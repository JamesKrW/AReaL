import os
import re
import sys
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.utils.data import Subset
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    cycle_dataloader,
    tensor_container_to,
)
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_processor_and_tokenizer
from realhf.base import seeding, stats_tracker



from torch.utils.data import Dataset
from typing import List, Dict, Any
from dataset import build_env_dataset
from agent_args import AgentGRPOConfig
from workflow_new import VisionMultiTurnAgentEnvWorkflow
# ------------------------------------------------------



def main(args):
    config, _ = load_expr_config(args, AgentGRPOConfig)
    config: AgentGRPOConfig

    rank = int(os.getenv("RANK"))

    seeding.set_random_seed(config.seed, f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)


    train_dataset = build_env_dataset(
        config.envs,
        split="train",
        base_seed=config.seed,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
    )
    train_size = len(train_dataset)
    subset_size = int(1.0 * train_size)

    random_indices = torch.randperm(train_size).tolist()[:subset_size]

    subset_train_dataset = Subset(train_dataset, random_indices)
    valid_dataset = build_env_dataset(
        config.envs,
        split="valid",
        base_seed=config.seed,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
    )
    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        subset_train_dataset,
        batch_size=config.train_dataset.batch_size // actor.data_parallel_world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size // actor.data_parallel_world_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.valid_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    weight_update_meta = [
        WeightUpdateMeta.from_fsdp_xccl(
            AllocationMode.from_str(config.allocation_mode), actor
        )
    ]
    # weight_update_meta = [WeightUpdateMeta.from_disk(config.saver)]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
        config.gconfig_eval.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
        config.gconfig_eval.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = VisionMultiTurnAgentEnvWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        processor=processor,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    eval_workflow = VisionMultiTurnAgentEnvWorkflow(
        gconfig=config.gconfig_eval,
        tokenizer=tokenizer,
        processor=processor,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    data_generator = cycle_dataloader(train_dataloader)
    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            batch = None
            if actor.is_data_parallel_head():
                if config.async_training:
                    batch = rollout.prepare_batch(
                        train_dataloader,
                        workflow=workflow,
                        should_accept=lambda sample: True,
                    )
                else:
                    batch = rollout.rollout_batch(
                        next(data_generator),
                        workflow=workflow,
                        should_accept=lambda sample: True,
                    )
                batch = tensor_container_to(batch, actor.device)
            batch = broadcast_tensor_container(
                batch,
                src_rank=actor.current_data_parallel_head(),
                group=actor.context_and_model_parallel_group,
            )
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()
        
        if actor.is_data_parallel_head() and "rewards" in batch and "tag_id" in batch:
            rewards_tensor = batch["rewards"].detach().float().cpu().view(-1)
            tag_tensor = batch["tag_id"].detach().long().cpu().view(-1)
            if rewards_tensor.numel() > 0 and tag_tensor.numel() == rewards_tensor.numel():
                stats_tracker.scalar(train_avg_reward=float(rewards_tensor.mean().item()))
                unique_tags = torch.unique(tag_tensor)
                for tag_id in unique_tags.tolist():
                    mask = tag_tensor == tag_id
                    if mask.any():
                        tag_reward = rewards_tensor[mask].mean().item()
                        tag_key = f"tag_{tag_id}"
                        stats_tracker.scalar(**{f"train_avg_reward/{tag_key}": float(tag_reward)})
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()
                        
        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)

                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()


        with stats_tracker.record_timing("update_weights"):
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            current_platform.synchronize()

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(
                actor,
                epoch,
                step,
                global_step,
                tokenizer=tokenizer,
                processor=processor,
            )

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
                processor=processor,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                eval_avg_reward = None
                per_tag_totals: Dict[int, float] = {}
                per_tag_counts: Dict[int, int] = {}
                if actor.is_data_parallel_head():
                    # Stats are logged in workflow
                    # and will be exported later
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    results = eval_rollout.wait(cnt, timeout=None)
                    if "rewards" in results:
                        rewards_tensor = results["rewards"].float().view(-1).cpu()
                        if rewards_tensor.numel() > 0:
                            eval_avg_reward = float(rewards_tensor.mean().item())
                        if "tag_id" in results:
                            tag_tensor = results["tag_id"].long().view(-1).cpu()
                            if tag_tensor.numel() == rewards_tensor.numel():
                                for reward, tag_id in zip(rewards_tensor.tolist(), tag_tensor.tolist()):
                                    per_tag_totals[tag_id] = per_tag_totals.get(tag_id, 0.0) + reward
                                    per_tag_counts[tag_id] = per_tag_counts.get(tag_id, 0) + 1
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()

                if eval_avg_reward is not None:
                    stats_tracker.scalar(eval_avg_reward=eval_avg_reward)
                    for tag_id, total in per_tag_totals.items():
                        count = per_tag_counts.get(tag_id, 0)
                        if count > 0:
                            tag_key = f"tag_{tag_id}"
                            stats_tracker.scalar(**{f"eval_avg_reward/{tag_key}": total / count})

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats[0].update(
            stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        )
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
