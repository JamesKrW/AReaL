import os
import re
import sys
import itertools

import torch
import torch.distributed as dist
from torch.utils.data import Subset
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import AllocationMode, FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
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
from workflow import VisionMultiTurnAgentEnvWorkflow
# ------------------------------------------------------



def main(args):
    config, _ = load_expr_config(args, AgentGRPOConfig)
    config: AgentGRPOConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))


    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    train_dataset = build_env_dataset(
        config.envs, split="train", base_seed=config.seed, rank=rank, world_size=world_size
    )
    valid_dataset = build_env_dataset(
        config.envs, split="valid", base_seed=config.seed, rank=rank, world_size=world_size
    )
    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size // world_size,
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
    rollout.initialize(None, ft_spec)
    eval_rollout = RemoteSGLangEngine(config.rollout)
    eval_rollout.initialize(None, ft_spec)
    # NOTE: set a large version such that eval does not have any offpolicyness control
    eval_rollout.set_version(int(1e12))

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.initialize(None, ft_spec)

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_nccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    weight_update_meta = [WeightUpdateMeta.from_disk(
        experiment_name=config.saver.experiment_name,
        trial_name=config.saver.trial_name,
        file_root=config.saver.fileroot
        )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = VisionMultiTurnAgentEnvWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        processor=processor,
        max_turns=config.max_turns,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
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

    data_generator = itertools.cycle(train_dataloader)
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
            if config.async_training:
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            else:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(train_dataloader)
                    data = next(data_generator)
                batch = rollout.rollout_batch(data, workflow=workflow)

        batch = batch.to(actor.device)
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

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

        with stats_tracker.record_timing("update_weights"):
            rollout.pause()
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()
            rollout.resume()
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(
                actor,
                epoch,
                step,
                global_step,
                tokenizer=tokenizer,
                processor=processor,
            )

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                rollout.pause()
                cnt = 0
                for data in valid_dataloader:
                    for item in data:
                        eval_rollout.submit(item, workflow)
                        cnt += 1
                batch = eval_rollout.wait(cnt, timeout=None)
                rewards = batch["rewards"].float().to(actor.device)
                with stats_tracker.scope("grpo-eval"):
                    stats_tracker.denominator(
                        n_seqs=torch.ones(
                            rewards.shape[0],
                            device=rewards.device,
                            dtype=torch.bool,
                        )
                    )
                    stats_tracker.stat(task_reward=rewards, denominator="n_seqs")
                rollout.resume()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
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

        stats_logger.commit(epoch, step, global_step, stats)

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])