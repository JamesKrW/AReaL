# -*- coding: utf-8 -*-
# NOTE: Chat in Chinese; code comments in English.

import os
import re
import sys
import time
from copy import deepcopy
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from torch.utils.data import Subset, Dataset
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
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

from dataset import build_env_dataset
from agent_args import AgentGRPOConfig
from workflow_new import VisionMultiTurnAgentEnvWorkflow

# ------------------------------------------------------
# Utilities
# ------------------------------------------------------

def submit_with_backpressure(engine: RemoteSGLangEngine, items, workflow, max_inflight: int = 256):
    """
    Submit items to the workflow executor with backpressure.
    - Keep at most `max_inflight` requests in the input queue.
    - When reaching the limit, call wait(...) to drain results before continuing.
    - Convert transient "Input queue full" errors into short backoff retries.
    Returns:
        dict: aggregated results merged across multiple wait() calls (best-effort).
    """
    inflight = 0
    aggregated = {}
    # Submit loop
    for item in items:
        # Try to submit; if input queue is full, backoff for a short time and retry.
        while True:
            try:
                engine.submit(item, workflow)
                break
            except RuntimeError as e:
                if "Input queue full" in str(e):
                    time.sleep(0.02)  # short backoff
                    continue
                raise
        inflight += 1

        # Drain when inflight reaches the cap
        if inflight >= max_inflight:
            part = engine.wait(inflight, timeout=None)
            if isinstance(part, dict):
                for k, v in part.items():
                    if k not in aggregated:
                        aggregated[k] = v
                    else:
                        # Best-effort concatenate for tensor-like values
                        try:
                            aggregated[k] = torch.cat([aggregated[k], v], dim=0)
                        except Exception:
                            aggregated[k] = v
            inflight = 0

    # Drain remaining inflight requests
    if inflight > 0:
        part = engine.wait(inflight, timeout=None)
        if isinstance(part, dict):
            for k, v in part.items():
                if k not in aggregated:
                    aggregated[k] = v
                else:
                    try:
                        aggregated[k] = torch.cat([aggregated[k], v], dim=0)
                    except Exception:
                        aggregated[k] = v
    return aggregated


def run_evaluation(
    evaluator: Evaluator,
    eval_rollout: RemoteSGLangEngine,
    eval_workflow: VisionMultiTurnAgentEnvWorkflow,
    valid_dataloader: StatefulDataLoader,
    actor: FSDPPPOActor,
    epoch: int,
    step: int,
    global_step: int,
    force: bool = False,
):
    """
    Reusable evaluation entry used both at the beginning (eval_first) and each training step.
    - Only global rank 0 submits evaluation jobs to avoid queue flooding.
    - Backpressure submission with bounded inflight.
    - Tag-wise aggregation and logging via stats_tracker.
    - Proper synchronization barriers for multi-node/multi-GPU.
    """
    with stats_tracker.record_timing("eval"):

        def evaluate_fn():
            # Only global rank 0 submits evaluation jobs
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()
                return

            eval_avg_reward = None
            per_tag_totals: Dict[int, float] = {}
            per_tag_counts: Dict[int, int] = {}

            # Derive a conservative inflight cap from config if available.
            max_inflight = 256
            try:
                qsz = getattr(eval_rollout.config, "queue_size", None)
                mcr = getattr(eval_rollout.config, "max_concurrent_rollouts", 64)
                if qsz:
                    max_inflight = min(int(qsz), 2 * int(mcr))
            except Exception:
                pass

            # Stream submissions with backpressure
            def gen_items():
                for data in valid_dataloader:
                    for item in data:
                        yield item

            results = submit_with_backpressure(
                eval_rollout,
                gen_items(),
                eval_workflow,
                max_inflight=max_inflight,
            )

            # Aggregate metrics if present
            if isinstance(results, dict) and "rewards" in results:
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

        if force:
            evaluate_fn()
            if hasattr(evaluator, "freq_ctl") and evaluator.freq_ctl is not None:
                evaluator.freq_ctl.time_ctl.reset_time()
            return

        evaluator.evaluate(
            evaluate_fn,
            epoch,
            step,
            global_step,
        )

    dist.barrier(device_ids=[actor.device.index])
    current_platform.synchronize()


# ------------------------------------------------------
# Main
# ------------------------------------------------------

def main(args):
    config, _ = load_expr_config(args, AgentGRPOConfig)
    config: AgentGRPOConfig

    rank = int(os.getenv("RANK", "0"))

    # Set seeds
    seeding.set_random_seed(config.seed, f"trainer{rank}")

    # Allocation / parallel strategy
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine (actor) and process groups
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)

    # Tokenizer / Processor
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    # Build datasets (train / valid)
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

    # Dataloaders
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

    # Training spec
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Inference engines
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)

    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # NCCL/XCCL weight update meta (broadcast rank-0 info to all ranks)

    # Build workflows (train / eval)
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

    # Utilities
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    # Recover
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

    # ------------------------------------------------------
    # (NEW) Optional evaluation before training loop
    # ------------------------------------------------------
    # We keep versions at their initial values; this measures current model state.
    if getattr(config, "eval_first", False):
        # Use epoch=0, step=0, and current global_step (start_step) for bookkeeping.
        run_evaluation(
            evaluator=evaluator,
            eval_rollout=eval_rollout,
            eval_workflow=eval_workflow,
            valid_dataloader=valid_dataloader,
            actor=actor,
            epoch=0,
            step=0,
            global_step=start_step,
            force=True,
        )
        initial_eval_stats = stats_tracker.export_all(
            reduce_group=actor.data_parallel_group
        )
        if initial_eval_stats:
            # Force the very first commit to align wandb step with global_step.
            if not dist.is_initialized() or dist.get_rank() == 0:
                stats_logger._last_commit_step = start_step - 1
            initial_epoch = (
                start_step // steps_per_epoch if steps_per_epoch > 0 else 0
            )
            initial_epoch_step = (
                start_step % steps_per_epoch if steps_per_epoch > 0 else 0
            )
            stats_logger.commit(
                initial_epoch,
                initial_epoch_step,
                start_step,
                initial_eval_stats,
            )

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

        # ---------- Rollout ----------
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

        # ---------- Optional recompute logprob ----------
        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        # ---------- Reference model logprob ----------
        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        # ---------- Advantage ----------
        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        # ---------- PPO update ----------
        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # Pause rollout for updates / save / eval
        rollout.pause()

        # ---------- Update weights ----------
        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        # ---------- Save checkpoint ----------
        with stats_tracker.record_timing("save"):
            saver.save(
                actor,
                epoch,
                step,
                global_step,
                tokenizer=tokenizer,
                processor=processor,
            )

        # ---------- Recover checkpoint ----------
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

        # ---------- Evaluation ---------- (reused)
        run_evaluation(
            evaluator=evaluator,
            eval_rollout=eval_rollout,
            eval_workflow=eval_workflow,
            valid_dataloader=valid_dataloader,
            actor=actor,
            epoch=epoch,
            step=step,
            global_step=global_step,
        )

        # ---------- Log stats ----------
        stats[0].update(
            stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        )
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

    # ---------- Teardown ----------
    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
