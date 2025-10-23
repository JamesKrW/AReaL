# -*- coding: utf-8 -*-
# NOTE: Chat in Chinese; code comments in English.

import os
import re
import sys
import time
from copy import deepcopy
from typing import Dict, Any

import torch
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta

from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    cycle_dataloader,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.viewsuite.agent_dataset import build_env_dataset
from areal.viewsuite.agent_args import AgentGRPOConfig
from areal.viewsuite.workflow_v3 import VisionMultiTurnAgentEnvWorkflow


# ------------------------------------------------------
# Utilities
# ------------------------------------------------------


def main(args):
    # Load config (AgentGRPOConfig extends GRPOConfig with agent-specific fields)
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

    # ------------------------------------------------------------------
    # Datasets and Dataloaders (new API)
    #   - get_custom_dataset(split=..., dataset_config=..., processor=...)
    #   - create_dataloader(dataset, rank=..., world_size=..., dataset_config=...)
    #   Sharding is handled inside create_dataloader; dataset itself is unsharded.
    # ------------------------------------------------------------------
    train_dataset = build_env_dataset(
        config.envs,
        split="train",
        base_seed=config.seed,
    )
    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    valid_dataset = build_env_dataset(
        config.envs,
        split="valid",
        base_seed=config.seed,
       
    )
    
    # only rank0 submit eval task so it needs see full dataset
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.valid_dataset,
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

    # weight_update_meta = WeightUpdateMeta.from_disk(
    #     experiment_name=config.experiment_name,
    #     trial_name=config.trial_name,
    #     file_root=config.cluster.fileroot
    # )
    weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)
    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

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
            if config.async_training:
                batch = actor.prepare_batch(
                    train_dataloader,
                    granularity=actor.config.group_size,
                    workflow=workflow,
                    should_accept=lambda sample: True,
                )
            else:
                batch = actor.rollout_batch(
                    next(data_generator),
                    granularity=actor.config.group_size,
                    workflow=workflow,
                    should_accept=lambda sample: True,
                )

        
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

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                if actor.is_data_parallel_head():
                    # Stats are logged in workflow
                    # and will be exported later
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

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
