#!/bin/bash

# Fixed Low-resource Async-PPO Training Script
# This version fixes the model path issues and uses only available GPUs



    
python3 training/main_async_ppo.py \
    n_nodes=1 n_gpus_per_node=2 \
    allocation_mode=sglang.d1p1m1+d1p1m1 \
    cluster.fileroot=/datadrive_d/kangrui/experiments \
    actor.type._class=qwen3 \
    actor.path=/datadrive_d/kangrui/projects/toolusevlm/AReaL/experiments/models/Qwen--Qwen3-0.6B \
    ref.type._class=qwen3 \
    ref.path=/datadrive_d/kangrui/projects/toolusevlm/AReaL/experiments/models/Qwen--Qwen3-0.6B \
    dataset.path=hf-dataset://inclusionAI/AReaL-RL-Data/data/boba_106k_0319.jsonl \
    dataset.train_bs_n_seqs=128 \
    group_size=4 \
    ppo.gen.max_new_tokens=8192 \
    ppo.ppo_n_minibatches=4 \
    actor_train.mb_spec.max_tokens_per_mb=8192 \
    actor_inf.mb_spec.max_tokens_per_mb=8192 \
    ref_inf.mb_spec.max_tokens_per_mb=8192 \
    max_concurrent_rollouts=8 \
    max_head_offpolicyness=4 \
    cpus_per_generation_server=2 \
    mem_per_generation_server=20480 \
    cpus_per_gserver_manager=2 \
    mem_per_gserver_manager=5120 \
    cpus_per_rollout_worker=2 \
    mem_per_rollout_worker=10240 \
    cpus_per_master_worker=2 \
    mem_per_master_worker=10240 \
    cpus_per_model_worker=2 \
    mem_per_model_worker=20480 \
    actor.sglang.enable_memory_saver=true \
    actor.sglang.allow_auto_truncate=true \
    actor.sglang.context_length=8192 \
    actor.sglang.mem_fraction_static=0.6 \
    actor.sglang.max_running_requests=4 \
    actor.sglang.chunked_prefill_size=4096 \
    actor.sglang.max_prefill_tokens=8192 \
    actor.sglang.cpu_offload_gb=2
