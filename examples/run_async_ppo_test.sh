#!/bin/bash

# Set wandb environment variables
wandb login
# Add other wandb-related variables here if needed

# Get the absolute path of the current script directory
PWD_ABS="$(pwd)"

# Set ray_temp_path to the ray directory under the parent of the current directory (absolute path)
RAY_TEMP_PATH="$(dirname "$PWD_ABS")/ray"
export RAY_TEMP_PATH

# Set cluster.fileroot to the experiments directory under the current directory (absolute path)
CLUSTER_FILEROOT="$PWD_ABS/areal_experiments"
export CLUSTER_FILEROOT

python3 training/main_async_ppo.py \
    n_nodes=1 n_gpus_per_node=2 \
    allocation_mode=sglang.d1p1m1+d1p1m1 \
    cluster.fileroot=$CLUSTER_FILEROOT \
    wandb.mode=online \
    wandb.project=areal \
    wandb.name=test-areal-0.6b \
    actor.type._class=qwen3 \
    actor.path=Qwen/Qwen3-0.6B \
    ref.type._class=qwen3 \
    ref.path=Qwen/Qwen3-0.6B \
    dataset.path=hf-dataset://inclusionAI/AReaL-RL-Data/data/boba_106k_0319.jsonl \
    dataset.train_bs_n_seqs=8 \
    group_size=4 \
    ppo.gen.max_new_tokens=2048 \
    ppo.ppo_n_minibatches=4 \
    actor_train.mb_spec.max_tokens_per_mb=16384 \
    actor_inf.mb_spec.max_tokens_per_mb=16384 \
    max_concurrent_rollouts=4 \
    max_head_offpolicyness=4 \
    flush_request_timeout=900 \
    recover_retries=30 \
    recover_after=60 \
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
    actor.sglang.context_length=4096 \
    actor.sglang.mem_fraction_static=0.5 \
    actor.sglang.max_running_requests=2 \
    actor.sglang.chunked_prefill_size=2048 \
    actor.sglang.max_prefill_tokens=4096 \
    actor.sglang.cpu_offload_gb=0 \
    ray_temp_path=$RAY_TEMP_PATH