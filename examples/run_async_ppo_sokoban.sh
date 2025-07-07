#!/bin/bash
# Set wandb environment variables
wandb login

# Get the absolute path of the current script directory
PWD_ABS="$(pwd)"

# Set ray_temp_path to the ray directory under the parent of the current directory (absolute path)
RAY_TEMP_PATH="$(dirname "$PWD_ABS")/ray"
export RAY_TEMP_PATH

# Set cluster.fileroot to the experiments directory under the current directory (absolute path)
CLUSTER_FILEROOT="$PWD_ABS/areal_experiments"
export CLUSTER_FILEROOT

python3 training/main_async_ppo_sokoban.py \
    n_nodes=1 n_gpus_per_node=8 \
    allocation_mode=sglang.d4p1m1+d2p2m1 \
    cluster.fileroot=$CLUSTER_FILEROOT \
    actor.type._class=qwen2 \
    actor.path=Qwen/Qwen2.5-1.5B-Instruct \
    ref.type._class=qwen3 \
    ref.path=Qwen/Qwen2.5-1.5B-Instruct \
    dataset.path=placeholder_dataset.jsonl \
    dataset.train_bs_n_seqs=32 \
    group_size=1 \
    ppo.gen.max_new_tokens=1024 \
    ppo.ppo_n_minibatches=4 \
    actor_train.mb_spec.max_tokens_per_mb=32768 \
    actor_inf.mb_spec.max_tokens_per_mb=32768 \
    max_concurrent_rollouts=16 \
    max_head_offpolicyness=4 \
    flush_request_timeout=900 \
    recover_retries=30 \
    recover_after=60 \
    wandb.mode=online \
    wandb.project=areal \
    wandb.name=test-areal-sokoban \
    experiment_name=sokoban-ppo \
    trial_name=sokoban-trial-1 \
    ray_temp_path=$RAY_TEMP_PATH 