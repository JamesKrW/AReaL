#!/bin/bash
# Set wandb environment variables
wandb login

# Get the absolute path of the current script directory
PWD_ABS="$(pwd)"

# Set ray_temp_path to the ray directory under the parent of the current directory (absolute path)
RAY_TEMP_PATH="$(dirname "$PWD_ABS")/ray"
export RAY_TEMP_PATH

# Set cluster.fileroot to the experiments directory under the current directory (absolute path)
CLUSTER_FILEROOT="$PWD_ABS/experiments"
export CLUSTER_FILEROOT

python3 training/main_async_ppo.py \
    n_nodes=1 n_gpus_per_node=8 \
    allocation_mode=sglang.d4p1m1+d2p2m1 \
    cluster.fileroot=/home/aiscuser/projects/AReaL/experiments \
    actor.type._class=qwen3 \
    actor.path=Qwen/Qwen3-1.7B \
    ref.type._class=qwen3 \
    ref.path=Qwen/Qwen3-1.7B \
    dataset.path=hf-dataset://inclusionAI/AReaL-RL-Data/data/boba_106k_0319.jsonl \
    dataset.train_bs_n_seqs=32 \
    group_size=8 \
    ppo.gen.max_new_tokens=4096 \
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
    wandb.name=test-areal-1.7b \

