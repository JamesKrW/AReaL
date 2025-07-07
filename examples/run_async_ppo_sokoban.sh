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

python3 training/main_async_ppo_sokoban.py \
    n_nodes=1 n_gpus_per_node=8 \
    mode=ray \
    wandb.mode=online \
    wandb.project=areal \
    wandb.name=test-areal-sokoban \
    experiment_name=sokoban-ppo \
    trial_name=sokoban-trial-1 \
    # Add other arguments as needed 