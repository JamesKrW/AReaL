export PYTHONFAULTHANDLER=1
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_CPP_LOG_LEVEL=debug
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export RAY_DEDUP_LOGS=0
ulimit -c unlimited
ulimit -c unlimited  # 允许生成核心转储
 python3 -m areal.launcher.ray \
  AReaL/examples/viewsuite/train_v1.py \
  --config AReaL/examples/viewsuite/areal_scripts/job_launch/viewsuite_active_explore_7b_ray.yaml \
  > "$(pwd)/$(basename "$0" .sh).log" 2>&1