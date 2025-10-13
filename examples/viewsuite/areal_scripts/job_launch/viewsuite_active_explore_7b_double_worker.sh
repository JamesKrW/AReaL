 python3 -m areal.launcher.local \
  AReaL/examples/viewsuite/train_v2.py \
  --config AReaL/examples/viewsuite/areal_scripts/job_launch/viewsuite_active_explore_7b_double_worker.yaml \
  > "$(pwd)/$(basename "$0" .sh).log" 2>&1