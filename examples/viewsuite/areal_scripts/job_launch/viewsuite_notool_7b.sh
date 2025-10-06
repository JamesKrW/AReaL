 python3 -m areal.launcher.local \
  AReaL/examples/viewsuite/train_v0.py \
  --config AReaL/examples/viewsuite/areal_scripts/job_launch/viewsuite_notool_7b.yaml \
  > "$(pwd)/$(basename "$0" .sh).log" 2>&1