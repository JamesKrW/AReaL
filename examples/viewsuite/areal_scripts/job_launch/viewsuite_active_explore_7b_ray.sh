AREAL_SGLANG_ENTRY=/home/aiscuser/projects/AReaL/areal/launcher/sglang_server.py \
python3 -m areal.launcher.ray_abs \
  /home/aiscuser/projects/AReaL/examples/viewsuite/train_v3.py \
  --config AReaL/examples/viewsuite/areal_scripts/job_launch/viewsuite_active_explore_7b_ray.yaml \
  > "$(pwd)/$(basename "$0" .sh).log" 2>&1