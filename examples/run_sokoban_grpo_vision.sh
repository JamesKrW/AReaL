python3 -m areal.launcher.local \
  examples/lite/sokoban_grpo_vision.py \
  --config examples/lite/configs/sokoban_grpo_vision.yaml \
  cluster.fileroot=$HOME/tmp/areal/experiments \
  cluster.name_resolve.nfs_record_root=$HOME/tmp/areal/name_resolve \
  allocation_mode=sglang.d4p1t1+d4p1t1