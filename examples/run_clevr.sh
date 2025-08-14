python3 -m areal.launcher.local \
  examples/lite/clevr_count_70k_grpo_modified.py \
  --config examples/lite/configs/clevr_count_70k_grpo_modified.yaml \
  cluster.fileroot=$HOME/tmp/areal/experiments \
  cluster.name_resolve.nfs_record_root=$HOME/tmp/areal/name_resolve \
  allocation_mode=sglang.d2p1t1+d1p1t2