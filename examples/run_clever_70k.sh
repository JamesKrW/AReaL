python3 -m areal.launcher.local \
  examples/lite/clevr_count_70k_grpo_m.py \
  --config examples/lite/configs/clevr_count_70k_grpo_m.yaml \
  cluster.fileroot=$HOME/tmp/areal/experiments \
  cluster.name_resolve.nfs_record_root=$HOME/tmp/areal/name_resolve \
  allocation_mode=sglang.d2p1t1+d1p1t2 \
  cluster.n_gpus_per_node=4 \
  train_dataset.path=BUAADreamer/clevr_count_70k \
  valid_dataset.path=BUAADreamer/clevr_count_70k \
  actor.path=Qwen/Qwen2.5-VL-3B-Instruct \
  actor.gradient_checkpointing=true
