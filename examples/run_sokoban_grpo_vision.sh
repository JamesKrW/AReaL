python3 -m areal.launcher.local \
  examples/lite/sokoban_grpo_vision.py \
  --config examples/lite/configs/sokoban_grpo_vision.yaml \
  cluster.fileroot=$HOME/tmp/areal/experiments \
  cluster.name_resolve.nfs_record_root=$HOME/tmp/areal/name_resolve \
  allocation_mode=sglang.d4p1t1+d1p1t4 \
  train_dataset.batch_size=32 \
  gconfig.n_samples=16 \
  actor_train.mb_spec.max_tokens_per_mb=32768 \
  actor_inf.mb_spec.max_tokens_per_mb=32768 \
  ref.mb_spec.max_tokens_per_mb=32768 \
