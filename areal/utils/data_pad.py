import torch
from typing import List, Dict, Tuple
from tensordict import TensorDict

def _right_pad_dim1(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
    if x.size(1) == target_len:
        return x
    pad_len = target_len - x.size(1)
    pad_shape = (x.size(0), pad_len, *x.shape[2:])
    pad = x.new_full(pad_shape, pad_value)
    return torch.cat([x, pad], dim=1)

def _gather_lengths(tds: List[TensorDict], key: str) -> torch.Tensor:
    """Return per-sample lengths along dim=1 for `key`, shape (B,). Missing -> 0."""
    lens = []
    for td in tds:
        if key in td.keys():
            lens.append(td[key].shape[1])
        else:
            # assume batch size = td.batch_size[0]
            lens.extend([0] * td.batch_size[0])
    if len(lens) != sum(td.batch_size[0] for td in tds):
        # Fallback safer path
        lens = []
        for td in tds:
            L = td[key].shape[1] if key in td.keys() else 0
            lens.append(L)
    return torch.tensor(lens, dtype=torch.long, device=tds[0].device if hasattr(tds[0], "device") else None)

def concat_padded_tensors(
    tensor_dicts: List[TensorDict],
    pad_value: float = 0.0,
) -> TensorDict:
    """
    Batch-concat with padding:
      - Text keys pad to text_max_len (from attention_mask).
      - Vision keys ('pixel_values', 'image_grid_thw', 'video_grid_thw', 'second_per_grid_ts')
        also pad along dim=1 to their own max lens within the batch.
      - Additionally, emit '<key>_lengths' of shape (B,) giving true per-sample lengths
        so you can slice away padding later in prepare_mb_list.
    """
    if not tensor_dicts:
        return TensorDict()

    # New batch size
    batch_sizes = [tuple(d.batch_size) for d in tensor_dicts]
    B = sum(bs[0] for bs in batch_sizes)
    new_batch_size = [B, *batch_sizes[0][1:]]

    assert all("attention_mask" in td for td in tensor_dicts), "Missing attention_mask"
    text_max_len = max(td["attention_mask"].shape[1] for td in tensor_dicts)

    vision_keys = ("pixel_values", "image_grid_thw", "video_grid_thw", "second_per_grid_ts")

    # compute max lens for vision keys
    def _key_max_len(key: str) -> int:
        lens = [td[key].shape[1] for td in tensor_dicts if key in td.keys()]
        return max(lens) if lens else 0

    max_lens: Dict[str, int] = {k: _key_max_len(k) for k in vision_keys}

    # All keys union
    all_keys = set()
    for td in tensor_dicts:
        all_keys |= set(td.keys())

    result: Dict[str, torch.Tensor] = {}

    # First: build concatenated tensors with padding
    for key in all_keys:
        tensors = []
        is_vision = key in vision_keys
        target_len = max_lens[key] if is_vision else text_max_len

        # find a ref tensor for shape/dtype if needed
        ref = next((td[key] for td in tensor_dicts if key in td.keys()), None)

        for td in tensor_dicts:
            if key not in td.keys():
                # create empty (Bi, 0, *tail) to be padded up to target_len
                if ref is None:
                    tensors = []
                    break
                Bi = td.batch_size[0]
                if ref.dim() == 1:
                    empty = ref.new_empty((Bi,))
                else:
                    tail = ref.shape[2:]
                    empty = ref.new_empty((Bi, 0, *tail)) if len(tail) > 0 else ref.new_empty((Bi, 0))
                t = empty
            else:
                t = td[key]

            if t.dim() == 1:
                # 1D tensors (e.g., rewards) directly collected
                tensors.append(t)
            else:
                padv = 0 if key == "attention_mask" else (0.0 if key in ("pixel_values",) else pad_value)
                t = _right_pad_dim1(t, target_len, padv)
                tensors.append(t)

        if not tensors:
            continue

        try:
            result[key] = torch.cat(tensors, dim=0)
        except RuntimeError as e:
            shapes = [list(t.shape) for t in tensors]
            raise RuntimeError(f"Concat failed for key '{key}' with shapes {shapes}") from e

    # --- build per-sample true lengths for vision keys ---
    device = next(iter(result.values())).device if len(result) else None
    vision_keys = ("pixel_values", "image_grid_thw", "video_grid_thw", "second_per_grid_ts")

    for k in vision_keys:
        if max_lens.get(k, 0) == 0:
            continue
        per = []
        for td in tensor_dicts:
            Bi = td.batch_size[0]
            if f"{k}_lengths" in td.keys():
                v = td[f"{k}_lengths"]
                # ensure shape (Bi,)
                v = v.to(device=device).view(-1)
                assert v.numel() == Bi, f"{k}_lengths numel={v.numel()} != batch {Bi} in one td"
                per.append(v)
            elif k in td.keys():
                L = td[k].shape[1]
                per.append(torch.full((Bi,), L, dtype=torch.long, device=device))
            else:
                per.append(torch.zeros((Bi,), dtype=torch.long, device=device))
        result[f"{k}_lengths"] = torch.cat(per, dim=0)  # shape (B,)
    B = new_batch_size[0]
    bad = []
    for k, v in result.items():
        if isinstance(v, torch.Tensor) and v.dim() == 1 and v.size(0) != B:
            bad.append((k, tuple(v.shape)))
    if bad:
        raise RuntimeError(f"1D key(s) not matching batch B={B}: {bad}")
    return TensorDict(result, batch_size=new_batch_size)
