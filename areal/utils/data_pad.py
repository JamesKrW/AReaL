import torch
from typing import List, Tuple, Dict
from tensordict import TensorDict

def _right_pad_dim1(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
    """
    Right-pad a tensor along dimension 1 to `target_len`.
    Works for any rank >= 2. Shape: (B, L, *tail) -> (B, target_len, *tail)
    """
    if x.size(1) == target_len:
        return x
    pad_len = target_len - x.size(1)
    pad_shape = (x.size(0), pad_len, *x.shape[2:])
    pad = x.new_full(pad_shape, pad_value)
    return torch.cat([x, pad], dim=1)

def _concat_ragged_and_offsets(
    xs: List[torch.Tensor],
    squeeze_leading_batch: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate a list of variable-length tensors along dim=0 and
    return (flat_tensor, offsets). Each element is assumed to have shape:
      - (N_i, *tail)         if squeeze_leading_batch=False or already squeezed
      - (1, N_i, *tail)      if squeeze_leading_batch=True (we will squeeze(0) to (N_i, *tail))

    Returns:
      flat:     torch.Tensor with shape (sum_i N_i, *tail)
      offsets:  torch.LongTensor of shape (B+1,), prefix sums with offsets[0]=0
    """
    pieces = []
    lengths = []
    for x in xs:
        t = x
        if squeeze_leading_batch and t.dim() >= 2 and t.size(0) == 1:
            t = t.squeeze(0)  # (N_i, *tail)
        # Allow empty per-sample tensors (N_i == 0)
        n_i = t.size(0) if t.dim() >= 1 else 0
        pieces.append(t)
        lengths.append(n_i)

    # Handle the case with all-zero lengths
    total = sum(lengths)
    if total == 0:
        # Build an empty tensor with a reasonable inferred dtype/device from the first non-empty
        ref = next((p for p in pieces if p.numel() > 0), None)
        if ref is None:
            # Fallback: use float32 CPU 1D zero tensor
            flat = torch.empty((0,), dtype=torch.float32)
        else:
            flat = ref.new_empty((0, *ref.shape[1:]))
        offsets = torch.zeros(len(xs) + 1, dtype=torch.long, device=flat.device)
        return flat, offsets

    flat = torch.cat(pieces, dim=0)
    offsets = torch.tensor([0] + list(torch.as_tensor(lengths).cumsum(0).tolist()),
                           dtype=torch.long, device=flat.device)
    return flat, offsets

def concat_padded_tensors(
    tensor_dicts: List[TensorDict],
    pad_value: float = 0.0,
    ragged_keys: Tuple[str, ...] = ("pixel_values", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"),
) -> TensorDict:
    """
    Batch-concatenate a list of TensorDicts:

    - Text-like fields (variable sequence length along dim=1) are padded to the
      max length derived from 'attention_mask', then concatenated along dim=0.

    - Vision-like fields (ragged) are NOT padded into a fake batch. Instead, we
      build:
        * a single flat tensor concatenated along dim=0 across samples, and
        * a companion '<key>_offsets' of shape (B+1) so you can slice per-sample
          ranges later (Plan B).

      Examples:
        image_grid_thw: (1, N_i, 3) or (N_i, 3) per sample   → flat (sum N_i, 3) + offsets
        pixel_values:   (1, N_i, C, H, W) → (sum N_i, C, H, W) + offsets

    - 1D tensors (e.g., rewards with shape (B,)) are concatenated along dim=0 as-is.

    Returns:
      A TensorDict containing:
        - Padded & concatenated text fields (e.g., input_ids, attention_mask, ...)
        - For each ragged key 'k':
            k_flat:       concatenated flat tensor
            k_offsets:    LongTensor (B+1) prefix sums
    """
    if not tensor_dicts:
        return TensorDict()

    # New batch size: sum of per-sample batch sizes along dim=0
    batch_sizes = [tuple(d.batch_size) for d in tensor_dicts]
    B = sum(bs[0] for bs in batch_sizes)
    new_batch_size = [B, *batch_sizes[0][1:]]

    # --- Text padding target ---
    assert all("attention_mask" in td for td in tensor_dicts), "Missing 'attention_mask' in some TensorDicts."
    text_max_len = max(td["attention_mask"].shape[1] for td in tensor_dicts)

    # Gather all keys present in any sample
    all_keys = set()
    for td in tensor_dicts:
        all_keys |= set(td.keys())

    result: Dict[str, torch.Tensor] = {}

    # First pass: handle ragged vision keys into flat+offsets; skip adding the original key
    for key in all_keys:
        if key not in ragged_keys:
            continue
        vals = [td[key] for td in tensor_dicts if key in td.keys()]
        if len(vals) == 0:
            # Nothing to do for this key
            continue

        # Build a per-sample list aligned to tensor_dicts (missing → empty)
        aligned_vals = []
        ref = vals[0]
        tail = ref.shape[2:] if (ref.dim() >= 3 and ref.size(0) == 1) else (ref.shape[1:] if ref.dim() >= 2 else ())
        device = ref.device
        dtype = ref.dtype

        for td in tensor_dicts:
            if key in td.keys():
                aligned_vals.append(td[key])
            else:
                # Missing → create (1, 0, *tail) or (0, *tail) so that squeeze works uniformly
                empty = torch.empty((1, 0, *tail), dtype=dtype, device=device) if len(tail) > 0 else torch.empty((1, 0), dtype=dtype, device=device)
                aligned_vals.append(empty)

        flat, offsets = _concat_ragged_and_offsets(aligned_vals, squeeze_leading_batch=True)
        result[f"{key}_flat"] = flat
        result[f"{key}_offsets"] = offsets

    # Second pass: handle non-ragged keys with text padding
    for key in all_keys:
        if key in ragged_keys:
            # The original ragged tensors are not included; we provide *_flat + *_offsets instead.
            continue

        tensors = []
        for td in tensor_dicts:
            if key not in td.keys():
                # If a sample is missing this key, create a matching empty/padded tensor
                # Try to infer from any existing tensor for this key
                ref = next((tdd[key] for tdd in tensor_dicts if key in tdd.keys()), None)
                if ref is None:
                    # Skip entirely if no sample contains this key
                    tensors = []
                    break
                if ref.dim() == 1:
                    tensors.append(ref.new_empty((0,)))
                else:
                    # Create (B_i, 0, *tail)
                    Bi = td.batch_size[0]
                    tail = ref.shape[2:]
                    empty = ref.new_empty((Bi, 0, *tail)) if len(tail) > 0 else ref.new_empty((Bi, 0))
                    tensors.append(empty)
                continue

            t = td[key]
            if t.dim() == 1:
                tensors.append(t)
            else:
                # Pad along dim=1 to text_max_len (for text-like tensors)
                t = _right_pad_dim1(t, text_max_len, pad_value if key != "attention_mask" else 0)
                tensors.append(t)

        if not tensors:
            continue

        try:
            result[key] = torch.cat(tensors, dim=0)
        except RuntimeError as e:
            shapes = [list(t.shape) for t in tensors]
            raise RuntimeError(f"Concat failed for key '{key}' with shapes {shapes}") from e

    return TensorDict(result, batch_size=new_batch_size)
