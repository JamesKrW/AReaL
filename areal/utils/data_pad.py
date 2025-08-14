import torch
from typing import List
from tensordict import TensorDict

def concat_padded_tensors(
    tensor_dicts: List[TensorDict],
    pad_value: float = 0.0,
    image_keys: List[str] = ("pixel_values", "image_grid_thw"),
) -> TensorDict:
    """
    Concatenate and pad tensors from multiple padded tensor dictionaries.

    Rules:
    - Text-like tensors (seq along dim=1) are padded to the max L derived from `attention_mask`.
    - Image-like tensors in `image_keys` are padded to the max length of their own dim=1.
    - 1D tensors (e.g., rewards) are concatenated without padding.
    - `attention_mask` is always padded with 0s; others use `pad_value`.
    """
    if not tensor_dicts:
        return TensorDict()

    # batch size aggregation: sum B over dicts, keep trailing batch dims from the first
    batch_sizes = [tuple(d.batch_size) for d in tensor_dicts]
    new_batch_size = [sum(b[0] for b in batch_sizes), *batch_sizes[0][1:]]

    # ---- figure out target lengths per key-group ----
    # text length from attention_mask (required for text-like tensors)
    assert all("attention_mask" in td for td in tensor_dicts), "attention_mask is required in all TensorDicts"
    text_max_L = max(td["attention_mask"].shape[1] for td in tensor_dicts)

    # per-image-key max lengths (if key missing in a dict, treat its length as 0)
    image_key_to_maxL = {}
    for k in image_keys:
        maxL = 0
        for td in tensor_dicts:
            if k in td:
                t = td[k]
                if t.dim() < 2:
                    raise ValueError(f"{k} must have at least 2 dims [B, L, ...], got shape {tuple(t.shape)}")
                maxL = max(maxL, t.shape[1])
        if maxL > 0:
            image_key_to_maxL[k] = maxL

    result = {}
    all_keys = list(tensor_dicts[0].keys())

    for key in all_keys:
        tensors_to_concat = []
        is_image_key = key in image_key_to_maxL  # only if we actually observed it
        target_L = image_key_to_maxL[key] if is_image_key else text_max_L

        for td in tensor_dicts:
            if key not in td:
                # Create an empty (all-pad) tensor for missing keys in this dict
                # We need a reference shape from any dict that has the key
                ref = None
                for tdr in tensor_dicts:
                    if key in tdr:
                        ref = tdr[key]
                        break
                if ref is None:
                    raise KeyError(f"Key '{key}' not found in any TensorDicts.")
                if ref.dim() == 1:
                    # For 1D signals, if missing, create zeros of length B (safe choice)
                    B = td.batch_size[0]
                    filler = ref.new_zeros((B,), dtype=ref.dtype)
                elif ref.dim() >= 2:
                    B = td.batch_size[0]
                    if is_image_key:
                        filler = ref.new_full((B, target_L, *ref.shape[2:]), pad_value)
                    else:
                        if key == "attention_mask":
                            filler = ref.new_zeros((B, target_L), dtype=ref.dtype)
                        else:
                            filler = ref.new_full((B, target_L, *ref.shape[2:]), pad_value)
                else:
                    raise ValueError(f"Unsupported dim for key '{key}'.")
                tensors_to_concat.append(filler)
                continue

            tensor = td[key]

            # 1D tensors: concat directly
            if tensor.dim() == 1:
                tensors_to_concat.append(tensor)
                continue

            # sanity: ensure batch matches this td
            if tensor.shape[0] != td.batch_size[0]:
                raise ValueError(f"Key '{key}' has B={tensor.shape[0]} but td.batch_size[0]={td.batch_size[0]}.")

            cur_L = tensor.shape[1]
            if cur_L == target_L:
                tensors_to_concat.append(tensor)
                continue

            if cur_L > target_L:
                # If some sequences are already longer, we truncate to target_L
                # (rare, but makes function robust)
                tensor = tensor[:, :target_L, ...]
                tensors_to_concat.append(tensor)
                continue

            # pad along dim=1 to target_L
            pad_width = target_L - cur_L
            if key == "attention_mask":
                pad = tensor.new_zeros((tensor.shape[0], pad_width), dtype=tensor.dtype)
            else:
                pad = tensor.new_full((tensor.shape[0], pad_width, *tensor.shape[2:]), pad_value)

            tensor = torch.cat([tensor, pad], dim=1)
            tensors_to_concat.append(tensor)

        # concat across batch
        result[key] = torch.cat(tensors_to_concat, dim=0)

    return TensorDict(result, batch_size=new_batch_size)
