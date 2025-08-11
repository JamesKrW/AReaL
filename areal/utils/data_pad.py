import torch
from typing import List
from tensordict import TensorDict
# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

# Pad/unpad operations are modified from flash-attention under BSD-3 license.
# Copyright (c) 2023, Tri Dao.

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict

from areal.api.cli_args import MicroBatchSpec
from realhf.base import datapack, logging

logger = logging.getLogger("data_pad utils")
def _right_pad_dim1(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
    """
    Right-pad a tensor along dimension 1 to `target_len`.
    Preserves all other dimensions unchanged.

    Args:
        x: Input tensor of shape (B, N, ...).
        target_len: Desired length along dim=1 after padding.
        pad_value: Value to fill in the padded positions.

    Returns:
        Tensor padded along dim=1 to shape (B, target_len, ...).
    """
    if x.size(1) == target_len:
        return x
    pad_len = target_len - x.size(1)
    pad_shape = (x.size(0), pad_len, *x.shape[2:])
    pad = x.new_full(pad_shape, pad_value)
    return torch.cat([x, pad], dim=1)

def concat_padded_tensors(
    tensor_dicts: List[TensorDict], pad_value: float = 0.0
) -> TensorDict:
    """
    Concatenate multiple TensorDicts along the batch dimension (dim=0),
    padding sequences along dim=1 so that all tensors for the same key
    have matching shapes.

    This function is designed for multi-sample rollouts where:
    - Text-related fields (input_ids, attention_mask, etc.) may have
      variable sequence lengths.
    - Vision-related fields (pixel_values, image_grid_thw) may have
      variable numbers of images/tokens per sample.
    - Scalar or 1D tensors (e.g., rewards) are concatenated directly.

    Padding rules:
    - Text fields are padded to the max sequence length determined from
      attention_mask.
    - Visual fields are padded separately to their own max length (dim=1).
    - Missing keys in a sample are filled with empty tensors of shape
      (B, 0, *tail) before padding.

    Args:
        tensor_dicts: List of TensorDict objects, each representing one sample.
        pad_value: Value used for padding non-mask fields.

    Returns:
        A single TensorDict containing concatenated and padded tensors.
    """
    if not tensor_dicts:
        return TensorDict()

    # Compute new batch size (sum over dim=0 of each TensorDict)
    batch_sizes = [tuple(d.batch_size) for d in tensor_dicts]
    new_batch_size = [sum(bs[0] for bs in batch_sizes), *batch_sizes[0][1:]]

    # Determine max sequence length for text fields
    assert all("attention_mask" in td for td in tensor_dicts), "Missing attention_mask in some TensorDicts"
    text_max_len = max(td["attention_mask"].shape[1] for td in tensor_dicts)

    # Determine max sequence length (dim=1) for visual fields
    def _key_max_len(key: str) -> int:
        lens = [td[key].shape[1] for td in tensor_dicts if key in td.keys()]
        return max(lens) if lens else 0

    pv_max_len  = _key_max_len("pixel_values")
    thw_max_len = _key_max_len("image_grid_thw")

    # Collect all keys across all samples (not just from the first sample)
    all_keys = set()
    for td in tensor_dicts:
        all_keys |= set(td.keys())

    result = {}
    for key in all_keys:
        tensors = []

        # Decide padding length and pad value per key type
        if key == "pixel_values":
            target_len = pv_max_len
            key_pad = 0.0  # Fill pixel data with zeros
        elif key == "image_grid_thw":
            target_len = thw_max_len
            key_pad = 0   # Fill grid indices with zeros
        else:
            target_len = text_max_len
            key_pad = pad_value

        # Find a reference tensor to infer tail shape for empty placeholders
        ref_tensor = next((td[key] for td in tensor_dicts if key in td.keys()), None)

        for td in tensor_dicts:
            if key not in td.keys():
                # If this sample is missing the key, create an empty tensor
                if ref_tensor is None:
                    # All samples missing the key → skip
                    tensors = []
                    break
                B = td.batch_size[0]
                tail = ref_tensor.shape[2:] if ref_tensor.dim() >= 2 else ()
                empty = ref_tensor.new_empty((B, 0, *tail)) if tail else ref_tensor.new_empty((B,))
                tensor = empty
            else:
                tensor = td[key]

            # Scalars or 1D tensors can be concatenated directly
            if tensor.dim() == 1:
                tensors.append(tensor)
                continue

            # Pad sequences along dim=1
            tensor = _right_pad_dim1(tensor, target_len, key_pad)
            tensors.append(tensor)

        if not tensors:
            continue

        try:
            result[key] = torch.cat(tensors, dim=0)
        except RuntimeError as e:
            shapes = [list(t.shape) for t in tensors]
            raise RuntimeError(f"Concat failed for key '{key}' with shapes {shapes}") from e

    return TensorDict(result, batch_size=new_batch_size)
