import math
from typing import List, Tuple, Dict, Any
from types import SimpleNamespace

import torch
from tensordict import TensorDict
from transformers import AutoProcessor


# =======================
# Factor helpers
# =======================
def best_factor_pair(q: int) -> Tuple[int, int]:
    """
    Return (b, c) s.t. b*c=q and |b-c| minimized, i.e., closest to sqrt(q).  q>=1
    """
    r = int(math.isqrt(q))
    for b in range(r, 0, -1):
        if q % b == 0:
            return b, q // b
    return 1, q


def generate_arrays(n: int, m: int) -> List[List[int]]:
    """
    Generate m arrays [a, b, c] with a∈{0,1}, b,c>=0, s.t. sum(a*b*c)=n.
    Split n as evenly as possible (diff ≤ 1) and choose square-like (b,c) per part.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    base, rem = divmod(n, m)
    parts = [base + 1 if i < rem else base for i in range(m)]
    out = []
    for q in parts:
        if q <= 0:
            out.append([0, 0, 0])
        else:
            b, c = best_factor_pair(q)
            out.append([1, b, c])
    assert sum(a * b * c for a, b, c in out) == n
    return out


# =======================
# Validators
# =======================
def _sum_thw_products(thw: torch.Tensor) -> torch.Tensor:
    """
    thw: (B, N, 3) -> (B,) sums of t*h*w over N
    """
    if thw.numel() == 0:
        return torch.zeros(thw.shape[0], dtype=torch.long, device=thw.device)
    prod = thw[..., 0] * thw[..., 1] * thw[..., 2]  # (B,N)
    return prod.sum(dim=1)  # (B,)


def validate_pixel_grid_alignment(td: TensorDict, pixel_key="pixel_values", grid_key="image_grid_thw") -> None:
    """
    Ensure L == sum(t*h*w) per row.
    pixel_values: (B, L, D)
    image_grid_thw: (B, N, 3)
    """
    if pixel_key not in td or grid_key not in td:
        return
    pv = td[pixel_key]
    thw = td[grid_key]
    assert pv.dim() == 3 and thw.dim() == 3 and thw.shape[-1] == 3
    B, L, _ = pv.shape
    assert thw.shape[0] == B, "batch mismatch between pixel_values and image_grid_thw"
    sums = _sum_thw_products(thw)
    assert torch.all(sums == L), f"Misalignment: L={L}, sums={sums.tolist()}"

import functools
from transformers import AutoProcessor

@functools.lru_cache(maxsize=1)
def get_default_processor(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    return AutoProcessor.from_pretrained(model_id)
# =======================
# Main concat/pad
# =======================
def concat_padded_tensors(
    tensor_dicts: List[TensorDict],
    pad_value: float = 0.0,
    processor: Any = None,
    image_keys: Tuple[str, str, str] = ("pixel_values", "image_grid_thw", "input_ids"),
) -> TensorDict:
    """
    Concatenate and pad a list of compatible TensorDicts along batch dim.

    Expected shapes (when images present):
      - pixel_values:  (B, L, D)    # D fixed, L depends on images
      - image_grid_thw:(B, N, 3)    # N groups, each [t,h,w]
      - input_ids:     (B, T)

    We compute global max_L (over pixel_values) and max_N (over groups). For each td:
      1) Pad pixel_values to max_L.
      2) Append groups to image_grid_thw to reach max_N; choose thw_allocs so
         sum(products) == (max_L - L).  (Uses generate_arrays.)
      3) Append corresponding vision special tokens to input_ids:
           <|vision_start|> + <|image_pad|> * (t*h*w) + <|vision_end|>
         for each newly added group. (Requires processor.tokenizer.)
      4) Compute final max_T across (possibly extended) input_ids.
      5) Pad input_ids (again if needed), attention_mask (zeros), and any other 2D [B,T]
         tensors to max_T. 1D tensors are concatenated, other ranks concatenated as-is.
    """
    if not tensor_dicts:
        return TensorDict()

    # new batch size
    batch_sizes = [tuple(d.batch_size) for d in tensor_dicts]
    new_batch_size = [sum(x[0] for x in batch_sizes), *batch_sizes[0][1:]]

    # Presence and maxima
    has_images = any("pixel_values" in td and "image_grid_thw" in td for td in tensor_dicts)
    max_L = 0
    max_N = 0
    if has_images:
        processor = processor or get_default_processor()
        for td in tensor_dicts:
            if "pixel_values" in td:
                max_L = max(max_L, td["pixel_values"].shape[1])
            if "image_grid_thw" in td:
                max_N = max(max_N, td["image_grid_thw"].shape[1])
        if any((max_L - td["pixel_values"].shape[1] > 0 and max_N == td["image_grid_thw"].shape[1]) 
            for td in tensor_dicts):
            max_N += 1
    # Build per-td plans
    pad_plans: List[Dict[str, Any]] = []
    for td in tensor_dicts:
        plan: Dict[str, Any] = {}
        if has_images and ("pixel_values" in td) and ("image_grid_thw" in td):
            pv = td["pixel_values"]
            thw = td["image_grid_thw"]
            B, L, _ = pv.shape
            _, N, _ = thw.shape
            pad_L = max_L - L
            pad_N = max_N - N

            if pad_N > 0:
                if pad_L < 0:
                    raise ValueError("pad_L cannot be negative.")
                thw_allocs = generate_arrays(pad_L, pad_N) if pad_L > 0 else [[0, 0, 0]] * pad_N
            else:
                thw_allocs = []

            if thw_allocs:
                assert sum(a * b * c for a, b, c in thw_allocs) == pad_L

            plan.update({"pad_L": pad_L, "pad_N": pad_N, "thw_allocs": thw_allocs, "B": B})
        pad_plans.append(plan)

    result: Dict[str, torch.Tensor] = {}

    # Stage 1: pixel_values -> pad to max_L
    if has_images and "pixel_values" in tensor_dicts[0]:
        pv_list = []
        for td in tensor_dicts:
            pv = td["pixel_values"]
            B, L, D = pv.shape
            device, dtype = pv.device, pv.dtype
            if L < max_L:
                pad = torch.zeros((B, max_L - L, D), device=device, dtype=dtype)
                pv = torch.cat([pv, pad], dim=1)
            pv_list.append(pv)
        result["pixel_values"] = torch.cat(pv_list, dim=0)

    # Stage 2: image_grid_thw -> append thw_allocs to reach max_N
    if has_images and "image_grid_thw" in tensor_dicts[0]:
        ig_list = []
        for td, plan in zip(tensor_dicts, pad_plans):
            thw = td["image_grid_thw"]
            device, dtype = thw.device, thw.dtype
            if plan.get("thw_allocs"):
                alloc = torch.tensor(plan["thw_allocs"], device=device, dtype=dtype)  # (pad_N,3)
                pad_block = alloc.unsqueeze(0).expand(thw.shape[0], -1, -1)          # (B,pad_N,3)
                thw = torch.cat([thw, pad_block], dim=1)
            ig_list.append(thw)
        result["image_grid_thw"] = torch.cat(ig_list, dim=0)

    # Stage 3: input_ids -> append special tokens for NEW groups per-td, but DO NOT concat yet
    ii_per_td: List[torch.Tensor] = []
    if any("input_ids" in td for td in tensor_dicts):
        for td, plan in zip(tensor_dicts, pad_plans):
            if "input_ids" not in td:
                ii_per_td.append(None)
                continue
            ii = td["input_ids"]
            device, dtype = ii.device, ii.dtype
            if has_images and processor is not None and plan.get("thw_allocs"):
                toks_all = []
                for a, b, c in plan["thw_allocs"]:
                    count = a * b * c
                    group_str = "<|vision_start|>" + ("<|image_pad|>" * count) + "<|vision_end|>"
                    toks = processor.tokenizer.encode(group_str, add_special_tokens=False)
                    toks_all.extend(toks)
                if toks_all:
                    pad_tensor = torch.tensor(toks_all, device=device, dtype=dtype).unsqueeze(0)
                    pad_tensor = pad_tensor.expand(ii.shape[0], -1)
                    ii = torch.cat([ii, pad_tensor], dim=1)
            ii_per_td.append(ii)
    # else: no input_ids in any td

    # Stage 4: compute final max_T (considering appended tokens)
    if ii_per_td and any(x is not None for x in ii_per_td):
        max_T = max(x.shape[1] for x in ii_per_td if x is not None)
    else:
        # fallback to any 2D key
        candidate_keys = [k for k in tensor_dicts[0].keys() if tensor_dicts[0][k].dim() == 2]
        max_T = max((td[candidate_keys[0]].shape[1] for td in tensor_dicts), default=0)

    # Stage 5: process remaining keys, INCLUDING input_ids to unify to max_T
    all_keys = list(tensor_dicts[0].keys())
    for k in all_keys:
        # Skip keys already finalized, EXCEPT "input_ids" which we still need to pad to max_T
        if (k in result) and (k != "input_ids"):
            continue

        tensors = []
        for idx, td in enumerate(tensor_dicts):
            if k == "input_ids":
                x = ii_per_td[idx] if (idx < len(ii_per_td)) else None
                if x is None:
                    continue
            else:
                if k not in td.keys(True, True):
                    continue
                x = td[k]

            device, dtype = x.device, x.dtype

            # 1D -> concat as-is
            if x.dim() == 1:
                tensors.append(x)
                continue

            # 2D -> pad to max_T; other ranks -> concat as-is
            if x.dim() == 2:
                B, T = x.shape
                if T < max_T:
                    pad_width = max_T - T
                    if k == "attention_mask":
                        padding = torch.zeros((B, pad_width), device=device, dtype=dtype)
                    elif k == "input_ids":
                        pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
                        if pad_id is None:
                            pad_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", 0)
                        padding = torch.full((B, pad_width), fill_value=pad_id, device=device, dtype=dtype)
                    else:
                        padding = torch.full((B, pad_width), fill_value=pad_value, device=device, dtype=dtype)
                    x = torch.cat([x, padding], dim=1)
                tensors.append(x)
            else:
                tensors.append(x)

        if tensors:
            result[k] = torch.cat(tensors, dim=0)

    out = TensorDict(result, batch_size=new_batch_size)

    # Final validation if images present
    if has_images:
        validate_pixel_grid_alignment(out)

    return out





if __name__ == "__main__":
    # =======================
# Minimal tests
# =======================
    def _make_td(
        B: int,
        thw_groups: List[Tuple[int, int, int]],
        D: int = 8,
        T: int = 12,
        device: str = "cpu",
    ) -> TensorDict:
        L = sum(t * h * w for (t, h, w) in thw_groups)
        pixel_values = torch.randn(B, L, D, device=device)
        image_grid_thw = torch.tensor(thw_groups, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1, -1)
        input_ids = torch.randint(100, (B, T), device=device, dtype=torch.long)
        attention_mask = torch.ones(B, T, device=device, dtype=torch.long)
        rewards = torch.randint(0, 2, (B,), device=device, dtype=torch.long)
        return TensorDict(
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "rewards": rewards,
            },
            batch_size=[B],
        )


    def _test_basic_alignment_and_padding(processor):
        device = "cpu"
        D = 8
        # TD1: L=12 (1,3,4), N=1
        td1 = _make_td(B=2, thw_groups=[(1, 3, 4)], D=D, T=10, device=device)
        # TD2: L=20 (1,4,5), N=1
        td2 = _make_td(B=3, thw_groups=[(1, 4, 5)], D=D, T=7, device=device)
        # TD3: L=6  (1,2,3), N=1
        td3 = _make_td(B=1, thw_groups=[(1, 2, 3)], D=D, T=12, device=device)

        out = concat_padded_tensors([td1, td2, td3], processor=processor)

        # Shapes
        assert out["pixel_values"].shape == (2 + 3 + 1, 21, D)
        assert out["image_grid_thw"].shape == (6, 2, 3)
        # input_ids & attention_mask same T
        assert out["input_ids"].shape[1] == out["attention_mask"].shape[1]

        validate_pixel_grid_alignment(out)
        print("[OK] _test_basic_alignment_and_padding")


    def _test_group_additions_and_token_append(processor):
        device = "cpu"
        D = 4
        # tdA: L=8 (1,2,4), N=1
        tdA = _make_td(B=1, thw_groups=[(1, 2, 4)], D=D, T=5, device=device)
        # tdB: L=8 (1,2,4), N=1
        tdB = _make_td(B=1, thw_groups=[(1, 2, 4)], D=D, T=6, device=device)
        # tdC: L=14 (1,2,7), N=3  -> raise max_N to 3
        tdC = _make_td(B=1, thw_groups=[(1, 2, 7)], D=D, T=7, device=device)
        tdC["image_grid_thw"] = torch.tensor([[(1, 2, 7), (0, 0, 0), (0, 0, 0)]], dtype=torch.long).expand(1, -1, -1)

        out = concat_padded_tensors([tdA, tdB, tdC], processor=processor)

        # N padded to 3
        assert out["image_grid_thw"].shape[1] == 4
        # input_ids & attention_mask same T
        assert out["input_ids"].shape[1] == out["attention_mask"].shape[1]

        validate_pixel_grid_alignment(out)
        print("[OK] _test_group_additions_and_token_append")
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    # Quick check for generate_arrays
    arr = generate_arrays(n=17, m=5)
    assert sum(a * b * c for a, b, c in arr) == 17

    _test_basic_alignment_and_padding(processor)
    _test_group_additions_and_token_append(processor)
    print("All tests passed.")
