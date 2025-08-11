#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
validate_vlm_delta_vs_full_qwen25vl.py

Compare incremental (per-turn) vs one-shot tokenization for multi-turn, multi-image
conversations on Qwen2.5-VL-style processors, including:
- input_ids equality (delta concat vs one-shot)
- pixel_values equality (per-turn concat along dim=0 vs one-shot)
- image_grid_thw equality (per-turn concat along dim=0 vs one-shot)
- optional per-image pixel consistency (single-image vs its slice in batch)

Run:
  python validate_vlm_delta_vs_full_qwen25vl.py \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --image /path/to/img.png \
    --rounds 4
"""

import argparse
import random
import re
from typing import List, Dict, Tuple, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


# ------------------- Image & text helpers ------------------- #

VISION_START = "<|vision_start|>"
VISION_PAD   = "<|image_pad|>"
VISION_END   = "<|vision_end|>"

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def get_text_with_image_token(text_content: str, processor) -> str:
    """Replace '<image>' placeholder with the model's actual image token sequence."""
    if processor is None:
        image_token = "<image>"
    else:
        iproc = getattr(processor, "image_processor", None)
        if iproc and hasattr(iproc, "image_processor_type") and \
           "qwen" in str(iproc.image_processor_type).lower():
            image_token = f"{VISION_START}{VISION_PAD}{VISION_END}"
        else:
            image_token = getattr(processor, "image_token", "<image>")
    return text_content.replace("<image>", image_token)

def count_image_segments(serialized_text: str, processor) -> int:
    """Count image segments in serialized text (Qwen: count '<|vision_start|>')."""
    iproc = getattr(processor, "image_processor", None)
    if iproc and hasattr(iproc, "image_processor_type") and \
       "qwen" in str(iproc.image_processor_type).lower():
        return len(re.findall(r"<\|vision_start\|>", serialized_text))
    image_tok = getattr(processor, "image_token", "<image>")
    return serialized_text.count(image_tok)

def tokenize_with_processor(processor, text: str, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
    """
    processor(text, images) -> dict with at least input_ids/attention_mask.
    May also include pixel_values and image_grid_thw depending on the processor.
    """
    out = processor(text=text, images=images if images else None,
                    padding=False, return_tensors="pt")
    return out


# ------------------- Random message generation ------------------- #

_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
    "pi rho sigma tau upsilon phi chi psi omega quick brown fox jumps over lazy dog "
    "neural vision language model prompt reward rollout environment token image video"
).split()

def random_text(n_words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))

def random_user_with_image_marker(rng: random.Random) -> str:
    """Build random user text and insert one '<image>' marker."""
    n_left = rng.randint(3, 8)
    n_right = rng.randint(3, 8)
    left = random_text(n_left, rng)
    right = random_text(n_right, rng)
    return f"{left} <image> {right}"


# ------------------- Utilities ------------------- #

def compare_id_lists(a: List[int], b: List[int]) -> Tuple[bool, int]:
    """Compare token ID lists; return (equal?, first_mismatch_index or -1)."""
    if len(a) != len(b):
        L = min(len(a), len(b))
        for i in range(L):
            if a[i] != b[i]:
                return False, i
        return False, L
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return False, i
    return True, -1

def first_tensor_mismatch(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, float, Tuple[int, ...]]:
    """
    Compare tensors by shape and values.
    Return (equal?, max_abs_diff, mismatch_shape_tuple)
    """
    if a is None and b is None:
        return True, 0.0, tuple()
    if (a is None) != (b is None):
        return False, float("inf"), tuple()
    if a.shape != b.shape:
        return False, float("inf"), a.shape
    diff = (a - b).abs()
    return bool((diff.max() == 0)), float(diff.max().item()), a.shape

def split_by_images_flat(pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> List[torch.Tensor]:
    """
    Qwen2.5-VL fast processor typically returns flattened visual tokens:
      pixel_values: (sum_tokens, D)
      image_grid_thw: (N, 3) with (T, H, W) per image
    We split the flat sequence into N chunks, proportionally to T*H*W.

    Note: actual token count may include an internal merge factor. We do proportional
    splitting and fix rounding errors on the last chunk.
    """
    assert pixel_values.dim() == 2, f"Expect (sum_tokens, D), got {tuple(pixel_values.shape)}"
    assert image_grid_thw.dim() == 2 and image_grid_thw.size(-1) == 3, \
        f"Expect (N,3), got {tuple(image_grid_thw.shape)}"

    thw = image_grid_thw.tolist()
    counts = [int(t*h*w) for t, h, w in thw]
    total_units = sum(counts)
    L = pixel_values.shape[0]
    # map proportions to actual length L
    scaled = [round(c / total_units * L) for c in counts]
    # fix rounding drift
    drift = L - sum(scaled)
    if drift != 0:
        scaled[-1] += drift

    chunks, st = [], 0
    for n in scaled:
        chunks.append(pixel_values[st:st+n])
        st += n
    return chunks


# ------------------- Core builders ------------------- #

def build_incremental_ids_via_deltas(
    processor,
    messages: List[Dict[str, str]],
    per_turn_images: List[List[Image.Image]],
    verbose: bool = True,
) -> List[int]:
    """
    Token IDs only (delta-based, cumulative text each turn).
    """
    ptok = processor.tokenizer
    inc_ids: List[int] = []
    all_images: List[Image.Image] = []

    for i in range(len(messages)):
        if i == 0:
            prev_ids = []
            k_prev = 0
        else:
            prev_text = ptok.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=False)
            k_prev = count_image_segments(prev_text, processor)
            prev_out = tokenize_with_processor(processor, prev_text, all_images[:k_prev])
            prev_ids = prev_out["input_ids"][0].tolist()

        curr_text = ptok.apply_chat_template(messages[:i+1], tokenize=False, add_generation_prompt=False)
        k_curr = count_image_segments(curr_text, processor)

        new_imgs = per_turn_images[i] if i < len(per_turn_images) else []
        assert k_curr == k_prev + len(new_imgs), \
            f"Image count mismatch at turn {i}: k_curr={k_curr}, k_prev={k_prev}, new={len(new_imgs)}"

        curr_out = tokenize_with_processor(processor, curr_text, all_images[:k_prev] + new_imgs)
        curr_ids = curr_out["input_ids"][0].tolist()

        assert curr_ids[:len(prev_ids)] == prev_ids, \
            f"Prefix mismatch at turn {i} — tokenizer/template misalignment."

        delta = curr_ids[len(prev_ids):]
        if verbose:
            print(f"[turn {i}] delta_len={len(delta)} (role={messages[i]['role']})")
        inc_ids += delta

        if new_imgs:
            all_images.extend(new_imgs)

    return inc_ids


def build_per_turn_visual_concat(
    processor,
    messages: List[Dict[str, str]],
    per_turn_images: List[List[Image.Image]],
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Visual side ONLY (per-turn processing → concat along dim=0).
    For each turn i, call processor(text=messages[i]['content'], images=per_turn_images[i])
    and collect pixel_values + image_grid_thw if present.
    """
    pv_list, thw_list = [], []
    for i, (msg, imgs) in enumerate(zip(messages, per_turn_images)):
        if len(imgs) == 0:
            continue
        out = tokenize_with_processor(processor, msg["content"], imgs)
        if "pixel_values" not in out or "image_grid_thw" not in out:
            if verbose:
                print(f"[turn {i}] (role={msg['role']}) has images but processor did not return pixel_values/image_grid_thw; skipping visual concat.")
            continue
        pv = out["pixel_values"]       # (sum_tokens_turn, D)
        thw = out["image_grid_thw"]    # (n_images_turn, 3)
        if pv is not None and pv.numel() > 0:
            pv_list.append(pv)
            thw_list.append(thw)

    if len(pv_list) == 0:
        return None, None
    pixel_values   = torch.cat(pv_list, dim=0)      # (sum_tokens_all, D)
    image_grid_thw = torch.cat(thw_list, dim=0)     # (sum_images_all, 3)
    return pixel_values, image_grid_thw


def build_one_shot_all(
    processor,
    messages: List[Dict[str, str]],
    all_images: List[Image.Image],
) -> Dict[str, torch.Tensor]:
    """
    One-shot processor on the full conversation (no generation prompt).
    Returns dict with input_ids/attention_mask and, if available, pixel_values/image_grid_thw.
    """
    ptok = processor.tokenizer
    full_text = ptok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    k_full = count_image_segments(full_text, processor)
    assert k_full == len(all_images), f"Final image count mismatch: k_full={k_full}, images={len(all_images)}"
    out = tokenize_with_processor(processor, full_text, all_images)
    return out


def per_image_pixel_consistency(
    processor,
    images: List[Image.Image],
    verbose: bool = True,
) -> List[float]:
    """
    Compare each image processed alone vs its slice in a batch.
    Uses proportional split by image_grid_thw (T*H*W).
    """
    diffs: List[float] = []

    # Batch first
    dummy = " ".join(["<image>"] * len(images))
    full_text = get_text_with_image_token(dummy, processor)
    batch = tokenize_with_processor(processor, full_text, images)

    if "pixel_values" not in batch or "image_grid_thw" not in batch:
        if verbose:
            print(f"(keys: {list(batch.keys())}) pixel_values/image_grid_thw not available; skip per-image check.")
        return []

    pv_b = batch["pixel_values"]          # (L, D)
    thw_b = batch["image_grid_thw"]       # (N, 3)
    chunks_b = split_by_images_flat(pv_b, thw_b)

    # Singles
    single_text = get_text_with_image_token("<image>", processor)
    for i, img in enumerate(images):
        single = tokenize_with_processor(processor, single_text, [img])
        if "pixel_values" not in single or "image_grid_thw" not in single:
            diffs.append(float("nan"))
            continue
        pv_s = single["pixel_values"]     # (l_i, D)
        # Align to min length to tolerate ±1 rounding in proportional split
        n = min(pv_s.shape[0], chunks_b[i].shape[0])
        diffs.append(float((pv_s[:n] - chunks_b[i][:n]).abs().max().item()))
    return diffs


# ------------------- Public entry ------------------- #

def run_validation(
    processor,
    image_path: str,
    rounds: int = 4,
    seed: int = 1234,
    verbose: bool = True,
):
    rng = random.Random(seed)
    img = load_image(image_path)

    # Build messages + per-turn images
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    per_turn_images: List[List[Image.Image]] = [[]]  # system: no images

    # user(<image>) -> assistant -> user(<image>) -> assistant ...
    for r in range(rounds):
        u = random_user_with_image_marker(rng)
        u = get_text_with_image_token(u, processor)  # replace placeholder
        messages.append({"role": "user", "content": u})
        per_turn_images.append([img])  # one image at this user turn

        a = random_text(rng.randint(5, 12), rng)
        messages.append({"role": "assistant", "content": a})
        per_turn_images.append([])

    # Build all_images in encounter order
    all_images: List[Image.Image] = []
    for lst in per_turn_images:
        all_images.extend(lst)

    # (A) Token IDs: incremental (delta on cumulative text)
    inc_ids = build_incremental_ids_via_deltas(
        processor, messages, per_turn_images, verbose=verbose
    )

    # (B) Token IDs + Visuals: one-shot on full conversation
    full_out = build_one_shot_all(processor, messages, all_images)
    full_ids = full_out["input_ids"][0].tolist()

    # Compare token ids
    equal_ids, idx = compare_id_lists(inc_ids, full_ids)
    print("\n=== TOKEN IDS EQUALITY ===")
    print(f"Equal? {equal_ids}")
    print(f"Lengths: incremental={len(inc_ids)}, full={len(full_ids)}")
    if not equal_ids:
        print(f"First mismatch at index {idx}: inc={inc_ids[idx]}, full={full_ids[idx]}")

    # (C) Visuals: per-turn concat (our required approach)
    perturn_pv, perturn_thw = build_per_turn_visual_concat(
        processor, messages, per_turn_images, verbose=verbose
    )

    # Extract one-shot visuals (if provided)
    full_pv  = full_out.get("pixel_values", None)
    full_thw = full_out.get("image_grid_thw", None)

    # Compare grid_thw
    print("\n=== IMAGE_GRID_THW COMPARISON ===")
    if perturn_thw is None or full_thw is None:
        print("image_grid_thw not available from processor; skipping.")
        equal_thw = None
    else:
        same_thw = (perturn_thw.shape == full_thw.shape) and torch.equal(perturn_thw, full_thw)
        print(f"Equal? {same_thw} | per-turn shape={tuple(perturn_thw.shape)} | one-shot shape={tuple(full_thw.shape)}")
        if not same_thw:
            # show small diff if shapes match but values differ
            if perturn_thw.shape == full_thw.shape:
                print("Max |Δ| on thw:", (perturn_thw - full_thw).abs().max().item())
        equal_thw = bool(same_thw)

    # Compare pixel_values
    print("\n=== PIXEL_VALUES COMPARISON ===")
    if perturn_pv is None or full_pv is None:
        print("pixel_values not available from processor; skipping.")
        equal_pv = None
        pv_max_delta = None
    else:
        if perturn_pv.shape != full_pv.shape:
            print(f"Shapes differ: per-turn={tuple(perturn_pv.shape)} vs one-shot={tuple(full_pv.shape)}")
            equal_pv = False
            pv_max_delta = float("inf")
        else:
            diff = (perturn_pv - full_pv).abs()
            pv_max_delta = float(diff.max().item())
            equal_pv = bool(pv_max_delta == 0.0)
            print(f"Equal? {equal_pv} | shape={tuple(full_pv.shape)} | max|Δ|={pv_max_delta:.6g}")

    # (D) Per-image pixel consistency: single vs its slice in one-shot batch
    diffs = []
    if full_pv is not None and full_thw is not None:
        # repeat the same image 'rounds' times (since we inserted one per user turn)
        diffs = per_image_pixel_consistency(processor, [img] * rounds, verbose=verbose)
        if diffs:
            print("\n=== PER-IMAGE PIXEL CONSISTENCY (single vs its slice in batch) ===")
            for i, d in enumerate(diffs, 1):
                print(f"Image #{i}: max|Δ| = {d:.6g}")

    return {
        "equal_ids": equal_ids,
        "first_mismatch_idx": (-1 if equal_ids else idx),
        "len_incremental": len(inc_ids),
        "len_full": len(full_ids),
        "equal_thw": equal_thw,
        "equal_pv": equal_pv,
        "pv_max_delta": pv_max_delta if perturn_pv is not None and full_pv is not None else None,
        "per_image_pixel_diffs": diffs,
    }


# ------------------- CLI ------------------- #

def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="HF model id, e.g., Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to an image file.")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_fast", action="store_true",
                        help="Force fast processor if supported (HF defaults may already be fast).")
    args = parser.parse_args()

    # Load processor (and tokenizer implicitly)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=args.use_fast)
    if not hasattr(processor, "tokenizer"):
        # Fallback: ensure tokenizer exists
        _ = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

    run_validation(processor, args.image, rounds=args.rounds, seed=args.seed)


if __name__ == "__main__":
    _cli()
