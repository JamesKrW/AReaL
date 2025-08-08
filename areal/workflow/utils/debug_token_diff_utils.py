import os

# Toggle with env var: export AREAL_DEBUG_TOKEN_ALIGN=1
DEBUG_TOKEN_ALIGNMENT = os.getenv("AREAL_DEBUG_TOKEN_ALIGN", "0") not in ("", "0", "false", "False")

def _first_mismatch(a, b):
    """
    Return the first index where two sequences differ, or None if all equal (up to min length).
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n  # Diverge at the end (one is a prefix of the other)
    return None

def _safe_decode(tokenizer, ids):
    """
    Decode without throwing; keep special tokens so we can see template markers.
    """
    try:
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except Exception as e:
        return f"<decode-error: {e}>"

def debug_diff_token_sequences(tokenizer, left, right, left_name="left", right_name="right", ctx=24):
    """
    Print a compact diff between two token sequences around the first mismatch.
    """
    i = _first_mismatch(left, right)
    print("=" * 80)
    print("[TOKEN ALIGNMENT DEBUG]")
    print(f"{left_name}: len={len(left)} | {right_name}: len={len(right)}")
    if i is None:
        print("No element-wise mismatch. Lengths may differ or sequences identical.")
        print("=" * 80)
        return

    lo = max(0, i - ctx)
    hiL = min(len(left), i + ctx)
    hiR = min(len(right), i + ctx)

    print(f"First mismatch at index: {i}")
    print(f"\n-- {left_name}[{lo}:{hiL}] tokens:\n{left[lo:hiL]}")
    print(f"-- {right_name}[{lo}:{hiR}] tokens:\n{right[lo:hiR]}")

    print("\n-- Decoded windows (keep specials) --")
    print(f"{left_name} decoded:\n{_safe_decode(tokenizer, left[lo:hiL])}")
    print(f"{right_name} decoded:\n{_safe_decode(tokenizer, right[lo:hiR])}")

    # Also show tails to detect extra prefixes/suffixes
    tailw = min(64, max(len(left) - i, len(right) - i))
    print("\n-- Tails from mismatch --")
    print(f"{left_name}[{i}:{i+tailw}] tokens:\n{left[i:i+tailw]}")
    print(f"{right_name}[{i}:{i+tailw}] tokens:\n{right[i:i+tailw]}")
    print(f"{left_name} tail decoded:\n{_safe_decode(tokenizer, left[i:i+tailw])}")
    print(f"{right_name} tail decoded:\n{_safe_decode(tokenizer, right[i:i+tailw])}")
    print("=" * 80)
