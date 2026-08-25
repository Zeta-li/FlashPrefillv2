"""flash_attn_func window/normal correctness sweep vs float64 eager.

Covers the boundary cases test_swa_sink.py (single window 511) does not:
normal full causal, window smaller than the 128-wide tile, non-tile-aligned
window, odd sequence lengths, LQ == LKV, and non-causal symmetric windows.

FA window semantics (docstring of flash_attn_func): query i attends keys in
[i + seqlen_k - seqlen_q - left, i + seqlen_k - seqlen_q + right]; causal=True
is equivalent to right = 0.

Run:  python test_window_sweep.py
"""

import math
import sys

import torch

from flashprefill.flash_attn_interface import flash_attn_func

DEV = "cuda"


def eager(q_b, k, v, left, right, causal, scale):
    """float64 reference; q_b (LQ,HQ,D), k/v (LKV,HKV,D)."""
    LQ, HQ, _ = q_b.shape
    LKV, HKV, _ = k.shape
    g = HQ // HKV
    kx = k.repeat_interleave(g, dim=1)
    vx = v.repeat_interleave(g, dim=1)
    s = torch.einsum("qhd,khd->hqk", q_b, kx) * scale
    row = torch.arange(LQ, device=q_b.device)[:, None]
    col = torch.arange(LKV, device=q_b.device)[None, :]
    limit = row + LKV - LQ  # bottom-right alignment
    mask = torch.ones_like(s, dtype=torch.bool)
    if left >= 0:
        mask &= col >= (limit - left)
    eff_right = 0 if causal else right
    if eff_right >= 0:
        mask &= col <= (limit + eff_right)
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, vx)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  ({detail})")
    return cond


def run(HQ, HKV, D, LQ, LKV, cases, tol):
    tag = f"H{HQ}/{HKV} D{D} L{LQ}/{LKV}"
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(LQ, HQ, D, device=DEV).to(torch.bfloat16)
    k = torch.randn(LKV, HKV, D, device=DEV).to(torch.bfloat16)
    v = torch.randn(LKV, HKV, D, device=DEV).to(torch.bfloat16)
    ok = True
    for (left, right, causal) in cases:
        ref = eager(q.double(), k.double(), v.double(), left, right, causal, scale)
        out = flash_attn_func(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            causal=causal, window_size=(left, right), num_splits=1,
        ).squeeze(0)
        rel = ((out.double() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        ok &= check(f"[{tag}] window=({left},{right}) causal={causal}",
                    rel < tol, f"rel={rel:.2e}")
    return ok


def main():
    ok = True
    tol = 2e-2
    # GQA, head_dim 256 (welm), odd lengths stress tile/mask boundaries
    ok &= run(24, 2, 256, 320, 4099, [
        (-1, -1, True),      # normal full causal
        (511, 0, True),      # welm SWA
        (63, 0, True),       # window < tile width (128)
        (100, 0, True),      # non-tile-aligned window
        (5000, 0, True),     # window >= LKV (degenerates to full causal)
    ], tol)
    # LQ == LKV equal-length causal + small window
    ok &= run(8, 2, 128, 777, 777, [
        (-1, -1, True),
        (31, 0, True),
        (776, 0, True),
    ], tol)
    # non-causal symmetric window (both sides), GQA
    ok &= run(8, 2, 128, 256, 256, [
        (64, 64, False),
        (0, 0, False),       # diagonal only
    ], tol)
    print("=" * 60)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
