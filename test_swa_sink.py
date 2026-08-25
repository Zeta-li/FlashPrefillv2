"""SWA (sliding-window / local attention) correctness test, with and without
learnable sinks, in BF16 and FP8.

The welm SWA layers use window=512 in transformers semantics (current token
plus the 511 tokens to its left), i.e. FA window_size=(511, 0) under
bottom-right causal alignment.

Reference is float64 eager with the same window mask; the sink is folded into
the softmax denominator. Run:  python test_swa_sink.py
"""

import math
import sys

import torch
import torch.nn.functional as F

from flashprefill.flash_attn_interface import (
    flash_attn_func,
    flash_attn_with_kvcache,
)

DEV = "cuda"
HQ, HKV, D = 24, 2, 128
LQ, LKV = 512, 32768
W = 512                       # transformers-style window size
SCALE = 1.0 / math.sqrt(D)


def eager_swa_sink(q_b, k, v, sink):
    g = HQ // HKV
    kx = k.repeat_interleave(g, dim=1)
    vx = v.repeat_interleave(g, dim=1)
    s = torch.einsum("qhd,khd->hqk", q_b, kx) * SCALE
    row = torch.arange(LQ, device=q_b.device)[:, None]
    col = torch.arange(LKV, device=q_b.device)[None, :]
    limit = row + LKV - LQ                       # bottom-right causal limit
    mask = (col <= limit) & (col > limit - W)    # sliding window
    s = s.masked_fill(~mask, float("-inf"))
    m = s.max(dim=-1, keepdim=True).values
    p = torch.exp(s - m)
    denom = p.sum(-1, keepdim=True)
    if sink is not None:
        denom = denom + torch.exp(sink.double().view(HQ, 1, 1) - m)
    return torch.einsum("hqk,khd->qhd", p / denom, vx)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  ({detail})")
    return cond


def run(dtype, use_sink, tol):
    tag = "bf16" if dtype == torch.bfloat16 else "fp8"
    torch.manual_seed(0)
    q = torch.randn(LQ, HQ, D, device=DEV).to(dtype)
    k = torch.randn(LKV, HKV, D, device=DEV).to(dtype)
    v = torch.randn(LKV, HKV, D, device=DEV).to(dtype)
    sinks = (torch.randn(HQ, dtype=torch.float32) * 0.5).to(dtype).to(DEV) if use_sink else None
    ref = eager_swa_sink(q.double(), k.double(), v.double(),
                         sinks.double() if use_sink else None)

    # A) dense flash_attn_func with window
    out = flash_attn_func(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        causal=True, window_size=(W - 1, 0), num_splits=1,
        sinks=sinks, return_attn_probs=False).squeeze(0)
    rel = ((out.double() - ref).norm() / ref.norm()).item()

    # B) paged with_kvcache with window (SWA serving path)
    k_cache = k.unsqueeze(1).contiguous()
    v_cache = v.unsqueeze(1).contiguous()
    page_table = torch.arange(LKV, dtype=torch.int32, device=DEV).unsqueeze(0)
    out2 = flash_attn_with_kvcache(
        q, k_cache, v_cache, page_table=page_table,
        cache_seqlens=torch.tensor([LKV], dtype=torch.int32, device=DEV),
        cu_seqlens_q=torch.tensor([0, LQ], dtype=torch.int32, device=DEV),
        max_seqlen_q=LQ, causal=True, window_size=(W - 1, 0),
        num_splits=1, sinks=sinks)
    rel2 = ((out2.double() - ref).norm() / ref.norm()).item()

    ok = True
    ok &= check(f"[{tag}] dense window_size=({W-1},0) sink={use_sink}", rel < tol, f"rel={rel:.2e}")
    ok &= check(f"[{tag}] paged window_size=({W-1},0) sink={use_sink}", rel2 < tol, f"rel={rel2:.2e}")
    return ok


def main():
    ok = True
    ok &= run(torch.bfloat16, use_sink=False, tol=2e-2)
    ok &= run(torch.bfloat16, use_sink=True, tol=2e-2)
    ok &= run(torch.float8_e4m3fn, use_sink=False, tol=8e-2)
    ok &= run(torch.float8_e4m3fn, use_sink=True, tol=8e-2)
    print("=" * 60)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
