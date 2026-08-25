"""Per-head learnable-sink correctness test for the PackGQA kernel paths,
in both BF16 and FP8 (e4m3).

Background: sinks are per QUERY head, but a PackGQA M tile packs rows of G
different query heads; the old Has_Sink path applied one scalar sink per tile
(sinks[kv_head]), which is wrong for every head except one when G > 1. The
fix gathers the sink logit per fragment row (q_head = kv_head*g + R % g).

Strategy: give every head a DISTINCT sink value so a per-head mix-up shows up
immediately, check the exact algebraic identity
    out_with_sink == out_sink_free * sigmoid(lse_sink_free - sink)
and compare against a float64 eager reference, with per-head error breakdown
(the dense reference is exact, so per-head errors attribute cleanly).

Coverage: A) dense path (public API; packing decided internally);
B) block-sparse paged path with mean correction -- the PackGQA path that the
per-row sink fix targets (paged + block-sparse forces PackGQA).

Run:  python test_sinks_packgqa.py
"""

import math
import sys

import torch
import torch.nn.functional as F

from flashprefill.flash_attn_interface import (
    flash_attn_func,
    flash_attn_with_kvcache,
)
from flash_block_sparse_index_triton import (
    SparseIndexWorkspace,
    build_block_sparse_index_fast,
)

DEV = "cuda"
BLOCK_M, BLOCK_N, SUB = 128, 128, 2
HQ, HKV, D = 24, 2, 256          # GQA ratio 8, divides BLOCK_M
LQ, LKV = 512, 8192
SCALE = 1.0 / math.sqrt(D)


def eager_with_sink(q_b, k, v, sink):
    """float64 causal reference, sink folded into the softmax denominator."""
    g = HQ // HKV
    kx = k.repeat_interleave(g, dim=1)
    vx = v.repeat_interleave(g, dim=1)
    s = torch.einsum("qhd,khd->hqk", q_b, kx) * SCALE
    row = torch.arange(LQ, device=q_b.device)[:, None]
    col = torch.arange(LKV, device=q_b.device)[None, :]
    s = s.masked_fill(col > (row + LKV - LQ), float("-inf"))
    m = s.max(dim=-1, keepdim=True).values
    p = torch.exp(s - m)
    denom = p.sum(-1, keepdim=True) + torch.exp(sink.double().view(HQ, 1, 1) - m)
    return torch.einsum("hqk,khd->qhd", p / denom, vx)  # (LQ, HQ, D)


def per_head_err(out, ref):
    d = (out.double() - ref).norm(dim=(0, 2)) / ref.norm(dim=(0, 2)).clamp_min(1e-9)
    return d  # (HQ,)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  ({detail})")
    return cond


def run_dtype(dtype, id_tol, eager_tol, sparse_tol):
    """Full sink test for one dtype. id_tol: algebraic identity;
    eager_tol: dense vs float64 eager; sparse_tol: sparse sanity."""
    tag = "bf16" if dtype == torch.bfloat16 else "fp8"
    ok = True
    torch.manual_seed(0)
    # Distinct per-head sinks: any per-head mix-up is immediately visible.
    # (kernel requires sinks in the same dtype as q; fp8 rounds them coarsely,
    # and every check below uses the same rounded values)
    sinks = (torch.arange(HQ, dtype=torch.float32) * 0.7 - 5.0).to(dtype).to(DEV)
    sinks_ref = sinks.double()  # exact values of the rounded sinks

    q = torch.randn(LQ, HQ, D, device=DEV).to(dtype)
    k = torch.randn(LKV, HKV, D, device=DEV).to(dtype)
    v = torch.randn(LKV, HKV, D, device=DEV).to(dtype)
    ref = eager_with_sink(q.double(), k.double(), v.double(), sinks_ref)

    # ---- A. dense path (public API; packing decided internally) ----
    out_s, lse_s = flash_attn_func(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        causal=True, num_splits=1, sinks=sinks,
        return_attn_probs=True)
    out_0, lse_0 = flash_attn_func(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        causal=True, num_splits=1,
        return_attn_probs=True)
    out_s, out_0 = out_s.squeeze(0), out_0.squeeze(0)

    comp = out_0.double() * torch.sigmoid(
        lse_0.squeeze(0).t().unsqueeze(-1) - sinks_ref.view(1, HQ, 1))
    rel = ((out_s.double() - comp).norm() / comp.norm()).item()
    ok &= check(f"[{tag}] dense: sinks vs sink-free*sigmoid(lse-s)",
                rel < id_tol, f"rel={rel:.2e}")

    errs = per_head_err(out_s, ref)
    ok &= check(f"[{tag}] dense: sinks vs float64 eager (per head)",
                errs.max().item() < eager_tol,
                f"max_rel={errs.max().item():.2e} at head {errs.argmax().item()}")

    # ---- B. block-sparse paged path with mean correction ----
    g = HQ // HKV
    k_cache = k.unsqueeze(1).contiguous()          # (LKV, 1, HKV, D) page_size=1
    v_cache = v.unsqueeze(1).contiguous()
    page_table = torch.arange(LKV, dtype=torch.int32, device=DEV).unsqueeze(0)
    cache_seqlens = torch.tensor([LKV], dtype=torch.int32, device=DEV)
    cu_q = torch.tensor([0, LQ], dtype=torch.int32, device=DEV)
    n_tiles = (LQ * g + BLOCK_M - 1) // BLOCK_M
    cu_q_tiles = torch.tensor([0, n_tiles], dtype=torch.int32, device=DEV)
    ws = SparseIndexWorkspace(
        batch_size=1, num_kv_heads=HKV, head_dim=D,
        total_q_tiles=n_tiles, max_q_tiles=n_tiles,
        max_k_blocks=(LKV + BLOCK_N - 1) // BLOCK_N,
        dtype=dtype, device=DEV, cu_q_tiles=cu_q_tiles,
        n_sub=SUB, use_mean_correction=True)
    cu, idx, total_tiles, cu_q_tiles = build_block_sparse_index_fast(
        q, k_cache, page_table, cache_seqlens, cu_q, ws,
        v_cache=v_cache, k_block_m=BLOCK_M, k_block_n=BLOCK_N,
        abs_threshold=0.1, attention_sink=2, window_size=4, last_n_blocks=8,
        causal=True)
    sparse = dict(page_table=page_table, cache_seqlens=cache_seqlens,
                  cu_seqlens_q=cu_q, max_seqlen_q=LQ, causal=True, num_splits=1,
                  block_sparse_cu=cu, block_sparse_idx=idx,
                  total_q_tiles=total_tiles, cu_q_tiles=cu_q_tiles,
                  k_mean=ws.k_mean, v_mean=ws.v_mean,
                  mean_k_block_size=BLOCK_N, return_softmax_lse=True)
    out_ss, lse_ss, _, _ = flash_attn_with_kvcache(q, k_cache, v_cache, sinks=sinks, **sparse)
    out_s0, lse_s0, _, _ = flash_attn_with_kvcache(q, k_cache, v_cache, **sparse)

    comp = out_s0.double() * torch.sigmoid(
        lse_s0.t().unsqueeze(-1) - sinks_ref.view(1, HQ, 1))
    rel = ((out_ss.double() - comp).norm() / comp.norm()).item()
    ok &= check(f"[{tag}] sparse+correction: sinks vs sink-free*sigmoid(lse-s)",
                rel < id_tol, f"rel={rel:.2e}")

    # sparse is approximate vs dense reference: only a loose sanity bound here
    errs = per_head_err(out_ss, ref)
    ok &= check(f"[{tag}] sparse+correction: sinks vs float64 eager (sanity)",
                errs.max().item() < sparse_tol,
                f"max_rel={errs.max().item():.2e} at head {errs.argmax().item()}")
    return ok


def main():
    ok = True
    # identity tolerances: bf16 noise ~1e-3; fp8 rounds sinks/inputs coarsely
    ok &= run_dtype(torch.bfloat16, id_tol=1e-2, eager_tol=2e-2, sparse_tol=0.10)
    ok &= run_dtype(torch.float8_e4m3fn, id_tol=3e-2, eager_tol=8e-2, sparse_tol=0.20)
    print("=" * 60)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
