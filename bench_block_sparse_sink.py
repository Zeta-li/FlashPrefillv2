"""Block-sparse forward bench: pre-patch vs patched (per-row sink) kernel.

Self-contained; forward only (no bwd). Uses the installed package by default:

    python bench_block_sparse_sink.py                       # installed (patched) build
    python bench_block_sparse_sink.py <old_pkg_parent_dir>  # pre-patch build

Each run times the paged block-sparse forward with sinks off and on, so the
two runs together show (a) no-sink regression from the patch and (b) the cost
of the sink path itself.
"""

import os
import sys

if len(sys.argv) > 1:
    sys.path.insert(0, os.path.abspath(sys.argv[1]))

import torch  # noqa: E402

from flashprefill.flash_attn_interface import flash_attn_with_kvcache  # noqa: E402
from flash_block_sparse_index_triton import (  # noqa: E402
    SparseIndexWorkspace,
    build_block_sparse_index_fast,
)

DEV = "cuda"
DTYPES = [torch.bfloat16, torch.float8_e4m3fn]
BLOCK_M, BLOCK_N, SUB = 128, 128, 2
HQ, HKV, D = 24, 2, 256
LQ = 512


def timed(fn, iters=20):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    g = HQ // HKV
    import flashprefill
    print(f"pkg={os.path.dirname(flashprefill.__file__)}")
    for dtype in DTYPES:
        tag = "bf16" if dtype == torch.bfloat16 else "fp8"
        sinks = torch.zeros(HQ, dtype=dtype, device=DEV)
        for lkv in [4096, 32768, 131072]:
            torch.manual_seed(0)
            q = torch.randn(LQ, HQ, D, device=DEV).to(dtype)
            k_cache = torch.randn(lkv, 1, HKV, D, device=DEV).to(dtype)
            v_cache = torch.randn(lkv, 1, HKV, D, device=DEV).to(dtype)
            page_table = torch.arange(lkv, dtype=torch.int32, device=DEV).unsqueeze(0)
            cache_seqlens = torch.tensor([lkv], dtype=torch.int32, device=DEV)
            cu_q = torch.tensor([0, LQ], dtype=torch.int32, device=DEV)
            n_tiles = (LQ * g + BLOCK_M - 1) // BLOCK_M
            cu_q_tiles = torch.tensor([0, n_tiles], dtype=torch.int32, device=DEV)
            ws = SparseIndexWorkspace(
                batch_size=1, num_kv_heads=HKV, head_dim=D,
                total_q_tiles=n_tiles, max_q_tiles=n_tiles,
                max_k_blocks=(lkv + BLOCK_N - 1) // BLOCK_N,
                dtype=dtype, device=DEV, cu_q_tiles=cu_q_tiles,
                n_sub=SUB, use_mean_correction=True)
            cu, idx, total_tiles, cu_q_tiles = build_block_sparse_index_fast(
                q, k_cache, page_table, cache_seqlens, cu_q, ws,
                v_cache=v_cache, k_block_m=BLOCK_M, k_block_n=BLOCK_N,
                abs_threshold=0.1, attention_sink=2, window_size=4,
                last_n_blocks=8, causal=True)
            sparse = dict(page_table=page_table, cache_seqlens=cache_seqlens,
                          cu_seqlens_q=cu_q, max_seqlen_q=LQ, causal=True,
                          num_splits=1, block_sparse_cu=cu, block_sparse_idx=idx,
                          total_q_tiles=total_tiles, cu_q_tiles=cu_q_tiles,
                          k_mean=ws.k_mean, v_mean=ws.v_mean,
                          mean_k_block_size=BLOCK_N)
            t_off = timed(lambda: flash_attn_with_kvcache(q, k_cache, v_cache, **sparse))
            t_on = timed(lambda: flash_attn_with_kvcache(q, k_cache, v_cache, sinks=sinks, **sparse))
            print(f"[{tag}] kv_len={lkv:>7d}  sinks_off={t_off:8.3f} ms  sinks_on={t_on:8.3f} ms")


if __name__ == "__main__":
    main()
