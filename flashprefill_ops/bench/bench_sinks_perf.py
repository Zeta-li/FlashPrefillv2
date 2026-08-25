"""Performance check: sinks on vs sinks off on the block-sparse paged path.

The per-row sink fix runs once per M-tile in the epilogue, so the expected
delta is unmeasurable (<1%). If it is larger, the sink gather should be
moved to an smem prefetch per CTA.

Run:  python bench/bench_sinks_perf.py
"""

import torch

from flashprefill.flash_attn_interface import flash_attn_with_kvcache
from flash_block_sparse_index_triton import (
    SparseIndexWorkspace,
    build_block_sparse_index_fast,
)

DEV = "cuda"
DTYPE = torch.bfloat16
BLOCK_M, BLOCK_N, SUB = 128, 128, 2
HQ, HKV, D = 32, 4, 128
LQ = 512


def bench(lkv, sinks, index_pack, iters=20):
    q = torch.randn(LQ, HQ, D, dtype=DTYPE, device=DEV)
    k_cache, v_cache, page_table, cache_seqlens, cu_q = index_pack["static"]
    sparse = dict(index_pack["sparse"])
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    # warmup
    for _ in range(5):
        flash_attn_with_kvcache(q, k_cache, v_cache, sinks=sinks, **sparse)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start.record()
        flash_attn_with_kvcache(q, k_cache, v_cache, sinks=sinks, **sparse)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def main():
    g = HQ // HKV
    for lkv in [4096, 32768, 131072]:
        k_cache = torch.randn(lkv, 1, HKV, D, dtype=DTYPE, device=DEV)
        v_cache = torch.randn(lkv, 1, HKV, D, dtype=DTYPE, device=DEV)
        page_table = torch.arange(lkv, dtype=torch.int32, device=DEV).unsqueeze(0)
        cache_seqlens = torch.tensor([lkv], dtype=torch.int32, device=DEV)
        cu_q = torch.tensor([0, LQ], dtype=torch.int32, device=DEV)
        q = torch.randn(LQ, HQ, D, dtype=DTYPE, device=DEV)
        n_tiles = (LQ * g + BLOCK_M - 1) // BLOCK_M
        cu_q_tiles = torch.tensor([0, n_tiles], dtype=torch.int32, device=DEV)
        ws = SparseIndexWorkspace(
            batch_size=1, num_kv_heads=HKV, head_dim=D,
            total_q_tiles=n_tiles, max_q_tiles=n_tiles,
            max_k_blocks=(lkv + BLOCK_N - 1) // BLOCK_N,
            dtype=DTYPE, device=DEV, cu_q_tiles=cu_q_tiles,
            n_sub=SUB, use_mean_correction=True)
        cu, idx, total_tiles, cu_q_tiles = build_block_sparse_index_fast(
            q, k_cache, page_table, cache_seqlens, cu_q, ws,
            v_cache=v_cache, k_block_m=BLOCK_M, k_block_n=BLOCK_N,
            abs_threshold=0.1, attention_sink=2, window_size=4,
            last_n_blocks=8, causal=True)
        index_pack = {
            "static": (k_cache, v_cache, page_table, cache_seqlens, cu_q),
            "sparse": dict(page_table=page_table, cache_seqlens=cache_seqlens,
                           cu_seqlens_q=cu_q, max_seqlen_q=LQ, causal=True,
                           num_splits=1, block_sparse_cu=cu,
                           block_sparse_idx=idx, total_q_tiles=total_tiles,
                           cu_q_tiles=cu_q_tiles, k_mean=ws.k_mean,
                           v_mean=ws.v_mean, mean_k_block_size=BLOCK_N),
        }
        sinks = torch.zeros(HQ, dtype=DTYPE, device=DEV)
        t_off = bench(lkv, None, index_pack)
        t_on = bench(lkv, sinks, index_pack)
        delta = (t_on - t_off) / t_off * 100
        print(f"kv_len={lkv:>7d}  sinks_off={t_off:8.3f} ms  "
              f"sinks_on={t_on:8.3f} ms  delta={delta:+.2f}%")


if __name__ == "__main__":
    main()
