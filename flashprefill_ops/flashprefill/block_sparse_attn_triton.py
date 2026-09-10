"""Triton block-sparse paged attention for SM100+ (Blackwell, e.g. B300/sm_103).

The CUDA/CuTe PackGQA kernel in this repo targets Hopper (sm_90a, wgmma) and
cannot run on Blackwell (sm_100/sm_103, where wgmma was replaced by tcgen05).
This module re-implements the SAME semantics in Triton so the flashprefill
package works on Blackwell GPUs:

  * PackGQA row layout: packed row r -> q_pos = r // gqa_ratio,
    q_head = kv_head * gqa_ratio + r % gqa_ratio  (tile = K_BLOCK_M rows)
  * CSR sparse index produced by flash_block_sparse_index_triton:
    row = kv_head * total_q_tiles + global_q_tile, values are physical
    64-token tile ids (ascending), page_table maps tokens to physical pages
  * optional zero-order mean correction for unselected blocks
    (score += q @ k_mean[j] * scale + log2(len_j), contrib p_j * v_mean[j])
  * optional per-query-head attention sinks (denominator += exp(sink - m))
  * FP8 (e4m3) q/k/v with per-(batch, kv_head) dequant scales

Only the forward pass is provided (prefill serving path).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import triton
import triton.language as tl

_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _fp_packgqa_fwd_kernel(
    Q,
    KC,
    VC,
    PT,
    CacheSeqlens,
    CuSeqlensQ,
    CuQTiles,
    BsCu,
    BsIdx,
    KMean,
    VMean,
    KD,
    VD,
    Sinks,
    Out,
    scale_log2,
    total_q_tiles,
    stride_qt,
    stride_qh,
    stride_kp,
    stride_kt,
    stride_kh,
    stride_vp,
    stride_vt,
    stride_vh,
    stride_ptb,
    stride_ptp,
    stride_kmb,
    stride_kmj,
    stride_kmh,
    stride_vmb,
    stride_vmj,
    stride_vmh,
    stride_kdb,
    stride_kdh,
    stride_vdb,
    stride_vdh,
    stride_ot,
    stride_oh,
    NUM_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_BLOCK_M: tl.constexpr,
    TILE_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SPARSE: tl.constexpr,
    MEAN_CORR: tl.constexpr,
    N_SUB: tl.constexpr,
    MEAN_K_BLOCK: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    HAS_KD: tl.constexpr,
    HAS_VD: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    m_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // NUM_KV_HEADS
    kv_head = batch_head % NUM_KV_HEADS

    q_tile_begin = tl.load(CuQTiles + batch)
    q_tile_end = tl.load(CuQTiles + batch + 1)
    q_tile_count = q_tile_end - q_tile_begin
    if m_block >= q_tile_count:
        return
    global_q_tile = q_tile_begin + m_block

    q_begin = tl.load(CuSeqlensQ + batch)
    q_end = tl.load(CuSeqlensQ + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(CacheSeqlens + batch)
    prefix_len = kv_len - q_len

    # PackGQA packed rows of this M tile.
    packed = m_block * K_BLOCK_M + tl.arange(0, K_BLOCK_M)
    q_pos = packed // GQA_RATIO
    q_head = kv_head * GQA_RATIO + packed % GQA_RATIO
    valid_q = q_pos < q_len

    dims = tl.arange(0, HEAD_DIM)
    dims_v = tl.arange(0, HEAD_DIM_V)

    q = tl.load(
        Q + (q_begin + q_pos)[:, None] * stride_qt + q_head[:, None] * stride_qh + dims[None, :],
        mask=valid_q[:, None],
        other=0.0,
    )
    if IS_FP8:
        q = q.to(tl.bfloat16)

    # Per-(batch, kv_head) dequant scales (folded into the softmax score).
    kd = 1.0
    if HAS_KD:
        kd = tl.load(KD + batch * stride_kdb + kv_head * stride_kdh).to(tl.float32)
    qk_scale = scale_log2 * kd
    vd = 1.0
    if HAS_VD:
        vd = tl.load(VD + batch * stride_vdb + kv_head * stride_vdh).to(tl.float32)

    packed_end = tl.minimum((m_block + 1) * K_BLOCK_M, q_len * GQA_RATIO)
    q_last = (packed_end - 1) // GQA_RATIO
    q_pos_min = (m_block * K_BLOCK_M) // GQA_RATIO

    if SPARSE:
        row = kv_head * total_q_tiles + global_q_tile
        lo = tl.load(BsCu + row)
        hi = tl.load(BsCu + row + 1)
    else:
        if IS_CAUSAL:
            last_tile = (prefix_len + q_last) // TILE_N
        else:
            last_tile = (kv_len - 1) // TILE_N
        lo = 0
        hi = last_tile + 1

    m_i = tl.full((K_BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((K_BLOCK_M,), tl.float32)
    acc = tl.zeros((K_BLOCK_M, HEAD_DIM_V), tl.float32)

    for i in range(lo, hi):
        if SPARSE:
            tile = tl.load(BsIdx + i)
        else:
            tile = i
        k_tok = tile * TILE_N + tl.arange(0, TILE_N)
        tok_valid = k_tok < kv_len
        page = tl.load(
            PT + batch * stride_ptb + (k_tok // PAGE_SIZE) * stride_ptp,
            mask=tok_valid,
            other=0,
        )
        k_slot = page * stride_kp + (k_tok % PAGE_SIZE) * stride_kt
        k = tl.load(
            KC + k_slot[:, None] + kv_head * stride_kh + dims[None, :],
            mask=tok_valid[:, None],
            other=0.0,
        )
        if IS_FP8:
            k = k.to(tl.bfloat16)
        score = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale
        vis = tok_valid[None, :] & valid_q[:, None]
        if IS_CAUSAL:
            vis &= k_tok[None, :] <= (prefix_len + q_pos)[:, None]
        if WINDOW_LEFT >= 0:
            vis &= (prefix_len + q_pos)[:, None] - k_tok[None, :] <= WINDOW_LEFT
        if WINDOW_RIGHT >= 0:
            vis &= k_tok[None, :] - (prefix_len + q_pos)[:, None] <= WINDOW_RIGHT
        score = tl.where(vis, score, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(score, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(score - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v_slot = page * stride_vp + (k_tok % PAGE_SIZE) * stride_vt
        v = tl.load(
            VC + v_slot[:, None] + kv_head * stride_vh + dims_v[None, :],
            mask=tok_valid[:, None],
            other=0.0,
        )
        if IS_FP8:
            v = v.to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    # Zero-order mean correction: each UNSELECTED logical block j contributes
    # exp2(q @ k_mean[j] * scale + log2(len_j) - m) * v_mean[j], where len_j is
    # the block's valid token count. `unsel` mirrors the fp64 reference:
    # logical blocks < j_hi that are absent from the CSR (ascending order lets
    # a monotone pointer answer membership).
    if MEAN_CORR:
        n_logical = (kv_len + MEAN_K_BLOCK - 1) // MEAN_K_BLOCK
        if IS_CAUSAL:
            j_hi = tl.minimum(n_logical, (prefix_len + q_pos_min) // MEAN_K_BLOCK)
        else:
            j_hi = n_logical
        ptr = lo
        for j in range(0, j_hi):
            cur = tl.load(BsIdx + ptr, mask=ptr < hi, other=2147483647)
            while (ptr < hi) & (cur < j * N_SUB):
                ptr += 1
                cur = tl.load(BsIdx + ptr, mask=ptr < hi, other=2147483647)
            is_sel = (ptr < hi) & (cur // N_SUB == j)
            km = tl.load(
                KMean + batch * stride_kmb + j * stride_kmj + kv_head * stride_kmh + dims
            )
            if IS_FP8:
                km = km.to(tl.bfloat16)
            s = tl.sum(q.to(tl.float32) * km.to(tl.float32)[None, :], axis=1) * qk_scale
            blk_len = tl.minimum(kv_len - j * MEAN_K_BLOCK, MEAN_K_BLOCK).to(tl.float32)
            s = s + tl.log2(blk_len)
            # Selected blocks are computed exactly in the main loop: suppress
            # their mean contribution. Also suppress invalid q rows.
            s = tl.where(valid_q & (is_sel == 0), s, float("-inf"))
            m_new = tl.maximum(m_i, s)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new)
            l_i = l_i * alpha + p
            vm = tl.load(
                VMean + batch * stride_vmb + j * stride_vmj + kv_head * stride_vmh + dims_v
            )
            if IS_FP8:
                vm = vm.to(tl.bfloat16)
            acc = acc * alpha[:, None] + p[:, None] * vm.to(tl.float32)[None, :]
            m_i = m_new

    # Per-query-head attention sink: denominator += exp(sink - m). Sinks are
    # logits (no softmax_scale), gathered per packed row's query head.
    if HAS_SINKS:
        sink_vals = tl.load(Sinks + q_head, mask=valid_q, other=float("-inf")).to(tl.float32)
        l_i += tl.exp2(sink_vals * _LOG2E - m_i)

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    # V is loaded as raw fp8 (undescaled), so acc holds sum(p * v_true / v_scale);
    # fold the per-(batch,kv_head) V descale back in here. vd == 1 when HAS_VD is
    # false, so this is a no-op for bf16/fp16 KV.
    out = acc * vd / l_safe[:, None]
    tl.store(
        Out + (q_begin + q_pos)[:, None] * stride_ot + q_head[:, None] * stride_oh + dims_v[None, :],
        out.to(Out.dtype.element_ty),
        mask=valid_q[:, None],
    )


def _host_q_lens(cu_seqlens_q: torch.Tensor) -> list:
    cu = cu_seqlens_q.tolist()
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]


def flash_attn_with_kvcache_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Sequence[int] = (-1, -1),
    q_descale: Optional[Union[float, torch.Tensor]] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    block_sparse_cu: Optional[torch.Tensor] = None,
    block_sparse_idx: Optional[torch.Tensor] = None,
    total_q_tiles: Optional[int] = None,
    cu_q_tiles: Optional[torch.Tensor] = None,
    k_mean: Optional[torch.Tensor] = None,
    v_mean: Optional[torch.Tensor] = None,
    mean_k_block_size: int = 0,
    num_splits: int = 1,  # accepted for API parity; single-pass kernel
    **unused,
) -> torch.Tensor:
    """Drop-in Triton replacement of ``flash_attn_with_kvcache`` (fwd only)."""
    del max_seqlen_q, num_splits, unused
    if q.ndim != 3:
        raise ValueError("q must have shape (total_q, num_q_heads, head_dim)")
    if k_cache.ndim != 4:
        raise ValueError("k_cache must be (num_pages, page_size, num_kv_heads, head_dim)")
    total_q, num_q_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    head_dim_v = v_cache.shape[-1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    gqa_ratio = num_q_heads // num_kv_heads
    batch = cache_seqlens.numel()
    device = q.device

    is_fp8 = q.dtype == torch.float8_e4m3fn
    out_dtype = torch.bfloat16 if is_fp8 else q.dtype
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    sparse = block_sparse_cu is not None
    k_block_m = 128
    q_lens = _host_q_lens(cu_seqlens_q)
    q_tiles = [(ql * gqa_ratio + k_block_m - 1) // k_block_m for ql in q_lens]
    max_q_tiles = max(q_tiles) if q_tiles else 0
    if max_q_tiles == 0:
        return torch.empty(total_q, num_q_heads, head_dim_v, dtype=out_dtype, device=device)

    if sparse:
        if block_sparse_idx is None or total_q_tiles is None or cu_q_tiles is None:
            raise ValueError("block_sparse_cu/idx/total_q_tiles/cu_q_tiles must come together")
        cu_q_tiles_arg = cu_q_tiles.contiguous()
        bs_cu = block_sparse_cu.contiguous()
        bs_idx = block_sparse_idx.contiguous()
        total_q_tiles = int(total_q_tiles)
    else:
        tiles_cum = [0]
        for t in q_tiles:
            tiles_cum.append(tiles_cum[-1] + t)
        cu_q_tiles_arg = torch.tensor(tiles_cum, dtype=torch.int32, device=device)
        total_q_tiles = tiles_cum[-1]
        bs_cu = cu_q_tiles_arg  # placeholder, unused when SPARSE=False
        bs_idx = cu_q_tiles_arg

    mean_corr = sparse and k_mean is not None and v_mean is not None and mean_k_block_size > 0
    if mean_corr:
        n_sub = mean_k_block_size // 64
        km = k_mean
        vm = v_mean
        s_kmb, s_kmj, s_kmh, _ = km.stride()
        s_vmb, s_vmj, s_vmh, _ = vm.stride()
    else:
        n_sub = 1
        mean_k_block_size = 64
        km = vm = q  # placeholders, unused
        s_kmb = s_kmj = s_kmh = s_vmb = s_vmj = s_vmh = 0

    qd = 1.0
    if q_descale is not None:
        if isinstance(q_descale, torch.Tensor):
            if q_descale.numel() != 1:
                raise NotImplementedError("only scalar q_descale is supported by the Triton backend")
            qd = float(q_descale.item())
        else:
            qd = float(q_descale)

    has_kd = k_descale is not None
    has_vd = v_descale is not None
    if has_kd:
        s_kdb, s_kdh = k_descale.stride() if k_descale.ndim == 2 else (0, 0)
        kd_arg = k_descale
    else:
        s_kdb = s_kdh = 0
        kd_arg = q
    if has_vd:
        s_vdb, s_vdh = v_descale.stride() if v_descale.ndim == 2 else (0, 0)
        vd_arg = v_descale
    else:
        s_vdb = s_vdh = 0
        vd_arg = q

    has_sinks = sinks is not None
    sinks_arg = sinks if has_sinks else q

    out = torch.empty(total_q, num_q_heads, head_dim_v, dtype=out_dtype, device=device)

    grid = (max_q_tiles, batch * num_kv_heads)
    _fp_packgqa_fwd_kernel[grid](
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        cu_q_tiles_arg,
        bs_cu,
        bs_idx,
        km,
        vm,
        kd_arg,
        vd_arg,
        sinks_arg,
        out,
        softmax_scale * _LOG2E * qd,
        total_q_tiles,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        page_table.stride(0),
        page_table.stride(1),
        s_kmb,
        s_kmj,
        s_kmh,
        s_vmb,
        s_vmj,
        s_vmh,
        s_kdb,
        s_kdh,
        s_vdb,
        s_vdh,
        out.stride(0),
        out.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GQA_RATIO=gqa_ratio,
        PAGE_SIZE=page_size,
        K_BLOCK_M=k_block_m,
        TILE_N=64,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim_v,
        IS_CAUSAL=causal,
        SPARSE=sparse,
        MEAN_CORR=mean_corr,
        N_SUB=n_sub,
        MEAN_K_BLOCK=mean_k_block_size,
        HAS_SINKS=has_sinks,
        HAS_KD=has_kd,
        HAS_VD=has_vd,
        WINDOW_LEFT=window_size[0],
        WINDOW_RIGHT=window_size[1],
        IS_FP8=is_fp8,
        num_warps=8,
        num_stages=2,
    )
    return out


def flash_attn_func_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Sequence[int] = (-1, -1),
    q_descale: Optional[Union[float, torch.Tensor]] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    **unused,
) -> torch.Tensor:
    """Dense contiguous (batch, seqlen, nheads, headdim) forward via the
    PackGQA kernel recast as a single-page-per-request paged problem."""
    if q.ndim != 4:
        raise ValueError("q must have shape (batch, seqlen, nheads, headdim)")
    b, s, h, d = q.shape
    nkv = k.shape[2]
    dv = v.shape[-1]
    q_flat = q.reshape(b * s, h, d)
    k_cache = k.reshape(b, s, nkv, d)
    v_cache = v.reshape(b, s, nkv, dv)
    page_table = torch.arange(b, dtype=torch.int32, device=q.device).unsqueeze(1)
    cache_seqlens = torch.full((b,), s, dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.arange(0, (b + 1) * s, s, dtype=torch.int32, device=q.device)
    out = flash_attn_with_kvcache_triton(
        q_flat,
        k_cache,
        v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
    )
    return out.reshape(b, s, h, dv)


__all__ = ["flash_attn_with_kvcache_triton", "flash_attn_func_triton"]
