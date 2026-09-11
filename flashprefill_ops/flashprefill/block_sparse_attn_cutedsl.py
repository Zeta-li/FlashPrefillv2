"""CuteDSL block-sparse paged prefill attention for Blackwell (sm_103 / B300).

Drop-in replacement for the Triton backend (`block_sparse_attn_triton.py`), built
on the CUTLASS Ampere flash-attention-v2 CuteDSL example (`cutlass_fa2_base.py`,
register-based cp.async + `MmaF16BF16Op`, which runs on Blackwell via backward
compat and — unlike the tcgen05 warp-spec base — supports head_dim=256).

Replicates the SAME semantics as the Triton kernel:
  * paged KV cache (num_pages, page_size, num_kv_heads, head_dim) + page_table,
    with page_size == TILE_N == n_block_size == 64 (one logical tile == one page)
  * GQA (num_q_heads = gqa_ratio * num_kv_heads); each CTA owns one (batch, q_head)
  * varlen via cu_seqlens_q (packed q of shape (total_q, num_q_heads, head_dim))
  * fp8 e4m3 q/k/v loaded and up-cast to bf16, with per-(batch, kv_head) dequant:
    K descale folded into the softmax scale, V descale multiplied at the epilogue
    (mirrors the fixed V-descale semantics of the Triton kernel)
  * block-sparse CSR tile iteration (block_sparse_cu / block_sparse_idx)
  * causal / sliding-window masking, attention sink            [staged; see below]

Correctness-first: this is the simple, semantically-faithful path. The tcgen05
warp-specialized base is kept in `cutlass_blackwell_fmha.py` for the tuning stage.

Only the forward pass is provided (prefill serving path).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Sequence, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack

from .cutlass_fa2_base import FlashAttentionForwardAmpere

LOG2_E = 1.4426950408889634074

# PackGQA query-tile height used by the *production* block-sparse CSR index
# builder (see `block_sparse_attn_triton.py` / `prefill.py`, `k_block_m=128`).
# The CSR selection set is keyed on (kv_head, q_tile) where a q_tile packs
# `PACKGQA_K_BLOCK_M` rows = `PACKGQA_K_BLOCK_M // gqa_ratio` query positions
# (shared across a kv_head's gqa heads). This kernel processes one q_head per
# CTA, so its query-tile height on the sparse path MUST equal that many
# positions to read the identical selection set as Triton (see the sparse
# `m_block_size` derivation in the wrapper).
PACKGQA_K_BLOCK_M = 128


class FlashPrefillCuteDSL(FlashAttentionForwardAmpere):
    """Paged, GQA, varlen, (optionally block-sparse) flash-attention forward.

    Subclasses the Ampere FA2 example to reuse its smem layouts, tiled MMA,
    ldmatrix smem-copy atoms, online-softmax math and threadquad reductions,
    while replacing the mainloop with a per-tile paged / CSR-driven loop and
    explicit token-position masking that mirrors the Triton ground truth.
    """

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        gqa_ratio: int,
        num_kv_heads: int,
        page_size: int,
        m_block_size: int = 64,
        n_block_size: int = 64,
        num_threads: int = 128,
        is_causal: bool = True,
        is_fp8: bool = False,
        has_kd: bool = False,
        has_vd: bool = False,
        sparse: bool = False,
        window_left: int = -1,
        window_right: int = -1,
        has_sinks: bool = False,
        mean_corr: bool = False,
        mean_k_block: int = 64,
        pack_gqa: bool = False,
    ):
        super().__init__(head_dim, m_block_size, n_block_size, num_threads, is_causal)
        assert page_size == n_block_size, "kernel assumes one logical tile == one page"
        self._head_dim_v = head_dim_v
        self._head_dim_v_padded = (head_dim_v + 31) // 32 * 32
        self._gqa_ratio = gqa_ratio
        self._num_kv_heads = num_kv_heads
        self._page_size = page_size
        self._is_fp8 = is_fp8
        self._has_kd = has_kd
        self._has_vd = has_vd
        self._sparse = sparse
        # PackGQA: one CTA processes m_block_size *packed* rows (query-position ×
        # q-head interleaved, `packed = q_pos*gqa + head`) for a single kv_head, so
        # its KV is loaded ONCE and reused across all gqa q-heads (4× less KV
        # traffic vs. the one-q_head-per-CTA path). Enabled for HKV==1 where the
        # packed rows are contiguous in q.reshape(total_q*num_q_heads, head_dim).
        self._pack_gqa = pack_gqa
        self._window_left = window_left
        self._window_right = window_right
        self._has_sinks = has_sinks
        # Zero-order mean correction for UNSELECTED logical blocks (default off).
        self._mean_corr = mean_corr
        self._mean_k_block = mean_k_block          # multiple of n_block_size (64)
        self._n_sub = mean_k_block // n_block_size  # 64-tiles per logical mean block
        # Compute dtype fed to the tensor cores.
        self._compute_dtype = cutlass.BFloat16

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,          # (total_q, num_q_heads, head_dim)
        mK: cute.Tensor,          # (num_pages, page_size, num_kv_heads, head_dim)
        mV: cute.Tensor,          # (num_pages, page_size, num_kv_heads, head_dim_v)
        mO: cute.Tensor,          # (total_q, num_q_heads, head_dim_v)
        mPT: cute.Tensor,         # (batch, max_pages) int32
        mCacheSeqlens: cute.Tensor,   # (batch,) int32
        mCuSeqlensQ: cute.Tensor,     # (batch+1,) int32
        mCuQTiles: cute.Tensor,       # (batch+1,) int32
        mBsCu: cute.Tensor,           # (num_rows+1,) int32   (sparse only, else dummy)
        mBsIdx: cute.Tensor,          # (nnz,) int32          (sparse only, else dummy)
        mKD: cute.Tensor,             # (batch, num_kv_heads) f32 (dummy if not has_kd)
        mVD: cute.Tensor,             # (batch, num_kv_heads) f32 (dummy if not has_vd)
        mSinks: cute.Tensor,          # (num_q_heads,) f32 (dummy if not has_sinks)
        mKmean: cute.Tensor,          # (batch, n_logical, num_kv_heads, head_dim) (dummy if not mean_corr)
        mVmean: cute.Tensor,          # (batch, n_logical, num_kv_heads, head_dim_v) (dummy if not mean_corr)
        softmax_scale: cutlass.Float32,
        total_q_tiles: cutlass.Int32,
        max_q_blocks: cutlass.Constexpr,
        batch: cutlass.Constexpr,
        num_q_heads: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self._dtype = mQ.element_type

        # --- smem layouts (Q uses head_dim, K uses head_dim, V uses head_dim_v) ---
        def make_slayout(rows, kdim_padded):
            smem_k_block = 64 if kdim_padded % 64 == 0 else 32
            swizzle_bits = 3 if smem_k_block == 64 else 2
            atom = cute.make_composed_layout(
                cute.make_swizzle(swizzle_bits, 3, 3),
                0,
                cute.make_layout((8, smem_k_block), stride=(smem_k_block, 1)),
            )
            return cute.tile_to_shape(atom, (rows, kdim_padded), (0, 1))

        sQ_layout = make_slayout(self._m_block_size, self._head_dim_padded)
        sK_layout = make_slayout(self._n_block_size, self._head_dim_padded)
        sV_layout = make_slayout(self._n_block_size, self._head_dim_v_padded)
        sO_layout = make_slayout(self._m_block_size, self._head_dim_v_padded)

        # smem tiles are always in the compute dtype (bf16); fp8 is up-cast on load.
        cdt = self._compute_dtype

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[cdt, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[cdt, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[cdt, cute.cosize(sV_layout)], 1024]

        # --- gmem tiled copies (built for the *load* dtype = mQ.element_type) ---
        copy_bits = 128
        async_elems = copy_bits // self._dtype.width
        atom_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=copy_bits,
        )
        # thread layout for QKV load, sized off the K-major smem atom of Q
        smem_k_block = 64 if self._head_dim_padded % 64 == 0 else 32
        tdim1 = smem_k_block // async_elems
        t_layout = cute.make_layout(
            (self._num_threads // tdim1, tdim1), stride=(tdim1, 1)
        )
        v_layout = cute.make_layout((1, async_elems))
        gmem_copy_QKV = cute.make_tiled_copy_tv(atom_async, t_layout, v_layout)

        # For fp8 inputs the smem/MMA operands stay bf16 (`cdt`); we cannot cp.async
        # straight into a bf16 tile (element widths differ). Instead do a synchronous
        # gmem(fp8)->rmem(fp8)->rmem(bf16)->smem(bf16) load, converting in registers.
        # `gmem_copy_src` (fp8, universal) and `gmem_copy_dst` (bf16, universal) share
        # the SAME thread/value layout, so element i of thread t is the same logical
        # (row, col) in both — the conversion is layout-correct regardless of swizzle.
        if cutlass.const_expr(self._is_fp8):
            atom_ld_src = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self._dtype, num_bits_per_copy=copy_bits
            )
            atom_ld_dst = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cdt, num_bits_per_copy=cdt.width * async_elems
            )
            gmem_copy_src = cute.make_tiled_copy_tv(atom_ld_src, t_layout, v_layout)
            gmem_copy_dst = cute.make_tiled_copy_tv(atom_ld_dst, t_layout, v_layout)
        else:
            gmem_copy_src = gmem_copy_QKV
            gmem_copy_dst = gmem_copy_QKV

        atom_univ = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cdt, num_bits_per_copy=copy_bits
        )
        oa_elems = copy_bits // cdt.width
        smem_k_block_v = 64 if self._head_dim_v_padded % 64 == 0 else 32
        tdim1_o = smem_k_block_v // oa_elems
        tO_layout = cute.make_layout(
            (self._num_threads // tdim1_o, tdim1_o), stride=(tdim1_o, 1)
        )
        vO_layout = cute.make_layout((1, oa_elems))
        gmem_copy_O = cute.make_tiled_copy_tv(atom_univ, tO_layout, vO_layout)

        # --- tiled mma (bf16 inputs, f32 accum) ---
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(cdt, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # PackGQA packs a kv_head's gqa q-heads into one CTA, so the grid's head
        # axis is num_kv_heads; otherwise one CTA per q_head.
        grid_heads = (
            num_q_heads // self._gqa_ratio if self._pack_gqa else num_q_heads
        )
        grid_dim = (max_q_blocks, batch, grid_heads)
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            mQ, mK, mV, mO, mPT, mCacheSeqlens, mCuSeqlensQ, mCuQTiles,
            mBsCu, mBsIdx, mKD, mVD, mSinks, mKmean, mVmean,
            softmax_scale_log2, total_q_tiles,
            sQ_layout, sK_layout, sV_layout, sO_layout,
            gmem_copy_src, gmem_copy_dst, gmem_copy_O, tiled_mma, SharedStorage,
        ).launch(
            grid=grid_dim, block=[self._num_threads, 1, 1], stream=stream
        )

    # ---------------------------------------------------------------- device
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mPT: cute.Tensor,
        mCacheSeqlens: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mCuQTiles: cute.Tensor,
        mBsCu: cute.Tensor,
        mBsIdx: cute.Tensor,
        mKD: cute.Tensor,
        mVD: cute.Tensor,
        mSinks: cute.Tensor,
        mKmean: cute.Tensor,
        mVmean: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        total_q_tiles: cutlass.Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_copy_src: cute.TiledCopy,
        gmem_copy_dst: cute.TiledCopy,
        gmem_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch, head_z = cute.arch.block_idx()

        # In PackGQA the grid's head axis is the kv_head (one CTA packs its gqa
        # q-heads); otherwise it is the q_head and kv_head is derived by division.
        if cutlass.const_expr(self._pack_gqa):
            kv_head = head_z
            q_head = head_z * self._gqa_ratio   # base q_head of this kv_head's group
        else:
            q_head = head_z
            kv_head = q_head // self._gqa_ratio

        # --- varlen bookkeeping ---
        q_tile_begin = mCuQTiles[batch]
        q_tile_end = mCuQTiles[batch + 1]
        # Early exit is not allowed inside a staged kernel; instead flag out-of-range
        # CTAs and clamp their tile range to empty (their rows are all >= q_len, so
        # the store is predicated off anyway).
        oob = m_block >= (q_tile_end - q_tile_begin)
        global_q_tile = q_tile_begin + m_block

        q_begin = mCuSeqlensQ[batch]
        q_end = mCuSeqlensQ[batch + 1]
        q_len = q_end - q_begin
        kv_len = mCacheSeqlens[batch]
        prefix_len = kv_len - q_len

        q_base = m_block * self._m_block_size          # local (packed) row offset
        # Number of valid local rows in this batch: packed rows (q_len*gqa) under
        # PackGQA, plain query positions otherwise.
        if cutlass.const_expr(self._pack_gqa):
            q_row_limit = q_len * self._gqa_ratio
        else:
            q_row_limit = q_len
        # max q position (global) touched by this tile, for the causal tile bound.
        # Under PackGQA a packed row maps to q position `row // gqa`.
        q_row_last = cutlass.min(q_base + self._m_block_size - 1, q_row_limit - 1)
        if cutlass.const_expr(self._pack_gqa):
            q_pos_last = prefix_len + q_row_last // self._gqa_ratio
        else:
            q_pos_last = prefix_len + q_row_last

        # --- descale scalars ---
        scale_log2 = softmax_scale_log2
        if cutlass.const_expr(self._has_kd):
            scale_log2 = scale_log2 * mKD[batch, kv_head]
        vd = cutlass.Float32(1.0)
        if cutlass.const_expr(self._has_vd):
            vd = mVD[batch, kv_head]

        # --- CSR / dense tile range [lo, hi) ---
        if cutlass.const_expr(self._sparse):
            row = kv_head * total_q_tiles + global_q_tile
            lo = mBsCu[row]
            hi = mBsCu[row + 1]
        else:
            if cutlass.const_expr(self._is_causal):
                last_tile = q_pos_last // self._n_block_size
            else:
                last_tile = (kv_len - 1) // self._n_block_size
            lo = cutlass.Int32(0)
            hi = last_tile + 1
        if oob:
            hi = lo

        # --- smem ---
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim_v_padded, self._n_block_size),
                stride=(self._n_block_size, 1),
            ),
        )

        # --- Q global tile (packed varlen): offset rows by the batch's q_begin ---
        # PackGQA: mQ is the 2D reshape (total_q*num_q_heads, head_dim); packed rows
        # are contiguous, so this batch's rows start at q_begin*gqa and the CTA reads
        # a plain (m_block_size, head_dim) tile spanning gqa interleaved q-heads.
        if cutlass.const_expr(self._pack_gqa):
            mQ_h = cute.domain_offset((q_begin * self._gqa_ratio, 0), mQ)
        else:
            mQ_h = cute.domain_offset((q_begin, 0), mQ[None, q_head, None])
        gQ = cute.local_tile(
            mQ_h, (self._m_block_size, self._head_dim_padded), (m_block, 0)
        )

        gmem_thr_src = gmem_copy_src.get_slice(tidx)
        gmem_thr_dst = gmem_copy_dst.get_slice(tidx)
        tQgQ = gmem_thr_src.partition_S(gQ)
        tQsQ = gmem_thr_dst.partition_D(sQ)
        tKsK = gmem_thr_dst.partition_D(sK)
        tVsV = gmem_thr_dst.partition_D(sV)

        # --- mma partitions / accumulators ---
        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_O = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_block_size, self._head_dim_v_padded)),
            cutlass.Float32,
        )
        acc_O.fill(0.0)

        # --- smem->rmem ldmatrix copies ---
        atom_Q = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._compute_dtype
        )
        atom_K = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._compute_dtype
        )
        atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._compute_dtype
        )
        smem_copy_Q = cute.make_tiled_copy_A(atom_Q, tiled_mma)
        smem_copy_K = cute.make_tiled_copy_B(atom_K, tiled_mma)
        smem_copy_V = cute.make_tiled_copy_B(atom_V, tiled_mma)
        smem_thr_Q = smem_copy_Q.get_slice(tidx)
        smem_thr_K = smem_copy_K.get_slice(tidx)
        smem_thr_V = smem_copy_V.get_slice(tidx)
        tSsQ = smem_thr_Q.partition_S(sQ)
        tSrQ_view = smem_thr_Q.retile(tSrQ)
        tSsK = smem_thr_K.partition_S(sK)
        tSrK_view = smem_thr_K.retile(tSrK)
        tOsVt = smem_thr_V.partition_S(sVt)
        tOrVt_view = smem_thr_V.retile(tOrVt)

        # --- identity coords for masking (local row/col within the tile) ---
        cS = cute.make_identity_tensor((self._m_block_size, self._n_block_size))
        tScS = thr_mma.partition_C(cS)
        tScS_mn = self._make_acc_tensor_mn_view(tScS)

        # --- load Q once (predicate rows against q_len, head_dim is full/aligned) ---
        cQ = cute.make_identity_tensor(
            (self._m_block_size, self._head_dim_padded)
        )
        tQcQ = gmem_thr_src.partition_S(cQ)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            # identity coords are tile-local; add q_base for the batch-global row.
            if cute.elem_less(q_base + tQcQ[0, m, 0][0], q_row_limit):
                if cutlass.const_expr(self._is_fp8):
                    self._load_convert(tQgQ[None, m, None], tQsQ[None, m, None])
                else:
                    cute.copy(gmem_copy_src, tQgQ[None, m, None], tQsQ[None, m, None])
            else:
                tQsQ[None, m, None].fill(0)
        if cutlass.const_expr(not self._is_fp8):
            cute.arch.cp_async_commit_group()

        # --- softmax running state ---
        row_max = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_sum = cute.make_rmem_tensor(
            (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
        )
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        params = SimpleNamespace(
            mK=mK, mV=mV, mPT=mPT, mBsIdx=mBsIdx, mKmean=mKmean, mVmean=mVmean,
            gmem_copy_src=gmem_copy_src, gmem_thr_src=gmem_thr_src,
            tKsK=tKsK, tVsV=tVsV,
            smem_copy_Q=smem_copy_Q, smem_copy_K=smem_copy_K, smem_copy_V=smem_copy_V,
            tSsQ=tSsQ, tSrQ_view=tSrQ_view, tSrQ=tSrQ,
            tSsK=tSsK, tSrK_view=tSrK_view, tSrK=tSrK,
            tOsVt=tOsVt, tOrVt_view=tOrVt_view, tOrVt=tOrVt,
            thr_mma=thr_mma, tiled_mma=tiled_mma, acc_O=acc_O,
            tScS_mn=tScS_mn, row_max=row_max, row_sum=row_sum,
            batch=batch, kv_head=kv_head, kv_len=kv_len, prefix_len=prefix_len,
            q_base=q_base, q_len=q_len, q_row_limit=q_row_limit,
            scale_log2=scale_log2,
        )

        # --- mainloop over selected tiles (uniform online-softmax, no first-flag) ---
        i = lo
        while i < hi:
            if cutlass.const_expr(self._sparse):
                tile = mBsIdx[i]
            else:
                tile = i
            self._compute_one_tile(params, tile)
            i += 1

        # --- zero-order mean correction for UNSELECTED logical blocks ---
        # Each logical block j < j_hi that is absent from the CSR contributes a
        # single synthetic key/value pair (k_mean[j], v_mean[j]) with score bias
        # log2(len_j). Processed as extra synthetic KV tiles through the same
        # QK/PV MMA + online-softmax path (online softmax is order-independent, so
        # batching all blocks per 64-wide tile matches Triton's per-block loop).
        if cutlass.const_expr(self._mean_corr):
            n_logical = (kv_len + self._mean_k_block - 1) // self._mean_k_block
            # smallest query position in this tile (packed row base // gqa)
            if cutlass.const_expr(self._pack_gqa):
                q_pos_min = q_base // self._gqa_ratio
            else:
                q_pos_min = q_base
            if cutlass.const_expr(self._is_causal):
                j_hi = cutlass.min(
                    n_logical, (prefix_len + q_pos_min) // self._mean_k_block
                )
            else:
                j_hi = n_logical
            if oob:
                j_hi = cutlass.Int32(0)
            n_mean_tiles = (j_hi + self._n_block_size - 1) // self._n_block_size
            bt = cutlass.Int32(0)
            while bt < n_mean_tiles:
                self._compute_mean_tile(params, bt, n_logical, j_hi, lo, hi)
                bt += 1

        # --- attention sink: add per-q-head sink logit to the denominator only.
        # Sinks are raw logits (no softmax_scale); row_max*scale_log2 is the running
        # scaled max (matches Triton's `l_i += exp2(sink*LOG2E - m_i)`). row_sum here
        # is a per-thread partial column sum that normalize_softmax later reduces over
        # the 4-lane quad, so add the term on ONE representative lane (lane%4==0) to
        # avoid counting it four times.
        if cutlass.const_expr(self._has_sinks):
            lane = cute.arch.lane_idx()
            if lane % 4 == 0:
                for r in cutlass.range_constexpr(cute.size(row_sum)):
                    grow = q_base + tScS_mn[r, 0][0]
                    if grow < q_row_limit:
                        # PackGQA: each packed row belongs to q_head = base + row%gqa;
                        # otherwise the whole CTA is a single q_head.
                        if cutlass.const_expr(self._pack_gqa):
                            sink_val = mSinks[q_head + grow % self._gqa_ratio]
                        else:
                            sink_val = mSinks[q_head]
                        row_sum[r] = row_sum[r] + cute.math.exp2(
                            sink_val * LOG2_E - row_max[r] * scale_log2, fastmath=True
                        )

        # --- epilogue: normalize, apply V descale, store ---
        self.normalize_softmax(acc_O, row_sum)
        if cutlass.const_expr(self._has_vd):
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            for r in cutlass.range_constexpr(cute.size(row_sum)):
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * vd

        self._store_O(mO, sQ, sO_layout, acc_O, gmem_copy_O, tiled_mma,
                      q_begin, q_head, m_block, q_base, q_row_limit)

    # ------------------------------------------------------- one tile
    @cute.jit
    def _compute_one_tile(self, p, tile: cutlass.Int32):
        n_block_size = self._n_block_size
        k_base = tile * n_block_size

        # physical page for this logical tile (page_size == n_block_size, so the
        # page id is exactly the row-block index into the flattened KV cache).
        # int64: page ids index a large global KV pool (num_pages can exceed
        # 2^31 / (page_size*num_kv_heads*head_dim) ~= 131072). Used raw as the
        # local_tile coordinate, `page * row_stride` (== page*16384 here) is
        # computed in int32 and overflows -> illegal address. Promote to int64
        # so crd2idx does the whole offset in 64-bit. (Xid 31 fix; mirrors the
        # triton kernels.)
        page = cutlass.Int64(p.mPT[p.batch, tile])

        # --- load K page -> sK, V page -> sV (up-cast fp8->bf16 handled below) ---
        # mK/mV are flattened to (num_pages*page_size, num_kv_heads, head_dim);
        # local_tile keeps the tile shape static (mirrors the base) and preserves
        # the 128-bit alignment annotation through the runtime page coordinate.
        gK = cute.local_tile(
            p.mK[None, p.kv_head, None],
            (self._n_block_size, self._head_dim_padded), (page, 0),
        )
        gV = cute.local_tile(
            p.mV[None, p.kv_head, None],
            (self._n_block_size, self._head_dim_v_padded), (page, 0),
        )
        tKgK = p.gmem_thr_src.partition_S(gK)
        tVgV = p.gmem_thr_src.partition_S(gV)

        # token predicate (residual): token = k_base + local_n < kv_len
        cKV = cute.make_identity_tensor((n_block_size, self._head_dim_padded))
        tKVcKV = p.gmem_thr_src.partition_S(cKV)
        for n in cutlass.range_constexpr(cute.size(p.tKsK.shape[1])):
            if cute.elem_less(k_base + tKVcKV[0, n, 0][0], p.kv_len):
                if cutlass.const_expr(self._is_fp8):
                    self._load_convert(tKgK[None, n, None], p.tKsK[None, n, None])
                    self._load_convert(tVgV[None, n, None], p.tVsV[None, n, None])
                else:
                    cute.copy(p.gmem_copy_src, tKgK[None, n, None], p.tKsK[None, n, None])
                    cute.copy(p.gmem_copy_src, tVgV[None, n, None], p.tVsV[None, n, None])
            else:
                p.tKsK[None, n, None].fill(0)
                p.tVsV[None, n, None].fill(0)
        if cutlass.const_expr(not self._is_fp8):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # --- S = Q @ K^T ---
        acc_S = cute.make_rmem_tensor(
            p.thr_mma.partition_shape_C((self._m_block_size, n_block_size)),
            cutlass.Float32,
        )
        acc_S.fill(0.0)
        cute.copy(p.smem_copy_Q, p.tSsQ[None, None, 0], p.tSrQ_view[None, None, 0])
        cute.copy(p.smem_copy_K, p.tSsK[None, None, 0], p.tSrK_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(p.tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(p.tSsQ.shape[2])
            cute.copy(p.smem_copy_Q, p.tSsQ[None, None, k_next], p.tSrQ_view[None, None, k_next])
            cute.copy(p.smem_copy_K, p.tSsK[None, None, k_next], p.tSrK_view[None, None, k_next])
            cute.gemm(p.tiled_mma, acc_S, p.tSrQ[None, None, k], p.tSrK[None, None, k], acc_S)

        # --- mask: causal / window / residual, on real token positions ---
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        tScS_mn = p.tScS_mn
        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            grow = p.q_base + tScS_mn[r, 0][0]
            q_valid = grow < p.q_row_limit
            # PackGQA: a packed row maps to query position `row // gqa`.
            if cutlass.const_expr(self._pack_gqa):
                q_pos = p.prefix_len + grow // self._gqa_ratio
            else:
                q_pos = p.prefix_len + grow
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                k_tok = k_base + tScS_mn[0, c][1]
                vis = q_valid and (k_tok < p.kv_len)
                if cutlass.const_expr(self._is_causal):
                    vis = vis and (k_tok <= q_pos)
                if cutlass.const_expr(self._window_left >= 0):
                    vis = vis and (q_pos - k_tok <= self._window_left)
                if cutlass.const_expr(self._window_right >= 0):
                    vis = vis and (k_tok - q_pos <= self._window_right)
                if not vis:
                    acc_S_mn[r, c] = -cutlass.Float32.inf

        # --- online softmax rescale ---
        self._online_softmax(acc_S, p.acc_O, p.row_max, p.row_sum, p.scale_log2)

        # --- O += P @ V ---
        rP = cute.make_fragment_like(acc_S, self._compute_dtype)
        rP.store(acc_S.load().to(self._compute_dtype))
        rP_div = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            ((rP_div.shape[0], rP_div.shape[2][0]), rP_div.shape[1], rP_div.shape[2][1]),
            stride=((rP_div.stride[0], rP_div.stride[2][0]), rP_div.stride[1], rP_div.stride[2][1]),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
        cute.copy(p.smem_copy_V, p.tOsVt[None, None, 0], p.tOrVt_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(p.smem_copy_V, p.tOsVt[None, None, k_next], p.tOrVt_view[None, None, k_next])
            cute.gemm(p.tiled_mma, p.acc_O, tOrS[None, None, k], p.tOrVt[None, None, k], p.acc_O)
        self.cta_sync_barrier.arrive_and_wait()

    # ------------------------------------------------- one mean-correction tile
    @cute.jit
    def _compute_mean_tile(self, p, bt: cutlass.Int32, n_logical: cutlass.Int32,
                           j_hi: cutlass.Int32, lo: cutlass.Int32, hi: cutlass.Int32):
        """Synthetic KV tile of up to ``n_block_size`` logical mean blocks.

        Keys/values come from ``k_mean``/``v_mean`` (rows = logical blocks). Column
        c holds logical block ``j = bt*n_block_size + c``; its score gets a
        +log2(len_j) bias (added in the pre-scale domain as log2(len_j)/scale_log2,
        since online-softmax multiplies by scale_log2), and is masked to -inf when
        the block is out of range (>= j_hi) or SELECTED (present in the CSR, since
        selected blocks are already computed exactly in the main loop)."""
        n_block_size = self._n_block_size
        blk_base = bt * n_block_size

        # --- load k_mean / v_mean logical-block rows -> sK / sV ---
        gKM = cute.local_tile(
            p.mKmean[p.batch, None, p.kv_head, None],
            (self._n_block_size, self._head_dim_padded), (bt, 0),
        )
        gVM = cute.local_tile(
            p.mVmean[p.batch, None, p.kv_head, None],
            (self._n_block_size, self._head_dim_v_padded), (bt, 0),
        )
        tKgK = p.gmem_thr_src.partition_S(gKM)
        tVgV = p.gmem_thr_src.partition_S(gVM)

        cKV = cute.make_identity_tensor((n_block_size, self._head_dim_padded))
        tKVcKV = p.gmem_thr_src.partition_S(cKV)
        for n in cutlass.range_constexpr(cute.size(p.tKsK.shape[1])):
            if cute.elem_less(blk_base + tKVcKV[0, n, 0][0], n_logical):
                if cutlass.const_expr(self._is_fp8):
                    self._load_convert(tKgK[None, n, None], p.tKsK[None, n, None])
                    self._load_convert(tVgV[None, n, None], p.tVsV[None, n, None])
                else:
                    cute.copy(p.gmem_copy_src, tKgK[None, n, None], p.tKsK[None, n, None])
                    cute.copy(p.gmem_copy_src, tVgV[None, n, None], p.tVsV[None, n, None])
            else:
                p.tKsK[None, n, None].fill(0)
                p.tVsV[None, n, None].fill(0)
        if cutlass.const_expr(not self._is_fp8):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # --- S = Q @ K_mean^T ---
        acc_S = cute.make_rmem_tensor(
            p.thr_mma.partition_shape_C((self._m_block_size, n_block_size)),
            cutlass.Float32,
        )
        acc_S.fill(0.0)
        cute.copy(p.smem_copy_Q, p.tSsQ[None, None, 0], p.tSrQ_view[None, None, 0])
        cute.copy(p.smem_copy_K, p.tSsK[None, None, 0], p.tSrK_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(p.tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(p.tSsQ.shape[2])
            cute.copy(p.smem_copy_Q, p.tSsQ[None, None, k_next], p.tSrQ_view[None, None, k_next])
            cute.copy(p.smem_copy_K, p.tSsK[None, None, k_next], p.tSrK_view[None, None, k_next])
            cute.gemm(p.tiled_mma, acc_S, p.tSrQ[None, None, k], p.tSrK[None, None, k], acc_S)

        # --- score bias + not-selected mask (per logical block / column) ---
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        tScS_mn = p.tScS_mn
        inv_scale = 1.0 / p.scale_log2
        for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
            j = blk_base + tScS_mn[0, c][1]
            # CSR membership: block j selected iff some selected 64-tile in
            # [j*n_sub, (j+1)*n_sub). BsIdx[lo:hi] holds selected 64-tile indices.
            sel = cutlass.Int32(0)
            t = lo
            while t < hi:
                if (p.mBsIdx[t] // self._n_sub) == j:
                    sel = cutlass.Int32(1)
                t += 1
            col_ok = (j < j_hi) and (sel == 0)
            blk_len = cutlass.min(p.kv_len - j * self._mean_k_block, self._mean_k_block)
            blk_len_f = cute.arch.fmax(cutlass.Float32(blk_len), 1.0)
            bias = cute.math.log2(blk_len_f, fastmath=True) * inv_scale
            for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                q_valid = (p.q_base + tScS_mn[r, 0][0]) < p.q_row_limit
                if q_valid and col_ok:
                    acc_S_mn[r, c] = acc_S_mn[r, c] + bias
                else:
                    acc_S_mn[r, c] = -cutlass.Float32.inf

        # --- online softmax rescale (shared with the main loop) ---
        self._online_softmax(acc_S, p.acc_O, p.row_max, p.row_sum, p.scale_log2)

        # --- O += P @ V_mean ---
        rP = cute.make_fragment_like(acc_S, self._compute_dtype)
        rP.store(acc_S.load().to(self._compute_dtype))
        rP_div = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            ((rP_div.shape[0], rP_div.shape[2][0]), rP_div.shape[1], rP_div.shape[2][1]),
            stride=((rP_div.stride[0], rP_div.stride[2][0]), rP_div.stride[1], rP_div.stride[2][1]),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
        cute.copy(p.smem_copy_V, p.tOsVt[None, None, 0], p.tOrVt_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(p.smem_copy_V, p.tOsVt[None, None, k_next], p.tOrVt_view[None, None, k_next])
            cute.gemm(p.tiled_mma, p.acc_O, tOrS[None, None, k], p.tOrVt[None, None, k], p.acc_O)
        self.cta_sync_barrier.arrive_and_wait()

    # ------------------------------------------------------- fp8 load+convert
    @cute.jit
    def _load_convert(self, gsrc_slice, sdst_slice):
        """Synchronous gmem(fp8) -> rmem(fp8) -> rmem(bf16) -> smem(bf16) load.
        `gsrc_slice`/`sdst_slice` are matching thread partitions (same TV layout),
        so the register up-cast is element-for-element layout correct."""
        atom_src = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._dtype, num_bits_per_copy=128
        )
        atom_dst = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self._compute_dtype
        )
        frag = cute.make_fragment_like(gsrc_slice, self._dtype)
        cute.copy(atom_src, gsrc_slice, frag)
        fragb = cute.make_fragment_like(sdst_slice, self._compute_dtype)
        fragb.store(frag.load().to(self._compute_dtype))
        cute.copy(atom_dst, fragb, sdst_slice)

    # ------------------------------------------------------- online softmax
    @cute.jit
    def _online_softmax(self, acc_S, acc_O, row_max, row_sum, scale_log2):
        # Uniform path (no first-tile special case): on the first tile row_max is
        # -inf so corr = exp2(-inf) = 0, which correctly zeroes the (already-zero)
        # acc_O and drops the empty running sum.
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_max)):
            acc_S_row = acc_S_mn[r, None].load()
            cur = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            cur = self._threadquad_reduce_max(cur)
            prev = row_max[r]
            new = cute.arch.fmax(prev, cur)
            # guard fully-masked rows to avoid nan
            new = 0.0 if new == -cutlass.Float32.inf else new
            exp_row = cute.math.exp2(
                acc_S_row * scale_log2 - new * scale_log2, fastmath=True
            )
            s = exp_row.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            corr = cute.math.exp2(prev * scale_log2 - new * scale_log2, fastmath=True)
            s = s + row_sum[r] * corr
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * corr
            row_max[r] = new
            row_sum[r] = s
            acc_S_mn[r, None] = exp_row

    # ------------------------------------------------------- store O
    @cute.jit
    def _store_O(self, mO, sQ, sO_layout, acc_O, gmem_copy_O, tiled_mma,
                 q_begin, q_head, m_block, q_base, q_row_limit):
        rO = cute.make_fragment_like(acc_O, self._compute_dtype)
        rO.store(acc_O.load().to(self._compute_dtype))
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        tidx, _, _ = cute.arch.thread_idx()
        atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._compute_dtype)
        smem_copy_O = cute.make_tiled_copy_C(atom_O, tiled_mma)
        smem_thr_O = smem_copy_O.get_slice(tidx)
        cute.copy(atom_O, smem_thr_O.retile(rO), smem_thr_O.partition_D(sO))

        # PackGQA: mO is the 2D reshape (total_q*num_q_heads, head_dim_v); this
        # batch's packed rows start at q_begin*gqa. Otherwise index the q_head.
        if cutlass.const_expr(self._pack_gqa):
            mO_h = cute.domain_offset((q_begin * self._gqa_ratio, 0), mO)
        else:
            mO_h = cute.domain_offset((q_begin, 0), mO[None, q_head, None])
        gO = cute.local_tile(
            mO_h, (self._m_block_size, self._head_dim_v_padded), (m_block, 0)
        )
        gmem_thr_O = gmem_copy_O.get_slice(tidx)
        tOsO = gmem_thr_O.partition_S(sO)
        tOgO = gmem_thr_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self._compute_dtype)
        self.cta_sync_barrier.arrive_and_wait()
        cute.copy(gmem_copy_O, tOsO, tOrO)

        cO = cute.make_identity_tensor((self._m_block_size, self._head_dim_v_padded))
        tOcO = gmem_thr_O.partition_D(cO)
        for m in cutlass.range_constexpr(cute.size(tOgO.shape[1])):
            if cute.elem_less(q_base + tOcO[0, m, 0][0], q_row_limit):
                cute.copy(gmem_copy_O, tOrO[None, m, None], tOgO[None, m, None])


# ====================================================================== host wrapper
_KERNEL_CACHE = {}


def _cute_tensor(t, leading_dim=None):
    ld = t.ndim - 1 if leading_dim is None else leading_dim
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=ld)


def _elem_bits(t):
    import torch
    return {torch.float8_e4m3fn: 8, torch.bfloat16: 16, torch.float16: 16,
            torch.float32: 32}[t.dtype]


def _cute_tensor_aligned(t):
    """Like `_cute_tensor` but also declares the contiguous last dim 128-bit
    aligned (required for the cp.async 128-bit load atoms)."""
    ld = t.ndim - 1
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=ld)
        .mark_compact_shape_dynamic(
            mode=ld, stride_order=t.dim_order(), divisibility=128 // _elem_bits(t)
        )
    )


def flash_attn_with_kvcache_cutedsl(
    q,
    k_cache,
    v_cache,
    *,
    page_table,
    cache_seqlens,
    cu_seqlens_q,
    max_seqlen_q: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Sequence[int] = (-1, -1),
    q_descale: Optional[Union[float, "object"]] = None,
    k_descale=None,
    v_descale=None,
    sinks=None,
    block_sparse_cu=None,
    block_sparse_idx=None,
    total_q_tiles: Optional[int] = None,
    cu_q_tiles=None,
    k_mean=None,
    v_mean=None,
    mean_k_block_size: int = 0,
    num_splits: int = 1,
    **unused,
):
    """Drop-in CuteDSL replacement of ``flash_attn_with_kvcache`` (fwd only)."""
    import torch

    del max_seqlen_q, num_splits, unused

    total_q, num_q_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    head_dim_v = v_cache.shape[-1]
    gqa_ratio = num_q_heads // num_kv_heads
    batch = cache_seqlens.numel()
    device = q.device

    is_fp8 = q.dtype == torch.float8_e4m3fn
    out_dtype = torch.bfloat16 if is_fp8 else q.dtype
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    sparse = block_sparse_cu is not None
    n_block_size = page_size

    # PackGQA (perf path, HKV==1): the (only) kv_head's gqa q-heads are packed
    # into one CTA of `PACKGQA_K_BLOCK_M` contiguous rows of the reshaped tensor
    # q.reshape(total_q*num_q_heads, head_dim) — packed row `r` = query position
    # `r//gqa`, q-head `r%gqa`. KV is then loaded ONCE per tile and reused across
    # all gqa heads (vs. gqa_ratio× reloads for one-q_head-per-CTA), and the tile
    # natively matches the production CSR granularity (k_block_m=PACKGQA_K_BLOCK_M).
    # PackGQA (128-row, one CTA per kv_head, KV loaded once) is implemented and
    # verified numerically identical, but measured SLOWER than the 32-row per-head
    # path on this d=256 config (1.01ms vs 0.88ms on the 9724-tok dump): MMA FLOPs
    # are identical, so the only PackGQA win is 4× less KV load — which is negated
    # by occupancy collapse (its 128KB smem tile fits only 1 CTA/SM vs 2 for the
    # 80KB 32-row tile, and 1 CTA/SM cannot hide the synchronous fp8->bf16 load).
    # Default is therefore the faster 32-row path; PackGQA stays behind the env
    # override (may win once the load is cp.async double-buffered — stage-6 TODO).
    pack_gqa = False
    import os as _os
    _pack_env = _os.environ.get("FLASHPREFILL_CUTEDSL_PACK")
    if _pack_env == "1" and sparse and num_kv_heads == 1:
        pack_gqa = True

    # Query-tile height. The block-sparse CSR is produced by the production
    # PackGQA index builder with `k_block_m = PACKGQA_K_BLOCK_M` packed rows,
    # i.e. `PACKGQA_K_BLOCK_M // gqa_ratio` query positions per CSR tile (the
    # selection set is shared across a kv_head's gqa heads). PackGQA tiles exactly
    # PACKGQA_K_BLOCK_M packed rows = that many positions × gqa heads. Otherwise
    # this CTA owns one q_head, so its query-tile height must equal that many
    # positions to read the identical selection set as Triton — a wider tile would
    # straddle two production CSR tiles and merge their distinct selections. The
    # dense path is self-consistent at any height, so it keeps a 64-row tile.
    if sparse:
        if PACKGQA_K_BLOCK_M % gqa_ratio != 0:
            raise ValueError(
                f"PACKGQA_K_BLOCK_M ({PACKGQA_K_BLOCK_M}) must be divisible by "
                f"gqa_ratio ({gqa_ratio}) to match the production CSR granularity")
        m_block_size = PACKGQA_K_BLOCK_M if pack_gqa else PACKGQA_K_BLOCK_M // gqa_ratio
    else:
        m_block_size = 64
    # Thread count. This is a register-file-bound kernel: the O accumulator holds
    # `m_block_size * head_dim_v / num_threads` f32 registers per thread, and with
    # head_dim_v=256 that must stay ~128 to avoid spilling acc_O to local memory
    # (which is catastrophic — every MMA-accumulate becomes a global round-trip).
    # PackGQA's 128-row tile therefore needs 256 threads (8 warps → 128*256/256 =
    # 128 regs), matching the dense 64-row/128-thread pressure. The 32-row per-head
    # path uses 64 threads. The FA2 base only requires (m_block_size*2)%num_threads==0.
    if pack_gqa:
        num_threads = 256
    else:
        num_threads = 128 if (m_block_size * 2) % 128 == 0 else 64

    # PackGQA tiles packed rows (q_len*gqa), so its tile count is over that many.
    row_mult = gqa_ratio if pack_gqa else 1
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    q_tiles = [(ql * row_mult + m_block_size - 1) // m_block_size for ql in q_lens]
    max_q_blocks = max(q_tiles) if q_tiles else 0
    if max_q_blocks == 0:
        return torch.empty(total_q, num_q_heads, head_dim_v, dtype=out_dtype, device=device)

    if sparse:
        cu_q_tiles_arg = cu_q_tiles.contiguous().to(torch.int32)
        bs_cu = block_sparse_cu.contiguous().to(torch.int32)
        bs_idx = block_sparse_idx.contiguous().to(torch.int32)
        total_q_tiles = int(total_q_tiles)
    else:
        tiles_cum = [0]
        for t in q_tiles:
            tiles_cum.append(tiles_cum[-1] + t)
        cu_q_tiles_arg = torch.tensor(tiles_cum, dtype=torch.int32, device=device)
        total_q_tiles = tiles_cum[-1]
        bs_cu = cu_q_tiles_arg
        bs_idx = cu_q_tiles_arg

    has_kd = k_descale is not None
    has_vd = v_descale is not None
    # sglang passes descale as a stride-0 broadcast view (layer.k_scale.expand((batch,
    # num_kv_heads))). `.to(f32).contiguous()` does NOT materialize it when the tensor is
    # already f32 AND all dims are size 1 (batch==1 single request) — torch flags an
    # all-size-1 tensor as "contiguous" so .contiguous() returns self, keeping stride 0.
    # _cute_tensor()'s mark_layout_dynamic then rejects strides[leading_dim]==0. Force a
    # freshly-allocated standard-stride (num_kv_heads, 1) tensor via empty+copy_ (copy_
    # broadcasts the expanded source), guaranteeing strides[leading_dim]==1 for any input.
    def _materialize_descale(t):
        # A flat (batch*num_kv_heads,) descale reshapes to 2D (old .view() semantics);
        # 2D / broadcast / (num_kv_heads,) sources are handled by copy_'s broadcasting.
        if t.ndim == 1 and t.numel() == batch * num_kv_heads:
            t = t.reshape(batch, num_kv_heads)
        out = torch.empty(batch, num_kv_heads, dtype=torch.float32, device=device)
        out.copy_(t)  # copy_ broadcasts stride-0 / (batch,1) / (num_kv_heads,) sources
        return out
    kd_arg = (_materialize_descale(k_descale) if has_kd
              else torch.ones(batch, num_kv_heads, dtype=torch.float32, device=device))
    vd_arg = (_materialize_descale(v_descale) if has_vd
              else torch.ones(batch, num_kv_heads, dtype=torch.float32, device=device))

    has_sinks = sinks is not None
    sinks_arg = (sinks.to(torch.float32).contiguous() if has_sinks
                 else torch.zeros(num_q_heads, dtype=torch.float32, device=device))

    # Zero-order mean correction (matches Triton: requires sparse + both means +
    # a positive block size that is a multiple of the 64-wide tile).
    mean_corr = (sparse and k_mean is not None and v_mean is not None
                 and mean_k_block_size > 0)
    if mean_corr:
        if mean_k_block_size % n_block_size != 0:
            raise ValueError("mean_k_block_size must be a multiple of the tile size (64)")
        km_arg = k_mean.contiguous()
        vm_arg = v_mean.contiguous()
        mean_k_block = int(mean_k_block_size)
    else:
        mean_k_block = n_block_size
        km_arg = torch.zeros(batch, 1, num_kv_heads, head_dim, dtype=q.dtype, device=device)
        vm_arg = torch.zeros(batch, 1, num_kv_heads, head_dim_v, dtype=q.dtype, device=device)

    # fp8 e4m3: Q/K/V are up-cast to bf16 in-kernel (raw values); the per-(b,h) K
    # descale folds into softmax_scale, the scalar Q descale too, V descale at the
    # epilogue (mirrors the Triton path). Fold scalar q_descale here.
    if is_fp8 and q_descale is not None:
        if isinstance(q_descale, torch.Tensor):
            if q_descale.numel() != 1:
                raise NotImplementedError("only scalar q_descale is supported")
            softmax_scale = softmax_scale * float(q_descale.item())
        else:
            softmax_scale = softmax_scale * float(q_descale)

    out = torch.empty(total_q, num_q_heads, head_dim_v, dtype=out_dtype, device=device)

    key = (q.dtype, head_dim, head_dim_v, gqa_ratio, num_kv_heads, page_size,
           m_block_size, n_block_size, num_threads, causal, is_fp8, has_kd, has_vd,
           sparse, pack_gqa, has_sinks, window_size[0], window_size[1],
           mean_corr, mean_k_block,
           int(max_q_blocks), int(batch), int(num_q_heads))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Flatten the paged cache to (num_pages*page_size, num_kv_heads, head_dim) so
    # a page id doubles as a static-shaped local_tile row-block coordinate.
    k_flat = k_cache.reshape(num_pages * page_size, num_kv_heads, head_dim)
    v_flat = v_cache.reshape(num_pages * page_size, num_kv_heads, head_dim_v)

    # PackGQA reads Q/O as the 2D reshape (total_q*num_q_heads, head_dim); packed
    # rows are contiguous so this is a plain view of the contiguous q/out tensors.
    if pack_gqa:
        mQ = _cute_tensor_aligned(q.reshape(total_q * num_q_heads, head_dim))
        mO = _cute_tensor_aligned(out.reshape(total_q * num_q_heads, head_dim_v))
    else:
        mQ = _cute_tensor_aligned(q)
        mO = _cute_tensor_aligned(out)
    mK = _cute_tensor_aligned(k_flat)
    mV = _cute_tensor_aligned(v_flat)
    mPT = _cute_tensor(page_table.contiguous().to(torch.int32))
    mCS = _cute_tensor(cache_seqlens.contiguous().to(torch.int32), leading_dim=0)
    mCQ = _cute_tensor(cu_seqlens_q.contiguous().to(torch.int32), leading_dim=0)
    mCQT = _cute_tensor(cu_q_tiles_arg, leading_dim=0)
    mBsCu = _cute_tensor(bs_cu, leading_dim=0)
    mBsIdx = _cute_tensor(bs_idx, leading_dim=0)
    mKD = _cute_tensor(kd_arg)
    mVD = _cute_tensor(vd_arg)
    mSinks = _cute_tensor(sinks_arg, leading_dim=0)
    mKmean = _cute_tensor_aligned(km_arg)
    mVmean = _cute_tensor_aligned(vm_arg)

    if key not in _KERNEL_CACHE:
        op = FlashPrefillCuteDSL(
            head_dim, head_dim_v, gqa_ratio, num_kv_heads, page_size,
            m_block_size, n_block_size, num_threads, causal, is_fp8, has_kd, has_vd,
            sparse, window_size[0], window_size[1], has_sinks, mean_corr, mean_k_block,
            pack_gqa,
        )
        _KERNEL_CACHE[key] = cute.compile(
            op, mQ, mK, mV, mO, mPT, mCS, mCQ, mCQT, mBsCu, mBsIdx, mKD, mVD, mSinks,
            mKmean, mVmean,
            cutlass.Float32(softmax_scale), cutlass.Int32(total_q_tiles),
            int(max_q_blocks), int(batch), int(num_q_heads), stream,
        )
    _KERNEL_CACHE[key](
        mQ, mK, mV, mO, mPT, mCS, mCQ, mCQT, mBsCu, mBsIdx, mKD, mVD, mSinks,
        mKmean, mVmean,
        cutlass.Float32(softmax_scale), cutlass.Int32(total_q_tiles), stream,
    )
    return out


__all__ = ["flash_attn_with_kvcache_cutedsl", "FlashPrefillCuteDSL"]
