"""
Fused MoE W8A8 Hopper Kernel — CuTe DSL (SM90) — NoRAPS Variant
================================================================
Same as RAPS variant but with cp.async gather for activations (no TMA,
no pre-scatter). All other optimizations (double-buffered sOut, TMA
descriptor prefetch, SwiGLU tanh, paired FP8, FastDivmod, register
balance, grid ordering) are IDENTICAL to the RAPS kernel.
Used for incremental benchmarking to isolate the RAPS contribution.

Optimized architecture with per-sub-tile WGMMA matching baseline patterns.

Architecture:
- WGMMA: m64n{BN}k32 (per sub-tile), iterated over WN/WARPGROUPS sub-tiles
- A = activations (M=64 tokens), B = weights (N=BN per sub-tile)
- BM = block_m = token count for grid sizing (pad to 64 for WGMMA M)
- BN = tile_n = 32 or 64 (weight columns per sub-tile)
- WN = num_n_chunks = 4 or 8 (total weight sub-tiles = WN*BN/BN = WN)
- STAGES = 1..5 (pipeline depth)
- Consumer threads = WN*32, Producer threads = 128

Phases:
  Phase 1 — Up-Projection:
    Producer:  cp.async-gather X + TMA W_up into smem
    Consumer:  For each TN sub-tile: WGMMA(X, W_sub) → tile_acc[4]
               Apply block-wise scales: bf16x2_acc += scale_w * scale_x * tile_acc

  Phase 2 — SwiGLU + FP8 Quantisation:
    SwiGLU(gate, up) per (tn, tm) pair
    Per-token max-abs → FP8 quantise → swizzled store to sX_down
    Incorporate topk_weights * scaling_factor into token_scale

  Phase 3 — Down-Projection:
    Producer:  TMA W_down into smem
    Consumer:  For each TN2 sub-tile: WGMMA(sX_down, W_down_sub) → tile_acc
               Scale → BF16 → stmatrix to sOut[block_m * PAD]
               cp.reduce.async.bulk scatter-add to global output
"""

import functools
import json
from typing import Tuple


import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.utils.layout import LayoutEnum
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass._mlir.dialects import llvm

from quack.pipeline import PipelineTmaCpAsync, make_pipeline_state, PipelineStateWAdvance
from quack.copy_utils import swizzle_int, gather_m_get_copy_fn


# ═══════════════════════════════════════════════════════════════════════
#  Auto-tune config helpers
# ═══════════════════════════════════════════════════════════════════════

@functools.lru_cache()
def get_best_cute_config(path: str, n_tokens: int) -> dict:
    """Load the best CuTe DSL kernel config for a given batch size.

    Returns dict with keys: block_m, block_n, warp_n, stages.
    Uses nearest-batch-size selection (same as Alpha's get_best_config).
    """
    with open(path, "r") as f:
        best_conf = json.load(f)
    dist = float("inf")
    ret = None
    for nt, val in best_conf.items():
        if abs(int(nt) - n_tokens) < dist:
            dist = abs(int(nt) - n_tokens)
            ret = val
    return ret


def config_to_kernel_params(config: dict) -> dict:
    """Convert auto-tune config to FusedMoE_W8A8_UpDownAcc constructor kwargs.

    Maps (block_n, warp_n) → (tile_shape_n, num_n_chunks):
      (64, 4) → tile_shape_n=64, num_n_chunks=4
      (32, 8) → tile_shape_n=32, num_n_chunks=8
    """
    bn = config["block_n"]
    wn = config["warp_n"]
    return {
        "tile_shape_n": bn,
        "num_n_chunks": wn,
        "num_stages": config["stages"],
        "block_m": config["block_m"],
    }


# ═══════════════════════════════════════════════════════════════════════
#  PTX intrinsics (inline PTX for operations without CuTe DSL equivalents)

#  Remaining inline PTX (no CuTe DSL equivalent):
#    tanh_ptx         — tanh.approx.f32 (no cute.math.tanh fastmath support)
#    tanh_ptx         — tanh.approx.f32 (SiLU/SwiGLU)
#    cp_reduce_async_bulk_bf16 — non-TMA scatter-add to global
#    cvt_f32x2_to_fp8x2_e4m3  — packed FP8 conversion
#    st_shared_b8_int — byte-level shared memory store (FP8 quantize)
#    cp_async_ca_4b   — 4B scatter copy for scales
#    stmatrix_x4_trans_ptx — register-to-smem matrix store
#    bf16x2 pack/unpack/abs/max/add — packed BF16 operations
#    umulhi           — mul.hi.u32 for FastDivmod
# ═══════════════════════════════════════════════════════════════════════

@dsl_user_op
def tanh_ptx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(), [Float32(x).ir_value()],
            "tanh.approx.f32 $0, $1;",
            "=f,f", has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


def silu_swiglu(gate, up):
    """SiLU(gate) * up — using identity: SiLU(x) = x/2 * tanh(x/2) + x/2
    Compiles to FMUL, TANH, FFMA, FMUL (4 ops instead of 5)."""
    half_gate = gate * Float32(0.5)
    tanh_val = tanh_ptx(half_gate)
    silu_gate = half_gate * tanh_val + half_gate  # FFMA: saves one FMUL vs separate sigmoid
    return silu_gate * up


@dsl_user_op
def cp_reduce_async_bulk_bf16(dst_gmem_ptr, src_smem_ptr, num_bytes: Int32,
                               *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as llvm_dialect
    from cutlass._mlir import ir as mlir_ir
    dst_val = dst_gmem_ptr.ir_value(loc=loc, ip=ip) if hasattr(dst_gmem_ptr, 'ir_value') else dst_gmem_ptr
    src_val = src_smem_ptr.ir_value(loc=loc, ip=ip) if hasattr(src_smem_ptr, 'ir_value') else src_smem_ptr
    size_val = Int32(num_bytes).ir_value(loc=loc, ip=ip)
    i32_ty = mlir_ir.IntegerType.get_signless(32)
    i64_ty = mlir_ir.IntegerType.get_signless(64)
    dst_int = llvm_dialect.ptrtoint(i64_ty, dst_val, loc=loc, ip=ip)
    src_int = llvm_dialect.ptrtoint(i32_ty, src_val, loc=loc, ip=ip)
    llvm.inline_asm(
        None, [dst_int, src_int, size_val],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;",
        "l,r,r", has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def cvt_f32x2_to_fp8x2_e4m3(hi: Float32, lo: Float32, *, loc=None, ip=None) -> Int32:
    """Convert two f32 values to packed fp8x2 with a SINGLE cvt instruction.

    Returns an Int32 where low byte = fp8(lo), high byte = fp8(hi).
    """
    from cutlass._mlir import ir as mlir_ir
    hi_ir = Float32(hi).ir_value(loc=loc, ip=ip)
    lo_ir = Float32(lo).ir_value(loc=loc, ip=ip)
    i16_ty = mlir_ir.IntegerType.get_signless(16)
    packed = llvm.inline_asm(
        i16_ty,
        [hi_ir, lo_ir],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f",
        has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    i32_ty = mlir_ir.IntegerType.get_signless(32)
    return Int32(llvm.zext(i32_ty, packed, loc=loc, ip=ip))


@dsl_user_op
def st_shared_b8_int(addr: Int32, val: Int32, *, loc=None, ip=None):
    a = Int32(addr).ir_value()
    v = Int32(val).ir_value()
    llvm.inline_asm(
        None, [a, v],
        "st.shared.b8 [$0], $1;",
        "r,r", has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def cp_async_ca_4b(dst_smem_addr: Int32,
                    src_gmem_addr: cutlass.Int64,
                    *, loc=None, ip=None):
    """cp.async.ca.shared.global [smem], [gmem], 4 — 4-byte async copy.
    Used for scale loading to avoid fence_proxy_async overhead.
    Matches Alpha's CP_ASYNC_CG4 macro. Participates in cp.async.commit_group
    and cp_async_mbarrier_arrive_noinc for barrier tracking."""
    dst_int = Int32(dst_smem_addr).ir_value()
    src_int = cutlass.Int64(src_gmem_addr).ir_value()
    llvm.inline_asm(
        None, [dst_int, src_int],
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l", has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


# ═══════════════════════════════════════════════════════════════════════
#  StMatrix inline PTX (bypasses CuTe verification for unswizzled smem)
# ═══════════════════════════════════════════════════════════════════════

@dsl_user_op
def stmatrix_x4_trans_ptx(smem_addr: Int32, v0: Int32, v1: Int32,
                           v2: Int32, v3: Int32, *, loc=None, ip=None):
    """stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 — raw PTX."""
    addr = Int32(smem_addr).ir_value(loc=loc, ip=ip)
    r0 = Int32(v0).ir_value(loc=loc, ip=ip)
    r1 = Int32(v1).ir_value(loc=loc, ip=ip)
    r2 = Int32(v2).ir_value(loc=loc, ip=ip)
    r3 = Int32(v3).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None, [addr, r0, r1, r2, r3],
        "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [$0], {$1,$2,$3,$4};",
        "r,r,r,r,r", has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def stmatrix_x2_trans_ptx(smem_addr: Int32, v0: Int32, v1: Int32,
                           *, loc=None, ip=None):
    """stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 — raw PTX."""
    addr = Int32(smem_addr).ir_value(loc=loc, ip=ip)
    r0 = Int32(v0).ir_value(loc=loc, ip=ip)
    r1 = Int32(v1).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None, [addr, r0, r1],
        "stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [$0], {$1,$2};",
        "r,r,r", has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Packed BF16x2 accumulator intrinsics
# ═══════════════════════════════════════════════════════════════════════

@dsl_user_op
def f32_pair_to_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(),
        [Float32(hi).ir_value(), Float32(lo).ir_value()],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f", has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))


@dsl_user_op
def bf16x2_hadd(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(
        T.i32(),
        [Int32(a).ir_value(), Int32(b).ir_value()],
        "add.rn.bf16x2 $0, $1, $2;",
        "=r,r,r", has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))


@dsl_user_op
def bf16x2_lo_to_f32(packed: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(),
        [Int32(packed).ir_value()],
        "{ .reg .b16 lo; cvt.u16.u32 lo, $1; cvt.f32.bf16 $0, lo; }",
        "=f,r", has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))


@dsl_user_op
def bf16x2_hi_to_f32(packed: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.inline_asm(
        T.f32(),
        [Int32(packed).ir_value()],
        "{ .reg .b32 t; .reg .b16 hi; shr.b32 t, $1, 16; cvt.u16.u32 hi, t; cvt.f32.bf16 $0, hi; }",
        "=f,r", has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))


@dsl_user_op
def abs_bf16x2(packed: Int32, *, loc=None, ip=None) -> Int32:
    """abs.bf16x2 — per-element absolute value on packed bf16x2."""
    return Int32(llvm.inline_asm(
        T.i32(),
        [Int32(packed).ir_value()],
        "abs.bf16x2 $0, $1;",
        "=r,r", has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))


@dsl_user_op
def max_bf16x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """max.bf16x2 — per-element max on packed bf16x2."""
    return Int32(llvm.inline_asm(
        T.i32(),
        [Int32(a).ir_value(), Int32(b).ir_value()],
        "max.bf16x2 $0, $1, $2;",
        "=r,r,r", has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))


# ═══════════════════════════════════════════════════════════════════════
#  Fast Integer Division
# ═══════════════════════════════════════════════════════════════════════

@dsl_user_op
def umulhi(a: Int32, b: Int32, *, loc=None, ip=None) -> cutlass.Uint32:
    return cutlass.Uint32(
        llvm.inline_asm(
            cutlass._mlir.ir.IntegerType.get_signless(32),
            [Int32(a).ir_value(loc=loc, ip=ip),
             Int32(b).ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class FastDivmod:
    def __init__(self, divisor: Int32, multiplier: cutlass.Uint32,
                 shift_right: cutlass.Uint32):
        self.divisor = divisor
        self.multiplier = multiplier
        self.shift_right = shift_right

    @cute.jit
    def div(self, dividend: Int32) -> Int32:
        return (
            Int32(umulhi(dividend, self.multiplier) >> self.shift_right)
            if self.divisor != 1
            else dividend
        )

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.divisor, self.multiplier, self.shift_right]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.divisor, self.multiplier, self.shift_right],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FastDivmod(*tuple(obj_list))


def _compute_fast_divmod_params(d):
    if d <= 1:
        return 0, 0
    N = 32
    l = (d - 1).bit_length()
    m = ((1 << (N + l)) + d - 1) // d
    if m < (1 << N):
        return m, l
    m = ((1 << (N + l - 1)) + d - 1) // d
    return m, l - 1


# ═══════════════════════════════════════════════════════════════════════
#  Kernel configuration
# ═══════════════════════════════════════════════════════════════════════
class FusedMoEW8A8Kernel:
    """
    SM90 Fused MoE W8A8 Kernel with per-sub-tile WGMMA.

    WGMMA: m64n{BN}k32 iterated over TN sub-tiles
    - A = activations (M=64, always padded to 64 for WGMMA)
    - B = weights (N=BN per sub-tile, BN=32 or 64)
    - TN = total_tile_n / BN sub-tiles per k-tile

    Template parameters:
        BM (block_m) — tokens per CTA for grid sizing (8-128)
        BK (tile_k)  — 128 (fixed)
        BN (tile_n)  — 32 or 64 (weight columns per sub-tile WGMMA)
        WN (num_n_chunks) — 4 or 8 (TN = WN sub-tiles total)
        STAGES (num_stages) — 1-5 (pipeline depth)
    """

    arch = 90

    def __init__(
        self,
        tile_shape_m: int = 64,
        tile_shape_n: int = 64,
        tile_shape_k: int = 128,
        num_n_chunks: int = 4,
        num_stages: int = 4,
        block_scale_shape: Tuple[int, int] = (128, 128),
        block_m: int = None,
        cluster_y: int = 1,
        persistent_grid_y: int = 0,
        group_m: int = 32,
    ):
        # ─── Core parameters ───
        self.cluster_y = cluster_y
        self.persistent_grid_y = persistent_grid_y
        self.group_m = group_m
        self.tile_n = tile_shape_n             # BN: 32 or 64
        self.tile_k = tile_shape_k             # BK: 128
        self.num_n_chunks = num_n_chunks       # WN: 4 or 8
        self.total_tile_n = self.tile_n * self.num_n_chunks  # 256 typically
        self.num_stages = num_stages

        # ─── Block-m for grid sizing ───
        self.block_m = block_m if block_m is not None else 64

        # tile_m is no longer needed — both up-proj and down-proj use block_m
        # directly in the WGMMA N dimension (m64n{block_m}k32), eliminating
        # padding waste. Kept only for potential external reference.
        self.tile_m = ((self.block_m + 63) // 64) * 64

        # ─── Derived constants ───
        # Minimum WGMMA N-dimension: CuTe DSL's IR verifier rejects B descriptors
        # with shape (1,1):(0,0) when N=8 (single-atom fragment). Use N>=16 for WGMMA
        # and keep block_m for grid sizing / token loading / scatter-add.
        self.wgmma_bm = max(self.block_m, 16)

        # Transposed WGMMA: m64n{wgmma_bm}k32 for up-projection
        # tiles_N = number of 64-col weight sub-tiles
        self.tiles_N = self.total_tile_n // 64  # = 4 for total_tile_n=256
        self.tiles_M_up = self.wgmma_bm // 8   # token groups (wgmma_bm/8)

        # Down-projection parameters
        self.BK2 = self.total_tile_n // 2      # down-proj K (reduction) dimension
        self.BN2 = self.total_tile_n            # down-proj output columns per scatter (= total_tile_n)
        self.w_tiles_per_k_slice = self.BK2 // self.tile_k  # weight TMA loads per BK2 reduction

        # Thread layout
        self.consumer_threads = self.num_n_chunks * 32
        self.producer_threads = 128
        self.threads_per_cta = self.consumer_threads + self.producer_threads
        self.load_warp_id = self.consumer_threads // 32
        self.warpgroups = self.num_n_chunks // 4

        # Data types
        self.io_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = Float32
        self.out_dtype = cutlass.BFloat16

        # WGMMA atom layout — splits work across warpgroups
        # Both up-proj and down-proj use A=weights, B=activations → split M (= weight cols) across WGs
        self.atom_layout_mnk_up = (self.warpgroups, 1, 1)

        # Per-warpgroup sub-tile count (each WG handles tiles_N / warpgroups sub-tiles)
        self.tiles_N_per_wg = self.tiles_N // self.warpgroups

        # Dynamic register budget: allocate what we actually need + headroom.
        # data_regs = tiles_N_per_wg * tiles_M_up * 6 + tiles_M_up * 10 + 30
        # Round up to multiple of 8 (CUDA register allocation granularity).
        self.num_regs_load = 40
        data_regs = self.tiles_N_per_wg * self.tiles_M_up * 6 + self.tiles_M_up * 10 + 30
        max_avail = ((65536 - self.producer_threads * 40) // self.consumer_threads // 8) * 8
        self.num_regs_mma = min(((data_regs + 15) // 8) * 8, max_avail, 256)

        # Block-wise scale granularity
        self.block_scale_shape = block_scale_shape
        self.w_scales_per_cta = self.total_tile_n // block_scale_shape[1]  # 2 for ttn=256, 4 for ttn=512
        self.scales_per_scatter = self.BN2 // block_scale_shape[0]  # K-scale-blocks per output scatter (= ttn/128)

        # Number of threads per warp group
        self.num_threads_per_warp_group = 128

        # Consumer-only barrier
        self.consumer_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.consumer_threads,
        )
        self.producer_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.producer_threads,
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Host-side entry point
    # ═══════════════════════════════════════════════════════════════════
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mX_scale: cute.Tensor,
        mW_up: cute.Tensor,
        mW_up_scale: cute.Tensor,
        mW_down: cute.Tensor,
        mW_down_scale: cute.Tensor,
        mOut: cute.Tensor,
        mSortedTokenIds: cute.Tensor,
        mExpertIds: cute.Tensor,
        mNumTokensPostPadded: cute.Tensor,
        mTopkWeights: cute.Tensor,
        top_k: cutlass.Constexpr[int],
        M: int,
        K: int,
        N: int,
        num_experts: int,
        sorted_num: int,
        scaling_factor: float,
        debug_buf: cute.Tensor,
    ):
        self.top_k = top_k

        tile_n = self.tile_n
        tile_k = self.tile_k
        total_tile_n = self.total_tile_n
        block_m = self.block_m
        wgmma_bm = self.wgmma_bm
        BN2 = self.BN2
        BK2 = self.BK2

        # ─── WGMMA setup ───
        a_major = warpgroup.OperandMajorMode.K
        b_major = warpgroup.OperandMajorMode.K

        # Up-projection (transposed): m64n{wgmma_bm}k32 atom
        # permutation covers total_tile_n; cute.gemm handles all sub-tiles.
        # atom_layout=(warpgroups,1,1) supports both 1 and 2 warpgroups.
        # With 2 WGs, CuTe's permutation assigns sub-tiles interleaved across
        # WGs — Phase 2/3 column mappings use global_tn to handle this.
        mma_op_up = warpgroup.MmaF8Op(
            self.io_dtype, self.io_dtype, self.acc_dtype,
            (64, wgmma_bm, 32),
            warpgroup.OperandSource.SMEM,
            a_major, b_major,
        )
        tiled_mma_up = cute.make_tiled_mma(
            cute.make_mma_atom(mma_op_up),
            atom_layout_mnk=cute.make_layout(self.atom_layout_mnk_up),
            permutation_mnk=(total_tile_n, wgmma_bm, 32),
        )
        # Down-projection: REUSE the same m64n{block_m}k32 atom as up-projection.
        # Matching Alpha: A=W_down (64 output cols in M), B=X_down (block_m tokens in N).
        # This eliminates tile_m padding waste and dramatically reduces register pressure.
        # tiled_mma_up is used for both projections.

        mma_k = cute.size(tiled_mma_up.shape_mnk, mode=[2])
        self.k_blocks_per_stage = tile_k // mma_k

        # ─── Smem layouts ───
        # Weights: (total_tile_n, tile_k, stages) — A operand for up, B for down
        # Use make_smem_layout_a with the up-proj tiler (M=total_tile_n)
        # Note: for K-major FP8 with K=128, make_smem_layout_a and make_smem_layout_b
        # produce identical layouts (same SW128 swizzle), so this is compatible
        # with the down-projection's partition_B usage too.
        mma_tiler_w = (total_tile_n, wgmma_bm, tile_k)
        smem_layout_w = sm90_utils.make_smem_layout_a(
            a_layout=LayoutEnum.ROW_MAJOR,
            mma_tiler_mnk=mma_tiler_w,
            a_dtype=self.io_dtype,
            num_stages=self.num_stages,
        )

        # Activations: (wgmma_bm, tile_k, stages) — B operand for transposed up-proj
        mma_tiler_x = (total_tile_n, wgmma_bm, tile_k)
        smem_layout_x = sm90_utils.make_smem_layout_b(
            b_layout=LayoutEnum.ROW_MAJOR,
            mma_tiler_mnk=mma_tiler_x,
            b_dtype=self.io_dtype,
            num_stages=self.num_stages,
        )
        _x_one_stg = cute.slice_(smem_layout_x, (None, None, 0))

        # sX_down: (wgmma_bm, BK2) — B operand for down-projection (matching Alpha)
        # Same mma_tiler as up-proj: (total_tile_n, wgmma_bm, BK2) with BK2==tile_k
        mma_tiler_x_down = (total_tile_n, wgmma_bm, BK2)
        smem_layout_x_down = sm90_utils.make_smem_layout_b(
            b_layout=LayoutEnum.ROW_MAJOR,
            mma_tiler_mnk=mma_tiler_x_down,
            b_dtype=self.io_dtype,
            num_stages=1,
        )

        # ─── TMA descriptors ───
        smem_w_1stage = cute.slice_(smem_layout_w, (None, None, 0))
        smem_tile_nk = (total_tile_n, tile_k)

        # Up weights: (E*N, K) row-major
        w_up_2d = cute.make_tensor(
            mW_up.iterator,
            cute.make_layout((num_experts * N, K), stride=(K, 1))
        )
        tma_atom_w_up, tma_tensor_w_up = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            w_up_2d,
            smem_w_1stage,
            smem_tile_nk,
        )

        # Down weights: (E*K, N/2) row-major
        N_half = N // 2
        w_down_2d = cute.make_tensor(
            mW_down.iterator,
            cute.make_layout((num_experts * K, N_half), stride=(N_half, 1))
        )
        tma_atom_w_down, tma_tensor_w_down = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            w_down_2d,
            smem_w_1stage,
            smem_tile_nk,
        )

        # ─── TiledCopy for X activation gather (cp.async 128-bit) ───
        copy_atom_x = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.io_dtype,
            num_bits_per_copy=128,
        )
        copy_elems = 128 // self.io_dtype.width  # 16 FP8 elems per load
        k_threads = min(tile_k // copy_elems, 8)
        tiled_copy_x = cute.make_tiled_copy_tv(
            copy_atom_x,
            cute.make_layout(
                (self.producer_threads // k_threads, k_threads),
                stride=(k_threads, 1)),
            cute.make_layout((1, copy_elems)),
        )

        K_scale_cols = K // self.block_scale_shape[1]

        # ─── FastDivmod for top_k ───
        _magic_m, _magic_s = _compute_fast_divmod_params(int(top_k))
        top_k_divmod = FastDivmod(
            Int32(top_k), cutlass.Uint32(_magic_m), cutlass.Uint32(_magic_s)
        )

        # ─── Dimension values ───
        num_k_tiles = Int32(K // tile_k)
        # Total Phase 3 weight TMA loads: (output_K_slices) * (w_tiles_per_k_slice)
        # = (K / total_tile_n) * (BK2 / tile_k)  [= K / tile_k / n_x2 where n_x2 = (N//2)/tile_k]
        self.num_output_k_slices = K // self.BN2   # output K-slices to scatter
        w_tiles_per_k_slice = self.w_tiles_per_k_slice
        num_k_tiles_down = Int32(self.num_output_k_slices * w_tiles_per_k_slice)
        K_div_bs0 = Int32(K // self.block_scale_shape[0])
        K_div_bs1 = Int32(K // self.block_scale_shape[1])
        N_div_bs0 = Int32(N // self.block_scale_shape[0])
        N_div_bs1 = Int32(N // self.block_scale_shape[1])
        scale_rows_down = Int32(K // self.block_scale_shape[0])
        scale_cols_down = Int32((N // 2) // self.block_scale_shape[1])

        # ─── Shared storage ───
        w_smem_size = cute.cosize(smem_layout_w)
        x_smem_size = cute.cosize(smem_layout_x)
        x_down_smem_size = cute.cosize(smem_layout_x_down)
        tiles_M_up = self.tiles_M_up
        tiles_N = self.tiles_N
        num_consumer_warps = self.consumer_threads // 32
        # Packed bf16x2 format: one Int32 per (tm, lane, warp) instead of
        # two Float32 per (tm, lane, warp, {lo,hi}). Halves smem usage.
        block_max_size = tiles_M_up * 4 * num_consumer_warps

        # sOut: wgmma_bm × (BN2+8) bf16 elements — 8-column padding matches Alpha's
        # PAD pattern to avoid shared memory bank conflicts during stmatrix writes
        # Uses wgmma_bm (not block_m) because WGMMA processes wgmma_bm rows
        self.sOut_PAD = BN2 + 8
        sOut_size = wgmma_bm * self.sOut_PAD

        # Union sX (Phase 1) and sOut (Phase 3): they are never used simultaneously.
        # sX is FP8 (1 byte/elem), sOut is BF16 (2 bytes/elem).
        # Store as FP8 element count (= byte count since FP8 is 1 byte).
        sOut_bytes = sOut_size * 2        # BF16 = 2 bytes per element
        self.sOut_elems = sOut_size       # BF16 element count per buffer

        # Scale smem: pipelined through producer (matches baseline pattern)
        # Uses wgmma_bm entries per stage (padding entries zeroed for block_m < wgmma_bm)
        scale_x_up_smem_size = self.num_stages * wgmma_bm
        self.w_scales_per_cta = self.total_tile_n // self.block_scale_shape[1]  # 2 for ttn=256, 4 for ttn=512
        scale_w_up_smem_size = self.num_stages * self.w_scales_per_cta
        # Down-proj weight scales: scales_per_scatter per stage (= ttn/128)
        # For ttn=256: 2; for ttn=512: 4. Matches K-scale-blocks per output scatter.
        scale_w_down_smem_size = self.num_stages * self.scales_per_scatter

        # ── Double-buffered sOut ──
        # Two sOut buffers alternate: stmatrix writes buf[N%2], scatter-add reads
        # buf[(N-1)%2]. Eliminates wait_bulk(0) stall → use wait_bulk(1) instead.
        # Auto-enabled when 2× sOut fits in smem (H200: 227KB limit).
        SMEM_LIMIT = 227 * 1024
        sOut_double_bytes = sOut_bytes * 2
        smem_with_double = (w_smem_size + max(x_smem_size, sOut_double_bytes)
                            + x_down_smem_size + 8192)  # 8KB for barriers/scales/metadata
        self.use_double_sout = smem_with_double <= SMEM_LIMIT
        x_or_sOut_elems = max(x_smem_size, sOut_double_bytes if self.use_double_sout else sOut_bytes)

        @cute.struct
        class SharedStorage:
            mainloop_barriers: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, w_smem_size], 1024
            ]
            # Union: sX (Phase 1 activations) and sOut (Phase 3 output)
            # share this memory — they are never used simultaneously.
            sX_or_sOut: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, x_or_sOut_elems], 1024
            ]
            # sX_down is separate (used simultaneously with sOut in Phase 3)
            sX_down: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, x_down_smem_size], 1024
            ]
            block_max: cute.struct.MemRange[Int32, block_max_size]
            topk_scales: cute.struct.MemRange[Float32, wgmma_bm]
            scale_x_up: cute.struct.MemRange[Float32, scale_x_up_smem_size]
            scale_w_up: cute.struct.MemRange[Float32, scale_w_up_smem_size]
            scale_w_down: cute.struct.MemRange[Float32, scale_w_down_smem_size]
            gather_idx: cute.struct.MemRange[Int32, wgmma_bm]

        self.shared_storage = SharedStorage

        # ─── Grid (with cluster alignment + optional persistence) ───
        grid_x = cute.ceil_div(N, total_tile_n)
        grid_y_raw = cute.ceil_div(sorted_num, block_m)

        persistent_grid_y = self.persistent_grid_y
        cluster_y = self.cluster_y

        grid_y = grid_y_raw + (cluster_y - grid_y_raw % cluster_y) % cluster_y
        max_persistent_iters = 1
        if persistent_grid_y > 0:
            grid_y_p = max((min(grid_y_raw, persistent_grid_y) // cluster_y) * cluster_y,
                           cluster_y)
            grid_y = grid_y_p
            max_persistent_iters = (grid_y_raw + grid_y_p - 1) // grid_y_p
        cluster_y_launch = cluster_y

        # ─── Launch ───
        self.kernel(
            mX, mX_scale,
            mW_up_scale, mW_down_scale,
            tma_atom_w_up, tma_tensor_w_up,
            tma_tensor_w_down, tma_atom_w_down,
            mOut,
            mSortedTokenIds, mExpertIds, mNumTokensPostPadded, mTopkWeights,
            top_k_divmod,
            num_k_tiles, num_k_tiles_down,
            K_div_bs0, K_div_bs1, N_div_bs0, N_div_bs1,
            scale_rows_down, scale_cols_down,
            Int32(M), Int32(K), Int32(N),
            Float32(scaling_factor),
            Int32(grid_y_raw),
            Int32(max_persistent_iters),
            Int32(grid_y),
            Int32(grid_x),
            tiled_mma_up, tiled_copy_x,
            smem_layout_x, smem_layout_w,
            smem_layout_x_down,
            debug_buf,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, cluster_y_launch, 1),
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Device kernel
    # ═══════════════════════════════════════════════════════════════════
    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        x_scale: cute.Tensor,
        w_up_scale: cute.Tensor,
        w_down_scale: cute.Tensor,
        tma_atom_w_up: cute.CopyAtom,
        w_up_tensor: cute.Tensor,
        w_down_tensor: cute.Tensor,
        tma_atom_w_down: cute.CopyAtom,
        out: cute.Tensor,
        sorted_token_ids: cute.Tensor,
        expert_ids: cute.Tensor,
        num_tokens_post_padded: cute.Tensor,
        topk_weights: cute.Tensor,
        top_k_divmod: FastDivmod,
        num_k_tiles: Int32,
        num_k_tiles_down: Int32,
        K_div_bs0: Int32,
        K_div_bs1: Int32,
        N_div_bs0: Int32,
        N_div_bs1: Int32,
        scale_rows_down: Int32,
        scale_cols_down: Int32,
        M: Int32,
        K: Int32,
        N: Int32,
        scaling_factor: Float32,
        actual_tiles_y: Int32,
        max_persistent_iters: Int32,
        grid_y_dim: Int32,
        num_n_tiles: Int32,
        tiled_mma_up: cute.TiledMma,
        tiled_copy_x: cute.TiledCopy,
        x_smem_layout: cute.ComposedLayout,
        w_smem_layout: cute.ComposedLayout,
        x_down_smem_layout: cute.ComposedLayout,
        debug_buf: cute.Tensor,
    ):
        # ─── Compile-time constants ───
        tile_n = self.tile_n
        tile_k = self.tile_k
        total_tile_n = self.total_tile_n
        block_m = self.block_m
        wgmma_bm = self.wgmma_bm
        num_stages = self.num_stages
        consumer_threads = self.consumer_threads
        producer_threads = self.producer_threads
        load_warp_id = self.load_warp_id
        tiles_M_up = self.tiles_M_up
        tiles_N = self.tiles_N
        tiles_N_per_wg = self.tiles_N_per_wg
        BK2 = self.BK2
        BN2 = self.BN2
        num_k_blocks = self.k_blocks_per_stage
        num_consumer_warps = self.consumer_threads // 32
        warpgroups = self.warpgroups
        w_scales_per_cta = self.w_scales_per_cta
        w_tiles_per_k_slice = self.w_tiles_per_k_slice
        num_output_k_slices = self.num_output_k_slices
        scales_per_scatter = self.scales_per_scatter

        # ─── Thread/Block identity ───
        # Grid is launched [N-tiles, M-tiles] matching Alpha's dimGrid(N, M):
        # N varies fastest → adjacent CTAs share M-tile expert → weight L2 hits
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        consumer_tidx = tidx
        consumer_warp_id = consumer_tidx // 32
        consumer_lane_id = consumer_tidx % 32
        lane_id_mod4 = consumer_lane_id % 4
        lane_id_div4 = consumer_lane_id // 4
        warp_id_mod4 = consumer_warp_id % 4
        warp_id_div4 = consumer_warp_id // 4

        producer_tidx = tidx - consumer_threads

        # ─── Shared memory ───
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # sX (Phase 1) lives in sX_or_sOut union
        sX = storage.sX_or_sOut.get_tensor(
            x_smem_layout.outer, swizzle=x_smem_layout.inner
        )
        sW = storage.sW.get_tensor(
            w_smem_layout.outer, swizzle=w_smem_layout.inner
        )
        # sX_down (Phase 2-3) has its own allocation (used with sOut simultaneously)
        sX_down = storage.sX_down.get_tensor(
            x_down_smem_layout.outer, swizzle=x_down_smem_layout.inner
        )

        topk_scales_smem = cute.make_tensor(
            storage.topk_scales.data_ptr(),
            cute.make_layout((wgmma_bm,))
        )

        sOut_PAD = self.sOut_PAD

        # Scale smem tensors: use 1D FLAT layout to avoid CuTe DSL's broken
        # dynamic multi-dimensional indexing. Index as flat[stg * stride + offset].
        scale_x_up_smem = cute.make_tensor(
            storage.scale_x_up.data_ptr(),
            cute.make_layout((num_stages * wgmma_bm,))
        )
        scale_w_up_smem = cute.make_tensor(
            storage.scale_w_up.data_ptr(),
            cute.make_layout((num_stages * w_scales_per_cta,))
        )
        scale_w_down_smem = cute.make_tensor(
            storage.scale_w_down.data_ptr(),
            cute.make_layout((num_stages * scales_per_scatter,))
        )

        sX_down_base_addr = storage.sX_down.data_ptr().toint()
        # sOut reuses sX_or_sOut memory (Phase 3, after Phase 1 sX is no longer needed)
        # BF16 tensor view with padded stride — enables CuTe pointer arithmetic
        # instead of raw byte offsets for stmatrix and scatter-add addressing
        sOut_bf16 = cute.make_tensor(
            cute.recast_ptr(storage.sX_or_sOut.data_ptr(), dtype=cutlass.BFloat16),
            cute.make_layout((wgmma_bm, BN2), stride=(sOut_PAD, 1))
        )
        # Double-buffered sOut: compile-time constants that degenerate to
        # single-buffer when disabled (stride=0 → buf_off always 0, wait(0)).
        sout_buf_stride = self.sOut_elems if self.use_double_sout else 0
        sout_wait_n = 1 if self.use_double_sout else 0

        # Scale smem base addresses for cp.async.ca 4-byte loads
        # (eliminates fence_proxy_async in producer hot loop — matches Alpha's CP_ASYNC_CG4)
        scale_x_up_base_addr = storage.scale_x_up.data_ptr().toint()
        scale_w_up_base_addr = storage.scale_w_up.data_ptr().toint()
        scale_w_down_base_addr = storage.scale_w_down.data_ptr().toint()
        topk_scales_base_addr = storage.topk_scales.data_ptr().toint()
        block_max_base_addr = storage.block_max.data_ptr().toint()

        _swz = sX_down.iterator.type.swizzle_type
        swizzle_b_down = _swz.num_bits
        swizzle_m_down = _swz.num_base
        swizzle_s_down = _swz.num_shift

        # Packed bf16x2 reduction: one packed value per (tm, lane, warp)
        block_max_smem = cute.make_tensor(
            storage.block_max.data_ptr(),
            cute.make_layout((tiles_M_up, 4, num_consumer_warps))
        )

        gather_idx_smem = cute.make_tensor(
            storage.gather_idx.data_ptr(),
            cute.make_layout((wgmma_bm,))
        )

        # ─── Pipeline ───
        w_smem_one_stage = cute.slice_(w_smem_layout, (None, None, 0))
        tma_w_bytes = cute.size_in_bytes(self.io_dtype, w_smem_one_stage)
        tma_copy_bytes = tma_w_bytes  # Only W via TMA; X via cp.async gather

        producer_arrive_count = 1 + producer_threads
        mainloop_pipeline = PipelineTmaCpAsync.create(
            barrier_storage=storage.mainloop_barriers.data_ptr(),
            num_stages=num_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, producer_arrive_count
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, consumer_threads // cute.arch.WARP_SIZE
            ),
            tx_count=tma_copy_bytes,
        )

        # ─── One-time warp-specialized setup (before persistent loop) ───
        # Register allocations, WGMMA partitions, and compile-time constants are
        # hoisted here to avoid re-allocation inside the dynamic persistent loop.

        # -- Producer one-time setup --
        if warp_idx >= load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

            if producer_tidx == 0:
                cpasync.prefetch_descriptor(tma_atom_w_up)
                cpasync.prefetch_descriptor(tma_atom_w_down)

            is_tma_warp = warp_idx == load_warp_id

        # -- Consumer one-time setup --
        if warp_idx < load_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)

            warp_group_idx = consumer_tidx // self.num_threads_per_warp_group
            wg_thread_layout = cute.make_layout(
                self.warpgroups, stride=self.num_threads_per_warp_group
            )
            mma_thread_idx = wg_thread_layout(warp_group_idx)

            thr_mma_up = tiled_mma_up.get_slice(mma_thread_idx)
            tCsW_up = thr_mma_up.partition_A(sW)
            tCrW_up = thr_mma_up.make_fragment_A(tCsW_up)
            tCsX_up = thr_mma_up.partition_B(sX)
            tCrX_up = thr_mma_up.make_fragment_B(tCsX_up)

            mma_tiler_mn_up = (total_tile_n, wgmma_bm)
            acc_shape_up = thr_mma_up.partition_shape_C(mma_tiler_mn_up)
            tile_acc = cute.make_rmem_tensor(acc_shape_up, self.acc_dtype)

            bf16x2_acc = cute.make_rmem_tensor((tiles_N_per_wg, tiles_M_up, 2), Int32)
            reg_scale_x = cute.make_rmem_tensor((tiles_M_up, 2), Float32)

        # ─── Read actual ntp from GPU tensor (enables grid overlaunch) ───
        # Like Triton's tl.load(ntp_ptr) pattern: the grid may be launched
        # with an overestimate of sorted_num to avoid GPU→CPU sync. We read
        # the real ntp from the device tensor to bound the persistent loop.
        ntp_val = num_tokens_post_padded[0]
        actual_tiles_y = (ntp_val + Int32(block_m) - Int32(1)) // Int32(block_m)

        # ─── Persistent grid-stride loop ───
        # Non-persistent: max_persistent_iters=1, loop body runs once.
        for _persistent_iter in cutlass.range(max_persistent_iters):
            my_bidy = bidy + _persistent_iter * grid_y_dim

            is_valid_cta = my_bidy < actual_tiles_y
            effective_bidy = my_bidy
            if my_bidy >= actual_tiles_y:
                effective_bidy = actual_tiles_y - Int32(1)

            effective_bidx = bidx

            # GROUP_M swizzle: only when N-tiles >= 8 (weight working set exceeds L2)
            if self.group_m > 0 and num_n_tiles > Int32(8):
                group_m_val = Int32(self.group_m)
                linear_id = effective_bidy * num_n_tiles + effective_bidx
                group_id = linear_id // (group_m_val * num_n_tiles)
                first_in_group = group_id * group_m_val
                group_sz_raw = actual_tiles_y - first_in_group
                group_sz = group_sz_raw
                if group_sz_raw > group_m_val:
                    group_sz = group_m_val
                tiles_in_group = group_sz * num_n_tiles
                local_id = linear_id - group_id * (group_m_val * num_n_tiles)
                if local_id < tiles_in_group:
                    effective_bidy = first_in_group + local_id % group_sz
                    effective_bidx = local_id // group_sz

            expert_id = expert_ids[effective_bidy]

            if _persistent_iter > Int32(0):
                mainloop_pipeline.sync_object_full.mbarrier_init()
                mainloop_pipeline.sync_object_empty.mbarrier_init()
                cute.arch.mbarrier_init_fence()
                cute.arch.sync_threads()

            # ──────────────────────────────────────────────────────────────
            #  PRODUCER PATH
            # ──────────────────────────────────────────────────────────────
            if warp_idx >= load_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_load)

                if producer_tidx == 0:
                    cpasync.prefetch_descriptor(tma_atom_w_up)
                    cpasync.prefetch_descriptor(tma_atom_w_down)

                is_tma_warp = warp_idx == load_warp_id

                # TMA partition for up-projection weights
                n_tiles_per_expert = N // total_tile_n
                n_tile_idx = expert_id * n_tiles_per_expert + effective_bidx
                tile_shape_nk = (total_tile_n, tile_k)
                gW_up = cute.local_tile(w_up_tensor, tile_shape_nk, (n_tile_idx, None))

                sW_tma = cute.group_modes(sW, 0, 2)
                gW_up_tma = cute.group_modes(gW_up, 0, 2)
                tWsW_up, tWgW_up = cpasync.tma_partition(
                    tma_atom_w_up, 0, cute.make_layout(1), sW_tma, gW_up_tma,
                )

                # ── Per-token precomputation: gather indices + scales + topk ──
                my_tok_id = Int32(0)
                my_src_token = Int32(M)  # invalid by default
                if producer_tidx < wgmma_bm:
                    if producer_tidx < block_m:
                        my_tok_id = sorted_token_ids[effective_bidy * block_m + producer_tidx]
                        my_src_token = top_k_divmod.div(my_tok_id)

                    # Topk weight for this token
                    topk_val = Float32(0.0)
                    if producer_tidx < block_m and my_src_token < M:
                        topk_val = topk_weights[my_tok_id]
                    topk_scales_smem[producer_tidx] = topk_val

                    # Gather index for cp.async activation loads
                    gather_idx_smem[producer_tidx] = my_src_token

                scale_src_token = my_src_token

                # Sync producer threads before gather_m_get_copy_fn reads
                # indices written by other warps
                self.producer_barrier.arrive_and_wait()

                # Build CuTe gather copy (auto-handles swizzle + predication)
                thr_copy_x = tiled_copy_x.get_slice(producer_tidx)
                x_aligned = cute.make_tensor(
                    x.iterator,
                    cute.make_layout(
                        x.shape,
                        stride=tuple(
                            cute.assume(s, divby=128 // self.io_dtype.width)
                            if not cute.is_static(s) else s
                            for s in x.stride
                        )))
                copy_x_fn = gather_m_get_copy_fn(
                    thr_copy_x, x_aligned, sX, gather_idx_smem,
                    limit_m=Int32(block_m), limit_k=K)

                # ═══════════════════════════════════════════════════════════
                #  Phase 1 Producer: Load X + W_up (PipelineState)
                # ═══════════════════════════════════════════════════════════
                k_tile_p = Int32(0)
                p1_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, num_stages)

                for _k in cutlass.range(num_k_tiles):
                    smem_stg = p1_producer_state.index

                    mainloop_pipeline.producer_acquire(p1_producer_state, None, is_tma_warp)

                    if is_tma_warp:
                        tma_bar = mainloop_pipeline.producer_get_barrier(p1_producer_state)
                        # TMA W_up
                        cute.copy(
                            tma_atom_w_up,
                            tWgW_up[(None, k_tile_p)],
                            tWsW_up[(None, smem_stg)],
                            tma_bar_ptr=tma_bar,
                        )

                    # cp.async gather X activations (all producer threads)
                    copy_x_fn(k_tile_p, smem_stg)

                    if producer_tidx < wgmma_bm:
                        xs_src = x_scale.iterator.toint() + cutlass.Int64(
                            (scale_src_token * K_div_bs1 + k_tile_p) * Int32(4))
                        xs_dst = scale_x_up_base_addr + (
                            smem_stg * Int32(wgmma_bm) + producer_tidx) * Int32(4)
                        if scale_src_token < M:
                            cp_async_ca_4b(Int32(xs_dst), xs_src)

                    if producer_tidx < w_scales_per_cta:
                        w_scl_idx = (expert_id * N_div_bs1 * K_div_bs0
                                     + (effective_bidx * Int32(w_scales_per_cta) + producer_tidx) * K_div_bs0
                                     + k_tile_p)
                        ws_src = w_up_scale.iterator.toint() + cutlass.Int64(
                            w_scl_idx * Int32(4))
                        ws_dst = scale_w_up_base_addr + (
                            smem_stg * Int32(w_scales_per_cta) + producer_tidx) * Int32(4)
                        cp_async_ca_4b(Int32(ws_dst), ws_src)

                    mainloop_pipeline.producer_cpasync_commit(p1_producer_state)
                    p1_producer_state.advance()
                    k_tile_p = k_tile_p + Int32(1)

                p1_remainder = p1_producer_state.index
                p_phase = p1_producer_state.phase

                # ═══════════════════════════════════════════════════════════
                #  Phase 3 Producer: Load W_down (PipelineState)
                # ═══════════════════════════════════════════════════════════
                # Weight w2: (E, K, N//2), tiled as (K/ttn, (N//2)/tile_k).
                # Each CTA handles w_tiles_per_k_slice adjacent column tiles
                # (covering BK2 = ttn/2 reduction columns).
                k_rows_per_expert = K // total_tile_n
                row_tile_base_down = expert_id * k_rows_per_expert
                col_tile_base_down = effective_bidx * w_tiles_per_k_slice

                # Pre-partition TMA for each column tile within the BK2 slice.
                # Unrolled manually since CuTe DSL does not allow list mutation in loops.
                # Always create both partitions (d1 aliases d0 when w_tiles=1)
                # because CuTe DSL traces both branches of compile-time ifs.
                gW_d0 = cute.local_tile(w_down_tensor, tile_shape_nk,
                                        (None, col_tile_base_down))
                gW_d0_tma = cute.group_modes(gW_d0, 0, 2)
                tWsW_d0, tWgW_d0 = cpasync.tma_partition(
                    tma_atom_w_down, 0, cute.make_layout(1), sW_tma, gW_d0_tma,
                )
                col1 = col_tile_base_down + (1 if w_tiles_per_k_slice > 1 else 0)
                gW_d1 = cute.local_tile(w_down_tensor, tile_shape_nk, (None, col1))
                gW_d1_tma = cute.group_modes(gW_d1, 0, 2)
                tWsW_d1, tWgW_d1 = cpasync.tma_partition(
                    tma_atom_w_down, 0, cute.make_layout(1), sW_tma, gW_d1_tma,
                )

                p3_producer_state = PipelineStateWAdvance(
                    num_stages, num_k_tiles, p1_remainder, p_phase)
                k_tile_down_p = Int32(0)

                if num_k_tiles_down > 0:

                    for _dk in cutlass.range(num_k_tiles_down):
                        smem_stg = p3_producer_state.index

                        mainloop_pipeline.sync_object_empty.wait(
                            p3_producer_state.index, p3_producer_state.phase)
                        if is_tma_warp:
                            mainloop_pipeline.sync_object_full.arrive_and_expect_tx(
                                p3_producer_state.index, tma_w_bytes)

                        col_within_slice = k_tile_down_p % w_tiles_per_k_slice
                        row_within_k = k_tile_down_p // w_tiles_per_k_slice

                        if is_tma_warp:
                            tma_bar = mainloop_pipeline.producer_get_barrier(
                                p3_producer_state)
                            actual_row = row_tile_base_down + row_within_k
                            if w_tiles_per_k_slice == 1:
                                cute.copy(
                                    tma_atom_w_down,
                                    tWgW_d0[(None, actual_row)],
                                    tWsW_d0[(None, smem_stg)],
                                    tma_bar_ptr=tma_bar,
                                )
                            else:
                                if col_within_slice == 0:
                                    cute.copy(
                                        tma_atom_w_down,
                                        tWgW_d0[(None, actual_row)],
                                        tWsW_d0[(None, smem_stg)],
                                        tma_bar_ptr=tma_bar,
                                    )
                                if col_within_slice == 1:
                                    cute.copy(
                                        tma_atom_w_down,
                                        tWgW_d1[(None, actual_row)],
                                        tWsW_d1[(None, smem_stg)],
                                        tma_bar_ptr=tma_bar,
                                    )

                        if producer_tidx < scales_per_scatter:
                            n_half_col = col_tile_base_down + col_within_slice
                            down_scl_idx = (expert_id * scale_rows_down * scale_cols_down
                                            + (row_within_k * Int32(scales_per_scatter)
                                               + producer_tidx) * scale_cols_down
                                            + n_half_col)
                            ws_down_src = w_down_scale.iterator.toint() + cutlass.Int64(
                                down_scl_idx * Int32(4))
                            ws_down_dst = scale_w_down_base_addr + (
                                smem_stg * Int32(scales_per_scatter) + producer_tidx) * Int32(4)
                            cp_async_ca_4b(Int32(ws_down_dst), ws_down_src)

                        mainloop_pipeline.producer_cpasync_commit(p3_producer_state)
                        p3_producer_state.advance()
                        k_tile_down_p = k_tile_down_p + Int32(1)

                # Producer tail: drain pipeline
                for _tail_i in range(num_stages - 1):
                    p3_producer_state.advance()
                mainloop_pipeline.producer_acquire(
                    p3_producer_state, None, is_tma_warp)

            # ──────────────────────────────────────────────────────────────
            #  CONSUMER PATH
            # ──────────────────────────────────────────────────────────────
            if warp_idx < load_warp_id:
                cute.arch.warpgroup_reg_alloc(self.num_regs_mma)

                # ─── Fragment setup ───
                warp_group_idx = cute.arch.make_warp_uniform(consumer_tidx // self.num_threads_per_warp_group)
                wg_thread_layout = cute.make_layout(
                    self.warpgroups, stride=self.num_threads_per_warp_group
                )
                mma_thread_idx = wg_thread_layout(warp_group_idx)

                thr_mma_up = tiled_mma_up.get_slice(mma_thread_idx)
                tCsW_up = thr_mma_up.partition_A(sW)
                tCrW_up = thr_mma_up.make_fragment_A(tCsW_up)
                tCsX_up = thr_mma_up.partition_B(sX)
                tCrX_up = thr_mma_up.make_fragment_B(tCsX_up)

                mma_tiler_mn_up = (total_tile_n, wgmma_bm)
                acc_shape_up = thr_mma_up.partition_shape_C(mma_tiler_mn_up)
                tile_acc = cute.make_rmem_tensor(acc_shape_up, self.acc_dtype)

                bf16x2_acc = cute.make_rmem_tensor((tiles_N_per_wg, tiles_M_up, 2), Int32)
                for i_init in cutlass.range(tiles_N_per_wg * tiles_M_up * 2):
                    bf16x2_acc[i_init] = Int32(0)

                reg_scale_x = cute.make_rmem_tensor((tiles_M_up, 2), Float32)
                half_scales = w_scales_per_cta // 2
                num_scale_pairs = half_scales
                reg_sw_gate = cute.make_rmem_tensor(const_expr(num_scale_pairs), Float32)
                reg_sw_up = cute.make_rmem_tensor(const_expr(num_scale_pairs), Float32)

                # ── Phase 1 Consumer: PipelineState ──
                c1_consumer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, num_stages)
                for _k in cutlass.range(num_k_tiles):
                    smem_stg = c1_consumer_state.index

                    mainloop_pipeline.consumer_wait(c1_consumer_state)

                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )

                    warpgroup.fence()
                    mma_atom_p1 = cute.make_mma_atom(tiled_mma_up.op)
                    mma_atom_p1.set(warpgroup.Field.ACCUMULATE, False)
                    for k_blk in cutlass.range(num_k_blocks, unroll_full=True):
                        k_coord = (None, None, k_blk, smem_stg)
                        cute.gemm(
                            mma_atom_p1, tile_acc,
                            tCrW_up[k_coord], tCrX_up[k_coord],
                            tile_acc,
                        )
                        mma_atom_p1.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.commit_group()

                    for pi in cutlass.range_constexpr(num_scale_pairs):
                        reg_sw_gate[pi] = scale_w_up_smem[(
                            smem_stg * Int32(w_scales_per_cta) + Int32(pi * 2))]
                        reg_sw_up[pi] = scale_w_up_smem[(
                            smem_stg * Int32(w_scales_per_cta) + Int32(pi * 2 + 1))]

                    xs_flat_base = smem_stg * Int32(wgmma_bm)
                    for tm_pre in cutlass.range_constexpr(tiles_M_up):
                        token_row_pre = const_expr(tm_pre * 8) + lane_id_mod4 * 2
                        reg_scale_x[(tm_pre, 0)] = scale_x_up_smem[(xs_flat_base + token_row_pre)]
                        reg_scale_x[(tm_pre, 1)] = scale_x_up_smem[(xs_flat_base + token_row_pre + 1)]

                    warpgroup.wait_group(0)

                    mainloop_pipeline.consumer_release(c1_consumer_state)

                    for tm in cutlass.range_constexpr(tiles_M_up):
                        for tn in cutlass.range_constexpr(tiles_N_per_wg):
                            pair_idx = const_expr((tn * warpgroups) // (tiles_N // half_scales))
                            sw0 = reg_sw_gate[const_expr(pair_idx)]
                            sw1 = reg_sw_up[const_expr(pair_idx)]
                            sw0_sx0 = sw0 * reg_scale_x[(tm, 0)]
                            sw0_sx1 = sw0 * reg_scale_x[(tm, 1)]
                            sw1_sx0 = sw1 * reg_scale_x[(tm, 0)]
                            sw1_sx1 = sw1 * reg_scale_x[(tm, 1)]
                            base = const_expr(tn * tiles_M_up * 4 + tm * 4)

                            scaled_0 = f32_pair_to_bf16x2(
                                sw0_sx0 * tile_acc[base + 0],
                                sw0_sx1 * tile_acc[base + 1],
                            )
                            bf16x2_acc[(tn, tm, 0)] = bf16x2_hadd(
                                bf16x2_acc[(tn, tm, 0)], scaled_0
                            )

                            scaled_1 = f32_pair_to_bf16x2(
                                sw1_sx0 * tile_acc[base + 2],
                                sw1_sx1 * tile_acc[base + 3],
                            )
                            bf16x2_acc[(tn, tm, 1)] = bf16x2_hadd(
                                bf16x2_acc[(tn, tm, 1)], scaled_1
                            )

                    c1_consumer_state.advance()

                c1_remainder = c1_consumer_state.index
                c_phase = c1_consumer_state.phase

                # ═══════════════════════════════════════════════════════════
                #  Phase 2: SwiGLU + FP8 Quantisation
                # ═══════════════════════════════════════════════════════════
                FP8_MAX = 448.0
                FP8_MIN = -FP8_MAX
                MIN_SCALE = 1e-6

                # Fused SwiGLU + max-abs using packed bf16x2 (matching Alpha's approach).
                # Using bf16x2 abs/max operations halves instruction count vs f32 pairs,
                # and the packed format carries through lane/cross-warp reduction.
                EPS_PACKED = f32_pair_to_bf16x2(Float32(1e-10), Float32(1e-10))
                token_max_packed = cute.make_rmem_tensor(tiles_M_up, Int32)
                for tm_init in cutlass.range_constexpr(tiles_M_up):
                    token_max_packed[tm_init] = EPS_PACKED

                for tn in cutlass.range_constexpr(tiles_N_per_wg):
                    for tm in cutlass.range_constexpr(tiles_M_up):
                        gate_0 = bf16x2_lo_to_f32(bf16x2_acc[(tn, tm, 0)])
                        gate_1 = bf16x2_hi_to_f32(bf16x2_acc[(tn, tm, 0)])
                        up_0 = bf16x2_lo_to_f32(bf16x2_acc[(tn, tm, 1)])
                        up_1 = bf16x2_hi_to_f32(bf16x2_acc[(tn, tm, 1)])

                        sg_0 = silu_swiglu(gate_0, up_0)
                        sg_1 = silu_swiglu(gate_1, up_1)
                        sg_packed = f32_pair_to_bf16x2(sg_0, sg_1)
                        bf16x2_acc[(tn, tm, 0)] = sg_packed

                        # Fused max-abs using packed bf16x2 (1 abs + 1 max vs 4 ops)
                        abs_packed = abs_bf16x2(sg_packed)
                        token_max_packed[tm] = max_bf16x2(token_max_packed[tm], abs_packed)

                # Lane reduction using packed bf16x2 (halves shuffle + max count)
                for tm in cutlass.range_constexpr(tiles_M_up):
                    for offset in [16, 8, 4]:
                        shfl_packed = cute.arch.shuffle_sync_bfly(token_max_packed[tm], offset=Int32(offset))
                        token_max_packed[tm] = max_bf16x2(token_max_packed[tm], shfl_packed)

                # Cross-warp reduction via smem (packed bf16x2: one store/load per warp)
                if consumer_lane_id < 4:
                    for tm in cutlass.range_constexpr(tiles_M_up):
                        block_max_smem[(tm, consumer_lane_id, consumer_warp_id)] = token_max_packed[tm]

                self.consumer_barrier.arrive_and_wait()

                for tm in cutlass.range_constexpr(tiles_M_up):
                    for w in cutlass.range_constexpr(num_consumer_warps):
                        val_packed = block_max_smem[(tm, lane_id_mod4, w)]
                        token_max_packed[tm] = max_bf16x2(token_max_packed[tm], val_packed)

                # Unpack to f32 for scale computation
                token_max = cute.make_rmem_tensor((tiles_M_up, 2), Float32)
                for tm in cutlass.range_constexpr(tiles_M_up):
                    token_max[(tm, 0)] = bf16x2_lo_to_f32(token_max_packed[tm])
                    token_max[(tm, 1)] = bf16x2_hi_to_f32(token_max_packed[tm])

                # Token scale + pre-compute reciprocal (avoids division in quantize loop)
                token_scale = cute.make_rmem_tensor((tiles_M_up, 2), Float32)
                inv_token_scale = cute.make_rmem_tensor((tiles_M_up, 2), Float32)
                INV_FP8_MAX = Float32(1.0 / FP8_MAX)
                for tm in cutlass.range_constexpr(tiles_M_up):
                    ts0 = cute.arch.fmax(token_max[(tm, 0)] * INV_FP8_MAX, MIN_SCALE)
                    ts1 = cute.arch.fmax(token_max[(tm, 1)] * INV_FP8_MAX, MIN_SCALE)
                    token_scale[(tm, 0)] = ts0
                    token_scale[(tm, 1)] = ts1
                    inv_token_scale[(tm, 0)] = Float32(1.0) / ts0
                    inv_token_scale[(tm, 1)] = Float32(1.0) / ts1

                # FP8 quantise → swizzled store to sX_down (block_m rows only)
                # Column offset: each warpgroup writes to its own portion of sX_down
                # Matches Alpha: x_col = (warp_id/4)*(TN*32) + tn*32 + (warp_id%4)*8 + lane_id/4
                #
                # Only write block_m valid rows (no padding needed — down-proj WGMMA
                # now uses N=block_m, so no padding rows are read).
                # Optimization: use cvt_f32x2_to_fp8x2_e4m3 to convert BOTH row0/row1
                # values with a SINGLE CVT instruction (saves one CVT per iteration).
                for tm in cutlass.range_constexpr(tiles_M_up):
                    for tn in cutlass.range_constexpr(tiles_N_per_wg):
                        x_row0 = tm * 8 + lane_id_mod4 * 2 + 0
                        x_row1 = tm * 8 + lane_id_mod4 * 2 + 1
                        # Global sub-tile index: CuTe permutation interleaves WGs,
                        # so WG0 gets sub-tiles {0,2,...}, WG1 gets {1,3,...}.
                        global_tn = const_expr(tn * warpgroups) + warp_id_div4
                        x_col = global_tn * 32 + warp_id_mod4 * 8 + lane_id_div4

                        k_seg = x_col // Int32(128)
                        k_loc = x_col % Int32(128)
                        i0 = k_seg * Int32(wgmma_bm * 128) + x_row0 * Int32(128) + k_loc
                        i1 = k_seg * Int32(wgmma_bm * 128) + x_row1 * Int32(128) + k_loc

                        sw0 = swizzle_int(Int32(i0), swizzle_b_down, swizzle_m_down, swizzle_s_down)
                        sw1 = swizzle_int(Int32(i1), swizzle_b_down, swizzle_m_down, swizzle_s_down)

                        inv_s0 = inv_token_scale[(tm, 0)]
                        inv_s1 = inv_token_scale[(tm, 1)]
                        val0 = bf16x2_lo_to_f32(bf16x2_acc[(tn, tm, 0)])
                        val1 = bf16x2_hi_to_f32(bf16x2_acc[(tn, tm, 0)])

                        q_val0 = val0 * inv_s0
                        q_val1 = val1 * inv_s1
                        # No explicit clamp needed: cvt.rn.satfinite.e4m3x2
                        # automatically clamps to [-448, 448] range.

                        # Paired FP8 conversion: one CVT for both values
                        # packed = [hi_byte=fp8(q_val1), lo_byte=fp8(q_val0)]
                        packed_fp8 = cvt_f32x2_to_fp8x2_e4m3(q_val1, q_val0)
                        st_shared_b8_int(Int32(sX_down_base_addr + sw0), packed_fp8)
                        st_shared_b8_int(Int32(sX_down_base_addr + sw1), packed_fp8 >> Int32(8))

                # Incorporate topk_weights and scaling_factor
                for tm in cutlass.range_constexpr(tiles_M_up):
                    token_idx = tm * 8 + lane_id_mod4 * 2
                    topk_w_0 = topk_scales_smem[token_idx]
                    topk_w_1 = topk_scales_smem[token_idx + 1]
                    token_scale[(tm, 0)] = token_scale[(tm, 0)] * topk_w_0 * scaling_factor
                    token_scale[(tm, 1)] = token_scale[(tm, 1)] * topk_w_1 * scaling_factor

                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta,
                )
                self.consumer_barrier.arrive_and_wait()

                # (Phase 2 debug removed)

                # ═══════════════════════════════════════════════════════════
                #  Phase 3: Down-Projection with stmatrix + scatter-add
                #
                #  Operation ordering matches Alpha-MoE exactly:
                #    1. WGMMA (consume sW from pipeline)
                #    2. Release pipeline (sW freed — unblocks producer)
                #    3. Scale acc → BF16 register pairs (overlaps with producer)
                #    4. Wait for PREVIOUS scatter-add (sOut safe to overwrite)
                #    5. stmatrix to sOut
                #    6. Barrier (all stmatrix visible)
                #    7. scatter-add from sOut → gmem
                # ═══════════════════════════════════════════════════════════

                # Matching Alpha: A=W_down (64 output cols), B=X_down (block_m tokens)
                # REUSE thr_mma_up from Phase 1 setup (same partition, same sW buffer)
                tCsW_down = thr_mma_up.partition_A(sW)
                tCrW_down = thr_mma_up.make_fragment_A(tCsW_down)
                tCsX_down_part = thr_mma_up.partition_B(sX_down)
                tCrX_down = thr_mma_up.make_fragment_B(tCsX_down_part)

                # CRITICAL: Reuse tile_acc for down-projection (same WGMMA shape).
                # In C++, scoped {} blocks let the compiler reuse registers. In CuTe DSL's
                # flat Python scope, separate variables (tile_acc vs tile_acc) cause the
                # compiler to keep BOTH alive → 64-80 extra registers → spills.
                # Reusing tile_acc saves these registers, matching Alpha's C++ scoping.

                # Pre-compute scatter-add token source + output base (hoisted from inner loop)
                scatter_tok_src = Int32(M)  # invalid by default
                scatter_dst_base = Int32(0)
                scatter_src_off = Int32(0)
                if consumer_tidx < block_m:
                    scatter_tok_id = sorted_token_ids[effective_bidy * block_m + consumer_tidx]
                    scatter_tok_src = top_k_divmod.div(scatter_tok_id)
                    scatter_dst_base = scatter_tok_src * K
                    scatter_src_off = consumer_tidx * sOut_PAD

                # Pre-compute stmatrix address components (loop-invariant)
                # Reduces IMAD count inside the hot inner loops
                lane_mod8 = consumer_lane_id % 8
                lane_and8 = consumer_lane_id & Int32(8)
                lane_div16 = consumer_lane_id // 16
                # stmatrix lane-level address components (loop-invariant)
                out_col_lane = (
                    warp_id_mod4 * Int32(16)
                    + lane_and8
                    + lane_div16 * Int32(64)
                )

                if num_k_tiles_down > 0:

                    sout_buf = Int32(0)

                    c3_consumer_state = PipelineStateWAdvance(
                        num_stages, num_k_tiles, c1_remainder, c_phase)
                    k_tile_down_c = Int32(0)
                    wt_counter = Int32(0)
                    reg_sw_dn = cute.make_rmem_tensor(const_expr(scales_per_scatter), Float32)

                    for _dk in cutlass.range(num_k_tiles_down):
                        smem_stg = c3_consumer_state.index

                        mainloop_pipeline.consumer_wait(c3_consumer_state)

                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )

                        warpgroup.fence()
                        mma_atom_p3 = cute.make_mma_atom(tiled_mma_up.op)
                        mma_atom_p3.set(warpgroup.Field.ACCUMULATE, False)
                        if w_tiles_per_k_slice == 1:
                            for k_blk in cutlass.range(num_k_blocks, unroll_full=True):
                                cute.gemm(
                                    mma_atom_p3, tile_acc,
                                    tCrW_down[(None, None, k_blk, smem_stg)],
                                    tCrX_down[(None, None, k_blk, 0)],
                                    tile_acc,
                                )
                                mma_atom_p3.set(warpgroup.Field.ACCUMULATE, True)
                        else:
                            if wt_counter == 0:
                                for k_blk in cutlass.range(num_k_blocks, unroll_full=True):
                                    cute.gemm(
                                        mma_atom_p3, tile_acc,
                                        tCrW_down[(None, None, k_blk, smem_stg)],
                                        tCrX_down[(None, None, k_blk, 0)],
                                        tile_acc,
                                    )
                                    mma_atom_p3.set(warpgroup.Field.ACCUMULATE, True)
                            if wt_counter == 1:
                                for k_blk in cutlass.range(num_k_blocks, unroll_full=True):
                                    cute.gemm(
                                        mma_atom_p3, tile_acc,
                                        tCrW_down[(None, None, k_blk, smem_stg)],
                                        tCrX_down[(None, None, k_blk + const_expr(num_k_blocks), 0)],
                                        tile_acc,
                                    )
                                    mma_atom_p3.set(warpgroup.Field.ACCUMULATE, True)
                        warpgroup.commit_group()

                        for si_dn in cutlass.range_constexpr(scales_per_scatter):
                            reg_sw_dn[si_dn] = scale_w_down_smem[(
                                smem_stg * Int32(scales_per_scatter) + Int32(si_dn))]

                        warpgroup.wait_group(0)

                        mainloop_pipeline.consumer_release(c3_consumer_state)

                        cute.arch.cp_async_bulk_wait_group(sout_wait_n)
                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        self.consumer_barrier.arrive_and_wait()

                        buf_off = sout_buf * Int32(sout_buf_stride)
                        for tn in cutlass.range_constexpr(tiles_N_per_wg):
                            global_tn = const_expr(tn * warpgroups) + warp_id_div4
                            sw_dn = reg_sw_dn[const_expr((tn * warpgroups) // 2)]
                            for tm_idx in cutlass.range_constexpr(tiles_M_up):
                                ts0 = token_scale[(tm_idx, 0)] * sw_dn
                                ts1 = token_scale[(tm_idx, 1)] * sw_dn
                                base = const_expr(tn * tiles_M_up * 4 + tm_idx * 4)
                                v0 = f32_pair_to_bf16x2(
                                    tile_acc[base + 0] * ts0, tile_acc[base + 1] * ts1)
                                v1 = f32_pair_to_bf16x2(
                                    tile_acc[base + 2] * ts0, tile_acc[base + 3] * ts1)
                                out_row = Int32(tm_idx * 8) + lane_mod8
                                out_col = out_col_lane + global_tn * Int32(64)
                                smem_addr = (sOut_bf16.iterator + buf_off + (out_row * Int32(sOut_PAD) + out_col)).toint()
                                stmatrix_x2_trans_ptx(smem_addr, v0, v1)

                        cute.arch.fence_proxy(
                            cute.arch.ProxyKind.async_shared,
                            space=cute.arch.SharedSpace.shared_cta,
                        )
                        self.consumer_barrier.arrive_and_wait()

                        if consumer_tidx < block_m:
                            if scatter_tok_src < M and is_valid_cta:
                                k_off = k_tile_down_c * Int32(BN2)
                                dst_off = scatter_dst_base + k_off
                                dst_ptr = (out.iterator + dst_off).llvm_ptr
                                src_ptr = (sOut_bf16.iterator + buf_off + scatter_src_off).llvm_ptr
                                num_bytes = BN2 * 2
                                cp_reduce_async_bulk_bf16(dst_ptr, src_ptr, num_bytes)

                        cute.arch.cp_async_bulk_commit_group()
                        sout_buf = sout_buf ^ Int32(1)

                        wt_counter = wt_counter + Int32(1)
                        if wt_counter >= w_tiles_per_k_slice:
                            k_tile_down_c = k_tile_down_c + Int32(1)
                            wt_counter = Int32(0)

                        c3_consumer_state.advance()

                    # Final wait for last scatter-add
                    cute.arch.cp_async_bulk_wait_group(0)

            cute.arch.sync_threads()


# Alias for test harness compatibility
FusedMoE_W8A8_UpDownAcc = FusedMoEW8A8Kernel
