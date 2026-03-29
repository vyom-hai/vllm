# SPDX-License-Identifier: Apache-2.0
"""
RouteMoE dispatch: wave-based cost model + CuTe DSL kernel cache.

Core components:
  - WaveCostModel: fits overhead + tpw * eff_waves(grid) per config
  - KernelCache: JIT compiles and caches CuTe DSL kernels
  - interleave_tensor: gate/up weight interleaving for kernel layout
  - ensure_block_scales / ensure_act_scale: scale broadcasting
"""

import importlib
import json
import math
import os
import threading
import time as _time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")

SM_COUNT = 132  # H200 SXM

_CACHE_DIR = Path(os.environ.get(
    "ROUTEMOE_CACHE_DIR",
    str(Path.home() / ".cache" / "routemoe"),
))


# ── Wave cost model (Eq. 4-5 from paper) ─────────────────────────────────

def eff_waves(grid: int, sm: int = SM_COUNT) -> float:
    if grid <= 0:
        return 0.0
    return (grid // sm) + (grid % sm) / sm


def compute_grid(counts: list[int], bm: int, N: int, bn: int, wn: int) -> int:
    ttn = bn * wn
    m_tiles = sum(math.ceil(c / bm) for c in counts if c > 0)
    n_tiles = math.ceil(N / ttn)
    return m_tiles * n_tiles


def fit_wave_model(
    grids: list[int], times: list[float],
) -> tuple[float, float, float]:
    """Fit: time = overhead + tpw * eff_waves(grid) + cta_cost * grid.

    3-parameter model that captures:
      - overhead: fixed per-launch cost (pipeline fill/drain, SwiGLU)
      - tpw: time per SM wave in steady state
      - cta_cost: per-CTA scheduling overhead (penalises large grids)

    Returns (overhead, tpw, cta_cost), each clamped ≥ 0.
    """
    ew = np.array([eff_waves(g) for g in grids], dtype=np.float64)
    g = np.array(grids, dtype=np.float64)
    y = np.array(times, dtype=np.float64)
    # Design matrix: [1, eff_waves, grid]
    A = np.column_stack([np.ones_like(ew), ew, g])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return (max(0.0, float(coeffs[0])),   # overhead
            max(0.0, float(coeffs[1])),    # tpw
            max(0.0, float(coeffs[2])))    # cta_cost


def estimate_time(
    overhead: float, tpw: float, grid: int,
    cta_cost: float = 0.0,
) -> float:
    return overhead + tpw * eff_waves(grid) + cta_cost * grid


# ── Config validity (C1-C5 from paper) ───────────────────────────────────

def config_valid(bm: int, bn: int, wn: int, stg: int) -> bool:
    ttn = bn * wn
    wgmma_bm = max(bm, 16)
    tile_k = 128
    if ttn % 64 != 0 or wn % 4 != 0:
        return False
    tiles_N = ttn // 64
    warpgroups = wn // 4
    if tiles_N % warpgroups != 0:
        return False
    BK2 = ttn // 2
    if BK2 < tile_k or BK2 // tile_k > 2:
        return False
    if bn not in (16, 32, 64, 128):
        return False
    consumer_threads = wn * 32
    sw = ttn * tile_k * stg
    sx = wgmma_bm * tile_k * stg
    sxd = BK2 * wgmma_bm
    sout_double = wgmma_bm * (ttn + 8) * 2 * 2
    smem = sw + max(sx, sout_double) + sxd + 8192
    if smem > 227 * 1024:
        return False
    max_regs = min(((65536 - 128 * 40) // consumer_threads // 8) * 8, 256)
    tiles_N_per_wg = tiles_N // warpgroups
    tiles_M_up = wgmma_bm // 8
    data_regs = tiles_N_per_wg * tiles_M_up * 6 + tiles_M_up * 10 + 30
    needed_regs = ((data_regs + 15) // 8) * 8
    if needed_regs > max_regs:
        return False
    return True


def enumerate_valid_configs() -> list[tuple[int, int, int, int]]:
    configs = []
    for bn in [32, 64, 128]:
        for wn in [4, 8]:
            for stg in range(1, 6):
                for bm in range(8, 129, 8):
                    if config_valid(bm, bn, wn, stg):
                        configs.append((bm, bn, wn, stg))
    return configs


# ── Weight interleaving ──────────────────────────────────────────────────

def interleave_tensor(tensor: torch.Tensor, rep: int = 8) -> torch.Tensor:
    """Interleave gate+up halves at `rep`-column granularity.

    Input:  (E, 2*N, K) -- first N cols gate, second N cols up.
    Output: (E, 2*N, K) -- alternating rep-col blocks of gate and up.
    """
    E, full_N, K = tensor.shape
    N = full_N // 2
    first = tensor[:, :N, :].reshape(E, N // rep, rep, K)
    second = tensor[:, N:, :].reshape(E, N // rep, rep, K)
    interleaved = torch.stack([first, second], dim=2)
    return interleaved.reshape(E, full_N, K).contiguous()


# ── Scale broadcasting ───────────────────────────────────────────────────

def ensure_block_scales(
    w_scale: torch.Tensor, E: int, N_cols: int, K: int, block: int = 128,
) -> torch.Tensor:
    """Broadcast per-tensor/per-expert scales to block-scale format.

    The CuTe kernel expects (E, ceil(N_cols/128), ceil(K/128)) block scales.
    vLLM per-tensor FP8 provides (E,) or (E, 1) or (E, 1, 1).
    Block-quantized FP8 already provides the right shape.
    """
    target = (E, math.ceil(N_cols / block), math.ceil(K / block))
    if w_scale.shape == target:
        return w_scale
    return w_scale.reshape(
        E, *([1] * (3 - w_scale.dim()))
    ).expand(target).contiguous()


def ensure_act_scale(
    a_scale: torch.Tensor | None, M: int, K: int, block: int = 128,
) -> torch.Tensor:
    """Broadcast per-token activation scale to block-scale format.

    Kernel expects (M, ceil(K/128)). vLLM may provide (M,), (M, 1),
    or (M, ceil(K/128)) depending on quantization scheme.
    """
    target = (M, math.ceil(K / block))
    if a_scale is not None and a_scale.shape == target:
        return a_scale
    if a_scale is None:
        return torch.ones(target, dtype=torch.float32, device="cuda")
    return a_scale.reshape(
        M if a_scale.numel() >= M else 1, -1
    ).expand(target).contiguous()


# ── Torch -> CuTe tensor bridge ──────────────────────────────────────────

def torch_to_cute(t: torch.Tensor, dtype=None):
    from cutlass.torch import from_dlpack
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = t.view(torch.uint8)
    ct = from_dlpack(t, assumed_align=16, enable_tvm_ffi=True)
    if dtype is not None:
        ct.element_type = dtype
    ct = ct.mark_layout_dynamic(leading_dim=t.dim() - 1)
    return ct


# ── CuTe DSL kernel cache ───────────────────────────────────────────────

class KernelCache:
    """Compiles and caches CuTe DSL kernels via cute.compile().

    Uses cute.compile(..., options="--enable-tvm-ffi") which returns a
    compiled function callable with raw PyTorch tensors. The compiled
    functions are cached in a dict keyed by (bm, bn, wn, stg, K).
    """

    def __init__(self):
        self._cache: dict[tuple, object] = {}
        self._lock = threading.Lock()
        self._kernel_cls = None

    def _load_cls(self):
        if self._kernel_cls is not None:
            return
        here = Path(__file__).parent / "kernel.py"
        spec = importlib.util.spec_from_file_location(
            "routemoe_kernel", str(here))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._kernel_cls = mod.FusedMoE_W8A8_UpDownAcc

    def get_or_compile(
        self, bm: int, bn: int, wn: int, stg: int,
        example_x, example_xs, example_w1, example_w1s,
        example_w2, example_w2s, example_out,
        example_si, example_ei, example_ntp, example_tw,
        topk: int, S: int, K: int, N: int, E: int,
        sorted_num: int, scaling_factor: float, example_db,
    ):
        import cutlass
        import cutlass.cute as cute

        key = (bm, bn, wn, stg, K)
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            self._load_cls()
            kern = self._kernel_cls(
                tile_shape_n=bn, num_n_chunks=wn, num_stages=stg,
                block_m=bm, cluster_y=1, persistent_grid_y=0, group_m=32,
                block_scale_shape=(128, 128),
            )
            compiled = cute.compile(
                kern,
                torch_to_cute(example_x, cutlass.Float8E4M3FN),
                torch_to_cute(example_xs, cutlass.Float32),
                torch_to_cute(example_w1, cutlass.Float8E4M3FN),
                torch_to_cute(example_w1s, cutlass.Float32),
                torch_to_cute(example_w2, cutlass.Float8E4M3FN),
                torch_to_cute(example_w2s, cutlass.Float32),
                torch_to_cute(example_out, cutlass.BFloat16),
                torch_to_cute(example_si, cutlass.Int32),
                torch_to_cute(example_ei, cutlass.Int32),
                torch_to_cute(example_ntp, cutlass.Int32),
                torch_to_cute(example_tw, cutlass.Float32),
                topk, S, K, N, E, sorted_num, scaling_factor,
                torch_to_cute(example_db, cutlass.Float32),
                options="--enable-tvm-ffi",
            )
            self._cache[key] = compiled
            return compiled


# ── Wave cost model manager ──────────────────────────────────────────────

class WaveCostModel:
    def __init__(self, E: int, N: int, K: int, topk: int):
        self.E, self.N, self.K, self.topk = E, N, K, topk
        self.fitted: dict[tuple, tuple[float, float, float]] = {}
        self._profiled = False

    def _cache_path(self) -> Path:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return _CACHE_DIR / f"wave_{self.E}_{self.N}_{self.K}.json"

    def load_cache(self) -> bool:
        p = self._cache_path()
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
            self.fitted = {
                tuple(json.loads(k)): tuple(v) for k, v in data.items()
            }
            self._profiled = True
            return True
        except Exception:
            return False

    def save_cache(self):
        try:
            data = {json.dumps(list(k)): list(v)
                    for k, v in self.fitted.items()}
            self._cache_path().write_text(json.dumps(data))
        except Exception:
            pass

    def profile(self, kernel_cache: KernelCache,
                w1: torch.Tensor | None = None,
                w1_scale: torch.Tensor | None = None,
                w2: torch.Tensor | None = None,
                w2_scale: torch.Tensor | None = None):
        """Profile all valid configs at 4 batch sizes (~7 min cold).

        If actual model weights (w1, w2) are provided, uses them for
        profiling so the cost model fits real memory traffic patterns.
        Otherwise falls back to random synthetic weights.
        """
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        configs = enumerate_valid_configs()
        # Profile at varied batch sizes AND routing distributions.
        # Using both balanced and skewed routing ensures the fitted
        # (overhead, tpw) parameters reflect real-world conditions
        # where not all experts are equally loaded.
        s_values = [32, 128, 512, 1024]
        E, N, K, topk = self.E, self.N, self.K, self.topk
        dev = "cuda"
        total = len(configs)
        t0 = _time.time()
        ok, fail = 0, 0

        print(f"[RouteMoE] Profiling {total} configs x {len(s_values)} "
              f"S-values (E={E}, N={N}, K={K})...", flush=True)

        if w1 is not None and w2 is not None:
            # Use actual model weights for accurate profiling
            w_up = w1
            w_up_s = w1_scale
            w_dn = w2
            w_dn_s = w2_scale
            print(f"[RouteMoE] Using actual model weights "
                  f"(w1={list(w1.shape)}, w2={list(w2.shape)})",
                  flush=True)
        else:
            # Synthetic weights matching actual shapes
            w_up = interleave_tensor(
                torch.randn(E, N, K, device=dev,
                            dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            )
            w_up_s = torch.ones(
                E, N // 128, K // 128, dtype=torch.float32, device=dev)
            N_half = N // 2
            w_dn = torch.randn(
                E, K, N_half, device=dev, dtype=torch.bfloat16
            ).to(torch.float8_e4m3fn)
            w_dn_s = torch.ones(
                E, K // 128, N_half // 128, dtype=torch.float32,
                device=dev)
        dbg = torch.zeros(max(256, E + 4), device=dev, dtype=torch.float32)

        # Validation pass: test each (config, S) once to filter out
        # configs that crash.  CUDA errors corrupt device state, so we
        # do a quick single-launch test and skip any that fail.
        valid_configs = []
        for cfg in configs:
            bm_v, bn_v, wn_v, stg_v = cfg
            ok = True
            for S_v in s_values:
                try:
                    num_tok_v = S_v * topk
                    x_v = torch.randn(S_v, K, dtype=torch.bfloat16,
                                      device=dev).to(torch.float8_e4m3fn)
                    xs_v = torch.ones(S_v, K // 128, dtype=torch.float32,
                                      device=dev)
                    out_v = torch.zeros(S_v, K, dtype=torch.bfloat16,
                                        device=dev)
                    seq = torch.arange(E, device=dev, dtype=torch.int32
                                       ).repeat(math.ceil(num_tok_v / E)
                                                )[:num_tok_v]
                    tid_v = seq.reshape(S_v, topk)
                    tw_v = torch.ones(num_tok_v, dtype=torch.float32,
                                      device=dev) / topk
                    si_v, ei_v, nt_v = moe_align_block_size(tid_v, bm_v, E)
                    sn_v = nt_v.item()
                    c_v = kernel_cache.get_or_compile(
                        bm_v, bn_v, wn_v, stg_v,
                        x_v, xs_v, w_up, w_up_s, w_dn, w_dn_s, out_v,
                        si_v, ei_v, nt_v, tw_v,
                        topk, S_v, K, N, E, sn_v, 1.0, dbg)
                    c_v(x_v.view(torch.uint8), xs_v,
                        w_up.view(torch.uint8), w_up_s,
                        w_dn.view(torch.uint8), w_dn_s,
                        out_v, si_v, ei_v, nt_v, tw_v,
                        S_v, K, N, E, sn_v, 1.0, dbg)
                    torch.cuda.synchronize()
                except Exception:
                    ok = False
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    break
            if ok:
                valid_configs.append(cfg)
        print(f"[RouteMoE] Validation: {len(valid_configs)}/{len(configs)} "
              f"configs passed", flush=True)
        configs = valid_configs

        # Build profiling schedule: 2 S values × 2 routing patterns = 4 points.
        # Paper showed 2-point profiling achieves 0.98x parity with 4-point.
        # Using balanced + skewed (25% experts active) breaks grid degeneracy
        # between small-bm and large-bm configs.
        s_profile = [s_values[0], s_values[-1]]  # endpoints: smallest + largest
        profile_points = []
        for S in s_profile:
            num_tokens = S * topk
            # Balanced: all experts get equal tokens
            bal_counts = [num_tokens // E] * E
            for i in range(num_tokens % E):
                bal_counts[i] += 1
            profile_points.append((S, bal_counts))
            # Skewed: 25% of experts get all tokens
            active = max(1, E // 4)
            skew_counts = [0] * E
            per = num_tokens // active
            for i in range(active):
                skew_counts[i] = per
            skew_counts[0] += num_tokens - per * active
            profile_points.append((S, skew_counts))

        for ci, cfg in enumerate(configs):
            bm, bn, wn, stg = cfg
            grids, times = [], []
            cfg_t0 = _time.time()

            for S, counts in profile_points:
                try:
                    num_tokens = S * topk
                    grid = compute_grid(counts, bm, N, bn, wn)

                    x = torch.randn(
                        S, K, dtype=torch.bfloat16, device=dev
                    ).to(torch.float8_e4m3fn)
                    x_s = torch.ones(
                        S, K // 128, dtype=torch.float32, device=dev)
                    out = torch.zeros(S, K, dtype=torch.bfloat16, device=dev)
                    # Generate topk_ids matching the target counts
                    ids_list = []
                    for eid, cnt in enumerate(counts):
                        ids_list.extend([eid] * cnt)
                    ids_t = torch.tensor(
                        ids_list[:num_tokens], device=dev, dtype=torch.int32)
                    topk_ids = ids_t.reshape(S, topk)
                    topk_w = torch.ones(
                        S * topk, dtype=torch.float32, device=dev
                    ) / topk
                    sids, eids, ntp = moe_align_block_size(topk_ids, bm, E)
                    sn = ntp.item()

                    compiled = kernel_cache.get_or_compile(
                        bm, bn, wn, stg,
                        x, x_s, w_up, w_up_s, w_dn, w_dn_s, out,
                        sids, eids, ntp, topk_w,
                        topk, S, K, N, E, sn, 1.0, dbg,
                    )

                    def run(out=out, x=x, x_s=x_s, w_up=w_up, w_up_s=w_up_s,
                            w_dn=w_dn, w_dn_s=w_dn_s, sids=sids, eids=eids,
                            ntp=ntp, topk_w=topk_w, compiled=compiled,
                            _S=S, _K=K, _N=N, _E=E, _sn=sn, _dbg=dbg):
                        out.zero_()
                        compiled(
                            x.view(torch.uint8), x_s,
                            w_up.view(torch.uint8), w_up_s,
                            w_dn.view(torch.uint8), w_dn_s,
                            out, sids, eids, ntp, topk_w,
                            _S, _K, _N, _E, _sn, 1.0, _dbg,
                        )

                    for _ in range(3):
                        run()
                    torch.cuda.synchronize()

                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    iters = 10
                    start_ev.record()
                    for _ in range(iters):
                        run()
                    end_ev.record()
                    torch.cuda.synchronize()
                    t_us = start_ev.elapsed_time(end_ev) * 1000.0 / iters
                    grids.append(grid)
                    times.append(t_us)
                except Exception as e:
                    print(f"  FAIL [{ci+1}/{total}] cfg={cfg} S={S}: {e}",
                          flush=True)
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    continue

            torch.cuda.empty_cache()

            if len(grids) >= 3:
                overhead, tpw, cta_cost = fit_wave_model(grids, times)
                self.fitted[cfg] = (overhead, tpw, cta_cost)
                ok += 1
            else:
                fail += 1

            elapsed = _time.time() - t0
            cfg_dt = _time.time() - cfg_t0
            eta = elapsed / (ci + 1) * (total - ci - 1)
            print(f"  [{ci+1}/{total}] ({bm},{bn},{wn},{stg}) "
                  f"{len(grids)}/4 pts  {cfg_dt:.1f}s  "
                  f"[ok={ok} fail={fail}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s]",
                  flush=True)

        self._profiled = True
        self.save_cache()
        total_t = _time.time() - t0
        print(f"[RouteMoE] Profiling done: {ok} fitted, {fail} failed "
              f"in {total_t:.0f}s ({total_t/60:.1f} min)", flush=True)

    def dispatch(
        self, counts: list[int], N: int,
    ) -> tuple[int, int, int, int]:
        """CPU-side dispatch (requires counts as Python list)."""
        best, best_cost = (64, 64, 4, 2), float("inf")
        for cfg, params in self.fitted.items():
            bm, bn, wn, stg = cfg
            grid = compute_grid(counts, bm, N, bn, wn)
            if grid == 0:
                continue
            overhead, tpw = params[0], params[1]
            cta_cost = params[2] if len(params) > 2 else 0.0
            cost = estimate_time(overhead, tpw, grid, cta_cost)
            if cost < best_cost:
                best_cost = cost
                best = cfg
        return best

    def get_unique_bm_values(self) -> list[int]:
        """Return sorted unique bm values across all fitted configs."""
        if not hasattr(self, '_unique_bm'):
            self._unique_bm = sorted(set(c[0] for c in self.fitted.keys()))
        return self._unique_bm

    # ── GPU-side dispatch (zero CPU sync) ────────────────────────────
    _gpu_tables_ready: bool = False

    def _build_gpu_tables(self, device: torch.device):
        """Upload cost model parameters to GPU tensors (once)."""
        if self._gpu_tables_ready:
            return
        configs = list(self.fitted.keys())
        params = [self.fitted[c] for c in configs]
        C = len(configs)
        if C == 0:
            return

        self._gpu_cfgs = configs                       # CPU list for index→cfg
        self._gpu_bm = torch.tensor(
            [c[0] for c in configs], dtype=torch.float32, device=device)
        self._gpu_n_tiles = torch.tensor(
            [math.ceil(self.N / (c[1] * c[2])) for c in configs],
            dtype=torch.float32, device=device)
        self._gpu_overhead = torch.tensor(
            [p[0] for p in params], dtype=torch.float32, device=device)
        self._gpu_tpw = torch.tensor(
            [p[1] for p in params], dtype=torch.float32, device=device)
        self._gpu_cta_cost = torch.tensor(
            [p[2] if len(p) > 2 else 0.0 for p in params],
            dtype=torch.float32, device=device)
        self._gpu_tables_ready = True

    def dispatch_gpu(
        self, counts_gpu: torch.Tensor, N: int,
    ) -> tuple[int, int, int, int]:
        """GPU-side dispatch: cost model eval on GPU, single int sync.

        Instead of transferring E expert counts to CPU and iterating
        83 configs in Python, this computes the full dispatch on GPU
        and transfers only the 1-int winning config index.

        Cost model:
            grid(c) = sum_e(ceil(count_e / bm_c)) * n_tiles_c
            eff_waves(g) = g // 132 + (g % 132) / 132
            est_time(c) = overhead_c + tpw_c * eff_waves(grid_c)
            best = argmin_c(est_time_c)
        """
        self._build_gpu_tables(counts_gpu.device)
        if not self._gpu_tables_ready:
            # Fallback
            return self.dispatch(counts_gpu.tolist(), N)

        # counts_gpu: (E,) int tensor on GPU
        # _gpu_bm: (C,) float tensor on GPU
        # Vectorized grid computation: (E, 1) / (1, C) → (E, C) → sum → (C,)
        counts_f = counts_gpu.float().unsqueeze(1)          # (E, 1)
        bm_f = self._gpu_bm.unsqueeze(0)                    # (1, C)
        m_tiles = torch.ceil(counts_f / bm_f).sum(dim=0)    # (C,)
        grid = m_tiles * self._gpu_n_tiles                   # (C,)

        # eff_waves: piecewise linear
        ew = torch.floor(grid / SM_COUNT) + torch.fmod(grid, SM_COUNT) / SM_COUNT

        # Estimated time per config (3-param model)
        est = (self._gpu_overhead + self._gpu_tpw * ew
               + self._gpu_cta_cost * grid)                   # (C,)

        # Mask out zero-grid configs (no valid tokens)
        est = torch.where(grid > 0, est, torch.tensor(
            float("inf"), device=est.device))

        # Single int transfer: winning config index
        best_idx = est.argmin().item()                       # 1-int sync
        return self._gpu_cfgs[best_idx]

    def dispatch_gpu_async(
        self, counts_gpu: torch.Tensor,
    ) -> torch.Tensor:
        """GPU-side dispatch returning argmin tensor (NO .item() call).

        Returns a scalar GPU tensor with the config index.  The caller
        reads it later via .item() after the stream completes, so the
        sync is free (stream already finished by then).
        """
        self._build_gpu_tables(counts_gpu.device)

        counts_f = counts_gpu.float().unsqueeze(1)           # (E, 1)
        bm_f = self._gpu_bm.unsqueeze(0)                     # (1, C)
        m_tiles = torch.ceil(counts_f / bm_f).sum(dim=0)     # (C,)
        grid = m_tiles * self._gpu_n_tiles                    # (C,)

        ew = torch.floor(grid / SM_COUNT) + torch.fmod(grid, SM_COUNT) / SM_COUNT
        est = (self._gpu_overhead + self._gpu_tpw * ew
               + self._gpu_cta_cost * grid)                    # (C,)
        est = torch.where(grid > 0, est, torch.tensor(
            float("inf"), device=est.device))

        return est.argmin()  # scalar GPU tensor — no sync


# ── Module-level singletons ──────────────────────────────────────────────

_kernel_cache = KernelCache()
_cost_models: dict[tuple, WaveCostModel] = {}
_init_lock = threading.Lock()
_precompile_done: set = set()


def get_cost_model(
    E: int, N: int, K: int, topk: int,
    w1: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> WaveCostModel:
    key = (E, N, K, topk)
    if key in _cost_models and _cost_models[key]._profiled:
        return _cost_models[key]
    with _init_lock:
        if key in _cost_models and _cost_models[key]._profiled:
            return _cost_models[key]
        cm = WaveCostModel(E, N, K, topk)
        if not cm.load_cache():
            print(f"[RouteMoE] Profiling cost model for E={E} N={N} K={K} "
                  f"(~7 min, cached for future runs)...")
            cm.profile(_kernel_cache, w1, w1_scale, w2, w2_scale)
            print(f"[RouteMoE] Done -- {len(cm.fitted)} configs fitted.")
        else:
            print(f"[RouteMoE] Loaded cached cost model "
                  f"({len(cm.fitted)} configs).")
        _cost_models[key] = cm
        return cm


def get_kernel_cache() -> KernelCache:
    return _kernel_cache


def precompile_top_configs(
    cm: WaveCostModel,
    kernel_cache: KernelCache,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    max_configs: int = 10,
):
    """Pre-compile the configs most likely to be dispatched."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    E, N, K, topk = cm.E, cm.N, cm.K, cm.topk
    dev = w1.device

    selected = set()
    for S in [1, 4, 16, 32, 64, 128, 256, 512, 1024]:
        for bal in [0.3, 0.5, 0.8, 1.0]:
            num_tokens = S * topk
            if bal >= 0.9:
                counts = [num_tokens // E] * E
                for i in range(num_tokens % E):
                    counts[i] += 1
            else:
                counts = [0] * E
                active = max(1, int(E * bal))
                per = num_tokens // active
                for i in range(active):
                    counts[i] = per
                counts[0] += num_tokens - per * active
            cfg = cm.dispatch(counts, N)
            selected.add(cfg)

    selected = list(selected)[:max_configs]
    t0 = _time.time()
    print(f"[RouteMoE] Pre-compiling {len(selected)} kernel configs...",
          flush=True)

    for i, (bm, bn, wn, stg) in enumerate(selected):
        try:
            S_ex = 32
            x = torch.randn(
                S_ex, K, dtype=torch.bfloat16, device=dev
            ).to(torch.float8_e4m3fn)
            x_s = torch.ones(
                S_ex, K // 128, dtype=torch.float32, device=dev)
            out = torch.zeros(S_ex, K, dtype=torch.bfloat16, device=dev)
            topk_ids = torch.randint(
                0, E, (S_ex, topk), device=dev, dtype=torch.int32)
            topk_w = torch.ones(
                S_ex * topk, dtype=torch.float32, device=dev)
            sids, eids, ntp = moe_align_block_size(topk_ids, bm, E)
            sn = ntp.item()
            dbg = torch.zeros(
                max(256, E + 4), dtype=torch.float32, device=dev)

            kernel_cache.get_or_compile(
                bm, bn, wn, stg,
                x, x_s, w1, w1_scale, w2, w2_scale, out,
                sids, eids, ntp, topk_w,
                topk, S_ex, K, N, E, sn, 1.0, dbg,
            )
            print(f"  [{i+1}/{len(selected)}] ({bm},{bn},{wn},{stg}) "
                  f"compiled", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(selected)}] ({bm},{bn},{wn},{stg}) "
                  f"FAIL: {e}", flush=True)

    dt = _time.time() - t0
    print(f"[RouteMoE] Pre-compilation done in {dt:.1f}s "
          f"({len(kernel_cache._cache)} cached)", flush=True)
