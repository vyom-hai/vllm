# SPDX-License-Identifier: Apache-2.0
"""
RouteMoE expert backend: routing-aware CuTe DSL fused 3-phase MoE kernel.

Zero-sync dispatch architecture:
  1. M-cached GPU dispatch: cost model runs on GPU, result cached by batch
     size M. First layer per step does one .item() sync; all other layers
     (and repeat M values) get a free cache hit.
  2. Sentinel-filled overlaunch: sorted_ids/expert_ids pre-filled with safe
     sentinel values before moe_align_block_size. Grid uses the allocation
     size (host-known, no sync) instead of ntp.item(). Extra CTAs read
     sentinel expert_id=0 and sorted_id=M*topk, hitting the kernel's
     existing padding check with zero output contribution.
"""

import math

import torch

from vllm import _custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

logger = init_logger(__name__)

# M-cached dispatch: shared across all RouteMoE layers in the same process.
# Key: M (batch size) -> (bm, bn, wn, stg)
# Within a forward step all 16 MoE layers see the same M; first layer pays
# one .item() sync, remaining layers get zero-sync cache hits.
_dispatch_cache: dict[int, tuple[int, int, int, int]] = {}


class RouteMoEExperts(mk.FusedMoEExpertsModular):
    """CuTe DSL fused 3-phase MoE expert with routing-aware dispatch."""

    _cost_model_initialized: bool = False

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype == torch.float8_e4m3fn, (
            "RouteMoE requires FP8 E4M3 quantization"
        )
        self._initialized = False
        self._cm = None
        self._kc = None
        self._dbg = None
        self._act_scale_k_blocks: int = 0
        self._ones_scale: torch.Tensor | None = None
        self._ones_scale_size: int = 0

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (current_platform.is_cuda()
                and current_platform.has_device_capability((9, 0)))

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticTensorSym, kFp8DynamicTokenSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUSTEP]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (1, max(N, K))
        workspace2 = (1, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _lazy_init(self, w1: torch.Tensor, w2: torch.Tensor,
                   E: int, topk: int):
        if self._initialized:
            return
        from vllm.model_executor.layers.fused_moe.routemoe.dispatch import (
            get_cost_model,
            get_kernel_cache,
            precompile_top_configs,
            _precompile_done,
        )
        _, full_N, K = w1.shape
        N = full_N
        self._N = N
        self._K = K
        self._E = E
        self._topk = topk
        self._act_scale_k_blocks = math.ceil(K / 128)

        self._cm = get_cost_model(E, N, K, topk,
                                   w1, self.w1_scale, w2, self.w2_scale)
        self._kc = get_kernel_cache()
        self._dbg = torch.zeros(
            max(256, E + 4), dtype=torch.float32, device=w1.device,
        )

        model_key = (E, N, K, topk)
        if model_key not in _precompile_done:
            precompile_top_configs(
                self._cm, self._kc, w1, self.w1_scale,
                w2, self.w2_scale,
            )
            _precompile_done.add(model_key)

        self._initialized = True

    def _get_ones_scale(self, M: int, k_blocks: int,
                        device: torch.device) -> torch.Tensor:
        needed = M * k_blocks
        if self._ones_scale is None or self._ones_scale_size < needed:
            alloc = max(needed, 4096)
            self._ones_scale = torch.ones(
                alloc, dtype=torch.float32, device=device)
            self._ones_scale_size = alloc
        return self._ones_scale[:needed].view(M, k_blocks)

    def _moe_align_with_sentinels(
        self,
        topk_ids: torch.Tensor,
        bm: int,
        E: int,
        expert_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """moe_align_block_size with sentinel-filled buffers.

        Pre-fills sorted_ids with M*topk (padding sentinel) and expert_ids
        with 0 (valid expert) so that over-launched CTAs read safe values.
        Returns (sorted_ids, expert_ids, ntp_tensor, sn) where sn is the
        allocation size (no GPU->CPU sync needed).
        """
        num_tokens = topk_ids.numel()
        max_padded = num_tokens + E * (bm - 1)
        if num_tokens < E:
            max_padded = min(num_tokens * bm, max_padded)
        num_blocks = triton.cdiv(max_padded, bm)

        dev = topk_ids.device
        sorted_ids = torch.full(
            (max_padded,), num_tokens, dtype=torch.int32, device=dev)
        expert_ids = torch.zeros(
            (num_blocks,), dtype=torch.int32, device=dev)
        ntp = torch.empty((1,), dtype=torch.int32, device=dev)

        ops.moe_align_block_size(
            topk_ids, E, bm, sorted_ids, expert_ids, ntp,
            expert_map if False else None,
        )

        if expert_map is not None:
            expert_ids = expert_map[expert_ids]

        return sorted_ids, expert_ids, ntp, max_padded

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        E_local, full_N, K = w1.shape
        N = full_N
        topk = topk_ids.shape[1]
        M = hidden_states.shape[0]
        E = global_num_experts if global_num_experts > 0 else E_local

        self._lazy_init(w1, w2, E, topk)

        w1s = self.w1_scale
        w2s = self.w2_scale

        k_blocks = self._act_scale_k_blocks
        if a1q_scale is not None and a1q_scale.shape == (M, k_blocks):
            a1s = a1q_scale
        elif a1q_scale is None:
            a1s = self._get_ones_scale(M, k_blocks, hidden_states.device)
        else:
            a1s = a1q_scale.reshape(
                M if a1q_scale.numel() >= M else 1, -1
            ).expand(M, k_blocks).contiguous()

        # --- Change 1: M-cached GPU dispatch (zero sync on cache hit) ---
        # M is constant across all 16 MoE layers within a single forward
        # step. It only changes between steps when the scheduler adjusts
        # the batch. First layer at a new M pays one dispatch_gpu .item()
        # sync; all other layers (and future steps at the same M) hit cache.
        cached = _dispatch_cache.get(M)
        if cached is not None:
            bm, bn, wn, stg = cached
        else:
            flat_ids = topk_ids.flatten()
            if expert_map is not None:
                flat_ids = expert_map[flat_ids]
            counts_gpu = torch.bincount(flat_ids.clamp(min=0), minlength=E)
            bm, bn, wn, stg = self._cm.dispatch_gpu(counts_gpu, N)
            _dispatch_cache[M] = (bm, bn, wn, stg)

        # --- Change 2: Sentinel-filled overlaunch (zero sync for sn) ---
        sorted_ids, expert_ids, ntp, sn = self._moe_align_with_sentinels(
            topk_ids, bm, E, expert_map)

        output.zero_()
        topk_flat = topk_weights.flatten().contiguous()
        scaling_factor = 1.0

        compiled = self._kc.get_or_compile(
            bm, bn, wn, stg,
            hidden_states, a1s, w1, w1s, w2, w2s, output,
            sorted_ids, expert_ids, ntp, topk_flat,
            topk, M, K, N, E, sn, scaling_factor, self._dbg,
        )

        compiled(
            hidden_states.view(torch.uint8), a1s,
            w1.view(torch.uint8), w1s,
            w2.view(torch.uint8), w2s,
            output, sorted_ids, expert_ids, ntp, topk_flat,
            M, K, N, E, sn, scaling_factor, self._dbg,
        )
