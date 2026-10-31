# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM FP8 block weights with BF16 activations and FP32 block scales."""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# Research-only switch, read once at import rather than in the serving hot path.
_DENSE_SPLITK_RESEARCH = os.environ.get("VLLM_GLM_FP8_DENSE_SPLITK") == "1"
_DENSE_PREFILL_AUTO_RESEARCH = os.environ.get("VLLM_GLM_FP8_DENSE_PREFILL_AUTO") == "1"
_DENSE_PREFILL_BLOCK_M = int(os.environ.get("VLLM_GLM_FP8_DENSE_PREFILL_BLOCK_M", "16"))
_DENSE_PREFILL_WARPS = int(os.environ.get("VLLM_GLM_FP8_DENSE_PREFILL_WARPS", "4"))
_DENSE_PREFILL_STAGES = int(os.environ.get("VLLM_GLM_FP8_DENSE_PREFILL_STAGES", "2"))
_DENSE_PREFILL_WIDE_SHAPES = frozenset(
    {
        (2624, 6144),
        (2048, 2048),
        (3584, 512),
        (6144, 2048),
        (4096, 2048),
        (2816, 6144),
        (6144, 1408),
        (6144, 256),
    }
)
if _DENSE_PREFILL_BLOCK_M not in (16, 32, 64):
    raise ValueError("Research dense prefill BLOCK_M must be 16, 32, or 64")
if _DENSE_PREFILL_WARPS not in (4, 8) or _DENSE_PREFILL_STAGES not in (2, 3, 4):
    raise ValueError("Research dense prefill warps/stages outside supported sweep")
# Column tile of the ordered split-K reduce. The reduce sums every K128 partial
# of one output element in the original block order whatever the tile, so the
# tile only changes how many CTAs share the latency-bound reduction (128 columns
# per CTA leaves ~21 CTAs for the widest decode projection).
_DENSE_REDUCE_BLOCK_N = int(os.environ.get("VLLM_GLM_FP8_DENSE_REDUCE_BLOCK_N", "32"))
if _DENSE_REDUCE_BLOCK_N not in (16, 32, 64, 128):
    raise ValueError("Dense split-K reduce BLOCK_N must be 16, 32, 64, or 128")


@triton.jit(do_not_specialize=["m_size"])
def _fp8_block_w8a16_gemm(
    x_ptr,
    weight_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    m_size,
    n_size: tl.constexpr,
    out_stride: tl.constexpr,
    local_q: tl.constexpr,
    q_start: tl.constexpr,
    kv_dest_base: tl.constexpr,
    k_size: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    weight_stride_n: tl.constexpr,
    weight_stride_k: tl.constexpr,
    scale_stride_n: tl.constexpr,
    scale_stride_k: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    ks = tl.arange(0, 128)
    result = tl.full((block_m, 128), 0.0, tl.float32)
    for block_k in range(tl.cdiv(k_size, 128)):
        k = block_k * 128 + ks
        x = tl.load(
            x_ptr + rows[:, None] * x_stride_m + k[None, :] * x_stride_k,
            (rows[:, None] < m_size) & (k[None, :] < k_size),
            other=0,
        )
        weight = tl.load(
            weight_ptr + cols[:, None] * weight_stride_n + k[None, :] * weight_stride_k,
            (cols[:, None] < n_size) & (k[None, :] < k_size),
            other=0.0,
        )
        scale = tl.load(
            scale_ptr + (cols // 128) * scale_stride_n + block_k * scale_stride_k,
            cols < n_size,
            other=0,
        )
        # E4M3 values widen exactly to BF16. Each original K128 block is
        # accumulated before its original FP32 N128/K128 scale is applied.
        partial = tl.dot(x, tl.trans(weight).to(tl.bfloat16))
        result += partial * scale[None, :]
    if has_bias:
        bias = tl.load(bias_ptr + cols, cols < n_size, other=0).to(tl.float32)
        result += bias[None, :]
    if local_q == n_size and q_start == 0 and kv_dest_base == 0:
        dest_cols = cols
    else:
        dest_cols = tl.where(cols < local_q, q_start + cols, kv_dest_base + cols)
    tl.store(
        out_ptr + rows[:, None] * out_stride + dest_cols[None, :],
        result,
        (rows[:, None] < m_size) & (cols[None, :] < n_size),
    )


@triton.jit(do_not_specialize=["m_size"])
def _fp8_block_w8a16_partial(
    x_ptr,
    weight_ptr,
    partial_ptr,
    m_size,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    weight_stride_n: tl.constexpr,
    weight_stride_k: tl.constexpr,
    block_m: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    block_k = tl.program_id(2)
    k = block_k * 128 + tl.arange(0, 128)
    x = tl.load(
        x_ptr + rows[:, None] * x_stride_m + k[None, :] * x_stride_k,
        (rows[:, None] < m_size) & (k[None, :] < k_size),
        other=0,
    )
    weight = tl.load(
        weight_ptr + cols[:, None] * weight_stride_n + k[None, :] * weight_stride_k,
        (cols[:, None] < n_size) & (k[None, :] < k_size),
        other=0.0,
    )
    # Match the deployed kernel's one original K128 FP32 dot exactly.
    partial = tl.dot(x, tl.trans(weight).to(tl.bfloat16))
    tl.store(
        partial_ptr
        + block_k * m_size * n_size
        + rows[:, None] * n_size
        + cols[None, :],
        partial,
        (rows[:, None] < m_size) & (cols[None, :] < n_size),
    )


@triton.jit(do_not_specialize=["m_size"])
def _fp8_block_w8a16_ordered_reduce(
    partial_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    m_size,
    n_size: tl.constexpr,
    out_stride: tl.constexpr,
    local_q: tl.constexpr,
    q_start: tl.constexpr,
    kv_dest_base: tl.constexpr,
    k_blocks: tl.constexpr,
    scale_stride_n: tl.constexpr,
    scale_stride_k: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    result = tl.full((block_m, block_n), 0.0, tl.float32)
    for block_k in range(k_blocks):
        partial = tl.load(
            partial_ptr
            + block_k * m_size * n_size
            + rows[:, None] * n_size
            + cols[None, :],
            (rows[:, None] < m_size) & (cols[None, :] < n_size),
            other=0,
        )
        scale = tl.load(
            scale_ptr + (cols // 128) * scale_stride_n + block_k * scale_stride_k,
            cols < n_size,
            other=0,
        )
        # Preserve the original per-block FP32 scale and sequential sum order.
        result += partial * scale[None, :]
    if has_bias:
        bias = tl.load(bias_ptr + cols, cols < n_size, other=0).to(tl.float32)
        result += bias[None, :]
    if local_q == n_size and q_start == 0 and kv_dest_base == 0:
        dest_cols = cols
    else:
        dest_cols = tl.where(cols < local_q, q_start + cols, kv_dest_base + cols)
    tl.store(
        out_ptr + rows[:, None] * out_stride + dest_cols[None, :],
        result,
        (rows[:, None] < m_size) & (cols[None, :] < n_size),
    )


@register_weight_loader_v2_supported_method
class GlmFp8W8A16LinearMethod(Fp8LinearMethod):
    """Keep serialized E4M3 weights and FP32 scales unchanged after loading."""

    supports_glm_scatter_out = True

    def __init__(self, quant_config):
        if (
            not quant_config.is_checkpoint_fp8_serialized
            or quant_config.weight_block_size != [128, 128]
            or quant_config.activation_scheme != "dynamic"
        ):
            raise ValueError("GLM W8A16 requires serialized FP8 128x128 weights.")
        super().__init__(quant_config)
        self.use_marlin = False
        self.use_deep_gemm = False
        logger.info_once(
            "GLM FP8 W8A16 dense selected: E4M3 weights, BF16 activations, "
            "original FP32 128x128 block scales.",
            scope="process",
        )
        logger.info_once(
            "GLM FP8 dense research policy: splitk=%s prefill_auto=%s "
            "manual_block_m=%d warps=%d stages=%d",
            _DENSE_SPLITK_RESEARCH,
            _DENSE_PREFILL_AUTO_RESEARCH,
            _DENSE_PREFILL_BLOCK_M,
            _DENSE_PREFILL_WARPS,
            _DENSE_PREFILL_STAGES,
            scope="process",
        )

    def process_weights_after_loading(self, layer) -> None:
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise TypeError("GLM W8A16 requires E4M3 checkpoint weights.")
        if layer.weight_scale_inv.dtype != torch.float32:
            raise TypeError("GLM W8A16 requires original FP32 checkpoint scales.")
        n_size, k_size = layer.weight.shape
        expected_scale_shape = (triton.cdiv(n_size, 128), triton.cdiv(k_size, 128))
        if tuple(layer.weight_scale_inv.shape) != expected_scale_shape:
            raise ValueError(
                "GLM W8A16 scale shape does not match 128x128 weight blocks: "
                f"{tuple(layer.weight_scale_inv.shape)} != {expected_scale_shape}."
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        scatter: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or x.device.type != "cuda":
            raise TypeError("GLM W8A16 requires CUDA BF16 activations.")
        n_size, k_size = layer.weight.shape
        if x.shape[-1] != k_size:
            raise ValueError("GLM W8A16 input width does not match weight width.")
        flat_x = x.reshape(-1, k_size)
        if out is None:
            out = torch.empty((*x.shape[:-1], n_size), dtype=x.dtype, device=x.device)
            out_stride = n_size
            local_q = n_size
            q_start = 0
            kv_dest_base = 0
        else:
            if scatter is None or out.ndim != 2 or out.stride(1) != 1:
                raise ValueError("GLM scatter output needs a contiguous 2D row buffer")
            out_stride = out.stride(0)
            q_start, local_q, kv_dest_base = scatter
            if not 0 <= local_q <= n_size:
                raise ValueError("GLM scatter q width exceeds the local output")
        if flat_x.shape[0] == 0:
            return out
        prefill = flat_x.shape[0] >= 128
        block_m = _DENSE_PREFILL_BLOCK_M if prefill else 16
        if _DENSE_PREFILL_AUTO_RESEARCH and flat_x.shape[0] in (1024, 2048, 4096):
            if (n_size, k_size) in _DENSE_PREFILL_WIDE_SHAPES and (
                flat_x.shape[0] >= 2048 or (n_size, k_size) != (3584, 512)
            ):
                block_m = 64
            elif (n_size, k_size) in {(512, 6144), (3584, 512)}:
                block_m = 32
        num_warps = _DENSE_PREFILL_WARPS if prefill else 4
        num_stages = _DENSE_PREFILL_STAGES if prefill else 2
        if _DENSE_SPLITK_RESEARCH and flat_x.shape[0] <= 16 and k_size >= 512:
            k_blocks = triton.cdiv(k_size, 128)
            partial = torch.empty(
                (k_blocks, flat_x.shape[0], n_size),
                dtype=torch.float32,
                device=x.device,
            )
            grid = (
                triton.cdiv(flat_x.shape[0], block_m),
                triton.cdiv(n_size, 128),
            )
            _fp8_block_w8a16_partial[(*grid, k_blocks)](
                flat_x,
                layer.weight,
                partial,
                flat_x.shape[0],
                n_size,
                k_size,
                *flat_x.stride(),
                *layer.weight.stride(),
                block_m,
                num_warps=4,
                num_stages=2,
            )
            reduce_grid = (
                triton.cdiv(flat_x.shape[0], block_m),
                triton.cdiv(n_size, _DENSE_REDUCE_BLOCK_N),
            )
            _fp8_block_w8a16_ordered_reduce[reduce_grid](
                partial,
                layer.weight_scale_inv,
                bias if bias is not None else layer.weight_scale_inv,
                out,
                flat_x.shape[0],
                n_size,
                out_stride,
                local_q,
                q_start,
                kv_dest_base,
                k_blocks,
                *layer.weight_scale_inv.stride(),
                bias is not None,
                block_m,
                _DENSE_REDUCE_BLOCK_N,
                num_warps=4,
                num_stages=2,
            )
            return out
        _fp8_block_w8a16_gemm[
            (triton.cdiv(flat_x.shape[0], block_m), triton.cdiv(n_size, 128))
        ](
            flat_x,
            layer.weight,
            layer.weight_scale_inv,
            bias if bias is not None else layer.weight_scale_inv,
            out,
            flat_x.shape[0],
            n_size,
            out_stride,
            local_q,
            q_start,
            kv_dest_base,
            k_size,
            *flat_x.stride(),
            *layer.weight.stride(),
            *layer.weight_scale_inv.stride(),
            bias is not None,
            block_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out
