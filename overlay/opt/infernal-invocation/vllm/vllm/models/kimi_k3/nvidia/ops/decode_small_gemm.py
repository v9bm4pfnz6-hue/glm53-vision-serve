# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-batch BF16 GEMMs of the Kimi-K3 MoE front and latent tail without
cuBLAS split-K.

At decode row counts (one to sixteen rows) three unquantized BF16 projections
run once per MoE layer: the TP-sharded router gate (K 7168, N 104 per rank at
TP9, fp32 logits), the routed latent projection ``routed_expert_down_proj``
(K 7168, N 400 per rank) and the latent-tail up-projection on this rank's
packed input shard (K 400, N 7168). cuBLAS serves each as a split-K kernel
followed by a separate ``splitKreduce`` launch; on the served rank-0 decode
trace the three GEMMs and their reduces take about 24 us per layer, 2.2 ms per
step, almost all of it launch and latency rather than the 1.5-6 MB of weight
bytes each one streams.

This module replaces them with single launches that keep the full K extent
inside one CTA (no split-K, no reduce kernel):

* the CuTe DSL skinny GEMM of ``vllm.model_executor.kernels.linear.cute_dsl``
  (PDL-armed, one thread block per ``outputs_per_block`` columns, per-thread
  fp32 accumulation over the thread's K lane, warp and block tree reduction)
  for the two K 7168 shapes, with fp32 output for the gate;
* the fixed-order Triton kernel of :mod:`invariant_gemm` with decode tiles for
  the K 400 up-projection, whose K is not a multiple of the CuTe K tile.

Numerics: bf16 operands, fp32 accumulation and one rounding into the output,
the same class as cuBLAS but a different summation order, so results are not
bit-identical to the served path. Deployment follows the precision rule: the
window proof measures both against a float64 reference on real weights before
the switch is accepted.

Opt-in: ``VLLM_K3_DECODE_SMALL_GEMM=1``. ``VLLM_K3_DECODE_SMALL_GEMM_MAX_ROWS``
(default 16) bounds the row count; larger calls keep cuBLAS.
``VLLM_K3_DECODE_SMALL_GEMM_CONFIGS`` overrides the per-shape CuTe
configuration for measurement: ``N,K:block,outputs,unroll[,vector];...``.
Row counts are compiled for the decode capture sizes
(``VLLM_K3_DECODE_SMALL_GEMM_ROWS``, default ``1,2,4,8,16``) before graph
capture; other row counts fall back to cuBLAS.
"""

from __future__ import annotations

import functools
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.triton_utils import triton

from .invariant_gemm import _fixed_order_gemm_kernel

logger = init_logger(__name__)

# Decode tiles of the fixed-order Triton kernel for the short-K up-projection:
# one 16-row tile (rows past M are masked), 64 output columns per program.
_TRITON_TILES = (16, 64, 64, 4, 2)  # BLOCK_M, BLOCK_N, BLOCK_K, warps, stages


@functools.lru_cache(maxsize=1)
def settings() -> tuple[bool, int, tuple[int, ...]]:
    enabled = os.getenv("VLLM_K3_DECODE_SMALL_GEMM", "0") == "1"
    max_rows = max(
        1, min(16, int(os.getenv("VLLM_K3_DECODE_SMALL_GEMM_MAX_ROWS", "16")))
    )
    rows = tuple(
        sorted(
            {
                int(r)
                for r in os.getenv(
                    "VLLM_K3_DECODE_SMALL_GEMM_ROWS", "1,2,4,8,16"
                ).split(",")
                if r.strip()
            }
        )
    )
    return enabled, max_rows, rows


def _default_cute_params(n: int, k: int) -> tuple[int, int, int, int] | None:
    """``(block_size, outputs_per_block, k_unroll, vector_width)`` for a
    K 7168-class shape, or None when the shape is not a CuTe candidate."""
    if k % 1024 != 0 or n > 1024:
        # Only the narrow MoE-front projections are measured; wide outputs
        # keep cuBLAS until a window proof covers them.
        return None
    if n <= 128 and n % 2 == 0:
        return (128, 2, 1, 8)
    if n % 4 == 0:
        return (128, 4, 1, 8)
    if n % 2 == 0:
        return (128, 2, 1, 8)
    return None


@functools.lru_cache(maxsize=1)
def _cute_overrides() -> dict[tuple[int, int], tuple[int, int, int, int] | str]:
    raw = os.getenv("VLLM_K3_DECODE_SMALL_GEMM_CONFIGS", "").strip()
    out: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    if not raw:
        return out
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        shape, params = item.split(":")
        n, k = (int(v) for v in shape.split(","))
        if params.strip() == "triton":
            out[(n, k)] = "triton"  # type: ignore[assignment]
            continue
        values = [int(v) for v in params.split(",")]
        if len(values) == 3:
            values.append(8)
        if len(values) != 4:
            raise ValueError(
                "VLLM_K3_DECODE_SMALL_GEMM_CONFIGS entries are "
                "N,K:block,outputs,unroll[,vector]"
            )
        out[(n, k)] = tuple(values)  # type: ignore[assignment]
    return out


@functools.cache
def _plan(n: int, k: int) -> tuple[str, tuple[int, int, int, int] | None] | None:
    """``("cute", params)`` or ``("triton", None)`` for a local weight shape.

    A non-empty ``VLLM_K3_DECODE_SMALL_GEMM_CONFIGS`` is also the allowlist:
    shapes it does not name keep cuBLAS (``N,K:triton`` selects the Triton
    kernel for a listed shape)."""
    overrides = _cute_overrides()
    if overrides:
        entry = overrides.get((n, k))
        if entry is None:
            return None
        if entry == "triton":
            return ("triton", None) if k <= 1024 else None
        params = entry
    else:
        params = _default_cute_params(n, k)
    if params is not None:
        block, outputs, _unroll, vector = params
        if k % (block * vector) == 0 and n % outputs == 0:
            return ("cute", params)
    if k <= 1024:
        return ("triton", None)
    return None


def _config(rows: int, params: tuple[int, int, int, int]) -> SkinnyGemmConfig:
    block, outputs, unroll, vector = params
    return SkinnyGemmConfig(rows, block, outputs, unroll, vector)


def applies_to(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype | None = None
) -> bool:
    """Whether ``x @ weight.T`` (``weight`` is ``[N, K]``) takes this path."""
    enabled, max_rows, rows = settings()
    if not enabled:
        return False
    if (
        x.ndim != 2
        or weight.ndim != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_cuda
        or not weight.is_cuda
        or x.shape[1] != weight.shape[1]
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        return False
    m = x.shape[0]
    if m < 1 or m > max_rows:
        return False
    plan = _plan(weight.shape[0], weight.shape[1])
    if plan is None:
        return False
    if plan[0] == "cute":
        if m not in rows or not shape_dynamic_skinny_gemm.is_available():
            return False
        # A kernel that is not compiled yet must not be compiled under graph
        # capture; such a call keeps cuBLAS (the capture then holds cuBLAS for
        # that row count, which is why the warmup provider compiles the decode
        # row counts first).
        if torch.cuda.is_current_stream_capturing():
            return _compiled(m, plan[1], weight.dtype, out_dtype)  # type: ignore[arg-type]
        return True
    return True


def _compiled(
    rows: int,
    params: tuple[int, int, int, int],
    dtype: torch.dtype,
    out_dtype: torch.dtype | None,
) -> bool:
    key = shape_dynamic_skinny_gemm._key(dtype, _config(rows, params), False, out_dtype)
    return key in shape_dynamic_skinny_gemm._compiled


def mm(
    x: torch.Tensor, weight: torch.Tensor, *, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """``x @ weight.T`` for a call that passed :func:`applies_to`."""
    n, k = weight.shape
    kind, params = _plan(n, k)  # type: ignore[misc]
    if kind == "cute":
        assert params is not None
        return shape_dynamic_skinny_gemm(
            x, weight, _config(x.shape[0], params), None, out_dtype=out_dtype
        )
    out = torch.empty((x.shape[0], n), dtype=out_dtype or x.dtype, device=x.device)
    _launch_triton(x, weight.t(), out)
    return out


def _launch_triton(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
    block_m, block_n, block_k, num_warps, num_stages = _TRITON_TILES
    m, k = a.shape
    n = b.shape[1]
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fixed_order_gemm_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        ACCUMULATE=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def request_warmup(shapes: list[tuple[int, int, torch.dtype | None]]) -> None:
    """Ask the CuTe warmup provider to compile the decode row counts of each
    ``(n, k, out_dtype)`` before graph capture; shapes on the Triton path need
    no registration (Triton compiles at first eager call during warmup)."""
    enabled, _max_rows, rows = settings()
    if not enabled or not shape_dynamic_skinny_gemm.is_available():
        return
    for n, k, out_dtype in shapes:
        plan = _plan(n, k)
        if plan is None or plan[0] != "cute":
            continue
        configs = [_config(r, plan[1]) for r in rows]  # type: ignore[arg-type]
        shape_dynamic_skinny_gemm.request_warmup_configs(
            torch.bfloat16, configs, out_dtype=out_dtype
        )
        logger.info_once(
            "Kimi-K3 decode small GEMM: CuTe skinny GEMM for N=%d K=%d "
            "(block=%d outputs=%d unroll=%d vector=%d, out=%s) at rows %s",
            n,
            k,
            *plan[1],  # type: ignore[misc]
            out_dtype or torch.bfloat16,
            rows,
        )
