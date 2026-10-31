# SPDX-License-Identifier: Apache-2.0
"""Fusion of the rank-one GLP projection finish with original BF16 stages.

Call this after the existing BF16 ``stream = hidden + residual`` and the
existing ``coeff = einsum('...h,kh->...k', stream, directions)``. It preserves
the BF16 materialization after the second einsum, alpha multiply, and both
subtractions. Only alpha=1 is supported by this candidate.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["elements"])
def _glp_finish_rank1(
    stream_ptr,
    coeff_ptr,
    dirs_ptr,
    residual_ptr,
    output_ptr,
    cols: tl.constexpr,
    elements,
    block: tl.constexpr,
):
    flat = tl.program_id(0) * block + tl.arange(0, block)
    mask = flat < elements
    row = flat // cols
    col = flat % cols

    stream = tl.load(stream_ptr + flat, mask=mask, other=0).to(tl.float32)
    coeff = tl.load(coeff_ptr + row, mask=mask, other=0).to(tl.float32)
    direction = tl.load(dirs_ptr + col, mask=mask, other=0).to(tl.float32)
    residual = tl.load(residual_ptr + flat, mask=mask, other=0).to(tl.float32)

    # The rank-one einsum first writes BF16. A scalar alpha of exactly one
    # leaves that BF16 value unchanged. Materialize each later BF16 tensor too.
    projected = (coeff * direction).to(tl.bfloat16).to(tl.float32)
    steered = (stream - projected).to(tl.bfloat16).to(tl.float32)
    result = (steered - residual).to(tl.bfloat16)
    tl.store(output_ptr + flat, result, mask=mask)


def glp_finish(
    stream: torch.Tensor,
    coeff: torch.Tensor,
    directions: torch.Tensor,
    residual: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Finish one GLP rank-one projection, preserving the original stages."""
    if alpha != 1.0:
        raise ValueError("This exact-stage prototype supports alpha=1 only")
    if not all(
        x.is_cuda and x.dtype == torch.bfloat16
        for x in (stream, coeff, directions, residual)
    ):
        raise ValueError("All tensors must be CUDA BF16")
    if stream.ndim != 2 or not stream.is_contiguous() or stream.shape[1] == 0:
        raise ValueError("stream must be contiguous [M,H]")
    rows, cols = stream.shape
    if rows == 0 or residual.shape != stream.shape or not residual.is_contiguous():
        raise ValueError("residual must be contiguous [M,H], M>0")
    if coeff.shape != (rows, 1) or not coeff.is_contiguous():
        raise ValueError("coeff must be contiguous [M,1]")
    if directions.shape != (1, cols) or not directions.is_contiguous():
        raise ValueError("directions must be contiguous [1,H]")
    if len({x.device for x in (stream, coeff, directions, residual)}) != 1:
        raise ValueError("All tensors must be on the same device")

    out = torch.empty_like(stream)
    _glp_finish_rank1[(triton.cdiv(rows * cols, 256),)](
        stream,
        coeff,
        directions,
        residual,
        out,
        cols,
        rows * cols,
        256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
