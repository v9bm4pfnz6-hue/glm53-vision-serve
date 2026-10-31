# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 routed GEMM over unchanged E4M3 bytes and FP32 block scales."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["VALID"])
def _routed_block_gemm(
    a_ptr,
    b_ptr,
    scale_ptr,
    c_ptr,
    route_ptr,
    sorted_ptr,
    expert_ptr,
    padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    VALID,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    TOP_K: tl.constexpr,
    MUL_ROUTE: tl.constexpr,
    NAIVE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    if pm * BLOCK_M >= tl.load(padded_ptr):
        return
    rows = tl.arange(0, BLOCK_M)
    if NAIVE:
        tokens = tl.where(rows == 0, pm, VALID).to(tl.int64)
    else:
        tokens = tl.load(sorted_ptr + pm * BLOCK_M + rows).to(tl.int64)
    valid_rows = tokens < VALID
    expert = tl.load(expert_ptr + pm).to(tl.int64)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = c_ptr + tokens[:, None] * CM + cols[None, :] * CN
    output_mask = valid_rows[:, None] & (cols[None, :] < N)
    if expert < 0:
        tl.store(output_ptrs, tl.full((BLOCK_M, BLOCK_N), 0, tl.float32), output_mask)
        return

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(tl.cdiv(K, 128)):
        ks = kb * 128 + tl.arange(0, 128)
        a = tl.load(
            a_ptr + (tokens[:, None] // TOP_K) * AM + ks[None, :] * AK,
            valid_rows[:, None] & (ks[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + expert * BE + ks[:, None] * BK + cols[None, :] * BN,
            (ks[:, None] < K) & (cols[None, :] < N),
            other=0.0,
        ).to(tl.bfloat16)
        # Every finite E4M3 value is exact in BF16. Apply the original FP32
        # scale after each original 128-wide K block, before FP32 summation.
        partial = tl.dot(a, b, out_dtype=tl.float32)
        scale = tl.load(
            scale_ptr + expert * SE + (cols // 128) * SN + kb * SK,
            cols < N,
            other=0,
        )
        acc = acc + partial * scale[None, :]
    if MUL_ROUTE:
        route = tl.load(route_ptr + tokens, valid_rows, other=0)
        acc = acc * route[:, None]
    tl.store(output_ptrs, acc, output_mask)


def routed_fp8_w8a16(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    route_weights: torch.Tensor | None,
    sorted_tokens: torch.Tensor | None,
    expert_ids: torch.Tensor,
    padded_tokens: torch.Tensor,
    multiply_route: bool,
    top_k: int,
    block_m: int,
) -> None:
    if a.dtype != torch.bfloat16 or b.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "GLM W8A16 requires BF16 activations and original E4M3 weights"
        )
    if scales.dtype != torch.float32 or output.dtype != torch.bfloat16:
        raise ValueError("GLM W8A16 requires FP32 scales and BF16 output")
    if a.ndim != 2 or b.ndim != 3 or output.ndim != 3:
        raise ValueError("Unexpected routed GEMM tensor rank")
    if a.shape[1] != b.shape[2] or b.shape[2] % 128:
        raise ValueError("GLM W8A16 requires original 128-wide K blocks")
    if scales.shape != (b.shape[0], triton.cdiv(b.shape[1], 128), b.shape[2] // 128):
        raise ValueError("FP32 scale tensor does not match the original weight blocks")
    if multiply_route and route_weights is None:
        raise ValueError("Routed output requires route weights")
    valid = a.shape[0] * top_k
    if sorted_tokens is None:
        block_count = valid
    else:
        em = sorted_tokens.numel()
        if a.shape[0] < block_m:
            em = min(em, valid * block_m)
        block_count = triton.cdiv(em, block_m)
    _routed_block_gemm[(block_count, triton.cdiv(b.shape[1], 64))](
        a,
        b,
        scales,
        output,
        route_weights,
        sorted_tokens,
        expert_ids,
        padded_tokens,
        b.shape[1],
        b.shape[2],
        valid,
        *a.stride(),
        *b.stride(),
        *scales.stride(),
        output.stride(1),
        output.stride(2),
        TOP_K=top_k,
        MUL_ROUTE=multiply_route,
        NAIVE=sorted_tokens is None,
        BLOCK_M=block_m,
        BLOCK_N=64,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
