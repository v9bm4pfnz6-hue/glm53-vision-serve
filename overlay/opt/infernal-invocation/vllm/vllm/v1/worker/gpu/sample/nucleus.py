# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nucleus (top-p) truncation of probabilistic draft proposals.

Rejection sampling returns the target distribution for any proposal q, and a
draft token is accepted with probability sum(min(p, q)). The target samples
from its top-p nucleus, so an untruncated draft keeps proposing tail tokens
that the target can never accept. Masking the draft logits below their own
nucleus, before the Gumbel draw and before they are cached for the verifier,
proposes from a distribution closer to the target's.

Every tensor-parallel rank draws the draft independently and must draw the same
token, so the mask is a deterministic function of the logits: the kernels count
logits per value with integer atomics and reduce masses in a fixed order.

A logit's key is the upper 16 bits of its order-preserving fp32 bit pattern,
which is the logit itself for bf16 heads. The nucleus is the smallest set of
whole keys, taken from the top, whose mass reaches top_p.
"""

import torch

from vllm import envs
from vllm.triton_utils import tl, triton

_NUM_KEYS = 65536
_KEY_BLOCK = 256
_NUM_KEY_BLOCKS = _NUM_KEYS // _KEY_BLOCK
_VOCAB_BLOCK = 1024


def draft_nucleus_enabled() -> bool:
    return bool(envs.VLLM_K3_DRAFT_TOP_P) or bool(
        envs.VLLM_K3_DRAFT_TOP_P_REQUEST_OVERRIDE
    )


def draft_top_p(top_p: float, extra_args: dict | None) -> float:
    """Draft nucleus mass of a request; 1.0 leaves the draft untruncated."""
    mode = envs.VLLM_K3_DRAFT_TOP_P
    if not mode:
        value = 1.0
    elif mode == "request":
        value = float(top_p)
    else:
        value = float(mode)
    if envs.VLLM_K3_DRAFT_TOP_P_REQUEST_OVERRIDE and extra_args:
        override = extra_args.get("draft_top_p")
        if isinstance(override, (int, float)) and not isinstance(override, bool):
            value = float(override)
    if not value > 0.0:
        return 1.0
    return min(value, 1.0)


def key_values() -> torch.Tensor:
    """Smallest-magnitude fp32 value of every key, in key order."""
    keys = torch.arange(_NUM_KEYS, dtype=torch.int64)
    bits = torch.where(
        keys >= _NUM_KEYS // 2,
        (keys << 16) ^ 0x80000000,
        (~keys & 0xFFFF) << 16,
    )
    bits = torch.where(bits >= 2**31, bits - 2**32, bits)
    return bits.to(torch.int32).view(torch.float32)


@triton.jit
def _ordered_key(logits):
    bits = logits.to(tl.float32).to(tl.int32, bitcast=True).to(tl.int64)
    ordered = bits ^ ((bits >> 31) | 0x80000000)
    return (ordered >> 16) & 0xFFFF


@triton.jit
def _count_kernel(
    hist_ptr,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    temp_ptr,
    top_p_ptr,
    vocab_size,
    BLOCK: tl.constexpr,
    NUM_KEYS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    if tl.load(top_p_ptr + req_state_idx) >= 1.0:
        return
    if tl.load(temp_ptr + req_state_idx) == 0.0:
        return
    block = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + row * logits_stride + block, mask=mask, other=0.0)
    key = _ordered_key(logits)
    tl.atomic_add(
        hist_ptr + row * NUM_KEYS + key,
        tl.full((BLOCK,), 1, dtype=tl.int32),
        mask=mask,
    )


@triton.jit
def _mass_kernel(
    mass_ptr,
    block_sum_ptr,
    block_max_ptr,
    hist_ptr,
    key_value_ptr,
    idx_mapping_ptr,
    temp_ptr,
    top_p_ptr,
    KEY_BLOCK: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    NUM_KEY_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    if tl.load(top_p_ptr + req_state_idx) >= 1.0:
        return
    temp = tl.load(temp_ptr + req_state_idx)
    if temp == 0.0:
        return
    key_block = tl.program_id(1)
    keys = key_block * KEY_BLOCK + tl.arange(0, KEY_BLOCK)
    counts = tl.load(hist_ptr + row * NUM_KEYS + keys)
    # The histogram is all zero again once every key block has run.
    tl.store(hist_ptr + row * NUM_KEYS + keys, tl.zeros((KEY_BLOCK,), dtype=tl.int32))
    values = tl.load(key_value_ptr + keys)
    present = (counts > 0) & (values > float("-inf")) & (values < float("inf"))
    block_max = tl.max(tl.where(present, values, float("-inf")), axis=0)
    # Masses are relative to the block's own maximum; the floor kernel rescales
    # them with the row maximum, so no separate pass finds that maximum first.
    rel = (values.to(tl.float64) - block_max.to(tl.float64)) / temp.to(tl.float64)
    mass = tl.where(present, counts.to(tl.float64) * tl.exp(rel), 0.0)
    tl.store(mass_ptr + row * NUM_KEYS + keys, mass)
    tl.store(block_sum_ptr + row * NUM_KEY_BLOCKS + key_block, tl.sum(mass, axis=0))
    tl.store(block_max_ptr + row * NUM_KEY_BLOCKS + key_block, block_max)


@triton.jit
def _floor_kernel(
    floor_ptr,
    mass_ptr,
    block_sum_ptr,
    block_max_ptr,
    idx_mapping_ptr,
    temp_ptr,
    top_p_ptr,
    KEY_BLOCK: tl.constexpr,
    NUM_KEYS: tl.constexpr,
    NUM_KEY_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        tl.store(floor_ptr + row, 0)
        return
    top_p = tl.load(top_p_ptr + req_state_idx)
    if top_p >= 1.0:
        tl.store(floor_ptr + row, 0)
        return
    temp = tl.load(temp_ptr + req_state_idx)
    if temp == 0.0:
        tl.store(floor_ptr + row, 0)
        return

    # Key blocks from the highest key down.
    rank = tl.arange(0, NUM_KEY_BLOCKS)
    blocks = (NUM_KEY_BLOCKS - 1) - rank
    sums = tl.load(block_sum_ptr + row * NUM_KEY_BLOCKS + blocks)
    maxes = tl.load(block_max_ptr + row * NUM_KEY_BLOCKS + blocks)
    row_max = tl.max(maxes, axis=0)
    rel = (maxes.to(tl.float64) - row_max.to(tl.float64)) / temp.to(tl.float64)
    scale = tl.where(maxes > float("-inf"), tl.exp(rel), 0.0)
    block_mass = sums * scale
    total = tl.sum(block_mass, axis=0)
    target = top_p.to(tl.float64) * total
    cum = tl.cumsum(block_mass, axis=0)
    last = tl.minimum(tl.sum((cum < target).to(tl.int32), axis=0), NUM_KEY_BLOCKS - 1)
    above = tl.sum(tl.where(rank == last - 1, cum, 0.0), axis=0)
    last_scale = tl.sum(tl.where(rank == last, scale, 0.0), axis=0)
    key_block = (NUM_KEY_BLOCKS - 1) - last

    # Keys of that block from the highest down.
    key_rank = tl.arange(0, KEY_BLOCK)
    keys = key_block * KEY_BLOCK + (KEY_BLOCK - 1) - key_rank
    key_mass = tl.load(mass_ptr + row * NUM_KEYS + keys) * last_scale
    key_cum = above + tl.cumsum(key_mass, axis=0)
    last_key = tl.minimum(
        tl.sum((key_cum < target).to(tl.int32), axis=0), KEY_BLOCK - 1
    )
    floor = key_block * KEY_BLOCK + (KEY_BLOCK - 1) - last_key
    # No finite mass (or a non-finite total): keep every logit.
    floor = tl.where(total > 0.0, floor, 0)
    tl.store(floor_ptr + row, floor)


@triton.jit
def _mask_kernel(
    logits_ptr,
    logits_stride,
    floor_ptr,
    vocab_size,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    floor = tl.load(floor_ptr + row)
    if floor <= 0:
        return
    block = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + row * logits_stride + block, mask=mask, other=0.0)
    drop = mask & (_ordered_key(logits) < floor)
    tl.store(
        logits_ptr + row * logits_stride + block,
        tl.full((BLOCK,), float("-inf"), dtype=tl.float32),
        mask=drop,
    )


class DraftNucleus:
    """Masks draft logits below their top-p nucleus, in place."""

    def __init__(self, max_rows: int, device: torch.device):
        self.max_rows = max_rows
        self.hist = torch.zeros(max_rows, _NUM_KEYS, dtype=torch.int32, device=device)
        self.mass = torch.zeros(max_rows, _NUM_KEYS, dtype=torch.float64, device=device)
        self.block_sum = torch.zeros(
            max_rows, _NUM_KEY_BLOCKS, dtype=torch.float64, device=device
        )
        self.block_max = torch.zeros(
            max_rows, _NUM_KEY_BLOCKS, dtype=torch.float32, device=device
        )
        self.floor = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.key_value = key_values().to(device)

    def apply(
        self,
        logits: torch.Tensor,  # [num_tokens, vocab_size]
        idx_mapping: torch.Tensor,  # [num_tokens], -1 for inert rows
        temperature: torch.Tensor,  # [max_num_reqs]
        top_p: torch.Tensor,  # [max_num_reqs]
    ) -> None:
        num_tokens, vocab_size = logits.shape
        if num_tokens > self.max_rows:
            raise ValueError(
                f"Draft nucleus rows {num_tokens} exceed the capacity {self.max_rows}."
            )
        if logits.stride(1) != 1:
            raise ValueError("Draft nucleus requires unit-stride logit rows.")
        idx_mapping = idx_mapping.contiguous()
        num_blocks = triton.cdiv(vocab_size, _VOCAB_BLOCK)
        _count_kernel[(num_tokens, num_blocks)](
            self.hist,
            logits,
            logits.stride(0),
            idx_mapping,
            temperature,
            top_p,
            vocab_size,
            BLOCK=_VOCAB_BLOCK,
            NUM_KEYS=_NUM_KEYS,
        )
        _mass_kernel[(num_tokens, _NUM_KEY_BLOCKS)](
            self.mass,
            self.block_sum,
            self.block_max,
            self.hist,
            self.key_value,
            idx_mapping,
            temperature,
            top_p,
            KEY_BLOCK=_KEY_BLOCK,
            NUM_KEYS=_NUM_KEYS,
            NUM_KEY_BLOCKS=_NUM_KEY_BLOCKS,
            num_warps=1,
        )
        _floor_kernel[(num_tokens,)](
            self.floor,
            self.mass,
            self.block_sum,
            self.block_max,
            idx_mapping,
            temperature,
            top_p,
            KEY_BLOCK=_KEY_BLOCK,
            NUM_KEYS=_NUM_KEYS,
            NUM_KEY_BLOCKS=_NUM_KEY_BLOCKS,
        )
        _mask_kernel[(num_tokens, num_blocks)](
            logits,
            logits.stride(0),
            self.floor,
            vocab_size,
            BLOCK=_VOCAB_BLOCK,
        )
