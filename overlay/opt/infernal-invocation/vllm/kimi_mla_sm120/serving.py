# SPDX-License-Identifier: Apache-2.0
"""Rebind live serving tensors to preallocated SM120 MLA storage."""

import math

import torch
import triton

from .kernel import _attention, _merge
from .plan import choose_splits


def _geometry(rows, heads, batch, query_length, capacity, sms):
    block_rows = 32 if heads >= 64 and query_length >= 4 else 16
    items = batch * triton.cdiv(query_length * heads, block_rows)
    splits = choose_splits(items, capacity, sms)
    return block_rows, splits, rows * heads * splits * 514 * 4


def workspace_bytes(max_rows, heads, capacity, sms):
    """Cover every live one-row or four-row request shape within capacity."""
    return max(
        _geometry(rows, heads, rows // query_length, query_length, capacity, sms)[2]
        for rows in range(1, max_rows + 1)
        for query_length in (1, 4)
        if rows % query_length == 0
    )


class ServingLauncher:
    """Retain compile identities, never retain per-layer input/cache buffers."""

    def __init__(self, device):
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.compiled = set()

    def run(self, *, query, cache, table, lengths, cu_query, query_lengths,
            output, lse, scratch, capacity, scale, value_scale):
        rows, heads, dim = query.shape
        batch = lengths.numel()
        if dim != 576 or rows % batch:
            raise ValueError("SM120 MLA requires uniform query lengths and 576-wide queries")
        query_length = rows // batch
        if query_length not in (1, 4):
            raise ValueError("The serving adapter supports flattened or four-query decode")
        if query.dtype != torch.float8_e4m3fn:
            raise ValueError("SM120 serving queries must be native E4M3")
        if cache.dtype == torch.uint8:
            cache = cache.view(torch.float8_e4m3fn)
        if cache.dtype != torch.float8_e4m3fn or cache.shape[-1] != 576:
            raise ValueError("SM120 serving cache must use 576-byte native E4M3 records")
        if not (query.is_contiguous() and cache.is_contiguous() and output.is_contiguous()):
            raise ValueError("SM120 serving query/cache/output must be contiguous")
        if table.dtype != torch.int32 or table.stride(1) != 1:
            raise ValueError("SM120 page tables must have contiguous int32 columns")
        block_rows, splits, needed = _geometry(
            rows, heads, batch, query_length, capacity, self.sms
        )
        if scratch.dtype != torch.uint8 or scratch.numel() < needed:
            raise ValueError(f"SM120 scratch needs {needed} bytes")
        flat = scratch[:needed].view(torch.float32)
        slots = rows * heads * splits
        partial_o = flat[:slots * 512]
        partial_m = flat[slots * 512:slots * 513]
        partial_l = flat[slots * 513:slots * 514]
        page_size = int(cache.shape[1])
        key = (rows, heads, batch, query_length, cache.shape[0], page_size,
               table.shape[1], table.stride(0), splits, block_rows,
               query_lengths is not None, float(scale), float(value_scale))
        if torch.cuda.is_current_stream_capturing() and key not in self.compiled:
            raise RuntimeError("SM120 MLA layout was not exercised by eager graph warmup")
        _attention[(triton.cdiv(query_length * heads, block_rows), splits, batch)](
            query, cache, table, lengths, cu_query,
            lengths if query_lengths is None else query_lengths,
            output, lse, partial_o, partial_m, partial_l,
            float(scale) * math.log2(math.e), float(value_scale), heads, rows,
            query_length, int(cache.shape[0]), page_size, int(table.shape[1]),
            int(table.stride(0)), splits, True, query_lengths is not None,
            block_rows, 64, num_warps=8, num_stages=2, enable_fp_fusion=False,
        )
        if splits > 1:
            _merge[(query_length * heads, batch)](
                partial_o, partial_m, partial_l, output, lse, cu_query,
                heads, rows, query_length, splits, triton.next_power_of_2(splits),
                True, float(value_scale), 128, num_warps=4, enable_fp_fusion=False,
            )
        self.compiled.add(key)
        return output, lse
