# SPDX-License-Identifier: Apache-2.0
"""Capture-static native SM120 Kimi MLA plan and caller-owned buffers."""

import math

import torch
import triton

from .kernel import _attention, _merge


def _span(tensor):
    begin = tensor.data_ptr()
    if not tensor.numel():
        return begin, begin
    extent = 1 + sum(
        (n - 1) * s for n, s in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return begin, begin + extent * tensor.element_size()


def choose_splits(items: int, max_seq_len: int, sm_count: int) -> int:
    """One CTA wave, with at least two 64-token chunks per active split."""
    target = max(1, sm_count // max(items, 1))
    by_length = max(1, triton.cdiv(max_seq_len, 64) // 2)
    return min(target, by_length, 64)


class KimiMlaSm120Plan:
    """Prepare once; ``run`` reads live tensor contents without allocating.

    KV records are 512 latent plus 64 RoPE E4M3 values; values alias the
    latent prefix. Supports uniform 4D queries or 3D queries with device
    cumulative query lengths. ``query_cache_lengths`` optionally supplies
    the visible *local* KV length of each flattened query for DCP verification.
    Cache-page IDs and lengths must describe the declared capacity; padding
    uses negative page IDs. Returned LSE uses the natural logarithm.
    """

    def __init__(
        self,
        *,
        query,
        kv_cache,
        block_tables,
        seq_lens,
        out,
        bmm1_scale,
        bmm2_scale=1.0,
        workspace_buffer=None,
        lse=None,
        cum_seq_lens_q=None,
        max_q_len=None,
        max_seq_len=None,
        query_cache_lengths=None,
        num_split=None,
        block_rows=None,
    ):
        if query.device.type != "cuda":
            raise ValueError("SM120 Kimi MLA requires CUDA tensors")
        if torch.cuda.get_device_capability(query.device) != (12, 0):
            raise ValueError("This kernel is qualified only for SM120")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare Kimi MLA before CUDA graph capture")
        if query.dtype != torch.float8_e4m3fn or kv_cache.dtype != torch.float8_e4m3fn:
            raise ValueError("query and KV must be native E4M3")
        if (
            query.ndim not in (3, 4)
            or query.shape[-1] != 576
            or not query.is_contiguous()
        ):
            raise ValueError("query must be contiguous [Q,H,576] or [B,Q,H,576]")
        if kv_cache.ndim == 4 and kv_cache.shape[1] == 1:
            kv_cache = kv_cache[:, 0]
        if (
            kv_cache.ndim != 3
            or kv_cache.shape[-1] != 576
            or not kv_cache.is_contiguous()
        ):
            raise ValueError("KV must be contiguous [pages,page_size,576]")
        self.page_size = int(kv_cache.shape[1])
        if self.page_size < 64 or self.page_size % 64:
            raise ValueError("page size must be a positive multiple of 64")
        if (
            seq_lens.ndim != 1
            or seq_lens.dtype != torch.int32
            or not seq_lens.is_contiguous()
        ):
            raise ValueError("seq_lens must be contiguous int32 [B]")
        self.batch = int(seq_lens.numel())
        self.heads = int(query.shape[-2])
        if self.batch < 1 or self.heads < 1:
            raise ValueError("batch and head counts must be positive")
        if (
            block_tables.ndim != 2
            or block_tables.shape[0] != self.batch
            or block_tables.dtype != torch.int32
            or block_tables.stride(1) != 1
        ):
            raise ValueError(
                "block_tables must be int32 [B,width] with unit column stride"
            )
        if (
            out.shape != query.shape[:-1] + (512,)
            or out.dtype != torch.bfloat16
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous BF16 query.shape[:-1] + (512,)")
        self.ragged = query.ndim == 3
        if self.ragged:
            if (
                cum_seq_lens_q is None
                or cum_seq_lens_q.shape != (self.batch + 1,)
                or cum_seq_lens_q.dtype != torch.int32
                or not cum_seq_lens_q.is_contiguous()
                or max_q_len is None
                or max_q_len < 1
            ):
                raise ValueError(
                    "3D queries require int32 cu_seqlens [B+1] and max_q_len"
                )
            self.max_q = int(max_q_len)
            self.total_q = int(query.shape[0])
        else:
            if query.shape[0] != self.batch or query.shape[1] < 1:
                raise ValueError("uniform query batch/length does not match seq_lens")
            self.max_q = int(query.shape[1])
            self.total_q = self.batch * self.max_q
            if cum_seq_lens_q is not None:
                raise ValueError("cu_seqlens is only used with 3D queries")
        if query_cache_lengths is not None and (
            query_cache_lengths.shape != (self.total_q,)
            or query_cache_lengths.dtype != torch.int32
            or not query_cache_lengths.is_contiguous()
        ):
            raise ValueError("query_cache_lengths must be contiguous int32 [total_q]")
        if block_rows is None:
            # Wide-head verification benefits from reusing each KV tile for
            # 32 packed rows. Small-head draft plans retain the smaller tile;
            # neither setting automatically replaces another serving backend.
            block_rows = 32 if self.heads >= 64 and self.max_q >= 4 else 16
        if block_rows not in (16, 32):
            raise ValueError("block_rows must be 16 or 32")
        self.block_rows = int(block_rows)
        max_capacity = int(block_tables.shape[1]) * self.page_size
        max_seq_len = max_capacity if max_seq_len is None else int(max_seq_len)
        if max_seq_len < 0 or max_seq_len > max_capacity:
            raise ValueError("max_seq_len exceeds the page-table capacity")
        items = self.batch * triton.cdiv(self.max_q * self.heads, self.block_rows)
        sm_count = torch.cuda.get_device_properties(query.device).multi_processor_count
        self.num_split = (
            choose_splits(items, max_seq_len, sm_count)
            if num_split is None
            else int(num_split)
        )
        if not 1 <= self.num_split <= 64:
            raise ValueError("num_split must be between 1 and 64")
        if not math.isfinite(bmm1_scale) or not math.isfinite(bmm2_scale):
            raise ValueError("attention scales must be finite")
        self.scale_log2 = float(bmm1_scale) * math.log2(math.e)
        self.value_scale = float(bmm2_scale)
        self.query = query
        self.kv_cache = kv_cache
        self.block_tables = block_tables
        self.seq_lens = seq_lens
        self.cu = seq_lens if cum_seq_lens_q is None else cum_seq_lens_q
        self.query_lengths = (
            seq_lens if query_cache_lengths is None else query_cache_lengths
        )
        self.per_query_length = query_cache_lengths is not None
        self.out = out
        self.lse = (
            torch.empty(query.shape[:-1], device=query.device, dtype=torch.float32)
            if lse is None
            else lse
        )
        if (
            self.lse.shape != query.shape[:-1]
            or self.lse.dtype != torch.float32
            or not self.lse.is_contiguous()
        ):
            raise ValueError("lse must be contiguous float32 query.shape[:-1]")
        slots = self.total_q * self.heads * self.num_split
        self.workspace_bytes = slots * (512 + 2) * 4
        if workspace_buffer is None:
            workspace_buffer = torch.empty(
                self.workspace_bytes, dtype=torch.uint8, device=query.device
            )
        if (
            workspace_buffer.dtype != torch.uint8
            or not workspace_buffer.is_contiguous()
            or workspace_buffer.numel() < self.workspace_bytes
            or workspace_buffer.data_ptr() % 4
        ):
            raise ValueError(
                "workspace must be aligned contiguous uint8 with sufficient capacity"
            )
        self.workspace = workspace_buffer
        flat = workspace_buffer.view(-1)[: self.workspace_bytes].view(torch.float32)
        self.partial_o = flat[: slots * 512]
        self.partial_m = flat[slots * 512 : slots * 513]
        self.partial_l = flat[slots * 513 : slots * 514]
        tensors = (
            kv_cache,
            block_tables,
            seq_lens,
            self.cu,
            self.query_lengths,
            out,
            self.lse,
            workspace_buffer,
        )
        if any(t.device != query.device for t in tensors):
            raise ValueError("all tensors must be on the query device")
        readers = (query, kv_cache, block_tables, seq_lens, self.cu, self.query_lengths)
        writers = (out, self.lse, workspace_buffer)
        for index, writer in enumerate(writers):
            begin, end = _span(writer)
            for other in readers + writers[:index]:
                other_begin, other_end = _span(other)
                if begin < other_end and other_begin < end:
                    raise ValueError(
                        "output, LSE and workspace must not overlap inputs or each other"
                    )
        # Warm both launchers with the real layouts. Capture/replay subsequently
        # changes only contents of those retained tensors, never compile keys.
        self.run()

    def run(self):
        grid = (
            triton.cdiv(self.max_q * self.heads, self.block_rows),
            self.num_split,
            self.batch,
        )
        _attention[grid](
            self.query,
            self.kv_cache,
            self.block_tables,
            self.seq_lens,
            self.cu,
            self.query_lengths,
            self.out,
            self.lse,
            self.partial_o,
            self.partial_m,
            self.partial_l,
            self.scale_log2,
            self.value_scale,
            self.heads,
            self.total_q,
            self.max_q,
            int(self.kv_cache.shape[0]),
            self.page_size,
            int(self.block_tables.shape[1]),
            int(self.block_tables.stride(0)),
            self.num_split,
            self.ragged,
            self.per_query_length,
            self.block_rows,
            64,
            num_warps=8,
            num_stages=2,
            enable_fp_fusion=False,
        )
        if self.num_split > 1:
            _merge[(self.max_q * self.heads, self.batch)](
                self.partial_o,
                self.partial_m,
                self.partial_l,
                self.out,
                self.lse,
                self.cu,
                self.heads,
                self.total_q,
                self.max_q,
                self.num_split,
                triton.next_power_of_2(self.num_split),
                self.ragged,
                self.value_scale,
                128,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return self.out, self.lse
