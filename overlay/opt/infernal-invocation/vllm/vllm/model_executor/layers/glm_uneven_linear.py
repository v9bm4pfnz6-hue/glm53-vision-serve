# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uneven column-parallel linears for the GLM FP8 TP9 composition.

The MLA low-rank down-projections (``q_a_proj``, ``kv_a_proj_with_mqa``), the
sparse indexer's ``wq_b`` and the MTP draft's ``eh_proj`` are
``ReplicatedLinear``/``nn.Linear`` tensors, so every TP rank loads the full
weight. On TP9 that duplicates 1.43 GiB per rank. Sharding the output rows
across the TP group and all-gathering the activations removes the duplication
while every consumer keeps seeing the full-width result.

The partition assigns whole quantization blocks to ranks: 128-row blocks for
FP8 weights with 128x128 scale blocks, 64-row units for the 576-row kv_a
projection (576 = 9 x 64). Unit counts per rank differ by at most one, so no
zero padding is allocated: the loaded weight and scale bytes across the group
equal the original checkpoint bytes. A 64-row shard shares its 128-row scale
block with the neighbouring rank and loads that (identical) scale row.

Forward evaluates the local shard through the stock quantization path, pads
only the activation to the widest shard, all-gathers it, and restores the
original row order with a single ``index_select``. No cross-rank reduction is
introduced. Per-row GEMM results can still vary at kernel-tiling level when
the shard width changes, so a teacher-forced score gate is required before
serving.

The ``arscatter`` transport places the local rows directly at their original
row positions inside a persistent zero buffer sized to the exact full output
width and all-reduces that buffer out of place. Every row position receives
exactly one non-zero contribution, so the reduction equals the concatenation
bit for bit and its result is already in the original row order: no
zero-fill, no per-rank padding on the wire (9-11 % fewer bytes) and no
post-reduction ``index_select`` per projection. The buffer keeps the zeros
outside this rank's rows across calls, so only the local rows are rewritten
each step; caching is limited to decode-sized row counts and oversized
(prefill) batches allocate a fresh buffer instead.
"""

from __future__ import annotations

import os
from itertools import accumulate

import torch
import torch.nn.functional as F

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

_GATHER_TRANSPORT: str | None = None

# Row counts at or below this keep a persistent scatter buffer; larger
# (eager prefill) batches allocate a fresh zero buffer per call instead so
# the per-instance cache stays small. Decode graphs never exceed this.
_SCATTER_CACHE_MAX_ROWS = 64


def glm_uneven_gather_transport() -> str:
    """How the uneven shards are gathered (read once from the environment).

    ``allgather`` (default) is the group all-gather. ``allreduce`` places the
    local rows in this rank's slot of a zero buffer and all-reduces it: every
    slot receives exactly one non-zero contribution, so the sum equals the
    concatenation bit for bit in any wire format, while the decode-sized
    payload (a few tens of KB per layer) takes the PCIe all-reduce path that
    already serves the decode graphs instead of a NCCL all-gather.
    ``arscatter`` is the all-reduce form with the local rows scattered
    straight to their original row positions in an exact-width persistent
    zero buffer, so the reduction result is the final layout and no reordering
    or per-rank padding exists at all. ``argemm`` additionally lets supported
    FP8 GEMMs store directly into that buffer instead of copying their local
    output; other quantization methods use the ``arscatter`` fallback.
    """
    global _GATHER_TRANSPORT
    if _GATHER_TRANSPORT is None:
        transport = (
            os.environ.get("VLLM_GLM_UNEVEN_GATHER", "allgather").strip().lower()
        )
        if transport not in ("allgather", "allreduce", "arscatter", "argemm"):
            raise ValueError(
                "VLLM_GLM_UNEVEN_GATHER must be allgather, allreduce, "
                "arscatter or argemm"
            )
        _GATHER_TRANSPORT = transport
    return _GATHER_TRANSPORT


def _gather_uneven(
    local: torch.Tensor, rank: int, world_size: int, max_local_rows: int
):
    if glm_uneven_gather_transport() == "allreduce":
        slots = local.new_zeros((*local.shape[:-1], world_size * max_local_rows))
        start = rank * max_local_rows
        slots[..., start : start + local.shape[-1]] = local
        return tensor_model_parallel_all_reduce(slots)
    pad = max_local_rows - local.shape[-1]
    if pad:
        local = F.pad(local, (0, pad))
    return tensor_model_parallel_all_gather(local)


def glm_uneven_low_rank_enabled() -> bool:
    """Opt-in gate: GLM FP8 TP9 only, with an env opt-out for A/B."""
    return (
        os.environ.get("VLLM_GLM_FP8_TP9") == "1"
        and os.environ.get("VLLM_GLM_FP8_SHARD_LOW_RANK", "1") == "1"
        and get_tensor_model_parallel_world_size() > 1
    )


def uneven_row_ranges(
    output_size: int, block_size: int, world_size: int
) -> list[tuple[int, int]]:
    """Contiguous per-rank row ranges; unit counts differ by at most one."""
    if output_size % block_size:
        raise ValueError(f"{output_size} is not a multiple of {block_size}")
    units = output_size // block_size
    base, extra = divmod(units, world_size)
    if base == 0:
        raise ValueError(
            f"Cannot shard {output_size} rows over {world_size} ranks at "
            f"{block_size}-row granularity: fewer units than ranks"
        )
    counts = [base + (1 if rank < extra else 0) for rank in range(world_size)]
    starts = [0, *accumulate(count * block_size for count in counts[:-1])]
    return [(start, start + count * block_size) for start, count in zip(starts, counts)]


class GlmUnevenColumnLinear(ReplicatedLinear):
    """ReplicatedLinear whose output rows are sharded unevenly across TP ranks."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        block_size: int = 128,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        params_dtype: torch.dtype | None = None,
        return_bias: bool = True,
    ):
        self.full_output_size = output_size
        self.block_size = block_size
        self.world_size = get_tensor_model_parallel_world_size()
        self.rank = get_tensor_model_parallel_rank()
        ranges = uneven_row_ranges(output_size, block_size, self.world_size)
        self.row_start, self.row_end = ranges[self.rank]
        local_rows = self.row_end - self.row_start
        self.max_local_rows = max(end - start for start, end in ranges)
        super().__init__(
            input_size,
            local_rows,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            params_dtype=params_dtype,
            return_bias=return_bias,
        )
        indices = [
            rank * self.max_local_rows + offset
            for rank, (start, end) in enumerate(ranges)
            for offset in range(end - start)
        ]
        self.register_buffer(
            "gather_indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self._scatter_slots: dict = {}

    def weight_loader(self, param, loaded_weight: torch.Tensor):
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == self.full_output_size:
            param.data.copy_(loaded_weight[self.row_start : self.row_end])
            return
        scale_rows = (self.full_output_size + 127) // 128
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == scale_rows:
            param.data.copy_(
                loaded_weight[self.row_start // 128 : (self.row_end + 127) // 128]
            )
            return
        raise ValueError(
            f"Unexpected checkpoint tensor shape {tuple(loaded_weight.shape)} "
            f"for a {self.full_output_size}-row projection"
        )

    def _scatter_local(self, slots: torch.Tensor, local: torch.Tensor):
        """Write this rank's local rows at their original row positions."""
        slots[..., self.row_start : self.row_end] = local

    def _scatter_spec(self) -> tuple[int, int, int]:
        return self.row_start, self.weight.shape[0], 0

    def _scatter_slots_for(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        cacheable = 0 < rows <= _SCATTER_CACHE_MAX_ROWS
        key = (tuple(x.shape[:-1]), x.dtype, x.device)
        slots = self._scatter_slots.get(key) if cacheable else None
        if slots is None:
            slots = x.new_zeros((*x.shape[:-1], self.full_output_size))
            if cacheable:
                self._scatter_slots[key] = slots
        return slots

    def _gather_scatter(self, local: torch.Tensor) -> torch.Tensor:
        """All-reduce a persistent zero buffer carrying the local rows at
        their original positions; each output row sums one contribution with
        eight zeros, so the result is the full-width output in the original
        row order."""
        slots = self._scatter_slots_for(local)
        self._scatter_local(slots, local)
        return tensor_model_parallel_all_reduce(slots)

    def forward(self, x: torch.Tensor):
        transport = glm_uneven_gather_transport()
        direct = transport == "argemm" and getattr(
            self.quant_method, "supports_glm_scatter_out", False
        )
        if self.world_size > 1 and direct:
            slots = self._scatter_slots_for(x)
            self.quant_method.apply(
                self, x, None, out=slots, scatter=self._scatter_spec()
            )
            local = tensor_model_parallel_all_reduce(slots)
        else:
            local = self.quant_method.apply(self, x, None)
            if self.world_size > 1:
                if transport in ("arscatter", "argemm"):
                    local = self._gather_scatter(local)
                else:
                    gathered = _gather_uneven(
                        local, self.rank, self.world_size, self.max_local_rows
                    )
                    local = gathered.index_select(-1, self.gather_indices)
        if not self.return_bias:
            return local
        return local, None


class GlmUnevenFusedQkvALinear(GlmUnevenColumnLinear):
    """Uneven TP shard of the fused [q_lora | kv_lora+rope] projection.

    The checkpoint keeps ``q_a_proj`` (2048 rows) and ``kv_a_proj_with_mqa``
    (576 rows) as separate tensors; the fused parameter packs this rank's q
    rows followed by its 64 kv rows. All-gather restores the fused
    [2048 | 576] layout, so downstream consumers are unchanged. kv uses 64-row
    units (576 = 9 x 64); a 64-row shard shares its 128-row scale block with
    the neighbouring rank. The fused local width is q_local + 64, whose scale
    row count ceil((q_local+64)/128) equals q_blocks + 1 because q_local is a
    multiple of 128, so the stock FP8 weight/scale allocation is exact.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        params_dtype: torch.dtype | None = None,
    ):
        if len(output_sizes) != 2:
            raise ValueError("The fused projection has exactly two components")
        q_size, kv_size = output_sizes
        if kv_size % 64:
            raise ValueError("kv component must be a multiple of 64")
        self.component_sizes = [q_size, kv_size]
        world = get_tensor_model_parallel_world_size()
        self.component_ranges = [
            uneven_row_ranges(q_size, 128, world),
            uneven_row_ranges(kv_size, 64, world),
        ]
        rank = get_tensor_model_parallel_rank()
        qs, qe = self.component_ranges[0][rank]
        ks, ke = self.component_ranges[1][rank]
        self.local_q = qe - qs
        self.local_kv = ke - ks
        self._q_start = qs
        self._kv_start = ks
        self.max_local_rows = max(
            (qe_ - qs_) + (ke_ - ks_)
            for (qs_, qe_), (ks_, ke_) in zip(*self.component_ranges)
        )
        # Initialize the replicated-linear machinery with the fused local
        # width, skipping GlmUnevenColumnLinear's single-range partition.
        ReplicatedLinear.__init__(
            self,
            input_size,
            self.local_q + self.local_kv,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            params_dtype=params_dtype,
        )
        # The MTP loader's kv 576->640 padding targets the stock fused layout;
        # this shard keeps the original 576 rows, so mark the parameter to
        # skip that padding.
        self.weight.glm_uneven_fused = True
        self.full_output_size = q_size + kv_size
        self.block_size = 128
        self.world_size = world
        self.rank = rank
        # The all-gathered layout holds per-rank [q | kv] chunks; the fused
        # reference layout is [all q rows | all kv rows], so gather the q
        # indices of every rank first, then the kv indices.
        q_indices: list[int] = []
        kv_indices: list[int] = []
        for other in range(world):
            qs_, qe_ = self.component_ranges[0][other]
            ks_, ke_ = self.component_ranges[1][other]
            base = other * self.max_local_rows
            q_indices.extend(range(base, base + (qe_ - qs_)))
            kv_base = base + (qe_ - qs_)
            kv_indices.extend(range(kv_base, kv_base + (ke_ - ks_)))
        indices = q_indices + kv_indices
        self.register_buffer(
            "gather_indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        # arscatter: local position -> original row position, matching the
        # [local q | local kv] packing of the fused local weight above.
        q_dest = list(range(qs, qe))
        kv_dest = list(range(q_size + ks, q_size + ke))
        self.register_buffer(
            "_scatter_index",
            torch.tensor(q_dest + kv_dest, dtype=torch.long),
            persistent=False,
        )
        self._scatter_slots: dict = {}

    def _scatter_local(self, slots: torch.Tensor, local: torch.Tensor):
        slots.index_copy_(-1, self._scatter_index, local)

    def _scatter_spec(self) -> tuple[int, int, int]:
        return (
            self._q_start,
            self.local_q,
            self.component_sizes[0] + self._kv_start - self.local_q,
        )

    def weight_loader(self, param, loaded_weight: torch.Tensor, shard_id: int = 0):
        if shard_id not in (0, 1):
            raise ValueError(f"Unexpected fused shard id {shard_id}")
        full_rows = self.component_sizes[shard_id]
        start, end = self.component_ranges[shard_id][self.rank]
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == full_rows:
            offset = 0 if shard_id == 0 else self.local_q
            param.data[offset : offset + (end - start)].copy_(loaded_weight[start:end])
            return
        scale_rows = (full_rows + 127) // 128
        if loaded_weight.ndim >= 1 and loaded_weight.shape[0] == scale_rows:
            offset = 0 if shard_id == 0 else (self.local_q + 127) // 128
            scale_start = start // 128
            scale_end = (end + 127) // 128
            param.data[offset : offset + (scale_end - scale_start)].copy_(
                loaded_weight[scale_start:scale_end]
            )
            return
        raise ValueError(
            f"Unexpected checkpoint tensor shape {tuple(loaded_weight.shape)} "
            f"for fused component {shard_id}"
        )
