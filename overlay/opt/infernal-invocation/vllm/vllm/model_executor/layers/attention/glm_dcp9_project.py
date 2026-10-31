# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Value-project FP32 sparse-MLA partials before the GLM TP9/DCP9 combine.

The nine-rank DCP combine of GLM-5.3 (``kv_lora_rank`` 512, ``v_head_dim`` 256)
exchanges each rank's FP32 partial attention output for every destination
rank's eight heads: 512 FP32 elements per head. The value up-projection
``W_UV`` is linear per head, so applying it to every rank's partial before the
exchange and combining the projected partials with the same softmax weights is
the same computation in exact arithmetic while the exchange moves 256 FP32
elements per head (half the bytes on the PCIe fabric that bounds the combine).

Numerics. The projection kernel reads the checkpoint's E4M3 ``kv_b_proj``
weights and their FP32 128x128 block scales (the served decode path uses a
BF16 copy of ``W_UV``), forms each weight in FP32 (``fp8 * scale``, one FP32
rounding) and accumulates the 512-term dot products with FP32 FMAs
(``tl.dot(..., input_precision="ieee")``). The b12x combine then weights the
projected partials in FP32, accumulates in FP64 and rounds once to BF16.
Relative to the served path (FP64 combine, BF16 rounding of the 512-wide
latent, BF16 ``W_UV`` bmm with FP32 accumulation, BF16 rounding) this removes
one BF16 rounding of the activations and the BF16 rounding of the weights;
``tests/model_executor/layers/test_glm_dcp9_project.py`` measures both paths
against an FP64 reference.

Weights. A rank projects its partial for every head: its own eight heads come
straight from its ``kv_b_proj`` shard (no copy), the other ranks' real heads
from a table gathered once after loading (E4M3 rows plus the FP32 scale of
every (row, 128-wide latent block)), and padded heads (all-zero ``kv_b_proj``
rows, the TP9 port's heads 64-71) are written as zeros without storage. The
table is 8.65 MB per layer for 64 remote heads and lives in one contiguous
arena per device (``_TableArena``), gathered over the CPU (gloo) group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton

_SCALE_BLOCK = 128
_BLOCK_ROWS = 16
_BLOCK_N = 32
_BLOCK_K = 32
PADDED_SLOT = -1


@dataclass
class GlmDcp9ProjectionWeights:
    """Per-rank operands of ``glm_dcp9_project_partials``.

    ``local_weight``/``local_scale`` are this rank's ``kv_b_proj`` E4M3 weight
    ``[heads_per_rank * (qk_nope + v), latent]`` and FP32 block scales
    ``[rows // 128, latent // 128]``; heads ``local_head_start`` to
    ``local_head_start + heads_per_rank - 1`` are read from them.
    ``table``/``table_scale`` hold the other ranks' real heads as
    ``[n, v, latent]`` E4M3 and ``[n, v, latent // 128]`` FP32; ``slots[h]`` is
    the table index of head ``h`` or ``PADDED_SLOT`` for local and padded heads.
    """

    local_weight: torch.Tensor
    local_scale: torch.Tensor
    local_head_start: int
    heads_per_rank: int
    qk_nope_head_dim: int
    v_head_dim: int
    table: torch.Tensor
    table_scale: torch.Tensor
    slots: torch.Tensor
    padded_heads: tuple[int, ...]

    @property
    def total_heads(self) -> int:
        return int(self.slots.numel())

    @property
    def table_bytes(self) -> int:
        return self.table.numel() + self.table_scale.numel() * 4


@triton.jit(do_not_specialize=["num_rows", "local_head_start"])
def _glm_dcp9_project_kernel(
    partial_ptr,
    local_weight_ptr,
    local_scale_ptr,
    table_ptr,
    table_scale_ptr,
    slot_ptr,
    out_ptr,
    num_rows,
    local_head_start,
    partial_stride_row,
    partial_stride_head,
    total_heads: tl.constexpr,
    heads_per_rank: tl.constexpr,
    head_rows: tl.constexpr,
    qk_nope_head_dim: tl.constexpr,
    latent_dim: tl.constexpr,
    v_dim: tl.constexpr,
    block_rows: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    head = tl.program_id(0)
    n_block = tl.program_id(1)
    row_block = tl.program_id(2)
    rows = row_block * block_rows + tl.arange(0, block_rows)
    cols = n_block * block_n + tl.arange(0, block_n)
    ks = tl.arange(0, block_k)
    row_mask = rows < num_rows
    k_blocks: tl.constexpr = latent_dim // 128
    partial_head = partial_ptr + head.to(tl.int64) * partial_stride_head
    slot = tl.load(slot_ptr + head)
    local_head = head - local_head_start
    is_local = (local_head >= 0) & (local_head < heads_per_rank)
    acc = tl.zeros((block_rows, block_n), tl.float32)
    if is_local:
        # This rank's kv_b_proj shard: value rows of local head l are rows
        # l * head_rows + qk_nope .. + v; their 128-row scale blocks need not
        # align with head boundaries.
        weight_rows = local_head * head_rows + qk_nope_head_dim + cols
        weight_base = local_weight_ptr + weight_rows.to(tl.int64)[:, None] * latent_dim
        scale_base = local_scale_ptr + (weight_rows // 128) * k_blocks
        for k_block in range(k_blocks):
            scale = tl.load(scale_base + k_block)
            # One FP32 scale per (row, 128-wide latent block); the block is
            # consumed in block_k-wide sub-tiles so the FMA dot (no TF32) stays
            # in registers: a 128-wide tile spills to a per-thread stack that
            # the driver reserves device-wide at the kernel's first launch.
            for sub in range(128 // block_k):
                k = k_block * 128 + sub * block_k + ks
                partial = tl.load(
                    partial_head
                    + rows[:, None].to(tl.int64) * partial_stride_row
                    + k[None, :],
                    row_mask[:, None],
                    other=0.0,
                )
                weight = tl.load(weight_base + k[None, :])
                weight32 = weight.to(tl.float32) * scale[:, None]
                acc += tl.dot(partial, tl.trans(weight32), input_precision="ieee")
    elif slot >= 0:
        weight_base = (
            table_ptr
            + slot.to(tl.int64) * v_dim * latent_dim
            + cols.to(tl.int64)[:, None] * latent_dim
        )
        scale_base = (
            table_scale_ptr + slot.to(tl.int64) * v_dim * k_blocks + cols * k_blocks
        )
        for k_block in range(k_blocks):
            scale = tl.load(scale_base + k_block)
            # One FP32 scale per (row, 128-wide latent block); the block is
            # consumed in block_k-wide sub-tiles so the FMA dot (no TF32) stays
            # in registers: a 128-wide tile spills to a per-thread stack that
            # the driver reserves device-wide at the kernel's first launch.
            for sub in range(128 // block_k):
                k = k_block * 128 + sub * block_k + ks
                partial = tl.load(
                    partial_head
                    + rows[:, None].to(tl.int64) * partial_stride_row
                    + k[None, :],
                    row_mask[:, None],
                    other=0.0,
                )
                weight = tl.load(weight_base + k[None, :])
                weight32 = weight.to(tl.float32) * scale[:, None]
                acc += tl.dot(partial, tl.trans(weight32), input_precision="ieee")
    # Padded heads (slot < 0, not local) keep acc == 0.
    tl.store(
        out_ptr
        + rows[:, None].to(tl.int64) * (total_heads * v_dim)
        + head.to(tl.int64) * v_dim
        + cols[None, :],
        acc,
        row_mask[:, None],
    )


@triton.jit(do_not_specialize=["num_rows", "local_head_start"])
def _glm_dcp9_project_kernel_v2(
    partial_ptr,
    local_weight_ptr,
    local_scale_ptr,
    table_ptr,
    table_scale_ptr,
    slot_ptr,
    out_ptr,
    num_rows,
    local_head_start,
    partial_stride_row,
    partial_stride_head,
    total_heads: tl.constexpr,
    heads_per_rank: tl.constexpr,
    head_rows: tl.constexpr,
    qk_nope_head_dim: tl.constexpr,
    latent_dim: tl.constexpr,
    v_dim: tl.constexpr,
    block_rows: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_stages: tl.constexpr,
):
    """Same arithmetic as ``_glm_dcp9_project_kernel``, pipelined.

    Each output element is still the sequence of ``latent_dim // block_k``
    ``tl.dot(..., input_precision="ieee")`` sub-tiles over ascending latent
    blocks, added to the accumulator in order, with ``fp8 * scale`` formed in
    FP32; the two kernels are bitwise identical. The weight sub-tile is loaded
    directly in ``[block_k, block_n]`` order (no transpose through shared
    memory) and the 128-wide scale-block loop is a pipelined ``tl.range`` so
    the next block's loads overlap the current block's FMA dots instead of
    serializing one global-memory latency per sub-tile.
    """
    head = tl.program_id(0)
    n_block = tl.program_id(1)
    row_block = tl.program_id(2)
    rows = row_block * block_rows + tl.arange(0, block_rows)
    cols = n_block * block_n + tl.arange(0, block_n)
    ks = tl.arange(0, block_k)
    row_mask = rows < num_rows
    k_blocks: tl.constexpr = latent_dim // 128
    subs: tl.constexpr = 128 // block_k
    partial_head = partial_ptr + head.to(tl.int64) * partial_stride_head
    partial_rows = partial_head + rows[:, None].to(tl.int64) * partial_stride_row
    slot = tl.load(slot_ptr + head)
    local_head = head - local_head_start
    is_local = (local_head >= 0) & (local_head < heads_per_rank)
    acc = tl.zeros((block_rows, block_n), tl.float32)
    if is_local:
        weight_rows = local_head * head_rows + qk_nope_head_dim + cols
        # [block_k, block_n]: column n is the contiguous latent row of weight
        # row n, read in block_k-byte segments.
        weight_cols = local_weight_ptr + weight_rows.to(tl.int64)[None, :] * latent_dim
        scale_cols = local_scale_ptr + (weight_rows // 128) * k_blocks
        for k_block in tl.range(0, k_blocks, num_stages=num_stages):
            scale = tl.load(scale_cols + k_block)
            for sub in tl.static_range(subs):
                k = k_block * 128 + sub * block_k + ks
                partial = tl.load(
                    partial_rows + k[None, :], row_mask[:, None], other=0.0
                )
                weight = tl.load(weight_cols + k[:, None])
                weight32 = weight.to(tl.float32) * scale[None, :]
                acc += tl.dot(partial, weight32, input_precision="ieee")
    elif slot >= 0:
        weight_cols = (
            table_ptr
            + slot.to(tl.int64) * v_dim * latent_dim
            + cols.to(tl.int64)[None, :] * latent_dim
        )
        scale_cols = (
            table_scale_ptr + slot.to(tl.int64) * v_dim * k_blocks + cols * k_blocks
        )
        for k_block in tl.range(0, k_blocks, num_stages=num_stages):
            scale = tl.load(scale_cols + k_block)
            for sub in tl.static_range(subs):
                k = k_block * 128 + sub * block_k + ks
                partial = tl.load(
                    partial_rows + k[None, :], row_mask[:, None], other=0.0
                )
                weight = tl.load(weight_cols + k[:, None])
                weight32 = weight.to(tl.float32) * scale[None, :]
                acc += tl.dot(partial, weight32, input_precision="ieee")
    tl.store(
        out_ptr
        + rows[:, None].to(tl.int64) * (total_heads * v_dim)
        + head.to(tl.int64) * v_dim
        + cols[None, :],
        acc,
        row_mask[:, None],
    )


@triton.jit(do_not_specialize=["num_rows", "local_head_start"])
def _glm_dcp9_project_kernel_v3(
    partial_ptr,
    local_weight_ptr,
    local_scale_ptr,
    table_ptr,
    table_scale_ptr,
    slot_ptr,
    out_ptr,
    num_rows,
    local_head_start,
    partial_stride_row,
    partial_stride_head,
    total_heads: tl.constexpr,
    heads_per_rank: tl.constexpr,
    head_rows: tl.constexpr,
    qk_nope_head_dim: tl.constexpr,
    latent_dim: tl.constexpr,
    v_dim: tl.constexpr,
    block_rows: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_stages: tl.constexpr,
):
    """v2 with one flat pipelined loop over all latent sub-tiles.

    Identical arithmetic and order to v1/v2 (``fp8 * scale`` in FP32, one
    ``tl.dot(..., input_precision="ieee")`` per ascending block_k sub-tile,
    accumulated in order). The loop runs over all ``latent_dim // block_k``
    sub-tiles instead of nesting four sub-tiles inside each 128-wide scale
    block, so the software pipeline prefetches across scale-block boundaries;
    the per-sub-tile scale load re-reads the same FP32 value.
    """
    head = tl.program_id(0)
    n_block = tl.program_id(1)
    row_block = tl.program_id(2)
    rows = row_block * block_rows + tl.arange(0, block_rows)
    cols = n_block * block_n + tl.arange(0, block_n)
    ks = tl.arange(0, block_k)
    row_mask = rows < num_rows
    k_blocks: tl.constexpr = latent_dim // 128
    subs: tl.constexpr = 128 // block_k
    tiles: tl.constexpr = latent_dim // block_k
    partial_head = partial_ptr + head.to(tl.int64) * partial_stride_head
    partial_rows = partial_head + rows[:, None].to(tl.int64) * partial_stride_row
    slot = tl.load(slot_ptr + head)
    local_head = head - local_head_start
    is_local = (local_head >= 0) & (local_head < heads_per_rank)
    acc = tl.zeros((block_rows, block_n), tl.float32)
    if is_local:
        weight_rows = local_head * head_rows + qk_nope_head_dim + cols
        weight_cols = local_weight_ptr + weight_rows.to(tl.int64)[None, :] * latent_dim
        scale_cols = local_scale_ptr + (weight_rows // 128) * k_blocks
        for tile in tl.range(0, tiles, num_stages=num_stages):
            scale = tl.load(scale_cols + tile // subs)
            k = tile * block_k + ks
            partial = tl.load(partial_rows + k[None, :], row_mask[:, None], other=0.0)
            weight = tl.load(weight_cols + k[:, None])
            weight32 = weight.to(tl.float32) * scale[None, :]
            acc += tl.dot(partial, weight32, input_precision="ieee")
    elif slot >= 0:
        weight_cols = (
            table_ptr
            + slot.to(tl.int64) * v_dim * latent_dim
            + cols.to(tl.int64)[None, :] * latent_dim
        )
        scale_cols = (
            table_scale_ptr + slot.to(tl.int64) * v_dim * k_blocks + cols * k_blocks
        )
        for tile in tl.range(0, tiles, num_stages=num_stages):
            scale = tl.load(scale_cols + tile // subs)
            k = tile * block_k + ks
            partial = tl.load(partial_rows + k[None, :], row_mask[:, None], other=0.0)
            weight = tl.load(weight_cols + k[:, None])
            weight32 = weight.to(tl.float32) * scale[None, :]
            acc += tl.dot(partial, weight32, input_precision="ieee")
    tl.store(
        out_ptr
        + rows[:, None].to(tl.int64) * (total_heads * v_dim)
        + head.to(tl.int64) * v_dim
        + cols[None, :],
        acc,
        row_mask[:, None],
    )


def _projection_kernel_variant() -> str:
    import os

    variant = os.environ.get("VLLM_GLM_DCP9_PROJECT_KERNEL", "v1").strip().lower()
    if variant not in ("v1", "v2", "v3"):
        raise ValueError("VLLM_GLM_DCP9_PROJECT_KERNEL must be v1, v2 or v3")
    return variant


def _projection_v2_config() -> tuple[int, int, int]:
    """(block_n, num_stages, num_warps) for the pipelined kernel."""
    import os

    spec = os.environ.get("VLLM_GLM_DCP9_PROJECT_V2_CONFIG", "32,3,4")
    block_n, num_stages, num_warps = (int(part) for part in spec.split(","))
    return block_n, num_stages, num_warps


_PROJECTION_SCRATCH: dict[tuple, torch.Tensor] = {}


def glm_dcp9_projection_scratch(
    device: torch.device, rows: int, heads: int, v_dim: int, capacity_rows: int
) -> torch.Tensor:
    """Static FP32 ``[rows, heads, v]`` output for the projection kernel.

    One buffer per (device, geometry) is shared by every layer: the layers run
    sequentially on one stream and the exchange kernel consumes the projection
    before the next layer overwrites it. A fresh allocation per layer would be
    kept alive for the whole CUDA graph capture by the exchange's stream
    bookkeeping (about 1.2 MiB x 78 layers per captured decode graph).
    """
    if rows > capacity_rows:
        raise ValueError(
            f"{rows} rows exceed the projection scratch capacity {capacity_rows}"
        )
    key = (str(device), heads, v_dim, capacity_rows)
    buffer = _PROJECTION_SCRATCH.get(key)
    if buffer is None:
        buffer = torch.empty(
            (capacity_rows, heads, v_dim), dtype=torch.float32, device=device
        )
        _PROJECTION_SCRATCH[key] = buffer
    return buffer[:rows]


def glm_dcp9_project_partials(
    partial: torch.Tensor,
    weights: GlmDcp9ProjectionWeights,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project FP32 partials ``[rows, heads, latent]`` to ``[rows, heads, v]``."""
    if partial.dtype != torch.float32 or partial.ndim != 3 or partial.stride(2) != 1:
        raise ValueError("partial must be a 3-D FP32 tensor with unit last stride")
    rows, heads, latent = (int(v) for v in partial.shape)
    v_dim = weights.v_head_dim
    head_rows = weights.qk_nope_head_dim + v_dim
    if heads != weights.total_heads:
        raise ValueError(
            f"partial has {heads} heads, weights cover {weights.total_heads}"
        )
    if latent % 128 != 0 or v_dim % _BLOCK_N != 0:
        raise ValueError("latent must be a multiple of 128 and v a multiple of 64")
    k_blocks = latent // _SCALE_BLOCK
    local_weight, local_scale = weights.local_weight, weights.local_scale
    if (
        local_weight.dtype != torch.float8_e4m3fn
        or not local_weight.is_contiguous()
        or tuple(local_weight.shape) != (weights.heads_per_rank * head_rows, latent)
        or local_scale.dtype != torch.float32
        or not local_scale.is_contiguous()
        or tuple(local_scale.shape)
        != (triton.cdiv(weights.heads_per_rank * head_rows, _SCALE_BLOCK), k_blocks)
    ):
        raise ValueError(
            "local kv_b_proj weight/scale do not match the projection geometry"
        )
    table, table_scale = weights.table, weights.table_scale
    if (
        table.dtype != torch.float8_e4m3fn
        or not table.is_contiguous()
        or table.ndim != 3
        or tuple(table.shape[1:]) != (v_dim, latent)
        or table_scale.dtype != torch.float32
        or not table_scale.is_contiguous()
        or tuple(table_scale.shape) != (table.shape[0], v_dim, k_blocks)
    ):
        raise ValueError("remote W_UV table/scale do not match the projection geometry")
    if weights.slots.dtype != torch.int32 or not weights.slots.is_contiguous():
        raise ValueError("slots must be a contiguous int32 tensor")
    if out is None:
        out = torch.empty(
            (rows, heads, v_dim), dtype=torch.float32, device=partial.device
        )
    elif (
        out.dtype != torch.float32
        or tuple(out.shape) != (rows, heads, v_dim)
        or not out.is_contiguous()
    ):
        raise ValueError("out must be a contiguous FP32 [rows, heads, v] tensor")
    if rows == 0:
        return out
    variant = _projection_kernel_variant()
    if variant in ("v2", "v3"):
        block_n, num_stages, num_warps = _projection_v2_config()
        if v_dim % block_n != 0:
            raise ValueError("v2/v3 block_n must divide v_head_dim")
        grid = (heads, v_dim // block_n, triton.cdiv(rows, _BLOCK_ROWS))
        kernel = (
            _glm_dcp9_project_kernel_v2
            if variant == "v2"
            else _glm_dcp9_project_kernel_v3
        )
        kernel[grid](
            partial,
            local_weight,
            local_scale,
            table,
            table_scale,
            weights.slots,
            out,
            rows,
            weights.local_head_start,
            partial.stride(0),
            partial.stride(1),
            total_heads=heads,
            heads_per_rank=weights.heads_per_rank,
            head_rows=head_rows,
            qk_nope_head_dim=weights.qk_nope_head_dim,
            latent_dim=latent,
            v_dim=v_dim,
            block_rows=_BLOCK_ROWS,
            block_n=block_n,
            block_k=_BLOCK_K,
            num_stages=num_stages,
            num_warps=num_warps,
        )
        return out
    grid = (heads, v_dim // _BLOCK_N, triton.cdiv(rows, _BLOCK_ROWS))
    _glm_dcp9_project_kernel[grid](
        partial,
        local_weight,
        local_scale,
        table,
        table_scale,
        weights.slots,
        out,
        rows,
        weights.local_head_start,
        partial.stride(0),
        partial.stride(1),
        total_heads=heads,
        heads_per_rank=weights.heads_per_rank,
        head_rows=head_rows,
        qk_nope_head_dim=weights.qk_nope_head_dim,
        latent_dim=latent,
        v_dim=v_dim,
        block_rows=_BLOCK_ROWS,
        block_n=_BLOCK_N,
        block_k=_BLOCK_K,
        num_warps=4,
    )
    return out


class _TableArena:
    """Contiguous per-device storage for the remote-head tables of many layers.

    One allocation per (device, table geometry) sized for ``layers_hint`` layers
    keeps the 7-8 MiB per-layer tables out of the caching allocator's 20 MiB
    large-block segments (no stranded fragments inside partially used segments)
    and makes their memory cost exact. A request past the capacity opens a new
    chunk for the remaining layers.
    """

    def __init__(self, device, table_elems: int, scale_elems: int, layers: int):
        self.table_elems = table_elems
        self.scale_elems = scale_elems
        self.capacity = max(1, layers)
        self.used = 0
        self.tables = torch.empty(
            self.capacity * table_elems, dtype=torch.float8_e4m3fn, device=device
        )
        self.scales = torch.empty(
            self.capacity * scale_elems, dtype=torch.float32, device=device
        )

    def take(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.used >= self.capacity:
            return None
        index = self.used
        self.used += 1
        table = self.tables[index * self.table_elems : (index + 1) * self.table_elems]
        scale = self.scales[index * self.scale_elems : (index + 1) * self.scale_elems]
        return table, scale


_ARENAS: dict[tuple, list[_TableArena]] = {}


def _arena_slices(
    device: torch.device,
    shape_table: tuple[int, int, int],
    shape_scale: tuple[int, int, int],
    layers_hint: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    table_elems = shape_table[0] * shape_table[1] * shape_table[2]
    scale_elems = shape_scale[0] * shape_scale[1] * shape_scale[2]
    key = (str(device), table_elems, scale_elems)
    chunks = _ARENAS.setdefault(key, [])
    if not chunks or chunks[-1].used >= chunks[-1].capacity:
        # The first chunk covers the layers the caller expects; later chunks
        # (another model instance such as the MTP draft) take one layer each.
        chunks.append(
            _TableArena(
                device, table_elems, scale_elems, layers_hint if not chunks else 1
            )
        )
    taken = chunks[-1].take()
    assert taken is not None
    table, scale = taken
    return table.view(*shape_table), scale.view(*shape_scale)


def build_glm_dcp9_projection_weights(
    local_weight: torch.Tensor,
    local_scale: torch.Tensor,
    gathered_weight: torch.Tensor,
    gathered_scale: torch.Tensor,
    *,
    rank: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    layers_hint: int = 1,
) -> GlmDcp9ProjectionWeights:
    """Assemble the per-rank projection operands from the gathered shards.

    ``gathered_weight`` is every rank's ``kv_b_proj`` shard concatenated in rank
    order (``[world * heads_per_rank * (qk_nope + v), latent]`` E4M3; rank order
    is head order) and ``gathered_scale`` the matching ``[rows // 128,
    latent // 128]`` FP32 block scales; both may live on the CPU. Heads whose
    value rows are all zero are padding and get no table entry. The table and
    its scales are written into the arena of ``local_weight``'s device.
    """
    if (
        gathered_weight.dtype != torch.float8_e4m3fn
        or local_weight.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("GLM DCP9 projected combine requires E4M3 kv_b_proj weights")
    if gathered_scale.dtype != torch.float32 or local_scale.dtype != torch.float32:
        raise TypeError("GLM DCP9 projected combine requires FP32 kv_b_proj scales")
    head_rows = qk_nope_head_dim + v_head_dim
    n_rows, latent = (int(v) for v in gathered_weight.shape)
    local_rows = int(local_weight.shape[0])
    if (
        n_rows % local_rows != 0
        or local_rows % head_rows != 0
        or latent % _SCALE_BLOCK != 0
    ):
        raise ValueError(
            f"kv_b_proj geometry: gathered rows {n_rows}, local rows {local_rows}, "
            f"head rows {head_rows}, latent {latent}"
        )
    world = n_rows // local_rows
    heads_per_rank = local_rows // head_rows
    total_heads = world * heads_per_rank
    expected_scale = (triton.cdiv(n_rows, _SCALE_BLOCK), latent // _SCALE_BLOCK)
    if tuple(gathered_scale.shape) != expected_scale:
        raise ValueError(
            f"kv_b_proj scale shape {tuple(gathered_scale.shape)} != {expected_scale}"
        )
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} outside world {world}")
    source = gathered_weight.device
    device = local_weight.device
    value_rows = gathered_weight.view(total_heads, head_rows, latent)[
        :, qk_nope_head_dim:, :
    ]
    row_index = (
        torch.arange(total_heads, device=source)[:, None] * head_rows
        + qk_nope_head_dim
        + torch.arange(v_head_dim, device=source)[None, :]
    )
    scale_rows = gathered_scale[row_index // _SCALE_BLOCK]
    nonzero = torch.count_nonzero(
        value_rows.reshape(total_heads, -1).view(torch.uint8), dim=1
    ).tolist()
    local_start = rank * heads_per_rank
    slots = torch.full((total_heads,), PADDED_SLOT, dtype=torch.int32)
    remote: list[int] = []
    padded: list[int] = []
    for head in range(total_heads):
        if local_start <= head < local_start + heads_per_rank:
            continue
        if nonzero[head] == 0:
            padded.append(head)
            continue
        slots[head] = len(remote)
        remote.append(head)
    k_blocks = latent // _SCALE_BLOCK
    table, table_scale = _arena_slices(
        device,
        (len(remote), v_head_dim, latent),
        (len(remote), v_head_dim, k_blocks),
        layers_hint,
    )
    index = torch.tensor(remote, dtype=torch.long, device=source)
    table.copy_(value_rows[index])
    table_scale.copy_(scale_rows[index])
    return GlmDcp9ProjectionWeights(
        local_weight=local_weight.contiguous(),
        local_scale=local_scale.contiguous(),
        local_head_start=local_start,
        heads_per_rank=heads_per_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        table=table,
        table_scale=table_scale,
        slots=slots.to(device),
        padded_heads=tuple(padded),
    )


def gather_glm_dcp9_projection_weights(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    cpu_group: dist.ProcessGroup,
    *,
    rank: int,
    world_size: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    layers_hint: int = 1,
) -> GlmDcp9ProjectionWeights:
    """Gather every rank's E4M3 ``kv_b_proj`` shard over the CPU (gloo) group.

    The gather stays off the GPU communicators on purpose: a first NCCL
    collective of a new shape on the DCP group would allocate communicator
    buffers that the KV-cache memory profiler then has to pay for.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError("GLM DCP9 projected combine requires E4M3 kv_b_proj weights")
    if weight_scale_inv.dtype != torch.float32:
        raise TypeError("GLM DCP9 projected combine requires FP32 kv_b_proj scales")
    local_weight = weight.contiguous()
    local_scale = weight_scale_inv.contiguous()
    weight_cpu = local_weight.view(torch.uint8).cpu()
    scale_cpu = local_scale.cpu()
    weight_parts = [torch.empty_like(weight_cpu) for _ in range(world_size)]
    scale_parts = [torch.empty_like(scale_cpu) for _ in range(world_size)]
    dist.all_gather(weight_parts, weight_cpu, group=cpu_group)
    dist.all_gather(scale_parts, scale_cpu, group=cpu_group)
    if not torch.equal(weight_parts[rank], weight_cpu):
        raise RuntimeError("GLM DCP9 projected combine: gathered shard order mismatch")
    return build_glm_dcp9_projection_weights(
        local_weight,
        local_scale,
        torch.cat(weight_parts, dim=0).view(torch.float8_e4m3fn),
        torch.cat(scale_parts, dim=0),
        rank=rank,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        layers_hint=layers_hint,
    )


def dequant_glm_dcp9_w_uv_reference(
    weights: GlmDcp9ProjectionWeights, head: int, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Exact (fp8 * scale) ``W_UV`` of one head as ``[latent, v]`` in ``dtype``."""
    local_head = head - weights.local_head_start
    head_rows = weights.qk_nope_head_dim + weights.v_head_dim
    if 0 <= local_head < weights.heads_per_rank:
        rows = torch.arange(weights.v_head_dim, device=weights.local_weight.device)
        rows = local_head * head_rows + weights.qk_nope_head_dim + rows
        values = weights.local_weight[rows].to(dtype)
        scale = weights.local_scale[rows // _SCALE_BLOCK].to(dtype)
    else:
        slot = int(weights.slots[head])
        if slot < 0:
            return torch.zeros(
                (weights.local_weight.shape[1], weights.v_head_dim),
                dtype=dtype,
                device=weights.local_weight.device,
            )
        values = weights.table[slot].to(dtype)
        scale = weights.table_scale[slot].to(dtype)
    return (values * scale.repeat_interleave(_SCALE_BLOCK, dim=1)).T.contiguous()
