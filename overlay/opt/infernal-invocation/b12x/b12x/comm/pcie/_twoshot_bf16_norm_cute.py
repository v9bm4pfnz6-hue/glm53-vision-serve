"""BF16 two-shot push all-reduce with a fused RMSNorm-shard epilogue (CuTe DSL).

The kernel is the served push all-reduce (:class:`_TwoShotPushAllReduceLaunch`
in :mod:`_twoshot_bf16_cute`: phase one pushes shards to peers, a barrier,
phase two reduces this rank's shard in fixed rank order and publishes it,
a barrier, phase three copies the peers' reduced shards) followed, in the
single launched CTA, by the Kimi-K3 latent RMSNorm of every reduced row and
the store of this rank's column block of the normalized rows:

* the payload is ``num_rows`` rows of ``hidden_packs * 8`` bf16 values; after
  phase three the CTA holds the full reduced rows in ``output`` (its own
  stores, visible after the block barrier because the launch is one CTA);
* per row, four warps (128 threads) accumulate the squares of the bf16 values
  in float64 (bf16 squares are exact in fp64; the sum rounds at 2^-53) and
  one thread reduces the 128 partials in a fixed order, so the variance is at
  least as precise as any fp32 block reduction; the scale
  ``1 / sqrt(variance / hidden + eps)`` is computed in float64 with correctly
  rounded operations and rounded once to fp32, where the served CUDA kernel
  evaluates ``rsqrtf`` (an approximation of up to 2 ulp) on an fp32 variance;
* the normalized value is ``(x * scale) * weight`` in fp32 — the served
  kernel's operation order — rounded once to bf16, for the shard packs
  ``[shard_pack0, shard_pack0 + shard_packs)`` of every row; packs beyond
  the logical width (the last rank's padding) are stored as zeros, matching
  ``pack_projection_shard``.

Sixteen warps serve four rows per pass, so decode batches of up to sixteen
rows cost at most four passes. The full reduced rows are still written to
``output`` exactly as the served kernel writes them.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float64, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import ld_global_nc_v4_u32, ld_global_v4_u32, st_global_v4_u32
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._cute_intrinsics import (
    graph_epoch_arrive_serialized,
    ld_relaxed_gpu_u32,
    pack_f32x2_to_bf16x2,
    unpack_bf16x2,
)
from ._twoshot_bf16_cute import (
    _PACK_ELEMS,
    _PREPARED_BF16_LAUNCHERS,
    _TwoShotPushAllReduceLaunch,
    _bf16_process_key,
)
from ._twoshot_cute import (
    _GRAPH_BLOCKS_ARRIVED_OFFSET,
    _GRAPH_EPOCH_OFFSET,
    _ld_generic_v4_u32,
    _st_generic_v4_u32,
)

#: Rows served per pass: sixteen warps, four per row.
_ROWS_PER_PASS = 4
_THREADS_PER_ROW = 128


class _TwoShotPushAllReduceNormShardLaunch(_TwoShotPushAllReduceLaunch):
    """The push all-reduce plus the RMSNorm-shard epilogue (one CTA)."""

    @cute.jit
    def __call__(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        output: cute.Pointer,
        weight: cute.Pointer,
        shard: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        reduced_offset: Int64,
        slot_bytes: Int64,
        rows_per_rank: Int32,
        remainder_packs: Int32,
        num_rows: Int32,
        hidden_packs: Int32,
        shard_pack0: Int32,
        shard_packs: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            payload,
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            output,
            weight,
            shard,
            rank,
            pack_stride,
            reduced_offset,
            slot_bytes,
            rows_per_rank,
            remainder_packs,
            num_rows,
            hidden_packs,
            shard_pack0,
            shard_packs,
            eps,
        ).launch(
            grid=(1, 1, 1),
            block=[self._threads, 1, 1],
            max_number_threads=(512, 1, 1),
            min_blocks_per_mp=1,
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        payload: cute.Pointer,
        staging0: cute.Pointer,
        staging1: cute.Pointer,
        staging2: cute.Pointer,
        staging3: cute.Pointer,
        staging4: cute.Pointer,
        staging5: cute.Pointer,
        staging6: cute.Pointer,
        staging7: cute.Pointer,
        staging8: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        output: cute.Pointer,
        weight: cute.Pointer,
        shard: cute.Pointer,
        rank: Int32,
        pack_stride: Int64,
        reduced_offset: Int64,
        slot_bytes: Int64,
        rows_per_rank: Int32,
        remainder_packs: Int32,
        num_rows: Int32,
        hidden_packs: Int32,
        shard_pack0: Int32,
        shard_packs: Int32,
        eps: Float32,
    ) -> None:
        staging = (
            staging0,
            staging1,
            staging2,
            staging3,
            staging4,
            staging5,
            staging6,
            staging7,
            staging8,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        if cutlass.const_expr(self._static_peers):
            local_rank = Int32(self._rank)
        else:
            local_rank = rank
        packs_per_row = Int32(self._row_elems // _PACK_ELEMS)
        base_packs = Int64(rows_per_rank) * Int64(packs_per_row)
        shard_base = Int64(local_rank) * base_packs + Int64(
            cutlass.min(local_rank, remainder_packs)
        )
        shard_packs_owned = base_packs
        if local_rank < remainder_packs:
            shard_packs_owned = base_packs + Int64(1)
        threads = Int64(self._threads)
        grid_threads = Int64(gdim) * threads
        flat = Int64(bidx) * threads + Int64(tidx)

        payload_address = Int64(payload.toint())
        output_address = Int64(output.toint())
        staging_slot_offset = Int64(0)
        self_signal = self._select_address(signals, local_rank)
        if cutlass.const_expr(self._device_slot_selection):
            generation = ld_relaxed_gpu_u32(self_signal + Int64(_GRAPH_EPOCH_OFFSET))
            staging_slot_offset = (
                Int64((generation + Uint32(self._slot_bias)) % Uint32(2)) * slot_bytes
            )
            cute.arch.barrier()
            if Int32(tidx) == Int32(self._threads - 1):
                graph_epoch_arrive_serialized(
                    self_signal + Int64(_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_GRAPH_BLOCKS_ARRIVED_OFFSET),
                    Uint32(gdim),
                )
        self_base = self._select_address(staging, local_rank) + staging_slot_offset

        # Phase one: push this rank's contribution to each peer's shard into
        # the peer's payload region, at this rank's source slot.
        peer_index = Int32(1)
        while peer_index < Int32(self._world_size):
            destination = (local_rank + peer_index) % Int32(self._world_size)
            destination_shard_base = Int64(destination) * base_packs + Int64(
                cutlass.min(destination, remainder_packs)
            )
            destination_shard_packs = base_packs
            if destination < remainder_packs:
                destination_shard_packs = base_packs + Int64(1)
            destination_slot = (
                self._select_address(staging, destination)
                + staging_slot_offset
                + Int64(local_rank) * pack_stride * Int64(16)
            )
            index = flat
            while index < destination_shard_packs:
                words = ld_global_nc_v4_u32(
                    payload_address + (destination_shard_base + index) * Int64(16)
                )
                _st_generic_v4_u32(
                    destination_slot + index * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                index += grid_threads
            peer_index += Int32(1)

        self._barrier(signals, local_rank)

        # Phase two: reduce this rank's shard from local memory, publish the
        # result to the output and to every peer's reduced region.
        index = flat
        while index < shard_packs_owned:
            accumulator = cute.make_rmem_tensor((_PACK_ELEMS,), cutlass.Float32)
            for lane in cutlass.range_constexpr(_PACK_ELEMS):
                accumulator[lane] = Float32(0.0)
            local_words = ld_global_nc_v4_u32(
                payload_address + (shard_base + index) * Int64(16)
            )
            peer_words = []
            for peer_index in cutlass.range_constexpr(1, self._world_size):
                source_rank = (local_rank + Int32(peer_index)) % Int32(self._world_size)
                staged_pack = (
                    self_base
                    + Int64(source_rank) * pack_stride * Int64(16)
                    + index * Int64(16)
                )
                peer_words.append(_ld_generic_v4_u32(staged_pack))
            self._accumulate_words(accumulator, local_words)
            for peer_index in cutlass.range_constexpr(self._world_size - 1):
                self._accumulate_words(accumulator, peer_words[peer_index])
            self._store_pack(
                output_address + (shard_base + index) * Int64(16), accumulator
            )
            reduced_words = (
                pack_f32x2_to_bf16x2(accumulator[0], accumulator[1]),
                pack_f32x2_to_bf16x2(accumulator[2], accumulator[3]),
                pack_f32x2_to_bf16x2(accumulator[4], accumulator[5]),
                pack_f32x2_to_bf16x2(accumulator[6], accumulator[7]),
            )
            for peer_index in cutlass.range_constexpr(1, self._world_size):
                destination = (local_rank + Int32(peer_index)) % Int32(self._world_size)
                destination_reduced = (
                    self._select_address(staging, destination)
                    + staging_slot_offset
                    + reduced_offset
                    + Int64(local_rank) * pack_stride * Int64(16)
                )
                _st_generic_v4_u32(
                    destination_reduced + index * Int64(16),
                    reduced_words[0],
                    reduced_words[1],
                    reduced_words[2],
                    reduced_words[3],
                )
            index += grid_threads

        self._barrier(signals, local_rank)

        # Phase three: copy the peers' reduced shards from local memory.
        for peer_index in cutlass.range_constexpr(1, self._world_size):
            source_rank = (local_rank + Int32(peer_index)) % Int32(self._world_size)
            source_reduced = (
                self_base
                + reduced_offset
                + Int64(source_rank) * pack_stride * Int64(16)
            )
            source_shard_base = Int64(source_rank) * base_packs + Int64(
                cutlass.min(source_rank, remainder_packs)
            )
            source_shard_packs = base_packs
            if source_rank < remainder_packs:
                source_shard_packs = base_packs + Int64(1)
            destination = output_address + source_shard_base * Int64(16)
            index = flat
            while index < source_shard_packs:
                words = _ld_generic_v4_u32(source_reduced + index * Int64(16))
                st_global_v4_u32(
                    destination + index * Int64(16),
                    words[0],
                    words[1],
                    words[2],
                    words[3],
                )
                index += grid_threads

        # Epilogue: every reduced row is complete in ``output`` (this CTA's own
        # stores; the block barrier orders them before the reads below).
        cute.arch.sync_threads()

        smem_alloc = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            partials: cute.struct.Align[
                cute.struct.MemRange[Float64, _ROWS_PER_PASS * _THREADS_PER_ROW], 16
            ]
            scales: cute.struct.Align[cute.struct.MemRange[Float32, _ROWS_PER_PASS], 16]

        storage = smem_alloc.allocate(SharedStorage)
        partials = storage.partials.get_tensor(
            cute.make_layout((_ROWS_PER_PASS * _THREADS_PER_ROW,), stride=(1,))
        )
        scales = storage.scales.get_tensor(
            cute.make_layout((_ROWS_PER_PASS,), stride=(1,))
        )

        weight_address = Int64(weight.toint())
        shard_address = Int64(shard.toint())
        thread = Int32(tidx)
        group = thread // Int32(_THREADS_PER_ROW)
        tig = thread - group * Int32(_THREADS_PER_ROW)
        row0 = Int32(0)
        while row0 < num_rows:
            rows_here = num_rows - row0
            if rows_here > Int32(_ROWS_PER_PASS):
                rows_here = Int32(_ROWS_PER_PASS)
            row = row0 + group
            row_address = output_address + Int64(row) * Int64(hidden_packs) * Int64(16)

            # 1. Sum of squares of the row in float64, 128 threads per row.
            acc = Float64(0.0)
            if group < rows_here:
                pack = tig
                while pack < hidden_packs:
                    words = ld_global_v4_u32(row_address + Int64(pack) * Int64(16))
                    for word_index in cutlass.range_constexpr(4):
                        lo, hi = unpack_bf16x2(words[word_index])
                        lo64 = Float64(lo)
                        hi64 = Float64(hi)
                        acc = acc + lo64 * lo64
                        acc = acc + hi64 * hi64
                    pack += Int32(_THREADS_PER_ROW)
            partials[thread] = acc
            cute.arch.sync_threads()

            # 2. One thread per row reduces the partials in a fixed order and
            #    forms the fp32 scale from a float64 rsqrt.
            if tig == Int32(0) and group < rows_here:
                total = Float64(0.0)
                for item in cutlass.range_constexpr(_THREADS_PER_ROW):
                    total = total + partials[group * Int32(_THREADS_PER_ROW) + Int32(item)]
                hidden_elems = Float64(hidden_packs * Int32(_PACK_ELEMS))
                variance = total / hidden_elems + Float64(eps)
                scale64 = Float64(1.0) / cute.math.sqrt(variance, fastmath=False)
                scales[group] = Float32(scale64)
            cute.arch.sync_threads()

            # 3. Normalize and store this rank's column block of the row.
            if group < rows_here:
                scale = scales[group]
                shard_row_address = (
                    shard_address + Int64(row) * Int64(shard_packs) * Int64(16)
                )
                local_pack = tig
                while local_pack < shard_packs:
                    pack = shard_pack0 + local_pack
                    out_words = cute.make_rmem_tensor((4,), Uint32)
                    if pack < hidden_packs:
                        words = ld_global_v4_u32(row_address + Int64(pack) * Int64(16))
                        weights = ld_global_v4_u32(weight_address + Int64(pack) * Int64(16))
                        for word_index in cutlass.range_constexpr(4):
                            lo, hi = unpack_bf16x2(words[word_index])
                            wlo, whi = unpack_bf16x2(weights[word_index])
                            out_words[word_index] = pack_f32x2_to_bf16x2(
                                (lo * scale) * wlo, (hi * scale) * whi
                            )
                    else:
                        for word_index in cutlass.range_constexpr(4):
                            out_words[word_index] = Uint32(0)
                    st_global_v4_u32(
                        shard_row_address + Int64(local_pack) * Int64(16),
                        out_words[0],
                        out_words[1],
                        out_words[2],
                        out_words[3],
                    )
                    local_pack += Int32(_THREADS_PER_ROW)
            # The next pass reuses the partials and scales.
            cute.arch.sync_threads()
            row0 += Int32(_ROWS_PER_PASS)


NORM_SHARD_MODES = ("push_norm_shard", "push_static_norm_shard")


@functools.cache
def get_twoshot_bf16_allreduce_norm_shard_launcher(
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
    mode: str = "push_norm_shard",
) -> Callable[..., None]:
    """Compile the push all-reduce + RMSNorm-shard specialization.

    ``mode`` is ``"push_norm_shard"`` or ``"push_static_norm_shard"`` (the
    TP9 static-peer specialization); the all-reduce part keeps the served
    push kernel's shard partition and sum order.
    """
    if mode not in NORM_SHARD_MODES:
        raise ValueError(f"invalid norm-shard all-reduce mode {mode!r}")
    static = mode == "push_static_norm_shard"
    if static and world_size != 9:
        raise ValueError("the static-peer push specialization supports TP9 only")
    operation = f"all_reduce_{mode}"
    process_key = _bf16_process_key(
        operation,
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
        device_index,
    )
    del device_index
    if world_size not in (2, 4, 8, 9):
        raise ValueError(f"unsupported world size {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}")
    if threads != 512:
        raise ValueError("the norm-shard epilogue runs four rows per pass on 512 threads")
    if row_elems != _PACK_ELEMS:
        raise ValueError("the norm-shard all-reduce needs single-pack rows")
    slot_bias = int(slot_bias) & 1
    launch = _TwoShotPushAllReduceNormShardLaunch(
        world_size,
        rank,
        device_slot_selection,
        slot_bias,
        threads,
        row_elems,
        static_peers=static,
    )
    cache_key = (
        "bf16",
        operation,
        int(world_size),
        int(rank),
        bool(device_slot_selection),
        slot_bias,
        int(threads),
        int(row_elems),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    p16 = lambda: make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16)  # noqa: E731
    p4 = lambda: make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=4)  # noqa: E731
    raw = b12x_compile(
        launch,
        p16(),
        *(p16() for _ in range(9)),
        *(p4() for _ in range(9)),
        p16(),
        p16(),
        p16(),
        0, 1, 1, 1, 1, 0, 1, 448, 0, 50, 1.0e-6,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            f"comm.pcie.twoshot_bf16.{operation}",
            1,
            cache_key,
        ),
    )
    compiled_rank = int(rank)

    def run(
        payload_address: int,
        staging_addresses: Sequence[int],
        signal_addresses: Sequence[int],
        output_address: int,
        weight_address: int,
        shard_address: int,
        rank: int,
        reduced_offset: int,
        slot_bytes: int,
        rows_per_rank: int,
        remainder_packs: int,
        pack_stride: int,
        num_rows: int,
        hidden_packs: int,
        shard_pack0: int,
        shard_packs: int,
        eps: float,
    ) -> None:
        if len(staging_addresses) != 9 or len(signal_addresses) != 9:
            raise ValueError("two-shot scalar pointer ABI requires nine peer slots")
        if not 0 <= int(remainder_packs) < world_size:
            raise ValueError("remainder_packs must be below the world size")
        if static and rank != compiled_rank:
            raise ValueError("static-peer launcher rank does not match its specialization")
        if pack_stride <= 0:
            raise ValueError("the push all-reduce needs a positive pack_stride")
        raw(
            make_ptr(cutlass.Uint32, payload_address, cute.AddressSpace.gmem, assumed_align=16),
            *(
                make_ptr(cutlass.Uint32, address, cute.AddressSpace.gmem, assumed_align=16)
                for address in staging_addresses
            ),
            *(
                make_ptr(cutlass.Uint32, address, cute.AddressSpace.gmem, assumed_align=4)
                for address in signal_addresses
            ),
            make_ptr(cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, weight_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, shard_address, cute.AddressSpace.gmem, assumed_align=16),
            int(rank),
            int(pack_stride),
            int(reduced_offset),
            int(slot_bytes),
            int(rows_per_rank),
            int(remainder_packs),
            int(num_rows),
            int(hidden_packs),
            int(shard_pack0),
            int(shard_packs),
            float(eps),
            current_cuda_stream(),
        )

    _PREPARED_BF16_LAUNCHERS.add(process_key)
    return run


def is_twoshot_bf16_allreduce_norm_shard_launcher_prepared(
    world_size: int,
    rank: int,
    device_slot_selection: bool,
    slot_bias: int,
    threads: int,
    row_elems: int,
    device_index: int,
    mode: str = "push_norm_shard",
) -> bool:
    return (
        _bf16_process_key(
            f"all_reduce_{mode}",
            world_size,
            rank,
            device_slot_selection,
            slot_bias,
            threads,
            row_elems,
            device_index,
        )
        in _PREPARED_BF16_LAUNCHERS
    )


__all__ = [
    "NORM_SHARD_MODES",
    "get_twoshot_bf16_allreduce_norm_shard_launcher",
    "is_twoshot_bf16_allreduce_norm_shard_launcher_prepared",
]
