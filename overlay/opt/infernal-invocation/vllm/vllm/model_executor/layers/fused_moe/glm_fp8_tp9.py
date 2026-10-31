# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exact 128-channel checkpoint shards for GLM block-FP8 TP9 MoE."""


def get_glm_fp8_tp9_shard(
    layer_idx: int,
    tp_rank: int,
    *,
    original_size: int = 2048,
    first_sparse_layer: int = 3,
    tp_size: int = 9,
    block_size: int = 128,
) -> tuple[int, int]:
    """Return the original-channel start and width for one routed MoE rank."""
    if layer_idx < first_sparse_layer:
        raise ValueError("GLM FP8 TP9 sharding requires a sparse or MTP layer")
    if tp_size != 9 or not 0 <= tp_rank < tp_size:
        raise ValueError("GLM FP8 TP9 sharding requires rank 0..8 of TP9")
    if original_size != 2048 or block_size != 128:
        raise ValueError("GLM FP8 TP9 sharding requires 2048 by 128 blocks")

    ordinal = layer_idx - first_sparse_layer
    short_rank = (2 * ordinal) % tp_size
    short_ranks = {short_rank, (short_rank + 1) % tp_size}
    widths = [
        block_size if rank in short_ranks else 2 * block_size for rank in range(tp_size)
    ]
    assert sum(widths) == original_size
    return sum(widths[:tp_rank]), widths[tp_rank]
