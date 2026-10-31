# SPDX-License-Identifier: Apache-2.0
"""SM120 warp-MMA port of packed-row, split-KV Kimi MLA attention.

Q/K/V retain their native E4M3 representation. Probability HIGH/LOW E4M3
and FP32 partials preserve the arithmetic precision of the deployed B12X
SM120 reader; no BF16 partial-output or single-FP8 probability shortcut is
used. Planning is static; lengths and page IDs remain replay-time inputs.
"""

import triton
import triton.language as tl


@triton.jit
def _attention(
    Q,
    KV,
    TABLE,
    LENGTHS,
    CU_Q,
    QUERY_LENGTHS,
    O,
    LSE,
    PARTIAL_O,
    PARTIAL_M,
    PARTIAL_L,
    SCALE_LOG2: tl.constexpr,
    VALUE_SCALE: tl.constexpr,
    HEADS: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    MAX_Q: tl.constexpr,
    PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    RAGGED: tl.constexpr,
    PER_QUERY_LENGTH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    tile = tl.program_id(0)
    split = tl.program_id(1)
    request = tl.program_id(2)
    if RAGGED:
        q_start = tl.load(CU_Q + request)
        q_end = tl.load(CU_Q + request + 1)
        q_count = tl.minimum(tl.maximum(q_end - q_start, 0), MAX_Q)
    else:
        q_start = request * MAX_Q
        q_count = MAX_Q
    packed_row = tile * BM + tl.arange(0, BM)
    query_row = q_start + packed_row // HEADS
    head = packed_row % HEADS
    live_row = (packed_row < q_count * HEADS) & (query_row >= 0) & (query_row < TOTAL_Q)
    output_row = query_row.to(tl.int64) * HEADS + head
    if PER_QUERY_LENGTH:
        visible = tl.load(QUERY_LENGTHS + query_row, live_row, other=0)
    else:
        visible = tl.full((BM,), tl.load(LENGTHS + request), tl.int32)
    visible = tl.minimum(tl.maximum(visible, 0), TABLE_WIDTH * PAGE_SIZE)
    visible = tl.where(live_row, visible, 0)
    max_visible = tl.max(visible, 0)
    chunks = tl.cdiv(max_visible, BN)
    chunks_per_split = tl.cdiv(chunks, SPLITS)
    begin = split * chunks_per_split
    end = tl.minimum(begin + chunks_per_split, chunks)

    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    q_latent = tl.load(
        Q + output_row[:, None] * 576 + d[None, :], live_row[:, None], other=0.0
    )
    q_rope = tl.load(
        Q + output_row[:, None] * 576 + 512 + r[None, :], live_row[:, None], other=0.0
    )
    acc = tl.full((BM, 512), 0, tl.float32)
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    for chunk in range(begin, end):
        token = chunk * BN + tl.arange(0, BN)
        table_col = token // PAGE_SIZE
        page = tl.load(
            TABLE + request.to(tl.int64) * TABLE_STRIDE + table_col,
            table_col < TABLE_WIDTH,
            other=-1,
        )
        valid_page = (page >= 0) & (page < PAGES) & (token < max_visible)
        # Promote before multiplying: a recycled physical page may start past
        # 2 GiB even when a request has only a handful of visible tokens.
        cache_row = page.to(tl.int64) * PAGE_SIZE + token % PAGE_SIZE
        k_latent = tl.load(
            KV + cache_row[:, None] * 576 + d[None, :], valid_page[:, None], other=0.0
        )
        k_rope = tl.load(
            KV + cache_row[:, None] * 576 + 512 + r[None, :],
            valid_page[:, None],
            other=0.0,
        )
        scores = tl.dot(
            q_latent, tl.trans(k_latent), out_dtype=tl.float32, max_num_imprecise_acc=0
        )
        scores = tl.dot(
            q_rope,
            tl.trans(k_rope),
            scores,
            out_dtype=tl.float32,
            max_num_imprecise_acc=0,
        )
        valid = (
            live_row[:, None]
            & valid_page[None, :]
            & (token[None, :] < visible[:, None])
        )
        scores = tl.where(valid, scores * SCALE_LOG2, float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(next_maximum == float("-inf"), 0.0, next_maximum)
        rescale = tl.exp2(maximum - safe_maximum)
        probability = tl.where(valid, tl.exp2(scores - safe_maximum[:, None]), 0.0)
        denominator = denominator * rescale + tl.sum(probability, 1)
        acc = acc * rescale[:, None]

        # The 64-token quantization domain and both E4M3 terms match the
        # existing SM120 reader. BN stays 64 to retain that precision contract.
        p_scale = tl.maximum(tl.max(probability, 1), 1.0e-10) * (1.0 / 448.0)
        inverse_scale = tl.div_rn(1.0, p_scale)
        scaled = tl.minimum(
            tl.maximum(probability * inverse_scale[:, None], -448.0), 448.0
        )
        high = scaled.to(tl.float8e4nv)
        residual = tl.minimum(tl.maximum(scaled - high.to(tl.float32), -448.0), 448.0)
        low = residual.to(tl.float8e4nv)
        high_value = tl.dot(
            high, k_latent, out_dtype=tl.float32, max_num_imprecise_acc=0
        )
        low_value = tl.dot(low, k_latent, out_dtype=tl.float32, max_num_imprecise_acc=0)
        acc = acc + (high_value + low_value) * p_scale[:, None]
        maximum = next_maximum

    if SPLITS == 1:
        reciprocal = tl.div_rn(1.0, tl.where(denominator > 0.0, denominator, 1.0))
        result = acc * reciprocal[:, None] * VALUE_SCALE
        logsum = tl.where(
            denominator > 0.0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("-inf"),
        )
        tl.store(O + output_row[:, None] * 512 + d[None, :], result, live_row[:, None])
        tl.store(LSE + output_row, logsum, live_row)
    else:
        slot = output_row * SPLITS + split
        if begin < end:
            tl.store(
                PARTIAL_O + slot[:, None] * 512 + d[None, :], acc, live_row[:, None]
            )
        tl.store(PARTIAL_M + slot, maximum, live_row)
        tl.store(PARTIAL_L + slot, denominator, live_row)


@triton.jit
def _merge(
    PARTIAL_O,
    PARTIAL_M,
    PARTIAL_L,
    O,
    LSE,
    CU_Q,
    HEADS: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    MAX_Q: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
    RAGGED: tl.constexpr,
    VALUE_SCALE: tl.constexpr,
    BD: tl.constexpr,
):
    packed_row = tl.program_id(0)
    request = tl.program_id(1)
    if RAGGED:
        q_start = tl.load(CU_Q + request)
        q_count = tl.minimum(
            tl.maximum(tl.load(CU_Q + request + 1) - q_start, 0), MAX_Q
        )
    else:
        q_start = request * MAX_Q
        q_count = MAX_Q
    query_row = q_start + packed_row // HEADS
    live = (packed_row < q_count * HEADS) & (query_row >= 0) & (query_row < TOTAL_Q)
    row = query_row.to(tl.int64) * HEADS + packed_row % HEADS
    s = tl.arange(0, SPLITS_PAD)
    slot = row * SPLITS + s
    m = tl.load(PARTIAL_M + slot, live & (s < SPLITS), other=float("-inf"))
    l = tl.load(PARTIAL_L + slot, live & (s < SPLITS), other=0.0)
    active_s = live & (s < SPLITS) & (l > 0.0)
    maximum = tl.max(m, 0)
    safe_maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    weight = tl.exp2(m - safe_maximum)
    denominator = tl.sum(l * weight, 0)
    reciprocal = tl.div_rn(1.0, tl.where(denominator > 0.0, denominator, 1.0))
    for block in range(512 // BD):
        d = block * BD + tl.arange(0, BD)
        partial = tl.load(
            PARTIAL_O + slot[:, None] * 512 + d[None, :],
            active_s[:, None],
            other=0.0,
        )
        result = tl.sum(partial * weight[:, None], 0) * reciprocal * VALUE_SCALE
        tl.store(O + row * 512 + d, result, live)
    logsum = tl.where(
        denominator > 0.0,
        maximum * 0.6931471805599453 + tl.log(denominator),
        float("-inf"),
    )
    tl.store(LSE + row, logsum, live)
