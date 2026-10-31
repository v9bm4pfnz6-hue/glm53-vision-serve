"""Lossless CPU adapter from coupled K2 atom extents to common Trellis tensors."""

from __future__ import annotations

import torch

from b12x.moe.fused_moe.config import TrellisConfig
from b12x.moe.fused_moe.source import TrellisExtent, TrellisSource
from b12x.moe.fused_moe.weights import ScaleFactors, TrellisWeights


def trellis_from_qsrt_atoms_v2(
    atom_payload: torch.Tensor,
    *,
    first_atom_slot: int,
    num_experts: int,
    hidden_size: int,
    global_intermediate_size: int,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    down_svh: torch.Tensor,
    rotation_draws: torch.Tensor,
) -> tuple[TrellisSource, TrellisWeights]:
    """Expose codeword views and FP16 tables without decoding or GPU allocation.

    Atom records interleave codewords and scale vectors per expert. An explicit
    expert stride excludes those scale bytes without a second extent-sized CPU
    copy; common preparation stages only the logical codeword bundles.
    """
    if atom_payload.device.type != "cpu" or atom_payload.dtype != torch.uint8:
        raise ValueError("QSRT atom payload must be CPU uint8")
    if atom_payload.ndim != 2 or atom_payload.stride(1) != 1:
        raise ValueError("QSRT atom payload must contain contiguous logical rows")
    if type(first_atom_slot) is not int or first_atom_slot < 0:
        raise ValueError("first_atom_slot must be a nonnegative integer")
    if hidden_size <= 0 or hidden_size % 512 or num_experts <= 0:
        raise ValueError(
            "coupled QSRT requires a positive expert count and H % 512 == 0"
        )
    if global_intermediate_size <= 0 or global_intermediate_size % 256:
        raise ValueError(
            "coupled QSRT global intermediate width must be divisible by 256"
        )
    slots = atom_payload.shape[0]
    global_slots = global_intermediate_size // 32
    if slots <= 0 or first_atom_slot % 4 or slots % 4:
        raise ValueError("coupled QSRT extents must close 128-channel blocks")
    if first_atom_slot + slots > global_slots:
        raise ValueError("QSRT extent exceeds the global intermediate axis")
    half = global_slots // 2
    if first_atom_slot < half < first_atom_slot + slots:
        raise ValueError("QSRT extent crosses the FC1 side-table boundary")
    for name, value in (
        ("gate_suh", gate_suh),
        ("up_suh", up_suh),
        ("down_svh", down_svh),
    ):
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float16
            or value.shape != (hidden_size,)
        ):
            raise ValueError(f"{name} must be CPU FP16 [hidden_size]")
    if (
        rotation_draws.device.type != "cpu"
        or rotation_draws.dtype != torch.uint8
        or rotation_draws.shape != (num_experts,)
        or bool(torch.any(rotation_draws > 7))
    ):
        raise ValueError("rotation_draws must be CPU uint8[experts] in 0..7")
    section = (hidden_size // 16) * 64 * 2
    bundle_bytes = 3 * section + 3 * 32 * 2
    payload_bytes = num_experts * bundle_bytes
    if atom_payload.shape[1] < payload_bytes:
        raise ValueError("QSRT row is shorter than the declared expert bundles")
    if bool(torch.any(atom_payload[:, payload_bytes:] != 0)):
        raise ValueError("QSRT row padding must be zero")
    bundles = atom_payload[:, :payload_bytes].unflatten(1, (num_experts, bundle_bytes))
    codes = bundles[:, :, : 3 * section]
    scales = bundles[:, :, 3 * section :].contiguous().view(torch.float16)
    intermediate = (
        scales.reshape(slots, num_experts, 3, 32)
        .permute(1, 2, 0, 3)
        .reshape(num_experts, 3, slots * 32)
        .contiguous()
    )
    selected_input = gate_suh if first_atom_slot < half else up_suh
    config = TrellisConfig.from_dict(
        {
            "version": 2,
            "codebook": "lut_e4m3",
            "rate": {"granularity": "uniform"},
            "scale": {
                "input_scales": {"vectors": "per_layer", "gains": "none"},
                "intermediate_scales": {"vectors": "per_expert", "gains": "none"},
                "output_scales": {"vectors": "per_layer", "gains": "none"},
            },
            "transform": {
                "projection": {"kind": "scaled_hadamard", "block_size": 128},
                "expert": {
                    "kind": "intermediate_hadamard",
                    "pre_block_size": 512,
                    "post_block_size": 128,
                    "sign_pattern_granularity": "per_expert",
                },
            },
        }
    )
    return TrellisSource(
        config=config,
        uniform_bits=2,
        extent=TrellisExtent(
            global_intermediate_size=global_intermediate_size,
            first_slot=first_atom_slot,
            slot_count=slots,
        ),
    ), TrellisWeights(
        codes=codes,
        rate=torch.tensor([0x22], dtype=torch.uint8),
        input_scales=ScaleFactors(selected_input.contiguous()),
        intermediate_scales=ScaleFactors(intermediate),
        output_scales=ScaleFactors(down_svh.contiguous()),
        expert_sign_patterns=rotation_draws,
    )
