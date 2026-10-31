"""Common uniform Trellis preparation backed by the served W4A16 execution API."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .config import TrellisCodebook, TrellisConfig
from .source import TrellisSource
from .weights import TrellisWeights


@dataclass(frozen=True, kw_only=True)
class MoEGeometry:
    num_experts: int
    hidden_size: int
    intermediate_size: int

    def __post_init__(self):
        for name in ("num_experts", "hidden_size", "intermediate_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, kw_only=True)
class ActivationSpec:
    mode: str
    nonlinearity: str
    io_dtype: torch.dtype
    rotation_dtype: torch.dtype | None = None

    def __post_init__(self):
        if self.mode != "a16" or self.nonlinearity not in {"silu", "situ"}:
            raise ValueError(
                "common Trellis execution requires A16 activations with SiLU or SiTU"
            )
        if self.io_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Trellis public I/O must be float16 or bfloat16")
        rotation = self.io_dtype if self.rotation_dtype is None else self.rotation_dtype
        if rotation not in (torch.float16, torch.bfloat16):
            raise ValueError("full-rotation arithmetic must be FP16 or BF16")


def plan_weights(*, source=None, geometry=None, activation, **kwargs):
    from . import _impl

    if source is None:
        if geometry is not None or isinstance(activation, ActivationSpec):
            raise ValueError(
                "typed geometry/activation requires a common Trellis source"
            )
        return _impl.plan_b12x_fp4_moe_weights(activation=activation, **kwargs)
    if isinstance(source, TrellisConfig):
        if source.codebook is not TrellisCodebook.LUT_E4M3:
            raise NotImplementedError("declare uniform_bits through TrellisSource")
        source = TrellisSource(config=source, uniform_bits=3)
    if not isinstance(source, TrellisSource):
        raise TypeError("source must be TrellisSource or TrellisConfig")
    if not isinstance(geometry, MoEGeometry) or not isinstance(
        activation, ActivationSpec
    ):
        raise TypeError("common preparation requires MoEGeometry and ActivationSpec")
    rotation = (
        activation.io_dtype
        if activation.rotation_dtype is None
        else activation.rotation_dtype
    )
    if (
        rotation == torch.bfloat16
        and source.config.codebook is not TrellisCodebook.LUT_E4M3
    ):
        raise NotImplementedError(
            "BF16 full rotation requires E4M3 codewords to preserve decoded weights"
        )
    if source.uniform_bits is None:
        raise NotImplementedError(
            "this execution branch requires a declared uniform rate"
        )
    if (
        source.extent is not None
        and source.extent.intermediate_size != geometry.intermediate_size
    ):
        raise ValueError("source extent differs from the local intermediate width")
    projection = source.config.transform.projection
    if projection.kind != "scaled_hadamard" or projection.block_size != 128:
        raise NotImplementedError("common preparation requires scaled_hadamard(128)")
    expert = source.config.transform.expert
    coupled = expert.kind == "intermediate_hadamard"
    if geometry.hidden_size % 128 or geometry.intermediate_size % 128:
        raise ValueError("served full-rotation geometry must be aligned to 128")
    if coupled:
        if expert.pre_block_size != 512 or expert.post_block_size != 128:
            raise NotImplementedError(
                "served intermediate transforms require 512/128 blocks"
            )
        if geometry.hidden_size % 512:
            raise ValueError(
                "coupled full rotation requires hidden_size divisible by 512"
            )
    tile = kwargs.pop(
        "trellis_tile_config", (128, 128, 128, 128) if coupled else (64, 256, 64, 256)
    )
    if kwargs:
        raise TypeError(f"unsupported common preparation arguments: {sorted(kwargs)}")
    plan = _impl.plan_b12x_fp4_moe_weights(
        quant_modes="w4a16",
        source_format="b12x_trellis",
        activation=activation.nonlinearity,
        params_dtype=activation.io_dtype,
        num_experts=geometry.num_experts,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        trellis_bits=source.uniform_bits,
        trellis_tile_config=tile,
        coupled_hadamard=coupled,
        trellis_source=source,
    )
    return replace(plan, rotation_dtype=str(rotation).removeprefix("torch."))


def prepare_weights(*, plan, weights=None, device=None, staging=None, **kwargs):
    from b12x.moe._shared.execution import PreparedWeightLayout
    from . import _impl
    from .trellis import prepare_trellis_weights

    if plan.source_format != "b12x_trellis":
        if weights is not None or device is not None or staging is not None:
            raise ValueError("common weight tensors require a common Trellis plan")
        return _impl.prepare_b12x_fp4_moe_weights(plan=plan, **kwargs)
    if not isinstance(weights, TrellisWeights):
        raise TypeError("common preparation requires TrellisWeights")
    if kwargs:
        raise TypeError(f"unsupported common preparation arguments: {sorted(kwargs)}")
    value = prepare_trellis_weights(
        plan.trellis_source,
        weights,
        activation=plan.activation,
        params_dtype=getattr(torch, plan.rotation_dtype),
        num_experts=plan.num_experts,
        hidden_size=plan.hidden_size,
        intermediate_size=plan.intermediate_size,
        device=device,
        staging=staging,
        tile_config=plan.trellis_tile_config,
    )
    unit = torch.ones((), dtype=torch.float32, device=value.w13.device)
    return _impl.B12XFP4ExpertWeights(
        plan=plan,
        a1_gscale=unit,
        a2_gscale=unit,
        w1_fp4=value.w13,
        w1_blockscale=value.w13_scale,
        w1_alphas=value.w13_global_scale,
        w2_fp4=value.w2,
        w2_blockscale=value.w2_scale,
        w2_alphas=value.w2_global_scale,
        representation=_impl._PreparedWeightRepresentation(
            quant_mode="w4a16",
            layout=PreparedWeightLayout.TRELLIS_NATIVE,
            value=value,
        ),
    )
