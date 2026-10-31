"""Container-independent tensors for common Trellis preparation."""

from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ScaleFactors:
    """One scale boundary represented as vectors times optional gains."""

    vectors: torch.Tensor
    gains: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.vectors, torch.Tensor):
            raise TypeError("ScaleFactors.vectors must be a torch.Tensor")
        if self.gains is not None and not isinstance(self.gains, torch.Tensor):
            raise TypeError("ScaleFactors.gains must be a torch.Tensor or None")


@dataclass(frozen=True)
class TrellisWeights:
    """Container-independent layer-local trellis tensors.

    ``codes`` is the rank-local ``[I_local/32, payload_bytes]`` uint8 payload,
    or ``[I_local/32, experts, codeword_bytes_per_expert]`` with an explicit
    expert axis. Physical strides may include padding outside logical bytes.
    ``rate`` is a view selected from the single model-level uint8 rate tensor;
    it is never copied merely to give each layer its own rate parameter.
    CPU payloads may be prepared onto an explicitly selected CUDA device.
    Row padding is removed and checked by the checkpoint adapter, not by the
    execution planner. All scale vectors and gains retain FP16 storage.
    """

    codes: torch.Tensor
    rate: torch.Tensor
    input_scales: ScaleFactors
    intermediate_scales: ScaleFactors
    output_scales: ScaleFactors
    expert_sign_patterns: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.codes, torch.Tensor):
            raise TypeError("TrellisWeights.codes must be a torch.Tensor")
        if not isinstance(self.rate, torch.Tensor):
            raise TypeError("TrellisWeights.rate must be a torch.Tensor")
        for name in (
            "input_scales",
            "intermediate_scales",
            "output_scales",
        ):
            if not isinstance(getattr(self, name), ScaleFactors):
                raise TypeError(f"TrellisWeights.{name} must be ScaleFactors")
        if self.expert_sign_patterns is not None and not isinstance(
            self.expert_sign_patterns, torch.Tensor
        ):
            raise TypeError(
                "TrellisWeights.expert_sign_patterns must be a tensor or None"
            )
