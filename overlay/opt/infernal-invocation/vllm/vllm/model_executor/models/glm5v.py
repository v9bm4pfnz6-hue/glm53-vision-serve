# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Vision: MoonViT tower + PatchMerger projector + GLM-5.3 text model.

The checkpoint keeps the official GLM-5.3 text layout (``model.layers.*``,
``model.embed_tokens``, ``model.norm``, ``lm_head``) next to
``vision_tower.*`` and ``mm_projector.*`` tensors, so the model is the in-tree
``KimiK25ForConditionalGeneration`` with GLM-specific weight naming, the GLM
image placeholder tokens, an unquantized (BF16) tower and projector, and a
weight loader that zero-pads the tower to the virtual-TP geometry.
"""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.models.kimi_k25 import (
    KimiK25DummyInputsBuilder,
    KimiK25ForConditionalGeneration,
    KimiK25MultiModalProcessor,
    KimiK25ProcessingInfo,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.virtual_tp import get_virtual_tp_axis_padded_size
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.configs.glm5v import Glm5vConfig

logger = init_logger(__name__)

GLM5V_IMAGE_PLACEHOLDER = "<|begin_of_image|><|image|><|end_of_image|>"


class Glm5vProcessingInfo(KimiK25ProcessingInfo):
    def get_hf_config(self) -> Glm5vConfig:
        return self.ctx.get_hf_config(Glm5vConfig)


def _pad_dim(
    tensor: torch.Tensor, dim: int, original: int, padded: int
) -> torch.Tensor:
    if tensor.shape[dim] != original:
        raise ValueError(
            f"Expected vision tensor size {original} on dim {dim}, "
            f"got {tuple(tensor.shape)}"
        )
    if padded == original:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = padded
    out = tensor.new_zeros(shape)
    out.narrow(dim, 0, original).copy_(tensor)
    return out


def _pad_fused_qkv(tensor: torch.Tensor, original: int, padded: int) -> torch.Tensor:
    if tensor.shape[0] != 3 * original:
        raise ValueError(
            f"Expected fused QKV size {3 * original} on dim 0, "
            f"got {tuple(tensor.shape)}"
        )
    parts = tensor.split(original, dim=0)
    return torch.cat([_pad_dim(part, 0, original, padded) for part in parts], dim=0)


def keep_bare_ignored_layers(ignored_layers: list[str] | None) -> list[str]:
    """Keep both the mapped and the checkpoint spelling of FP8 ignore entries.

    ``SupportsQuant.__new__`` rewrites the checkpoint's ignore list through the
    wrapper's mapper ("model.layers.N.mlp.gate" becomes
    "language_model.model.layers.N.mlp.gate"). The MTP draft is built from the
    same quantization config under its own prefix ("model.layers.78..."), so
    the bare names must survive. The wrapper's modules all live under
    ``language_model.``, ``vision_tower.`` or ``mm_projector.``, so a bare name
    can never match a target module.
    """
    ignored = list(ignored_layers or [])
    bare = [
        name.removeprefix("language_model.")
        for name in ignored
        if name.startswith("language_model.")
    ]
    return list(dict.fromkeys(ignored + bare))


@MULTIMODAL_REGISTRY.register_processor(
    KimiK25MultiModalProcessor,
    info=Glm5vProcessingInfo,
    dummy_inputs=KimiK25DummyInputsBuilder,
)
class Glm5vForConditionalGeneration(KimiK25ForConditionalGeneration):
    # The text shards use the standalone GLM-5.3 names; vLLM nests the decoder
    # under ``language_model``. Tower and projector names already match.
    # "lm_head" without a dot so the quantization ignore entry "lm_head" maps
    # together with "lm_head.weight".
    hf_to_vllm_mapper = KimiK25ForConditionalGeneration.hf_to_vllm_mapper | (
        WeightsMapper(
            orig_to_new_prefix={
                "model.": "language_model.model.",
                "lm_head": "language_model.lm_head",
            }
        )
    )

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        quant_config = vllm_config.quant_config
        if isinstance(quant_config, Fp8Config):
            quant_config.ignored_layers = keep_bare_ignored_layers(
                quant_config.ignored_layers
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality in ("image", "vision_chunk"):
            return GLM5V_IMAGE_PLACEHOLDER
        return super().get_placeholder_str(modality, i)

    def _maybe_ignore_quant_config(self, quant_config: QuantizationConfig):
        # The FP8 block scales describe only the GLM text linears; the tower
        # and projector tensors are BF16 without scales and run unquantized.
        if isinstance(quant_config, Fp8Config):
            return None
        return super()._maybe_ignore_quant_config(quant_config)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return super().load_weights(self._pad_vision_weights(weights))

    def _pad_vision_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Zero-pad tower/projector tensors to the virtual-TP geometry.

        Padded attention heads get zero QKV rows and zero output-projection
        columns; padded MLP channels get zero first-linear rows (and bias) and
        zero second-linear columns, so every padded unit contributes exactly
        zero and the per-rank shards line up with the standard loaders.
        """
        vision_config = self.config.vision_config
        heads = int(vision_config.num_attention_heads)
        qkv = int(
            getattr(vision_config, "qkv_hidden_size", None) or vision_config.hidden_size
        )
        original_qkv = int(getattr(vision_config, "original_qkv_hidden_size", qkv))
        inter = int(vision_config.intermediate_size)
        original_inter = int(
            getattr(vision_config, "original_intermediate_size", inter)
        )
        merge_h, merge_w = vision_config.merge_kernel_size
        projector_hidden = int(vision_config.hidden_size) * int(merge_h) * int(merge_w)
        padded_projector_hidden = get_virtual_tp_axis_padded_size(
            "vision_projector_hidden_size", projector_hidden, config=vision_config
        )
        if (qkv, inter, padded_projector_hidden) == (
            original_qkv,
            original_inter,
            projector_hidden,
        ):
            yield from weights
            return
        logger.info_once(
            "glm5v vision padding: heads %d (qkv %d -> %d), MLP %d -> %d, "
            "projector %d -> %d",
            heads,
            original_qkv,
            qkv,
            original_inter,
            inter,
            projector_hidden,
            padded_projector_hidden,
        )
        for name, tensor in weights:
            if name.startswith("vision_tower.encoder.blocks."):
                _, module, kind = name.rsplit(".", 2)
                if module == "wqkv":
                    tensor = _pad_fused_qkv(tensor, original_qkv, qkv)
                elif module == "wo" and kind == "weight":
                    tensor = _pad_dim(tensor, 1, original_qkv, qkv)
                elif module == "fc0":
                    tensor = _pad_dim(tensor, 0, original_inter, inter)
                elif module == "fc1" and kind == "weight":
                    tensor = _pad_dim(tensor, 1, original_inter, inter)
            elif name.startswith("mm_projector."):
                _, module, kind = name.rsplit(".", 2)
                if module == "linear_1":
                    tensor = _pad_dim(
                        tensor, 0, projector_hidden, padded_projector_hidden
                    )
                elif module == "linear_2" and kind == "weight":
                    tensor = _pad_dim(
                        tensor, 1, projector_hidden, padded_projector_hidden
                    )
            yield name, tensor
