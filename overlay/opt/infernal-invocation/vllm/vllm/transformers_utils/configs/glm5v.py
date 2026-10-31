# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Vision (``glm5v``) configuration.

The checkpoint is the official ``glm_moe_dsa`` GLM-5.3 text model with a MoonViT
tower and a PatchMerger projector attached (``Glm5vForConditionalGeneration``).
The layout mirrors ``KimiK25Config`` (``vision_config`` + ``text_config`` +
``media_placeholder_token_id``) so the Kimi-K2.5 multimodal path can be reused;
the text configuration is the transformers-native ``GlmMoeDsaConfig`` built from
the nested ``text_config`` dictionary, which is byte-identical to the standalone
GLM-5.3 ``config.json``.
"""

from typing import Any

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from vllm.transformers_utils.configs.kimi_k25 import KimiK25VisionConfig

GLM5V_IMAGE_TOKEN_ID = 154854  # GLM tokenizer ``<|image|>``
GLM5V_PAD_TOKEN_ID = 154820


class Glm5vVisionConfig(KimiK25VisionConfig):
    """MoonViT tower + PatchMerger projector configuration for glm5v.

    ``text_hidden_size`` is the projector output width (the GLM hidden size);
    ``vt_*`` duplicates of the tower fields arrive through ``**kwargs`` and are
    kept as plain attributes.
    """

    model_type = "glm5v_vision"

    def __init__(self, text_hidden_size: int = 6144, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.text_hidden_size = text_hidden_size


class Glm5vConfig(PretrainedConfig):
    model_type = "glm5v"
    sub_configs = {
        "vision_config": Glm5vVisionConfig,
        "text_config": AutoConfig,
    }

    def __init__(
        self,
        vision_config: dict | Glm5vVisionConfig | None = None,
        text_config: dict | PretrainedConfig | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = GLM5V_IMAGE_TOKEN_ID,
        pad_token_id: int = GLM5V_PAD_TOKEN_ID,
        use_unified_vision_chunk: bool = True,
        video_placeholder: str = "<|glm5v_video_placeholder|>",
        **kwargs: Any,
    ) -> None:
        if vision_config is None:
            self.vision_config = Glm5vVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = Glm5vVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if text_config is None:
            self.text_config = AutoConfig.for_model("glm_moe_dsa")
        elif isinstance(text_config, dict):
            text_kwargs = dict(text_config)
            text_kwargs.setdefault("model_type", "glm_moe_dsa")
            self.text_config = AutoConfig.for_model(**text_kwargs)
        else:
            self.text_config = text_config

        # The projector emits GLM hidden-size vectors (KimiK25Config pattern).
        if self.vision_config.mm_hidden_size == self.vision_config.hidden_size:
            self.vision_config.mm_hidden_size = self.text_config.hidden_size

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.use_unified_vision_chunk = use_unified_vision_chunk
        self.video_placeholder = video_placeholder

        # Only the GLM text linears are FP8; the tower and projector stay BF16.
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    def __getattr__(self, name: str) -> Any:
        # Text-model fields (DSA indexer geometry, layer counts, MLA ranks, ...)
        # resolve to the nested text configuration, so a reader holding the
        # wrapper configuration sees the same values as one holding the
        # standalone GLM-5.3 configuration. Dunder and private names and the
        # sub-configurations themselves are never forwarded.
        if name.startswith("_") or name in ("text_config", "vision_config"):
            raise AttributeError(name)
        try:
            text_config = object.__getattribute__(self, "__dict__")["text_config"]
        except KeyError:
            raise AttributeError(name) from None
        try:
            return getattr(text_config, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            ) from None


__all__ = ["Glm5vConfig", "Glm5vVisionConfig"]
