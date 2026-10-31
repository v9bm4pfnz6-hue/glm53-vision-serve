# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Kimi-K3 multimodal preprocessing (images and videos)."""

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k3 import KimiK3Processor
from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
    KimiK25FusedVisionProcessor,
)
from vllm.utils.import_utils import is_numba_available

logger = init_logger(__name__)


_PATCH_LIMIT_KEYS = (
    "in_patch_limit",
    "patch_limit_on_one_side",
    "in_patch_limit_video",
)

_VIDEO_PLACEHOLDER = "<|kimi_video_placeholder|>"
_VIDEO_CHUNK_TYPE = "video_chunk"


def _timestamp_as_str(timestamp: float, timestamp_mode: str) -> str:
    """Render a video chunk start time in the checkpoint's timestamp_mode."""
    if timestamp_mode == "hh:mm:ss.fff":
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%H:%M:%S"
        ) + f".{int((timestamp % 1) * 1000):03d}"
    if timestamp_mode == "mm:ss.fff":
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%M:%S"
        ) + f".{int((timestamp % 1) * 1000):03d}"
    if timestamp_mode == "mm:ss":
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%M:%S")
    raise ValueError(f"Invalid timestamp mode: {timestamp_mode}")


def _unwrap_video_item(video: Any) -> tuple[Any, Mapping[str, Any] | None]:
    """Return (frames, metadata) from a parsed video item."""
    if isinstance(video, MediaWithBytes):
        return _unwrap_video_item(video.media)
    if isinstance(video, tuple):
        return video
    return video, None


def _frame_to_pil(frame: Any) -> PILImage.Image:
    if isinstance(frame, PILImage.Image):
        return frame if frame.mode == "RGB" else frame.convert("RGB")
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if isinstance(frame, np.ndarray):
        return PILImage.fromarray(frame).convert("RGB")
    raise ValueError(f"Unsupported video frame type: {type(frame)}")


def _video_to_chunk_medias(
    video: Any,
    media_proc_cfg: Mapping[str, Any],
    image_processor: Any = None,
) -> list[dict[str, Any]]:
    """Split a parsed video item into canonical MoonViT video chunks.

    Frames are sampled at the checkpoint's ``sample_fps`` (capped by the
    source fps) and grouped into ``temporal_merge_kernel_size``-frame chunks;
    each chunk carries its start timestamp in ``timestamp_mode``. The MoonViT
    tower accepts at most ``init_pos_emb_time`` (4) frames per grid row, so
    one video becomes one or more chunk media dicts. Total frames are capped
    to the checkpoint's per-video patch budget (``in_patch_limit_video``);
    oversized videos are downsampled uniformly to keep full-duration
    coverage.
    """
    frames, metadata = _unwrap_video_item(video)
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError("Video item has no frames")

    sample_fps = media_proc_cfg["sample_fps"]
    temporal_merge = int(media_proc_cfg["temporal_merge_kernel_size"])
    timestamp_mode = media_proc_cfg["timestamp_mode"]

    src_fps = (metadata or {}).get("fps")
    src_total = (metadata or {}).get("total_num_frames") or total_frames
    if isinstance(src_fps, (int, float)) and src_fps > 0 and src_total == total_frames:
        # Full-frame decode: sample indices at sample_fps on the source timeline.
        effective_fps = min(sample_fps, src_fps)
        sampled = max(round(total_frames * effective_fps / src_fps), 1)
        indices = np.linspace(0, total_frames - 1, sampled).round().astype(int)
        start_times = indices / src_fps
    else:
        # Pre-sampled (loader-side fps/num_frames policy) or metadata-free
        # input: frames are already the sample. Spread timestamps uniformly
        # over the source duration when known so chunk times stay on the
        # source timeline regardless of the loader's sampling rate.
        indices = np.arange(total_frames)
        duration = (metadata or {}).get("duration")
        if isinstance(duration, (int, float)) and duration > 0 and total_frames > 1:
            start_times = indices * (duration / (total_frames - 1))
        else:
            start_times = indices / sample_fps

    frame_budget = media_proc_cfg.get("in_patch_limit_video")
    if frame_budget and image_processor is not None and len(indices) > 0:
        probe = _frame_to_pil(frames[indices[0]])
        tokens_per_frame = image_processor.media_tokens_calculator(
            {"type": _VIDEO_CHUNK_TYPE, "video_chunk": [probe]}
        )
        merge = int(media_proc_cfg["merge_kernel_size"])
        patches_per_frame = tokens_per_frame * merge * merge
        frames_cap = max(temporal_merge, int(frame_budget) // max(patches_per_frame, 1))
        if len(indices) > frames_cap:
            keep = np.linspace(0, len(indices) - 1, frames_cap).round().astype(int)
            indices = indices[keep]
            start_times = start_times[keep]
            logger.info_once(
                "Kimi-K3 video exceeds the in_patch_limit_video budget; "
                "downsampled to %d frames.",
                frames_cap,
            )

    pil_frames = [_frame_to_pil(frames[i]) for i in indices]
    chunks = []
    for start in range(0, len(pil_frames), temporal_merge):
        chunks.append(
            {
                "type": _VIDEO_CHUNK_TYPE,
                "video_chunk": pil_frames[start : start + temporal_merge],
                "timestamp": _timestamp_as_str(
                    float(start_times[start]), timestamp_mode
                ),
            }
        )
    return chunks


def _apply_image_patch_limits(
    image_processor: Any,
    processor_kwargs: Mapping[str, object],
) -> Any:
    """Return a processor with deployment-wide Kimi media limits applied.

    Kimi's image processor resizes inputs to ``in_patch_limit`` and
    ``patch_limit_on_one_side``; ``in_patch_limit_video`` is the per-video
    patch budget the video chunker enforces. Restricting those checkpoint
    defaults bounds both the startup memory profile and request-time vision
    activations. The cached checkpoint processor is copied so another model
    configuration in the same process cannot inherit these deployment-specific
    limits.
    """
    overrides = {
        key: processor_kwargs[key]
        for key in _PATCH_LIMIT_KEYS
        if key in processor_kwargs
    }
    if not overrides:
        return image_processor

    media_proc_cfg = getattr(image_processor, "media_proc_cfg", None)
    if not isinstance(media_proc_cfg, Mapping):
        raise ValueError(
            "Kimi-K3 image patch limits require an image processor with "
            "a media_proc_cfg mapping"
        )

    bounded_cfg = dict(media_proc_cfg)
    for key, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Kimi-K3 {key} must be a positive integer")
        checkpoint_limit = bounded_cfg.get(key)
        if not isinstance(checkpoint_limit, int) or checkpoint_limit <= 0:
            raise ValueError(
                f"Kimi-K3 checkpoint does not define a positive {key} limit"
            )
        if value > checkpoint_limit:
            raise ValueError(
                f"Kimi-K3 {key}={value} exceeds the checkpoint limit "
                f"of {checkpoint_limit}"
            )
        bounded_cfg[key] = value

    bounded_processor = deepcopy(image_processor)
    bounded_processor.media_proc_cfg = bounded_cfg
    logger.info(
        "Kimi-K3 media processing limits: in_patch_limit=%d, "
        "patch_limit_on_one_side=%d, in_patch_limit_video=%d",
        bounded_cfg["in_patch_limit"],
        bounded_cfg["patch_limit_on_one_side"],
        bounded_cfg["in_patch_limit_video"],
    )
    return bounded_processor


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
):
    # Apply the patch limits.
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, patch_limit_on_one_side * patch_size)
    new_h = min(new_h, patch_limit_on_one_side * patch_size)

    factor = merge_kernel_size * patch_size

    pad_height = (factor - new_h % factor) % factor
    pad_width = (factor - new_w % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        # Calculate new dimensions after padding and patching
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor

        assert token_height * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_height {token_height} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )
        assert token_width * merge_kernel_size <= patch_limit_on_one_side, (
            f"token_width {token_width} * merge_kernel_size {merge_kernel_size} > "
            f"patch_limit_on_one_side {patch_limit_on_one_side}"
        )

        num_tokens = token_height * token_width
    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


class KimiK3ProcessingInfo(BaseProcessingInfo):
    """Processing information for the Kimi-K3 model.

    K3 uses the standard ``image`` modality for images (unlike K2.5's unified
    ``vision_chunk``), so it builds its own ``KimiK3Processor`` wrapper around
    the checkpoint's image processor and resolves the ``<|media_pad|>`` token
    id the same way K2.5 does. Videos use the standard ``video`` modality:
    parsed video items are sampled and split into MoonViT chunks by
    ``_video_to_chunk_medias`` (see the module helpers).
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)

        self.hf_config = hf_config = self.get_hf_config()

        tokenizer = self.get_tokenizer()
        processor_cls = KimiK25FusedVisionProcessor if is_numba_available() else None
        logger.info_once(
            "Using %s image preprocessing for Kimi-K3 images.",
            "fused CPU" if processor_cls is not None else "remote HF",
        )
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
            processor_cls_overrides=processor_cls,
        )
        get_mm_config = getattr(self.ctx.model_config, "get_multimodal_config", None)
        if get_mm_config is not None:
            mm_config = get_mm_config()
            processor_kwargs = mm_config.mm_processor_kwargs or {}
            image_processor = _apply_image_patch_limits(
                image_processor,
                processor_kwargs,
            )

        # Resolve token ID from the tokenizer because transformers v5
        # may remap token IDs vs config.json.
        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|media_pad|>")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            unk_token_id is None or resolved_token_id != unk_token_id
        )
        if is_valid_resolved and resolved_token_id != config_token_id:
            logger.warning_once(
                "Kimi-K3 config.media_placeholder_token_id (%d) disagrees "
                "with tokenizer mapping for <|media_pad|> (%d). "
                "Using tokenizer value.",
                config_token_id,
                resolved_token_id,
            )
            media_token_id = resolved_token_id
            # Patch config so downstream code also sees the correct ID.
            hf_config.media_placeholder_token_id = resolved_token_id
        else:
            media_token_id = config_token_id

        self.media_token_id = media_token_id
        self.media_token = tokenizer.decode(media_token_id)

        self.image_processor = image_processor
        self.hf_processor = KimiK3Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_processor(self, **kwargs: object) -> KimiK3Processor:
        return self.hf_processor

    def get_hf_config(self) -> KimiK3Config:
        return self.ctx.get_hf_config(KimiK3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # None means unlimited
        return {"image": None, "video": None}

    def get_data_parser(self) -> MultiModalDataParser:
        # Video items keep their container metadata (source fps and frame
        # count) so the processor can sample frames and timestamp chunks.
        return MultiModalDataParser(video_needs_metadata=True)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        """Declare per-item token maximums analytically.

        Returning values here keeps the encoder-cache budget independent of
        the profiling dummy: the checkpoint's per-video budget is far larger
        than the image budget, and a dummy video at that budget makes the
        startup memory probe demand several GiB of transient vision
        activations, which the 0.97 memory-utilization profile cannot spare.
        """
        media_proc_cfg = self.image_processor.media_proc_cfg
        merge = media_proc_cfg["merge_kernel_size"]
        if isinstance(merge, (list, tuple)):
            merge = merge[0]
        spatial_merge = int(merge) * int(merge)
        temporal_merge = int(media_proc_cfg["temporal_merge_kernel_size"])
        image_max = self.get_max_image_size(
            media_proc_cfg["patch_size"],
            media_proc_cfg["merge_kernel_size"],
            media_proc_cfg["in_patch_limit"],
            media_proc_cfg["patch_limit_on_one_side"],
            media_proc_cfg["fixed_output_tokens"],
        )
        image_tokens = self.media_tokens_calculator(
            {
                "type": "image",
                "image": PILImage.new("RGB", (image_max.width, image_max.height)),
            }
        )
        video_tokens = int(media_proc_cfg["in_patch_limit_video"]) // (
            spatial_merge * temporal_merge
        )
        return {"image": int(image_tokens), "video": int(video_tokens)}

    @classmethod
    def get_max_image_size(
        cls,
        patch_size: int,
        merge_kernel_size: int,
        in_patch_limit: int,
        patch_limit_on_one_side: int,
        fixed_output_tokens: int | None,
    ) -> ImageSize:
        max_side = patch_limit_on_one_side * patch_size
        best_score = (-1, -1)
        best_size = (max_side, max_side)

        for width_patches in range(patch_limit_on_one_side + 1):
            width = min((width_patches + 1) * patch_size - 1, max_side)
            for height_patches in range(width_patches, patch_limit_on_one_side + 1):
                height = min((height_patches + 1) * patch_size - 1, max_side)
                resize_config = navit_resize_image(
                    width,
                    height,
                    patch_size,
                    merge_kernel_size,
                    in_patch_limit,
                    patch_limit_on_one_side,
                    fixed_output_tokens,
                )
                padded_width = resize_config["new_width"] + resize_config["pad_width"]
                padded_height = (
                    resize_config["new_height"] + resize_config["pad_height"]
                )
                num_patches = padded_width // patch_size * (padded_height // patch_size)
                score = (resize_config["num_tokens"], num_patches)
                if score > best_score:
                    best_score = score
                    best_size = (width, height)
        return ImageSize(width=best_size[0], height=best_size[1])


class KimiK3DummyInputsBuilder(BaseDummyInputsBuilder[KimiK3ProcessingInfo]):
    """Builds image- and video-based dummy inputs for K3 profiling.

    The dummy text is made of ``<|kimi_image_placeholder|>`` and
    ``<|kimi_video_placeholder|>`` markers — exactly the placeholders that
    K3's ``_get_prompt_updates`` expands. Image dummy data is a plain list of
    PIL images under the ``image`` key; video dummy data is a list of
    ``(frames, metadata)`` tuples under the ``video`` key, each holding one
    temporal-merge-sized chunk of frames at the per-frame patch cap
    (metadata is required by the video-needs-metadata parser). The
    encoder-cache budget is declared analytically by
    ``KimiK3ProcessingInfo.get_mm_max_tokens_per_item``, so this dummy only
    exercises the vision path within the startup memory-probe headroom.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        return (
            self.info.get_hf_config().image_placeholder * num_images
            + _VIDEO_PLACEHOLDER * num_videos
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        media_proc_cfg = self.info.image_processor.media_proc_cfg
        max_size = self.info.get_max_image_size(
            media_proc_cfg["patch_size"],
            media_proc_cfg["merge_kernel_size"],
            media_proc_cfg["in_patch_limit"],
            media_proc_cfg["patch_limit_on_one_side"],
            media_proc_cfg["fixed_output_tokens"],
        )
        num_images = mm_counts.get("image", 0)
        image_overrides = cast(
            ImageDummyOptions | None,
            mm_options.get("image") if mm_options else None,
        )
        mm_data: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=max_size.width,
                height=max_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }

        num_videos = mm_counts.get("video", 0)
        if num_videos:
            temporal_merge = int(media_proc_cfg["temporal_merge_kernel_size"])
            frame_size = self.info.get_max_image_size(
                media_proc_cfg["patch_size"],
                media_proc_cfg["merge_kernel_size"],
                media_proc_cfg["in_patch_limit_each_frame"],
                media_proc_cfg["patch_limit_on_one_side"],
                media_proc_cfg["fixed_output_tokens"],
            )
            # One chunk of frames at the per-frame cap: the encoder-cache
            # budget comes from `get_mm_max_tokens_per_item`, so the dummy
            # only has to exercise the vision path within the startup
            # memory-probe headroom.
            max_frames = temporal_merge
            videos = []
            sample_fps = float(media_proc_cfg["sample_fps"])
            for _ in range(num_videos):
                frames = self._get_dummy_images(
                    width=frame_size.width,
                    height=frame_size.height,
                    num_images=max_frames,
                    overrides=None,
                )
                metadata = {
                    "fps": sample_fps,
                    "total_num_frames": max_frames,
                    "duration": max_frames / sample_fps,
                }
                videos.append((np.stack([np.asarray(f) for f in frames]), metadata))
            mm_data["video"] = videos
        return mm_data


class KimiK3MultiModalProcessor(BaseMultiModalProcessor[KimiK3ProcessingInfo]):
    """Multi-modal processor for Kimi-K3 images and videos."""

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Override so the base always routes through the text+mm path
        # (`KimiK3Processor.__call__`). Otherwise the mm-only fast path calls
        # the checkpoint image processor directly with bare PIL images, but it
        # requires `{"type": "image", "image": PIL}` media dicts that only our
        # wrapper builds.
        #
        # Videos are split into MoonViT frame chunks and preprocessed by the
        # same (fused) vision processor; their tensors land in separate
        # `*_videos` keys because one HF output key maps to exactly one
        # modality downstream.
        videos = list(mm_data.get("videos") or [])
        mm_data_images_only = {
            key: value for key, value in mm_data.items() if key != "videos"
        }
        hf_inputs = super()._call_hf_processor(
            prompt, mm_data_images_only, mm_kwargs, tok_kwargs
        )
        if videos:
            media_proc_cfg = self.info.image_processor.media_proc_cfg
            chunk_medias = []
            chunk_counts = []
            for video in videos:
                chunks = _video_to_chunk_medias(
                    video, media_proc_cfg, self.info.image_processor
                )
                chunk_medias.extend(
                    {"type": chunk["type"], "video_chunk": chunk["video_chunk"]}
                    for chunk in chunks
                )
                chunk_counts.append(len(chunks))
            video_inputs = self.info.image_processor.preprocess(
                chunk_medias, return_tensors="pt"
            )
            pixel_values_videos = video_inputs.get("pixel_values")
            if pixel_values_videos is not None:
                hf_inputs["pixel_values_videos"] = pixel_values_videos
                hf_inputs["grid_thws_videos"] = video_inputs["grid_thws"]
                hf_inputs["video_chunk_counts"] = torch.tensor(
                    chunk_counts, dtype=torch.long
                )
        return hf_inputs

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Slice the flattened patch tensor back into per-image items.

        ``pixel_values`` holds all patches from every image concatenated; each
        image's patch count is ``prod(grid_thws[i])``. ``grid_thws`` is one
        ``[N_t, N_h, N_w]`` row per image. Video tensors live in the parallel
        ``*_videos`` keys; one video item spans ``video_chunk_counts[i]``
        consecutive chunk rows.
        """
        grid_thws = hf_inputs.get("grid_thws", torch.empty((0, 3)))
        grid_sizes = grid_thws.prod(-1)

        configs = dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid_sizes),
            grid_thws=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

        video_grid = hf_inputs.get("grid_thws_videos")
        chunk_counts = hf_inputs.get("video_chunk_counts")
        if video_grid is not None and chunk_counts is not None and len(chunk_counts):
            row_sizes = video_grid.prod(-1)
            patches_per_item = torch.tensor(
                [int(rows.sum()) for rows in row_sizes.split(chunk_counts.tolist())],
                dtype=torch.long,
            )
            configs["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes(
                "video", patches_per_item
            )
            configs["grid_thws_videos"] = MultiModalFieldConfig.flat_from_sizes(
                "video", chunk_counts, keep_on_cpu=True
            )
            configs["video_chunk_counts"] = MultiModalFieldConfig.flat_from_sizes(
                "video", torch.ones_like(chunk_counts), keep_on_cpu=True
            )
        return configs

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Expand each K3 media placeholder into a resolution-aware update.

        K3's prompt carries a single ``<|kimi_image_placeholder|>`` token per
        image. This replaces that token with
        ``<|media_begin|>image {w}x{h}<|media_content|>{pads}<|media_end|>``,
        embedding the per-image resolution in the prompt and marking only the
        ``<|media_pad|>`` positions as embedding slots (the number of pads is
        the feature size returned by ``media_tokens_calculator``).

        Each video placeholder ``<|kimi_video_placeholder|>`` expands to the
        canonical MoonViT chunk sequence
        ``{timestamp}<|media_begin|>video<|media_content|>{pads}<|media_end|>``
        per ≤4-frame chunk, concatenated in chunk order; chunk pad counts come
        from the same calculator on the chunk media dict.
        """
        media_token_id = self.info.media_token_id
        media_token = self.info.media_token
        image_placeholder = self.info.get_hf_config().image_placeholder

        def get_replacement(item_idx: int) -> PromptUpdateDetails[str]:
            images = mm_items.get_items("image", ImageProcessorItems)
            image = images.get(item_idx)
            if image is None:
                raise ValueError(f"Missing image data at index {item_idx}")

            # The checkpoint image processor works on media dicts, so wrap the
            # PIL image before asking it for the token count.
            num_media_token = self.info.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            pads = media_token * num_media_token

            # NOTE: `width`/`height` are the ORIGINAL upload dimensions, not the
            # post-preprocess (smart-resized) ones. `image` comes from the
            # untouched parsed `mm_items`; the checkpoint image processor
            # (`KimiK3VisionProcessor.preprocess`) only produces new tensors via
            # `image.resize(...)` and never mutates the stored PIL. This matches
            # the reference HF processor (`KimiK3Processor.preprocess_medias`),
            # which also builds the prompt from the original `img.size`. The
            # resize is reflected only in the pad count above.
            width, height = images.get_image_size(item_idx)
            full = (
                f"<|media_begin|>image {width}x{height}<|media_content|>"
                f"{pads}<|media_end|>"
            )

            return PromptUpdateDetails.select_token_id(full, media_token_id)

        def get_video_replacement(item_idx: int) -> PromptUpdateDetails[str]:
            videos = mm_items.get_items("video", VideoProcessorItems)
            video = videos.get(item_idx)
            if video is None:
                raise ValueError(f"Missing video data at index {item_idx}")

            media_proc_cfg = self.info.image_processor.media_proc_cfg
            chunks = _video_to_chunk_medias(
                video, media_proc_cfg, self.info.image_processor
            )
            parts = []
            for chunk in chunks:
                num_media_token = self.info.media_tokens_calculator(
                    {"type": _VIDEO_CHUNK_TYPE, "video_chunk": chunk["video_chunk"]}
                )
                parts.append(
                    f"{chunk['timestamp']}"
                    f"<|media_begin|>video<|media_content|>"
                    f"{media_token * num_media_token}<|media_end|>"
                )
            full = "".join(parts)

            return PromptUpdateDetails.select_token_id(full, media_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=image_placeholder,
                replacement=get_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=_VIDEO_PLACEHOLDER,
                replacement=get_video_replacement,
            ),
        ]
