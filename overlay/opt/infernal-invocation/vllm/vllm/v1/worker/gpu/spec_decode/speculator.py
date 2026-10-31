# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.nucleus import DraftNucleus, draft_nucleus_enabled
from vllm.v1.worker.gpu.spec_decode.utils import (
    DRAFT_GUMBEL_POS_OFFSET,
    draft_gumbel_pos,
)
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.dspark.online_sts import DSparkOnlineSTS

logger = init_logger(__name__)

CUDAGraphCapturePhase = Literal["profile", "production"]


class BaseSpeculator(ABC):
    # Draft-token capacity surface, implemented by speculators with a
    # confidence head (see DSparkSpeculator).
    use_draft_token_capacity: bool = False
    online_sts: "DSparkOnlineSTS | None" = None
    wants_auto_sps_curve: bool = False

    def warmup_capacity_kernels(self) -> None:  # noqa: B027
        pass

    def set_sps_curve(self, sps_curve: list[tuple[int, float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass

    @abstractmethod
    def capture(self, *, capture_phase: CUDAGraphCapturePhase) -> None:
        pass

    def get_cudagraph_managers(self) -> tuple["CudaGraphManager", ...]:
        return ()

    def clear_cudagraphs(self) -> None:
        for manager in self.get_cudagraph_managers():
            manager.clear()

    @abstractmethod
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [num_prefill_lookahead, max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_speculative_tokens: int | None = None,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        pass

    def compute_capacities(self, input_batch: InputBatch) -> torch.Tensor | None:
        return None


class DraftModelSpeculator(BaseSpeculator):
    # Draft nucleus (top-p) truncation; off unless __init__ enables it.
    _draft_nucleus: DraftNucleus | None = None
    _draft_top_p_source: UvaBackedTensor | None = None

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.draft_max_seq_len = self.max_model_len
        self.rebuild_prefill_attn_metadata = False
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()
        # Widen for HC-multiplexed residuals (e.g. DeepSeek V4 feeds the MTP
        # draft the target's pre-hc_head (T, hc_mult * hidden_size) residual).
        # Non-HC models default to hc_mult=1 and are unaffected.
        hc_mult = getattr(self.draft_model_config.hf_config, "hc_mult", 1)
        self.hidden_size = self.hidden_size * hc_mult
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype
        self.use_fp64_gumbel = vllm_config.model_config.use_fp64_gumbel
        self.use_local_argmax_reduction = (
            self.speculative_config.use_local_argmax_reduction
        )

        # DP configuration
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.eplb_state: EplbState | None = None

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.temperature = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.seeds = torch.zeros(self.max_num_reqs, dtype=torch.int64, device=device)
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.arange = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device="cpu"
        )

        self.draft_logits: torch.Tensor | None = None
        if self.speculative_config.draft_sample_method == "probabilistic":
            # Pre-temperature logits, cached from the previous decode step.
            self.draft_logits = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.vocab_size,
                dtype=vllm_config.model_config.head_dtype,
                device=device,
            )

        # The per-request nucleus mass lives in the sampler's states; draft
        # graphs read this fixed-address copy.
        self.draft_top_p = torch.ones(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        if self.draft_logits is not None and draft_nucleus_enabled():
            self._draft_nucleus = DraftNucleus(
                self.max_num_reqs * self.num_speculative_steps, device
            )
            logger.info_once(
                "Draft proposals are sampled from their top-p nucleus "
                "(VLLM_K3_DRAFT_TOP_P)."
            )

        self._draft_logit_bias: torch.Tensor | None = None
        if envs.VLLM_K3_DRAFT_LOGIT_BIAS != 0.0:
            ids_path = envs.VLLM_K3_DRAFT_LOGIT_BIAS_IDS
            if not ids_path:
                raise ValueError(
                    "VLLM_K3_DRAFT_LOGIT_BIAS requires VLLM_K3_DRAFT_LOGIT_BIAS_IDS "
                    "to name a torch-saved int64 tensor of token ids."
                )
            ids = torch.load(ids_path, map_location="cpu", weights_only=True).to(
                torch.int64
            )
            if ids.ndim != 1 or ids.numel() == 0 or (ids < 0).any() or (
                ids >= self.vocab_size
            ).any():
                raise ValueError(
                    f"Draft logit bias ids from {ids_path} must be a non-empty "
                    f"1-D tensor within [0, {self.vocab_size})."
                )
            head_dtype = getattr(vllm_config.model_config, "head_dtype", None)
            bias = torch.zeros(
                self.vocab_size, dtype=head_dtype or torch.float32
            )
            bias[ids] = envs.VLLM_K3_DRAFT_LOGIT_BIAS
            self._draft_logit_bias = bias.to(device)
            logger.info_once(
                "Draft logit bias %s on %d token ids from %s.",
                envs.VLLM_K3_DRAFT_LOGIT_BIAS,
                ids.numel(),
                ids_path,
            )

        self._draft_temperature = envs.VLLM_K3_DRAFT_TEMPERATURE
        if self._draft_temperature != 1.0:
            logger.info_once(
                "Draft proposal logits scaled by 1/%s "
                "(VLLM_K3_DRAFT_TEMPERATURE).",
                self._draft_temperature,
            )

        self.supports_mm_inputs = False

    @abstractmethod
    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        pass

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        self.model = self.load_draft_model(target_model, target_attn_layer_names)
        self._validate_local_argmax_reduction()

        all_attn_layers = set[str](
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        self.draft_attn_layer_names = all_attn_layers - target_attn_layer_names

        target_supports_mm = MULTIMODAL_REGISTRY.supports_multimodal_inputs(
            self.vllm_config.model_config
        )
        draft_supports_mm = supports_multimodal_embeddings(self.model)
        self.supports_mm_inputs = target_supports_mm and draft_supports_mm
        if target_supports_mm and not draft_supports_mm:
            logger.warning_once(
                "Draft model %s does not support external multimodal embeddings. "
                "Embeddings from the target model will not be passed to the "
                "drafter; using text-only draft inputs instead.",
                type(self.model).__name__,
            )

    def set_eplb_state(self, eplb_state: EplbState) -> None:
        """Inject EPLB state after construction."""
        self.eplb_state = eplb_state

    def _prepare_eplb_forward(self, num_unpadded_tokens: int) -> None:
        """Call EPLB prepare_forward if EPLB is active for the draft model."""
        if self.eplb_state is not None:
            self.eplb_state.prepare_forward(
                self.speculative_config.draft_model_config,
                num_unpadded_tokens,
            )

    @property
    def attn_vllm_config(self) -> VllmConfig:
        """Config for the draft's attention metadata builders. Overridden by
        speculators whose attention mode differs from the target's."""
        return self.vllm_config

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.attn_groups, self.attn_cg_support, _ = init_attn_backend(
            kv_cache_config,
            self.attn_vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )
        self.block_tables = block_tables
        # The target model runner's buffers and attention groups. Draft
        # prefill reuses the target model's attention metadata, so its
        # cudagraph capture must build dummy metadata through the same
        # builders and buffers.
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups
        self.rebuild_prefill_attn_metadata = block_tables.cp_size > 1 and any(
            getattr(group.kv_cache_spec, "dcp_replicated", False)
            and any(
                layer_name in self.draft_attn_layer_names
                for layer_name in group.layer_names
            )
            for group in kv_cache_config.kv_cache_groups
        )

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        num_query_per_req: int = 1,
        causal: bool | Mapping[int, bool] = True,
        max_seq_len_upper_bound: int | None = None,
        query_start_loc_np: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        if query_start_loc_np is not None:
            # Non-uniform query layout (e.g. multi-module MTP's mixed
            # prefill/decode queries); num_query_per_req is ignored.
            query_start_loc_cpu = torch.empty(num_reqs_padded + 1, dtype=torch.int32)
            query_start_loc_cpu[: num_reqs + 1] = torch.from_numpy(
                query_start_loc_np[: num_reqs + 1]
            )
            query_start_loc_cpu[num_reqs:] = query_start_loc_cpu[num_reqs]
            max_query_len = (
                int(
                    (
                        query_start_loc_cpu[1 : num_reqs + 1]
                        - query_start_loc_cpu[:num_reqs]
                    )
                    .max()
                    .item()
                )
                if num_reqs > 0
                else num_query_per_req
            )
        else:
            # Uniform query: query_start_loc[i] = min(i, num_reqs) * num_query_per_req.
            # Clamp keeps the series non-decreasing past num_reqs, which some
            # attention backends require.
            query_start_loc_cpu = (
                torch.clamp(self.arange[: num_reqs_padded + 1], max=num_reqs)
                * num_query_per_req
            )
            max_query_len = num_query_per_req
        block_tables = [
            x[:num_reqs_padded] for x in self.block_tables.input_block_tables
        ]
        slot_mappings = self.block_tables.slot_mappings[:, :num_tokens_padded]
        draft_seq_lens_cpu_upper_bound = torch.zeros(
            num_reqs_padded, dtype=torch.int32, device="cpu"
        )
        torch.add(
            seq_lens_cpu_upper_bound[:num_reqs],
            step,
            out=draft_seq_lens_cpu_upper_bound[:num_reqs],
        )
        draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.draft_max_seq_len)
        seq_lens = self.input_buffers.seq_lens[:num_reqs_padded]
        dcp_local_seq_lens = None
        if self.block_tables.cp_size > 1:
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.block_tables.cp_size,
                self.block_tables.cp_rank,
                self.block_tables.cp_interleave,
            )
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs_padded]
        with record_function_or_nullcontext("vllm:v2/speculator/build_attn_metadata"):
            attn_metadata = build_attn_metadata(
                attn_groups=self.attn_groups,
                num_reqs=num_reqs_padded,
                num_tokens=num_tokens_padded,
                query_start_loc_gpu=self.input_buffers.query_start_loc[
                    : num_reqs_padded + 1
                ],
                query_start_loc_cpu=query_start_loc_cpu,
                max_query_len=max_query_len,
                seq_lens=seq_lens,
                max_seq_len=self.draft_max_seq_len,
                block_tables=block_tables,
                slot_mappings=slot_mappings,
                kv_cache_config=self.kv_cache_config,
                causal=causal,
                dcp_local_seq_lens=dcp_local_seq_lens,
                seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,
                max_seq_len_upper_bound=max_seq_len_upper_bound,
            )
        return attn_metadata

    def _validate_local_argmax_reduction(self) -> None:
        if not self.use_local_argmax_reduction:
            return
        if self.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "use_local_argmax_reduction is not compatible with "
                "draft_sample_method='probabilistic'."
            )
        if not hasattr(self.model, "get_top_tokens"):
            raise ValueError(
                "use_local_argmax_reduction is enabled but draft model "
                f"{self.model.__class__.__name__} does not implement "
                "get_top_tokens()."
            )
        logger.info(
            "Using local argmax reduction for draft token generation "
            "(communication: O(2*tp_size) vs O(vocab_size))."
        )

    def _greedy_sample_draft(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_local_argmax_reduction:
            if self._draft_logit_bias is not None:
                logger.warning_once(
                    "Draft logit bias is not applied under local argmax "
                    "reduction."
                )
            return self.model.get_top_tokens(hidden_states)
        logits = self.model.compute_logits(hidden_states)
        if self._draft_logit_bias is not None:
            logits = logits + self._draft_logit_bias
        return logits.argmax(dim=-1)

    def set_draft_top_p_source(self, source: UvaBackedTensor) -> None:
        """Name the sampler state that holds each request's draft nucleus mass."""
        self._draft_top_p_source = source

    def _refresh_draft_top_p(self) -> None:
        # The source's device view moves between staged writes; draft graphs
        # need a fixed address, as for the temperature.
        if self._draft_nucleus is not None and self._draft_top_p_source is not None:
            self.draft_top_p.copy_(self._draft_top_p_source.gpu)

    def _sample_probabilistic_draft(
        self,
        logits: torch.Tensor,
        positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor,
        active_rows: torch.Tensor | None = None,
        *,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        """Sample a draft from a stream disjoint from verifier recovery."""
        if self._draft_nucleus is None and envs.VLLM_K3_FUSED_DRAFT_SAMPLE:
            return gumbel_sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                positions,
                apply_temperature=True,
                logits_cache=draft_logits,
                logits_cache_col=draft_step,
                logits_cache_active_rows=active_rows,
                use_fp64=self.use_fp64_gumbel,
                pos_offset=(1 + DRAFT_GUMBEL_POS_OFFSET) + int(pos_offset),
                draft_temperature=self._draft_temperature,
                logit_bias=self._draft_logit_bias,
            )
        if pos_offset != 0:
            positions = positions + pos_offset
        if self._draft_temperature != 1.0:
            # Shape the proposal distribution before any bias, the nucleus
            # mask, and the logits cached for the verifier.
            logits = logits / self._draft_temperature
        if self._draft_logit_bias is not None:
            # Deterministic shift of the proposal distribution, before the
            # nucleus mask and before the logits are cached for the verifier.
            logits = logits + self._draft_logit_bias
        if self._draft_nucleus is not None:
            self._draft_nucleus.apply(
                logits, idx_mapping, temperature, self.draft_top_p
            )
        return gumbel_sample(
            logits,
            idx_mapping,
            temperature,
            seeds,
            draft_gumbel_pos(positions),
            apply_temperature=True,
            logits_cache=draft_logits,
            logits_cache_col=draft_step,
            logits_cache_active_rows=active_rows,
            use_fp64=self.use_fp64_gumbel,
        )

    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if draft_logits is not None:
            logits = self.model.compute_logits(hidden_states)
            return self._sample_probabilistic_draft(
                logits=logits,
                positions=positions,
                idx_mapping=idx_mapping,
                temperature=temperature,
                seeds=seeds,
                draft_step=draft_step,
                draft_logits=draft_logits,
            )
        return self._greedy_sample_draft(hidden_states)

    def _copy_request_inputs(
        self,
        num_reqs: int,
        # [num_reqs]
        idx_mapping: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
    ) -> None:
        # Copy temperature, seeds, and idx mapping to the pre-allocated buffers.
        # NOTE(woosuk): For draft sampling, we only consider the temperature
        # and ignore the other sampling parameters such as top_k and top_p,
        # for simplicity and performance.
        # While this may slightly degrade the acceptance rate, it does not
        # affect the output distribution after rejection sampling.
        self.temperature.copy_(temperature)
        self.seeds.copy_(seeds)
        self._refresh_draft_top_p()
        self.idx_mapping[:num_reqs].copy_(idx_mapping)
        # idx_mapping for CG padded requests points to -1, which is ignored
        # during sampling to prevent writing stale values to draft logits.
        self.idx_mapping[num_reqs:].fill_(-1)
