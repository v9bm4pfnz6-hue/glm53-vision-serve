# GLM-5.3-Vision serving image — production build 61f90c4b12202a79 (2026-10-31).
# Base: runtime image with the vLLM/b12x/lmcache invocation trees the overlays are mounted into.
FROM ghcr.io/v9bm4pfnz6-hue/glm53-vision-runtime:20260918

# 146 overlay files from the qualified serving composition, placed at their container paths.
# sha256 of every file is pinned in manifest/build-61f90c4b12202a79.json (files.<dest-path>.sha256).
COPY overlay/ /

# Runtime environment of the serving composition (manifest `environment`).
# CUDA_VISIBLE_DEVICES is host-specific and intentionally NOT baked in; set it at run time
# (see run.sh — production uses the physical order 1,2,3,4,5,6,7,8,0).
ENV \
    B12X_PCIE_DCP_A2A_TRANSPORT=push \
    B12X_PCIE_DCP_GATHER_GROUPS=0,1,2,3,8;4,5,6,7 \
    B12X_PCIE_DCP_LSE_BLOCK_LIMIT=64 \
    B12X_PCIE_DCP_LSE_THREADS=64 \
    B12X_PCIE_DMA_FP8=off \
    B12X_PCIE_DMA_GRAPH_REPLAY=1 \
    B12X_PCIE_DMA_GRAPH_REPLAY_MAX_ENTRIES=2 \
    B12X_PCIE_RING_FP32_HOPS=1 \
    B12X_PCIE_TP9_PAIR_RELAY=1 \
    B12X_PCIE_TP9_STATIC_PEERS=1 \
    B12X_W4A8_TINY_DECODE=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    K3_PROFILE_RANK0_ONLY=1 \
    OMP_NUM_THREADS=1 \
    PYTHONPATH=/opt/infernal-invocation/b12x:/opt/infernal-invocation/vllm:/opt/venv/lib/python3.12/site-packages \
    VLLM_B12X_DENSE_ACTIVATION_MODE=a16 \
    VLLM_B12X_MLA_CKV_GATHER=1 \
    VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=131072 \
    VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=0 \
    VLLM_B12X_MOE_FP4_FORCE_A16=1 \
    VLLM_B12X_NVFP4_ACTIVATION_MODE=a16 \
    VLLM_CAPTURE_MEMORY_SNAPSHOT_DIR=/cache/capture-memory \
    VLLM_DCP_SHARD_DRAFT=1 \
    VLLM_DCP_TOPK_OWNER_MERGE=1 \
    VLLM_ENABLE_PCIE_ALLREDUCE=1 \
    VLLM_GLM_DCP9_B12X_FP32_COMBINE=1 \
    VLLM_GLM_DCP9_B12X_Q_GATHER=1 \
    VLLM_GLM_DCP9_B12X_TOPK_GATHER=0 \
    VLLM_GLM_DCP9_PROJECT_BEFORE_COMBINE=1 \
    VLLM_GLM_DCP9_PROJECT_KERNEL=v2 \
    VLLM_GLM_DCP9_PROJECT_V2_CONFIG=32,3,2 \
    VLLM_GLM_FP8_DENSE_PREFILL_AUTO=1 \
    VLLM_GLM_FP8_DENSE_REDUCE_BLOCK_N=16 \
    VLLM_GLM_FP8_DENSE_SPLITK=1 \
    VLLM_GLM_FP8_GLP_FUSED=0 \
    VLLM_GLM_FP8_ROUTED_PREFILL_AUTO=1 \
    VLLM_GLM_FP8_TAIL_CACHE=1 \
    VLLM_GLM_FP8_TAIL_REPLAY_TOKENS=4 \
    VLLM_GLM_FP8_TP9=1 \
    VLLM_GLM_UNEVEN_GATHER=argemm \
    VLLM_K3_MIXED_PREFILL_BUDGET=1024 \
    VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
    VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
    VLLM_PCIE_DMA_FP8=off \
    VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=8KB \
    VLLM_PCIE_TWOSHOT_ALLREDUCE_MODE=push \
    VLLM_PCIE_TWOSHOT_ROW_ELEMS=8 \
    VLLM_PROMPT_LOGPROBS_CHUNK_SIZE=128 \
    VLLM_USE_B12X_MOE=0

# vLLM serve command of the serving composition (manifest `server_args`).
# Expects the assembled composite weights at /model (see weights/assemble_weights.py)
# and a writable /cache volume. Listens on 127.0.0.1:8002; run with --network host.
# The base image's ENTRYPOINT is a fleet launcher wrapper; production replaces
# it with the venv python directly, so pin the interpreter here and keep CMD as
# its argument vector.
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["-m", "vllm.entrypoints.cli.main", "serve", "/model", "--served-model-name", "GLM-5.3", "--host", "127.0.0.1", "--port", "8002", "--tensor-parallel-size", "9", "--decode-context-parallel-size", "9", "--moe-backend", "triton", "--linear-backend", "triton", "--attention-backend", "B12X_MLA_SPARSE", "--dtype", "bfloat16", "--quantization", "fp8", "--kv-cache-dtype", "fp8_ds_mla", "--block-size", "64", "--load-format", "safetensors", "--hf-overrides", "{\"text_config\": {\"use_index_cache\": true, \"index_topk_pattern\": \"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS\"}}", "--max-model-len", "1048576", "--max-num-seqs", "4", "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.979", "--speculative-config", "{\"method\": \"mtp\", \"num_speculative_tokens\": 3, \"draft_tensor_parallel_size\": 9, \"kv_cache_dtype\": \"fp8_ds_mla\", \"attention_backend\": \"B12X_MLA_SPARSE\"}", "--compilation-config", "{\"mode\": 0, \"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"custom_ops\": [\"all\"], \"cudagraph_capture_sizes\": [1, 2, 4, 8, 16]}", "--enable-auto-tool-choice", "--enable-prompt-tokens-details", "--tool-call-parser", "glm47", "--reasoning-parser", "glm47", "--generation-config", "vllm", "--profiler-config", "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"/cache/profiles\", \"torch_profiler_with_stack\": false, \"torch_profiler_record_shapes\": false, \"torch_profiler_with_memory\": false, \"torch_profiler_with_flops\": false, \"torch_profiler_use_gzip\": true, \"torch_profiler_dump_cuda_time_total\": false, \"ignore_frontend\": true, \"delay_iterations\": 0, \"max_iterations\": 4, \"warmup_iterations\": 0, \"active_iterations\": 5, \"wait_iterations\": 0}", "--prefix-match-unit", "64", "--cp-kv-cache-interleave-size", "1", "--limit-mm-per-prompt", "{\"vision_chunk\": 16}", "--mm-encoder-tp-mode", "weights", "--mm-encoder-attn-backend", "TORCH_SDPA"]
