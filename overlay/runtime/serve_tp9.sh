#!/usr/bin/env bash
# Kimi-K3 TP9/DCP9 target with the DFlash draft sharded across the same nine
# ranks. Production entrypoint of candidates/k3-prod-levers-20260928: the
# entrypoint of candidates/k3-dcp-decode-caps-20260926 with the torch profiler
# opt-in (K3_TP9_PROFILER, default 0). With the variable unset the engine
# arguments equal the served ones minus the profiler arguments.
set -euo pipefail

export TP_SIZE=9 DCP_SIZE=9
export VLLM_K3_DRAFT_REMOTE_ADDRESS=""
export VLLM_K3_DRAFT_P2P=0
export VLLM_K3_DRAFT_DEFERRED_RESOLVE=0
export VLLM_DCP_SHARD_DRAFT=0
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export K3_PROFILE_RANK0_ONLY=1
# Decode collectives over the B12X PCIe IPC pool (the production TP8 path):
# DCP query-head gather, LSE reduce-scatter and the Kimi router/latent pair
# gather for batches of at most eight tokens; larger batches use NCCL AG+RS.
export VLLM_USE_B12X_DCP_A2A="${K3_TP9_USE_B12X_DCP_A2A:-1}"
export VLLM_DCP_A2A_MAX_TOKENS=${K3_DCP_A2A_MAX_TOKENS:-8}
export VLLM_DCP_A2A_LARGE_BACKEND=ag_rs
export B12X_PCIE_DCP_THREADS=512
export B12X_PCIE_DCP_BLOCK_LIMIT=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# MLA chunked-context workspace in tokens. The attention backend rounds it up
# to the DCP-aligned block (lcm(1536, 9) = 4608); 27648 = 6 x 4608 keeps the
# retained context projection buffer and the backend chunk the same size.
export VLLM_MLA_INTERNAL_CONTEXT_WORKSPACE_SIZE=27648
# Tensor-parallel all-reduce routing. The nine-peer one-shot pulls the whole
# payload from every peer (67 us per 57 KB call under graph replay); the
# push two-shot reduces the same payload in 20 us. "push" routes every
# all-reduce above 16 KB, which covers all decode sizes, through the two-shot
# push kernel; "island9" routes the same window through the two-island push
# kernel (ranks 0-3 and 4-7 reduce inside their PCIe switch clusters, rank 8
# contributes through island 0; fp32 island partials, one bf16 rounding);
# "oneshot" keeps the 112 KB one-shot window for A/B runs.
case "${K3_TP9_AR_ROUTE:-push}" in
  push|island9)
    export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=16KB
    export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=16KB
    export VLLM_PCIE_TWOSHOT_ALLREDUCE_MODE="${K3_TP9_AR_ROUTE:-push}"
    ;;
  oneshot)
    export VLLM_PCIE_TWOSHOT_ALLREDUCE_MODE=pull
    ;;
  *)
    echo "K3_TP9_AR_ROUTE must be push, island9 or oneshot" >&2
    exit 2
    ;;
esac
# B12X kernel resolution is frozen while CUDA graphs capture and every graph
# size runs one eager forward before capture, so captured graphs hold the
# resolved launcher of every B12X operation (the TP8 production entrypoint
# sets the same).
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
# Fault localization: synchronous launches make a CUDA error surface at the
# launch that caused it instead of at a later allocation or collective.
if [[ "${K3_TP9_LAUNCH_BLOCKING:-0}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
fi
export PYTHON_BIN=/opt/venv/bin/python
export PYTHONPATH=/opt/infernal-invocation/b12x:/opt/infernal-invocation/vllm:/opt/infernal-invocation/lmcache:/opt/exllamav3:/opt/venv/lib/python3.12/site-packages

K3_TP9_PORT="${PORT:-8094}"
# KV pool per rank. 2281701376 is the TP8 production value (2.12 GiB, about
# one 900k-token request); kv_budget.py derives the largest value the
# measured headroom allows.
# Context length: the target (kimi_linear) and the draft (qwen3) both declare
# max_position_embeddings 1048576; the same value is passed to the draft
# speculative config so the draft never truncates a target-admissible prompt.
K3_TP9_MAX_MODEL_LEN="${K3_TP9_MAX_MODEL_LEN:-1048576}"
export MAX_MODEL_LEN="${K3_TP9_MAX_MODEL_LEN}"
# Torch profiler (rank-0 step traces through /start_profile): off for serving.
# K3_TP9_PROFILER=1 registers the profiler and its endpoints for a research
# boot; a serving boot passes no profiler argument, so /start_profile and
# /stop_profile do not exist and no trace directory is created.
K3_TP9_PROFILER_ARGS=()
case "${K3_TP9_PROFILER:-0}" in
  1)
    K3_TP9_PROFILE_DIR="${VLLM_PROFILER_DIR:-/cache/tp9-colocated-prod/${K3_TP9_SOURCE_ID}/torch-profile}/$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir -p "${K3_TP9_PROFILE_DIR}"
    K3_TP9_PROFILER_ARGS=(
      --profiler-config.profiler=torch
      --profiler-config.torch_profiler_dir="${K3_TP9_PROFILE_DIR}"
      --profiler-config.torch_profiler_with_stack="${K3_TP9_PROFILER_WITH_STACK:-false}"
      --profiler-config.torch_profiler_record_shapes=false
      --profiler-config.torch_profiler_with_memory=false
      --profiler-config.torch_profiler_with_flops=false
      --profiler-config.torch_profiler_use_gzip=true
      --profiler-config.torch_profiler_dump_cuda_time_total=false
      --profiler-config.ignore_frontend=true
      --profiler-config.delay_iterations=0
      --profiler-config.max_iterations="${K3_TP9_PROFILER_MAX_ITERATIONS:-4}"
      --profiler-config.warmup_iterations=0
      --profiler-config.active_iterations=5
      --profiler-config.wait_iterations=0
    )
    ;;
  0) ;;
  *) echo "K3_TP9_PROFILER must be 0 or 1" >&2; exit 2 ;;
esac

# DFlash2 draft (lightseekorg/kimi-k3-dflash2, MLA layers on the K3 DSpark
# backbone): B12X_MLA draft attention, fp8 latent draft KV (the TP9/DCP9
# DSpark value), K3_TP9_SPEC_TOKENS speculative tokens (default four:
# verify width five, one-shot all-reduce payload).
K3_TP9_SPEC='{"method":"dflash","model":"/draft-model","num_speculative_tokens":'"${K3_TP9_SPEC_TOKENS:-4}"',"draft_tensor_parallel_size":9,"max_model_len":'"${K3_TP9_MAX_MODEL_LEN}"',"attention_backend":"B12X_MLA","kv_cache_dtype":"'"${K3_TP9_DRAFT_KV_DTYPE:-fp8}"'","draft_sample_method":"probabilistic","rejection_sample_method":"block","draft_load_config":{"load_format":"safetensors"}}'
K3_TP9_COMPILE='{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16],"pass_config":{"fuse_allreduce_rms":true}}'

# Vision: the MoonViT tower (27 layers, 12 heads, 0.83 GiB bf16) is replicated
# on every rank (12 heads do not split 9 ways, so the encoder runs
# data-parallel) and images enter the prefill as merged 2x2 patch tokens
# (in_patch_limit 40960 patches = 10240 tokens per image; image count
# per prompt is uncapped by default, overridable via
# K3_TP9_MAX_IMAGES_PER_PROMPT), the same vision tower limits as the TP8
# stack. Videos are decoded at the checkpoint's 8 fps policy
# (--media-io-kwargs video.fps, uncapped; the vLLM default is 32 frames) and
# split into 4-frame MoonViT chunks (per-frame cap 16384 patches = 4096
# tokens per chunk); video count per prompt defaults to 8, overridable via
# K3_TP9_MAX_VIDEOS_PER_PROMPT, the decode rate via K3_TP9_VIDEO_SAMPLE_FPS
# and the per-video patch budget via K3_TP9_VIDEO_PATCH_BUDGET (327680
# patches = 20480 tokens; the checkpoint default of 655360 makes the
# startup memory probe demand ~5.5 GiB of vision activations and trips the
# 0.97 memory-utilization headroom). The processor cache holds one item's
# pixel tensors; budget-sized videos reach ~0.8 GiB, so its default is
# 4 GiB (K3_TP9_MM_PROCESSOR_CACHE_GB). K3_TP9_VISION=0 serves the
# language model only.
case "${K3_TP9_VISION:-1}" in
  1)
    K3_TP9_MM_ARGS=(
      --limit-mm-per-prompt "{\"image\":${K3_TP9_MAX_IMAGES_PER_PROMPT:-999999},\"video\":${K3_TP9_MAX_VIDEOS_PER_PROMPT:-8}}"
      --mm-processor-kwargs "{\"in_patch_limit\":${K3_TP9_VISION_IN_PATCH_LIMIT:-40960},\"patch_limit_on_one_side\":${K3_TP9_VISION_PATCH_LIMIT_ON_ONE_SIDE:-512},\"in_patch_limit_video\":${K3_TP9_VIDEO_PATCH_BUDGET:-327680}}"
      --media-io-kwargs "{\"video\":{\"fps\":${K3_TP9_VIDEO_SAMPLE_FPS:-8},\"num_frames\":-1}}"
      --mm-processor-cache-gb "${K3_TP9_MM_PROCESSOR_CACHE_GB:-4}"
      --mm-encoder-tp-mode "${K3_TP9_MM_ENCODER_TP_MODE:-weights}"
    )
    ;;
  0) K3_TP9_MM_ARGS=(--language-model-only) ;;
  *) echo "K3_TP9_VISION must be 0 or 1" >&2; exit 2 ;;
esac

K3_TP9_ARGS=(
  /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model
  --served-model-name Kimi-K3
  --trust-remote-code "${K3_TP9_MM_ARGS[@]}"
  --host 127.0.0.1 --port "${K3_TP9_PORT}"
  --tensor-parallel-size 9 --decode-context-parallel-size 9
  --dcp-comm-backend a2a --dcp-kv-cache-interleave-size 1
  --mamba-block-size 4608 --prefix-match-unit 1536
  --load-format instanttensor --moe-backend b12x --linear-backend b12x
  --attention-backend B12X_MLA --kda-prefill-backend flashkda
  --dtype bfloat16 --kv-cache-dtype "${K3_TP9_TARGET_KV_DTYPE:-fp8_ds_mla}"
  --kv-cache-memory-bytes "${K3_TP9_KV_BYTES:-2281701376}" --max-model-len "${K3_TP9_MAX_MODEL_LEN}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4624}" --max-num-scheduled-tokens "${MAX_NUM_SCHEDULED_TOKENS:-4624}"
  --max-num-seqs 4 --gpu-memory-utilization 0.97
  --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align
  --speculative-config "${K3_TP9_SPEC}"
  --compilation-config "${K3_TP9_COMPILE}"
  --generation-config vllm --reasoning-parser kimi_k3
  --reasoning-config '{"loop_break_min_pattern_size":1,"loop_break_max_pattern_size":16,"loop_break_min_count":16,"loop_break_min_reasoning_tokens":16,"loop_break_check_interval":8,"loop_break_release":"force"}'
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 131072}'
  --tool-call-parser kimi_k3 --enable-auto-tool-choice
  --enable-prompt-tokens-details
  --attention-config.flash_attn_version=2
  "${K3_TP9_PROFILER_ARGS[@]}"
)

if [[ "${1:-}" == --print-command ]]; then
  printf '%q ' "${K3_TP9_ARGS[@]}"
  printf '\n'
  exit 0
fi

# CPU pinning (K3_TP9_CPU_PIN=1, default 1). The host is one 16C/32T socket
# (CCD0 = cores 0-7, CCD1 = 8-15, SMT sibling of core c = c+16, read from
# sysfs when present). Under load the nine worker main threads and the
# EngineCore main thread each spin at 100 %; unpinned they share physical
# cores and crowd CCD0. Layout (research/prefill-campaign-20260906/
# system-levers.md section 3.2): worker rank r main thread -> physical core
# 8+r for r < 8 and core 7 for r = 8 (with its SMT sibling), EngineCore
# main thread -> core 6 (+22), API server and LMCache MP server (all
# threads, plus their direct child processes) -> cores 0-5 (+16-21). Every
# other worker and EngineCore thread keeps the full CPU mask. A background
# job applies the layout once the server answers /health (all
# initialization threads exist by then and keep the inherited full mask),
# then re-applies it every K3_TP9_CPU_PIN_INTERVAL s (60) for
# K3_TP9_CPU_PIN_ROUNDS rounds (10) to widen threads created later, which
# inherit the pinned main thread's mask.
# Idempotent, logs only changes, never fails the boot.
# K3_TP9_CPU_PIN_DRY_RUN=1 reports the layout without applying it.
k3_tp9_cpu_pin() {
  local port="$1" parent="$2" log="[k3-cpu-pin]"
  local timeout="${K3_TP9_CPU_PIN_TIMEOUT:-5400}" rounds="${K3_TP9_CPU_PIN_ROUNDS:-10}"
  local interval="${K3_TP9_CPU_PIN_INTERVAL:-60}"
  local dry_run="${K3_TP9_CPU_PIN_DRY_RUN:-0}"
  local ncpu; ncpu="$(nproc --all 2>/dev/null || echo 0)"
  if (( ncpu < 32 )); then
    echo "$log skipped: ${ncpu} CPUs visible, the layout needs the 16C/32T host" >&2
    return 0
  fi
  if ! command -v taskset >/dev/null 2>&1; then
    echo "$log skipped: taskset not found" >&2
    return 0
  fi
  local full="0-$((ncpu - 1))" service="0-5,16-21" changed=0

  siblings_of() {
    local f="/sys/devices/system/cpu/cpu$1/topology/thread_siblings_list"
    if [[ -r "$f" ]]; then tr -d '\n' < "$f"; else printf '%s,%s' "$1" "$(($1 + 16))"; fi
  }
  # The helpers return masks through the variable ``mask`` (no command
  # substitution: the scan covers ~1,300 threads and must not fork per
  # thread). Masks use the hex form taskset -p prints.
  local mask=""
  # mask_of <tid>: the thread's affinity mask from /proc.
  mask_of() {
    local key val
    mask=""
    [[ -r "/proc/$1/status" ]] || return 1
    while IFS=$'\t' read -r key val; do
      if [[ "$key" == "Cpus_allowed:" ]]; then printf -v mask '%x' "$((16#${val//,/}))"; return 0; fi
    done < "/proc/$1/status" 2>/dev/null
    return 1
  }
  # mask_for_list "0-5,16-21": the mask of a CPU list (memoized).
  local -A list_masks=()
  mask_for_list() {
    local part lo hi c m=0
    local -a parts
    if [[ -n "${list_masks[$1]:-}" ]]; then mask="${list_masks[$1]}"; return 0; fi
    IFS=, read -ra parts <<< "$1"
    for part in "${parts[@]}"; do
      lo="${part%-*}"; hi="${part#*-}"
      for ((c = lo; c <= hi; c++)); do m=$(( m | (1 << c) )); done
    done
    printf -v mask '%x' "$m"
    list_masks[$1]="$mask"
  }
  # pin_tid <tid> <cpulist> <label>: set one thread's affinity when it
  # differs from the target, and log the change.
  pin_tid() {
    local before after target
    mask_of "$1" || return 0
    before="$mask"
    mask_for_list "$2"
    target="$mask"
    [[ "$before" == "$target" ]] && return 0
    if [[ "$dry_run" == 1 ]]; then
      echo "$log dry-run: $3 tid $1 mask $before -> $target ($2)" >&2
      changed=$((changed + 1))
      return 0
    fi
    taskset -pc "$2" "$1" >/dev/null 2>&1 || { echo "$log WARN cannot pin $3 tid $1" >&2; return 0; }
    mask_of "$1" || return 0
    after="$mask"
    if [[ "$before" != "$after" ]]; then
      echo "$log $3 tid $1: mask $before -> $after ($2)" >&2
      changed=$((changed + 1))
    fi
  }
  # pin_threads <pid> <main cpulist> <helper cpulist> <label>
  pin_threads() {
    local t tid
    for t in /proc/"$1"/task/[0-9]*; do
      tid="${t##*/}"
      if [[ "$tid" == "$1" ]]; then pin_tid "$tid" "$2" "$4 main"; else pin_tid "$tid" "$3" "$4 thread"; fi
    done
  }
  ppid_of() {
    local s
    [[ -r "/proc/$1/stat" ]] || return 1
    { read -r s < "/proc/$1/stat"; } 2>/dev/null || return 1
    s="${s##*) }"; set -- $s; echo "$2"
  }
  apply_layout() {
    local d pid title rank core engine="" api="" lmcache=""
    local -a workers=()
    local -a argv
    for d in /proc/[0-9]*; do
      pid="${d#/proc/}"
      [[ -r "$d/cmdline" ]] || continue
      argv=()
      { mapfile -t -d '' argv < "$d/cmdline"; } 2>/dev/null || continue
      title="${argv[*]:-}"
      if [[ "$title" =~ ^VLLM::Worker_TP([0-9]+) ]]; then
        workers+=("${BASH_REMATCH[1]}:$pid")
      elif [[ "$title" == "VLLM::EngineCore"* ]]; then
        engine="$pid"
      elif [[ "$title" =~ ^[^[:space:]]*python[0-9.]*\ -m\ vllm\.entrypoints\.cli\.main\ serve\  ]]; then
        api="$pid"
      elif [[ "$title" =~ ^[^[:space:]]*python[0-9.]*\ [^[:space:]]*/lmcache\ server\  ]]; then
        lmcache="$pid"
      fi
    done
    if (( ${#workers[@]} != 9 )) || [[ -z "$engine" ]]; then
      echo "$log WARN found ${#workers[@]} workers, EngineCore=${engine:-none}; pinning what exists" >&2
    fi
    local entry
    for entry in "${workers[@]}"; do
      rank="${entry%%:*}"; pid="${entry#*:}"
      if (( rank < 8 )); then core=$((8 + rank)); elif (( rank == 8 )); then core=7; else
        echo "$log WARN worker rank $rank has no core in the TP9 layout" >&2; continue
      fi
      pin_threads "$pid" "$(siblings_of "$core")" "$full" "worker TP$rank"
    done
    [[ -n "$engine" ]] && pin_threads "$engine" "$(siblings_of 6)" "$full" "EngineCore"
    local svc child
    for svc in $api $lmcache; do
      pin_threads "$svc" "$service" "$service" "service pid $svc"
      for d in /proc/[0-9]*; do
        child="${d#/proc/}"
        [[ "$child" == "$engine" ]] && continue
        [[ "$(ppid_of "$child" 2>/dev/null)" == "$svc" ]] || continue
        pin_threads "$child" "$service" "$service" "service child pid $child"
      done
    done
    echo "$log layout: workers=${#workers[@]} EngineCore=${engine:-none} api=${api:-none} lmcache=${lmcache:-none} changed=$changed" >&2
  }

  local waited=0
  until curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    if ! kill -0 "$parent" 2>/dev/null; then
      echo "$log server process exited before /health answered; nothing pinned" >&2
      return 0
    fi
    if (( waited >= timeout )); then
      echo "$log /health not answered within ${timeout}s; nothing pinned" >&2
      return 0
    fi
    sleep 5; waited=$((waited + 5))
  done
  local round=0
  while :; do
    changed=0
    apply_layout
    round=$((round + 1))
    if (( round >= rounds )) || [[ "$dry_run" == 1 ]] || ! kill -0 "$parent" 2>/dev/null; then break; fi
    sleep "$interval"
  done
  echo "$log finished after $round round(s)" >&2
  return 0
}

if [[ "${K3_TP9_CPU_PIN:-1}" == "1" ]]; then
  ( set +e; k3_tp9_cpu_pin "${K3_TP9_PORT}" "$$" ) &
fi

# The wrapper adds the matching LMCache connector arguments and owns the
# private cache-server process. Its source is pinned in this runtime bundle.
exec /bin/bash /runtime/lmcache-mp-wrapper.sh "${K3_TP9_ARGS[@]}"