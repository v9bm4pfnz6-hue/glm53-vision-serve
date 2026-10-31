#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "ERROR: lmcache-mp-wrapper.sh requires a model server command" >&2
  exit 2
fi

mode="${LMCACHE_MODE:-off}"
mode="${mode,,}"
case "${mode}" in
  off|0)
    exec "$@"
    ;;
  ram|memory|1)
    mode=ram
    ;;
  disk|ram-disk|memory-disk)
    mode=disk
    ;;
  *)
    echo "ERROR: LMCACHE_MODE must be off, ram, or disk; got ${mode}" >&2
    exit 2
    ;;
esac

command -v lmcache >/dev/null || {
  echo "ERROR: LMCACHE_MODE=${mode}, but the lmcache CLI is not installed" >&2
  exit 2
}

lmcache_expected_source_root="${LMCACHE_EXPECTED_SOURCE_ROOT:-}"
if [[ -n "${lmcache_expected_source_root}" ]]; then
  "${PYTHON_BIN:-python3}" - "${lmcache_expected_source_root}" <<'PY'
from pathlib import Path
import sys

import lmcache

expected = Path(sys.argv[1]).resolve()
loaded = Path(lmcache.__file__).resolve()
if not loaded.is_relative_to(expected):
    raise SystemExit(
        f"LMCache imported from {loaded}; expected a module below {expected}"
    )
PY
fi

service_port="${PORT:-8000}"
port_offset=0
if [[ "${service_port}" =~ ^[0-9]+$ ]] && (( service_port >= 8000 )); then
  port_offset=$((service_port - 8000))
fi

lmcache_host="${LMCACHE_HOST:-127.0.0.1}"
# Derived ports assume each service uses a unique PORT offset. Deployments with
# custom spacing must set both LMCACHE_PORT and LMCACHE_HTTP_PORT explicitly.
lmcache_port="${LMCACHE_PORT:-$((5555 + port_offset))}"
lmcache_http_port="${LMCACHE_HTTP_PORT:-$((8099 + port_offset))}"
lmcache_chunk_size="${LMCACHE_CHUNK_SIZE:-}"
if [[ -z "${lmcache_chunk_size}" ]]; then
  # LMCache chunks must align to every effective DCP paged-cache block.
  # TP6 uses 192-token (DCP3) or 384-token (DCP6) manager blocks; 512 is
  # invalid for both. Power-of-two DCP layouts retain the established 512.
  case "${DCP_SIZE:-${DCP:-1}}" in
    3|6) lmcache_chunk_size=384 ;;
    *) lmcache_chunk_size=512 ;;
  esac
fi
lmcache_l1_gb="${LMCACHE_L1_GB:-24}"
lmcache_l1_init_gb="${LMCACHE_L1_INIT_GB:-${lmcache_l1_gb}}"
# Every TP rank registers an independent GPU client, including at DCP1. Give
# each client its own affinity worker so rank transfers are not serialized.
# Constrained hosts can still override this explicitly.
lmcache_gpu_workers="${LMCACHE_MAX_GPU_WORKERS:-${TP_SIZE:-${TP:-1}}}"
lmcache_cpu_workers="${LMCACHE_MAX_CPU_WORKERS:-4}"
lmcache_log="${LMCACHE_LOG:-/tmp/lmcache-mp-${service_port}.log}"
lmcache_transfer_mode="${LMCACHE_TRANSFER_MODE:-auto}"
lmcache_transfer_mode="${lmcache_transfer_mode,,}"
case "${lmcache_transfer_mode}" in
  auto|lmcache_driven|engine_driven) ;;
  *)
    echo "ERROR: LMCACHE_TRANSFER_MODE must be auto, lmcache_driven, or engine_driven; got ${lmcache_transfer_mode}" >&2
    exit 2
    ;;
esac

lmcache_kv_load_failure_policy="${LMCACHE_KV_LOAD_FAILURE_POLICY:-recompute}"
lmcache_kv_load_failure_policy="${lmcache_kv_load_failure_policy,,}"
case "${lmcache_kv_load_failure_policy}" in
  recompute|fail) ;;
  *)
    echo "ERROR: LMCACHE_KV_LOAD_FAILURE_POLICY must be recompute or fail; got ${lmcache_kv_load_failure_policy}" >&2
    exit 2
    ;;
esac

lmcache_mq_timeout="${LMCACHE_MQ_TIMEOUT:-60}"
lmcache_heartbeat_interval="${LMCACHE_HEARTBEAT_INTERVAL:-10}"
lmcache_worker_reap_timeout="${LMCACHE_WORKER_REAP_TIMEOUT:-120}"
lmcache_worker_registration_grace="${LMCACHE_WORKER_REGISTRATION_GRACE:-3600}"

lmcache_separate_object_groups="${LMCACHE_SEPARATE_OBJECT_GROUPS:-0}"
lmcache_separate_object_groups="${lmcache_separate_object_groups,,}"
case "${lmcache_separate_object_groups}" in
  1|true|yes|on) lmcache_separate_object_groups=1 ;;
  0|false|no|off) lmcache_separate_object_groups=0 ;;
  *)
    echo "ERROR: LMCACHE_SEPARATE_OBJECT_GROUPS must be a boolean; got ${lmcache_separate_object_groups}" >&2
    exit 2
    ;;
esac

# L1 (shared-memory pool) eviction. The controller polls once per second and,
# whenever usage exceeds trigger_watermark x pool size, evicts eviction_ratio
# x (number of keys) least-recently-used objects (write-back to L2 first when
# --write-back-on-evict is set). Headroom after an eviction is what a cold
# external restore can reserve without the emergency path, so the pair must
# leave more than the largest expected restore free. These knobs are separate
# from LMCACHE_L2_EVICTION_* (the disk tier's controller).
lmcache_l1_eviction_watermark="${LMCACHE_L1_EVICTION_WATERMARK:-0.90}"
lmcache_l1_eviction_ratio="${LMCACHE_L1_EVICTION_RATIO:-0.10}"
python3 - "${lmcache_l1_eviction_watermark}" "${lmcache_l1_eviction_ratio}" <<'PY' || exit 2
import sys
watermark, ratio = (float(v) for v in sys.argv[1:3])
if not 0.0 < watermark <= 1.0 or not 0.0 < ratio <= 1.0:
    raise SystemExit(
        "LMCACHE_L1_EVICTION_WATERMARK and LMCACHE_L1_EVICTION_RATIO must be in (0, 1]"
    )
PY

server_args=(
  server
  --host "${lmcache_host}"
  --port "${lmcache_port}"
  --chunk-size "${lmcache_chunk_size}"
  --max-gpu-workers "${lmcache_gpu_workers}"
  --max-cpu-workers "${lmcache_cpu_workers}"
  --supported-transfer-mode "${lmcache_transfer_mode}"
  --l1-size-gb "${lmcache_l1_gb}"
  --l1-init-size-gb "${lmcache_l1_init_gb}"
  # L1 residency TTLs. Default 7 days: the LRU watermark eviction is the real
  # capacity mechanism, so the TTL only guards against genuinely stale data.
  # Short TTLs (1-2 h) evict idle-but-active sessions and force every resumed
  # conversation to re-restore from L2 or re-prefill.
  --l1-write-ttl-seconds "${LMCACHE_L1_WRITE_TTL_SECONDS:-604800}"
  --l1-read-ttl-seconds "${LMCACHE_L1_READ_TTL_SECONDS:-604800}"
  --eviction-policy LRU
  --eviction-trigger-watermark "${lmcache_l1_eviction_watermark}"
  --eviction-ratio "${lmcache_l1_eviction_ratio}"
  --l2-store-policy default
  --l2-prefetch-policy default
  --l2-prefetch-max-in-flight "${LMCACHE_L2_PREFETCH_MAX_IN_FLIGHT:-1}"
  # windowed restore: a loading lookup pins at most this many chunks in L1 and reports the rest from the L2 index; workers restore the rest in windows (0 = load the whole prefix at lookup time)
  --restore-pin-limit-chunks "${LMCACHE_RESTORE_PIN_LIMIT_CHUNKS:-0}"
  # below-watermark backup flush: copies evictable L1 keys to L2 ahead of eviction so victims are durable when evicted (0 = off)
  --periodic-flush-interval "${LMCACHE_L1_PERIODIC_FLUSH_INTERVAL:-0}"
  --http-host 127.0.0.1 \
  --http-port "${lmcache_http_port}"
  --worker-reap-timeout-seconds "${lmcache_worker_reap_timeout}"
  --worker-registration-grace-seconds "${lmcache_worker_registration_grace}"
)

# External prefix restores reserve L1 space before the L2-to-L1 transfer.
# Steady-state L1 headroom is only (1 - watermark) x L1 size, which is
# smaller than a full multi-chunk prefix restore, so without these flags
# lookup hits are trimmed to the free space available at that moment.
# Persist evicted L1 keys to L2 and let the prefetch controller synchronously
# evict LRU victims, so restores fit regardless of L1 pressure. Either flag
# can be disabled by setting the corresponding env var to 0; the emergency
# eviction flag is only valid together with write-back.
if [[ "${LMCACHE_WRITE_BACK_ON_EVICT:-1}" != "0" ]]; then
  server_args+=(--write-back-on-evict)
  if [[ "${LMCACHE_EMERGENCY_EVICT_FOR_PREFETCH:-1}" != "0" ]]; then
    server_args+=(--emergency-evict-for-prefetch)
  fi
fi

# An explicitly empty shared-memory name selects bounded per-request transfer
# buffers. This avoids mapping and pinning the entire L1 pool in every vLLM
# worker when the engine-driven transfer path is used.
if [[ -v LMCACHE_SHM_NAME ]]; then
  server_args+=(--shm-name "${LMCACHE_SHM_NAME}")
fi
if [[ "${lmcache_separate_object_groups}" == 1 ]]; then
  server_args+=(--separate-object-groups)
fi

lmcache_l2_path=disabled
if [[ "${mode}" == "disk" ]]; then
  lmcache_l2_path="${LMCACHE_L2_PATH:-/cache/lmcache/${service_port}}"
  lmcache_l2_gb="${LMCACHE_L2_GB:-256}"
  lmcache_l2_workers="${LMCACHE_L2_WORKERS:-4}"
  # L2 (disk) eviction. The fs_native adapter only rejects stores once
  # max_capacity_gb is reached ("capacity would exceed"); it never frees
  # space by itself. The adapter JSON accepts an "eviction" block that wires
  # the adapter into the MP server's L2EvictionController: the startup scan
  # seeds an LRU index ordered by file mtime, lookups/loads touch keys, and
  # whenever usage exceeds trigger_watermark x capacity the controller
  # deletes eviction_ratio x (number of keys) least-recently-used objects.
  # LMCACHE_L2_EVICTION_POLICY=noop disables it (previous behaviour).
  lmcache_l2_eviction_policy="${LMCACHE_L2_EVICTION_POLICY:-LRU}"
  lmcache_l2_eviction_watermark="${LMCACHE_L2_EVICTION_WATERMARK:-0.90}"
  lmcache_l2_eviction_ratio="${LMCACHE_L2_EVICTION_RATIO:-0.05}"
  case "${lmcache_l2_eviction_policy}" in
    LRU|IsolatedLRU|noop) ;;
    *)
      echo "ERROR: LMCACHE_L2_EVICTION_POLICY must be LRU, IsolatedLRU or noop; got ${lmcache_l2_eviction_policy}" >&2
      exit 2
      ;;
  esac
  mkdir -p "${lmcache_l2_path}"
  l2_config="$(
    LMCACHE_JSON_PATH="${lmcache_l2_path}" \
    LMCACHE_JSON_WORKERS="${lmcache_l2_workers}" \
    LMCACHE_JSON_CAPACITY_GB="${lmcache_l2_gb}" \
    LMCACHE_JSON_EVICTION_POLICY="${lmcache_l2_eviction_policy}" \
    LMCACHE_JSON_EVICTION_WATERMARK="${lmcache_l2_eviction_watermark}" \
    LMCACHE_JSON_EVICTION_RATIO="${lmcache_l2_eviction_ratio}" \
      python3 - <<'PY'
import json
import os

config = {
    "type": "fs_native",
    "base_path": os.environ["LMCACHE_JSON_PATH"],
    "num_workers": int(os.environ["LMCACHE_JSON_WORKERS"]),
    "use_odirect": False,
    "max_capacity_gb": float(os.environ["LMCACHE_JSON_CAPACITY_GB"]),
}
policy = os.environ["LMCACHE_JSON_EVICTION_POLICY"]
if policy != "noop":
    watermark = float(os.environ["LMCACHE_JSON_EVICTION_WATERMARK"])
    ratio = float(os.environ["LMCACHE_JSON_EVICTION_RATIO"])
    if not 0.0 < watermark <= 1.0 or not 0.0 < ratio <= 1.0:
        raise SystemExit(
            "LMCACHE_L2_EVICTION_WATERMARK and LMCACHE_L2_EVICTION_RATIO must be in (0, 1]"
        )
    config["eviction"] = {
        "eviction_policy": policy,
        "trigger_watermark": watermark,
        "eviction_ratio": ratio,
    }
print(json.dumps(config, separators=(",", ":")))
PY
  )"
  server_args+=(--l2-adapter "${l2_config}")
fi

transfer_config="$(
  LMCACHE_JSON_HOST="${lmcache_host}" \
  LMCACHE_JSON_PORT="${lmcache_port}" \
  LMCACHE_JSON_TRANSFER_MODE="${lmcache_transfer_mode}" \
  LMCACHE_JSON_LOAD_FAILURE_POLICY="${lmcache_kv_load_failure_policy}" \
  LMCACHE_JSON_MQ_TIMEOUT="${lmcache_mq_timeout}" \
  LMCACHE_JSON_HEARTBEAT_INTERVAL="${lmcache_heartbeat_interval}" \
  LMCACHE_JSON_WORKER_REAP_TIMEOUT="${lmcache_worker_reap_timeout}" \
  LMCACHE_JSON_WORKER_REGISTRATION_GRACE="${lmcache_worker_registration_grace}" \
    python3 - <<'PY'
import json
import math
import os
import shutil

mq_timeout = float(os.environ["LMCACHE_JSON_MQ_TIMEOUT"])
heartbeat_interval = float(os.environ["LMCACHE_JSON_HEARTBEAT_INTERVAL"])
heartbeat_timeout = float(os.environ.get("LMCACHE_HEARTBEAT_TIMEOUT", "30"))
restore_pin_chunks = int(os.environ.get("LMCACHE_RESTORE_PIN_LIMIT_CHUNKS", "0"))
restore_window_chunks = int(os.environ.get("LMCACHE_RESTORE_WINDOW_CHUNKS", "0"))
if restore_pin_chunks < 0 or restore_window_chunks < 0:
    raise SystemExit("LMCache restore chunk counts must be non-negative")
if restore_pin_chunks > 0 and (
    restore_window_chunks == 0
    or os.environ["LMCACHE_JSON_TRANSFER_MODE"] != "engine_driven"
):
    raise SystemExit(
        "Pin-limited LMCache lookup requires engine_driven transfers and "
        "LMCACHE_RESTORE_WINDOW_CHUNKS > 0"
    )
if restore_pin_chunks > 0:
    if not os.environ.get("LMCACHE_SHM_NAME"):
        raise SystemExit("Pin-limited LMCache lookup requires a named SHM pool")
    pool_gib = float(os.environ.get("LMCACHE_L1_GB", "24"))
    reserve_gib = float(os.environ.get("LMCACHE_SHM_RESERVE_GB", "4"))
    if not math.isfinite(pool_gib) or pool_gib <= 0:
        raise SystemExit("LMCACHE_L1_GB must be finite and positive")
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise SystemExit("LMCACHE_SHM_RESERVE_GB must be finite and non-negative")
    required_bytes = int((pool_gib + reserve_gib) * 2**30)
    free_bytes = shutil.disk_usage("/dev/shm").free
    if free_bytes < required_bytes:
        raise SystemExit(
            "Insufficient /dev/shm capacity for windowed LMCache restore: "
            f"free={free_bytes}, required_with_reserve={required_bytes}. "
            "Run the deployment restart script to remove unmapped cache pools; "
            "pickle transport cannot restore a pin-limited prefix."
        )
worker_reap_timeout = float(os.environ["LMCACHE_JSON_WORKER_REAP_TIMEOUT"])
worker_registration_grace = float(
    os.environ["LMCACHE_JSON_WORKER_REGISTRATION_GRACE"]
)
for name, value in (
    ("LMCACHE_MQ_TIMEOUT", mq_timeout),
    ("LMCACHE_HEARTBEAT_INTERVAL", heartbeat_interval),
    ("LMCACHE_WORKER_REAP_TIMEOUT", worker_reap_timeout),
    ("LMCACHE_WORKER_REGISTRATION_GRACE", worker_registration_grace),
):
    if not math.isfinite(value):
        raise SystemExit(f"{name} must be finite; got {value}")
if mq_timeout <= 0:
    raise SystemExit(f"LMCACHE_MQ_TIMEOUT must be positive; got {mq_timeout}")
if heartbeat_interval <= 0:
    raise SystemExit(
        "LMCACHE_HEARTBEAT_INTERVAL must be positive; "
        f"got {heartbeat_interval}"
    )
if worker_reap_timeout != 0 and worker_reap_timeout < 30:
    raise SystemExit(
        "LMCACHE_WORKER_REAP_TIMEOUT must be 0 or at least 30 seconds; "
        f"got {worker_reap_timeout}"
    )
if worker_registration_grace < worker_reap_timeout:
    raise SystemExit(
        "LMCACHE_WORKER_REGISTRATION_GRACE must be at least "
        f"LMCACHE_WORKER_REAP_TIMEOUT; got {worker_registration_grace} "
        f"and {worker_reap_timeout}"
    )
if worker_reap_timeout and 3 * heartbeat_interval > worker_reap_timeout:
    raise SystemExit(
        "LMCACHE_WORKER_REAP_TIMEOUT must be at least three heartbeat "
        f"intervals; got {worker_reap_timeout} and {heartbeat_interval}"
    )

print(
    json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            # A cache-tier failure is not a model failure. Recompute discards
            # the affected external blocks and preserves the API request.
            "kv_load_failure_policy": os.environ[
                "LMCACHE_JSON_LOAD_FAILURE_POLICY"
            ],
            "kv_connector_extra_config": {
                "lmcache.mp.host": f"tcp://{os.environ['LMCACHE_JSON_HOST']}",
                "lmcache.mp.port": int(os.environ["LMCACHE_JSON_PORT"]),
                "lmcache.mp.mq_timeout": mq_timeout,
                "lmcache.mp.heartbeat_interval": heartbeat_interval,
                "lmcache.mp.heartbeat_timeout": heartbeat_timeout,
                "lmcache.mp.mp_transfer_mode": os.environ[
                    "LMCACHE_JSON_TRANSFER_MODE"
                ],
                # Spawned workers receive connector config even when their
                # process environment omits the restore-window variable.
                "lmcache.mp.restore_window_chunks": restore_window_chunks,
            },
        },
        separators=(",", ":"),
    )
)
PY
)"

# LMCache registers KV storage by address. PyTorch expandable segments can
# remap those virtual addresses, so vLLM intentionally rejects this pairing.
# Keep any unrelated allocator settings while forcing the incompatible option
# off before the downstream model helper applies its normal default.
allocator_config="${PYTORCH_CUDA_ALLOC_CONF:-}"
if [[ -z "${allocator_config}" ]]; then
  allocator_config="expandable_segments:False"
elif [[ "${allocator_config}" =~ (^|,)expandable_segments:True(,|$) ]]; then
  allocator_config="${allocator_config//expandable_segments:True/expandable_segments:False}"
  echo "LMCache requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False; overriding expandable_segments:True"
fi
export PYTORCH_CUDA_ALLOC_CONF="${allocator_config}"

rm -f "${lmcache_log}"
export LMCACHE_DISABLE_BANNER="${LMCACHE_DISABLE_BANNER:-1}"
lmcache_server_command=(lmcache)
if [[ "${lmcache_transfer_mode}" == engine_driven ]]; then
  # GPU visibility is removed only from the standalone cache server. GPU
  # gather/scatter operations remain in the existing vLLM worker processes.
  lmcache_server_env="${LMCACHE_SERVER_ENV-CUDA_VISIBLE_DEVICES= CUDA_MODULE_LOADING=LAZY}"
  read -r -a lmcache_server_env_args <<< "${lmcache_server_env}"
  for assignment in "${lmcache_server_env_args[@]}"; do
    if [[ ! "${assignment}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
      echo "ERROR: invalid LMCACHE_SERVER_ENV assignment: ${assignment}" >&2
      exit 2
    fi
  done
  lmcache_server_command=(env "${lmcache_server_env_args[@]}" lmcache)
fi
"${lmcache_server_command[@]}" "${server_args[@]}" >"${lmcache_log}" 2>&1 &
lmcache_pid=$!
model_pid=""
shutdown_requested=0

stop_children() {
  if [[ -n "${model_pid}" ]] && kill -0 "${model_pid}" 2>/dev/null; then
    kill -TERM "${model_pid}" 2>/dev/null || true
  fi
  if kill -0 "${lmcache_pid}" 2>/dev/null; then
    kill -TERM "${lmcache_pid}" 2>/dev/null || true
  fi
}

request_shutdown() {
  shutdown_requested=1
  stop_children
}
trap request_shutdown INT TERM HUP

ready=0
for _ in $(seq 1 "${LMCACHE_START_TIMEOUT:-120}"); do
  if ! kill -0 "${lmcache_pid}" 2>/dev/null; then
    break
  fi
  if curl --fail --silent --show-error --max-time 1 \
      "http://127.0.0.1:${lmcache_http_port}/healthcheck" \
      >/dev/null 2>&1; then
    ready=1
    break
  fi
  if grep -Fq "${LMCACHE_READY_LOG_TEXT:-LMCache ZMQ cache server is running}" \
      "${lmcache_log}"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  echo "ERROR: LMCache did not become ready; log follows" >&2
  sed -n '1,320p' "${lmcache_log}" >&2
  stop_children
  wait "${lmcache_pid}" 2>/dev/null || true
  exit 1
fi

printf 'LMCache ready: mode=%s transfer=%s L1=%sGB chunk=%s L2=%s load_failure=%s heartbeat=%ss health=http://%s:%s/healthcheck metrics=http://%s:%s/metrics log=%s\n' \
  "${mode}" "${lmcache_transfer_mode}" "${lmcache_l1_gb}" "${lmcache_chunk_size}" \
  "${lmcache_l2_path}" "${lmcache_kv_load_failure_policy}" \
  "${lmcache_heartbeat_interval}" "${lmcache_host}" "${lmcache_http_port}" \
  "${lmcache_host}" "${lmcache_http_port}" "${lmcache_log}"

"$@" --kv-transfer-config "${transfer_config}" &
model_pid=$!

set +e
completed_pid=""
wait -n -p completed_pid "${lmcache_pid}" "${model_pid}"
first_status=$?
set -e

if [[ "${shutdown_requested}" == 1 ]]; then
  set +e
  if [[ "${completed_pid:-}" == "${model_pid}" ]]; then
    model_status=${first_status}
  else
    wait "${model_pid}" 2>/dev/null
    model_status=$?
  fi
  if [[ "${completed_pid:-}" != "${lmcache_pid}" ]]; then
    wait "${lmcache_pid}" 2>/dev/null
  fi
  set -e
  exit "${model_status}"
fi

if [[ "${completed_pid:-}" == "${lmcache_pid}" ]]; then
  echo "ERROR: LMCache exited while the model server was running" >&2
  sed -n '1,320p' "${lmcache_log}" >&2
  stop_children
  wait "${model_pid}" 2>/dev/null || true
  exit 1
fi

stop_children
wait "${lmcache_pid}" 2>/dev/null || true
exit "${first_status}"
