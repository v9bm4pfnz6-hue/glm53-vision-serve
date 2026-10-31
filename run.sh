#!/usr/bin/env bash
# Run the GLM-5.3-Vision serving container the way the production rig does.
#
# Requirements:
#   - 9x NVIDIA RTX PRO 6000 (SM120 / Blackwell, 96 GB) visible to the container
#   - weights assembled by weights/assemble_weights.py at $WEIGHTS_DIR
#   - a writable cache dir for Triton JIT + CUDA graph throwaway state
#
# The engine listens on 127.0.0.1:8002 (baked into the image CMD), so the
# container uses host networking. Change --host/--port by overriding the CMD
# if you need a different bind.
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:?set WEIGHTS_DIR to the assembled GLM-5.3-Vision-FP8 composite dir}"
CACHE_DIR="${CACHE_DIR:-$PWD/glm53-vision-cache}"
IMAGE="${IMAGE:-ghcr.io/v9bm4pfnz6-hue/glm53-vision-serve:20261031}"

mkdir -p "$CACHE_DIR"

# CUDA_VISIBLE_DEVICES: production pins the physical order 1,2,3,4,5,6,7,8,0 so
# DCP ranks 0-3,8 land on one PCIe switch chip and 4-7 on the other (see README).
# Adjust to your topology or leave unset to use enumeration order.
docker run -d \
  --name glm53-vision \
  --network host \
  --gpus all \
  --shm-size 16g \
  --restart unless-stopped \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8}" \
  -v "$WEIGHTS_DIR":/model:ro \
  -v "$CACHE_DIR":/cache \
  "$IMAGE"
