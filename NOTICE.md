# NOTICE

This repository packages a serving engine for GLM-5.3-Vision. The engine-side
code here (the `overlay/` files, `patches/`, Dockerfiles, scripts and docs) is
released under the Apache-2.0 license in `LICENSE`.

## Derivation

- The overlay files are modifications and ports of
  [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0, Copyright
  contributors to the vLLM project) and of the b12x runtime trees (Local
  Inference Lab) that are pre-installed in the runtime base image
  `ghcr.io/v9bm4pfnz6-hue/glm53-vision-runtime:20260918`. Upstream headers and
  SPDX identifiers are preserved in the files.
- The runtime base image is a convenience artifact of an internal deployment;
  it has not been individually license-audited artifact-by-artifact.

## Model weights (not distributed here)

The composite weights assembled by `weights/assemble_weights.py` come from:

- `zai-org/GLM-5.3` — GLM-5.3 License (MIT-style; Model-as-a-Service use by
  businesses above $10B revenue requires a Z.ai security review).
- `lunaroute/GLM-5.3-Vision-FP8` — MIT assembly, containing:
  - MoonViT-3d vision tower (from Kimi-K2.6) — Moonshot Modified-MIT
    (attribution obligations above 100M MAU / $20M monthly).
  - PatchMerger projector (trained by Baseten against GLM-5.2, MIT).

No model weights are included in this repository or in the Docker images built
from it. Obtain the weights from the upstream repositories and comply with
their licenses.

## Trademarks

NVIDIA, RTX PRO 6000 and CUDA are trademarks of NVIDIA Corporation. GLM is a
trademark of Zhipu AI (Z.ai). This project is not affiliated with, endorsed by,
or supported by any of them.
