# glm53-vision-serve

Docker packaging of a production **GLM-5.3-Vision** serving engine: the official
GLM-5.3 FP8 text model with a MoonViT vision tower, served with tensor-parallel 9 /
decode-context-parallel 9 / MTP-3 (3-token speculative decoding) on nine
NVIDIA RTX PRO 6000 (SM120 / Blackwell) GPUs, fronted by a heavily tuned
`vllm`/`b12x` invocation stack with custom Triton/CUDA kernels for a dual-switch
PCIe fabric.

The image published from this repo is **file-exact** with the qualified serving
composition on the rig it was built on: every one of the 146 overlay files inside
the image sha256-matches the build manifest of that deployment.

## What you need

- **9x NVIDIA RTX PRO 6000** (or other SM120-class GPUs, 96 GB each). The serving
  composition is pinned to TP9/DCP9 and its PCIe collectives are tuned for a
  specific topology (see below); other SM120 counts need argument changes and
  will behave differently.
- Docker with the NVIDIA runtime.
- ~1 TB of disk for weights (see below) and roughly 90 GB of free VRAM per GPU
  after model load for the KV pool the default 1 M context targets.
- `huggingface_hub` for the weights assembly (`pip install "huggingface_hub[cli]"`).
- Access to the two public HF repos below (no gating encountered at assembly time).

## Quickstart

```bash
# 1. Assemble the composite weights from public sources (no redistribution here):
python3 weights/assemble_weights.py --out /data/GLM-5.3-Vision-FP8-build

# 2. Pull and run:
docker pull ghcr.io/v9bm4pfnz6-hue/glm53-vision-serve:20261031
WEIGHTS_DIR=/data/GLM-5.3-Vision-FP8-build ./run.sh

# 3. First boot takes on the order of 10 minutes (weight load + JIT warmup for
#    your input shapes). Then:
curl http://127.0.0.1:8002/v1/models
```

`docker-compose.yml` is provided as an alternative to `run.sh`.

## The model

The composite weights directory has exactly two upstreams, both public:

| Component | Source | Notes |
|---|---|---|
| Text backbone (141 shards) | [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3), FP8 E4M3, revision pinned in `weights/weights-reference.json` | Official release, byte-for-byte |
| Vision tower `vision_tower.safetensors` (~833 MB, BF16) | [`lunaroute/GLM-5.3-Vision-FP8`](https://huggingface.co/lunaroute/GLM-5.3-Vision-FP8) | MoonViT-3d, 27 layers, 1152-wide, 16 heads × 72 (from Kimi-K2.6) |
| Projector `mm_projector.safetensors` (~99 MB, BF16) | same | PatchMerger MLP 1152 → 4608 → 6144, trained by Baseten against frozen GLM-5.2 |
| Config / tokenizer / chat template / processor | same | Glm5v config whose `text_config` equals official GLM-5.3; chat template with an image-part branch |

Honest caveats carried over from the assembly provenance: the projector was
trained against GLM-**5.2**, and its transfer to GLM-5.3 was published by
lunaroute with no GLM-5.3-specific training anywhere; the public quality
evidence for the transfer is 60.0% MMMU-Pro standard (n=300, zero-shot) measured
on an NVFP4 sibling model. Treat vision quality accordingly. License chain:
GLM-5.3 License (MIT-style, with a MaaS condition above $10B revenue), lunaroute
assembly MIT, MoonViL tower Moonshot Modified-MIT, Baseten projector MIT.

Up to **16 images per prompt**, at most 4,225 image tokens per image; the image
encoder (torch SDPA per image — the vLLM flash-attention build has no SM120
kernel for 72-wide heads) and preprocessing run in-tree, `trust_remote_code`
stays off.

## What's inside the image

Two layers on top of a runtime base (published separately as
`ghcr.io/v9bm4pfnz6-hue/glm53-vision-runtime:20260918`, which carries the
`vllm` / `b12x` / `lmcache` invocation trees the overlays drop into):

- **146 overlay files** (`overlay/` in this repo, paths = container paths):
  - GLM-5.3 vision port — in-tree `Glm5vConfig` / `Glm5vForConditionalGeneration`
    (`glm5v.py`), vision axes in the GLM virtual-TP plan, fused CPU image
    processor.
  - TP9 uneven sharding of the replicated low-rank projections (fused q_a/kv_a,
    indexer `wq_b`, MTP `eh_proj`) — −1.31 GiB of weights per rank, zero-padded
    heads/channels that contribute exactly zero.
  - `argemm` transport: supported FP8 GEMM outputs written directly into
    original positions of an exact-width buffer gathered through the PCIe
    all-reduce (removes the local-output copy).
  - DCP9 FP32 combine: per-rank value up-projection of FP32 partials for all 72
    heads via a pipelined Triton kernel (`VLLM_GLM_DCP9_PROJECT_KERNEL=v2`,
    bitwise-identical outputs, 67 → 31 µs per layer), B12X PCIe FP32 exchange,
    owner-side FP64 accumulation rounded once to BF16.
  - W8A16 Triton GEMM family for the FP8 weights with original FP32 scales,
    ordered split-K, fused GLP finish, lossless PCIe DMA pipeline.
  - PCIe two-shot all-reduce (row width 8, static peers, pair relay), DCP
    address-of-a2a `push` transport, LSE gather, query gather with two switch
    groups.
  - MTP3 draft/verify fixes (rank-local KV slots, rejection-corrected GPU
    causal lengths), V1 runner changes, output processor, metrics loggers.
  - Shared runtime pieces inherited from the K3 swarm package (fused-MoE
    execution, communicator base, lmcache MP wrapper).
- **Runtime environment** (50 vars) and the **vLLM serve command** (64 args)
  baked exactly as the serving composition defines them (`manifest/`).

`patches/` carries the full source diffs (`vllm-source.patch`, 69 files;
`b12x-source.patch`, 8 files) of the overlay against the upstream trees, for
review or cherry-picking.

## Precision

Original E4M3 FP8 weights, BF16 activations, FP32 scales and accumulation, BF16
model boundaries, `fp8_ds_mla` KV — all unchanged from the official release.
DCP partial attention outputs stay FP32 and combine in FP64 on the owner rank
(measured RMS error vs an FP64 reference: 3.60e-4, vs 6.29e-4 for the previous
BF16-combine path). The lossless prefill DMA ring keeps its last
reduce-scatter hop in FP32. No lossy communication, no extra quantization, no
approximate arithmetic. GLP steering is **disabled** in this composition (the
steering basis file ships for completeness but is never loaded).

## Measured performance (reference rig, 9x RTX PRO 6000)

- Decode step at 8k context: **30.56 ms** (~78–83 tok/s single stream with MTP3
  at 2.3–2.6 accepted tokens/step).
- Cold TTFT (window-2 composition): 8K 2.57 s / 128K 46.76 s.
- Prefill batch 4096, mixed-prefill budget 1024, per-request context up to
  **1,048,576 tokens** (KV pool ≈ 1.26 M tokens ≈ 1.2x one full-length request).
- Full decode CUDA graphs for batches [1,2,4,8,16].

Your numbers will depend on your PCIe topology: the collectives assume two
switch groups (ranks 0–3,8 on one chip, 4–7 on the other) sharing a single x16
cascade link, and `B12X_PCIE_DCP_GATHER_GROUPS` is pinned accordingly.

## Verification provenance

On the rig this was qualified for production (2026-10-31): CPU gate 255 tests,
boot-log markers, vision probes 9/9 (OCR, chart, shape, photo, follow-up turn,
base64 image block, tool call), teacher-forced text-path identity vs a text-only
GLM-5.3 build passed with zero regressions at all 15 scored positions (14
bit-exact; the one 8k sequence position where the engine is not run-to-run
deterministic — even against itself across repeats — falls inside the
reference's own band). The engine is **not** run-to-run deterministic at
temperature 0; kernel-level bit-exactness vs FP64 references is the identity
evidence between builds.

The published image itself was checked for file-exactness: every overlay file
sha256-matches the serving build manifest, and the env/command match the
serving container. No GPU boot of the published copy was performed in this
repo's CI (there is none) — first boot on your hardware is your integration
test.

## Repo layout

```
Dockerfile                    # FROM runtime base; COPY overlay; ENV + CMD pinned
overlay/                      # 146 files at their container paths
manifest/                     # sanitized build manifest, runtime env, server args
weights/assemble_weights.py   # composite weights assembly from public HF repos
weights/weights-reference.json# pinned revisions + sha256 of every verified file
patches/                      # full source diffs of the overlay (review/cherry-pick)
run.sh, docker-compose.yml    # run wrappers
NOTICE.md, LICENSE            # Apache-2.0; attribution and weight-license chain
```

## Licensing

Engine-side code in this repo is released under Apache-2.0 (see `LICENSE`).
It is derivative work: the overlay files are ports/edits of
[`vllm`](https://github.com/vllm-project/vllm) (Apache-2.0) and the b12x
runtime trees compiled into the base image; see `NOTICE.md`. The base image is
published as a convenience artifact without an individual license audit of
everything baked into it. Weight licensing follows the upstream chain described
in *The model* above — this repo distributes no model weights.
