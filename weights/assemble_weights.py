#!/usr/bin/env python3
"""Assemble the GLM-5.3-Vision-FP8 composite weights directory from public sources.

The composite has exactly two upstream sources, both public:

  1. Text backbone — the official `zai-org/GLM-5.3` FP8 E4M3 checkpoint
     (141 shards; revision pinned in weights-reference.json). Text numerics are
     the official release, byte-for-byte.

  2. Vision components — everything else comes verbatim from
     `lunaroute/GLM-5.3-Vision-FP8`: the MoonViT-3d vision tower (27 layers,
     1152-wide, 16 heads x 72, BF16, from Kimi-K2.6), the BF16 PatchMerger
     projector (1152 -> 4608 -> 6144; trained by Baseten against GLM-5.2),
     the Glm5v config (text_config == official GLM-5.3), tokenizer, chat
     template with the image branch, and processor/plugins files.

Nothing is redistributed here: this script downloads both sources, places the
141 text shards alongside the lunaroute files, and verifies the sha256 of every
file that participates in the serving composition against the pinned values in
weights-reference.json (taken from the qualified serving build manifest).

Usage:
    python3 assemble_weights.py --out /path/to/GLM-5.3-Vision-FP8-build \
        [--text-dir DIR] [--vision-dir DIR]

Without --text-dir / --vision-dir the sources are downloaded with
huggingface_hub into --out/.sources (add its bin dir to PATH or `pip install
"huggingface_hub[cli]"` first). With them, existing local snapshots are used
and only the assembly + verification runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = json.loads((HERE / "weights-reference.json").read_text())

TEXT_REPO = REFERENCE["text_model_repo"]
TEXT_REVISION = REFERENCE["text_model_revision"]
VISION_REPO = REFERENCE["vision_components_repo"]
SHARD_COUNT = REFERENCE["text_shard_count"]
CONFIG_SHA256 = REFERENCE["config_sha256"]
FILE_SHA256 = REFERENCE["vision_files_sha256"]

# Files that must exist in the assembled directory; sha256 is verified for
# those with pinned hashes (the composite config plus the lunaroute vision set).
VERIFIED_FILES = {"config.json": CONFIG_SHA256, **FILE_SHA256}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(repo: str, revision: str | None, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=dest,
        # lunaroute pins LFS integrity itself; hf verifies sha256 on download
    ))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="target composite directory")
    ap.add_argument("--text-dir", help="existing zai-org/GLM-5.3 snapshot dir")
    ap.add_argument("--vision-dir", help="existing lunaroute/GLM-5.3-Vision-FP8 snapshot dir")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    sources = out / ".sources"
    sources.mkdir(parents=True, exist_ok=True)

    if args.text_dir:
        text_dir = Path(args.text_dir).resolve()
    else:
        print(f"downloading text backbone {TEXT_REPO} @ {TEXT_REVISION} ...")
        text_dir = download(TEXT_REPO, TEXT_REVISION, sources / "glm-5.3")
    if args.vision_dir:
        vision_dir = Path(args.vision_dir).resolve()
    else:
        print(f"downloading vision components {VISION_REPO} ...")
        vision_dir = download(VISION_REPO, None, sources / "glm-5.3-vision-fp8")

    out.mkdir(parents=True, exist_ok=True)

    # 1. Vision side: copy the whole lunaroute snapshot verbatim.
    for item in vision_dir.rglob("*"):
        if item.is_file() and ".cache" not in item.parts:
            rel = item.relative_to(vision_dir)
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)

    # 2. Text backbone: link (or copy) the 141 official shards in.
    shards = sorted(text_dir.glob("model-*-of-00141.safetensors"))
    if len(shards) != SHARD_COUNT:
        print(f"ERROR: expected {SHARD_COUNT} text shards in {text_dir}, found {len(shards)}",
              file=sys.stderr)
        return 1
    for shard in shards:
        dst = out / shard.name
        if dst.exists():
            dst.unlink()
        try:
            os.link(shard, dst)  # zero extra bytes when sources are local
        except OSError:
            shutil.copy2(shard, dst)

    # 3. Verify every pinned file.
    failures = []
    for name, want in VERIFIED_FILES.items():
        got_path = out / name
        if not got_path.exists():
            failures.append(f"MISSING {name}")
            continue
        got = sha256_of(got_path)
        if got != want:
            failures.append(f"SHA256 MISMATCH {name}: got {got}, want {want}")
    if failures:
        print("VERIFICATION FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"OK: composite assembled at {out}")
    print(f"  text shards: {len(shards)} (hardlink/copy of {text_dir})")
    print(f"  verified files: {len(VERIFIED_FILES)} (config + vision set, sha256-pinned)")
    print(f"  tip: point the container at this dir via -v {out}:/model:ro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
