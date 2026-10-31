"""GLP projective steering for kimi_k3 — vendored from msuiche/weightless.

Self-contained port of weightless_steer.container (GGUF reader + spec gates)
and weightless_steer.core (CUDA-graph-safe projective apply), scalar lane
only (no per-request control plane). Source: github.com/msuiche/weightless
vllm-plugin/weightless_steer/{container.py,core.py}, retrieved 2026-10-15.

Semantics (identical to the plugin lane):
- h <- h - alpha * (h . d) d at the post-layer residual stream, per GLOBAL
  layer id, from a dense zero-padded [num_layers, 1, hidden] fp-model-dtype
  stack (zero rows are numeric no-ops on unsteered layers).
- alpha is a registered tensor buffer, not a Python float.
- Fail-closed: WEIGHTLESS_STEER_PATH set but unreadable / wrong mode /
  wrong hook / wrong width / out-of-range layers -> raise at boot.
- WEIGHTLESS_STEER_PATH unset -> wire_steering returns None and the model
  skips wiring entirely, so a disabled boot captures exactly the stock
  graph (this fork runs the layer loop under CUDA-graph capture, not
  dynamo, so the Python gate is resolved at capture time).

Env: WEIGHTLESS_STEER_PATH (required to enable), WEIGHTLESS_STEER_ALPHA
(default: file glp.alpha_default, else 1.0), WEIGHTLESS_STEER_LAYERS
(optional comma list), WEIGHTLESS_STEER_HOOK (must equal
"residual_stream_post_layer" if set).
"""
from __future__ import annotations

import functools
import logging
import math
import os
import re
import struct

import numpy as np
import torch

logger = logging.getLogger(__name__)

HOOK = "residual_stream_post_layer"

_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FM = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
       7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

_DIRECTION_RE = re.compile(r"direction\.(\d+)(?:\.(\d+))?")


def _int_meta(meta: dict, key: str, default: int) -> int:
    v = meta.get(key)
    if v is None or v == "":
        return default
    return int(float(str(v)))


def read_gguf_cvec(path: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Minimal GGUF v3 reader for control vectors (F32 tensors only)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"GGUF":
        raise ValueError(f"{path}: bad magic, not a GGUF file")
    ver, n_tensors, n_kv = struct.unpack_from("<IQQ", data, 4)
    if ver != 3:
        raise ValueError(f"{path}: GGUF version {ver}, expected 3")
    off = 24

    def rd_string(o):
        (n,) = struct.unpack_from("<Q", data, o)
        o += 8
        return data[o:o + n].decode("utf-8"), o + n

    meta = {}
    for _ in range(n_kv):
        key, off = rd_string(off)
        (vtype,) = struct.unpack_from("<I", data, off)
        off += 4
        if vtype == 8:  # string
            val, off = rd_string(off)
        elif vtype == 9:  # array
            etype, cnt = struct.unpack_from("<IQ", data, off)
            off += 12
            if etype == 8:
                arr = []
                for _ in range(cnt):
                    s, off = rd_string(off)
                    arr.append(s)
                val = arr
            else:
                val = struct.unpack_from(f"<{cnt}{_FM[etype][1]}", data, off)
                off += _SZ[etype] * cnt
        else:
            (val,) = struct.unpack_from(_FM[vtype], data, off)
            off += _SZ[vtype]
        meta[key] = val

    infos = []
    for _ in range(n_tensors):
        name, off = rd_string(off)
        (nd,) = struct.unpack_from("<I", data, off)
        off += 4
        dims = struct.unpack_from(f"<{nd}Q", data, off)
        off += 8 * nd
        (tt,) = struct.unpack_from("<I", data, off)
        off += 4
        (toff,) = struct.unpack_from("<Q", data, off)
        off += 8
        infos.append((name, dims, tt, toff))

    align = meta.get("general.alignment", 32)
    if isinstance(align, str):
        align = int(align)
    base = (off + align - 1) // align * align
    tensors = {}
    for name, dims, tt, toff in infos:
        if tt != 0:
            raise ValueError(f"{path}: {name} is not F32 (ggml type {tt})")
        n = 1
        for d in dims:
            n *= d
        tensors[name] = np.frombuffer(
            data, dtype="<f4", count=n, offset=base + toff
        ).copy()
    return meta, tensors


def load_control_vector(
    path: str,
    hook: str = HOOK,
) -> tuple[dict, dict[int, torch.Tensor]]:
    """Load a projective GLP control vector: (metadata, {layer: tensor}).

    Enforces the spec gates (msuiche/weightless docs/spec/GLP.md): mode
    present and "project", hook_point exact match, direction.0 rejected,
    integer layer ids, uniform width, spec_version 1/2, rank-1 only on
    this lane, layer_ids_zero_based cross-check when present.
    """
    meta, tensors = read_gguf_cvec(path)

    mode = meta.get("glp.mode")
    if mode is None:
        raise ValueError(f"{path}: no glp.mode; refusing to guess")
    if mode != "project":
        raise ValueError(
            f"{path}: glp.mode={mode!r}, only projective ablation "
            "(h -= alpha*(h.d)d) is implemented. Refusing."
        )

    file_hook = meta.get("glp.hook_point")
    if file_hook != hook:
        raise ValueError(
            f"{path}: glp.hook_point={file_hook!r} != this hook ({hook}). "
            "Refusing to apply at the wrong site."
        )

    version = _int_meta(meta, "glp.spec_version", 1)
    if version not in (1, 2):
        raise ValueError(
            f"{path}: glp.spec_version={meta.get('glp.spec_version')!r} "
            "not implemented (1 and 2 are). Refusing."
        )
    rank_declared = _int_meta(meta, "glp.rank", 1)
    if version == 1 and (rank_declared != 1
                         or "glp.dir_scales" in meta
                         or "glp.layer_scales" in meta):
        raise ValueError(
            f"{path}: glp.rank={rank_declared} / alpha-scaling keys "
            "require spec_version 2 (file declares 1). Refusing."
        )
    # This lane applies ONE scalar alpha to every steered layer: alpha
    # multipliers would be served at the wrong strength silently.
    scales = sorted(set(meta) & {"glp.dir_scales", "glp.layer_scales"})
    if scales:
        raise ValueError(
            f"{path}: carries {', '.join(scales)}; this lane does not "
            "implement alpha multipliers. Refusing."
        )

    derived_at = meta.get("glp.derived_at")
    if derived_at and derived_at != file_hook:
        logger.warning(
            "%s: transferred vector -- glp.derived_at=%r, apply hook %r",
            path, derived_at, file_hook,
        )

    logger.info(
        "weightless GLP vector: mode=%s spec_version=%s base_model=%s rev=%s "
        "alpha_default=%s sha=%s",
        mode,
        meta.get("glp.spec_version", "?"),
        meta.get("general.base_model.0.name") or meta.get("glp.base_model"),
        str(meta.get("general.base_model.0.version")
            or meta.get("glp.base_revision") or "?")[:12],
        meta.get("glp.alpha_default", "?"),
        str(meta.get("glp.content_sha256", "?"))[:16],
    )

    grouped = {}
    for name, arr in tensors.items():
        m = _DIRECTION_RE.fullmatch(name)
        if m is None:
            if name.startswith("direction."):
                raise ValueError(
                    f"{path}: malformed tensor name {name!r}"
                )
            continue
        idx, j = int(m.group(1)), int(m.group(2) or 0)
        if idx < 1:
            raise ValueError(f"{path}: {name} invalid; direction.0 rejected")
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if not np.isfinite(arr).all() or not np.any(arr):
            raise ValueError(f"{path}: {name} must be finite and nonzero")
        grouped.setdefault(idx, {})[j] = torch.from_numpy(arr.copy())
    if not grouped:
        raise ValueError(f"{path}: no direction.<N> tensors found")

    out = {}
    for idx, js in grouped.items():
        if sorted(js) != list(range(len(js))):
            raise ValueError(
                f"{path}: layer {idx}: direction indices {sorted(js)} "
                "are not 0..k-1 -- refusing a partial subspace"
            )
        stack = torch.stack([js[j] for j in sorted(js)])
        out[idx] = stack[0] if len(js) == 1 else stack
    widths = {v.shape[-1] for v in out.values()}
    if len(widths) != 1:
        raise ValueError(
            f"{path}: inconsistent n_embd across directions: {widths}"
        )
    ks = {(v.shape[0] if v.dim() > 1 else 1) for v in out.values()}
    if len(ks) != 1:
        raise ValueError(
            f"{path}: direction count differs across layers: {sorted(ks)}"
        )
    k = ks.pop()
    if k != rank_declared:
        raise ValueError(
            f"{path}: glp.rank declares {rank_declared} but tensors carry "
            f"{k} direction(s) per layer. Refusing."
        )
    if k > 1:
        raise ValueError(
            f"{path}: rank-{k} (subspace) vector -- this lane implements "
            "rank 1 only. Refusing a partial apply."
        )

    declared = meta.get("glp.layer_ids_zero_based")
    if declared:
        try:
            want = sorted(int(x) for x in declared.split(",") if x.strip())
        except ValueError:
            raise ValueError(
                f"{path}: glp.layer_ids_zero_based is not a comma list: "
                f"{declared!r}"
            ) from None
        if want and want != sorted(out):
            raise ValueError(
                f"{path}: glp.layer_ids_zero_based declares {want[0]}.."
                f"{want[-1]} ({len(want)}) but tensors resolve to "
                f"{sorted(out)[0]}..{sorted(out)[-1]} ({len(out)}). "
                "Refusing to steer the wrong layers."
            )
    logger.info(
        "weightless GLP vector: %d steered layers, n_embd=%d, rank=%d, "
        "layers %d..%d",
        len(out), next(iter(widths)), k, min(out), max(out),
    )
    return meta, out


class SteeringCore:
    """A loaded GLP vector plus the buffers it is applied from (scalar lane)."""

    def __init__(self, dirs, alpha, num_layers, hidden_size, path=None):
        self.dirs = dirs
        self.alpha = float(alpha)
        if not math.isfinite(self.alpha):
            raise ValueError("steering alpha must be finite")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.path = path
        self._owner = None

    @classmethod
    def from_env(cls, *, num_layers, hidden_size):
        """Build a core from WEIGHTLESS_STEER_*, or None if steering is off.

        Fail-closed: WEIGHTLESS_STEER_PATH set but the file missing,
        non-project, wrong hook, wrong width, or out of range raises -- a
        boot asked for steering must not serve unsteered.
        """
        path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
        if not path:
            return None

        env_hook = (os.environ.get("WEIGHTLESS_STEER_HOOK") or HOOK).strip()
        if env_hook != HOOK:
            raise RuntimeError(
                f"WEIGHTLESS_STEER_HOOK={env_hook!r} is not implemented; "
                f"the only site here is {HOOK}. Refusing."
            )

        try:
            meta, raw = load_control_vector(path, hook=HOOK)

            alpha_env = os.environ.get("WEIGHTLESS_STEER_ALPHA", "").strip()
            alpha = (float(alpha_env) if alpha_env
                     else float(meta.get("glp.alpha_default", 1.0)))
            if not math.isfinite(alpha):
                raise ValueError("steering alpha must be finite")

            want = os.environ.get("WEIGHTLESS_STEER_LAYERS", "").strip()
            selected = (
                {int(t) for t in want.replace(" ", "").split(",") if t}
                if want else None
            )

            dirs = {}
            for layer_id, vec in raw.items():
                layer_id = int(layer_id)
                if selected is not None and layer_id not in selected:
                    continue
                vec = vec.detach().to(torch.float32).reshape(-1)
                if vec.numel() != hidden_size:
                    raise RuntimeError(
                        f"steering vector layer {layer_id} width "
                        f"{vec.numel()} != {hidden_size}"
                    )
                # Normalise in float64: a non-unit export must not silently
                # scale alpha; nonfinite elements make the norm nonfinite.
                norm = vec.double().norm()
                if not torch.isfinite(norm) or norm <= 0:
                    raise ValueError(
                        f"{path}: direction layer {layer_id} must be "
                        "finite and nonzero"
                    )
                dirs[layer_id] = (vec.double() / norm).float()

            out_of_range = sorted(
                int(k) for k in raw if not 0 <= int(k) < num_layers
            )
            if out_of_range:
                raise RuntimeError(
                    f"{path}: direction layers {out_of_range} out of range "
                    f"for this model ({num_layers} layers)"
                )
            if not dirs:
                raise RuntimeError(
                    f"WEIGHTLESS_STEER_PATH={path} matched no layers; "
                    "refusing to run unsteered"
                )
        except Exception as exc:
            logger.error("GLP steering load failed (%s); failing closed", exc)
            raise

        logger.info(
            "weightless GLP steering active: hook=%s alpha=%.3f "
            "layers=%d..%d (%d) width=%d",
            HOOK, alpha, min(dirs), max(dirs), len(dirs), hidden_size,
        )
        return cls(dirs, alpha, num_layers, hidden_size, path=path)

    def register_buffers(self, module, dtype):
        """Register (and fill) _steer_stack / _steer_alpha on `module`.

        Dense [num_layers, 1, hidden_size] stack indexed by GLOBAL layer id
        plus a scalar alpha; both non-persistent so load_weights never sees
        them as unexpected state-dict keys.
        """
        if (not math.isfinite(self.alpha)
                or abs(self.alpha) > torch.finfo(dtype).max):
            raise ValueError(
                f"steering alpha {self.alpha!r} must be finite and "
                f"representable in {dtype}"
            )
        for layer_id, vec in self.dirs.items():
            norm = vec.double().norm()
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError(
                    f"steering direction layer {layer_id} must be finite "
                    "and nonzero"
                )
            if not torch.isfinite(vec.to(dtype)).all():
                raise ValueError(
                    f"steering direction layer {layer_id} is not "
                    f"representable in {dtype}"
                )
        module.register_buffer(
            "_steer_stack",
            torch.zeros(self.num_layers, 1, self.hidden_size, dtype=dtype),
            persistent=False,
        )
        module.register_buffer(
            "_steer_alpha",
            torch.zeros((), dtype=dtype),
            persistent=False,
        )
        for layer_id, vec in self.dirs.items():
            module._steer_stack[layer_id, 0] = vec.to(dtype)
        module._steer_alpha.fill_(self.alpha)
        self._owner = module

    def apply(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        """h <- h - alpha * (h . d) d at GLOBAL layer id `layer_idx`.

        Branch-free: layers we do not steer have a zero stack row, making
        this a numeric no-op there.
        """
        owner = self._owner
        dirs = owner._steer_stack[layer_idx]
        coef = torch.einsum("...h,kh->...k", h, dirs)
        proj = torch.einsum("...k,kh->...h", coef, dirs)
        return h - owner._steer_alpha * proj

    def apply_post_layer(
        self, layer_idx: int, hidden_states: torch.Tensor, side_stream: torch.Tensor
    ) -> torch.Tensor:
        """The post-layer projection of the model's ``_steer_post_layer``:
        ``h = hidden_states + side_stream``, ``h' = h - alpha (h . d) d`` and
        the write-back ``h' - side_stream``, as one fused kernel when
        ``WEIGHTLESS_STEER_FUSED`` is set (default on) and the operands are
        2-D bf16/fp16 CUDA rows with a single direction per layer; otherwise
        the five-op torch chain.

        Fused arithmetic: every term is formed in fp32 from the bf16 inputs,
        the dot product is a fixed-order fp32 reduction, and the result
        ``hidden_states - alpha * (h . d) * d`` is rounded once into the output
        dtype. The torch chain rounds after each of its five ops (the sum
        ``h``, both einsum outputs, the scaled projection and the final
        subtraction), so on steered layers the fused result is the more precise
        of the two; on layers whose stack row is zero it returns
        ``hidden_states`` bit-for-bit, where the chain returns
        ``bf16(bf16(hidden + side) - side)``.
        """
        owner = self._owner
        if fused_steering_enabled() and _fused_applies(
            hidden_states, side_stream, owner._steer_stack
        ):
            return _fused_post_layer(
                hidden_states,
                side_stream,
                owner._steer_stack[layer_idx, 0],
                owner._steer_alpha,
            )
        h = hidden_states + side_stream
        return self.apply(layer_idx, h) - side_stream


def fused_steering_enabled() -> bool:
    """``WEIGHTLESS_STEER_FUSED`` (default ``1``): one Triton launch per
    steered layer instead of the five-op torch chain."""
    return os.environ.get("WEIGHTLESS_STEER_FUSED", "1").strip() == "1"


def _fused_applies(
    hidden_states: torch.Tensor, side_stream: torch.Tensor, stack: torch.Tensor
) -> bool:
    return (
        hidden_states.ndim == 2
        and hidden_states.shape == side_stream.shape
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and hidden_states.dtype == side_stream.dtype == stack.dtype
        and hidden_states.is_cuda
        and side_stream.is_cuda
        and hidden_states.stride(-1) == 1
        and side_stream.stride(-1) == 1
        and stack.ndim == 3
        and stack.shape[1] == 1
        and stack.shape[2] == hidden_states.shape[1]
        and stack.is_contiguous()
    )


def _block_d(hidden_size: int) -> int:
    import triton

    return max(128, triton.next_power_of_2(hidden_size))


@functools.lru_cache(maxsize=1)
def _fused_kernel():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _glp_post_layer_kernel(
        hidden_ptr,
        side_ptr,
        dir_ptr,
        alpha_ptr,
        out_ptr,
        stride_hidden_m,
        stride_side_m,
        stride_out_m,
        hidden_size: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < hidden_size
        hidden = tl.load(
            hidden_ptr + row * stride_hidden_m + offs, mask=mask, other=0.0
        ).to(tl.float32)
        side = tl.load(side_ptr + row * stride_side_m + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        direction = tl.load(dir_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        alpha = tl.load(alpha_ptr).to(tl.float32)
        h = hidden + side
        coef = tl.sum(h * direction, axis=0)
        out = hidden - alpha * coef * direction
        tl.store(
            out_ptr + row * stride_out_m + offs,
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    return _glp_post_layer_kernel


def _fused_post_layer(
    hidden_states: torch.Tensor,
    side_stream: torch.Tensor,
    direction: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(hidden_states)
    rows, hidden_size = hidden_states.shape
    if rows == 0:
        return out
    kernel = _fused_kernel()
    kernel[(rows,)](
        hidden_states,
        side_stream,
        direction,
        alpha,
        out,
        hidden_states.stride(0),
        side_stream.stride(0),
        out.stride(0),
        hidden_size=hidden_size,
        BLOCK_D=_block_d(hidden_size),
        num_warps=8,
    )
    return out


def wire_steering(module, *, num_layers, hidden_size, dtype):
    """Env-gated wiring entry point called from KimiLinearModel.__init__.

    Returns the SteeringCore (or None when WEIGHTLESS_STEER_PATH is unset,
    in which case no buffers are registered and the model captures exactly
    the stock graph).
    """
    core = SteeringCore.from_env(num_layers=num_layers, hidden_size=hidden_size)
    if core is None:
        return None
    core.register_buffers(module, dtype)
    return core
