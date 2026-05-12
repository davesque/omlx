"""
JANG / JANGTQ checkpoint loader, used when ``jang_config.json`` is present.

JANG is an adaptive mixed-precision quantization format published by the
JANGQ-AI org (and several downstream forks under e.g. ``dealignai/``).
Unlike stock mlx-vlm quantization it stores per-tensor bit-widths only
implicitly (inferred at load time from the shape relationship between
each ``weight`` and its ``scales``) plus an out-of-band
``jang_config.json`` carrying:

* per-tensor allocation rules (CRITICAL / IMPORTANT / COMPRESS tiers)
* AWQ per-channel scales that need to be applied at forward time
* capabilities metadata (reasoning_parser, tool_parser,
  think_in_template, supports_thinking)

Stock ``mlx_lm.load`` and ``mlx_vlm.utils.load`` have no slot for any of
this, so they silently load the raw ``QuantizedLinear`` modules with
``.bits`` defaulted -- the matmul math then runs against correctly-loaded
``uint32`` tensors but with the wrong bit-width interpretation, producing
garbage outputs without ever raising. The fix is to delegate to
``jang_tools``' loader, which after ``mx.load`` walks every
``QuantizedLinear`` and back-fills ``.bits`` and ``.group_size`` from
the per-tensor metadata.

This module is the single place omlx imports ``jang_tools``. The
package is declared as an optional dependency in ``pyproject.toml``
(``[jang]`` extra); when absent, the ``try_load_*`` helpers report a
clear ``ImportError`` so the calling engine can surface an actionable
error to the user rather than silently producing nonsense.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_local_path(model_name_or_path: str) -> Path | None:
    """
    Map a HuggingFace repo id or local path to a concrete on-disk dir.

    Returns ``None`` when nothing local matches (caller falls back to
    the default load path, which will trigger a download or its own
    resolution).

    The HF cache layout is ``~/.cache/huggingface/hub/models--<org>--<name>/
    snapshots/<sha>/`` with file symlinks; we walk through to the snapshot
    when given an ``org/name`` repo id so the JANG loader sees the same
    directory layout it would after ``huggingface_hub.snapshot_download``.
    """
    candidate = Path(model_name_or_path)
    if candidate.is_dir():
        return candidate

    if "/" in model_name_or_path and not model_name_or_path.startswith(("/", ".")):
        repo_dir = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--{model_name_or_path.replace('/', '--')}"
        )
        snapshots = repo_dir / "snapshots"
        if snapshots.is_dir():
            # Pick the most-recently-modified snapshot. HF caches don't
            # rotate snapshot ids unless the upstream commit changes, so
            # this is effectively "current revision".
            snaps = sorted(
                (p for p in snapshots.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if snaps:
                return snaps[0]

    return None


def looks_like_jang(model_name_or_path: str) -> bool:
    """
    Cheap pre-check: does ``model_name_or_path`` resolve to a directory
    with a ``jang_config.json`` next to its ``config.json``?

    Used by the engine's load path to decide *whether* to import
    ``jang_tools``. We can't just call ``jang_tools.is_jang_model``
    upfront because importing ``jang_tools`` on a non-Apple-Silicon
    machine (or without the optional dep installed) would either fail
    or pull in mlx eagerly. So we do a filesystem-only check first.

    Returns ``False`` for both "definitely not JANG" and "can't tell
    yet" (e.g., the path is a bare repo id with nothing cached). The
    caller should not treat ``False`` as a hard refusal -- it just
    means the JANG loader has nothing to do.
    """
    local = _resolve_local_path(model_name_or_path)
    if local is None:
        return False
    jang_cfg = local / "jang_config.json"
    return jang_cfg.is_file()


def read_capabilities(model_name_or_path: str) -> dict[str, Any] | None:
    """
    Return the ``capabilities`` block from ``jang_config.json``, or
    ``None`` if either the file or the block is absent.

    JANG checkpoints encode reasoning + tool-calling hints in
    ``jang_config.json["capabilities"]``: ``reasoning_parser``,
    ``tool_parser``, ``think_in_template``, ``supports_thinking``. The
    engine wires these into its chat-template + scheduler so that, for
    example, Gemma 4 IT's thinking channel is exposed only when the
    checkpoint actually supports it. Stock mlx-vlm loaders ignore this
    metadata entirely, which is part of why running JANG models
    through the default path produces protocol-mismatched outputs.
    """
    local = _resolve_local_path(model_name_or_path)
    if local is None:
        return None
    jang_cfg_path = local / "jang_config.json"
    if not jang_cfg_path.is_file():
        return None
    try:
        with jang_cfg_path.open() as f:
            jang_cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to read %s", jang_cfg_path)
        return None
    caps = jang_cfg.get("capabilities")
    if isinstance(caps, dict):
        return caps
    return None


def try_load_jang_vlm(model_name_or_path: str) -> tuple[Any, Any] | None:
    """
    Load a JANG-quantized VLM checkpoint into mlx-vlm-compatible objects.

    Returns ``(model, processor)`` -- the same tuple shape that
    ``mlx_vlm.utils.load`` returns -- so the caller can swap this in
    transparently. Returns ``None`` when the path is not a JANG model
    so the caller can fall through to its default loader.

    The heavy lifting is in ``jang_tools.load_jang_vlm_model``:

    * Reads ``jang_config.json`` for tier-bit assignments + AWQ scales.
    * Calls ``mx.load`` (or the v1 repack path) to materialize the
      tensors.
    * Walks the loaded ``QuantizedLinear`` modules and back-fills
      ``.bits`` and ``.group_size`` from the per-tensor metadata --
      this is the step stock mlx-vlm omits.
    * Installs AWQ per-channel input scales as a pre-multiplication
      hook on each affected layer (the standard AWQ folding:
      ``Y = (X / diag(s)) @ dequant(Q)``).

    Raises ``ImportError`` with a clear install hint when the optional
    ``jang`` dependency is missing, so the engine error path can guide
    the user to ``pip install 'omlx[jang]'`` instead of failing with
    an opaque ``ModuleNotFoundError``.
    """
    if not looks_like_jang(model_name_or_path):
        return None

    local = _resolve_local_path(model_name_or_path)
    assert local is not None  # looks_like_jang already proved this

    try:
        from jang_tools.loader import load_jang_vlm_model
    except ImportError as exc:
        raise ImportError(
            f"JANG checkpoint detected at {local} but the 'jang' optional "
            "dependency is not installed. Install with "
            "`pip install 'omlx[jang]'` (PyPI package name is `jang`, "
            "import name is `jang_tools`)."
        ) from exc

    logger.info("JANG VLM checkpoint detected at %s — loading via jang_tools", local)
    model, processor = load_jang_vlm_model(local)
    logger.info("JANG VLM load complete (model_type=%s)",
                getattr(getattr(model, "config", None), "model_type", "<unknown>"))
    return model, processor


def try_load_jang_text(model_name_or_path: str) -> tuple[Any, Any] | None:
    """
    Load a text-only JANG checkpoint via ``jang_tools.load_jang_model``.

    Returns ``(model, tokenizer)`` -- the same shape ``mlx_lm.load``
    returns -- or ``None`` when the path is not a JANG model.

    Used by the text-only batched engine. The shared infrastructure
    with :func:`try_load_jang_vlm` (path resolution, capability
    surfacing, import gating) lives in this module precisely so the
    two engine entry points stay one-liners.
    """
    if not looks_like_jang(model_name_or_path):
        return None

    local = _resolve_local_path(model_name_or_path)
    assert local is not None

    try:
        from jang_tools.loader import load_jang_model
    except ImportError as exc:
        raise ImportError(
            f"JANG checkpoint detected at {local} but the 'jang' optional "
            "dependency is not installed. Install with "
            "`pip install 'omlx[jang]'`."
        ) from exc

    logger.info("JANG text checkpoint detected at %s — loading via jang_tools", local)
    return load_jang_model(local)
