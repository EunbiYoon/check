"""Discover LoRA adapters from runs/<session>/lora/<variant>/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import config


def _adapter_ready(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def _read_variant(run_dir: Path) -> str | None:
    info = run_dir / "run_info.json"
    if not info.is_file():
        return run_dir.name if _adapter_ready(run_dir) else None
    manifest = json.loads(info.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        return None
    return manifest.get("resolved_config", {}).get("variant") or run_dir.name


def _variant_adapter_path(variant_dir: Path) -> Path | None:
    if _adapter_ready(variant_dir):
        return variant_dir
    nested = variant_dir / "adapter"
    if _adapter_ready(nested):
        return nested
    return None


def discover_run_adapters(runs_dir: Path | None = None) -> dict[str, Path]:
    """Map variant name -> adapter directory from completed runs."""
    runs_dir = runs_dir or config.RUNS_DIR
    found: dict[str, Path] = {}
    if not runs_dir.is_dir():
        return found

    session_filter = __import__("os").environ.get("RUN_ID")
    sessions = (
        [runs_dir / session_filter]
        if session_filter
        else sorted(p for p in runs_dir.iterdir() if p.is_dir())
    )

    for session in sessions:
        lora_dir = session / "lora"
        if not lora_dir.is_dir():
            continue
        for variant_dir in sorted(lora_dir.iterdir()):
            if variant_dir.name in ("eval_staging", "staging") or not variant_dir.is_dir():
                continue
            variant = _read_variant(variant_dir)
            if not variant:
                continue
            adapter = _variant_adapter_path(variant_dir)
            if adapter is not None:
                found[variant] = adapter
    return found


def discover_published_adapters(lora_dir: Path | None = None) -> dict[str, Path]:
    """Map variant name -> adapter from a lora root (runs/<session>/lora)."""
    lora_dir = lora_dir or config.RUNS_DIR
    found: dict[str, Path] = {}
    if not lora_dir.is_dir():
        return found

    # runs/<session>/lora layout
    if (lora_dir / "lora").is_dir():
        lora_dir = lora_dir / "lora"

    for path in sorted(lora_dir.iterdir()):
        if not path.is_dir() or path.name in ("eval_staging", "staging", "__pycache__"):
            continue
        adapter = _variant_adapter_path(path)
        if adapter is not None:
            found[path.name] = adapter
    return found


def discover_all_adapters(
    *,
    runs_dir: Path | None = None,
    lora_dir: Path | None = None,
) -> dict[str, Path]:
    """Prefer explicit lora_dir; fall back to scanning all sessions."""
    if lora_dir is not None and lora_dir.is_dir():
        adapters = discover_published_adapters(lora_dir)
        if adapters:
            return adapters
    adapters = discover_run_adapters(runs_dir)
    if lora_dir is not None:
        adapters.update(discover_published_adapters(lora_dir))
    return adapters


def prepare_checkpoint_dir(
    out_dir: Path,
    *,
    runs_dir: Path | None = None,
    lora_dir: Path | None = None,
    variants: list[str] | None = None,
    link: bool = True,
) -> dict[str, Path]:
    """Stage adapters under out_dir/<variant> for eval --checkpoint-dir."""
    adapters = discover_all_adapters(runs_dir=runs_dir, lora_dir=lora_dir)
    if variants:
        adapters = {k: v for k, v in adapters.items() if k in variants}

    out_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for variant, src in sorted(adapters.items()):
        dest = out_dir / variant
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink():
                dest.unlink()
            elif dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if link:
            dest.symlink_to(src.resolve())
        else:
            shutil.copytree(src, dest)
        staged[variant] = dest
    return staged


def list_available_variants(
    *,
    runs_dir: Path | None = None,
    lora_dir: Path | None = None,
) -> list[str]:
    return sorted(discover_all_adapters(runs_dir=runs_dir, lora_dir=lora_dir).keys())
