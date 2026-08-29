"""Unified training and rollout -> metrics -> tables evaluation layout."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config import PROJECT_ROOT, RUNS_DIR


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def active_session_id() -> str | None:
    return os.environ.get("RUN_ID")


def ensure_session(session_id: str | None = None) -> str:
    sid = session_id or active_session_id() or utc_stamp()
    os.environ["RUN_ID"] = sid
    (RUNS_DIR / sid / "lora").mkdir(parents=True, exist_ok=True)
    return sid


def session_dir(session_id: str | None = None) -> Path:
    sid = session_id or active_session_id() or ensure_session()
    return RUNS_DIR / sid


def lora_root(session_id: str | None = None) -> Path:
    root = session_dir(session_id) / "lora"
    root.mkdir(parents=True, exist_ok=True)
    return root


def lora_variant_dir(variant: str, session_id: str | None = None) -> Path:
    d = lora_root(session_id) / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_lora_variant_dir(variant: str, session_id: str | None = None) -> Path:
    ensure_session(session_id)
    run_dir = lora_variant_dir(variant, session_id)
    (run_dir / "adapter").mkdir(parents=True, exist_ok=True)
    return run_dir


def eval_run_dir(session_id: str | None = None, *, create: bool = True) -> Path:
    sid = session_id or active_session_id() or ensure_session()
    d = RUNS_DIR / sid / "eval"
    if create:
        (d / "rollouts").mkdir(parents=True, exist_ok=True)
        (d / "tables").mkdir(parents=True, exist_ok=True)
        (d / "metrics").mkdir(parents=True, exist_ok=True)
    return d


def new_eval_dir(run_id: str | None = None) -> Path:
    if run_id:
        os.environ.setdefault("RUN_ID", run_id)
    return eval_run_dir(run_id, create=True)


def _legacy_eval_path(p: Path) -> Path | None:
    """Map eval/runs/<name> -> runs/<name>/eval."""
    try:
        rel = p.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "eval" and parts[1] == "runs":
        return RUNS_DIR / parts[2] / "eval"
    return None


def resolve_eval_dir(path_or_id: Path | str) -> Path:
    p = Path(path_or_id)
    if not p.is_absolute():
        p = PROJECT_ROOT / p

    legacy = _legacy_eval_path(p)
    if legacy is not None:
        p = legacy

    p = p.resolve()
    if p.name == "eval" and p.parent.parent == RUNS_DIR.resolve():
        return p
    if (p / "eval").is_dir():
        return p / "eval"
    if (p / "rollouts").is_dir() or (p / "metrics").is_dir() or (p / "tables").is_dir():
        return p
    return (RUNS_DIR / p.name / "eval").resolve()


def write_latest_pointer(session_id: str | None = None) -> None:
    sid = session_id or active_session_id()
    if not sid:
        return
    root = RUNS_DIR.name  # "history" after the rename, "runs" before it
    payload = {
        "run_id": sid,
        "path": f"{root}/{sid}",
        "lora": f"{root}/{sid}/lora",
        "eval": f"{root}/{sid}/eval",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    latest = RUNS_DIR / "latest.json"
    temporary = RUNS_DIR / f".latest.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(latest)
