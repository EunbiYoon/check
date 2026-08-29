"""Eval output paths under rollouts -> metrics -> tables stage folders."""

from __future__ import annotations

from pathlib import Path

from history.paths import (
    new_eval_dir,
    resolve_eval_dir,
    utc_stamp,
    write_latest_pointer as write_session_latest,
)


def new_run_dir(run_id: str | None = None) -> Path:
    return new_eval_dir(run_id)


def resolve_run_dir(run_dir: Path) -> Path:
    return resolve_eval_dir(run_dir)


def table_out_path(run_dir: Path, table: int, *, variant: str | None = None) -> Path:
    if variant:
        return run_dir / "tables" / f"table{table}_{variant}.json"
    return run_dir / "tables" / f"table{table}.json"


def write_latest_pointer(run_dir: Path) -> None:
    """Update runs/latest.json for a runs/<session>/eval directory."""
    eval_dir = resolve_eval_dir(run_dir)
    write_session_latest(eval_dir.parent.name)
