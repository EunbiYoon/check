#!/usr/bin/env python3
"""Build table artifacts from one session's per-variant metric JSON."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import config
from eval.table.build_latex_md import write_latex_md
from eval.table.run_title import infer_train_epochs
from history.paths import active_session_id, write_latest_pointer


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _paper_refs() -> dict:
    return {f"table{i}": getattr(config, f"PAPER_TABLE{i}") for i in range(1, 8)}


def _training_manifest(lora_dir: Path) -> dict:
    manifest: dict[str, dict] = {}
    if not lora_dir.is_dir():
        return manifest
    for variant_dir in sorted(path for path in lora_dir.iterdir() if path.is_dir()):
        info_path = variant_dir / "run_info.json"
        info = {}
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                info = {}
        manifest[variant_dir.name] = {
            "completed": info.get("status") == "completed",
            "adapter": str(variant_dir / "adapter"),
            "run_info": str(info_path),
        }
    return manifest


def _result_md(suite: dict, variants: list[str]) -> str:
    lines = [
        f"# Evaluation tables — {suite['run_id']}",
        "",
        f"Generated: {suite['finished_at']}",
        "",
        "Pipeline: `rollouts/` → `metrics/` → `tables/`",
        "",
        "## Table 1 — Headline metrics",
        "",
        "| Variant | fc(ID) | fc(HO) | CR(ID) | CR(HO) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in variants:
        row = suite["table1"][variant]
        lines.append(
            f"| {variant} | {row.get('fc_id', 0):.3f} | {row.get('fc_ho', 0):.3f} | "
            f"{row.get('cr_id', 0):.3f} | {row.get('cr_ho', 0):.3f} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "- `table1.json` … `table7.json`: paper table data",
        "- `suite.json`: combined metrics and metadata",
        "- `paper_refs.json`: fixed paper reference values",
        "- `latex.md`: LaTeX table output",
        "",
    ]
    return "\n".join(lines)


def _run_id(explicit: str | None) -> str | None:
    if explicit or active_session_id():
        return explicit or active_session_id()
    latest = config.RUNS_DIR / "latest.json"
    if latest.is_file():
        try:
            return str(json.loads(latest.read_text(encoding="utf-8"))["run_id"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="metrics/*.json -> tables/*")
    parser.add_argument("--run-id", default=None, help="Session under runs/; default: RUN_ID or latest")
    parser.add_argument("--variants", default="", help="Comma-separated; default: every metric JSON")
    parser.add_argument("--games", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-merge", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    run_id = _run_id(args.run_id)
    if run_id is None:
        parser.error("--run-id is required when RUN_ID and runs/latest.json are unavailable")
    session_dir = config.RUNS_DIR / run_id
    eval_dir = session_dir / "eval"
    metrics_dir = eval_dir / "metrics"
    tables_dir = eval_dir / "tables"
    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    paths = [metrics_dir / f"{variant}.json" for variant in requested]
    if not requested:
        paths = sorted(path for path in metrics_dir.glob("*.json") if path.name != "_config.json")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error("missing metric file(s): " + ", ".join(str(path) for path in missing))
    if not paths:
        parser.error(f"no metric JSON files found under {metrics_dir}")

    rows = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in paths}
    ok_variants = [variant for variant, row in rows.items() if "error" not in row]
    rollout_manifest_path = eval_dir / "rollouts" / "_config.json"
    rollout_manifest = (
        json.loads(rollout_manifest_path.read_text(encoding="utf-8"))
        if rollout_manifest_path.is_file()
        else {}
    )
    manifest = _training_manifest(session_dir / "lora")
    STUDENT_MODEL = rollout_manifest.get("STUDENT_MODEL") or config.PAPER_STUDENT_MODEL
    hyperparams = {
        "STUDENT_MODEL": STUDENT_MODEL,
        "lora_r": config.PAPER_LORA_R,
        "lora_alpha": config.PAPER_LORA_ALPHA,
        "lora_dropout": config.LORA_DROPOUT,
        "dpo_beta": config.DPO_BETA,
        "merge_alpha": config.MERGE_ALPHA,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.NUM_EPOCHS,
        "train_epochs": infer_train_epochs(session_dir / "lora"),
        "episodes_per_env": rollout_manifest.get("episodes_per_game", config.EPISODES_PER_ENV),
        "gradient_accumulation": config.GRADIENT_ACCUMULATION,
        "checkpoint_dir": str(session_dir / "lora"),
        "variants_evaluated": ok_variants,
        "paper_mode": True,
        "use_4bit": config.USE_4BIT,
    }
    suite = {
        "run_id": run_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": 0.0,
        "timings_seconds": {},
        "hyperparams": hyperparams,
        "table1": {v: {k: rows[v].get(k) for k in ("fc_id", "fc_ho", "cr_id", "cr_ho")} for v in ok_variants},
        "table2": {v: rows[v].get("per_game_cr", {}) for v in ok_variants},
        "table3": {v: rows[v].get("per_axis_cr", {}) for v in ok_variants},
        "table4": {v: rows[v].get("per_game_fc", {}) for v in ok_variants},
        "table5_manifest": manifest,
        "table6": {"paper_reference": config.PAPER_TABLE6},
        "table7": {
            v: {
                "ac": rows[v].get("per_game_ac", {}),
                "fc": rows[v].get("per_game_fc", {}),
                "exploitability": rows[v].get("per_game_exploitability", {}),
            }
            for v in ok_variants
        },
        "variants": ok_variants,
        "rows": rows,
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    save_json(tables_dir / "suite.json", suite)
    save_json(tables_dir / "paper_refs.json", _paper_refs())
    for number in range(1, 8):
        key = f"table{number}"
        save_json(tables_dir / f"{key}.json", suite.get(key, {}))
    (tables_dir / "result.md").write_text(_result_md(suite, ok_variants), encoding="utf-8")
    write_latex_md(tables_dir, suite, ok_variants=ok_variants)
    write_latest_pointer(run_id)
    print(f"{len(ok_variants)} variants -> {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
