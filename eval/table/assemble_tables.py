#!/usr/bin/env python3
"""Assemble Tables 1–7 from the best error-free per-variant metrics on disk.

Picks the newest complete metrics/*.json per variant (prefers 3B runs), merges
frontier model logs where complete, and builds suite.json, table JSONs,
result.md, and latex.md under eval/runs/<out>/.

  python -m eval.table.assemble_tables
  python -m eval.table.assemble_tables --out assembled_tables
  python -m eval.table.assemble_tables --variants base,core,aux,all,rw,merge,filter_on,filter_off
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from eval.metric.metrics import _axis_for_opponent
from eval.rollout.resume import ParsedEpisode, parse_eval_log
from eval.table.run_eval_suite import (
    _hyperparams,
    _paper_refs,
    _training_manifest,
    build_result_md,
)
from eval.table.build_latex_md import write_latex_md
from eval.table.paths import new_run_dir, write_latest_pointer
from eval.table.run_paper_tables import _maybe_merge
from eval.rollout.checkpoints import prepare_checkpoint_dir
from eval.table.run_table import save_json

PAPER_VARIANTS = ["base", "core", "aux", "all", "rw", "merge", "filter_on", "filter_off"]
FRONTIER_KEYS = ("llama_8b", "gemma_27b", "llama_70b")
# Table 4 frontier CR benchmark and Table 5 reasoning-action coupling summary.
PAPER_TABLE4_FRONTIER_MODELS = ("haiku", "gemma_27b", "llama_70b", "llama_8b")
TABLE5_TRAINED_VARIANTS = ("core", "aux", "all", "rw", "merge")
TABLE5_FRONTIER_ORDER = ("gemma_27b", "llama_8b", "llama_70b", "haiku")
TABLE5_FRONTIER_LABELS = {
    "haiku": "Bedrock Haiku 4.5",
    "gemma_27b": "Gemma 3-27B",
    "llama_70b": "Llama 3.3-70B",
    "llama_8b": "Llama 3.1-8B",
}
PAPER_TABLE4_FRONTIER_CR = {
    "haiku": {"cr_id": 14.60, "cr_ho": 2.05, "tft_cr": 1.50},
    "gemma_27b": {"cr_id": 13.36, "cr_ho": 0.28, "tft_cr": 0.75},
    "llama_70b": {"cr_id": 12.29, "cr_ho": 1.52, "tft_cr": 1.50},
    "llama_8b": {"cr_id": 13.60, "cr_ho": -0.06, "tft_cr": 0.75},
}
PAPER_TABLE4_QWEN_BASE = {"cr_id": 14.04, "cr_ho": -1.45, "tft_cr": 1.50}
COUPLING_BY_GAME_COLUMNS = [
    "pd-c", "pd-t", "pd-h", "stag", "nego", "bos", "mp", "ttt", "auct", "dd", "p-b", "ipd",
]
TABLE5_COLUMNS = ("fc_id", "ac_id", "fc_ho")
TABLE6_FRONTIER_ROWS = ("llama_8b", "gemma_27b", "llama_70b")
TABLE6_TRAINED_ROWS = ("base", "filter_on", "filter_off", "core", "aux", "all", "rw", "merge")
TABLE6_LABELS = {
    **TABLE5_FRONTIER_LABELS,
    **config.VARIANT_LABELS,
    "filter_on": "Qwen 2.5-3B + A+β-on",
    "filter_off": "Qwen 2.5-3B + A+β-off",
}


def _per_game_fc_to_abbrev(per_game_fc: dict | None) -> dict[str, float | None]:
    src = per_game_fc or {}
    return {abbrev: src.get(game) for game, abbrev in config.GAME_ABBREV.items()}


def _build_table6_coupling_by_game(
    *,
    rows: dict[str, dict],
    ok_variants: list[str],
    frontier_rows: dict[str, dict],
    variant_sources: dict[str, str],
    frontier_sources: dict[str, str],
) -> dict:
    """Table 6: per-(model, environment) final-round coupling fc."""
    table_rows: dict[str, dict[str, float | None]] = {}
    sources: dict[str, str] = {}
    measured_cells = 0
    total_cells = 0

    frontier_present = [k for k in TABLE6_FRONTIER_ROWS if k in frontier_rows]
    trained_present = [k for k in TABLE6_TRAINED_ROWS if k in ok_variants]

    for key in frontier_present:
        metrics = frontier_rows[key]
        abbrev_row = _per_game_fc_to_abbrev(metrics.get("per_game_fc"))
        table_rows[key] = abbrev_row
        for v in abbrev_row.values():
            total_cells += 1
            if v is not None:
                measured_cells += 1
        if any(v is not None for v in abbrev_row.values()):
            sources[key] = frontier_sources.get(key, "local:frontier")
        else:
            sources[key] = f"{frontier_sources.get(key, 'local')}; fc not measured"

    for key in trained_present:
        metrics = rows[key]
        abbrev_row = _per_game_fc_to_abbrev(metrics.get("per_game_fc"))
        table_rows[key] = abbrev_row
        for v in abbrev_row.values():
            total_cells += 1
            if v is not None:
                measured_cells += 1
        if any(v is not None for v in abbrev_row.values()):
            sources[key] = variant_sources.get(key, "local:metrics")
        else:
            sources[key] = f"{variant_sources.get(key, 'local')}; fc not measured"

    return {
        "table": 6,
        "title": "Per-(model, env) final-round coupling rate (parsable rounds only)",
        "note": (
            "n=12 episodes per cell. null = coupling not measured in this eval run. "
            "Local metrics only — no paper fallback."
        ),
        "columns": COUPLING_BY_GAME_COLUMNS,
        "labels": {k: TABLE6_LABELS.get(k, k) for k in table_rows},
        "sections": {
            "frontier": frontier_present,
            "trained_3b": trained_present,
        },
        "rows": table_rows,
        "measured_cells": measured_cells,
        "total_cells": total_cells,
        "sources": sources,
    }


def _is_complete(metrics: dict) -> bool:
    if metrics.get("error"):
        return False
    cr = metrics.get("per_game_cr") or {}
    if len(cr) < len(config.ALL_GAMES):
        return False
    return all(cr.get(g) is not None for g in config.ALL_GAMES)


def _is_3b(run_dir: Path) -> bool:
    suite = run_dir / "suite.json"
    if suite.is_file():
        try:
            hp = json.loads(suite.read_text(encoding="utf-8")).get("hyperparams", {})
        except json.JSONDecodeError:
            hp = {}
        model = str(hp.get("STUDENT_MODEL") or "")
        if "3B" in model or "3b" in model:
            return True
    return False


def _score_candidate(run_dir: Path, metrics: dict, variant: str) -> tuple[int, float]:
    score = 0
    if _is_complete(metrics):
        score += 1000
    if _coupling_measured(metrics):
        score += 500
    if _is_3b(run_dir):
        score += 100
    if "20260630_180322" in run_dir.name:
        score += 50
    if run_dir.name.startswith("table5_"):
        score += 75
    mtime = (run_dir / "metrics" / f"{variant}.json").stat().st_mtime
    return score, mtime


def _merge_logs(paths: list[Path]) -> dict[str, list[ParsedEpisode]]:
    per_game: dict[str, dict[int, ParsedEpisode]] = defaultdict(dict)
    for path in paths:
        if not path.is_file():
            continue
        for game, eps in parse_eval_log(path).items():
            for p in eps:
                per_game[game][p.episode] = p
    return {g: [per_game[g][k] for k in sorted(per_game[g])] for g in per_game}


def _metrics_from_episodes(per_game_eps: dict[str, list[ParsedEpisode]]) -> dict:
    per_game: dict[str, list[float]] = defaultdict(list)
    axis_cr: dict[str, list[float]] = defaultdict(list)
    for game, eps in per_game_eps.items():
        for p in eps:
            per_game[game].append(p.cr)
            if p.opponent:
                axis = _axis_for_opponent(p.opponent)
                if axis:
                    bucket = f"{'ID' if game in config.ID_GAMES else 'HO'} {axis.title()}"
                    axis_cr[bucket].append(p.cr)
    cr_id_vals = [r for g in config.ID_GAMES for r in per_game[g]]
    cr_ho_vals = [r for g in config.HO_GAMES for r in per_game[g]]

    def mean_cr(g: str):
        return sum(per_game[g]) / len(per_game[g]) if per_game[g] else None

    return {
        "fc_id": 0.0,
        "fc_ho": 0.0,
        "cr_id": sum(cr_id_vals) / len(cr_id_vals) if cr_id_vals else 0.0,
        "cr_ho": sum(cr_ho_vals) / len(cr_ho_vals) if cr_ho_vals else 0.0,
        "per_game_cr": {g: mean_cr(g) for g in config.ALL_GAMES},
        "per_axis_cr": {k: sum(v) / len(v) for k, v in axis_cr.items()},
        "per_game_fc": {g: None for g in config.ALL_GAMES},
        "per_game_ac": {g: None for g in config.ALL_GAMES},
        "per_game_exploitability": {g: None for g in config.ALL_GAMES},
    }


def _load_table1_base_row(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    row = data.get("rows", {}).get("base")
    return row if isinstance(row, dict) else None


def _frontier_game_counts(per_game_eps: dict[str, list[ParsedEpisode]]) -> dict[str, int]:
    return {g: len(per_game_eps.get(g, [])) for g in config.ALL_GAMES}


def _is_frontier_complete(per_game_eps: dict[str, list[ParsedEpisode]]) -> bool:
    counts = _frontier_game_counts(per_game_eps)
    if not counts:
        return False
    if all(n >= config.EPISODES_PER_ENV for n in counts.values()):
        return True
    # Allow one game at EPISODES_PER_ENV-1 when all others are complete.
    short = [g for g, n in counts.items() if n < config.EPISODES_PER_ENV]
    if len(short) != 1:
        return False
    return counts[short[0]] >= config.EPISODES_PER_ENV - 1 and all(
        counts[g] >= config.EPISODES_PER_ENV for g in config.ALL_GAMES if g != short[0]
    )


def _discover_frontier_metrics() -> dict[str, tuple[str, dict]]:
    """frontier key -> (source label, metrics row)."""
    found: dict[str, tuple[str, dict]] = {}

    llama8b = _load_table1_base_row(config.RUNS_DIR / "frontier_llama8b/eval/tables/table1_base.json")
    if llama8b and all(llama8b.get("per_game_cr", {}).get(g) is not None for g in config.ALL_GAMES):
        found["llama_8b"] = ("frontier_llama8b/tables/table1_base.json", llama8b)

    gemma_logs = [
        config.RUNS_DIR / part / "eval/logs/eval_table1_base.log"
        for part in _GEMMA_RUN_PARTS
    ]
    gemma_merged = _merge_logs(gemma_logs)
    if _is_frontier_complete(gemma_merged):
        counts = _frontier_game_counts(gemma_merged)
        tag = "complete" if all(n >= config.EPISODES_PER_ENV for n in counts.values()) else "near-complete"
        rel = "+".join(p.parent.parent.name for p in gemma_logs if p.is_file())
        found["gemma_27b"] = (f"{rel}/logs (merged, {tag})", _metrics_from_episodes(gemma_merged))

    llama70_logs = [
        config.RUNS_DIR / "frontier_llama70b/eval/logs/eval_part1.log",
        config.RUNS_DIR / "frontier_llama70b/eval/logs/eval_part2.log",
        config.RUNS_DIR / "frontier_llama70b/eval/logs/eval_stag_nego_bos.log",
        config.RUNS_DIR / "frontier_llama70b/eval/logs/eval_pd_part.log",
    ]
    llama70_merged = _merge_logs(llama70_logs)
    if _is_frontier_complete(llama70_merged):
        rel = "frontier_llama70b/logs (merged)"
        found["llama_70b"] = (rel, _metrics_from_episodes(llama70_merged))

    return found


def _paper_table4_by_game() -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for model, row in config.PAPER_TABLE4.items():
        out[model] = {
            config.ABBREV_TO_GAME.get(abbrev, abbrev): val for abbrev, val in row.items()
        }
    return out


def _tft_cr_from_log_paths(
    paths: list[Path],
    *,
    variant: str | None = None,
) -> tuple[float | None, int]:
    crs: list[float] = []
    for path in paths:
        if not path.is_file():
            continue
        for _game, eps in parse_eval_log(path, variant=variant).items():
            for p in eps:
                if p.opponent == "tit_for_tat":
                    crs.append(p.cr)
    if not crs:
        return None, 0
    return sum(crs) / len(crs), len(crs)


_GEMMA_RUN_PARTS = (
    "frontier_gemma27b",
    "frontier_gemma27b_part2",
    "frontier_gemma27b_part3",
    "frontier_gemma27b_part4",
    "frontier_gemma27b_part_divide-dollar",
    "frontier_gemma27b_part_auction",
    "frontier_gemma27b_part_p-beauty",
)


def _frontier_log_paths(key: str) -> list[Path]:
    root = config.RUNS_DIR
    if key == "llama_8b":
        return [root / "frontier_llama8b/eval/logs/eval_table1_base.log"]
    if key == "gemma_27b":
        return [root / part / "eval/logs/eval_table1_base.log" for part in _GEMMA_RUN_PARTS]
    if key == "llama_70b":
        return [
            root / "frontier_llama70b/eval/logs/eval_part1.log",
            root / "frontier_llama70b/eval/logs/eval_part2.log",
            root / "frontier_llama70b/eval/logs/eval_stag_nego_bos.log",
            root / "frontier_llama70b/eval/logs/eval_pd_part.log",
            root / "frontier_llama70b/eval/logs/eval_pd_tight.log",
        ]
    return []


def _qwen_base_log_paths(run_dir: Path | None) -> list[Path]:
    """Eval logs for Qwen 3B base variant (for TFT-CR)."""
    if run_dir is None:
        return []
    paths: list[Path] = []
    for name in ("eval_suite.log", "eval_base.log", "eval_table1_base.log"):
        p = run_dir / "logs" / name
        if p.is_file():
            paths.append(p)
    for shard_log in sorted(config.RUNS_DIR.glob(f"{run_dir.parent.name}_shard_*/eval/logs/*.log")):
        if shard_log.is_file():
            paths.append(shard_log)
    return paths


def _format_table4_frontier_block(table4: dict) -> list[str]:
    labels = {
        "haiku": "Claude 3.5 Haiku",
        "gemma_27b": "Gemma 3 27B",
        "llama_70b": "Llama 3.1 70B",
        "llama_8b": "Llama 3.1 8B",
        "qwen_3b_base": "Qwen 2.5-3B base",
    }
    order = ("haiku", "gemma_27b", "llama_70b", "llama_8b", "qwen_3b_base")

    def cell(x: float | None) -> str:
        return f"{x:>+8.2f}" if x is not None else f"{'':>8}"

    lines = [
        "Model                          CR(ID)   CR(HO)   TFT-CR",
        "-" * 58,
    ]
    rows = table4.get("rows", {})
    for model in order:
        row = rows.get(model, {})
        label = labels.get(model, model)
        lines.append(
            f"{label:<30}"
            f"{cell(row.get('cr_id'))} {cell(row.get('cr_ho'))} {cell(row.get('tft_cr'))}"
        )
    src = table4.get("sources", {})
    if src:
        lines.append("")
        lines.append("Sources: " + "; ".join(f"{k}={v}" for k, v in src.items()))
    return lines


def _replace_table4_md(md: str, table4: dict) -> str:
    """Return frontier CR and TFT-CR while dropping the legacy coupling-fc block."""
    block = [
        "## Table 4 — Frontier benchmark (CR + TFT-CR)",
        "",
        "Same 12-environment evaluation. CR(ID), CR(HO), TFT-CR — **local rollouts only** (no paper reference values).",
        "",
        *_format_table4_frontier_block(table4),
        "",
        "---",
        "",
    ]
    pattern = r"## Table 4 —[^\n]*\n.*?(?=\n---\n\n## Table [5679])"
    if not re.search(pattern, md, flags=re.DOTALL):
        return md
    return re.sub(pattern, "\n".join(block), md, count=1, flags=re.DOTALL)


def _build_table4_frontier(
    *,
    table1: dict[str, dict],
    frontier_rows: dict[str, dict],
    frontier_sources: dict[str, str],
    qwen_run_dir: Path | None = None,
) -> dict:
    """Return Table 4 frontier CR(ID), CR(HO), and TFT-CR values."""
    rows: dict[str, dict] = {}
    sources: dict[str, str] = {}

    for model in PAPER_TABLE4_FRONTIER_MODELS:
        row: dict[str, float | None] = {"cr_id": None, "cr_ho": None, "tft_cr": None}
        parts: list[str] = []
        if model in table1:
            row["cr_id"] = table1[model].get("cr_id")
            row["cr_ho"] = table1[model].get("cr_ho")
            parts.append(frontier_sources.get(model, "local:table1"))
        tft, n = _tft_cr_from_log_paths(_frontier_log_paths(model))
        if tft is not None:
            row["tft_cr"] = tft
            parts.append(f"tft_cr from {n} TFT episodes")
        if parts:
            sources[model] = "; ".join(parts)
        rows[model] = row

    qwen: dict[str, float | None] = {"cr_id": None, "cr_ho": None, "tft_cr": None}
    qwen_parts: list[str] = []
    if "base" in table1:
        qwen["cr_id"] = table1["base"].get("cr_id")
        qwen["cr_ho"] = table1["base"].get("cr_ho")
        qwen_parts.append("local:table1/base")
    tft, n = _tft_cr_from_log_paths(_qwen_base_log_paths(qwen_run_dir), variant="base")
    if tft is not None:
        qwen["tft_cr"] = tft
        qwen_parts.append(f"tft_cr from {n} TFT episodes")
    if qwen_parts:
        sources["qwen_3b_base"] = "; ".join(qwen_parts)
    rows["qwen_3b_base"] = qwen

    return {
        "table": 4,
        "title": "Frontier reference rows on the same 12-env eval",
        "columns": ["cr_id", "cr_ho", "tft_cr"],
        "rows": rows,
        "sources": sources,
    }


def _coupling_measured(metrics: dict | None) -> bool:
    """True when at least one parsable coupling round was counted."""
    if not metrics:
        return False
    for g in config.ALL_GAMES:
        if metrics.get("per_game_fc", {}).get(g) is not None:
            return True
        if metrics.get("per_game_ac", {}).get(g) is not None:
            return True
    return False


def _coupling_row_from_metrics(metrics: dict | None) -> dict[str, float | None]:
    """Return only locally measured coupling values."""
    if not metrics or not _coupling_measured(metrics):
        return {c: None for c in TABLE5_COLUMNS}
    return {
        c: metrics.get(c) if metrics.get(c) is not None else None
        for c in TABLE5_COLUMNS
    }


def _build_table5_coupling(
    *,
    rows: dict[str, dict],
    frontier_rows: dict[str, dict],
    frontier_sources: dict[str, str],
    variant_sources: dict[str, str],
) -> dict:
    """Return Table 5 reasoning-action coupling from local evaluation."""
    trained: dict[str, dict] = {}
    frontier: dict[str, dict] = {}
    sources: dict[str, str] = {}
    measured: dict[str, bool] = {}

    for variant in TABLE5_TRAINED_VARIANTS:
        metrics = rows.get(variant)
        trained[variant] = _coupling_row_from_metrics(metrics)
        measured[variant] = _coupling_measured(metrics)
        sources[variant] = (
            variant_sources.get(variant, "local:metrics")
            if measured[variant]
            else f"{variant_sources.get(variant, 'missing')}; coupling not measured"
        )

    frontier_models = [m for m in TABLE5_FRONTIER_ORDER if m in frontier_rows]
    for model in frontier_models:
        metrics = frontier_rows.get(model)
        frontier[model] = _coupling_row_from_metrics(metrics)
        measured[model] = _coupling_measured(metrics)
        sources[model] = (
            frontier_sources.get(model, "local:frontier")
            if measured[model]
            else f"{frontier_sources.get(model, 'missing')}; coupling not measured"
        )

    labels = {v: config.VARIANT_LABELS.get(v, v) for v in TABLE5_TRAINED_VARIANTS}
    labels.update({m: TABLE5_FRONTIER_LABELS[m] for m in frontier_models})

    return {
        "table": 5,
        "title": "Reasoning-action coupling on the 12-env eval (paper_v01 Table 5)",
        "note": (
            "fc = final-round coupling; ac = all-round coupling. "
            "null = not measured in local eval (no paper fallback)."
        ),
        "columns": list(TABLE5_COLUMNS),
        "column_labels": {"fc_id": "fc(ID)", "ac_id": "ac(ID)", "fc_ho": "fc(HO)"},
        "sections": {
            "trained_3b": list(TABLE5_TRAINED_VARIANTS),
            "frontier": frontier_models,
        },
        "labels": labels,
        "rows": {**trained, **frontier},
        "measured": measured,
        "sources": sources,
    }


def _format_table5_coupling_block(table5: dict) -> list[str]:
    col_labels = table5.get("column_labels", {})
    cols = table5.get("columns", list(TABLE5_COLUMNS))

    def cell(x: float | None) -> str:
        return f"{x:>8.2f}" if x is not None else f"{'':>8}"

    header = f"{'Model':<32}" + "".join(f"{col_labels.get(c, c):>10}" for c in cols)
    lines = [header, "-" * len(header)]
    rows = table5.get("rows", {})
    labels = table5.get("labels", {})

    for section, keys in (
        ("trained_3b", table5.get("sections", {}).get("trained_3b", TABLE5_TRAINED_VARIANTS)),
        ("frontier", table5.get("sections", {}).get("frontier", [])),
    ):
        if not keys:
            continue
        for key in keys:
            row = rows.get(key, {})
            label = labels.get(key, key)
            lines.append(
                f"{label:<32}"
                + "".join(cell(row.get(c)) for c in cols)
            )
    src = table5.get("sources", {})
    if src:
        lines.append("")
        lines.append("Sources: " + "; ".join(f"{k}={v}" for k, v in src.items()))
    return lines


def _inject_table5_coupling_md(md: str, table5: dict) -> str:
    marker = "## Table 5 — Training manifest"
    if marker not in md:
        return md
    block = [
        "## Table 5 — Reasoning-action coupling (paper_v01)",
        "",
        "fc(ID), ac(ID), fc(HO) — **local evaluation only** (no paper reference values).",
        "",
        *_format_table5_coupling_block(table5),
        "",
        "---",
        "",
    ]
    return md.replace(marker, "\n".join(block) + marker, 1)


def _build_coupling_by_game(
    *,
    rows: dict[str, dict],
    ok_variants: list[str],
    frontier_rows: dict[str, dict],
    variant_sources: dict[str, str],
    frontier_sources: dict[str, str],
) -> dict:
    """Legacy alias — same payload as table6.json."""
    return _build_table6_coupling_by_game(
        rows=rows,
        ok_variants=ok_variants,
        frontier_rows=frontier_rows,
        variant_sources=variant_sources,
        frontier_sources=frontier_sources,
    )


PAPER_TABLE7_TRAINING_ORDER = (
    "filter_on",
    "filter_off",
    "core",
    "aux",
    "all",
    "rw",
    "merge",
)
PAPER_TABLE7_TRAINING_LABELS = {
    "filter_on": "B filter-on",
    "filter_off": "B filter-off",
    "core": "A+β-CORE",
    "aux": "A+β-AUX",
    "all": "A+β-ALL",
    "rw": "A+β-RW",
    "merge": "A+β-MERGE",
}


def _pair_count_from_data(path: str | None) -> int | None:
    if not path:
        return None
    p = ROOT / path
    if not p.is_file():
        return None
    return sum(1 for _ in p.open(encoding="utf-8"))


def _discover_local_training_runs() -> dict[str, dict]:
    """Latest completed publish per variant from runs/*/lora/*/run_info.json."""
    runs_root = config.RUNS_DIR
    best: dict[str, dict] = {}
    if not runs_root.is_dir():
        return best
    for run_info in runs_root.glob("*/lora/*/run_info.json"):
        try:
            data = json.loads(run_info.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("status") != "completed":
            continue
        resolved = data.get("resolved_config", {})
        variant = resolved.get("variant") or run_info.parent.name
        train_metrics = data.get("train_metrics") or {}
        finished_at = data.get("finished_at") or ""
        session_id = run_info.parent.parent.parent.name
        entry = {
            "run_id": session_id,
            "variant": variant,
            "pairs": resolved.get("pairs_count"),
            "global_step": train_metrics.get("global_step"),
            "train_loss": train_metrics.get("train_loss"),
            "publish_dir": str(run_info.parent),
            "finished_at": finished_at,
        }
        prev = best.get(variant)
        if prev is None or finished_at > prev.get("finished_at", ""):
            best[variant] = entry
    return best


def _build_table7_training(
    *,
    manifest: dict,
    hyperparams: dict,
    local_runs: dict[str, dict] | None = None,
) -> dict:
    """Table 7: DPO training manifest from local checkpoints and lora run logs."""
    local_runs = local_runs if local_runs is not None else _discover_local_training_runs()
    rows: dict[str, dict] = {}
    sources: dict[str, str] = {}
    for variant in PAPER_TABLE7_TRAINING_ORDER:
        spec = manifest.get(variant, {})
        local = local_runs.get(variant, {})
        pairs = local.get("pairs")
        if pairs is None:
            pairs = _pair_count_from_data(spec.get("data"))
        if pairs is None:
            pairs = spec.get("pairs")
        row = {
            "label": PAPER_TABLE7_TRAINING_LABELS.get(variant, variant),
            "pairs": pairs if pairs else None,
            "best_step": local.get("global_step"),
            "final_train_loss": local.get("train_loss"),
            "checkpoint_exists": spec.get("exists"),
        }
        rows[variant] = row
        if local.get("run_id"):
            sources[variant] = f"runs/{local['run_id']}/lora/{variant}"
        elif spec.get("exists"):
            sources[variant] = f"local:{spec.get('checkpoint', variant)}"
        elif variant == "merge":
            sources[variant] = "local:LoRA average of AUX + ALL (not trained)"
        else:
            sources[variant] = "missing checkpoint"
    return {
        "table": 7,
        "title": "Training hyperparameters (local DPO runs)",
        "note": (
            "Shared recipe: LoRA r/α from hyperparams, sigmoid DPO β=0.1, "
            "environment-token loss masking. MERGE is adapter average, not a trained run. "
            "Ckpt step and train loss come from local runs/*/lora/*/run_info.json "
            "(no eval-loss tracking in current trainer)."
        ),
        "hyperparams": {
            "STUDENT_MODEL": hyperparams.get("STUDENT_MODEL"),
            "train_epochs": hyperparams.get("train_epochs", hyperparams.get("epochs")),
            "lora_r": hyperparams.get("lora_r"),
            "lora_alpha": hyperparams.get("lora_alpha"),
            "dpo_beta": config.DPO_BETA,
        },
        "columns": ["label", "pairs", "best_step", "final_train_loss"],
        "rows": rows,
        "sources": sources,
    }


def _companion_row_from_metrics(metrics: dict | None) -> dict[str, dict[str, float | None]]:
    """Per-env ac, fc, exploitability — local metrics only."""
    empty = {g: None for g in config.ALL_GAMES}
    if not metrics:
        return {"ac": dict(empty), "fc": dict(empty), "exploitability": dict(empty)}
    return {
        "ac": {g: metrics.get("per_game_ac", {}).get(g) for g in config.ALL_GAMES},
        "fc": {g: metrics.get("per_game_fc", {}).get(g) for g in config.ALL_GAMES},
        "exploitability": {
            g: metrics.get("per_game_exploitability", {}).get(g) for g in config.ALL_GAMES
        },
    }


def _build_table9_companion(
    *,
    rows: dict[str, dict],
    ok_variants: list[str],
    frontier_rows: dict[str, dict],
) -> dict:
    """Return Table 9 per-environment companion metrics from local evaluation."""
    table_rows: dict[str, dict] = {}
    sources: dict[str, str] = {}
    for variant in ok_variants:
        table_rows[variant] = _companion_row_from_metrics(rows.get(variant))
        sources[variant] = "local:metrics"
    for model, metrics in frontier_rows.items():
        table_rows[model] = _companion_row_from_metrics(metrics)
        sources[model] = "local:frontier metrics"
    return {
        "table": 9,
        "title": "Per-env companion metrics (local eval)",
        "note": "ac = all-round coupling; fc = final-round coupling; exploitability via maxmin LP (2×2 games).",
        "columns": ["ac", "fc", "exploitability"],
        "rows": table_rows,
        "sources": sources,
    }


def _format_table7_training_block(table7: dict) -> list[str]:
    lines = [
        f"{'Variant':<16} {'Pairs':>8} {'Ckpt step':>10} {'Train loss':>12}",
        "-" * 50,
    ]
    for variant in PAPER_TABLE7_TRAINING_ORDER:
        row = table7.get("rows", {}).get(variant, {})
        label = row.get("label", variant)
        pairs = row.get("pairs")
        step = row.get("best_step")
        loss = row.get("final_train_loss", row.get("best_eval_loss"))
        pairs_s = str(pairs) if pairs is not None else "—"
        step_s = str(step) if step is not None else "—"
        loss_s = f"{loss:.4f}" if isinstance(loss, (int, float)) else "—"
        lines.append(f"{label:<16} {pairs_s:>8} {step_s:>10} {loss_s:>12}")
    hp = table7.get("hyperparams", {})
    lines.append("")
    lines.append(
        f"Model: {hp.get('STUDENT_MODEL', '?')} · train_epochs={hp.get('train_epochs', '?')} · "
        f"LoRA r={hp.get('lora_r', '?')} α={hp.get('lora_alpha', '?')} · DPO β={hp.get('dpo_beta', '?')}"
    )
    return lines


def _format_table9_companion_block(table9: dict, variant: str) -> list[str]:
    labels = {**config.VARIANT_LABELS, "gemma_27b": "Gemma 3 27B", "llama_8b": "Llama 3.1 8B"}
    title = labels.get(variant, variant)
    row = table9.get("rows", {}).get(variant, {})
    lines = [
        f"### {title} (`{variant}`)",
        "",
        f"{'env':<22} {'ac':>8} {'fc':>8} {'exploit':>8}",
        "-" * 50,
    ]

    def cell(x: float | None) -> str:
        if x is None:
            return f"{'':>8}"
        return f"{x:>8.2f}"

    for game in config.ALL_GAMES:
        lines.append(
            f"{game:<22}"
            f"{cell(row.get('ac', {}).get(game))}"
            f"{cell(row.get('fc', {}).get(game))}"
            f"{cell(row.get('exploitability', {}).get(game))}"
        )
    return lines


def _inject_table7_training_md(md: str, table7: dict) -> str:
    marker = "## Table 5 — Training manifest"
    if marker not in md:
        return md
    block = [
        "## Table 7 — Training hyperparameters (local)",
        "",
        "Pair count, published checkpoint step, and final training loss by DPO variant (from local lora/runs).",
        "",
        *_format_table7_training_block(table7),
        "",
        "---",
        "",
    ]
    return md.replace(marker, "\n".join(block) + marker, 1)


def _strip_table4_fc_sections_md(md: str) -> str:
    """Drop legacy Table 4 coupling-fc blocks."""
    md = re.sub(
        r"\n---\n\n## Table 4 — Final-round coupling[^\n]*\n.*?(?=\n---\n\n## )",
        "\n",
        md,
        flags=re.DOTALL,
    )
    return re.sub(
        r"(## Table 4 — Final-round coupling fc\n\n.*?)(### \[Paper\].*?)(?=\n---\n\n## )",
        r"\1",
        md,
        count=1,
        flags=re.DOTALL,
    )


def _inject_table9_companion_md(md: str, table9: dict, variants: list[str]) -> str:
    marker = "## Table 6 — Stag-hunt anti-coordination"
    if marker not in md:
        return md
    show = [v for v in variants if v in table9.get("rows", {})]
    block = [
        "## Table 9 — Companion metrics (local)",
        "",
        "Per-environment action consistency (ac), final-round fc, and exploitability. Local evaluation only (no paper reference values).",
        "",
    ]
    for v in show:
        block.extend(_format_table9_companion_block(table9, v))
        block.append("")
    block.append("---")
    block.append("")
    return md.replace(marker, "\n".join(block) + marker, 1)


def _write_assembled_table_jsons(run_dir: Path, suite: dict) -> None:
    tables = run_dir / "tables"
    save_json(tables / "table1.json", suite.get("table1", {}))
    save_json(tables / "table2.json", suite.get("table2", {}))
    save_json(tables / "table3.json", suite.get("table3", {}))
    save_json(tables / "table4.json", suite.get("table4_frontier", {}))
    save_json(tables / "table5.json", suite.get("table5_coupling", {}))
    save_json(tables / "coupling_by_game.json", suite.get("coupling_by_game", {}))
    save_json(tables / "training_manifest.json", suite.get("table5_manifest", {}))
    save_json(tables / "table6.json", suite.get("table6", {}))
    save_json(tables / "table6_paper_format.json", suite.get("table6", {}))
    save_json(tables / "table8_stag_hunt.json", suite.get("table8_stag_hunt", {}))
    save_json(tables / "table7.json", suite.get("table7_training", {}))
    save_json(tables / "table9.json", suite.get("table9_companion", {}))
    save_json(tables / "companion_metrics.json", suite.get("table9_companion", {}))
    if suite.get("table4_fc"):
        save_json(tables / "table4_fc.json", suite["table4_fc"])


def _discover_metrics(runs_root: Path) -> dict[str, tuple[Path, dict, Path]]:
    """variant -> (eval_run_dir, metrics dict, metrics file path)."""
    best: dict[str, tuple[int, float, Path, dict, Path]] = {}
    if not runs_root.is_dir():
        return {}

    eval_dirs: list[Path] = []
    for session in runs_root.iterdir():
        if not session.is_dir():
            continue
        eval_dir = session / "eval"
        if eval_dir.is_dir():
            eval_dirs.append(eval_dir)
        elif (session / "metrics").is_dir():
            eval_dirs.append(session)

    for run_dir in eval_dirs:
        metrics_dir = run_dir / "metrics"
        if not metrics_dir.is_dir():
            continue
        for mf in metrics_dir.glob("*.json"):
            variant = mf.stem
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if data.get("error"):
                continue
            sc, mtime = _score_candidate(run_dir, data, variant)
            prev = best.get(variant)
            if prev is None or (sc, mtime) > (prev[0], prev[1]):
                best[variant] = (sc, mtime, run_dir, data, mf)

    return {
        v: (run_dir, metrics, mf)
        for v, (_, _, run_dir, metrics, mf) in best.items()
    }


def _build_suite(
    *,
    run_id: str,
    rows: dict[str, dict],
    ok_variants: list[str],
    sources: dict[str, str],
    qwen_run_dir: Path | None = None,
    frontier_rows: dict[str, dict],
    frontier_sources: dict[str, str],
    checkpoint_dir: Path,
    paper: bool,
) -> dict:
    ns = argparse.Namespace(
        STUDENT_MODEL=config.PAPER_STUDENT_MODEL if paper else config.STUDENT_MODEL,
        lora_r=config.PAPER_LORA_R if paper else config.LORA_R,
        lora_alpha=config.PAPER_LORA_ALPHA if paper else config.LORA_ALPHA,
        episodes=config.EPISODES_PER_ENV,
        checkpoint_dir=str(checkpoint_dir),
        variants=ok_variants,
        paper=paper,
    )
    manifest = _training_manifest(checkpoint_dir)
    all_keys = ok_variants + [k for k in FRONTIER_KEYS if k in frontier_rows]

    table1 = {
        v: {k: rows[v][k] for k in ("fc_id", "fc_ho", "cr_id", "cr_ho") if k in rows[v]}
        for v in ok_variants
    }
    table2 = {v: rows[v].get("per_game_cr", {}) for v in ok_variants}
    table3 = {v: rows[v].get("per_axis_cr", {}) for v in ok_variants}
    table4_fc = {v: rows[v].get("per_game_fc", {}) for v in ok_variants}

    for key, m in frontier_rows.items():
        table1[key] = {k: m[k] for k in ("fc_id", "fc_ho", "cr_id", "cr_ho") if k in m}
        table2[key] = m.get("per_game_cr", {})
        table3[key] = m.get("per_axis_cr", {})
        table4_fc[key] = {g: m.get("per_game_fc", {}).get(g) for g in config.ALL_GAMES}

    table7_training = _build_table7_training(
        manifest=manifest,
        hyperparams=_hyperparams(ns),
        local_runs=_discover_local_training_runs(),
    )
    table9_companion = _build_table9_companion(
        rows=rows,
        ok_variants=ok_variants,
        frontier_rows=frontier_rows,
    )

    table4_frontier = _build_table4_frontier(
        table1=table1,
        frontier_rows=frontier_rows,
        frontier_sources=frontier_sources,
        qwen_run_dir=qwen_run_dir,
    )
    table5_coupling = _build_table5_coupling(
        rows=rows,
        frontier_rows=frontier_rows,
        frontier_sources=frontier_sources,
        variant_sources=sources,
    )
    table6 = _build_table6_coupling_by_game(
        rows=rows,
        ok_variants=ok_variants,
        frontier_rows=frontier_rows,
        variant_sources=sources,
        frontier_sources=frontier_sources,
    )
    coupling_by_game = table6

    merged_rows = dict(rows)
    merged_rows.update(frontier_rows)

    return {
        "run_id": run_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": 0.0,
        "timings_seconds": {},
        "hyperparams": _hyperparams(ns),
        "sources": sources,
        "frontier_sources": frontier_sources,
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "table4_frontier": table4_frontier,
        "table4_fc": table4_fc,
        "table5_coupling": table5_coupling,
        "coupling_by_game": coupling_by_game,
        "table5_manifest": manifest,
        "table6": table6,
        "table8_stag_hunt": {"paper_reference": config.PAPER_TABLE6},
        "table7_training": table7_training,
        "table9_companion": table9_companion,
        "table7": table9_companion.get("rows", {}),
        "variants": all_keys,
        "rows": merged_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble tables from best metrics on disk")
    parser.add_argument("--out", default=None, help="Session folder name under runs/ (default: assembled_<ts>)")
    parser.add_argument("--variants", default=",".join(PAPER_VARIANTS))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--staging-dir", type=Path, default=None)
    parser.add_argument("--no-latest", action="store_true", help="Do not update runs/latest.json")
    args = parser.parse_args(argv)

    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    discovered = _discover_metrics(config.RUNS_DIR)
    frontier = _discover_frontier_metrics()

    latest_lora = None
    latest_path = config.RUNS_DIR / "latest.json"
    if latest_path.is_file():
        try:
            latest_lora = config.RUNS_DIR / json.loads(latest_path.read_text())["run_id"] / "lora"
        except (json.JSONDecodeError, KeyError):
            latest_lora = None
    if args.checkpoint_dir is None:
        args.checkpoint_dir = latest_lora if latest_lora and latest_lora.is_dir() else config.RUNS_DIR
    if args.staging_dir is None and latest_lora is not None:
        args.staging_dir = latest_lora.parent / "eval" / "staging"

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir.is_dir() and (ckpt_dir / "lora").is_dir():
        ckpt_dir = ckpt_dir / "lora"
    args.checkpoint_dir = ckpt_dir
    rows: dict[str, dict] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []

    for v in want:
        hit = discovered.get(v)
        if not hit:
            missing.append(v)
            continue
        run_dir, metrics, mf = hit
        rows[v] = metrics
        tag = "complete" if _is_complete(metrics) else "partial"
        sources[v] = f"{run_dir.name}/metrics/{mf.name} ({tag})"

    ok_variants = [v for v in want if v in rows]
    if not ok_variants:
        print("ERROR: no metrics found for requested variants", file=sys.stderr)
        return 1

    frontier_rows = {k: m for k, (_, m) in frontier.items()}
    frontier_sources = {k: src for k, (src, _) in frontier.items()}

    run_id = args.out or f"assembled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = new_run_dir(run_id)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    for v in ok_variants:
        save_json(run_dir / "metrics" / f"{v}.json", rows[v])

    if args.checkpoint_dir.is_dir():
        prepare_checkpoint_dir(
            args.staging_dir,
            lora_dir=args.checkpoint_dir,
            variants=[v for v in ok_variants if v not in ("base", "merge")],
            link=True,
        )
        _maybe_merge(args.staging_dir, alpha=config.MERGE_ALPHA)
        manifest_dir = args.staging_dir
    else:
        manifest_dir = args.checkpoint_dir

    qwen_run_dir = discovered.get("base", (None, None, None))[0] if "base" in discovered else None

    suite = _build_suite(
        run_id=run_id,
        rows=rows,
        ok_variants=ok_variants,
        sources=sources,
        frontier_rows=frontier_rows,
        frontier_sources=frontier_sources,
        checkpoint_dir=manifest_dir,
        paper=True,
        qwen_run_dir=qwen_run_dir,
    )
    save_json(run_dir / "suite.json", suite)
    save_json(run_dir / "paper_refs.json", _paper_refs())
    _write_assembled_table_jsons(run_dir, suite)

    md = build_result_md(
        suite=suite,
        rows=suite["rows"],
        ok_variants=ok_variants,
        manifest=suite["table5_manifest"],
        run_dir=run_dir,
        run_id=run_id,
    )
    lines = md.splitlines()
    lines.insert(2, "")
    lines.insert(3, "**Assembled** from best on-disk metrics (see `suite.json` → `sources`).")
    if frontier_sources:
        lines.insert(4, f"Frontier models: {', '.join(f'{k}←{v}' for k, v in frontier_sources.items())}.")
        insert_at = 5
    else:
        insert_at = 4
    if missing:
        lines.insert(insert_at, f"Missing variants (no metrics): {', '.join(missing)}.")
    md_out = _replace_table4_md("\n".join(lines), suite.get("table4_frontier", {}))
    md_out = _strip_table4_fc_sections_md(md_out)
    md_out = _inject_table5_coupling_md(md_out, suite.get("table5_coupling", {}))
    md_out = _inject_table7_training_md(md_out, suite.get("table7_training", {}))
    md_out = _inject_table9_companion_md(md_out, suite.get("table9_companion", {}), ok_variants)
    # Drop legacy per-variant companion sections from build_result_md (now Table 9).
    md_out = re.sub(
        r"\n---\n\n## Table 7 — Companion metrics \(`[^`]+`\).*?(?=\n---\n\n## |\Z)",
        "\n",
        md_out,
        flags=re.DOTALL,
    )
    (run_dir / "result.md").write_text(md_out, encoding="utf-8")
    latex_path = write_latex_md(run_dir, suite, ok_variants=ok_variants)
    if not args.no_latest:
        write_latest_pointer(run_dir)

    print(f"Assembled {len(ok_variants)} 3B variants + {len(frontier_rows)} frontier -> {run_dir}")
    for v in ok_variants:
        print(f"  {v}: {sources[v]}")
    for k, src in frontier_sources.items():
        m = frontier_rows[k]
        print(f"  {k}: {src} | CR_ID={m['cr_id']:.2f} CR_HO={m['cr_ho']:.2f}")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    print(f"Saved {run_dir / 'result.md'}")
    print(f"Saved {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
