#!/usr/bin/env python3
"""Aggregate persistent rollout JSONL files into per-variant metric JSON."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from eval.metric.metrics import aggregate_episode_results, metrics_to_dict
from eval.rollout.games import EpisodeResult, RoundResult
from history.paths import active_session_id


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


def _load_rollout(path: Path) -> list[EpisodeResult]:
    episodes: list[EpisodeResult] = []
    seen: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row: dict[str, Any] = json.loads(line)
                if row.get("status", "completed") != "completed":
                    continue
                key = (str(row["game"]), int(row["episode"]))
                if key in seen:
                    continue
                seen.add(key)
                episodes.append(
                    EpisodeResult(
                        game=key[0],
                        opponent=str(row["opponent"]),
                        cumulative_reward=float(row["cumulative_reward"]),
                        rounds=[RoundResult(**round_row) for round_row in row.get("rounds", [])],
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid rollout record {path}:{line_no}: {exc}") from exc
    return episodes


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rollouts/*.jsonl -> metrics/*.json")
    parser.add_argument("--run-id", default=None, help="Session under runs/; default: RUN_ID or latest")
    parser.add_argument("--variants", default="", help="Comma-separated; default: every rollout JSONL")
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
    eval_dir = config.RUNS_DIR / run_id / "eval"
    rollout_dir = eval_dir / "rollouts"
    metrics_dir = eval_dir / "metrics"
    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    paths = [rollout_dir / f"{variant}.jsonl" for variant in requested]
    if not requested:
        paths = sorted(rollout_dir.glob("*.jsonl"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error("missing rollout file(s): " + ", ".join(str(path) for path in missing))
    if not paths:
        parser.error(f"no rollout JSONL files found under {rollout_dir}")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(rollout_dir),
        "variants": {},
    }
    for path in paths:
        episodes = _load_rollout(path)
        output = metrics_dir / f"{path.stem}.json"
        _write_json(output, metrics_to_dict(aggregate_episode_results(episodes)))
        manifest["variants"][path.stem] = {"episodes": len(episodes), "path": str(output)}
        print(f"{path.stem}: {len(episodes)} episodes -> {output}")
    _write_json(metrics_dir / "_config.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
