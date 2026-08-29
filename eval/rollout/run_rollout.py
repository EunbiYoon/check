#!/usr/bin/env python3
"""Run model/game rollouts and persist lossless episode records as JSONL.

This stage deliberately does not aggregate metrics.  The JSONL records contain
everything needed to compute CR, opponent-axis CR, reasoning/action coupling,
and final-action exploitability without loading a model again.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train.dpo_lora.gpu_env  # noqa: F401

import torch

import config
from eval.rollout.checkpoints import list_available_variants, prepare_checkpoint_dir
from eval.rollout.progress import EvalLogger
from eval.table.run_paper_tables import _maybe_merge
from eval.table.run_table import build_agent
from eval.rollout.games import EpisodeResult, run_game_batch
from history.paths import active_session_id


SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _load_completed(path: Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                # Compact rows contain completed episodes only.  Accept the
                # legacy explicit status field for existing rollout files.
                if row.get("status", "completed") == "completed":
                    completed.add((str(row["game"]), int(row["episode"])))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid rollout JSONL {path}:{line_no}: {exc}") from exc
    return completed


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=False, default=_json_default)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _episode_row(
    *,
    episode: int,
    result: EpisodeResult,
) -> dict[str, Any]:
    """Return only per-episode data; shared metadata lives in _config.json."""
    return {
        "game": result.game,
        "episode": episode,
        "opponent": result.opponent,
        "cumulative_reward": result.cumulative_reward,
        "rounds": [asdict(round_result) for round_result in result.rounds],
    }


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sharded rollout workers write concurrently. Give each atomic replacement
    # its own temporary file so workers cannot rename one another's temp file.
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persist eval rollouts without computing metrics")
    parser.add_argument("--run-id", default=None, help="Training session under runs/; default: RUN_ID or runs/latest.json")
    parser.add_argument("--variants", default="", help="Comma-separated; default: all available + base")
    parser.add_argument("--games", default="all", help="Comma-separated; default: all 12 games")
    parser.add_argument("--episodes", type=int, default=config.EPISODES_PER_ENV)
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED)
    parser.add_argument("--max-tokens", type=int, default=config.EVAL_MAX_TOKENS)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip completed JSONL episode records")
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")

    run_id = args.run_id or active_session_id()
    if run_id is None:
        latest = config.RUNS_DIR / "latest.json"
        if latest.is_file():
            try:
                run_id = str(json.loads(latest.read_text(encoding="utf-8"))["run_id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
    if run_id is None:
        parser.error("--run-id is required when RUN_ID and runs/latest.json are unavailable")
    args.run_id = run_id
    session_dir = config.RUNS_DIR / run_id
    lora_dir = session_dir / "lora"
    if not lora_dir.is_dir():
        parser.error(f"LoRA session not found: {lora_dir}")

    rollout_dir = session_dir / "eval" / "rollouts"
    # Each rollout worker gets its own staging directory.  Multiple game shards
    # commonly start at the same time, and sharing this directory lets one
    # worker unlink an adapter while another is creating or loading it.
    staging_dir = rollout_dir / "staging" / f"worker-{os.getpid()}"
    available = list_available_variants(lora_dir=lora_dir)
    prepare_checkpoint_dir(staging_dir, lora_dir=lora_dir, variants=available)
    if not args.no_merge:
        _maybe_merge(staging_dir, alpha=config.MERGE_ALPHA)
    available = list_available_variants(lora_dir=staging_dir)

    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not requested:
        requested = ["base", *available]
    missing = [v for v in requested if v != "base" and v not in available]
    if missing:
        parser.error(
            "requested adapter(s) are incomplete or missing: "
            + ", ".join(missing)
            + f"; completed adapters: {', '.join(available) or 'none'}"
        )

    if args.games.lower() == "all":
        games = list(config.ALL_GAMES)
    else:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
        invalid_games = [g for g in games if g not in config.ALL_GAMES]
        if invalid_games:
            parser.error("unknown game(s): " + ", ".join(invalid_games))

    STUDENT_MODEL = args.STUDENT_MODEL or config.PAPER_STUDENT_MODEL
    manifest_path = rollout_dir / "_config.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "updated_at": _utc_now(),
        "STUDENT_MODEL": STUDENT_MODEL,
        "variants": requested,
        "games": games,
        "episodes_per_game": args.episodes,
        "seed": args.seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "max_tokens": args.max_tokens,
        "merge_alpha": config.MERGE_ALPHA,
        "files": {},
    }
    _write_manifest(manifest_path, manifest)

    # Reuse the established model construction path without metric aggregation.
    eval_args = argparse.Namespace(
        mode="lora",
        STUDENT_MODEL=STUDENT_MODEL,
        checkpoint_dir=staging_dir,
        no_4bit=not config.USE_4BIT,
        lora_r=config.PAPER_LORA_R,
        lora_alpha=config.PAPER_LORA_ALPHA,
        lora_target=config.PAPER_LORA_TARGET_MODULES,
        max_tokens=args.max_tokens,
    )

    for variant in requested:
        output = rollout_dir / f"{variant}.jsonl"
        if output.exists() and not args.resume:
            parser.error(f"rollout already exists: {output}; pass --resume to continue")
        completed = _load_completed(output) if args.resume else set()
        logger = EvalLogger(
            log_path=rollout_dir / "logs" / f"{variant}.log",
            append=args.resume,
            label=f"variant={variant}",
        )
        agent, tag = build_agent(eval_args, variant)
        logger.variant_start(variant, tag, "rollout", args.episodes)
        for game_idx, game in enumerate(config.ALL_GAMES, start=1):
            if game not in games:
                continue
            skip = {episode for saved_game, episode in completed if saved_game == game}
            logger.game_start(game_idx, len(config.ALL_GAMES), game, args.episodes)

            def save_episode(ep_num: int, result: EpisodeResult) -> None:
                _append_jsonl(
                    output,
                    _episode_row(
                        episode=ep_num,
                        result=result,
                    ),
                )

            run_game_batch(
                game,
                agent,
                args.episodes,
                args.seed,
                logger=logger,
                skip_episodes=skip,
                on_episode=save_episode,
            )

        completed_now = _load_completed(output)
        manifest["files"][variant] = {
            "path": str(output),
            "completed_episodes": len(completed_now),
            "expected_episodes": len(games) * args.episodes,
        }
        manifest["updated_at"] = _utc_now()
        _write_manifest(manifest_path, manifest)
        del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["status"] = "completed"
    manifest["updated_at"] = _utc_now()
    _write_manifest(manifest_path, manifest)
    print(f"Saved rollout records under {rollout_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
