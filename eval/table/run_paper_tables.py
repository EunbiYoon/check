#!/usr/bin/env python3
"""One-shot entry: runs/<session>/lora adapters -> Tables 1-7 under runs/<session>/eval/.

Supports lightweight heuristic evaluation and full trained-variant evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train.dpo_lora.gpu_env  # noqa: F401

import config
from eval.rollout.checkpoints import list_available_variants, prepare_checkpoint_dir
from eval.table.run_eval_suite import main as run_eval_suite
from history.paths import active_session_id, ensure_session, eval_run_dir, lora_root


def _load_merge_adapters():
    """Load train/specialist-merge/merge_adapters.py (hyphenated dir, no plain import)."""
    path = ROOT / "train" / "specialist-merge" / "merge_adapters.py"
    spec = importlib.util.spec_from_file_location("specialist_merge_adapters", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load merge module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


merge_lora_adapters = _load_merge_adapters().merge_lora_adapters


def _default_variants(available: list[str]) -> list[str]:
    if config.PAPER_EVAL_VARIANTS:
        return [v for v in config.PAPER_EVAL_VARIANTS if v == "base" or v in available]
    paper_order = ["base", "core", "aux", "all", "rw", "merge", "filter_on", "filter_off"]
    out = ["base"]
    for v in paper_order[1:]:
        if v in available:
            out.append(v)
    if len(out) == 1:
        out.extend(v for v in available if v != "base")
    return out


def _maybe_merge(staging: Path, *, alpha: float) -> None:
    aux = staging / "aux"
    all_ckpt = staging / "all"
    merge_out = staging / "merge"
    if not aux.exists() or not all_ckpt.exists():
        return
    if (merge_out / "adapter_config.json").exists():
        return
    merge_lora_adapters(aux, all_ckpt, merge_out, alpha=alpha)
    for name in ("adapter_config.json",):
        src = all_ckpt / name
        if src.exists():
            shutil.copy2(src, merge_out / name)
    print(f"MERGE: alpha={alpha} -> {merge_out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Tables 1-7 from runs/<session>/lora")
    parser.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    parser.add_argument("--lora-dir", type=Path, default=None)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Symlink tree passed to eval as --checkpoint-dir",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,  # deprecated alias for --lora-dir (legacy scripts)
    )
    parser.add_argument("--variants", default=config.EVAL_VARIANTS, help="Comma-separated")
    parser.add_argument("--episodes", type=int, default=config.EPISODES_PER_ENV)
    parser.add_argument("--seed", type=int, default=config.EVAL_SEED)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-target", default=None)
    parser.add_argument("--max-tokens", type=int, default=config.EVAL_MAX_TOKENS)
    parser.add_argument(
        "--games",
        default=None if config.EVAL_GAMES.lower() == "all" else config.EVAL_GAMES,
        help="Comma-separated games; default: all",
    )
    parser.add_argument("--merge-alpha", type=float, default=config.MERGE_ALPHA)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--mode", choices=["lora", "heuristic"], default="lora")
    parser.add_argument("--list", action="store_true", help="List discovered variants and exit")
    parser.add_argument("--copy", action="store_true", help="Copy adapters instead of symlinks")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Existing runs/<run-id>/eval folder (with --resume)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing eval run (requires --run-id)",
    )
    args = parser.parse_args(argv)

    session_id = args.run_id or active_session_id()
    if session_id is None:
        session_id = ensure_session()

    if args.lora_dir is None:
        args.lora_dir = lora_root(session_id) if session_id else config.RUNS_DIR
    if args.staging_dir is None:
        args.staging_dir = eval_run_dir(session_id) / "rollouts" / "staging"

    if args.checkpoint_dir is not None:
        args.lora_dir = args.checkpoint_dir
        if args.staging_dir.name == "staging" and args.staging_dir.parent.name == "eval":
            pass
        else:
            args.staging_dir = args.checkpoint_dir / "eval_staging"

    available = list_available_variants(runs_dir=args.runs_dir, lora_dir=args.lora_dir)
    if args.list:
        print("Available LoRA variants:")
        for v in available:
            print(f"  - {v}")
        return 0

    variants = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants
        else _default_variants(available)
    )

    staged = prepare_checkpoint_dir(
        args.staging_dir,
        runs_dir=args.runs_dir,
        lora_dir=args.lora_dir,
        variants=[v for v in variants if v not in ("base", "merge")],
        link=not args.copy,
    )
    print(f"Staged {len(staged)} adapter(s) under {args.staging_dir}")
    for v, p in staged.items():
        print(f"  {v} -> {p}")

    if not args.no_merge:
        _maybe_merge(args.staging_dir, alpha=args.merge_alpha)

    suite_argv = [
        "--checkpoint-dir",
        str(args.staging_dir),
        "--variants",
        ",".join(variants),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(args.max_tokens),
    ]
    if args.STUDENT_MODEL:
        suite_argv.extend(["--model-id", args.STUDENT_MODEL])
    if args.lora_r is not None:
        suite_argv.extend(["--lora-r", str(args.lora_r)])
    if args.lora_alpha is not None:
        suite_argv.extend(["--lora-alpha", str(args.lora_alpha)])
    if args.lora_target:
        suite_argv.extend(["--lora-target", args.lora_target])
    if args.games:
        suite_argv.extend(["--games", args.games])
    if args.run_id:
        suite_argv.extend(["--run-id", args.run_id])
    elif session_id:
        suite_argv.extend(["--run-id", session_id])
    if args.resume:
        suite_argv.append("--resume")

    if args.mode == "heuristic":
        # Heuristic path: run table 1 only per variant (fast, no GPU)
        from argparse import Namespace

        from eval.table.run_table import eval_variant, save_json
        from eval.metric.metrics import metrics_to_dict
        from eval.table.paths import new_run_dir, utc_stamp, write_latest_pointer
        from eval.table.run_eval_suite import (
            _hyperparams,
            _paper_refs,
            _training_manifest,
            _write_table_jsons,
            build_result_md,
        )

        run_id = utc_stamp()
        run_dir = new_run_dir(run_id)
        eval_args = Namespace(
            mode="heuristic",
            STUDENT_MODEL=args.STUDENT_MODEL or config.STUDENT_MODEL,
            checkpoint_dir=args.staging_dir,
            episodes=args.episodes,
            seed=args.seed,
            lora_r=args.lora_r or config.LORA_R,
            lora_alpha=args.lora_alpha or config.LORA_ALPHA,
            lora_target=args.lora_target or "q_proj,v_proj",
            dpo_beta=config.DPO_BETA,
            merge_alpha=args.merge_alpha,
            lr=config.LEARNING_RATE,
            epochs=config.NUM_EPOCHS,
            no_4bit=not config.USE_4BIT,
            max_tokens=args.max_tokens,
            logger=None,
        )
        rows = {}
        for variant in variants:
            print(f"Evaluating {variant} (heuristic)...")
            m = eval_variant(eval_args, variant)
            rows[variant] = metrics_to_dict(m)
        ok_variants = list(rows.keys())
        ns = argparse.Namespace(
            STUDENT_MODEL=eval_args.STUDENT_MODEL,
            lora_r=eval_args.lora_r,
            lora_alpha=eval_args.lora_alpha,
            episodes=args.episodes,
            checkpoint_dir=args.staging_dir,
            variants=variants,
            paper=True,
        )
        suite = {
            "run_id": run_id,
            "finished_at": "",
            "wall_seconds": 0.0,
            "timings_seconds": {},
            "hyperparams": _hyperparams(ns),
            "table1": {v: {k: rows[v][k] for k in ("fc_id", "fc_ho", "cr_id", "cr_ho")} for v in ok_variants},
            "table2": {v: rows[v].get("per_game_cr", {}) for v in ok_variants},
            "table3": {v: rows[v].get("per_axis_cr", {}) for v in ok_variants},
            "table4": {v: rows[v].get("per_game_fc", {}) for v in ok_variants},
            "table5_manifest": _training_manifest(args.staging_dir),
            "table6": {"paper_reference": config.PAPER_TABLE6},
            "table7": {
                v: {
                    "ac": rows[v].get("per_game_ac", {}),
                    "fc": rows[v].get("per_game_fc", {}),
                    "exploitability": rows[v].get("per_game_exploitability", {}),
                }
                for v in ok_variants
            },
            "variants": variants,
            "rows": rows,
        }
        save_json(run_dir / "tables" / "suite.json", suite)
        save_json(run_dir / "tables" / "paper_refs.json", _paper_refs())
        _write_table_jsons(run_dir, suite)
        md = build_result_md(
            suite=suite,
            rows=rows,
            ok_variants=ok_variants,
            manifest=suite["table5_manifest"],
            run_dir=run_dir,
            run_id=run_id,
        )
        (run_dir / "tables" / "result.md").write_text(md, encoding="utf-8")
        from eval.table.build_latex_md import write_latex_md

        latex_path = write_latex_md(run_dir / "tables", suite, ok_variants=ok_variants)
        write_latest_pointer(run_dir)
        print(f"\nSaved {run_dir / 'tables' / 'result.md'}")
        print(f"Saved {latex_path}")
        return 0

    return run_eval_suite(suite_argv)


if __name__ == "__main__":
    raise SystemExit(main())
