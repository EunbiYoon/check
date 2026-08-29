"""Locate pair-construction code and prepare a training dataset."""

from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path

import config


def _load_module(name: str, *relative_paths: str):
    candidates = [config.PROJECT_ROOT / path for path in relative_paths]
    target = next((path for path in candidates if path.is_file()), None)
    if target is None:
        parts = Path(relative_paths[0]).parts
        leaf = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        target = next(config.PROJECT_ROOT.glob(f"**/{leaf}"), None)
    if target is None:
        raise FileNotFoundError(f"cannot locate {relative_paths[0]} (tried: " + ", ".join(str(candidate) for candidate in candidates) + ")")
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


frontier = _load_module("noleakage_frontier", "data/alpha-beta/noleakage-frontier/build_pairs.py", "data-construction/alpha-beta/noleakage-frontier/build_pairs.py")
pair_data = _load_module("solver_pinned_pair_data", "data/alpha-beta/solver-pinned/update_pairs.py", "data-construction/alpha-beta/solver-pinned/update_pairs.py")
build_dpo_trainer = frontier.build_dpo_trainer
extract_train_metrics = frontier.extract_train_metrics
prepare_model = frontier.prepare_model


def prepare_pairs(args, run_dir: Path):
    if args.trajectories is None:
        pairs_path = Path(args.pairs)
        if not pairs_path.is_absolute():
            pairs_path = config.PROJECT_ROOT / pairs_path
        return pairs_path, pair_data.load_pairs(pairs_path)
    trajectories_path = args.trajectories
    if not trajectories_path.is_absolute():
        trajectories_path = config.PROJECT_ROOT / trajectories_path
    pairs_path = run_dir / "constructed_pairs.jsonl"
    if args.rebuild_pairs or not pairs_path.is_file():
        count = frontier.build_file(
            trajectories_path, pairs_path, provider="teacher",
            teacher_STUDENT_MODEL=args.teacher_model or config.TEACHER_MODEL,
            teacher_max_new_tokens=args.teacher_max_new_tokens,
            counterfactual_mode=args.counterfactual_mode,
            require_frontier_model_match=not args.allow_unverified_frontier_model,
        )
        print(f"Constructed {count} verified pairs from {trajectories_path}", flush=True)
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pairs_path, pair_data.load_pairs(pairs_path)
