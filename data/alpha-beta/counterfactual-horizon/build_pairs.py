#!/usr/bin/env python3
"""Filter solver flips with fixed or horizon-aware counterfactual returns."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from counterfactual_horizon import (
    fixed_continuation_return,
    horizon_aware_return,
    is_reconstructible,
    recorded_return,
    validate_recorded_trajectory,
)

ACTION_RE = re.compile(r"(<action>)(.*?)(</action>)", re.DOTALL)


def _load_solver_best_action():
    path = Path(__file__).resolve().parents[1] / "solver-pinned" / "best_action.py"
    spec = importlib.util.spec_from_file_location("solver_pinned_best_action", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load solver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main, module.decode_matrix_payoffs


best_action, decode_matrix_payoffs = _load_solver_best_action()


def format_action(action: Any) -> str:
    return str(action)


def pin_action_at(response: str, round_index: int, action: Any) -> str:
    """Replace only the candidate round's action, preserving all other text."""
    matches = list(ACTION_RE.finditer(response))
    if not 0 <= round_index < len(matches):
        raise ValueError(f"response has no <action> for round {round_index}")
    match = matches[round_index]
    return response[: match.start(2)] + format_action(action) + response[match.end(2) :]


def build_pairs(row: dict[str, Any], mode: str = "horizon-aware") -> list[dict[str, Any]]:
    """Emit per-round solver flips whose counterfactual beats τ_blind."""
    if str(row.get("game_family", "")).replace("_", "-").lower() not in {
        "matrix",
        "matrix-game",
    }:
        raise ValueError("§2.4 horizon filter supports repeated matrix games only")
    response = row.get("response", row.get("rejected"))
    own_actions = row.get("own_actions", row.get("agent_actions"))
    opponent_actions = row.get("opponent_actions")
    legal_actions = row.get("legal_actions")
    opponent_name = row.get("opponent", row.get("opponent_name"))
    if not isinstance(response, str):
        raise ValueError("row requires a string response (or rejected)")
    if not isinstance(own_actions, list) or not isinstance(opponent_actions, list):
        raise ValueError("row requires own_actions and opponent_actions lists")
    if not isinstance(legal_actions, list) or not isinstance(opponent_name, str):
        raise ValueError("row requires legal_actions and opponent name")
    if len(ACTION_RE.findall(response)) != len(own_actions):
        raise ValueError("response <action> count must equal trajectory horizon")
    if not is_reconstructible(opponent_name):
        # §2.4: the filter needs a reconstructible opponent; skip this trajectory.
        return []
    validate_recorded_trajectory(own_actions, opponent_actions, opponent_name, legal_actions)
    payoffs = decode_matrix_payoffs(row["payoffs"])
    baseline = recorded_return(own_actions, opponent_actions, payoffs)
    return_fn = horizon_aware_return if mode == "horizon-aware" else fixed_continuation_return
    if mode not in {"horizon-aware", "fixed"}:
        raise ValueError("mode must be 'horizon-aware' or 'fixed'")

    pairs = []
    for round_index, (recorded, opponent) in enumerate(zip(own_actions, opponent_actions)):
        solver_action = best_action(row, opponent)
        if solver_action == recorded:
            continue
        counterfactual = return_fn(
            own_actions,
            opponent_actions,
            round_index,
            solver_action,
            opponent_name,
            legal_actions,
            payoffs,
        )
        if counterfactual <= baseline:
            continue
        pair = {
            "prompt": row["prompt"],
            "chosen": pin_action_at(response, round_index, solver_action),
            "rejected": response,
            "meta": {
                "filter": mode,
                "flip_round": round_index,
                "recorded_action": recorded,
                "solver_action": solver_action,
                "recorded_return": baseline,
                "counterfactual_return": counterfactual,
                "opponent": opponent_name,
            },
        }
        if row.get("id") is not None:
            pair["id"] = f"{row['id']}:round-{round_index}"
        pairs.append(pair)
    return pairs


def build_file(source: Path, destination: Path, mode: str = "horizon-aware") -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                pairs = build_pairs(json.loads(line), mode)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            for pair in pairs:
                dst.write(json.dumps(pair, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("horizon-aware", "fixed"), default="horizon-aware")
    args = parser.parse_args()
    try:
        count = build_file(args.input, args.output, args.mode)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"wrote {count} {args.mode} counterfactual pairs to {args.output}")


if __name__ == "__main__":
    main()
