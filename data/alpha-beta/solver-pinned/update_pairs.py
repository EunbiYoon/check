#!/usr/bin/env python3
"""Build and load solver-pinned DPO prompt/chosen/rejected pairs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


def _load_best_action():
    """Load the sibling module even when this file is loaded by file path."""
    path = Path(__file__).with_name("best_action.py")
    spec = importlib.util.spec_from_file_location("solver_pinned_best_action", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load solver action module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


best_action = _load_best_action()

ACTION_RE = re.compile(r"(<action>)(.*?)(</action>)", re.DOTALL)


def format_action(action: Any) -> str:
    if isinstance(action, (list, tuple)):
        return "[" + ",".join(str(value) for value in action) + "]"
    return str(action)

### get best_action to change action label
def pin_actions(response: str, actions: list[Any]) -> str:
    """Replace the nth action tag with the nth solver label."""
    matches = list(ACTION_RE.finditer(response))
    if len(matches) != len(actions):
        raise ValueError(
            f"response has {len(matches)} <action> tags but trajectory has {len(actions)} rounds"
        )
    pieces: list[str] = []
    cursor = 0
    for match, action in zip(matches, actions):
        pieces.extend(
            (
                response[cursor : match.start()],
                match.group(1),
                format_action(action),
                match.group(3),
            )
        )
        cursor = match.end()
    pieces.append(response[cursor:])
    return "".join(pieces)


def build_pair(row: dict[str, Any]) -> dict[str, Any]:
    """Construct one ``prompt/chosen/rejected`` row.

    Required fields are ``prompt``, ``response``, ``opponent_actions``, and the
    game-family fields consumed by ``best_action``.  ``response`` is the
    recorded acting-model completion and becomes ``rejected``; only its action
    tags are changed in ``chosen``, preserving the recorded reasoning/context.
    """
    response = row.get("response", row.get("rejected"))
    if not isinstance(response, str):
        raise ValueError("row requires a string response (or rejected)")
    opponent_actions = row.get("opponent_actions")
    if not isinstance(opponent_actions, list) or not opponent_actions:
        raise ValueError("row requires a non-empty opponent_actions list")
    actions = [best_action(row, opponent_action) for opponent_action in opponent_actions]
    pair = {
        "prompt": row["prompt"],
        "chosen": pin_actions(response, actions),
        "rejected": response,
    }
    if row.get("id") is not None:
        pair["id"] = row["id"]
    if row.get("include_metadata"):
        pair["solver_actions"] = actions
        pair["opponent_actions"] = opponent_actions
        pair["game_family"] = row["game_family"]
    return pair


def build_file(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as src, destination.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                pair = build_pair(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            dst.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"no records in {source}")
    return count


def load_pairs(path: Path):
    """Load completed pairs as a Hugging Face Dataset for DPO training."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("load_pairs requires the 'datasets' package") from exc

    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                rows.append({key: row[key] for key in ("prompt", "chosen", "rejected")})
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number}: missing {exc.args[0]!r}") from exc
    if not rows:
        raise ValueError(f"No pairs in {path}")
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Recorded trajectory JSONL")
    parser.add_argument("--output", type=Path, required=True, help="DPO pair JSONL")
    args = parser.parse_args()
    try:
        count = build_file(args.input, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"wrote {count} solver-pinned pairs to {args.output}")


if __name__ == "__main__":
    main()
