#!/usr/bin/env python3
"""Hypothesis B data construction (paper Appendix A).

Reads a blind rollout trajectory JSONL (frontier model as player 0 against
heuristic opponents), tags every round for reasoning-action coupling, builds
candidate DPO pairs with strategies S1-S4, and -- when the coupling filter is
on -- keeps only the pairs whose chosen-side *trajectory* is coupled.

    coupling filter OFF  ->  the ``filter_off`` variant
    coupling filter ON   ->  the ``filter_on`` variant

Every pair keeps the frontier's recorded completion as ``chosen`` (reasoning
verbatim, action = frontier's own play) and swaps only the ``<action>`` payload
for ``rejected``. Output rows are ``{prompt, chosen, rejected, ...}``, the schema
consumed by ``train/dpo-lora/variant.py --pairs``.

This pipeline is a reconstruction of a *falsified* baseline: the paper reports
that coupling-filter DPO did not bind reasoning to action and that the filter
narrowed the chosen-side distribution. See ``README.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from coupling_pool import PIPELINE, Pool, build_pool
from strategies import STRATEGIES, Candidate


def _assemble(group: list[Candidate], *, coupling_filter: bool,
              coupled_by_trajectory: dict[str, bool]) -> dict[str, Any]:
    first = group[0]
    chosen = first.chosen
    strategies = sorted({candidate.strategy for candidate in group})
    return {
        "prompt": chosen.prompt,
        "chosen": chosen.completion,
        "rejected": first.rejected_completion,
        "strategy": strategies[0] if len(strategies) == 1 else "+".join(strategies),
        "trajectory_id": chosen.trajectory_id,
        "round": chosen.round_number,
        "provenance": {
            "pipeline": PIPELINE,
            "chosen_reasoning": "frontier-blind",
            "chosen_action": "frontier-recorded",
            "rejected": "frontier-blind-reasoning-with-swapped-action",
            "coupling_filter": "on" if coupling_filter else "off",
            "strategies": strategies,
            "strategy_detail": {c.strategy: c.detail for c in group},
            "chosen_action_label": chosen.action_label,
            "rejected_action_label": first.rejected_label,
            "chosen_round_coupled": chosen.coupled,
            "chosen_round_ev_parsable": chosen.ev_parsable,
            "chosen_trajectory_coupled": coupled_by_trajectory.get(
                chosen.trajectory_id, False
            ),
            "cumulative_reward": chosen.trajectory_return,
            "game": chosen.game,
            "opponent": chosen.opponent,
        },
    }


def _dedup_key(candidate: Candidate, mode: str) -> tuple:
    if mode == "none":
        return (id(candidate),)              # never merges
    if mode == "full":
        # one pair per exact (prompt, chosen text, rejected text)
        return (candidate.prompt, candidate.chosen.completion, candidate.rejected_completion)
    if mode == "context":
        # one pair per (decision context, action contrast); the blind reasoning
        # text varies episode-to-episode but the preference is the same signal
        return (candidate.prompt, candidate.chosen.action_label, candidate.rejected_label)
    if mode == "position":
        # one pair per (game, round, action contrast) — the coarsest grouping
        c = candidate.chosen
        return (c.game, c.round_number, c.action_label, candidate.rejected_label)
    raise ValueError("dedup must be 'none', 'full', 'context', or 'position'")


def _representative(group: list[Candidate], coupled_by_trajectory: dict[str, bool]) -> Candidate:
    """Best candidate to stand for a merged group: coupled chosen first, then
    higher cumulative reward, then the longer (more explicit) reasoning."""
    return max(
        group,
        key=lambda cand: (
            coupled_by_trajectory.get(cand.chosen.trajectory_id, False),
            cand.chosen.trajectory_return,
            len(cand.chosen.reasoning),
        ),
    )


def build_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    coupling_filter: bool = False,
    strategies: Iterable[str] = ("S1", "S2", "S3", "S4"),
    dedup: str = "full",
    max_pairs: int | None = None,
) -> list[dict[str, Any]]:
    requested = [name.strip().upper() for name in strategies if name.strip()]
    unknown = [name for name in requested if name not in STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategy(ies): {', '.join(unknown)}")

    pool: Pool = build_pool(rows)

    candidates: list[Candidate] = []
    for name in requested:
        candidates.extend(STRATEGIES[name](pool))

    # Merge candidates that carry the same preference signal (the granularity is
    # set by `dedup`); >1 strategy hitting the same key just adds a provenance tag.
    merged: dict[tuple, list[Candidate]] = {}
    for candidate in candidates:
        merged.setdefault(_dedup_key(candidate, dedup), []).append(candidate)

    pairs: list[dict[str, Any]] = []
    for group in sorted(merged.values(), key=lambda g: g[0].prompt):
        keep = _representative(group, pool.coupled_by_trajectory)
        if coupling_filter and not pool.coupled_by_trajectory.get(keep.chosen.trajectory_id, False):
            continue
        # keep the representative first so its (prompt, chosen, rejected) is emitted
        ordered = [keep] + [c for c in group if c is not keep]
        pairs.append(
            _assemble(
                ordered,
                coupling_filter=coupling_filter,
                coupled_by_trajectory=pool.coupled_by_trajectory,
            )
        )

    if max_pairs is not None and len(pairs) > max_pairs:
        import random

        random.Random(0).shuffle(pairs)
        pairs = pairs[:max_pairs]
    return pairs


def _summary(pairs: list[dict[str, Any]], pool: Pool) -> str:
    by_strategy: dict[str, int] = {}
    for pair in pairs:
        for name in pair["provenance"]["strategies"]:
            by_strategy[name] = by_strategy.get(name, 0) + 1
    parts = [f"{name}={count}" for name, count in sorted(by_strategy.items())]
    coupled = sum(1 for value in pool.coupled_by_trajectory.values() if value)
    skipped = (
        f", skipped {len(pool.skipped)} unparsable row(s)" if pool.skipped else ""
    )
    return (
        f"{len(pairs)} pairs ({', '.join(parts) or 'none'}); "
        f"{coupled}/{len(pool.coupled_by_trajectory)} trajectories coupled{skipped}"
    )


def build_file(
    source: Path,
    destination: Path,
    *,
    coupling_filter: bool = False,
    strategies: Iterable[str] = ("S1", "S2", "S3", "S4"),
    dedup: str = "full",
    max_pairs: int | None = None,
) -> int:
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc

    pool = build_pool(rows)
    pairs = build_pairs(
        rows, coupling_filter=coupling_filter, strategies=strategies,
        dedup=dedup, max_pairs=max_pairs,
    )
    if not pairs:
        if coupling_filter and not any(pool.coupled_by_trajectory.values()):
            raise ValueError(
                "no Hypothesis B pairs were emitted: the coupling filter is on "
                "but no trajectory has a parsable [EV] slot in its final round "
                "(the frontier reasoning in this pool is not slot-form)"
            )
        raise ValueError(
            "no Hypothesis B pairs were emitted -- the trajectory pool has no "
            "strategy-preferred action with a distinct dispreferred alternative "
            "(the paper's 'training-distribution narrowness', Appendix A)"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(_summary(pairs, pool), file=sys.stderr)
    return len(pairs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="blind rollout trajectory JSONL")
    parser.add_argument("--output", type=Path, required=True,
                        help="DPO pair JSONL to write")
    parser.add_argument(
        "--coupling-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep only pairs whose chosen-side trajectory is coupled "
             "(--coupling-filter = filter_on, --no-coupling-filter = filter_off)",
    )
    parser.add_argument(
        "--strategies",
        default="S1,S2,S3,S4",
        help="comma-separated subset of S1,S2,S3,S4",
    )
    parser.add_argument(
        "--dedup",
        choices=("none", "full", "context", "position"),
        default="context",
        help="merge granularity: 'full' = one pair per exact (prompt, chosen, "
             "rejected) text; 'context' = one per (prompt, chosen action, "
             "rejected action) [default]; 'position' = one per (game, round, "
             "action contrast)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="cap the output at N pairs (deterministic seeded sample)",
    )
    args = parser.parse_args(argv)
    try:
        count = build_file(
            args.input,
            args.output,
            coupling_filter=args.coupling_filter,
            strategies=args.strategies.split(","),
            dedup=args.dedup,
            max_pairs=args.max_pairs,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    variant = "filter_on" if args.coupling_filter else "filter_off"
    print(f"wrote {count} {variant} pairs to {args.output}")


if __name__ == "__main__":
    main()
