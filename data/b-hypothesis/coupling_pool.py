#!/usr/bin/env python3
"""Hypothesis B, Algorithm lines 2-6: parse blind trajectories into a pooled
set of per-round observations and tag each one for reasoning-action coupling.

A blind rollout row (``train/ab-data-construction/blind-rollout``) is expanded
into one :class:`RoundObs` per recorded round. Each observation carries the
history-only prompt, the frontier completion, the played action, the realised
stage / cumulative reward, and a coupling flag computed from the ``[EV]`` slot
of the chain-of-thought (``eval.metric.coupling``).

The module has no dependency on the training stack (torch / trl); it can run
before those packages are installed.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

def _find_repo_root() -> Path:
    """Walk up until a directory holds both ``config.py`` and the coupling
    metric. Robust to how deeply this package is nested under ``train/``.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "config.py").exists() and (
            parent / "eval" / "metric" / "coupling.py"
        ).exists():
            return parent
    raise FileNotFoundError("cannot locate the repository root")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.metric.coupling import argmax_stated_ev, coupling_for_round, parse_ev_slot

PIPELINE = "hypothesis-b-coupling-filter-v1"

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"(<action>)(.*?)(</action>)", re.IGNORECASE | re.DOTALL)
_COMPLETION_RE = re.compile(
    r"<think>.*?</think>\s*<action>.*?</action>", re.IGNORECASE | re.DOTALL
)


def swap_action(completion: str, new_label: str) -> str:
    """Return ``completion`` with its single ``<action>`` payload replaced.

    Hypothesis B keeps the frontier reasoning verbatim and pins the chosen-side
    action to the frontier's own recorded play; the rejected side of a pair is
    the same completion with the action swapped to a strategy-dispreferred label
    (cf. ``solver-pinned/update_pairs.pin_actions``, which does the mirror image
    for the chosen side).
    """
    replaced, count = _ACTION_RE.subn(
        lambda m: f"{m.group(1)}{new_label}{m.group(3)}", completion
    )
    if count != 1:
        raise ValueError(f"expected exactly one <action> block, found {count}")
    return replaced

_MATRIX_FAMILIES = {
    "matrix",
    "matrix-game",
    "pd-classic",
    "pd-tight",
    "pd-high-temptation",
    "ipd-stage",
    "stag-hunt",
    "bos",
    "matching-pennies",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _solver_path() -> Path:
    # The A+β solver has moved across refactors (train/ab-data-construction ->
    # data-construction/alpha-beta -> data/alpha-beta); try the known homes,
    # then search the tree.
    candidates = [
        REPO_ROOT / "data" / "alpha-beta" / "solver-pinned" / "best_action.py",
        REPO_ROOT / "data-construction" / "alpha-beta" / "solver-pinned" / "best_action.py",
        REPO_ROOT / "train" / "ab-data-construction" / "solver-pinned" / "best_action.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = next(REPO_ROOT.glob("**/solver-pinned/best_action.py"), None)
    if found is None:
        raise FileNotFoundError("cannot locate solver-pinned/best_action.py")
    return found


solver = _load(_solver_path(), "hb_solver")
stage_utility = solver.stage_utility


def action_str(action: Any) -> str:
    """Canonical string form, matching the blind rollout's ``action_text``."""
    if isinstance(action, (list, tuple)):
        return json.dumps(list(action), separators=(",", ":"))
    return str(action)


def family_of(row: dict[str, Any]) -> str:
    return str(row.get("game_family", "")).replace("_", "-").lower()


@dataclass
class RoundObs:
    """One recorded decision in a blind trajectory."""

    trajectory_id: str
    game: str
    game_family: str
    opponent: str
    round_index: int          # 0-based
    round_number: int         # 1-based
    horizon: int
    prompt: str               # history-only prompt for this round
    completion: str           # <think>...</think><action>...</action>
    reasoning: str
    action: Any               # played action (recorded, canonical value)
    action_label: str         # action_str(action)
    opponent_action: Any
    legal_actions: list[Any]
    payoff_rows: list[dict[str, Any]] | None
    row: dict[str, Any] = field(repr=False, default_factory=dict)
    stage_reward: float = 0.0
    trajectory_return: float = 0.0      # cumulative own reward over the episode
    continuation_return: float = 0.0    # realised reward from this round onward
    ev_parsable: bool = False
    coupled: bool = False
    stated_ev: dict[str, float] = field(default_factory=dict)
    argmax_ev: str | None = None

    @property
    def is_matrix(self) -> bool:
        return self.game_family in _MATRIX_FAMILIES

    @property
    def signature(self) -> tuple[str, int]:
        """Structural position key used for cross-strategy consensus (S3)."""
        return (self.game, self.round_number)


def _round_completions(row: dict[str, Any], count: int) -> list[str]:
    supplied = row.get("round_completions")
    if isinstance(supplied, list) and all(isinstance(x, str) for x in supplied):
        return supplied
    response = row.get("response", row.get("rejected"))
    if isinstance(response, str):
        found = [m.group(0) for m in _COMPLETION_RE.finditer(response)]
        if len(found) == count:
            return found
    raise ValueError("row requires round_completions (or a matching response)")


def _round_prompts(row: dict[str, Any], count: int) -> list[str]:
    prompts = row.get("round_prompts")
    if isinstance(prompts, list) and len(prompts) == count and all(
        isinstance(x, str) and x.strip() for x in prompts
    ):
        return prompts
    if count == 1 and isinstance(row.get("prompt"), str):
        return [row["prompt"]]
    raise ValueError("row requires one history-only round_prompts entry per round")


def _reasoning_of(completion: str) -> str:
    match = _THINK_RE.search(completion)
    return match.group(1).strip() if match else ""


def expand_row(row: dict[str, Any]) -> list[RoundObs]:
    """Turn one blind trajectory into its list of :class:`RoundObs`."""
    trajectory_id = row.get("id") or row.get("trajectory_id")
    if not trajectory_id:
        raise ValueError("trajectory requires a stable id")
    own_actions = row.get("own_actions", row.get("agent_actions"))
    opponent_actions = row.get("opponent_actions")
    if not isinstance(own_actions, list) or not isinstance(opponent_actions, list):
        raise ValueError("trajectory requires own_actions and opponent_actions lists")
    horizon = len(own_actions)
    if horizon == 0 or len(opponent_actions) != horizon:
        raise ValueError("own_actions and opponent_actions must share a non-zero length")
    completions = _round_completions(row, horizon)
    prompts = _round_prompts(row, horizon)
    if len(completions) != horizon:
        raise ValueError("one blind completion is required per recorded round")

    family = family_of(row)
    legal_actions = list(row.get("legal_actions", []))
    payoff_rows = row.get("payoffs") if isinstance(row.get("payoffs"), list) else None

    # Realised per-round utilities; the solver understands the recorded schema.
    stage_rewards = [
        float(stage_utility(row, own_actions[i], opponent_actions[i]))
        for i in range(horizon)
    ]
    total = sum(stage_rewards)

    observations: list[RoundObs] = []
    for i in range(horizon):
        reasoning = _reasoning_of(completions[i])
        result = coupling_for_round(reasoning, action_str(own_actions[i]))
        stated = parse_ev_slot(reasoning)
        observations.append(
            RoundObs(
                trajectory_id=str(trajectory_id),
                game=str(row.get("game", family)),
                game_family=family,
                opponent=str(row.get("opponent", row.get("opponent_name", ""))),
                round_index=i,
                round_number=i + 1,
                horizon=horizon,
                prompt=prompts[i],
                completion=completions[i],
                reasoning=reasoning,
                action=own_actions[i],
                action_label=action_str(own_actions[i]),
                opponent_action=opponent_actions[i],
                legal_actions=legal_actions,
                payoff_rows=payoff_rows,
                row=row,
                stage_reward=stage_rewards[i],
                trajectory_return=total,
                continuation_return=sum(stage_rewards[i:]),
                ev_parsable=result.parsable,
                coupled=result.coupled,
                stated_ev=stated,
                argmax_ev=argmax_stated_ev(stated),
            )
        )
    return observations


def trajectory_coupled(observations: Iterable[RoundObs]) -> bool:
    """Algorithm line 5: ``coupled(tau)`` is the coupling of the last round
    whose ``[EV]`` slot is parsable (the paper's final-round coupling metric).

    A trajectory with no parsable EV slot in any round is treated as uncoupled,
    so the coupling filter drops it.
    """
    parsable = [obs for obs in observations if obs.ev_parsable]
    if not parsable:
        return False
    return parsable[-1].coupled


@dataclass
class Pool:
    observations: list[RoundObs]
    skipped: list[tuple[str, str]]                 # (row id, reason)
    coupled_by_trajectory: dict[str, bool]

    def by_prompt(self) -> dict[str, list[RoundObs]]:
        groups: dict[str, list[RoundObs]] = defaultdict(list)
        for obs in self.observations:
            groups[obs.prompt].append(obs)
        return groups

    def by_signature(self) -> dict[tuple[str, int], list[RoundObs]]:
        groups: dict[tuple[str, int], list[RoundObs]] = defaultdict(list)
        for obs in self.observations:
            groups[obs.signature].append(obs)
        return groups

    def observed_actions(self) -> dict[tuple[str, int], dict[str, Any]]:
        """``signature -> {action_label: action_value}`` seen anywhere in the
        pool, used to draw *real* counterfactual actions for a round."""
        seen: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
        for obs in self.observations:
            seen[obs.signature].setdefault(obs.action_label, obs.action)
        return seen


def build_pool(rows: Iterable[dict[str, Any]]) -> Pool:
    observations: list[RoundObs] = []
    skipped: list[tuple[str, str]] = []
    grouped: dict[str, list[RoundObs]] = defaultdict(list)
    for index, row in enumerate(rows):
        identifier = str(row.get("id") or row.get("trajectory_id") or f"row-{index}")
        try:
            expanded = expand_row(row)
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append((identifier, str(exc)))
            continue
        observations.extend(expanded)
        grouped[identifier].extend(expanded)
    coupled_by_trajectory = {
        tid: trajectory_coupled(obs_list) for tid, obs_list in grouped.items()
    }
    return Pool(observations, skipped, coupled_by_trajectory)
