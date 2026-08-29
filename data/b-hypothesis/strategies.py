#!/usr/bin/env python3
"""Hypothesis B, Algorithm line 7: candidate DPO-pair construction.

Every candidate keeps the frontier model's recorded completion as the **chosen**
side (reasoning verbatim, action = the frontier's own recorded play) and builds
the **rejected** side by swapping only the ``<action>`` payload to a
strategy-dispreferred label. This mirrors ``solver-pinned/update_pairs`` (which
does the swap on the chosen side for A+beta) and keeps chosen-side text 100%
real, so the only thing DPO learns is which action the same reasoning should
commit to.

Four strategies decide when the recorded action is *preferred* and which action
is the *dispreferred* alternative:

* **S1 - outcome / cumulative reward.** The recorded action earns a strictly
  higher realised stage payoff than the alternative, and the round comes from an
  episode whose cumulative reward is at or above the median for its
  (game, opponent) cell.
* **S2 - NE-aligned action.** The recorded action lies in the stage game's
  Nash-equilibrium support; the alternative does not.
* **S3 - cross-strategy consensus.** The recorded action is the one most
  frontier trajectories converge on for that game/round position (across
  opponents); the alternative is an observed deviation.
* **S4 - L2 vs L0 contrastive.** The recorded action is the level-2
  cognitive-hierarchy action (best response to the best response to naive
  play); the alternative is the level-0 anchor.

No dependency on the training stack.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from coupling_pool import Pool, RoundObs, action_str, solver, stage_utility, swap_action
from nash import ne_own_actions

# S3 only trusts a consensus that is both clear and well-supported.
S3_CONSENSUS_THRESHOLD = 0.5
S3_MIN_OBSERVATIONS = 3
# Matrix stages are small enough to sweep every legal alternative; larger action
# spaces (auction, bargaining) only draw alternatives actually seen in the pool.
_SWEEP_LEGAL_MAX = 4


@dataclass
class Candidate:
    chosen: RoundObs
    rejected_label: str
    rejected_completion: str
    strategy: str
    detail: dict[str, Any]

    @property
    def prompt(self) -> str:
        return self.chosen.prompt


def _alternative_actions(obs: RoundObs, pool: Pool) -> dict[str, Any]:
    """``{label: value}`` of candidate rejected actions for this round."""
    alternatives: dict[str, Any] = dict(pool.observed_actions().get(obs.signature, {}))
    legal = list(obs.legal_actions)
    if 0 < len(legal) <= _SWEEP_LEGAL_MAX:
        for value in legal:
            alternatives.setdefault(action_str(value), value)
    alternatives.pop(obs.action_label, None)
    return alternatives


def _emit(chosen: RoundObs, label: str, strategy: str, detail: dict[str, Any]
          ) -> Candidate | None:
    if label == chosen.action_label:
        return None
    try:
        rejected = swap_action(chosen.completion, label)
    except ValueError:
        return None
    if rejected == chosen.completion:
        return None
    return Candidate(chosen, label, rejected, strategy, detail)


# --------------------------------------------------------------------------- S1


def _cumulative_medians(pool: Pool) -> dict[tuple[str, str], float]:
    returns: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen: set[tuple[str, tuple[str, str]]] = set()
    for obs in pool.observations:
        cell = (obs.game, obs.opponent)
        if (obs.trajectory_id, cell) in seen:
            continue
        seen.add((obs.trajectory_id, cell))
        returns[cell].append(obs.trajectory_return)
    return {cell: statistics.median(values) for cell, values in returns.items()}


def strategy_outcome(pool: Pool) -> list[Candidate]:
    medians = _cumulative_medians(pool)
    candidates: list[Candidate] = []
    for obs in pool.observations:
        if obs.trajectory_return < medians.get((obs.game, obs.opponent), float("-inf")):
            continue  # this episode did badly overall; do not reinforce its play
        try:
            recorded_payoff = stage_utility(obs.row, obs.action, obs.opponent_action)
        except (KeyError, TypeError, ValueError):
            continue
        for label, value in _alternative_actions(obs, pool).items():
            try:
                alt_payoff = stage_utility(obs.row, value, obs.opponent_action)
            except (KeyError, TypeError, ValueError):
                continue
            if recorded_payoff <= alt_payoff:
                continue
            candidate = _emit(
                obs, label, "S1-outcome",
                {
                    "recorded_stage_payoff": recorded_payoff,
                    "rejected_stage_payoff": alt_payoff,
                    "cumulative_reward": obs.trajectory_return,
                },
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


# --------------------------------------------------------------------------- S2


def strategy_ne_aligned(pool: Pool) -> list[Candidate]:
    equilibrium_cache: dict[str, set[Any]] = {}
    candidates: list[Candidate] = []
    for obs in pool.observations:
        if not obs.is_matrix or not obs.payoff_rows:
            continue
        if obs.game not in equilibrium_cache:
            try:
                equilibrium_cache[obs.game] = ne_own_actions(
                    obs.payoff_rows, obs.legal_actions
                )
            except (KeyError, TypeError, ValueError):
                equilibrium_cache[obs.game] = set()
        equilibrium = equilibrium_cache[obs.game]
        if not equilibrium or obs.action not in equilibrium:
            continue
        if len(equilibrium) >= len(set(map(action_str, obs.legal_actions))):
            continue  # every action is NE-aligned; nothing to contrast
        for value in obs.legal_actions:
            if value in equilibrium:
                continue
            candidate = _emit(
                obs, action_str(value), "S2-ne",
                {
                    "ne_actions": sorted(action_str(a) for a in equilibrium),
                    "rejected_action": action_str(value),
                },
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


# --------------------------------------------------------------------------- S3


def consensus_actions(pool: Pool) -> dict[tuple[str, int], str]:
    consensus: dict[tuple[str, int], str] = {}
    for signature, group in pool.by_signature().items():
        if len(group) < S3_MIN_OBSERVATIONS:
            continue
        counts = Counter(obs.action_label for obs in group)
        label, hits = counts.most_common(1)[0]
        if hits < len(group) and hits / len(group) >= S3_CONSENSUS_THRESHOLD:
            consensus[signature] = label
    return consensus


def strategy_consensus(pool: Pool) -> list[Candidate]:
    consensus = consensus_actions(pool)
    observed = pool.observed_actions()
    candidates: list[Candidate] = []
    for obs in pool.observations:
        target = consensus.get(obs.signature)
        if target is None or obs.action_label != target:
            continue
        for label in observed.get(obs.signature, {}):
            if label == target:
                continue
            candidate = _emit(
                obs, label, "S3-consensus",
                {
                    "position": {"game": obs.game, "round": obs.round_number},
                    "consensus_action": target,
                    "deviating_action": label,
                },
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


# --------------------------------------------------------------------------- S4


def _matrix_levels(obs: RoundObs) -> tuple[str, str] | None:
    payoff = solver.decode_matrix_payoffs(obs.payoff_rows)
    legal = list(dict.fromkeys(obs.legal_actions))
    if len(legal) < 2:
        return None
    level0 = legal[0]  # naive anchor: first legal action
    level1 = solver.matrix_best_action(level0, legal, payoff)
    level2 = solver.matrix_best_action(level1, legal, payoff)
    return action_str(level0), action_str(level2)


def _auction_levels(obs: RoundObs) -> tuple[str, str] | None:
    try:
        value = solver._auction_value(obs.row)
        menu = [float(bid) for bid in solver._auction_bid_menu(obs.row)]
    except (KeyError, TypeError, ValueError):
        return None
    if not menu:
        return None
    level0 = min(menu, key=lambda bid: abs(bid - value))  # truthful bid ~ value
    level1 = solver.auction_best_action(level0, value, menu)
    level2 = solver.auction_best_action(level1, value, menu)
    return action_str(level0), action_str(level2)


def _levels(obs: RoundObs) -> tuple[str, str] | None:
    if obs.is_matrix and obs.payoff_rows:
        try:
            return _matrix_levels(obs)
        except (KeyError, TypeError, ValueError):
            return None
    if obs.game_family in {"auction", "first-price-auction"}:
        return _auction_levels(obs)
    return None


def strategy_l2_l0(pool: Pool) -> list[Candidate]:
    candidates: list[Candidate] = []
    for obs in pool.observations:
        levels = _levels(obs)
        if levels is None:
            continue
        level0, level2 = levels
        if level0 == level2 or obs.action_label != level2:
            continue
        candidate = _emit(
            obs, level0, "S4-l2-l0",
            {"level0_action": level0, "level2_action": level2},
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


STRATEGIES = {
    "S1": strategy_outcome,
    "S2": strategy_ne_aligned,
    "S3": strategy_consensus,
    "S4": strategy_l2_l0,
}
