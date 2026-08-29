#!/usr/bin/env python3
"""Stage-game Nash equilibrium action sets for Hypothesis B strategy S2.

Only the two-player normal-form matrix stages are handled (PD variants,
stag-hunt, BoS, matching-pennies, ipd-stage). ``ne_own_actions`` returns the
set of player-0 actions that are played with positive probability in some
equilibrium:

* if one or more pure-strategy equilibria exist, their player-0 actions;
* otherwise, for a 2x2 stage with no pure equilibrium, both actions (the
  unique fully-mixed equilibrium has full support);
* otherwise the empty set (S2 emits nothing for that stage).

No dependency on the training stack.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

Payoff = Mapping[tuple[Any, Any], tuple[float, float]]


def decode_bimatrix(payoff_rows: Sequence[Mapping[str, Any]]) -> Payoff:
    """``[{"own","opponent","payoff":[u0,u1]}, ...]`` -> ``{(own,opp):(u0,u1)}``."""
    table: dict[tuple[Any, Any], tuple[float, float]] = {}
    for entry in payoff_rows:
        vector = entry["payoff"]
        if not isinstance(vector, (list, tuple)) or len(vector) < 2:
            raise ValueError("S2 needs both players' payoffs in every payoff row")
        table[(entry["own"], entry["opponent"])] = (float(vector[0]), float(vector[1]))
    return table


def _unique(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def pure_equilibria(
    own_actions: Sequence[Any],
    opponent_actions: Sequence[Any],
    payoff: Payoff,
) -> set[tuple[Any, Any]]:
    equilibria: set[tuple[Any, Any]] = set()
    for a in own_actions:
        for b in opponent_actions:
            if (a, b) not in payoff:
                continue
            best_own = max(payoff[(x, b)][0] for x in own_actions if (x, b) in payoff)
            best_opp = max(payoff[(a, y)][1] for y in opponent_actions if (a, y) in payoff)
            if payoff[(a, b)][0] == best_own and payoff[(a, b)][1] == best_opp:
                equilibria.add((a, b))
    return equilibria


def ne_own_actions(
    payoff_rows: Sequence[Mapping[str, Any]],
    legal_actions: Sequence[Any],
) -> set[Any]:
    payoff = decode_bimatrix(payoff_rows)
    own_actions = _unique(legal_actions) or _unique(a for a, _ in payoff)
    opponent_actions = _unique(b for _, b in payoff) or list(own_actions)

    pure = pure_equilibria(own_actions, opponent_actions, payoff)
    if pure:
        return {a for a, _ in pure}
    if len(own_actions) == 2 and len(opponent_actions) == 2:
        return set(own_actions)
    return set()
