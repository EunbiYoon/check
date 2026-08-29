"""Best-response label construction.

This module deliberately has no dependency on the training stack so that pair
construction can be run before installing PyTorch/TRL.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

### Each Games and matching function##
#    game_family                                        function
#   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#    pd-classic, pd-tight, pd-high-temptation,          matrix_best_action()
#    ipd-stage, stag-hunt, bos, matching-pennies
#   ─────────────────────────────────────────────────  ─────────────────────────────────────
#    auction, first-price-auction                       auction_best_action()
#   ─────────────────────────────────────────────────  ─────────────────────────────────────
#    bargaining, multi-issue-bargaining, negotiation    bargaining_best_action()


### The action a that maximizes my payoff against the opponent’s realized action a1.
def matrix_best_action(
    opponent_action: Any,
    own_action: Sequence[Any],
    payoffs: Mapping[tuple[Any, Any], float],
) -> Any:
    """Return the first payoff-maximising legal action against a realised move.

    ``payoffs[(own_action, opponent_action)]`` is player 0's payoff.  Keeping
    legal-action order as the tie-break makes construction deterministic.
    """
    if not own_action:
        raise ValueError("own_action must not be empty")
    missing = [
        (action, opponent_action)
        for action in own_action
        if (action, opponent_action) not in payoffs
    ]
    if missing:
        raise ValueError(f"missing player-0 payoffs for {missing}")
    # compare own_action, opponent_action
    return max(own_action, key=lambda action: payoffs[(action, opponent_action)])

def decode_matrix_payoffs(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, Any], float]:
    """Decode JSON-friendly payoff rows into the lookup used by the solver.

    Each row is ``{"own": ..., "opponent": ..., "payoff": ...}``; a two-item
    payoff vector is also accepted and its first item is player 0's payoff.
    """
    result: dict[tuple[Any, Any], float] = {}
    for row in rows:
        value = row["payoff"]
        if isinstance(value, list):
            value = value[0]
        result[(row["own"], row["opponent"])] = float(value)
    return result


def auction_best_action(
    opponent_bid: float,
    own_value: float,
    own_actions: Sequence[float],
) -> float:
    """Return the smallest legal bid above b1 if profitable, otherwise zero.

    Paper §2.2 states the rule as "smallest legal bid strictly above b1 when
    v0 > b1 + epsilon". On a discrete bid grid the smallest bid strictly above
    b1 *is* ``b1 + epsilon`` (epsilon = the grid step), and in a first-price
    auction the winner pays its own bid, so the profitability test is
    ``own_value > winning`` (the price actually paid) rather than
    ``own_value > b1``. The two coincide as the grid step shrinks; the form
    here is the one that never emits a money-losing bid.

    A tie is not treated as a win.  Zero must be a legal bid because it is the
    deliberate-loss label when winning would not be profitable.
    """
    bids = sorted(set(own_actions))
    if not bids:
        raise ValueError("own_actions must not be empty")
    if 0 not in bids:
        raise ValueError("own_actions must contain zero")
    winning = next((bid for bid in bids if bid > opponent_bid), None)
    # compare own_action, opponent_action
    return winning if winning is not None and own_value > winning else 0


def bargaining_best_action(
    opponent_proposal: Sequence[int],
    capacity: int | Sequence[int],
    own_weights: Sequence[float],
) -> tuple[int, ...]:
    """Return the feasible take-the-rest proposal, c[i] - p1[i]."""
    proposal = tuple(opponent_proposal)
    weights = tuple(own_weights)
    capacities = (
        (capacity,) * len(proposal) if isinstance(capacity, int) else tuple(capacity)
    )
    if not proposal or len(proposal) != len(weights) or len(proposal) != len(capacities):
        raise ValueError("proposal, capacity, and weights must have the same non-zero length")
    if any(weight < 0 for weight in weights):
        raise ValueError("take-the-rest is optimal only for non-negative own weights")
    if any(p < 0 or p > c for p, c in zip(proposal, capacities)):
        raise ValueError("opponent proposal must lie between zero and capacity")
    return tuple(c - p for p, c in zip(proposal, capacities))


def _auction_value(spec: Mapping[str, Any]) -> float:
    """Bidder's private value, accepting either recorded key name."""
    for key in ("own_value", "private_value"):
        if spec.get(key) is not None:
            return float(spec[key])
    raise ValueError("auction spec requires own_value (or private_value)")


def _auction_bid_menu(spec: Mapping[str, Any]) -> Sequence[float]:
    """Legal bid menu. ``legal_actions`` is the recorded-trajectory key; the
    solver-only ``own_actions`` name is still accepted for standalone specs.
    """
    menu = spec.get("legal_actions")
    if menu is None:
        menu = spec.get("own_actions")
    if not menu:
        raise ValueError("auction spec requires a legal bid menu (legal_actions)")
    return menu


### Player-0 stage payoff u0(own_action, opponent_action) for one round. ###
def matrix_utility(
    own_action: Any,
    opponent_action: Any,
    payoffs: Mapping[tuple[Any, Any], float],
) -> float:
    try:
        return float(payoffs[(own_action, opponent_action)])
    except KeyError as exc:
        raise ValueError(
            f"missing player-0 payoff for {(own_action, opponent_action)!r}"
        ) from exc


def auction_utility(own_bid: float, opponent_bid: float, own_value: float) -> float:
    """First-price payoff: value - bid on a strict win, zero otherwise."""
    return float(own_value) - float(own_bid) if float(own_bid) > float(opponent_bid) else 0.0


def bargaining_utility(
    own_proposal: Sequence[int],
    opponent_proposal: Sequence[int],
    capacity: int | Sequence[int],
    own_weights: Sequence[float],
) -> float:
    """Weighted claim value when the joint proposal is feasible, else zero."""
    own = tuple(own_proposal)
    other = tuple(opponent_proposal)
    weights = tuple(own_weights)
    capacities = (
        (capacity,) * len(own) if isinstance(capacity, int) else tuple(capacity)
    )
    if len(own) != len(other) or len(own) != len(weights) or len(own) != len(capacities):
        raise ValueError("proposal, capacity, and weights must have the same length")
    if any(o + p > c for o, p, c in zip(own, other, capacities)):
        return 0.0
    return float(sum(w * o for w, o in zip(weights, own)))


def stage_utility(spec: Mapping[str, Any], own_action: Any, opponent_action: Any) -> float:
    """Return u0(own_action, opponent_action) for the game in ``spec``.

    Used by the pair pipeline to gate one-shot flips on Algorithm 1's
    ``R_cf > R(tau_blind)`` test (paper §2.4).
    """
    family = str(spec["game_family"]).replace("_", "-").lower()
    if family in {
        "matrix",
        "matrix-game",
        "pd-classic",
        "pd-tight",
        "pd-high-temptation",
        "ipd-stage",
        "stag-hunt",
        "bos",
        "matching-pennies",
    }:
        return matrix_utility(
            own_action, opponent_action, decode_matrix_payoffs(spec["payoffs"])
        )
    if family in {"auction", "first-price-auction"}:
        return auction_utility(
            float(own_action), float(opponent_action), _auction_value(spec)
        )
    if family in {"bargaining", "multi-issue-bargaining", "negotiation"}:
        return bargaining_utility(
            own_action, opponent_action, spec["capacity"], spec["own_weights"]
        )
    raise ValueError(f"unsupported game_family: {spec['game_family']!r}")


### Return the best action ###
def main(spec: Mapping[str, Any], opponent_action: Any) -> Any:
    """Return the solver-selected action for a JSON game specification."""
    family = str(spec["game_family"]).replace("_", "-").lower()
    if family in {
        "matrix",
        "matrix-game",
        "pd-classic",
        "pd-tight",
        "pd-high-temptation",
        "ipd-stage",
        "stag-hunt",
        "bos",
        "matching-pennies",
    }:
        return matrix_best_action(
            opponent_action,
            spec["legal_actions"],
            decode_matrix_payoffs(spec["payoffs"]),
        )
    if family in {"auction", "first-price-auction"}:
        return auction_best_action(
            float(opponent_action),
            _auction_value(spec),
            _auction_bid_menu(spec),
        )
    if family in {"bargaining", "multi-issue-bargaining", "negotiation"}:
        return bargaining_best_action(
            opponent_action,
            spec["capacity"],
            spec["own_weights"],
        )
    raise ValueError(f"unsupported game_family: {spec['game_family']!r}")
