"""Counterfactual-horizon returns for repeated matrix games."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence

History = tuple[tuple[Any, Any], ...]
Payoffs = Mapping[tuple[Any, Any], float]

_DETERMINISTIC_OPPONENTS = {
    "always_cooperate",
    "always_defect",
    "tit_for_tat",
    "grim_trigger",
    "pavlov",
    "tit_for_two_tats",
}


def is_reconstructible(name: str) -> bool:
    """True when the opponent policy can be replayed exactly from its name.

    The counterfactual horizon filter (paper §2.4) requires a deterministic
    opponent whose policy is a known function of history. Stochastic pools
    (``random``, ``epsilon_greedy``, ``generous_tft``) are not reconstructible;
    trajectories against them are skipped rather than filtered.
    """
    return name in _DETERMINISTIC_OPPONENTS


def opponent_action(name: str, history: History, legal_actions: Sequence[Any]) -> Any:
    """Reconstruct a deterministic opponent policy from its name and history."""
    if len(legal_actions) != 2:
        raise ValueError("counterfactual horizon requires exactly two legal actions")
    if name not in _DETERMINISTIC_OPPONENTS:
        raise ValueError(f"opponent {name!r} is not an exact deterministic policy")
    cooperate, defect = legal_actions
    if name == "always_cooperate":
        return cooperate
    if name == "always_defect":
        return defect
    if name == "tit_for_tat":
        return cooperate if not history else history[-1][0]
    if name == "grim_trigger":
        return defect if any(own == defect for own, _ in history) else cooperate
    if name == "pavlov":
        # Matches eval.rollout.opponents.Pavlov: repeat its own previous action.
        return cooperate if not history else history[-1][1]
    # Matches eval.rollout.opponents.TitForTwoTats: retaliate after two consecutive
    # defections, otherwise mirror the student's immediately preceding action.
    trailing_defections = 0
    for own, _ in reversed(history):
        if own != defect:
            break
        trailing_defections += 1
    if trailing_defections >= 2:
        return defect
    return cooperate if not history else history[-1][0]


def stage_reward(payoffs: Payoffs, own_action: Any, opponent: Any) -> float:
    try:
        return float(payoffs[(own_action, opponent)])
    except KeyError as exc:
        raise ValueError(f"missing payoff for {(own_action, opponent)!r}") from exc


def validate_recorded_trajectory(
    own_actions: Sequence[Any],
    opponent_actions: Sequence[Any],
    opponent_name: str,
    legal_actions: Sequence[Any],
) -> History:
    """Ensure the recorded opponent actions equal π_opp(h_s) at every round."""
    if not own_actions or len(own_actions) != len(opponent_actions):
        raise ValueError("own_actions and opponent_actions need the same non-zero length")
    if len(own_actions) > 10:
        raise ValueError("paper §2.4 implementation supports horizon T <= 10")
    if len(legal_actions) != 2 or any(action not in legal_actions for action in own_actions):
        raise ValueError("all own actions must belong to exactly two legal_actions")
    history: History = ()
    for round_index, (own, recorded_opponent) in enumerate(
        zip(own_actions, opponent_actions)
    ):
        expected = opponent_action(opponent_name, history, legal_actions)
        if recorded_opponent != expected:
            raise ValueError(
                f"round {round_index}: recorded opponent action {recorded_opponent!r} "
                f"does not match {opponent_name!r} policy action {expected!r}"
            )
        history += ((own, recorded_opponent),)
    return history


def recorded_return(own_actions: Sequence[Any], opponent_actions: Sequence[Any], payoffs: Payoffs) -> float:
    return sum(stage_reward(payoffs, own, opp) for own, opp in zip(own_actions, opponent_actions))


def fixed_continuation_return(
    own_actions: Sequence[Any],
    opponent_actions: Sequence[Any],
    flip_round: int,
    flip_action: Any,
    opponent_name: str,
    legal_actions: Sequence[Any],
    payoffs: Payoffs,
) -> float:
    """Compute Eq. 2: replay π_opp while holding later student actions fixed."""
    validate_recorded_trajectory(own_actions, opponent_actions, opponent_name, legal_actions)
    if not 0 <= flip_round < len(own_actions):
        raise IndexError("flip_round is outside the trajectory")
    if flip_action not in legal_actions:
        raise ValueError("flip_action is not legal")
    history: History = tuple(zip(own_actions[:flip_round], opponent_actions[:flip_round]))
    total = recorded_return(own_actions[:flip_round], opponent_actions[:flip_round], payoffs)
    # Simultaneous move: the flip cannot change the opponent action at round t.
    opponent = opponent_actions[flip_round]
    total += stage_reward(payoffs, flip_action, opponent)
    history += ((flip_action, opponent),)
    for own in own_actions[flip_round + 1 :]:
        opponent = opponent_action(opponent_name, history, legal_actions)
        total += stage_reward(payoffs, own, opponent)
        history += ((own, opponent),)
    return total


def optimal_continuation_value(
    start_round: int,
    horizon: int,
    history: History,
    opponent_name: str,
    legal_actions: Sequence[Any],
    payoffs: Payoffs,
) -> float:
    """Compute V(s, h_s) by exhaustive dynamic programming (Eq. 3)."""
    legal = tuple(legal_actions)

    @lru_cache(maxsize=None)
    def value(round_index: int, state: History) -> float:
        if round_index >= horizon:
            return 0.0
        opponent = opponent_action(opponent_name, state, legal)
        return max(
            stage_reward(payoffs, own, opponent)
            + value(round_index + 1, state + ((own, opponent),))
            for own in legal
        )

    return value(start_round, history)


def horizon_aware_return(
    own_actions: Sequence[Any],
    opponent_actions: Sequence[Any],
    flip_round: int,
    flip_action: Any,
    opponent_name: str,
    legal_actions: Sequence[Any],
    payoffs: Payoffs,
) -> float:
    """Compute R^HA_cf: recorded prefix, forced flip, then optimal play."""
    validate_recorded_trajectory(own_actions, opponent_actions, opponent_name, legal_actions)
    if not 0 <= flip_round < len(own_actions):
        raise IndexError("flip_round is outside the trajectory")
    if flip_action not in legal_actions:
        raise ValueError("flip_action is not legal")
    history: History = tuple(zip(own_actions[:flip_round], opponent_actions[:flip_round]))
    prefix = recorded_return(own_actions[:flip_round], opponent_actions[:flip_round], payoffs)
    opponent = opponent_actions[flip_round]
    immediate = stage_reward(payoffs, flip_action, opponent)
    flipped_history = history + ((flip_action, opponent),)
    continuation = optimal_continuation_value(
        flip_round + 1,
        len(own_actions),
        flipped_history,
        opponent_name,
        legal_actions,
        payoffs,
    )
    return prefix + immediate + continuation
