"""Paper Eq. 3: counterfactual return with an optimized remaining horizon."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from fixed_continuation import (
    History, Payoffs, opponent_action, recorded_return, stage_reward,
    validate_recorded_trajectory,
)

def optimal_continuation_value(start_round: int, horizon: int, history: History,
                               opponent_name: str, legal_actions: Sequence[Any], payoffs: Payoffs) -> float:
    """Compute the optimal continuation value by exhaustive dynamic programming."""
    legal = tuple(legal_actions)
    @lru_cache(maxsize=None)
    def value(round_index: int, state: History) -> float:
        if round_index >= horizon: return 0.0
        opponent = opponent_action(opponent_name, state, legal)
        return max(stage_reward(payoffs, own, opponent) + value(round_index + 1, state + ((own, opponent),)) for own in legal)
    return value(start_round, history)

def horizon_aware_return(own_actions: Sequence[Any], opponent_actions: Sequence[Any],
                         flip_round: int, flip_action: Any, opponent_name: str,
                         legal_actions: Sequence[Any], payoffs: Payoffs) -> float:
    """Use the recorded prefix, forced flip, and optimal continuation from Eq. 3."""
    validate_recorded_trajectory(own_actions, opponent_actions, opponent_name, legal_actions)
    if not 0 <= flip_round < len(own_actions): raise IndexError("flip_round is outside the trajectory")
    if flip_action not in legal_actions: raise ValueError("flip_action is not legal")
    history: History = tuple(zip(own_actions[:flip_round], opponent_actions[:flip_round]))
    prefix = recorded_return(own_actions[:flip_round], opponent_actions[:flip_round], payoffs)
    opponent = opponent_actions[flip_round]
    immediate = stage_reward(payoffs, flip_action, opponent)
    flipped = history + ((flip_action, opponent),)
    return prefix + immediate + optimal_continuation_value(flip_round + 1, len(own_actions), flipped, opponent_name, legal_actions, payoffs)
