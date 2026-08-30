"""Shared opponent reconstruction and payoff utilities for paper Eq. 2 and Eq. 3."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

History = tuple[tuple[Any, Any], ...]
Payoffs = Mapping[tuple[Any, Any], float]
_DETERMINISTIC_OPPONENTS = {
    "always_cooperate", "always_defect", "tit_for_tat", "grim_trigger",
    "pavlov", "tit_for_two_tats",
}

def is_reconstructible(name: str) -> bool:
    return name in _DETERMINISTIC_OPPONENTS

def opponent_action(name: str, history: History, legal_actions: Sequence[Any]) -> Any:
    if len(legal_actions) != 2:
        raise ValueError("counterfactual horizon requires exactly two legal actions")
    if name not in _DETERMINISTIC_OPPONENTS:
        raise ValueError(f"opponent {name!r} is not an exact deterministic policy")
    cooperate, defect = legal_actions
    if name == "always_cooperate": return cooperate
    if name == "always_defect": return defect
    if name == "tit_for_tat": return cooperate if not history else history[-1][0]
    if name == "grim_trigger": return defect if any(own == defect for own, _ in history) else cooperate
    if name == "pavlov": return cooperate if not history else history[-1][1]
    trailing = 0
    for own, _ in reversed(history):
        if own != defect: break
        trailing += 1
    if trailing >= 2: return defect
    return cooperate if not history else history[-1][0]

def stage_reward(payoffs: Payoffs, own_action: Any, opponent: Any) -> float:
    try: return float(payoffs[(own_action, opponent)])
    except KeyError as exc: raise ValueError(f"missing payoff for {(own_action, opponent)!r}") from exc

def validate_recorded_trajectory(own_actions: Sequence[Any], opponent_actions: Sequence[Any],
                                 opponent_name: str, legal_actions: Sequence[Any]) -> History:
    if not own_actions or len(own_actions) != len(opponent_actions):
        raise ValueError("own_actions and opponent_actions need the same non-zero length")
    if len(own_actions) > 10:
        raise ValueError("paper §2.4 implementation supports horizon T <= 10")
    if len(legal_actions) != 2 or any(action not in legal_actions for action in own_actions):
        raise ValueError("all own actions must belong to exactly two legal_actions")
    history: History = ()
    for index, (own, recorded) in enumerate(zip(own_actions, opponent_actions)):
        expected = opponent_action(opponent_name, history, legal_actions)
        if recorded != expected:
            raise ValueError(f"round {index}: recorded opponent action {recorded!r} does not match {opponent_name!r} policy action {expected!r}")
        history += ((own, recorded),)
    return history

def recorded_return(own_actions: Sequence[Any], opponent_actions: Sequence[Any], payoffs: Payoffs) -> float:
    return sum(stage_reward(payoffs, own, opp) for own, opp in zip(own_actions, opponent_actions))
