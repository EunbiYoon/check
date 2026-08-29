"""Fixed opponent policies for eval rollouts."""

from __future__ import annotations

import random
from typing import Any


class OpponentPolicy:
    name = "base"

    def reset(self, rng: random.Random, legal_actions: list[Any]) -> None:
        self._rng = rng
        self._legal = list(legal_actions)
        self._history: list[Any] = []

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        raise NotImplementedError

    def observe(self, agent_action: Any, opp_action: Any) -> None:
        self._history.append((agent_action, opp_action))


class AlwaysCooperate(OpponentPolicy):
    name = "always_cooperate"

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        for preferred in ("C", "Stag", "Opera", "H"):
            if preferred in legal_actions:
                return preferred
        return legal_actions[0]


class AlwaysDefect(OpponentPolicy):
    name = "always_defect"

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        for preferred in ("D", "Hare", "Football", "T"):
            if preferred in legal_actions:
                return preferred
        return legal_actions[-1]


class RandomOpponent(OpponentPolicy):
    name = "random"

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        return self._rng.choice(legal_actions)


class EpsilonGreedy(OpponentPolicy):
    name = "epsilon_greedy"

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        if self._rng.random() < 0.1:
            return self._rng.choice(legal_actions)
        return legal_actions[0]


class TitForTat(OpponentPolicy):
    name = "tit_for_tat"

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        if last_agent_action is None:
            for preferred in ("C", "Stag", "Opera", "H"):
                if preferred in legal_actions:
                    return preferred
            return legal_actions[0]
        if last_agent_action in legal_actions:
            return last_agent_action
        return legal_actions[0]


class GrimTrigger(OpponentPolicy):
    name = "grim_trigger"

    def reset(self, rng: random.Random, legal_actions: list[Any]) -> None:
        super().reset(rng, legal_actions)
        self._triggered = False

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        defect_like = {"D", "Hare", "Football", "T"}
        if last_agent_action in defect_like:
            self._triggered = True
        if self._triggered:
            for preferred in ("D", "Hare", "Football", "T"):
                if preferred in legal_actions:
                    return preferred
            return legal_actions[-1]
        for preferred in ("C", "Stag", "Opera", "H"):
            if preferred in legal_actions:
                return preferred
        return legal_actions[0]


class Pavlov(OpponentPolicy):
    name = "pavlov"

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        if not self._history:
            return legal_actions[0]
        _, last_opp = self._history[-1]
        if last_opp in legal_actions:
            return last_opp
        return legal_actions[0]


class TitForTwoTats(OpponentPolicy):
    name = "tit_for_two_tats"

    def reset(self, rng: random.Random, legal_actions: list[Any]) -> None:
        super().reset(rng, legal_actions)
        self._defect_streak = 0

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        if last_agent_action in {"D", "Hare", "Football", "T"}:
            self._defect_streak += 1
        else:
            self._defect_streak = 0
        if self._defect_streak >= 2:
            for preferred in ("D", "Hare", "Football", "T"):
                if preferred in legal_actions:
                    return preferred
            return legal_actions[-1]
        if last_agent_action in legal_actions and last_agent_action is not None:
            return last_agent_action
        return legal_actions[0]


class GenerousTFT(OpponentPolicy):
    name = "generous_tft"

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        if last_agent_action in {"D", "Hare", "Football", "T"} and self._rng.random() < 0.1:
            for preferred in ("C", "Stag", "Opera", "H"):
                if preferred in legal_actions:
                    return preferred
        return TitForTat().act(legal_actions=legal_actions, last_agent_action=last_agent_action)


class TruthfulBidder(OpponentPolicy):
    name = "truthful_bidder"

    def reset(self, rng: random.Random, legal_actions: list[Any]) -> None:
        super().reset(rng, legal_actions)
        self._value = rng.randint(50, 150)

    def act(self, *, legal_actions: list[Any], last_agent_action: Any | None = None) -> Any:
        nums = [a for a in legal_actions if isinstance(a, (int, float))]
        if nums:
            return min(nums, key=lambda x: abs(float(x) - self._value))
        return legal_actions[0]


OPPONENT_CLASSES: dict[str, type[OpponentPolicy]] = {
    cls.name: cls
    for cls in [
        AlwaysCooperate,
        AlwaysDefect,
        RandomOpponent,
        EpsilonGreedy,
        TitForTat,
        GrimTrigger,
        Pavlov,
        TitForTwoTats,
        GenerousTFT,
        TruthfulBidder,
    ]
}


def make_opponent(name: str) -> OpponentPolicy:
    if name not in OPPONENT_CLASSES:
        raise KeyError(f"Unknown opponent: {name}")
    return OPPONENT_CLASSES[name]()
