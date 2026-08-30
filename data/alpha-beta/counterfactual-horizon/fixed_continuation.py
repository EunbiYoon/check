"""Paper Eq. 2: counterfactual return with later student actions held fixed."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

_UTILS_NAME = "counterfactual_utils"
if _UTILS_NAME in sys.modules:
    _utils = sys.modules[_UTILS_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_UTILS_NAME, Path(__file__).with_name("utils.py"))
    if _spec is None or _spec.loader is None:
        raise RuntimeError("cannot load counterfactual-horizon/utils.py")
    _utils = importlib.util.module_from_spec(_spec)
    sys.modules[_UTILS_NAME] = _utils
    _spec.loader.exec_module(_utils)

History = _utils.History
Payoffs = _utils.Payoffs
opponent_action = _utils.opponent_action
recorded_return = _utils.recorded_return
stage_reward = _utils.stage_reward
validate_recorded_trajectory = _utils.validate_recorded_trajectory

def fixed_continuation_return(own_actions: Sequence[Any], opponent_actions: Sequence[Any],
                              flip_round: int, flip_action: Any, opponent_name: str,
                              legal_actions: Sequence[Any], payoffs: Payoffs) -> float:
    """Replay the opponent after the forced flip while retaining recorded student actions."""
    validate_recorded_trajectory(own_actions, opponent_actions, opponent_name, legal_actions)
    if not 0 <= flip_round < len(own_actions): raise IndexError("flip_round is outside the trajectory")
    if flip_action not in legal_actions: raise ValueError("flip_action is not legal")
    history: History = tuple(zip(own_actions[:flip_round], opponent_actions[:flip_round]))
    total = recorded_return(own_actions[:flip_round], opponent_actions[:flip_round], payoffs)
    opponent = opponent_actions[flip_round]
    total += stage_reward(payoffs, flip_action, opponent)
    history += ((flip_action, opponent),)
    for own in own_actions[flip_round + 1:]:
        opponent = opponent_action(opponent_name, history, legal_actions)
        total += stage_reward(payoffs, own, opponent)
        history += ((own, opponent),)
    return total
