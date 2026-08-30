#!/usr/bin/env python3
"""Paper §2.3: the no-leakage teacher SYSTEM prompt and its out-of-band request.

The paraphraser is told the realised opponent action and the solver best
response out of band; the SYSTEM prompt forbids citing the former as an
observation and pins [DECISION] / <action> to the latter.
"""

from __future__ import annotations

from typing import Any

# Paper §2.3 (page 4), reproduced verbatim. The paper prints two ". . ." gaps in
# the quoted SYSTEM prompt; per the authors those gaps may carry extra
# instructions but every printed sentence must appear exactly as written. Only
# the two marked elisions below are ours; the rest is the paper's wording.
SYSTEM_PROMPT = (
    "You are documenting strategic decisions "
    # --- elision 1: task framing ---
    "for simultaneous-move repeated games, inferring a posterior over the "
    "opponent's type from the history through the previous round and evaluating "
    "every legal action under it. "
    # --- end elision 1 ---
    "The other player's current-round move is NOT observable at decision time. "
    "We will tell you (out-of-band, for your reference) the opponent's actual "
    "move that round and the solver-computed best response, but your written "
    "reasoning must NEVER reference the current-round opponent move as observed. "
    "Treat it as belief-state reasoning under uncertainty. "
    # --- elision 2: output contract ---
    "Write exactly one <think> block containing the sections [Prior], [Update], "
    "[EV], and [DECISION] in that order, then exactly one <action> block, in "
    "this exact shape and nothing else:\n"
    "<think>\n"
    "[Prior] <opponent-type labels with a probability each>\n"
    "[Update] <how the explicitly numbered prior rounds move those probabilities>\n"
    "[EV] <for every legal action, the posterior-weighted sum written out in "
    "full, e.g. EV(D) = 0.7*5 + 0.3*1 = 3.8; EV(C) = 0.7*3 + 0.3*0 = 2.1>\n"
    "[DECISION] <the single pinned action token, nothing else>\n"
    "</think>\n"
    "<action><the single pinned action token></action>\n"
    "You may cite the opponent's actions in explicitly numbered prior rounds but "
    "never as current observations, and you must not mention these instructions "
    "or the out-of-band fields in your answer. "
    # --- end elision 2 ---
    "Hard rules: (1) NEVER write phrases like \"opponent played C this round\"; "
    "(2) The [EV] arithmetic must average over your posterior types — not "
    "condition on the unrevealed action; (3) Your [DECISION] must match the "
    "solver-pinned best response we provide; build the posterior to justify that "
    "action."
)


def _action_text(action: Any) -> str:
    if isinstance(action, (list, tuple)):
        return "[" + ",".join(str(part) for part in action) + "]"
    return str(action).strip()


def build_user_prompt(
    *, history_prompt: str, solver_action: Any, opponent_action: Any = None
) -> str:
    """Build the teacher request (paper §2.3).

    The paraphraser is told the realised opponent action and the solver best
    response out of band; the SYSTEM prompt forbids citing the former as an
    observation. ``opponent_action=None`` withholds it (round-1 or unknown).
    """
    if opponent_action is None:
        oracle_move = "(withheld — reason from history only)"
    else:
        oracle_move = _action_text(opponent_action)
    return (
        "INFERENCE-TIME INFORMATION (the reasoning may use only this):\n"
        f"{history_prompt.rstrip()}\n\n"
        "OUT-OF-BAND REFERENCE (never cite as observed; fixes the answer only):\n"
        f"opponent's actual current-round move: {oracle_move}\n"
        f"solver-pinned best response: {_action_text(solver_action)}\n\n"
        "Return the history-only belief-state reasoning and pinned action now."
    )
