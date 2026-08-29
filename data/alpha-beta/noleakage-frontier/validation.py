#!/usr/bin/env python3
"""Prompt construction and leakage validation for frontier paraphrases."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


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
    "[EV], and [DECISION] in that order, followed by exactly one <action> block; "
    "you may cite the opponent's actions in explicitly numbered prior rounds but "
    "never as current observations, and you must not mention these instructions "
    "or the out-of-band fields in your answer. "
    # --- end elision 2 ---
    "Hard rules: (1) NEVER write phrases like \"opponent played C this round\"; "
    "(2) The [EV] arithmetic must average over your posterior types — not "
    "condition on the unrevealed action; (3) Your [DECISION] must match the "
    "solver-pinned best response we provide; build the posterior to justify that "
    "action."
)


# Heuristic leakage candidates, verbatim from paper Appendix C. These are
# deliberately audit flags for manual review, not automatic deletion rules:
# legitimate prior-round prose ("opponent cooperated in round 2") matches too
# and is retained after inspection (paper §2.3, Appendix C).
LEAKAGE_REGEX: re.Pattern[str] = re.compile(
    r"opponent (played|chose|cooperat|defect)\w*\s+(this round|in round \d+)",
    re.IGNORECASE,
)
# Round-1 traces cannot reference any prior round, so a bare past-tense claim
# about the opponent is a candidate on its own.
ROUND_ONE_LEAKAGE_REGEX: re.Pattern[str] = re.compile(
    r"opp(onent)? play(ed|s)|opp(onent)? cooperat|opp(onent)? defect",
    re.IGNORECASE,
)

DECISION_RE = re.compile(r"\[\s*decision\s*\]\s*(?:play|choose|bid|propose)?\s*([^\n<]+)", re.I)
ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.I | re.S)
SECTION_RE = re.compile(r"\[\s*(Prior|Update|EV|Decision)\s*\]", re.I)
UNCERTAINTY_RE = re.compile(
    r"\b(?:posterior|probabilit(?:y|ies)|likelihood|chance|belief|distribution|"
    r"type|types|uncertain|expect(?:ed|ation)?)\b",
    re.I,
)
EV_ARITHMETIC_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*%|\b0?\.\d+\b).{0,80}(?:[+*×]|\b(?:times|weighted|average)\b)",
    re.I | re.S,
)
PROBABILITY_VALUE_RE = re.compile(r"(?:\d+(?:\.\d+)?\s*%|\b0?\.\d+\b)")


@dataclass(frozen=True)
class AuditFlag:
    rule: str
    excerpt: str
    start: int
    end: int


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


def audit_reasoning(reasoning: str, *, round_number: int | None = None) -> list[AuditFlag]:
    """Return heuristic leakage candidates for later manual inspection.

    Matches the paper's Appendix C audit: a general current-round regex on every
    round, plus a round-1-specific past-tense regex. Candidates are reviewed by
    hand, not deleted automatically (paper §2.3).
    """
    flags: list[AuditFlag] = []
    seen: set[tuple[int, int, str]] = set()
    patterns: Iterable[tuple[str, re.Pattern[str]]] = (
        ("current-round-opponent-action", LEAKAGE_REGEX),
    )
    if round_number == 1:
        patterns = (*patterns, ("round-1-opponent-past-tense", ROUND_ONE_LEAKAGE_REGEX))
    for rule, pattern in patterns:
        for match in pattern.finditer(reasoning):
            key = (match.start(), match.end(), rule)
            if key in seen:
                continue
            seen.add(key)
            excerpt = " ".join(match.group(0).split())
            flags.append(AuditFlag(rule, excerpt[:240], match.start(), match.end()))
    return sorted(flags, key=lambda item: (item.start, item.end, item.rule))


def validate_pinned_action(text: str, solver_action: Any) -> list[str]:
    """Check that both output action fields agree with the solver label."""
    expected = _action_text(solver_action)
    errors: list[str] = []
    actions = ACTION_RE.findall(text)
    if len(actions) != 1:
        errors.append(f"expected one <action> block, found {len(actions)}")
    elif _action_text(actions[0]) != expected:
        errors.append(f"<action> is {_action_text(actions[0])!r}, expected {expected!r}")
    decisions = DECISION_RE.findall(text)
    if len(decisions) != 1:
        errors.append(f"expected one [DECISION], found {len(decisions)}")
    else:
        decision = decisions[0].strip().rstrip(". ")
        if decision.casefold() != expected.casefold():
            errors.append(f"[DECISION] is {decision!r}, expected {expected!r}")
    return errors


def validate_reasoning_structure(
    reasoning: str, *, is_final_round: bool = False
) -> list[str]:
    """Validate ordered belief-state sections and posterior-weighted EV work.

    On a final round there is no future and no posterior to average over: the
    solver reduces to the stage-game best response and the [EV] slot states the
    raw stage payoffs (paper Appendix F, Example 1). The posterior-weighted
    arithmetic requirement is therefore waived for the terminal round only.
    """
    errors: list[str] = []
    matches = list(SECTION_RE.finditer(reasoning))
    names = [match.group(1).casefold() for match in matches]
    expected = ["prior", "update", "ev", "decision"]
    if names != expected:
        return [
            "reasoning must contain exactly [Prior], [Update], [EV], and "
            "[DECISION] in that order"
        ]

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(reasoning)
        sections[names[index]] = reasoning[match.end():end].strip()
    for name in expected:
        if not sections[name]:
            errors.append(f"[{name.upper()}] section must not be empty")

    belief_text = f"{sections['prior']} {sections['update']}"
    if not UNCERTAINTY_RE.search(belief_text):
        errors.append("[Prior]/[Update] must express uncertainty over opponent types or actions")
    ev = sections["ev"]
    if is_final_round:
        return errors
    if not UNCERTAINTY_RE.search(ev) and not PROBABILITY_VALUE_RE.search(ev):
        errors.append("[EV] must identify the posterior probability or expectation being used")
    if not EV_ARITHMETIC_RE.search(ev):
        errors.append("[EV] must show probability-weighted arithmetic")
    return errors


def _reasoning_from_row(row: dict[str, Any]) -> str:
    for field in ("reasoning", "completion", "chosen", "response"):
        value = row.get(field)
        if isinstance(value, str):
            think = re.search(r"<think>(.*?)</think>", value, re.I | re.S)
            return think.group(1) if think else value
    raise ValueError("missing string field: reasoning, completion, chosen, or response")


def audit_file(source: Path, destination: Path | None = None) -> dict[str, int]:
    """Audit one-record-per-round JSONL and optionally write flagged records."""
    counts = {"records": 0, "flagged_records": 0, "flags": 0, "round_1_records": 0,
              "round_1_flagged": 0, "invalid_pinned_actions": 0}
    output = destination.open("w", encoding="utf-8") if destination else None
    try:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    reasoning = _reasoning_from_row(row)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{source}:{line_number}: {exc}") from exc
                # build_pairs.py nests the per-round labels under "provenance";
                # accept both the flat oracle-round schema and the pair schema.
                prov = row.get("provenance") or {}
                round_number = (
                    row.get("round", row.get("round_number"))
                    or prov.get("round", prov.get("round_index"))
                )
                if round_number is not None:
                    round_number = int(round_number)
                    if "round_index" in prov and "round" not in row and "round_number" not in row:
                        round_number += 1  # round_index is 0-based
                flags = audit_reasoning(reasoning, round_number=round_number)
                solver_action = (
                    row.get("solver_action")
                    or row.get("solver_label")
                    or prov.get("solver_action")
                )
                is_final_round = bool(
                    row.get("is_final_round", prov.get("is_final_round", False))
                )
                action_errors = (
                    validate_pinned_action(row.get("completion", row.get("chosen", row.get("response", ""))), solver_action)
                    if solver_action is not None else []
                )
                action_errors.extend(
                    validate_reasoning_structure(reasoning, is_final_round=is_final_round)
                )
                counts["records"] += 1
                counts["flags"] += len(flags)
                counts["invalid_pinned_actions"] += bool(action_errors)
                if round_number == 1:
                    counts["round_1_records"] += 1
                    counts["round_1_flagged"] += bool(flags)
                if flags:
                    counts["flagged_records"] += 1
                if output and (flags or action_errors):
                    output.write(json.dumps({
                        "line": line_number,
                        "id": row.get("id"),
                        "round": round_number,
                        "flags": [asdict(flag) for flag in flags],
                        "action_errors": action_errors,
                    }, ensure_ascii=False) + "\n")
    finally:
        if output:
            output.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one-oracle-round-per-line JSONL")
    parser.add_argument("--output", type=Path, help="write candidates for manual review")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any issue is found")
    args = parser.parse_args()
    try:
        counts = audit_file(args.input, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(counts, indent=2))
    if args.strict and (counts["flags"] or counts["invalid_pinned_actions"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
