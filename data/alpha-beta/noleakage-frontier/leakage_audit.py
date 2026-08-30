#!/usr/bin/env python3
"""Paper Appendix C: heuristic current-round leakage flags.

These are deliberately audit candidates for manual review, not automatic
deletion rules: legitimate prior-round prose ("opponent cooperated in round 2")
matches too and is retained after inspection (paper §2.3, Appendix C).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Verbatim from paper Appendix C.
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


@dataclass(frozen=True)
class AuditFlag:
    rule: str
    excerpt: str
    start: int
    end: int


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
