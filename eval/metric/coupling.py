"""Reasoning–action coupling metrics.

Parses the [EV] slot, extracts per-action stated expected values (Patterns A/B/C),
and marks a round coupled when the chosen action equals the argmax stated EV.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# English tokens occasionally captured as fake action labels (Appendix D).
_PROSE_DROP = frozenset(
    {
        "round",
        "rounds",
        "value",
        "values",
        "playing",
        "play",
        "action",
        "actions",
        "total",
        "expected",
        "overall",
        "if",
        "for",
        "the",
        "this",
        "that",
        "with",
        "and",
        "or",
        "is",
        "are",
        "be",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "from",
        "legal",
        "payoff",
        "payoffs",
        "mutual",
        "cooperation",
        "defection",
        "cooperate",
        "defect",
        "number",
        "arithmetic",
        "ending",
        "per",
        "each",
        "both",
        "options",
        "option",
        "have",
        "same",
        "good",
        "based",
        "calculation",
        "assume",
        "assuming",
        "probability",
        "probabilities",
        "opponent",
        "might",
        "could",
        "would",
        "will",
        "when",
        "while",
        "where",
        "however",
        "therefore",
        "thus",
        "hence",
        "so",
        "not",
        "no",
        "yes",
        "step",
        "stage",
        "game",
        "games",
        "history",
        "prior",
        "update",
        "decision",
        "corner",
        "edge",
        "center",
        "cell",
        "cells",
        "index",
        "indices",
        "bid",
        "bids",
        "proposal",
        "proposals",
    }
)

_FINAL_NUM = re.compile(r"=\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*$", re.I)

# Pattern B: EV(C) = 0.30(3.0) + 0.40(0) = 1.2
_PATTERN_B = re.compile(
    r"EV\s*\(\s*([^)]+?)\s*\)\s*=\s*(.+?=\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
    re.I,
)

# Pattern C: Expected payoff for 'C' is ... = 1.8
_PATTERN_C = re.compile(
    r"(?:Expected payoff for|Total EV for|EV for)\s+['\"]?([^'\";\n]+?)['\"]?\s*(?:is|=)\s*(.+?=\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
    re.I,
)

# Pattern A / prose: action label + separator + arithmetic ending in = number
_PATTERN_A = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(?:Action\s+)?"
    r"([A-Za-z0-9\[\],\-\+\'\"][A-Za-z0-9\[\],\-\+\'\" \t]{0,24}?)"
    r"(?:\*\*)?\s*(?:[:→\-]|->)\s*"
    r"(.+?=\s*[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
    re.I | re.M,
)

# "For playing 'C':" / "If I play C:" with trailing Total EV ... = N
_PATTERN_PLAY = re.compile(
    r"(?:For playing|If I play|playing)\s+['\"]?([^'\":\n]+?)['\"]?\s*[:)]",
    re.I,
)
_PATTERN_TOTAL_EV = re.compile(
    r"Total EV for\s+['\"]?([^'\":\n=]+?)['\"]?\s*=\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
    re.I,
)

_EV_BLOCK = re.compile(
    r"\[EV\](.*?)(?=\[Decision\]|\[/Decision\]|</think>|\Z)",
    re.I | re.S,
)


@dataclass
class CouplingResult:
    parsable: bool
    coupled: bool


def _normalize_token(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^play\s+", "", text)
    text = re.sub(r"[^a-z0-9.\-+]", "", text)
    return text


def _clean_action_label(raw: str) -> str | None:
    label = raw.strip().strip("*").strip()
    label = re.sub(r"^\((.+)\)$", r"\1", label)
    label = re.sub(r"^action\s+", "", label, flags=re.I)
    label = label.strip("'\"")
    if not label:
        return None
    token = re.sub(r"[^A-Za-z0-9\[\],\-\+\.]", "", label.split()[0] if label.split() else label)
    if not token:
        return None
    if token.lower() in _PROSE_DROP:
        return None
    if len(token) > 24:
        return None
    return label.strip()


def _extract_final_value(expr: str) -> float | None:
    m = _FINAL_NUM.search(expr.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _needs_operator(expr: str) -> bool:
    """Appendix D: arithmetic expression with at least one operator."""
    body = expr.strip()
    if not re.search(r"[\+\-\*/]", body):
        return False
    return _extract_final_value(body) is not None


def _add_ev(evs: dict[str, float], action: str | None, expr: str) -> None:
    if not action:
        return
    cleaned = _clean_action_label(action)
    if not cleaned:
        return
    if not _needs_operator(expr):
        # Allow simple "3.0 (if ...) + 0.0 (if ...) = 3.0" style totals.
        val = _extract_final_value(expr)
        if val is None:
            return
    else:
        val = _extract_final_value(expr)
        if val is None:
            return
    key = cleaned
    # Keep the highest stated EV if duplicate labels appear.
    if key not in evs or val > evs[key]:
        evs[key] = val


def parse_ev_slot(text: str) -> dict[str, float]:
    """Return action -> stated EV from reasoning text (empty if none parsable)."""
    if not text or not text.strip():
        return {}

    block = text
    m = _EV_BLOCK.search(text)
    if m:
        block = m.group(1)
    elif "[EV]" not in text.upper():
        return {}

    evs: dict[str, float] = {}

    for pat in (_PATTERN_B, _PATTERN_C, _PATTERN_A):
        for m in pat.finditer(block):
            _add_ev(evs, m.group(1), m.group(2))

    # "Total EV for C = 1.5 + 0.0 = 1.5"
    for m in _PATTERN_TOTAL_EV.finditer(block):
        action = m.group(1)
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        cleaned = _clean_action_label(action)
        if cleaned and (cleaned not in evs or val > evs[cleaned]):
            evs[cleaned] = val

    # Pair "If I play C:" sections with nearby "= number" totals on same bullet block.
    lines = block.splitlines()
    current_action: str | None = None
    for line in lines:
        pm = _PATTERN_PLAY.search(line)
        if pm:
            current_action = pm.group(1)
            continue
        if current_action and "=" in line:
            val = _extract_final_value(line)
            if val is not None and re.search(r"[\+\-\*/]", line):
                _add_ev(evs, current_action, line)
                current_action = None

    return evs


def _actions_match(stated: str, chosen: str) -> bool:
    a = _normalize_token(stated)
    b = _normalize_token(chosen)
    if not a or not b:
        return False
    return (
        a == b
        or a.endswith(b)
        or b.endswith(a)
        or a in b
        or b in a
    )


def argmax_stated_ev(evs: dict[str, float]) -> str | None:
    if not evs:
        return None
    best_val = max(evs.values())
    best = [k for k, v in evs.items() if v == best_val]
    return best[0] if len(best) == 1 else None


def coupling_for_round(reasoning: str, action: str) -> CouplingResult:
    evs = parse_ev_slot(reasoning)
    if not evs:
        return CouplingResult(parsable=False, coupled=False)

    best = argmax_stated_ev(evs)
    if best is None:
        return CouplingResult(parsable=True, coupled=False)

    return CouplingResult(parsable=True, coupled=_actions_match(best, str(action)))
