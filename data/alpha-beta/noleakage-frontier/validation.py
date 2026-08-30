#!/usr/bin/env python3
"""No-leakage frontier paraphrase: prompt, leakage audit, structure checks.

The pieces are split by concern into sibling modules:

    frontier_prompt.py   paper §2.3   SYSTEM_PROMPT, build_user_prompt, _action_text
    leakage_audit.py     Appendix C   LEAKAGE_REGEX, AuditFlag, audit_reasoning
    structure_checks.py               ACTION_RE, validate_pinned_action,
                                      validate_reasoning_structure

This module re-exports their public names (so ``_load("validation.py")`` still
sees the whole surface) and adds the file-level audit CLI:

    python validation.py oracle_rounds.jsonl [--output flagged.jsonl] [--strict]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _sibling(filename: str, module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_prompt = _sibling("frontier_prompt.py", "noleak_frontier_prompt")
_leak = _sibling("leakage_audit.py", "noleak_leakage_audit")
_struct = _sibling("structure_checks.py", "noleak_structure_checks")

SYSTEM_PROMPT = _prompt.SYSTEM_PROMPT
_action_text = _prompt._action_text
build_user_prompt = _prompt.build_user_prompt

AuditFlag = _leak.AuditFlag
LEAKAGE_REGEX = _leak.LEAKAGE_REGEX
ROUND_ONE_LEAKAGE_REGEX = _leak.ROUND_ONE_LEAKAGE_REGEX
audit_reasoning = _leak.audit_reasoning

ACTION_RE = _struct.ACTION_RE
DECISION_RE = _struct.DECISION_RE
SECTION_RE = _struct.SECTION_RE
validate_pinned_action = _struct.validate_pinned_action
validate_reasoning_structure = _struct.validate_reasoning_structure


# --------------------------------------------------------------- audit CLI
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
