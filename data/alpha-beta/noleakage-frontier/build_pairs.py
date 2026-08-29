#!/usr/bin/env python3
"""Build round-level DPO pairs and assemble the resulting DPO trainer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[3]          # repo root
_HERE = Path(__file__).resolve().parents[1]          # train/ab-data-construction/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PAIR_PIPELINE = "solver-counterfactual-frontier-v1"
COMPLETION_RE = re.compile(
    r"<think>.*?</think>\s*<action>.*?</action>", re.IGNORECASE | re.DOTALL
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


solver = _load(_HERE / "solver-pinned" / "best_action.py", "pair_solver")
counterfactual = _load(
    _HERE / "counterfactual-horizon" / "counterfactual_horizon.py",
    "pair_counterfactual",
)
no_leak = _load(Path(__file__).with_name("validation.py"), "pair_no_leak")


_dpolora_cache: dict[str, Any] = {}


def _dpolora(module_name: str):
    """Load a helper from ``train/dpo-lora`` (a hyphenated, non-importable dir).

    The teacher paraphraser and the DPO-trainer assembly reuse the training
    code's model loader / checkpoint callback; that package lives in a
    hyphenated directory, so it is loaded by file path like the other siblings.
    """
    if module_name not in _dpolora_cache:
        path = None
        for folder in ("dpo_lora", "dpo-lora"):
            candidate = ROOT / "train" / folder / f"{module_name}.py"
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            path = next(ROOT.glob(f"**/dpo?lora/{module_name}.py"), None)
        if path is None:
            raise FileNotFoundError(f"cannot locate train/dpo_lora/{module_name}.py")
        _dpolora_cache[module_name] = _load(path, f"dpolora_{module_name}")
    return _dpolora_cache[module_name]


def _round_completions(row: dict[str, Any]) -> list[str]:
    supplied = row.get("round_completions")
    if isinstance(supplied, list) and all(isinstance(item, str) for item in supplied):
        return supplied
    response = row.get("response", row.get("rejected"))
    if not isinstance(response, str):
        raise ValueError("trajectory requires response or round_completions")
    return [match.group(0) for match in COMPLETION_RE.finditer(response)]


def _round_prompts(row: dict[str, Any], count: int) -> list[str]:
    prompts = row.get("round_prompts")
    if isinstance(prompts, list) and len(prompts) == count and all(
        isinstance(item, str) and item.strip() for item in prompts
    ):
        return prompts
    if count == 1 and isinstance(row.get("prompt"), str):
        return [row["prompt"]]
    raise ValueError(
        "multi-round trajectories require one round_prompts entry per round; "
        "each prompt must contain only information observable before that round"
    )


def _existing_paraphraser(row: dict[str, Any]) -> Callable[..., str]:
    values = row.get("frontier_completions")
    if not isinstance(values, list):
        raise ValueError("provider=existing requires frontier_completions")

    def paraphrase(*, round_index: int, **_: Any) -> str:
        try:
            value = values[round_index]
        except IndexError as exc:
            raise ValueError(f"missing frontier completion for round {round_index}") from exc
        if not isinstance(value, str):
            raise ValueError(f"frontier completion for round {round_index} is not text")
        return value

    return paraphrase


def teacher_paraphraser(
    STUDENT_MODEL: str, *, max_new_tokens: int = 1024
) -> Callable[..., str]:
    """Load the local 7B teacher once and return a deterministic generator."""
    import torch

    import config
    load_base_model = _dpolora("utils").load_base_model

    model, tokenizer = load_base_model(STUDENT_MODEL=STUDENT_MODEL, use_4bit=config.USE_4BIT)
    model.eval()

    def paraphrase(
        *, history_prompt: str, solver_action: Any,
        opponent_action: Any = None, **_: Any
    ) -> str:
        messages = [
            {"role": "system", "content": no_leak.SYSTEM_PROMPT},
            {"role": "user", "content": no_leak.build_user_prompt(
                history_prompt=history_prompt,
                solver_action=solver_action,
                opponent_action=opponent_action,
            )},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    return paraphrase


def _accepted_rounds(
    row: dict[str, Any], *, counterfactual_mode: str = "horizon-aware"
) -> list[dict[str, Any]]:
    if counterfactual_mode not in {"horizon-aware", "fixed"}:
        raise ValueError("counterfactual_mode must be 'horizon-aware' or 'fixed'")
    own_actions = row.get("own_actions", row.get("agent_actions"))
    opponent_actions = row.get("opponent_actions")
    if not isinstance(own_actions, list) or not isinstance(opponent_actions, list):
        raise ValueError("trajectory requires own_actions and opponent_actions")
    if not own_actions or len(own_actions) != len(opponent_actions):
        raise ValueError("own_actions and opponent_actions must have equal non-zero length")
    family = str(row.get("game_family", "")).replace("_", "-").lower()
    repeated = len(own_actions) > 1
    baseline: float | None = None
    payoffs = None
    if repeated and not counterfactual.is_reconstructible(row.get("opponent", "")):
        # §2.4: the horizon filter needs a reconstructible opponent. Without it
        # we cannot certify repeated-game flips, so emit nothing for this row.
        return []

    if family in {"matrix", "matrix-game"}:
        payoffs = solver.decode_matrix_payoffs(row["payoffs"])
        baseline = counterfactual.recorded_return(own_actions, opponent_actions, payoffs)
        if repeated:
            counterfactual.validate_recorded_trajectory(
                own_actions,
                opponent_actions,
                row["opponent"],
                row["legal_actions"],
            )

    accepted: list[dict[str, Any]] = []
    for index, (recorded, opponent) in enumerate(zip(own_actions, opponent_actions)):
        selected = solver.main(row, opponent)
        if selected == recorded:
            continue
        candidate_return: float | None = None
        if repeated:
            if payoffs is None:
                raise ValueError("multi-round horizon filtering supports matrix games only")
            return_fn = (
                counterfactual.horizon_aware_return
                if counterfactual_mode == "horizon-aware"
                else counterfactual.fixed_continuation_return
            )
            candidate_return = return_fn(
                own_actions,
                opponent_actions,
                index,
                selected,
                row["opponent"],
                row["legal_actions"],
                payoffs,
            )
            if candidate_return <= baseline:
                continue
        else:
            # One-shot: Algorithm 1 line 7 — keep the flip only if it strictly
            # improves the stage payoff u0(a*, a1) over the recorded action.
            recorded_reward = solver.stage_utility(row, recorded, opponent)
            candidate_return = solver.stage_utility(row, selected, opponent)
            baseline = recorded_reward
            if candidate_return <= recorded_reward:
                continue
        accepted.append({
            "round_index": index,
            "recorded_action": recorded,
            "opponent_action": opponent,
            "solver_action": selected,
            "recorded_return": baseline,
            "counterfactual_return": candidate_return,
        })
    return accepted


def _collect_candidates(
    row: dict[str, Any], *, counterfactual_mode: str = "horizon-aware"
) -> list[dict[str, Any]]:
    """Hydrate every improving solver flip in one trajectory (no paraphrase yet).

    Separating collection from paraphrasing lets ``build_file`` de-duplicate the
    decision contexts *before* paying for teacher inference — replayed game
    positions (e.g. from extra rollout seeds) collapse to a single pair.
    """
    trajectory_id = row.get("id")
    if trajectory_id is None:
        raise ValueError("trajectory requires a stable id for leakage-free splitting")
    completions = _round_completions(row)
    own_actions = row.get("own_actions", row.get("agent_actions"))
    if len(completions) != len(own_actions):
        raise ValueError("one blind completion is required per recorded round")
    prompts = _round_prompts(row, len(completions))
    round_count = len(completions)
    game = row.get("game")
    game_family = str(row.get("game_family", "")).replace("_", "-").lower()
    existing = row.get("frontier_completions")
    out: list[dict[str, Any]] = []
    for candidate in _accepted_rounds(row, counterfactual_mode=counterfactual_mode):
        index = candidate["round_index"]
        out.append({
            "round_index": index,
            "prompt": prompts[index],
            "rejected": completions[index],
            "opponent_action": candidate["opponent_action"],
            "solver_action": candidate["solver_action"],
            "recorded_action": candidate["recorded_action"],
            "is_final_round": index + 1 == round_count,
            # paper §2.3 frontier paraphrase is for simultaneous-move REPEATED
            # games; one-shot rounds take the §3.1 solver-labelled path instead.
            "one_shot": round_count == 1,
            "filter": counterfactual_mode if round_count > 1 else "one-shot-improvement",
            "trajectory_id": str(trajectory_id),
            "game": game,
            "game_family": game_family,
            "candidate": candidate,
            "frontier_completion": (
                existing[index]
                if isinstance(existing, list) and index < len(existing)
                else None
            ),
        })
    return out


_ACTION_TAG_RE = re.compile(r"(<action>)\s*.*?\s*(</action>)", re.I | re.S)


def _solver_labelled_chosen(blind_completion: str, solver_action: Any) -> str | None:
    """Paper §3.1: one-shot / large-action games (auction, negotiation, and the
    other one-round games) get SOLVER-LABELLED pairs — the blind reasoning is
    kept verbatim and only the ``<action>`` payload is repinned to the solver
    best response. The §2.3 posterior-EV frontier paraphrase is reserved for
    simultaneous-move repeated games, whose SYSTEM prompt it is written for.
    """
    label = no_leak._action_text(solver_action)
    if len(_ACTION_TAG_RE.findall(blind_completion)) != 1:
        return None
    return _ACTION_TAG_RE.sub(
        lambda m: f"{m.group(1)}{label}{m.group(2)}", blind_completion, count=1
    )


def _pair_from_candidate(
    cand: dict[str, Any],
    paraphrase: Callable[..., str] | None,
    *,
    on_invalid: str = "raise",
    skipped: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Build + structurally validate one collected candidate.

    Repeated-game rounds are paraphrased by ``paraphrase`` (the §2.3 no-leakage
    teacher) and checked for the posterior-EV structure. One-shot rounds skip
    the teacher entirely and take the §3.1 solver-labelled path.

    ``on_invalid="raise"`` (default) aborts on the first failure; ``"skip"``
    returns ``None`` and appends ``{"round", "errors"}`` to ``skipped`` so a
    batch ceiling can be enforced.
    """
    index = cand["round_index"]
    is_final_round = cand["is_final_round"]

    if cand.get("one_shot"):
        chosen = _solver_labelled_chosen(cand["rejected"], cand["solver_action"])
        want = no_leak._action_text(cand["solver_action"])
        if chosen is None:
            errors = ["one-shot blind completion has no single <action> tag to repin"]
        else:
            acts = no_leak.ACTION_RE.findall(chosen)
            if len(acts) != 1 or no_leak._action_text(acts[0]) != want:
                errors = [f"solver-labelled <action> mismatch (expected {want!r})"]
            else:
                errors = []
        flags: list[Any] = []
        chosen_reasoning = "blind-verbatim"
    else:
        if paraphrase is None:
            raise ValueError("a paraphraser is required for repeated-game rounds")
        chosen = paraphrase(
            round_index=index,
            history_prompt=cand["prompt"],
            opponent_action=cand["opponent_action"],
            solver_action=cand["solver_action"],
        ).strip()
        errors = no_leak.validate_pinned_action(chosen, cand["solver_action"])
        reasoning_match = re.search(r"<think>(.*?)</think>", chosen, re.I | re.S)
        if reasoning_match is None:
            errors.append("missing <think> block")
            flags = []
        else:
            reasoning = reasoning_match.group(1)
            errors.extend(
                no_leak.validate_reasoning_structure(reasoning, is_final_round=is_final_round)
            )
            flags = no_leak.audit_reasoning(reasoning, round_number=index + 1)
        chosen_reasoning = "frontier-no-leak"

    if errors:
        if on_invalid == "skip":
            if skipped is not None:
                skipped.append({"round": index + 1, "errors": errors})
            return None
        raise ValueError(
            f"round {index}: invalid frontier completion: {', '.join(errors)}"
        )
    # Leakage flags are heuristic candidates for manual review, not deletion
    # rules (paper §2.3): record them on the pair, keep the pair.
    leakage_audit = [{"rule": flag.rule, "excerpt": flag.excerpt} for flag in flags]
    return {
        "prompt": cand["prompt"],
        "chosen": chosen,
        "rejected": cand["rejected"],
        "trajectory_id": cand["trajectory_id"],
        "round": index + 1,
        "provenance": {
            "pipeline": PAIR_PIPELINE,
            "chosen_reasoning": chosen_reasoning,
            "chosen_action": "solver-best-response",
            "rejected": "blind-round-completion",
            "filter": cand["filter"],
            "leakage_audit": leakage_audit,
            "is_final_round": is_final_round,
            "game": cand["game"],
            "game_family": cand.get("game_family"),
            **cand["candidate"],
        },
    }


_DEDUP_MODES = ("none", "full", "context", "position")


def _pair_dedup_key(cand: dict[str, Any], mode: str) -> Any:
    if mode == "none":
        return id(cand)                       # never merges
    if mode == "full":
        return (cand["prompt"], cand["rejected"])
    if mode == "context":
        # one pair per (decision context, pinned action); the blind completion
        # text varies episode-to-episode but the supervision signal does not
        return (cand["prompt"], str(cand["solver_action"]))
    if mode == "position":
        return (str(cand["game"]), cand["round_index"],
                str(cand["recorded_action"]), str(cand["solver_action"]))
    raise ValueError(f"dedup must be one of {', '.join(_DEDUP_MODES)}")


def _flip_margin(cand: dict[str, Any]) -> float:
    detail = cand.get("candidate", cand)
    try:
        return float(detail["counterfactual_return"]) - float(detail["recorded_return"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _pick_representative(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep the clearest improving example: largest reward margin, then the
    longer (more explicit) blind completion."""
    return max(group, key=lambda cand: (_flip_margin(cand), len(cand["rejected"])))


def build_pairs(
    row: dict[str, Any],
    paraphrase: Callable[..., str] | None = None,
    *,
    counterfactual_mode: str = "horizon-aware",
    on_invalid: str = "raise",
    skipped: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Construct the round-level DPO pairs for one blind-rollout trajectory."""
    if on_invalid not in {"raise", "skip"}:
        raise ValueError("on_invalid must be 'raise' or 'skip'")
    candidates = _collect_candidates(row, counterfactual_mode=counterfactual_mode)
    if paraphrase is None and any(not c.get("one_shot") for c in candidates):
        paraphrase = _existing_paraphraser(row)
    pairs: list[dict[str, Any]] = []
    for cand in candidates:
        pair = _pair_from_candidate(cand, paraphrase, on_invalid=on_invalid, skipped=skipped)
        if pair is not None:
            pairs.append(pair)
    return pairs


def build_file(
    source: Path,
    destination: Path,
    provider: str = "existing",
    teacher_STUDENT_MODEL: str | None = None,
    teacher_max_new_tokens: int = 1024,
    counterfactual_mode: str = "horizon-aware",
    require_frontier_model_match: bool = True,
    max_invalid_fraction: float = 0.2,
    dedup: str = "context",
    max_pairs: int | None = None,
    allow_empty: bool = False,
) -> int:
    """Build the DPO pair file for a whole trajectory pool, in two passes.

    Pass 1 collects every improving solver flip and de-duplicates the decision
    contexts (``dedup``, default ``context`` = one pair per prompt + pinned
    action), then optionally caps the survivors at ``max_pairs`` (deterministic
    seeded sample). Pass 2 paraphrases only the retained contexts, so teacher
    inference scales with the deduped count, not the raw round count.

    Note: when the caller has already line-sharded ``source`` (algorithm1.sh's
    3×3 mode), pass ``dedup="none"`` here and de-duplicate the merged file with
    :func:`dedup_pairs_file` instead — a shard cannot see the whole pool.
    """
    import config

    if dedup not in _DEDUP_MODES:
        raise ValueError(f"dedup must be one of {', '.join(_DEDUP_MODES)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_model = teacher_STUDENT_MODEL or config.TEACHER_MODEL

    # -------- pass 1: collect improving flips + dedup (no paraphrase) --------
    candidates: list[dict[str, Any]] = []
    raw_flips = 0
    with source.open(encoding="utf-8") as src:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if provider == "teacher" and require_frontier_model_match:
                    recorded_model = row.get("frontier_model")
                    if recorded_model is None:
                        raise ValueError(
                            "paper Algorithm 1 requires frontier_model provenance; "
                            "use --allow-unverified-frontier-model only for legacy data"
                        )
                    if recorded_model != expected_model:
                        raise ValueError(
                            f"blind rollout model {recorded_model!r} does not match "
                            f"paraphraser model {expected_model!r}"
                        )
                collected = _collect_candidates(row, counterfactual_mode=counterfactual_mode)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            raw_flips += len(collected)
            candidates.extend(collected)

    if not candidates and allow_empty:
        destination.write_text("", encoding="utf-8")
        print(
            f"[{destination.name}] 0 improving flips; wrote an empty shard",
            file=sys.stderr,
            flush=True,
        )
        return 0
    if not candidates:
        raise ValueError("no improving solver flips were found in the trajectory pool")

    if dedup != "none":
        merged: dict[Any, list[dict[str, Any]]] = {}
        for cand in candidates:
            merged.setdefault(_pair_dedup_key(cand, dedup), []).append(cand)
        candidates = [_pick_representative(group) for group in merged.values()]

    _sort_key = lambda c: (c["prompt"], c["round_index"], str(c["solver_action"]))
    candidates.sort(key=_sort_key)
    deduped = len(candidates)
    if max_pairs is not None and deduped > max_pairs:
        rng = random.Random(0)
        rng.shuffle(candidates)
        candidates = candidates[:max_pairs]
        candidates.sort(key=_sort_key)

    print(
        f"[{destination.name}] {raw_flips} improving flips -> {deduped} unique "
        f"{dedup} contexts"
        + (f" -> capped to {len(candidates)}" if deduped != len(candidates) else "")
        + f"; paraphrasing {len(candidates)}",
        file=sys.stderr,
        flush=True,
    )

    # -------- pass 2: build the survivors --------
    # One-shot rounds are solver-labelled (§3.1) and never hit the teacher, so
    # skip the (expensive) model load when nothing repeated survived.
    needs_teacher = provider == "teacher" and any(not c.get("one_shot") for c in candidates)
    shared = (
        teacher_paraphraser(expected_model, max_new_tokens=teacher_max_new_tokens)
        if needs_teacher
        else None
    )
    count = 0
    skipped: list[dict[str, Any]] = []
    started = time.monotonic()
    with destination.open("w", encoding="utf-8") as dst:
        for position, cand in enumerate(candidates, 1):
            if cand.get("one_shot"):
                paraphrase: Callable[..., str] | None = None
            elif shared is not None:
                paraphrase = shared
            else:
                fc = cand.get("frontier_completion")
                if not isinstance(fc, str):
                    raise ValueError(
                        "provider=existing requires frontier_completions for every retained round"
                    )
                paraphrase = lambda _fc=fc, **_: _fc
            pair = _pair_from_candidate(cand, paraphrase, on_invalid="skip", skipped=skipped)
            if pair is not None:
                dst.write(json.dumps(pair, ensure_ascii=False) + "\n")
                dst.flush()
                count += 1
            if position % 25 == 0 or position == len(candidates):
                print(
                    f"[{destination.name}] {position}/{len(candidates)} contexts — "
                    f"{count} pairs, {len(skipped)} dropped, "
                    f"{time.monotonic() - started:.0f}s elapsed",
                    file=sys.stderr,
                    flush=True,
                )
    total = count + len(skipped)
    if count == 0:
        raise ValueError(
            "no improving solver flips were emitted"
            + (f" ({len(skipped)} rounds dropped for malformed teacher output)" if skipped else "")
        )
    if skipped:
        reasons: dict[str, int] = {}
        for item in skipped:
            for err in item["errors"]:
                reasons[err] = reasons.get(err, 0) + 1
        summary = "; ".join(f"{n}x {msg}" for msg, n in sorted(reasons.items(), key=lambda kv: -kv[1]))
        fraction = len(skipped) / total
        if fraction > max_invalid_fraction:
            raise ValueError(
                f"{len(skipped)}/{total} teacher paraphrases failed structural "
                f"validation ({fraction:.0%} > {max_invalid_fraction:.0%} ceiling) — {summary}"
            )
        print(
            f"warning: dropped {len(skipped)}/{total} rounds with malformed teacher "
            f"output ({fraction:.1%}) — {summary}",
            file=sys.stderr,
        )
    return count


def dedup_pairs_file(
    source: Path,
    destination: Path,
    *,
    dedup: str = "context",
    max_pairs: int | None = None,
) -> int:
    """De-duplicate an already-built pair JSONL in place-safe fashion.

    Used by algorithm1.sh's ``merge`` step: the pool was line-sharded across
    workers, so ``build_file`` could only dedup within a shard. This runs on the
    concatenated file with the whole pool visible. ``source`` may equal
    ``destination`` (the file is read fully before the write opens).
    """
    if dedup not in _DEDUP_MODES:
        raise ValueError(f"dedup must be one of {', '.join(_DEDUP_MODES)}")

    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"no pairs in {source}")

    def key(pair: dict[str, Any]) -> Any:
        prov = pair.get("provenance", {})
        if dedup == "none":
            return id(pair)
        if dedup == "full":
            return (pair.get("prompt"), pair.get("rejected"))
        if dedup == "context":
            return (pair.get("prompt"), str(prov.get("solver_action")))
        return (str(prov.get("game")), pair.get("round"),
                str(prov.get("recorded_action")), str(prov.get("solver_action")))

    def margin(pair: dict[str, Any]) -> float:
        prov = pair.get("provenance", {})
        try:
            return float(prov["counterfactual_return"]) - float(prov["recorded_return"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    groups: dict[Any, list[dict[str, Any]]] = {}
    for pair in rows:
        groups.setdefault(key(pair), []).append(pair)
    kept = [
        max(group, key=lambda p: (margin(p), len(p.get("rejected", ""))))
        for group in groups.values()
    ]
    kept.sort(key=lambda p: (p.get("prompt", ""), p.get("round", 0)))
    if max_pairs is not None and len(kept) > max_pairs:
        rng = random.Random(0)
        rng.shuffle(kept)
        kept = kept[:max_pairs]
        kept.sort(key=lambda p: (p.get("prompt", ""), p.get("round", 0)))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for pair in kept:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"dedup({dedup}): {len(rows)} -> {len(kept)} pairs", file=sys.stderr)
    return len(kept)


RW_UPSAMPLE_FAMILIES = ("auction", "bargaining", "negotiation", "divide-dollar")


def upsample_pairs_file(
    source: Path,
    destination: Path,
    *,
    families: Iterable[str] = RW_UPSAMPLE_FAMILIES,
    factor: int = 4,
) -> tuple[int, int]:
    """Paper §3.1 RW mix: the (deduped) ALL pair file with the auction- and
    negotiation-family subset upsampled ``factor``× — one original plus
    ``factor - 1`` extra copies of each matching pair.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    want = {str(f).replace("_", "-").lower() for f in families}
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"no pairs in {source}")
    extra: list[dict[str, Any]] = []
    for pair in rows:
        fam = str(pair.get("provenance", {}).get("game_family", "")).replace("_", "-").lower()
        if fam in want:
            extra.extend([pair] * (factor - 1))
    out = rows + extra
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for pair in out:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(
        f"upsample({','.join(sorted(want))} x{factor}): {len(rows)} -> {len(out)} "
        f"pairs (+{len(extra)} from {len(extra) // max(factor - 1, 1)} matched)",
        file=sys.stderr,
    )
    return len(rows), len(out)


def prepare_model(*, STUDENT_MODEL: str, lora_config, resume_checkpoint: Path | None):
    """Load the base model (or checkpoint) and attach its LoRA adapter."""
    import config
    _u = _dpolora("utils")
    attach_lora = _u.attach_lora
    load_base_model = _u.load_base_model
    load_lora_adapter = _u.load_lora_adapter

    if resume_checkpoint is not None:
        model, tokenizer = load_lora_adapter(
            resume_checkpoint, STUDENT_MODEL=STUDENT_MODEL, use_4bit=config.USE_4BIT
        )
        print(f"Resuming from {resume_checkpoint}", flush=True)
    else:
        model, tokenizer = load_base_model(STUDENT_MODEL=STUDENT_MODEL, use_4bit=config.USE_4BIT)
        model = attach_lora(model, lora_config)
    if config.GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    model.print_trainable_parameters()
    return model, tokenizer


def build_dpo_trainer(
    *,
    model,
    tokenizer,
    dataset,
    args,
    adapter_dir: Path,
    tensorboard_dir: Path,
    use_tensorboard: bool,
    save_total_limit: int,
):
    """Configure the masked DPO trainer for constructed preference pairs."""
    import config
    from trl import DPOConfig

    MinimalCheckpointCallback = _dpolora("lifecycle").MinimalCheckpointCallback

    masking = _load(Path(__file__).with_name("masking.py"), "noleakage_masking")
    EnvironmentMaskedDPOTrainer = masking.EnvironmentMaskedDPOTrainer

    train_kwargs = {
        "output_dir": str(adapter_dir),
        "per_device_train_batch_size": config.PER_DEVICE_BATCH,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
    }
    if args.max_steps is not None:
        train_kwargs["max_steps"] = args.max_steps
    dpo_args = DPOConfig(
        **train_kwargs,
        learning_rate=args.lr,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_ratio=config.WARMUP_RATIO,
        logging_steps=config.LOGGING_STEPS,
        save_steps=config.SAVE_STEPS,
        save_total_limit=save_total_limit,
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
        fp16=config.TRAIN_FP16,
        bf16=config.TRAIN_BF16,
        report_to="tensorboard" if use_tensorboard else "none",
        logging_dir=str(tensorboard_dir) if use_tensorboard else None,
        beta=args.beta,
        max_length=args.max_length,
    )
    return EnvironmentMaskedDPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[MinimalCheckpointCallback()],
    )


def extract_train_metrics(trainer) -> dict:
    """Extract stable summary fields from a completed trainer run."""
    metrics = {"global_step": trainer.state.global_step, "epoch": trainer.state.epoch}
    if trainer.state.log_history:
        last = trainer.state.log_history[-1]
        for key in (
            "train_runtime",
            "train_loss",
            "train_samples_per_second",
            "train_steps_per_second",
        ):
            if key in last:
                metrics[key] = last[key]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("existing", "teacher"), default="existing")
    parser.add_argument("--teacher-model", default=None, help="defaults to config.TEACHER_MODEL")
    parser.add_argument("--teacher-max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--counterfactual-mode",
        choices=("horizon-aware", "fixed"),
        default="horizon-aware",
        help="Eq. 3 when horizon-aware, Eq. 2 when fixed",
    )
    parser.add_argument(
        "--allow-unverified-frontier-model",
        action="store_true",
        help="accept legacy trajectories without matching frontier_model provenance",
    )
    parser.add_argument(
        "--max-invalid-fraction",
        type=float,
        default=0.2,
        help="abort if more than this fraction of teacher paraphrases fail "
        "structural validation (the rest are dropped with a warning)",
    )
    parser.add_argument(
        "--dedup",
        choices=_DEDUP_MODES,
        default="context",
        help="merge duplicate decision contexts before paraphrasing: 'context' "
        "(default) = one pair per prompt + pinned action; 'position' = per "
        "(game, round, action contrast); 'full' = exact prompt+rejected text; "
        "'none' = one pair per accepted round (pre-dedup behaviour)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="cap the output at N pairs (deterministic seeded sample)",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="write an empty output successfully when a distributed shard has no improving flips",
    )
    parser.add_argument(
        "--dedup-only",
        action="store_true",
        help="--input is an already-built pair JSONL: only de-duplicate and cap "
        "it (used by algorithm1.sh merge, where the pool was shard-split)",
    )
    parser.add_argument(
        "--upsample-only",
        action="store_true",
        help="--input is a built (deduped) pair JSONL: derive the paper RW mix "
        "by upsampling the auction/negotiation-family subset --upsample-factor x",
    )
    parser.add_argument(
        "--upsample-factor", type=int, default=4,
        help="RW upsample multiplier for the auction/negotiation subset (default 4)",
    )
    parser.add_argument(
        "--upsample-families",
        default=",".join(RW_UPSAMPLE_FAMILIES),
        help="comma-separated game_family values to upsample for the RW mix",
    )
    args = parser.parse_args()
    if args.upsample_only:
        try:
            _, count = upsample_pairs_file(
                args.input, args.output,
                families=args.upsample_families.split(","),
                factor=args.upsample_factor,
            )
        except (OSError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")
        print(f"wrote {count} pairs (RW upsample) to {args.output}")
        return
    if args.dedup_only:
        try:
            count = dedup_pairs_file(
                args.input, args.output, dedup=args.dedup, max_pairs=args.max_pairs
            )
        except (OSError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")
        print(f"wrote {count} de-duplicated pairs to {args.output}")
        return
    try:
        count = build_file(
            args.input,
            args.output,
            provider=args.provider,
            teacher_STUDENT_MODEL=args.teacher_model,
            teacher_max_new_tokens=args.teacher_max_new_tokens,
            counterfactual_mode=args.counterfactual_mode,
            require_frontier_model_match=not args.allow_unverified_frontier_model,
            max_invalid_fraction=args.max_invalid_fraction,
            dedup=args.dedup,
            max_pairs=args.max_pairs,
            allow_empty=args.allow_empty,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"wrote {count} verified round-level pairs to {args.output}")


if __name__ == "__main__":
    main()
