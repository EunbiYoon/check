#!/usr/bin/env python3
"""Build round-level DPO pairs and assemble the resulting DPO trainer."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable


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
    from train.dpo_lora.utils import load_base_model

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


def build_pairs(
    row: dict[str, Any],
    paraphrase: Callable[..., str] | None = None,
    *,
    counterfactual_mode: str = "horizon-aware",
) -> list[dict[str, Any]]:
    trajectory_id = row.get("id")
    if trajectory_id is None:
        raise ValueError("trajectory requires a stable id for leakage-free splitting")
    completions = _round_completions(row)
    own_actions = row.get("own_actions", row.get("agent_actions"))
    if len(completions) != len(own_actions):
        raise ValueError("one blind completion is required per recorded round")
    prompts = _round_prompts(row, len(completions))
    paraphrase = paraphrase or _existing_paraphraser(row)
    pairs: list[dict[str, Any]] = []
    for candidate in _accepted_rounds(row, counterfactual_mode=counterfactual_mode):
        index = candidate["round_index"]
        chosen = paraphrase(
            round_index=index,
            history_prompt=prompts[index],
            opponent_action=candidate["opponent_action"],
            solver_action=candidate["solver_action"],
        ).strip()
        is_final_round = index + 1 == len(completions)
        action_errors = no_leak.validate_pinned_action(chosen, candidate["solver_action"])
        reasoning_match = re.search(r"<think>(.*?)</think>", chosen, re.I | re.S)
        if reasoning_match is None:
            action_errors.append("missing <think> block")
            flags = []
        else:
            reasoning = reasoning_match.group(1)
            action_errors.extend(
                no_leak.validate_reasoning_structure(
                    reasoning, is_final_round=is_final_round
                )
            )
            flags = no_leak.audit_reasoning(reasoning, round_number=index + 1)
        if action_errors:
            raise ValueError(
                f"round {index}: invalid frontier completion: {', '.join(action_errors)}"
            )
        # Leakage flags are heuristic candidates for manual review, not deletion
        # rules (paper §2.3): record them on the pair, keep the pair.
        leakage_audit = [
            {"rule": flag.rule, "excerpt": flag.excerpt} for flag in flags
        ]
        pairs.append({
            "prompt": prompts[index],
            "chosen": chosen,
            "rejected": completions[index],
            "trajectory_id": str(trajectory_id),
            "round": index + 1,
            "provenance": {
                "pipeline": PAIR_PIPELINE,
                "chosen_reasoning": "frontier-no-leak",
                "chosen_action": "solver-best-response",
                "rejected": "blind-round-completion",
                "filter": counterfactual_mode if len(completions) > 1 else "one-shot-improvement",
                "leakage_audit": leakage_audit,
                "is_final_round": is_final_round,
                **candidate,
            },
        })
    return pairs


def build_file(
    source: Path,
    destination: Path,
    provider: str = "existing",
    teacher_STUDENT_MODEL: str | None = None,
    teacher_max_new_tokens: int = 1024,
    counterfactual_mode: str = "horizon-aware",
    require_frontier_model_match: bool = True,
) -> int:
    import config

    destination.parent.mkdir(parents=True, exist_ok=True)
    shared = (
        teacher_paraphraser(
            teacher_STUDENT_MODEL or config.TEACHER_MODEL,
            max_new_tokens=teacher_max_new_tokens,
        )
        if provider == "teacher"
        else None
    )
    count = 0
    with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if provider == "teacher" and require_frontier_model_match:
                    recorded_model = row.get("frontier_model")
                    expected_model = teacher_STUDENT_MODEL or config.TEACHER_MODEL
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
                for pair in build_pairs(
                    row, shared, counterfactual_mode=counterfactual_mode
                ):
                    dst.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    count += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if count == 0:
        raise ValueError("no improving solver flips were emitted")
    return count


def prepare_model(*, STUDENT_MODEL: str, lora_config, resume_checkpoint: Path | None):
    """Load the base model (or checkpoint) and attach its LoRA adapter."""
    import config
    from train.dpo_lora.utils import attach_lora, load_base_model, load_lora_adapter

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

    from train.dpo_lora.lifecycle import MinimalCheckpointCallback

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
    args = parser.parse_args()
    try:
        count = build_file(
            args.input,
            args.output,
            provider=args.provider,
            teacher_STUDENT_MODEL=args.teacher_model,
            teacher_max_new_tokens=args.teacher_max_new_tokens,
            counterfactual_mode=args.counterfactual_mode,
            require_frontier_model_match=not args.allow_unverified_frontier_model,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"wrote {count} verified round-level pairs to {args.output}")


if __name__ == "__main__":
    main()
