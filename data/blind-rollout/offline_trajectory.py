#!/usr/bin/env python3
"""Algorithm 1 line 2: generate lossless blind-trajectory JSONL."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "config.py").is_file()
)  # repo root (robust to how deeply this file is nested under data/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from eval.rollout.agents import LoRALLMAgent
from eval.rollout.games import EpisodeResult, run_game_batch


def _load_blind_model(model_id: str):
    """BF16 causal-LM load for the blind rollout.

    Self-contained on purpose: blind rollout is data construction and must not
    depend on the training package (no bitsandbytes, no LoRA, no 4-bit).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return model, tokenizer


SUPPORTED_GAMES = (
    "pd-classic", "pd-tight", "pd-high-temptation", "stag-hunt",
    "bos", "matching-pennies", "negotiation", "auction", "ipd-stage",
)

# Repeated-game pools exclude stochastic policies that cannot be replayed by
# the exact counterfactual filter. One-shot games only need the realised move.
REPEATED_OPPONENTS = (
    "always_cooperate", "always_defect", "tit_for_tat", "grim_trigger",
    "pavlov", "tit_for_two_tats",
)
ONE_SHOT_MATRIX_OPPONENTS = REPEATED_OPPONENTS + ("random", "epsilon_greedy")
OPPONENTS_BY_GAME = {
    "pd-classic": REPEATED_OPPONENTS,
    "pd-tight": REPEATED_OPPONENTS,
    "pd-high-temptation": REPEATED_OPPONENTS,
    "stag-hunt": REPEATED_OPPONENTS,
    "bos": ONE_SHOT_MATRIX_OPPONENTS,
    "matching-pennies": ONE_SHOT_MATRIX_OPPONENTS,
    "ipd-stage": ONE_SHOT_MATRIX_OPPONENTS,
    "negotiation": ("always_cooperate", "always_defect", "random", "epsilon_greedy"),
    "auction": ("truthful_bidder", "always_cooperate", "random", "epsilon_greedy"),
}


def _matrix_rows(
    actions: Iterable[Any], payoff: dict[tuple[Any, Any], tuple[float, float]]
) -> list[dict[str, Any]]:
    return [
        {"own": own, "opponent": opp, "payoff": list(payoff[(own, opp)])}
        for own in actions
        for opp in actions
    ]


def game_spec(game: str, result: EpisodeResult) -> dict[str, Any]:
    """Return the solver inputs required by Algorithm 1 lines 4–7."""
    if game in config.PD_VARIANTS:
        p = config.PD_VARIANTS[game]
        payoff = {
            ("C", "C"): (p.R, p.R), ("C", "D"): (p.S, p.T),
            ("D", "C"): (p.T, p.S), ("D", "D"): (p.P, p.P),
        }
        return {"game_family": "matrix", "legal_actions": ["C", "D"],
                "payoffs": _matrix_rows(("C", "D"), payoff)}
    if game == "stag-hunt":
        payoff = {
            ("Stag", "Stag"): (4, 4), ("Stag", "Hare"): (0, 3),
            ("Hare", "Stag"): (3, 0), ("Hare", "Hare"): (1, 1),
        }
        return {"game_family": "matrix", "legal_actions": ["Stag", "Hare"],
                "payoffs": _matrix_rows(("Stag", "Hare"), payoff)}
    if game in {"bos", "matching-pennies"}:
        actions = ("Opera", "Football") if game == "bos" else ("H", "T")
        raw = config.BOS_PAYOFFS if game == "bos" else config.MP_PAYOFFS
        payoff = {(actions[i], actions[j]): raw[(i, j)] for i in range(2) for j in range(2)}
        return {"game_family": game, "legal_actions": list(actions),
                "payoffs": _matrix_rows(actions, payoff)}
    if game == "negotiation":
        return {"game_family": "bargaining", "capacity": 4,
                "own_weights": [3, 4, 3]}
    if game == "auction":
        prompt = result.rounds[0].prompt or ""
        match = re.search(r"private value is (\d+)", prompt)
        if match is None:
            raise ValueError("auction rollout prompt is missing the private value")
        return {"game_family": "auction", "private_value": int(match.group(1)),
                "legal_actions": list(range(201))}
    raise ValueError(f"Algorithm 1 solver does not support game {game!r}")


def game_unit_count(game: str, episodes_per_combination: int) -> int:
    """Number of indivisible (opponent, episode) work units for one game."""
    return len(OPPONENTS_BY_GAME[game]) * episodes_per_combination


def shard_unit_total(
    games: Iterable[str], episodes_per_combination: int,
    *, num_shards: int = 1, shard_index: int = 0,
) -> int:
    """Count the work units this shard will run across ``games``.

    The unit order is opponent-major, episode-minor, restarting per game — the
    exact order ``rollout_game`` walks — so a modulo split here matches what the
    generator keeps.
    """
    total = 0
    for game in games:
        units = game_unit_count(game, episodes_per_combination)
        total += len(range(shard_index, units, num_shards)) if num_shards > 1 else units
    return total


def rollout_game(
    *, game: str, agent: LoRALLMAgent, episodes_per_combination: int,
    seed: int, STUDENT_MODEL: str, num_shards: int = 1, shard_index: int = 0,
) -> Iterable[dict[str, Any]]:
    """Yield each episode immediately so progress never waits for a full batch.

    ``num_shards``/``shard_index`` keep only every ``num_shards``-th (opponent,
    episode) unit (offset ``shard_index``), so one game/seed can be spread over
    several GPUs. The per-unit seed is derived from ``opponent_index`` and
    ``episode_index`` alone, so the split never changes which trajectory a unit
    produces.
    """
    unit = -1
    for opponent_index, opponent in enumerate(OPPONENTS_BY_GAME[game]):
        combination_seed = seed + opponent_index * 1_000_000
        announced = False
        for episode_index in range(episodes_per_combination):
            unit += 1
            if num_shards > 1 and unit % num_shards != shard_index:
                continue
            if not announced:
                print(f"starting {game} / {opponent}", flush=True)
                announced = True
            episode = episode_index + 1
            print(
                f"  running episode {episode}/{episodes_per_combination} "
                f"({game} / {opponent})",
                flush=True,
            )
            # run_game_batch numbers its only episode as zero internally. The
            # adjusted base preserves the original batch's per-episode seed.
            result = run_game_batch(
                game, agent, 1,
                seed_base=combination_seed + episode_index * 997,
                opponent_name=opponent,
            )[0]
            yield result.to_training_trajectory(
                # The seed segment keeps ids unique when the same game is rolled
                # at several seeds and the shards are merged into one file; pair
                # construction needs a stable, collision-free trajectory id.
                trajectory_id=f"{game}-{opponent}-s{seed}-{episode}",
                game_spec=game_spec(game, result),
                frontier_model=STUDENT_MODEL,
            )


def write_trajectories(
    rows: Iterable[dict[str, Any]], output: Path, *, total: int | None = None,
    progress_every: int = 1,
) -> int:
    """Stream trajectory rows to ``output`` as they are produced.

    Each row is written and flushed immediately, so a running shard can be
    watched live with ``tail -f <output>``.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    started = time.monotonic()
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            completions = row.get("round_completions", [])
            empty_rounds = []
            for index, completion in enumerate(completions, start=1):
                match = re.search(r"<think>(.*?)</think>", str(completion), re.DOTALL)
                reasoning = match.group(1).strip() if match is not None else ""
                if not reasoning or reasoning.lower().startswith("fallback:"):
                    empty_rounds.append(index)
            if empty_rounds:
                raise RuntimeError(
                    f"refusing to write trajectory "
                    f"{row.get('trajectory_id', row.get('id'))!r}: "
                    f"empty or fallback reasoning in round(s) {empty_rounds}"
                )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            if (total is not None and count == total) or (
                progress_every and count % progress_every == 0
            ):
                elapsed = time.monotonic() - started
                rate = count / elapsed if elapsed else 0.0
                if total is None:
                    status = f"progress {count}"
                    eta = "ETA unknown"
                else:
                    remaining = (total - count) / rate if rate else 0.0
                    status = f"progress {count}/{total} ({count / total:.1%})"
                    eta = f"ETA {_duration(remaining)}"
                print(f"{status} | elapsed {_duration(elapsed)} | {eta} | "
                      f"{rate:.2f} trajectories/s", flush=True)
    if count == 0:
        output.unlink(missing_ok=True)
        raise ValueError("blind rollout emitted no trajectories")
    return count


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", default=",".join(config.BLIND_GAMES))
    parser.add_argument("--episodes-per-combination", type=int,
                        default=config.BLIND_EPISODES_PER_COMBINATION)
    parser.add_argument("--seed", type=int, default=config.BLIND_SEED)
    parser.add_argument("--model", default=config.BLIND_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=config.BLIND_MAX_NEW_TOKENS)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction,
                        default=config.BLIND_DO_SAMPLE)
    parser.add_argument("--temperature", type=float, default=config.BLIND_TEMPERATURE)
    parser.add_argument(
        "--reasoning-format",
        choices=("concise", "slots"),
        default=os.environ.get("BLIND_REASONING_FORMAT", "slots"),
        help="'slots' elicits [Prior][Update][EV][Decision] structure so the "
             "coupling filter (Hypothesis B) has an [EV] slot to parse; "
             "'concise' is free-text <=80 words. A+beta ignores the format.",
    )
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split the (opponent, episode) work list this many "
                             "ways so one game/seed can run on several GPUs")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="0-based index of this shard in [0, --num-shards)")
    args = parser.parse_args()
    games = [game.strip() for game in args.games.split(",") if game.strip()]
    invalid = [game for game in games if game not in SUPPORTED_GAMES]
    if invalid:
        parser.error("unsupported Algorithm 1 game(s): " + ", ".join(invalid))
    if args.episodes_per_combination < 1:
        parser.error("--episodes-per-combination must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    total = shard_unit_total(
        games, args.episodes_per_combination,
        num_shards=args.num_shards, shard_index=args.shard_index,
    )
    if total == 0:
        # Legal when a game/seed has fewer work units than shards; leave an empty
        # file so the merge step has something to glob and skip loading the model.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")
        print(
            f"shard {args.shard_index}/{args.num_shards} has no work units; "
            f"wrote empty {args.output}",
            flush=True,
        )
        return

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading blind model {args.model} (this can take a few minutes)...", flush=True)
    # Qwen 7B fits on an L40S in BF16; bitsandbytes 4-bit produced degenerate
    # repeated-token generations, so data generation stays full precision.
    model, tokenizer = _load_blind_model(args.model)
    first_device = next(model.parameters()).device
    print(f"Loaded blind model on {first_device}; CUDA available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"CUDA memory allocated after load: {allocated:.2f} GiB", flush=True)
    print(f"Blind reasoning format: {args.reasoning_format}", flush=True)
    agent = LoRALLMAgent(
        model, tokenizer,
        max_new_tokens=args.max_new_tokens,
        strict_output=True,
        do_sample=args.do_sample,
        temperature=args.temperature,
        reasoning_format=args.reasoning_format,
    )
    rows = (
        row
        for game in games
        for row in rollout_game(
            game=game, agent=agent,
            episodes_per_combination=args.episodes_per_combination,
            seed=args.seed, STUDENT_MODEL=args.model,
            num_shards=args.num_shards, shard_index=args.shard_index,
        )
    )
    if args.num_shards > 1:
        print(f"shard {args.shard_index}/{args.num_shards}", flush=True)
    print(f"Expected trajectories: {total}", flush=True)
    count = write_trajectories(rows, args.output, total=total)
    print(f"wrote {count} blind trajectories to {args.output}")


if __name__ == "__main__":
    main()
