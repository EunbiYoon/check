"""Extended episode-level robustness evaluation using the rollout path.

This module deliberately reuses ``eval.table.run_table.eval_variant`` so prompts,
opponents, decoding, rewards, model loading, and game execution stay unchanged.
Set PYTHONHASHSEED to make the legacy hash-derived episode seeds reproducible.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from eval.rollout.games import episode_seed
from eval.rollout.checkpoints import discover_all_adapters, prepare_checkpoint_dir
from eval.rollout.progress import EvalLogger
from eval.rollout.resume import ParsedEpisode
from eval.table.run_table import eval_variant

FIELDS = ["model", "environment", "opponent", "episode_id", "seed", "cumulative_reward"]
REQUESTED = ["base", "filter_on", "filter_off", "core", "aux", "all", "rw", "merge"]


def _model_local(STUDENT_MODEL: str) -> bool:
    path = Path(STUDENT_MODEL).expanduser()
    if path.exists():
        return True
    try:
        from huggingface_hub import scan_cache_dir
        return any(repo.repo_id == STUDENT_MODEL for repo in scan_cache_dir().repos)
    except Exception:
        return False


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prior(rows: list[dict[str, str]], model: str) -> dict[str, list[ParsedEpisode]]:
    out: dict[str, list[ParsedEpisode]] = defaultdict(list)
    for row in rows:
        if row["model"] == model:
            out[row["environment"]].append(
                ParsedEpisode(int(row["episode_id"]), float(row["cumulative_reward"]), row["opponent"])
            )
    return dict(out)


class EpisodeCSVLogger(EvalLogger):
    def __init__(self, log_path: Path, csv_path: Path, model: str, seed_base: int, append: bool):
        super().__init__(log_path, append=append)
        self.csv_path, self.model, self.seed_base = csv_path, model, seed_base
        self._opponents: dict[tuple[str, int], str] = {}
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists():
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writeheader()

    def episode_start(self, game: str, episode: int, n_episodes: int, opponent: str) -> None:
        self._opponents[(game, episode)] = opponent
        super().episode_start(game, episode, n_episodes, opponent)

    def episode_done(self, game: str, episode: int, cr: float, n_rounds: int) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writerow({
                "model": self.model,
                "environment": game,
                "opponent": self._opponents.get((game, episode), ""),
                "episode_id": episode,
                "seed": episode_seed(game, episode - 1, self.seed_base),
                "cumulative_reward": repr(float(cr)),
            })
            handle.flush()
        super().episode_done(game, episode, cr, n_rounds)


def _stats(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    mean = sum(values) / n
    ordered = sorted(values)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    se = std / math.sqrt(n)
    return {"n": n, "mean_cr": mean, "median_cr": median, "std_cr": std,
            "variance_cr": variance, "se_cr": se, "ci95_low": mean - 1.96 * se,
            "ci95_high": mean + 1.96 * se, "min_cr": min(values), "max_cr": max(values)}


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyse(out_dir: Path) -> None:
    episode_path = out_dir / "episode_rewards.csv"
    rows = _read_rows(episode_path)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    keyed: dict[tuple[str, str, int], float] = {}
    for row in rows:
        key = (row["model"], row["environment"])
        grouped[key].append(float(row["cumulative_reward"]))
        keyed[(row["model"], row["environment"], int(row["episode_id"]))] = float(row["cumulative_reward"])
    summary = [{"model": m, "environment": e, **_stats(vals)} for (m, e), vals in sorted(grouped.items())]
    summary_fields = ["model", "environment", "n", "mean_cr", "median_cr", "std_cr", "variance_cr",
                      "se_cr", "ci95_low", "ci95_high", "min_cr", "max_cr"]
    _write_csv(out_dir / "summary_by_model_env.csv", summary, summary_fields)

    overall: dict[str, list[float]] = defaultdict(list)
    for (model, _), vals in grouped.items(): overall[model].extend(vals)
    trained = [m for m in overall if m not in ("base", "filter_on", "filter_off")]
    best = max(trained, key=lambda m: sum(overall[m]) / len(overall[m])) if trained else None
    pairs = [("rw", "merge", "RW vs MERGE")]
    if best:
        pairs += [(best, "base", "best trained vs base"), (best, "filter_on", "best trained vs DPO filter-on"),
                  (best, "filter_off", "best trained vs DPO filter-off")]
    comparisons: list[dict] = []
    for left, right, label in pairs:
        for env in config.ALL_GAMES + ["ALL"]:
            keys = sorted(set((e, ep) for m, e, ep in keyed if m == left and (env == "ALL" or e == env)) &
                          set((e, ep) for m, e, ep in keyed if m == right and (env == "ALL" or e == env)))
            diffs = [keyed[(left, e, ep)] - keyed[(right, e, ep)] for e, ep in keys]
            if not diffs: continue
            s = _stats(diffs)
            comparisons.append({"comparison": label, "model_a": left, "model_b": right,
                                "environment": env, "n_paired": s["n"], "mean_cr_difference": s["mean_cr"],
                                "se_difference": s["se_cr"], "ci95_low": s["ci95_low"], "ci95_high": s["ci95_high"]})
    pair_fields = ["comparison", "model_a", "model_b", "environment", "n_paired", "mean_cr_difference",
                   "se_difference", "ci95_low", "ci95_high"]
    _write_csv(out_dir / "pairwise_comparisons.csv", comparisons, pair_fields)
    _plots(out_dir, rows, summary, comparisons)


def _plots(out_dir: Path, rows: list[dict[str, str]], summary: list[dict], comparisons: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (out_dir / "PLOTS_UNAVAILABLE.txt").write_text("matplotlib is not installed\n", encoding="utf-8")
        return
    envs = config.ALL_GAMES
    models = sorted({r["model"] for r in rows})
    means = {(r["model"], r["environment"]): r for r in summary}
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model in enumerate(models):
        xs = [j + (i - (len(models)-1)/2) * 0.08 for j in range(len(envs))]
        ys = [means.get((model, e), {}).get("mean_cr", math.nan) for e in envs]
        es = [1.96 * means.get((model, e), {}).get("se_cr", 0) for e in envs]
        ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=1, capsize=2, label=model)
    ax.set_xticks(range(len(envs)), envs, rotation=45, ha="right"); ax.set_ylabel("Mean cumulative reward (95% CI)"); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(out_dir / "mean_cr_ci95_by_model.png", dpi=300); plt.close(fig)
    main = [m for m in ("core", "aux", "all", "rw", "merge") if m in models]
    if main:
        fig, ax = plt.subplots(figsize=(10, 6)); ax.boxplot([[float(r["cumulative_reward"]) for r in rows if r["model"] == m] for m in main], labels=main, showfliers=False)
        ax.set_ylabel("Cumulative reward"); fig.tight_layout(); fig.savefig(out_dir / "trained_variant_reward_distributions.png", dpi=300); plt.close(fig)
    rw = [r for r in comparisons if r["comparison"] == "RW vs MERGE" and r["environment"] != "ALL"]
    if rw:
        fig, ax = plt.subplots(figsize=(11, 5)); x = range(len(rw)); y = [float(r["mean_cr_difference"]) for r in rw]
        err = [1.96 * float(r["se_difference"]) for r in rw]; ax.errorbar(x, y, yerr=err, fmt="o", capsize=3); ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(list(x), [r["environment"] for r in rw], rotation=45, ha="right"); ax.set_ylabel("RW - MERGE mean CR (95% CI)")
        fig.tight_layout(); fig.savefig(out_dir / "rw_vs_merge_ci95.png", dpi=300); plt.close(fig)
    comparable = [(r["model"], r["environment"], r["mean_cr"], config.PAPER_TABLE2.get(r["environment"], {}).get(r["model"]))
                  for r in summary if config.PAPER_TABLE2.get(r["environment"], {}).get(r["model"]) is not None]
    if comparable:
        old = [float(x[3]) for x in comparable]; new = [float(x[2]) for x in comparable]
        fig, ax = plt.subplots(figsize=(6, 6)); ax.scatter(old, new, alpha=.75)
        lo, hi = min(old + new), max(old + new); ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("Original n=12 mean CR"); ax.set_ylabel("Extended mean CR"); fig.tight_layout()
        fig.savefig(out_dir / "original_n12_vs_extended.png", dpi=300); plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="1000-episode robustness evaluation")
    p.add_argument("--episodes", type=int, default=1000); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "extended_eval_1000")
    p.add_argument("--lora-dir", type=Path, default=None); p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--variants", default=",".join(REQUESTED)); p.add_argument("--model-id", default=config.PAPER_STUDENT_MODEL)
    p.add_argument("--max-tokens", type=int, default=192); p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--analyse-only", action="store_true"); p.add_argument("--list", action="store_true", help="Report local availability without loading models")
    args = p.parse_args(argv); args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.analyse_only: analyse(args.out_dir); return 0
    if "PYTHONHASHSEED" not in os.environ:
        p.error("Set PYTHONHASHSEED (recommended: PYTHONHASHSEED=0) before launching")
    adapters = discover_all_adapters(lora_dir=args.lora_dir)
    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    available = (["base"] if _model_local(args.STUDENT_MODEL) else []) + [v for v in requested if v in adapters]
    if "merge" in requested and "aux" in adapters and "all" in adapters: available.append("merge")
    available = [v for v in requested if v in set(available)]
    unavailable = [v for v in requested if v not in available]
    (args.out_dir / "availability.txt").write_text(
        "available: " + ", ".join(available) + "\nunavailable: " + ", ".join(unavailable) +
        "\nexternal/frontier unavailable: haiku (no paid API calls attempted)\n", encoding="utf-8")
    if args.list:
        print((args.out_dir / "availability.txt").read_text(encoding="utf-8"), end="")
        return 0
    if not available:
        print("No requested model is available locally; wrote availability.txt")
        return 2
    # Reuse the existing evaluator's argument contract.
    from argparse import Namespace
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = args.out_dir / "staging"
        prepare_checkpoint_dir(checkpoint_dir, lora_dir=args.lora_dir,
                               variants=[v for v in available if v not in ("base", "merge")])
        if "merge" in available:
            from eval.table.run_paper_tables import _maybe_merge
            _maybe_merge(checkpoint_dir, alpha=config.MERGE_ALPHA)
    episode_csv = args.out_dir / "episode_rewards.csv"; existing = _read_rows(episode_csv)
    for variant in available:
        logger = EpisodeCSVLogger(args.out_dir / "extended_eval.log", episode_csv, variant, args.seed, append=True)
        ns = Namespace(mode="lora", STUDENT_MODEL=args.STUDENT_MODEL, checkpoint_dir=checkpoint_dir, episodes=args.episodes,
            seed=args.seed, lora_r=config.PAPER_LORA_R, lora_alpha=config.PAPER_LORA_ALPHA,
            lora_target=config.PAPER_LORA_TARGET_MODULES, dpo_beta=config.DPO_BETA, merge_alpha=config.MERGE_ALPHA,
            lr=config.LEARNING_RATE, epochs=config.NUM_EPOCHS, no_4bit=args.no_4bit,
            max_tokens=args.max_tokens, logger=logger, resume=False, games_list=None, run_dir=args.out_dir,
            log_file=args.out_dir / "extended_eval.log")
        # Supply CSV rewards for interruption-safe continuation.
        prior = _prior(existing, variant)
        agent_metrics = __import__("eval.metric.metrics", fromlist=["evaluate_agent"])
        from eval.table.run_table import build_agent
        agent, tag = build_agent(ns, variant); logger.variant_start(variant, tag, ns.mode, ns.episodes)
        agent_metrics.evaluate_agent(agent, n_episodes=ns.episodes, seed_base=ns.seed, logger=logger, prior_episodes=prior)
        existing = _read_rows(episode_csv)
    analyse(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
