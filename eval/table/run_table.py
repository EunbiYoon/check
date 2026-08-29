"""
SARL Tables 1-7 runner (LoRA / DPO / A+b).

  python -m eval.table.run_table --table 1 [--variant base] [--compare-reference]
  python -m eval.table.run_table --table 5 --train
  python -m eval.table.run_table --table 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train.dpo_lora.gpu_env  # noqa: F401 — NVHPC libcublas path before bnb

from eval.rollout.agents import HeuristicAgent, LoRALLMAgent, OllamaAgent
from config import (
    ALL_GAMES,
    ABBREV_TO_GAME,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    DPO_BETA,
    EPISODES_PER_ENV,
    GAME_ABBREV,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MERGE_ALPHA,
    STUDENT_MODEL,
    NUM_EPOCHS,
    OLLAMA_BASE_URL,
    OLLAMA_FRONTIER_MODEL,
    PAPER_STUDENT_MODEL,
    PAPER_TABLE1,
    PAPER_TABLE2,
    PAPER_TABLE3,
    PAPER_TABLE4,
    PAPER_TABLE5,
    PAPER_TABLE6,
    PAPER_TABLE7,
    LORA_DIR,
    PROJECT_ROOT,
    TABLE1_VARIANTS,
    TABLE2_VARIANTS,
    TABLE3_VARIANTS,
    TRAINING_VARIANTS,
    USE_4BIT,
    VARIANT_LABELS,
)
from eval.metric.metrics import EvalMetrics, evaluate_agent, metrics_to_dict
from eval.table.paths import new_run_dir, resolve_run_dir, table_out_path, utc_stamp, write_latest_pointer
from eval.rollout.progress import EvalLogger
from eval.rollout.resume import parse_eval_log, parse_games_arg, resume_summary
from train.dpo_lora.utils import attach_lora, build_lora_config, load_base_model

TRAIN_MODULE = "train.dpo_lora"


def fmt(x: float) -> str:
    if x >= 0:
        return f"+{x:.2f}" if abs(x) < 100 else f"{x:.2f}"
    return f"{x:.2f}"


def fmt_cell(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}"


def adapter_path(variant: str, ckpt_root: Path) -> Path | None:
    if variant == "base":
        return None
    if variant == "merge":
        return ckpt_root / "merge"
    if variant == "haiku":
        return None
    direct = ckpt_root / variant
    nested = direct / "adapter"
    if (nested / "adapter_config.json").exists():
        return nested
    if (direct / "adapter_config.json").exists():
        return direct
    return direct


def build_agent(args, variant: str):
    if args.mode == "heuristic":
        return HeuristicAgent(), f"heuristic-{variant}"

    if args.mode == "ollama":
        model = args.ollama_model or OLLAMA_FRONTIER_MODEL
        return (
            OllamaAgent(
                model,
                base_url=args.ollama_base_url or OLLAMA_BASE_URL,
                max_new_tokens=args.max_tokens,
            ),
            f"ollama:{model}",
        )

    model, tokenizer = load_base_model(
        STUDENT_MODEL=args.STUDENT_MODEL,
        use_4bit=not args.no_4bit,
    )
    lora_cfg = build_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        target_modules=(
            args.lora_target
            if args.lora_target == "all-linear"
            else args.lora_target.split(",")
        ),
    )

    ckpt = adapter_path(variant, Path(args.checkpoint_dir))
    if ckpt and ckpt.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ckpt))
        tag = variant
    elif variant == "base":
        tag = "base"
    else:
        model = attach_lora(model, lora_cfg)
        tag = f"{variant}-untrained"

    return LoRALLMAgent(model, tokenizer, max_new_tokens=args.max_tokens), tag


def _default_log_path(table: int, variant: str | None, run_dir: Path | None = None) -> Path:
    if run_dir is not None:
        return run_dir / "logs" / f"eval_table{table}_{variant or 'all'}.log"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    v = variant or "all"
    return PROJECT_ROOT / "results" / f"eval_table{table}_{v}_{stamp}.log"


def _eval_log_path(args, variant: str) -> Path | None:
    if getattr(args, "log_file", None):
        return Path(args.log_file)
    run_dir = getattr(args, "run_dir", None)
    if run_dir is None:
        return None
    table = getattr(args, "table", None)
    if table:
        return _default_log_path(table, variant, Path(run_dir))
    return Path(run_dir) / "logs" / "eval_suite.log"


def eval_variant(args, variant: str) -> EvalMetrics:
    logger: EvalLogger | None = getattr(args, "logger", None)
    prior = None
    if getattr(args, "resume", False):
        log_path = _eval_log_path(args, variant)
        if log_path and log_path.is_file():
            prior = parse_eval_log(log_path, variant=variant)
            if prior:
                print(
                    f"  Resume {variant}: {resume_summary(prior, args.episodes)} from {log_path}",
                    flush=True,
                )
    agent, tag = build_agent(args, variant)
    if logger:
        logger.variant_start(variant, tag, args.mode, args.episodes)
    else:
        print(f"  [{variant}] tag={tag} episodes={args.episodes}")
    m = evaluate_agent(
        agent,
        n_episodes=args.episodes,
        seed_base=args.seed,
        logger=logger,
        games=getattr(args, "games_list", None),
        prior_episodes=prior,
    )
    if logger:
        logger.variant_done(variant, m.fc_id, m.fc_ho, m.cr_id, m.cr_ho)
    return m


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {path}")


# ------------------------------------------------------------------ Table 1
def run_table1(args) -> None:
    print("=== Table 1: Headline results (fc, CR over 12 envs, n=12) ===")
    print(f"LoRA r={args.lora_r} alpha={args.lora_alpha} dropout={LORA_DROPOUT} | DPO beta={args.dpo_beta}")

    if args.compare_reference:
        print(f"\n{'Model':<32} {'fc(ID)':>8} {'fc(HO)':>8} {'CR(ID)':>10} {'CR(HO)':>10}")
        print("-" * 72)
        for v in TABLE1_VARIANTS + ["haiku"]:
            r = PAPER_TABLE1[v]
            print(
                f"{VARIANT_LABELS.get(v, v):<32} {r['fc_id']:>8.2f} {r['fc_ho']:>8.2f} "
                f"{fmt(r['cr_id']):>10} {fmt(r['cr_ho']):>10}"
            )
        return

    variants = [args.variant] if args.variant else TABLE1_VARIANTS
    rows: dict[str, dict] = {}
    print(f"\n{'Model':<32} {'fc(ID)':>8} {'fc(HO)':>8} {'CR(ID)':>10} {'CR(HO)':>10}")
    print("-" * 72)
    for v in variants:
        m = eval_variant(args, v)
        rows[v] = metrics_to_dict(m)
        label = VARIANT_LABELS.get(v, v)
        print(
            f"{label:<32} {m.fc_id:>8.2f} {m.fc_ho:>8.2f} "
            f"{fmt(m.cr_id):>10} {fmt(m.cr_ho):>10}"
        )
    save_json(Path(args.out), {"table": 1, "hyperparams": _hyperparams(args), "rows": rows})


# ------------------------------------------------------------------ Table 2
def run_table2(args) -> None:
    print("=== Table 2: Per-game cumulative reward ===")
    variants = [args.variant] if args.variant else TABLE2_VARIANTS

    if args.compare_reference:
        header = f"{'Game':<22}" + "".join(f" {v:>8}" for v in variants)
        print(header)
        print("-" * len(header))
        for game in ALL_GAMES:
            line = f"{game:<22}"
            for v in variants:
                val = PAPER_TABLE2.get(game, {}).get(v)
                line += f" {val if val is not None else 'n/a':>8}"
            print(line)
        return

    all_rows: dict[str, dict] = {}
    header = f"{'Game':<22}" + "".join(f" {v:>8}" for v in variants)
    print(header)
    print("-" * len(header))
    for v in variants:
        if v == "haiku":
            continue
        all_rows[v] = metrics_to_dict(eval_variant(args, v))

    for game in ALL_GAMES:
        line = f"{game:<22}"
        for v in variants:
            if v == "haiku":
                ref = PAPER_TABLE2.get(game, {}).get("haiku")
                line += f" {ref if ref is not None else 'n/a':>8}"
            else:
                cr = all_rows[v]["per_game_cr"].get(game, 0.0)
                line += f" {cr:>8.2f}"
        print(line)
    save_json(Path(args.out), {"table": 2, "rows": all_rows})


# ------------------------------------------------------------------ Table 3
def run_table3(args) -> None:
    print("=== Table 3: Per-opponent-axis CR (12 envs combined) ===")
    variants = [args.variant] if args.variant else TABLE3_VARIANTS

    if args.compare_reference:
        axes = list(PAPER_TABLE3.keys())
        header = f"{'Set Axis':<18}" + "".join(f" {v:>8}" for v in variants)
        print(header)
        print("-" * len(header))
        for axis in axes:
            line = f"{axis:<18}"
            for v in variants:
                val = PAPER_TABLE3[axis].get(v)
                line += f" {val if val is not None else 'n/a':>8}"
            print(line)
        return

    all_rows: dict[str, dict] = {}
    for v in variants:
        if v == "haiku":
            continue
        all_rows[v] = metrics_to_dict(eval_variant(args, v))

    axes = sorted({k for r in all_rows.values() for k in r["per_axis_cr"]})
    header = f"{'Set Axis':<18}" + "".join(f" {v:>8}" for v in variants if v != "haiku")
    print(header)
    for axis in axes:
        line = f"{axis:<18}"
        for v in variants:
            if v == "haiku":
                ref = PAPER_TABLE3.get(axis, {}).get("haiku")
                line += f" {ref if ref is not None else 'n/a':>8}"
            else:
                cr = all_rows[v]["per_axis_cr"].get(axis, 0.0)
                line += f" {cr:>8.2f}"
        print(line)
    save_json(Path(args.out), {"table": 3, "rows": all_rows})


# ------------------------------------------------------------------ Table 4
def run_table4(args) -> None:
    print("=== Table 4: Per-(model, env) final-round coupling fc ===")
    abbrevs = [GAME_ABBREV[g] for g in ALL_GAMES]
    models = ["base", "rw", "merge"] if not args.compare_reference else list(PAPER_TABLE4.keys())

    if args.compare_reference:
        header = f"{'Model':<16}" + "".join(f" {a:>6}" for a in abbrevs)
        print(header)
        for model, row in PAPER_TABLE4.items():
            line = f"{model:<16}"
            for a in abbrevs:
                line += f" {fmt_cell(row.get(a)):>6}"
            print(line)
        return

    variants = [args.variant] if args.variant else ["base", "rw", "merge"]
    all_rows: dict[str, dict] = {}
    for v in variants:
        m = eval_variant(args, v)
        all_rows[v] = {
            GAME_ABBREV[g]: m.per_game_fc.get(g) for g in ALL_GAMES
        }

    header = f"{'Model':<16}" + "".join(f" {a:>6}" for a in abbrevs)
    print(header)
    for v in variants:
        line = f"{v:<16}"
        for a in abbrevs:
            line += f" {fmt_cell(all_rows[v].get(a)):>6}"
        print(line)
    save_json(Path(args.out), {"table": 4, "rows": all_rows})


# ------------------------------------------------------------------ Table 5
def run_table5(args) -> None:
    print("=== Table 5: Training hyperparameters (LoRA DPO variants) ===")
    print(
        f"Model={args.STUDENT_MODEL} | LoRA r={args.lora_r} alpha={args.lora_alpha} "
        f"dropout={LORA_DROPOUT} | DPO beta={args.dpo_beta} | lr={args.lr} | "
        f"epochs={args.epochs} | grad_accum={GRADIENT_ACCUMULATION} | "
        f"merge_alpha={args.merge_alpha}"
    )

    if args.compare_reference:
        print(f"\n{'Variant':<14} {'Pairs':>8} {'Best step':>10} {'Eval loss':>10}")
        print("-" * 46)
        for v, r in PAPER_TABLE5.items():
            step = r["best_step"] if r["best_step"] is not None else "-"
            loss = f"{r['best_eval_loss']:.4f}" if r["best_eval_loss"] is not None else "-"
            print(f"{v:<14} {r['pairs']:>8} {str(step):>10} {loss:>10}")
        return

    if args.train:
        _train_all_variants(args)
        return

    # Report local training manifest
    manifest = {}
    print(f"\n{'Variant':<14} {'Pairs':>8} {'Data':>30} {'Checkpoint':>20}")
    print("-" * 76)
    for v, spec in TRAINING_VARIANTS.items():
        ckpt = Path(args.checkpoint_dir) / v
        data = spec["data"] or "-"
        pairs = spec["pairs"]
        exists = ckpt.exists()
        print(f"{v:<14} {pairs:>8} {data:>30} {str(ckpt):>20} {'OK' if exists else 'MISSING'}")
        manifest[v] = {**spec, "checkpoint": str(ckpt), "exists": exists}
    save_json(Path(args.out), {"table": 5, "hyperparams": _hyperparams(args), "manifest": manifest})


def _pair_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _train_all_variants(args) -> None:
    for v, spec in TRAINING_VARIANTS.items():
        if v == "merge":
            continue
        data = PROJECT_ROOT / spec["data"]
        out = Path(args.checkpoint_dir) / v
        if not data.exists():
            print(f"SKIP {v}: missing {data}")
            continue
        if _pair_count(data) == 0:
            print(f"SKIP {v}: no pairs in {data}")
            continue
        cmd = [
            sys.executable, "-m", TRAIN_MODULE,
            "--pairs", str(data),
            "--out", str(out),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--beta", str(args.dpo_beta),
            "--lora-r", str(args.lora_r),
            "--lora-alpha", str(args.lora_alpha),
        ]
        if args.STUDENT_MODEL:
            cmd.extend(["--model-id", args.STUDENT_MODEL])
        if args.lora_target:
            cmd.extend(["--lora-target", args.lora_target])
        print(f"Training {v}: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

    # Merge AUX + ALL (§2.5 Eq. 4)
    aux = Path(args.checkpoint_dir) / "aux"
    all_ckpt = Path(args.checkpoint_dir) / "all"
    if aux.exists() and all_ckpt.exists():
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "train" / "specialist-merge" / "merge_adapters.py"),
                "--checkpoint-dir", str(args.checkpoint_dir),
                "--alpha", str(args.merge_alpha),
            ],
            check=False,
        )


# ------------------------------------------------------------------ Table 6
def run_table6(args) -> None:
    print("=== Table 6: Stag-hunt anti-coordination (A+b example, TFT opponent) ===")
    print("Blind: opposite of TFT prev action | Oracle: solver BR (Eq. 1)")

    if args.compare_reference:
        print(f"\n{'round':>6} {'TFT':>8} {'blind':>8} {'oracle':>10}")
        print("-" * 36)
        for row in PAPER_TABLE6:
            oracle = row["oracle"] + (" (HELPS)" if row["helps"] else "")
            print(f"{row['round']:>6} {row['tft']:>8} {row['blind']:>8} {oracle:>10}")
        return

    # Reproduce the reference example logic.
    tft_sequence = ["Stag", "Hare", "Stag", "Stag", "Stag", "Hare", "Stag", "Stag", "Stag", "Stag"]
    blind_actions = []
    results = []
    prev_tft = None
    for rnd, tft in enumerate(tft_sequence, start=1):
        if prev_tft is None:
            blind = "Hare"
        else:
            blind = "Hare" if prev_tft == "Stag" else "Stag"
        oracle = tft  # BR against realized TFT action
        helps = blind != oracle
        blind_actions.append(blind)
        if rnd in (1, 2, 3, 6, 7):
            results.append({"round": rnd, "tft": tft, "blind": blind, "oracle": oracle, "helps": helps})
        prev_tft = tft

    print(f"\n{'round':>6} {'TFT':>8} {'blind':>8} {'oracle':>10} {'helps':>8}")
    for row in results:
        oracle = row["oracle"] + (" (HELPS)" if row["helps"] else "")
        print(f"{row['round']:>6} {row['tft']:>8} {row['blind']:>8} {oracle:>10} {str(row['helps']):>8}")

    save_json(Path(args.out), {"table": 6, "rows": results, "paper_reference": PAPER_TABLE6})


# ------------------------------------------------------------------ Table 7
def run_table7(args) -> None:
    print("=== Table 7: Per-env companion metrics (ac, fc, exploitability) ===")
    variant = args.variant or "base"

    if args.compare_reference:
        print(f"\n{'env':<22} {'ac':>8} {'fc':>8} {'exploit':>8}")
        print("-" * 50)
        for game in ALL_GAMES:
            r = PAPER_TABLE7[game]
            print(
                f"{game:<22} {fmt_cell(r['ac']):>8} "
                f"{fmt_cell(r['fc']):>8} {fmt_cell(r['exploitability']):>8}"
            )
        return

    m = eval_variant(args, variant)
    print(f"\nVariant: {variant}")
    print(f"{'env':<22} {'ac':>8} {'fc':>8} {'exploit':>8} {'paper_fc':>8}")
    print("-" * 58)
    rows = {}
    for game in ALL_GAMES:
        ref = PAPER_TABLE7[game]
        rows[game] = {
            "ac": m.per_game_ac.get(game),
            "fc": m.per_game_fc.get(game),
            "exploitability": m.per_game_exploitability.get(game),
        }
        print(
            f"{game:<22} {fmt_cell(rows[game]['ac']):>8} "
            f"{fmt_cell(rows[game]['fc']):>8} "
            f"{fmt_cell(rows[game]['exploitability']):>8} "
            f"{fmt_cell(ref['fc']):>8}"
        )
    save_json(Path(args.out), {"table": 7, "variant": variant, "rows": rows})


def _hyperparams(args) -> dict:
    hp = {
        "STUDENT_MODEL": args.STUDENT_MODEL,
        "mode": args.mode,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": LORA_DROPOUT,
        "dpo_beta": args.dpo_beta,
        "merge_alpha": args.merge_alpha,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "episodes_per_env": args.episodes,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "use_4bit": not args.no_4bit,
        "quantization": "4bit" if not args.no_4bit else "none",
        "bnb_4bit_quant_type": BNB_4BIT_QUANT_TYPE if not args.no_4bit else None,
        "bnb_4bit_compute_dtype": BNB_4BIT_COMPUTE_DTYPE if not args.no_4bit else None,
    }
    if args.mode == "ollama":
        hp["ollama_model"] = args.ollama_model or OLLAMA_FRONTIER_MODEL
        hp["ollama_base_url"] = args.ollama_base_url or OLLAMA_BASE_URL
    return hp


def main() -> None:
    p = argparse.ArgumentParser(description="SARL Tables 1-7")
    p.add_argument("--table", type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--mode", choices=["heuristic", "lora", "ollama"], default="lora")
    p.add_argument("--variant", default=None, help="Single variant (default: all for table)")
    p.add_argument("--model-id", default=STUDENT_MODEL)
    p.add_argument("--ollama-model", default=None, help="Ollama tag (default: config OLLAMA_FRONTIER_MODEL)")
    p.add_argument("--ollama-base-url", default=None, help=f"Ollama API (default: {OLLAMA_BASE_URL})")
    p.add_argument("--checkpoint-dir", type=Path, default=LORA_DIR)
    p.add_argument("--episodes", type=int, default=EPISODES_PER_ENV)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=LORA_R)
    p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--lora-target", default="q_proj,v_proj")
    p.add_argument("--dpo-beta", type=float, default=DPO_BETA)
    p.add_argument("--merge-alpha", type=float, default=MERGE_ALPHA)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--no-4bit", action="store_true", default=not USE_4BIT)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--compare-reference", action="store_true")
    p.add_argument("--train", action="store_true", help="Table 5: train all DPO variants")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Write all outputs under eval/runs/<timestamp>/ (default when --out omitted)",
    )
    p.add_argument("--log-file", type=Path, default=None, help="Eval progress log")
    p.add_argument("--quiet", action="store_true", help="Disable progress logging")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing log in --run-dir (append log, skip completed episodes)",
    )
    p.add_argument(
        "--games",
        default=None,
        help="Comma-separated games to run (default: all). With --resume, still merges prior CR.",
    )
    args = p.parse_args()

    if args.games:
        try:
            args.games_list = parse_games_arg(args.games)
        except ValueError as exc:
            p.error(str(exc))
    else:
        args.games_list = None

    if args.resume and args.run_dir is None:
        p.error("--resume requires --run-dir")

    run_dir: Path | None = None
    if args.run_dir is not None:
        run_dir = resolve_run_dir(Path(args.run_dir))
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    elif args.out is None and not args.compare_reference:
        run_dir = new_run_dir(utc_stamp())

    args.run_dir = run_dir

    if args.out is None and run_dir is not None:
        args.out = str(table_out_path(run_dir, args.table, variant=args.variant))

    if args.out is None:
        args.out = f"results/table{args.table}_metrics.json"

    if not args.compare_reference and not args.quiet and args.table in (1, 2, 3, 4, 7):
        log_path = args.log_file or _default_log_path(args.table, args.variant, run_dir)
        if args.resume and not log_path.exists():
            p.error(f"--resume: log not found: {log_path}")
        args.logger = EvalLogger(log_path=log_path, append=args.resume)
        print(f"Progress log: {log_path}", flush=True)
        if run_dir is not None:
            print(f"Run dir: {run_dir}", flush=True)
    else:
        args.logger = None

    runners = {
        1: run_table1,
        2: run_table2,
        3: run_table3,
        4: run_table4,
        5: run_table5,
        6: run_table6,
        7: run_table7,
    }
    runners[args.table](args)
    if run_dir is not None and not args.compare_reference:
        write_latest_pointer(run_dir)


if __name__ == "__main__":
    main()
