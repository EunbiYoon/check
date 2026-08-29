"""Command-line orchestration for LoRA DPO training."""

from __future__ import annotations

import train.dpo_lora.gpu_env  # noqa: F401 — set BNB_CUDA_VERSION before HF/bnb import

import argparse
import importlib.util
import json
import sys
from pathlib import Path

def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config.py").is_file():
            return parent
    raise FileNotFoundError("cannot locate the repository root (no config.py above this file)")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from history.paths import ensure_session, lora_variant_dir

from .lifecycle import (
    build_config_snapshot,
    finalize_run_manifest,
    find_resume_for_publish,
    infer_variant,
    persist_run_manifest,
    publish_adapter,
    resolve_checkpoint_path,
    resolve_run_dirs,
    write_latest_pointer,
)
from .utils import build_lora_config


def _load_module(name: str, *relative_paths: str):
    """Load an A+β pair-construction sibling module by file path.

    The A+β construction packages live in hyphenated directories that cannot be
    imported normally, and their location has moved across refactors, so each
    candidate relative path is tried in turn and, failing that, the tree is
    searched for the leaf filename.
    """
    candidates = [ROOT / rel for rel in relative_paths]
    target = next((path for path in candidates if path.is_file()), None)
    if target is None:
        parts = Path(relative_paths[0]).parts
        leaf = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        target = next(ROOT.glob(f"**/{leaf}"), None)
    if target is None:
        raise FileNotFoundError(
            f"cannot locate {relative_paths[0]} (tried: "
            + ", ".join(str(c) for c in candidates)
            + ")"
        )
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


frontier = _load_module(
    "noleakage_frontier",
    "data/alpha-beta/noleakage-frontier/build_pairs.py",
    "data-construction/alpha-beta/noleakage-frontier/build_pairs.py",
)
build_dpo_trainer = frontier.build_dpo_trainer
extract_train_metrics = frontier.extract_train_metrics
prepare_model = frontier.prepare_model


pair_data = _load_module(
    "solver_pinned_pair_data",
    "data/alpha-beta/solver-pinned/update_pairs.py",
    "data-construction/alpha-beta/solver-pinned/update_pairs.py",
)
load_pairs = pair_data.load_pairs


def _resolve_train_args(args: argparse.Namespace) -> None:
    args.STUDENT_MODEL = args.STUDENT_MODEL or config.PAPER_STUDENT_MODEL
    args.lora_r = config.PAPER_LORA_R if args.lora_r is None else args.lora_r
    args.lora_alpha = config.PAPER_LORA_ALPHA if args.lora_alpha is None else args.lora_alpha
    args.lora_target = args.lora_target or config.PAPER_LORA_TARGET_MODULES
    if args.out in ("lora/all", "lora_3b/all"):
        args.out = "all"
    if args.max_steps is None:
        args.max_steps = config.max_train_steps_for_variant(infer_variant(Path(args.out)))
    if args.max_length is None:
        args.max_length = config.MAX_SEQ_LENGTH
    if args.grad_accum is None:
        args.grad_accum = config.GRADIENT_ACCUMULATION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=None, help="JSONL with prompt/chosen/rejected")
    parser.add_argument("--trajectories", type=Path, default=None)
    parser.add_argument("--rebuild-pairs", action="store_true")
    parser.add_argument("--teacher-model", default=None, help="Frontier paraphraser; defaults to TEACHER_MODEL")
    parser.add_argument("--teacher-max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--allow-unverified-frontier-model",
        action="store_true",
        help="accept legacy trajectories lacking Algorithm 1 model provenance",
    )
    parser.add_argument(
        "--counterfactual-mode",
        choices=("horizon-aware", "fixed"),
        default="horizon-aware",
        help="Algorithm 1 flag h: Eq. 3 when horizon-aware, Eq. 2 when fixed",
    )
    parser.add_argument("--out", default="lora/all", help="Variant publish dir (e.g. lora/all)")
    parser.add_argument(
        "--model-id",
        dest="STUDENT_MODEL",
        default=None,
        help="Base HF model (default: config.STUDENT_MODEL)",
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Stop after N optimizer steps (overrides --epochs when set). "
            "Defaults to TRAIN_MAX_STEPS_<VARIANT>, then TRAIN_MAX_STEPS."
        ),
    )
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=config.DPO_BETA)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument(
        "--lora-target",
        default=None,
        help="LoRA target modules, e.g. q_proj,v_proj or all-linear",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=f"Max tokens per DPO example (default: {config.MAX_SEQ_LENGTH}).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help=f"Gradient accumulation steps (default: {config.GRADIENT_ACCUMULATION}).",
    )
    parser.add_argument(
        "--no-timestamp-out",
        action="store_true",
        help="Write directly to --out (legacy). Default: runs/<session>/lora/<variant>/",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Log train loss to TensorBoard under <run_dir>/tensorboard/",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help=f"Max checkpoints to keep (default: {config.PAPER_SAVE_TOTAL_LIMIT})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume latest unfinished checkpoint for --out variant (under runs/<session>/lora/)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume from explicit checkpoint dir (e.g. runs/<id>/lora/all/adapter/checkpoint-500)",
    )
    args = parser.parse_args()
    _resolve_train_args(args)
    if args.resume and args.resume_from_checkpoint:
        parser.error("Use only one of --resume or --resume-from-checkpoint")

    publish_dir = Path(args.out)
    if not publish_dir.is_absolute():
        publish_dir = config.PROJECT_ROOT / publish_dir
    variant_name = infer_variant(publish_dir)
    if not args.no_timestamp_out:
        ensure_session()
        publish_dir = lora_variant_dir(variant_name)

    resume_triple: tuple[Path, Path, Path] | None = None
    if args.resume_from_checkpoint is not None:
        resume_triple = resolve_checkpoint_path(args.resume_from_checkpoint)
    elif args.resume:
        resume_triple = find_resume_for_publish(publish_dir)
        if resume_triple is None:
            parser.error(f"--resume: no unfinished checkpoint found for {publish_dir}")

    use_tensorboard = not args.no_tensorboard
    save_total_limit = (
        args.save_total_limit
        if args.save_total_limit is not None
        else config.PAPER_SAVE_TOTAL_LIMIT
    )

    STUDENT_MODEL = args.STUDENT_MODEL or config.STUDENT_MODEL
    lora_r = args.lora_r if args.lora_r is not None else config.LORA_R
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else config.LORA_ALPHA
    lora_target = args.lora_target or ",".join(config.LORA_TARGET_MODULES)
    target_modules = (
        lora_target if lora_target == "all-linear" else lora_target.split(",")
    )
    lora_cfg = build_lora_config(r=lora_r, alpha=lora_alpha, target_modules=target_modules)

    if bool(args.pairs) == bool(args.trajectories):
        parser.error("use exactly one of --pairs or --trajectories")

    run_dir, adapter_dir, publish_dir = resolve_run_dirs(
        args, publish_dir=publish_dir, resume=resume_triple
    )
    resume_ckpt = resume_triple[2] if resume_triple else None
    if args.trajectories is not None:
        trajectories_path = args.trajectories
        if not trajectories_path.is_absolute():
            trajectories_path = config.PROJECT_ROOT / trajectories_path
        pairs_path = run_dir / "constructed_pairs.jsonl"
        if args.rebuild_pairs or not pairs_path.is_file():
            count = frontier.build_file(
                trajectories_path,
                pairs_path,
                provider="teacher",
                teacher_STUDENT_MODEL=args.teacher_model or config.TEACHER_MODEL,
                teacher_max_new_tokens=args.teacher_max_new_tokens,
                counterfactual_mode=args.counterfactual_mode,
                require_frontier_model_match=not args.allow_unverified_frontier_model,
            )
            print(f"Constructed {count} verified pairs from {trajectories_path}", flush=True)
            # Release the frontier model before loading the DPO student.
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        pairs_path = Path(args.pairs)
        if not pairs_path.is_absolute():
            pairs_path = config.PROJECT_ROOT / pairs_path

    model, tokenizer = prepare_model(
        STUDENT_MODEL=STUDENT_MODEL,
        lora_config=lora_cfg,
        resume_checkpoint=resume_ckpt,
    )

    dataset = load_pairs(pairs_path)
    variant = infer_variant(publish_dir)
    snapshot = build_config_snapshot(
        variant=variant,
        pairs_path=pairs_path,
        pairs_count=len(dataset),
        STUDENT_MODEL=STUDENT_MODEL,
        paper=True,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target=lora_target,
        max_length=args.max_length,
        grad_accum=args.grad_accum,
        publish_dir=publish_dir,
    )
    if resume_ckpt is None:
        persist_run_manifest(run_dir, argv=sys.argv, args=args, snapshot=snapshot)
    else:
        info = run_dir / "run_info.json"
        if info.is_file():
            manifest = json.loads(info.read_text(encoding="utf-8"))
            manifest["status"] = "running"
            manifest["resumed_from"] = str(
                resume_ckpt.relative_to(config.PROJECT_ROOT)
                if resume_ckpt.is_relative_to(config.PROJECT_ROOT)
                else resume_ckpt
            )
            manifest["resume_command"] = " ".join(sys.argv)
            info.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"DPO train model={STUDENT_MODEL} pairs={len(dataset)} "
        f"epochs={args.epochs} max_steps={args.max_steps} "
        f"max_length={args.max_length} grad_accum={args.grad_accum} "
        f"run_dir={run_dir} publish={publish_dir} "
        f"tensorboard={use_tensorboard} save_total_limit={save_total_limit} "
        f"resume={resume_ckpt}",
        flush=True,
    )
    tb_dir = run_dir / "tensorboard"
    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        args=args,
        adapter_dir=adapter_dir,
        tensorboard_dir=tb_dir,
        use_tensorboard=use_tensorboard,
        save_total_limit=save_total_limit,
    )
    status = "completed"
    train_metrics: dict | None = None
    try:
        trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
        trainer.save_model(str(adapter_dir))
        train_metrics = extract_train_metrics(trainer)
    except Exception:
        status = "failed"
        finalize_run_manifest(run_dir, status=status)
        raise
    else:
        publish_adapter(adapter_dir, publish_dir)
        finalize_run_manifest(run_dir, status=status, train_metrics=train_metrics)
        if not args.no_timestamp_out:
            write_latest_pointer(run_dir)

    print(
        f"Saved LoRA adapter run={run_dir} publish={publish_dir} (base={STUDENT_MODEL})",
        flush=True,
    )
    if use_tensorboard:
        print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}", flush=True)
    print(
        f"Checkpoints: {adapter_dir}/checkpoint-* (every {config.SAVE_STEPS} steps, "
        f"keep last {save_total_limit})",
        flush=True,
    )


if __name__ == "__main__":
    main()
