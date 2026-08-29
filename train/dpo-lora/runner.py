"""Single-variant DPO training workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "config.py").is_file())
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config
from history.paths import ensure_session, lora_variant_dir
from lifecycle import (build_config_snapshot, finalize_run_manifest, find_resume_for_publish, infer_variant, persist_run_manifest, publish_adapter, resolve_checkpoint_path, resolve_run_dirs, write_latest_pointer)
from args import parse_train_args
from data import build_dpo_trainer, extract_train_metrics, prepare_model, prepare_pairs
from utils import build_lora_config


def _resolve_output(args):
    publish_dir = Path(args.out)
    if not publish_dir.is_absolute():
        publish_dir = config.PROJECT_ROOT / publish_dir
    if not args.no_timestamp_out:
        ensure_session()
        publish_dir = lora_variant_dir(infer_variant(publish_dir))
    resume = None
    if args.resume_from_checkpoint is not None:
        resume = resolve_checkpoint_path(args.resume_from_checkpoint)
    elif args.resume:
        resume = find_resume_for_publish(publish_dir)
        if resume is None:
            raise SystemExit(f"--resume: no unfinished checkpoint found for {publish_dir}")
    return publish_dir, resume


def _mark_resumed(run_dir: Path, resume_ckpt: Path) -> None:
    info = run_dir / "run_info.json"
    if not info.is_file():
        return
    manifest = json.loads(info.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["resumed_from"] = str(resume_ckpt.relative_to(config.PROJECT_ROOT) if resume_ckpt.is_relative_to(config.PROJECT_ROOT) else resume_ckpt)
    manifest["resume_command"] = " ".join(sys.argv)
    info.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_training(argv: list[str] | None = None) -> None:
    args = parse_train_args(argv)
    publish_dir, resume = _resolve_output(args)
    run_dir, adapter_dir, publish_dir = resolve_run_dirs(args, publish_dir=publish_dir, resume=resume)
    resume_ckpt = resume[2] if resume else None
    pairs_path, dataset = prepare_pairs(args, run_dir)
    student_model = args.student_model
    lora_target = args.lora_target
    target_modules = lora_target if lora_target == "all-linear" else lora_target.split(",")
    lora_config = build_lora_config(r=args.lora_r, alpha=args.lora_alpha, target_modules=target_modules)
    model, tokenizer = prepare_model(STUDENT_MODEL=student_model, lora_config=lora_config, resume_checkpoint=resume_ckpt)
    snapshot = build_config_snapshot(
        variant=infer_variant(publish_dir), pairs_path=pairs_path, pairs_count=len(dataset),
        STUDENT_MODEL=student_model, paper=True, epochs=args.epochs, lr=args.lr,
        beta=args.beta, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_target=lora_target, max_length=args.max_length, grad_accum=args.grad_accum,
        publish_dir=publish_dir,
    )
    if resume_ckpt is None:
        persist_run_manifest(run_dir, argv=sys.argv if argv is None else argv, args=args, snapshot=snapshot)
    else:
        _mark_resumed(run_dir, resume_ckpt)
    use_tensorboard = not args.no_tensorboard
    save_total_limit = args.save_total_limit if args.save_total_limit is not None else config.PAPER_SAVE_TOTAL_LIMIT
    print(f"DPO train model={student_model} pairs={len(dataset)} epochs={args.epochs} max_steps={args.max_steps} max_length={args.max_length} grad_accum={args.grad_accum} run_dir={run_dir} publish={publish_dir} tensorboard={use_tensorboard} save_total_limit={save_total_limit} resume={resume_ckpt}", flush=True)
    trainer = build_dpo_trainer(model=model, tokenizer=tokenizer, dataset=dataset, args=args, adapter_dir=adapter_dir, tensorboard_dir=run_dir / "tensorboard", use_tensorboard=use_tensorboard, save_total_limit=save_total_limit)
    try:
        trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
        trainer.save_model(str(adapter_dir))
        metrics = extract_train_metrics(trainer)
    except Exception:
        finalize_run_manifest(run_dir, status="failed")
        raise
    publish_adapter(adapter_dir, publish_dir)
    finalize_run_manifest(run_dir, status="completed", train_metrics=metrics)
    if not args.no_timestamp_out:
        write_latest_pointer(run_dir)
    print(f"Saved LoRA adapter run={run_dir} publish={publish_dir} (base={student_model})", flush=True)
    if use_tensorboard:
        print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}", flush=True)
    print(f"Checkpoints: {adapter_dir}/checkpoint-* (every {config.SAVE_STEPS} steps, keep last {save_total_limit})", flush=True)
