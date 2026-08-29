"""Argument parsing and default resolution for single-variant training."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from lifecycle import infer_variant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one LoRA/DPO variant.")
    parser.add_argument("--pairs", default=None, help="JSONL with prompt/chosen/rejected")
    parser.add_argument("--trajectories", type=Path, default=None)
    parser.add_argument("--rebuild-pairs", action="store_true")
    parser.add_argument("--teacher-model", default=None, help="Frontier paraphraser; defaults to TEACHER_MODEL")
    parser.add_argument("--teacher-max-new-tokens", type=int, default=1024)
    parser.add_argument("--allow-unverified-frontier-model", action="store_true", help="accept legacy trajectories lacking Algorithm 1 model provenance")
    parser.add_argument("--counterfactual-mode", choices=("horizon-aware", "fixed"), default="horizon-aware", help="Algorithm 1 flag h: Eq. 3 when horizon-aware, Eq. 2 when fixed")
    parser.add_argument("--out", default="lora/all", help="Variant publish dir (e.g. lora/all)")
    parser.add_argument("--model-id", dest="student_model", default=None, help="Base HF model (default: config.PAPER_STUDENT_MODEL)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps; defaults to the configured variant limit.")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=config.DPO_BETA)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-target", default=None, help="LoRA modules or all-linear")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--no-timestamp-out", action="store_true", help="Write directly to --out instead of runs/<session>/lora/<variant>/")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume the latest unfinished run")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.student_model = args.student_model or config.PAPER_STUDENT_MODEL
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
    if args.resume and args.resume_from_checkpoint:
        parser.error("Use only one of --resume or --resume-from-checkpoint")
    if bool(args.pairs) == bool(args.trajectories):
        parser.error("use exactly one of --pairs or --trajectories")
    return args
