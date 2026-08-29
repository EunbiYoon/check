"""Human-readable run titles for result.md and latex.md."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from config import (
    EPISODES_PER_ENV,
    LORA_ALPHA,
    LORA_R,
    STUDENT_MODEL,
    NUM_EPOCHS,
    PAPER_LORA_ALPHA,
    PAPER_LORA_R,
    PAPER_STUDENT_MODEL,
    RUNS_DIR,
)


def model_size_label(STUDENT_MODEL: str | None) -> str:
    if not STUDENT_MODEL:
        return "?"
    mid = STUDENT_MODEL.lower()
    if "0.5b" in mid:
        return "0.5B"
    if "3b" in mid:
        return "3B"
    if "7b" in mid:
        return "7B"
    return STUDENT_MODEL.rsplit("/", 1)[-1]


def infer_train_epochs(checkpoint_dir: Path | str | None) -> int | None:
    """Read max `epochs` from runs/*/lora/*/run_info.json for staged variants."""
    if checkpoint_dir is None:
        return None
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_dir():
        return None

    epochs: list[int] = []
    for variant_dir in ckpt.iterdir():
        if not variant_dir.is_dir():
            continue
        run_dir = variant_dir
        if (variant_dir / "adapter").is_symlink():
            run_dir = variant_dir / "adapter"
            run_dir = run_dir.resolve().parent
        elif (variant_dir / "run_info.json").is_file():
            run_dir = variant_dir
        elif (RUNS_DIR / variant_dir.name / "lora" / variant_dir.name / "run_info.json").is_file():
            run_dir = RUNS_DIR / variant_dir.name / "lora" / variant_dir.name
        else:
            continue
        info = run_dir / "run_info.json"
        if not info.is_file():
            continue
        manifest = json.loads(info.read_text(encoding="utf-8"))
        ep = manifest.get("resolved_config", {}).get("epochs")
        if ep is None:
            ep = manifest.get("cli", {}).get("epochs")
        if ep is not None:
            epochs.append(int(ep))

    return max(epochs) if epochs else None


def _train_epochs(hp: dict) -> int | str:
    env = os.environ.get("TRAIN_EPOCHS")
    if env:
        return int(env)
    if hp.get("train_epochs") is not None:
        return hp["train_epochs"]
    inferred = infer_train_epochs(hp.get("checkpoint_dir"))
    if inferred is not None:
        return inferred
    return hp.get("epochs", NUM_EPOCHS)


def build_run_title(hp: dict, *, kind: str = "local") -> str:
    """Return a one-line tag distinguishing this run from the reference.

    Override entirely: ``EVAL_TITLE=my label``
    Extra suffix: ``RUN_LABEL=pd-only``
    Train epochs: ``TRAIN_EPOCHS=1`` (if not in run_info)
    """
    if kind == "local":
        override = os.environ.get("EVAL_TITLE")
        if override:
            return override.strip()

    if kind == "paper":
        model = model_size_label(PAPER_STUDENT_MODEL)
        train_ep = NUM_EPOCHS
        eval_ep = EPISODES_PER_ENV
        r, alpha = PAPER_LORA_R, PAPER_LORA_ALPHA
        prefix = "Paper"
    else:
        STUDENT_MODEL = hp.get("STUDENT_MODEL") or STUDENT_MODEL
        model = model_size_label(STUDENT_MODEL)
        train_ep = _train_epochs(hp)
        eval_ep = hp.get("episodes_per_env", "?")
        r = hp.get("lora_r") if hp.get("lora_r") is not None else LORA_R
        alpha = hp.get("lora_alpha") if hp.get("lora_alpha") is not None else LORA_ALPHA
        prefix = "Local"

    extra = os.environ.get("RUN_LABEL", "").strip()
    title = f"[{prefix}] Qwen2.5-{model} · train_ep={train_ep} · eval_ep={eval_ep} · LoRA r={r} α={alpha}"
    if extra and kind == "local":
        title += f" · {extra}"
    return title


def build_run_heading(hp: dict, *, run_id: str, kind: str = "local") -> str:
    return f"# {build_run_title(hp, kind=kind)} — `{run_id}`"


def latex_run_note(hp: dict, *, kind: str = "local") -> str:
    note = build_run_title(hp, kind=kind)
    note = re.sub(r"\[|\]", "", note)
    return _tex_escape(note)


def _tex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = str(text)
    for ch, esc in repl.items():
        out = out.replace(ch, esc)
    return out
