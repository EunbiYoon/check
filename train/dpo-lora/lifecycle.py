"""Training-run paths, manifests, checkpoint cleanup, and resume discovery."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import TrainerCallback
import yaml

import config
from history.paths import ensure_session, new_lora_variant_dir, write_latest_pointer as write_session_latest

_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
}


def infer_variant(publish_dir: Path) -> str:
    """Return the variant component of a training output path."""
    return publish_dir.name


def write_latest_pointer(run_dir: Path) -> None:
    """Point runs/latest at the session containing ``run_dir``."""
    write_session_latest(run_dir.parent.parent.name)


def _is_checkpoint_file(path: Path) -> bool:
    """Keep the files required to resume a run; drop everything else."""
    return path.name in _CHECKPOINT_FILES


class MinimalCheckpointCallback(TrainerCallback):
    """Trim checkpoints while retaining the files required for resume."""

    def on_save(self, args, state, control, **kwargs):
        if not args.should_save:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.is_dir():
            return control
        for path in checkpoint_dir.iterdir():
            if _is_checkpoint_file(path):
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return control


def _latest_checkpoint(adapter_dir: Path) -> Path | None:
    checkpoints = sorted(
        adapter_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def _publish_dir_matches(manifest: dict, publish_dir: Path) -> bool:
    variant = infer_variant(publish_dir)
    snapshot = manifest.get("resolved_config") or {}
    if snapshot.get("variant") == variant:
        return True
    published = manifest.get("publish_dir") or snapshot.get("publish_dir") or ""
    try:
        return Path(published).resolve() == publish_dir.resolve()
    except OSError:
        return str(published).endswith(f"/{variant}") or str(published).endswith(variant)


def find_resume_for_publish(publish_dir: Path) -> tuple[Path, Path, Path] | None:
    """Return (run_dir, adapter_dir, checkpoint_dir) for the latest unfinished run."""
    best: tuple[float, Path, Path, Path] | None = None
    if not config.RUNS_DIR.is_dir():
        return None
    session_filter = os.environ.get("RUN_ID")
    sessions = (
        [config.RUNS_DIR / session_filter]
        if session_filter
        else sorted(path for path in config.RUNS_DIR.iterdir() if path.is_dir())
    )
    variant = infer_variant(publish_dir)
    for session in sessions:
        run_dir = session / "lora" / variant
        info_path = run_dir / "run_info.json"
        adapter_dir = run_dir / "adapter"
        if not info_path.is_file() or not adapter_dir.is_dir():
            continue
        manifest = json.loads(info_path.read_text(encoding="utf-8"))
        checkpoint = _latest_checkpoint(adapter_dir)
        if (
            not _publish_dir_matches(manifest, publish_dir)
            or checkpoint is None
            or manifest.get("status") == "completed"
        ):
            continue
        candidate = (checkpoint.stat().st_mtime, run_dir, adapter_dir, checkpoint)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return None if best is None else best[1:]


def resolve_checkpoint_path(path: Path) -> tuple[Path, Path, Path]:
    """Resolve a supplied path to (run_dir, adapter_dir, checkpoint_dir)."""
    path = path.resolve()
    if path.name.startswith("checkpoint-"):
        return path.parent.parent, path.parent, path
    if (path / "trainer_state.json").is_file():
        return path.parent.parent, path.parent, path
    if path.is_dir() and (path / "adapter_config.json").is_file():
        checkpoint = _latest_checkpoint(path)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint-* under {path}")
        return path.parent, path, checkpoint
    raise FileNotFoundError(f"Not a checkpoint path: {path}")


def resolve_run_dirs(
    args,
    *,
    publish_dir: Path,
    resume: tuple[Path, Path, Path] | None,
) -> tuple[Path, Path, Path]:
    """Return (run_dir, adapter_dir, publish_dir)."""
    if resume is not None:
        run_dir, adapter_dir, _checkpoint = resume
        return run_dir, adapter_dir, run_dir
    if args.no_timestamp_out:
        publish_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = publish_dir / "adapter"
        if not adapter_dir.is_dir():
            adapter_dir = publish_dir
        return publish_dir, adapter_dir, publish_dir
    ensure_session()
    run_dir = new_lora_variant_dir(infer_variant(publish_dir))
    return run_dir, run_dir / "adapter", run_dir


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_path(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(config.PROJECT_ROOT).as_posix() if path.is_relative_to(config.PROJECT_ROOT) else str(path)


def build_config_snapshot(
    *, variant: str, pairs_path: Path, pairs_count: int, STUDENT_MODEL: str,
    paper: bool, epochs: int, lr: float, beta: float, lora_r: int,
    lora_alpha: int, lora_target: str, max_length: int, grad_accum: int,
    publish_dir: Path,
) -> dict[str, Any]:
    """Capture the resolved settings and provenance for a training run."""
    pairs_path = pairs_path.resolve()
    snapshot: dict[str, Any] = {
        "variant": variant, "pairs": _project_path(pairs_path),
        "pairs_count": pairs_count, "STUDENT_MODEL": STUDENT_MODEL,
        "student_model": STUDENT_MODEL or config.STUDENT_MODEL,
        "teacher_model": config.TEACHER_MODEL, "paper": paper, "epochs": epochs,
        "learning_rate": lr, "dpo_beta": beta, "lora_r": lora_r,
        "lora_alpha": lora_alpha, "lora_target": lora_target,
        "max_length": max_length, "gradient_accumulation_steps": grad_accum,
        "per_device_train_batch_size": config.PER_DEVICE_BATCH,
        "use_4bit": config.USE_4BIT,
        "quantization": "4bit" if config.USE_4BIT else "none",
        "bnb_4bit_quant_type": config.BNB_4BIT_QUANT_TYPE if config.USE_4BIT else None,
        "bnb_4bit_compute_dtype": config.BNB_4BIT_COMPUTE_DTYPE if config.USE_4BIT else None,
        "fp16": config.TRAIN_FP16, "bf16": config.TRAIN_BF16,
        "publish_dir": _project_path(publish_dir),
    }
    data_manifest = pairs_path.parent / "latest_manifest.json"
    if data_manifest.is_file():
        snapshot["pairs_data_manifest"] = _project_path(data_manifest)
    dpo_run = _infer_dpo_run_dir(pairs_path)
    if dpo_run is not None:
        snapshot["dpo_source_run"] = _project_path(dpo_run)
        dpo_snapshot = dpo_run / "config_snapshot.yaml"
        if dpo_snapshot.is_file():
            snapshot["dpo_config_snapshot"] = _project_path(dpo_snapshot)
    return snapshot


def _infer_dpo_run_dir(pairs_path: Path) -> Path | None:
    """Map a named DPO data run to its run directory when present."""
    if pairs_path.parent.name == "data":
        return None
    candidate = config.PROJECT_ROOT / "dpo" / "runs" / pairs_path.parent.name
    return candidate if candidate.is_dir() else None


def persist_run_manifest(
    run_dir: Path, *, argv: list[str] | None, args: Namespace,
    snapshot: dict[str, Any],
) -> None:
    """Write the initial config snapshot, pair sources, and run manifest."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd_argv = list(argv if argv is not None else sys.argv)
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    pairs_path = Path(snapshot["pairs"])
    if not pairs_path.is_absolute():
        pairs_path = config.PROJECT_ROOT / pairs_path
    if pairs_path.is_file():
        shutil.copy2(pairs_path, run_dir / "pairs_source.jsonl")
    data_manifest = pairs_path.parent / "latest_manifest.json"
    if data_manifest.is_file():
        shutil.copy2(data_manifest, run_dir / "pairs_manifest.json")
    publish_dir = Path(snapshot["publish_dir"])
    if not publish_dir.is_absolute():
        publish_dir = config.PROJECT_ROOT / publish_dir
    manifest: dict[str, Any] = {
        "started_at": utc_now_iso(), "finished_at": None, "status": "running",
        "command": shlex.join(cmd_argv), "argv": cmd_argv,
        "config_snapshot": "config_snapshot.yaml", "run_dir": str(run_dir.resolve()),
        "adapter_dir": str((run_dir / "adapter").resolve()),
        "publish_dir": str(publish_dir.resolve()),
        "timestamped_out": not getattr(args, "no_timestamp_out", False),
        "cli": {"pairs": snapshot["pairs"], "out": snapshot["publish_dir"],
                "paper": snapshot["paper"], "epochs": snapshot["epochs"],
                "STUDENT_MODEL": snapshot["STUDENT_MODEL"], "max_length": snapshot["max_length"],
                "grad_accum": snapshot["gradient_accumulation_steps"]},
        "resolved_config": snapshot, "train_metrics": None,
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def finalize_run_manifest(
    run_dir: Path, *, status: str = "completed",
    train_metrics: dict[str, Any] | None = None,
) -> None:
    """Record the terminal state and metrics for a run."""
    path = run_dir / "run_info.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["finished_at"] = utc_now_iso()
    manifest["status"] = status
    if train_metrics is not None:
        manifest["train_metrics"] = train_metrics
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def publish_adapter(adapter_dir: Path, publish_dir: Path) -> None:
    """Copy final adapter files to the variant directory, excluding checkpoints."""
    publish_dir.mkdir(parents=True, exist_ok=True)
    for item in adapter_dir.iterdir():
        if item.name.startswith("checkpoint-"):
            continue
        destination = publish_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
