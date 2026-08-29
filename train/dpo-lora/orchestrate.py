"""Multi-variant LoRA/DPO training orchestrator.

Trains one adapter per requested variant, one worker per GPU (variants are
stride-assigned across the workers), then optionally merges AUX+ALL
(paper §2.5 Eq. 4). Each variant is trained by ``variant.py``
subprocess -- this module only schedules them.

``train/dpo-lora/train.sh`` is the thin launcher: it sets up conda / CUDA / the
``.env`` file and then ``exec``s ``python train/dpo-lora/orchestrate.py``.
(``train/dpo-lora`` is a hyphenated dir, so it is run by path, not ``-m``.)

Config comes from CLI flags, falling back to environment variables (which the
launcher seeds from ``.env``):

    --variants        TRAIN_VARIANTS      filter_on,filter_off,core,aux,all,rw
    --num-gpus        TRAIN_NUM_GPUS      3
    --data-dir        DATA_DIR            data/alpha-beta/result
    --visible-devices TRAIN_VISIBLE_DEVICES (unset -> one GPU per worker)
    --run-id          RUN_ID             (unset -> new UTC-timestamp session)
    --no-merge        TRAIN_AUTO_MERGE=false
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config.py").is_file():
            return parent
    raise FileNotFoundError("cannot locate the repository root (no config.py)")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# RUN_ID from the real environment only; an unspecified run starts a new session,
# so a stale RUN_ID in .env must not be inherited.
_ENV_RUN_ID = os.environ.get("RUN_ID") or None


def _load_dotenv(root: Path) -> None:
    """Merge KEY=VALUE lines from .env, letting the real environment win."""
    path = root / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv(ROOT)

import config  # noqa: E402  (after sys.path + .env)

PAIR_FILE_BY_VARIANT = {
    # Hypothesis B pairs carry a "b_" prefix (data/b-hypothesis/result/), distinct
    # from the A+β blind-subset "filter_*.jsonl" trajectory files.
    "filter_on": "b_filter_on.jsonl",
    "filter_off": "b_filter_off.jsonl",
    "core": "a_beta_core.jsonl",
    "aux": "a_beta_aux.jsonl",
    "all": "a_beta_all.jsonl",
    "rw": "a_beta_rw.jsonl",
}

_PRINT_LOCK = threading.Lock()


def _log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# --------------------------------------------------------------------------- args
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    env = os.environ.get
    parser = argparse.ArgumentParser(prog="python train/dpo-lora/orchestrate.py",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants",
                        default=env("TRAIN_VARIANTS", "filter_on,filter_off,core,aux,all,rw"))
    parser.add_argument("--num-gpus", type=int, default=int(env("TRAIN_NUM_GPUS", "3")))
    parser.add_argument("--data-dir", default=env("DATA_DIR", "data/alpha-beta/result"))
    parser.add_argument("--visible-devices", default=env("TRAIN_VISIBLE_DEVICES", "") or None)
    parser.add_argument("--run-id", default=_ENV_RUN_ID)
    merge = parser.add_mutually_exclusive_group()
    merge.add_argument("--merge", dest="merge", action="store_true", default=None)
    merge.add_argument("--no-merge", dest="merge", action="store_false")
    args = parser.parse_args(argv)
    if args.merge is None:
        args.merge = env("TRAIN_AUTO_MERGE", "true").lower() != "false"
    return args


# ---------------------------------------------------------------- preflight
def _gpu_preflight() -> None:
    """Fail fast if CUDA / bitsandbytes 4-bit is unavailable (matches train.sh)."""
    check = (
        "import torch, bitsandbytes.cextension as e;"
        "assert torch.cuda.is_available(), 'torch.cuda is unavailable';"
        "lib = e.lib;"
        "assert lib is not None and getattr(lib, 'compiled_with_cuda', False),"
        " 'bitsandbytes CUDA is unavailable'"
    )
    result = subprocess.run([sys.executable, "-c", check], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit("ERROR: GPU preflight failed — run train.sh from a CUDA GPU allocation")


def _visible_gpu_count() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    return len([line for line in out.splitlines() if line.strip()])


# ---------------------------------------------------------------- one variant
def _variant_completed(run_info: Path) -> bool:
    try:
        return json.loads(run_info.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, ValueError):
        return False


def _train_variant(*, gpu: int, variant: str, args: argparse.Namespace,
                   session_dir: Path, log_dir: Path) -> int:
    visible = args.visible_devices or str(gpu)
    variant_dir = session_dir / "lora" / variant
    run_info = variant_dir / "run_info.json"

    if run_info.is_file() and _variant_completed(run_info):
        _log(f"[{_now()}] GPU {gpu} -> {variant}: already completed; skipping")
        return 0

    cmd = [sys.executable, str(Path(__file__).with_name("variant.py")),
           "--tensorboard", "--out", variant]

    cmd += ["--pairs", str(ROOT / args.data_dir / PAIR_FILE_BY_VARIANT[variant])]

    if list((variant_dir / "adapter").glob("checkpoint-*")):
        cmd.append("--resume")
        _log(f"[{_now()}] GPU {gpu} -> {variant}: resuming latest checkpoint")

    _log(f"[{_now()}] GPU {gpu} -> {variant}: {cmd[-1]}")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible)
    prefix = f"[GPU {visible}][{variant}] "
    log_path = log_dir / f"{variant}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            log_file.write(line + "\n")
            log_file.flush()
            _log(prefix + line)
        return proc.wait()


# ---------------------------------------------------------------- merge
def _auto_merge(session_dir: Path) -> None:
    merge_script = next(ROOT.glob("**/specialist-merge/merge_adapters.py"), None)
    aux = session_dir / "lora" / "aux" / "adapter_config.json"
    all_ = session_dir / "lora" / "all" / "adapter_config.json"
    if merge_script and aux.is_file() and all_.is_file():
        subprocess.run([sys.executable, str(merge_script),
                        "--checkpoint-dir", str(session_dir / "lora")], cwd=ROOT, check=True)
    else:
        _log("Merge skipped (requires the merge script plus completed aux and all)")


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.environ["RUN_ID"] = run_id  # child variant.py joins this session

    if args.num_gpus < 1 or args.num_gpus > 6:
        raise SystemExit("ERROR: --num-gpus must be an integer from 1 to 6")
    if args.visible_devices and args.num_gpus != 1:
        raise SystemExit("ERROR: --visible-devices model sharding requires --num-gpus 1")

    _gpu_preflight()
    visible = _visible_gpu_count()
    if visible and visible < args.num_gpus:
        raise SystemExit(f"ERROR: --num-gpus {args.num_gpus}, but only {visible} GPU(s) visible")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in PAIR_FILE_BY_VARIANT]
    if unknown:
        raise SystemExit(f"ERROR: invalid variant(s): {', '.join(unknown)}")
    if not variants:
        raise SystemExit("ERROR: no variants requested")

    missing = [str(ROOT / args.data_dir / PAIR_FILE_BY_VARIANT[v])
               for v in variants
               if not (ROOT / args.data_dir / PAIR_FILE_BY_VARIANT[v]).is_file()]
    if missing:
        raise SystemExit("ERROR: missing pair input(s):\n  " + "\n  ".join(missing))

    session_dir = config.RUNS_DIR / run_id
    log_dir = session_dir / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    worker_count = min(args.num_gpus, len(variants))
    _log(f"Session: {run_id}; variants: {' '.join(variants)}; parallel workers: {worker_count}")
    _log(f"Pair source: {args.data_dir}")

    failures: list[str] = []
    fail_lock = threading.Lock()

    def worker(gpu: int) -> None:
        for index in range(gpu, len(variants), worker_count):
            variant = variants[index]
            code = _train_variant(gpu=gpu, variant=variant, args=args,
                                  session_dir=session_dir, log_dir=log_dir)
            if code != 0:
                with fail_lock:
                    failures.append(variant)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(worker_count)]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        _log("interrupted — waiting for in-flight variants to exit")
        for thread in threads:
            thread.join()
        return 130

    if failures:
        _log(f"ERROR: training failed for: {', '.join(sorted(failures))}; merge skipped")
        return 1

    if args.merge:
        _auto_merge(session_dir)
    else:
        _log("Merge skipped (--no-merge)")

    _log(f"Training complete: {session_dir / 'lora'}")
    _log(f"Worker logs: {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
