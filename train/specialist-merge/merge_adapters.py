#!/usr/bin/env python3
"""§2.5 Eq. 4: merge the AUX and ALL specialist LoRA adapters.

    W_merge = alpha * W_AUX + (1 - alpha) * W_ALL

Per-tensor weighted sum of the two adapter safetensors; ``adapter_config.json``
is copied from the AUX adapter. Standalone — no torch/trl/peft, only
``safetensors`` — so it runs without a training allocation.

    python train/dpo/specialist-merge/merge_adapters.py --checkpoint-dir runs/<id>/lora
    python train/dpo/specialist-merge/merge_adapters.py --aux <dir> --all <dir> --out <dir> [--alpha 0.5]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def merge_lora_adapters(
    aux_path: str | Path,
    all_path: str | Path,
    out_path: str | Path,
    alpha: float = 0.5,
    STUDENT_MODEL: str | None = None,
) -> Path:
    """Return W_merge = alpha * W_AUX + (1-alpha) * W_ALL."""
    from safetensors.torch import load_file, save_file

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    aux_files = list(Path(aux_path).glob("*.safetensors"))
    all_files = list(Path(all_path).glob("*.safetensors"))
    if not aux_files or not all_files:
        raise FileNotFoundError("Adapter safetensors not found in aux/all paths")

    aux_sd = load_file(str(aux_files[0]))
    all_sd = load_file(str(all_files[0]))
    merged = {}
    for key in aux_sd:
        if key in all_sd:
            merged[key] = alpha * aux_sd[key] + (1.0 - alpha) * all_sd[key]
        else:
            merged[key] = aux_sd[key]

    save_file(merged, str(out_path / aux_files[0].name))
    adapter_config = Path(aux_path) / "adapter_config.json"
    if adapter_config.is_file():
        shutil.copy2(adapter_config, out_path / adapter_config.name)

    return out_path


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.checkpoint_dir is not None:
        aux = args.aux or args.checkpoint_dir / "aux"
        all_ckpt = args.all_dir or args.checkpoint_dir / "all"
        out = args.out or args.checkpoint_dir / "merge"
        return aux, all_ckpt, out
    if not (args.aux and args.all_dir and args.out):
        raise SystemExit("error: pass --checkpoint-dir, or all of --aux/--all/--out")
    return args.aux, args.all_dir, args.out


def main(argv: list[str] | None = None) -> int:
    from config import MERGE_ALPHA

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="LoRA run dir holding aux/ and all/ (merge/ is written alongside)",
    )
    p.add_argument("--aux", type=Path, default=None, help="AUX adapter dir (overrides --checkpoint-dir)")
    p.add_argument("--all", dest="all_dir", type=Path, default=None, help="ALL adapter dir")
    p.add_argument("--out", type=Path, default=None, help="Output merge dir")
    p.add_argument(
        "--alpha",
        type=float,
        default=MERGE_ALPHA,
        help=f"Eq. 4 weight on AUX (default: config.MERGE_ALPHA={MERGE_ALPHA})",
    )
    args = p.parse_args(argv)

    aux, all_ckpt, out = _resolve_paths(args)
    if not (aux / "adapter_config.json").is_file() or not (all_ckpt / "adapter_config.json").is_file():
        print(
            f"Merge skipped: requires completed aux and all adapters ({aux}, {all_ckpt})",
            file=sys.stderr,
        )
        return 1

    merge_lora_adapters(aux, all_ckpt, out, alpha=args.alpha)
    print(f"MERGE done: alpha={args.alpha} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
