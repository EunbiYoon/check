#!/usr/bin/env python3
"""Train one LoRA/DPO variant."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gpu_env  # noqa: F401,E402 — configure CUDA before HF/bnb imports


def main() -> None:
    from runner import run_training

    run_training()


if __name__ == "__main__":
    main()
