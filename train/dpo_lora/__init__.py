"""LoRA/DPO training for a single variant.

Entry point: ``python -m train.dpo_lora`` (= :func:`cli.main`). The multi-variant
orchestrator is ``train/dpo_lora/train.sh``.
"""

from .cli import main

__all__ = ["main"]
