"""§2.5 Eq. 4 specialist adapter merge: W_merge = alpha*W_AUX + (1-alpha)*W_ALL."""

from .merge_adapters import merge_lora_adapters

__all__ = ["merge_lora_adapters"]
