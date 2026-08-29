"""DPO loss masking for game-environment messages.

Targets TRL 0.15.x. Environment text (``[round continues] ...`` blocks) stays
visible to the model as context but is excluded from the DPO loss.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from trl import DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference

_ENVIRONMENT_BLOCK = re.compile(
    r"(?:\r?\n){0,2}\[round continues\].*?(?=(?:\r?\n){2,}<think>|\Z)", re.DOTALL
)


def environment_spans(response: str) -> list[tuple[int, int]]:
    return [match.span() for match in _ENVIRONMENT_BLOCK.finditer(response)]


def _overlaps_any(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return end > start and any(
        start < span_end and end > span_start for span_start, span_end in spans
    )


def _completion_loss_mask(text: str, tokenizer: Any, token_count: int) -> list[int]:
    spans = environment_spans(text)
    offsets = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    mask = [0 if _overlaps_any(start, end, spans) else 1 for start, end in offsets]
    mask.append(1)  # EOS appended by DPOTrainer stays supervised.
    if len(mask) < token_count:
        mask.extend([1] * (token_count - len(mask)))
    return mask[:token_count]


@dataclass
class EnvironmentMaskCollator(DataCollatorForPreference):
    """Pad the per-token environment loss masks alongside the preference batch."""

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        output = super().torch_call(examples)
        for side in ("chosen", "rejected"):
            masks = [torch.tensor(example[f"{side}_loss_mask"]) for example in examples]
            output[f"{side}_loss_mask"] = torch.nn.utils.rnn.pad_sequence(
                masks, batch_first=True, padding_value=0
            )
        return output


class _RestoreContextAttention:
    """Restore masked environment positions for the model's attention only."""

    def __init__(self, model: Any):
        self.model = model

    def __call__(self, input_ids: torch.Tensor, **kwargs: Any):
        attention = kwargs.get("attention_mask")
        if attention is not None:
            restored = attention.clone()
            for row in range(restored.shape[0]):
                active = restored[row].nonzero(as_tuple=False)
                if active.numel():
                    restored[row, active[0, 0] : active[-1, 0] + 1] = 1
            kwargs["attention_mask"] = restored
        return self.model(input_ids, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


class EnvironmentMaskedDPOTrainer(DPOTrainer):
    """Keep environment text as context while excluding it from DPO loss."""

    def __init__(self, *args: Any, **kwargs: Any):
        processing_class = kwargs.get("processing_class")
        if processing_class is None:
            raise ValueError("processing_class is required for environment masking")
        kwargs.setdefault(
            "data_collator",
            EnvironmentMaskCollator(pad_token_id=processing_class.pad_token_id),
        )
        super().__init__(*args, **kwargs)

    @staticmethod
    def tokenize_row(
        features, processing_class, max_prompt_length, max_completion_length, add_special_tokens
    ):
        batch = DPOTrainer.tokenize_row(
            features, processing_class, max_prompt_length, max_completion_length, add_special_tokens
        )
        for side in ("chosen", "rejected"):
            batch[f"{side}_loss_mask"] = _completion_loss_mask(
                features[side], processing_class, len(batch[f"{side}_input_ids"])
            )
        return batch

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        for column in ("chosen_loss_mask", "rejected_loss_mask"):
            if column not in self._signature_columns:
                self._signature_columns.append(column)

    def concatenated_forward(self, model, batch):
        masked_batch = dict(batch)
        masked_batch["chosen_attention_mask"] = batch["chosen_loss_mask"]
        masked_batch["rejected_attention_mask"] = batch["rejected_loss_mask"]
        return super().concatenated_forward(_RestoreContextAttention(model), masked_batch)
