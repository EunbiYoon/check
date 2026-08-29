"""LoRA model-loading utilities.

Adapter merging (§2.5 Eq. 4) lives in ``train/specialist-merge/merge_adapters.py``.

``train/dpo-lora`` is a hyphenated directory, so it is not importable as a
package: run ``variant.py`` / ``orchestrate.py`` directly, and load the shared
helpers here by file path (see the ``_load`` helpers in
``data/alpha-beta/noleakage-frontier/build_pairs.py`` and ``eval/table``). The
bootstrap below puts this directory on ``sys.path`` so the siblings resolve by
bare name.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import gpu_env  # noqa: F401,E402 — set BNB_CUDA_VERSION before HF/bnb import

import os  # noqa: E402
from typing import Any  # noqa: E402

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config


def _check_bitsandbytes_cuda() -> None:
    import bitsandbytes.functional as bnb_fn
    import bitsandbytes.cextension as bnb_ext

    lib = bnb_ext.lib
    if lib is None or not getattr(lib, "compiled_with_cuda", False):
        raise RuntimeError(
            "bitsandbytes loaded CPU library (4-bit quantize unavailable). "
            "Load CUDA 12.6 and its math libraries before training "
            f"(BNB_CUDA_VERSION={os.environ.get('BNB_CUDA_VERSION', '')!r})"
        )
    probe = torch.randn(8, 8, device="cuda", dtype=torch.float16)
    try:
        bnb_fn.quantize_4bit(probe)
    except Exception as exc:
        raise RuntimeError(
            "bitsandbytes 4-bit probe failed on GPU. "
            "Run train/train.sh from a CUDA GPU allocation."
        ) from exc


def build_bnb_config() -> BitsAndBytesConfig:
    compute_dtype = getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
    )


def build_lora_config(
    r: int | None = None,
    alpha: int | None = None,
    target_modules: list[str] | str | None = None,
) -> LoraConfig:
    tm = target_modules if target_modules is not None else config.LORA_TARGET_MODULES
    return LoraConfig(
        r=r or config.LORA_R,
        lora_alpha=alpha or config.LORA_ALPHA,
        target_modules=tm,
        lora_dropout=config.LORA_DROPOUT,
        task_type="CAUSAL_LM",
    )


def load_base_model(
    STUDENT_MODEL: str | None = None,
    use_4bit: bool = True,
    attn_implementation: str | None = None,
    torch_dtype: torch.dtype | str | None = None,
) -> tuple[Any, Any]:
    STUDENT_MODEL = STUDENT_MODEL or config.STUDENT_MODEL
    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit load requires a visible GPU (torch.cuda.is_available() is False). "
                "On the cluster run: module load cuda/12.6 && nvidia-smi"
            )
        _check_bitsandbytes_cuda()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Single-process training: one worker per GPU, pinned by CUDA_VISIBLE_DEVICES
    # in train/train.sh. device_map="auto" then places the model on the visible
    # device(s); with several visible it shards one worker across them.
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if use_4bit:
        kwargs["quantization_config"] = build_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL, **kwargs)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    return model, tokenizer


def attach_lora(model, lora_config: LoraConfig | None = None):
    lora_config = lora_config or build_lora_config()
    return get_peft_model(model, lora_config)


def load_lora_adapter(
    adapter_path: str | Path,
    STUDENT_MODEL: str | None = None,
    use_4bit: bool = True,
):
    model, tokenizer = load_base_model(STUDENT_MODEL=STUDENT_MODEL, use_4bit=use_4bit)
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    return model, tokenizer
