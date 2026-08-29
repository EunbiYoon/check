"""LLM and heuristic agents for eval rollouts."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

import torch
from transformers import GenerationConfig

from config import OLLAMA_BASE_URL


class HeuristicAgent:
    """Fast baseline without GPU inference."""

    def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
        legal = obs.get("legal_actions") or []
        game = obs.get("game", "")
        if not legal:
            return "heuristic fallback", None
        if game in ("pd-classic", "pd-tight", "pd-high-temptation", "ipd-stage"):
            action = "C" if "C" in legal else legal[0]
        elif game == "stag-hunt":
            action = "Stag" if "Stag" in legal else legal[0]
        elif game == "bos":
            action = "Opera" if "Opera" in legal else legal[0]
        elif game == "matching-pennies":
            action = legal[0]
        elif game == "negotiation":
            action = (1, 1, 1)
        elif game == "tic-tac-toe":
            action = legal[0]
        elif game == "auction":
            action = obs.get("private_value", 100)
        elif game == "divide-dollar":
            action = 0.5
        elif game == "p-beauty":
            action = 33
        else:
            action = legal[0]
        return f"heuristic: choose {action}", action


class LoRALLMAgent:
    """Qwen + LoRA agent; output format matches DPO training data."""

    def __init__(
        self, model, tokenizer, max_new_tokens: int = 192, *,
        strict_output: bool = False, do_sample: bool = False,
        temperature: float = 0.7, reasoning_format: str = "concise",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.strict_output = strict_output
        self.do_sample = do_sample
        self.temperature = temperature
        if reasoning_format not in _REASONING_FORMATS:
            raise ValueError(f"reasoning_format must be one of {sorted(_REASONING_FORMATS)}")
        self.reasoning_format = reasoning_format

    def _generate(self, obs: dict[str, Any]) -> str:
        text = _format_llm_prompt(self.tokenizer, obs["prompt"], self.reasoning_format)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.model.eval()
        # Kwarg do_sample=False overrides model.generation_config (Llama base defaults to sample).
        with torch.inference_mode():
            out = self.model.generate(**inputs, generation_config=gen_cfg, do_sample=self.do_sample)
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        vocab_size = len(self.tokenizer)
        token_ids = [int(token_id) for token_id in new_tokens.detach().cpu().tolist()]
        invalid_ids = [token_id for token_id in token_ids if not 0 <= token_id < vocab_size]
        if invalid_ids:
            raise RuntimeError(
                "model.generate() returned token IDs outside the tokenizer vocabulary: "
                f"vocab_size={vocab_size}, invalid_ids={invalid_ids[:10]}. "
                "Do not filter these IDs: that silently corrupts rollout reasoning."
            )
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
        # One malformed sample (e.g. <p>…</p> instead of <think>…</think>) must
        # not kill a multi-hour shard: regenerate once before giving up.
        attempts = 2 if self.strict_output else 1
        last_response = ""
        for attempt in range(attempts):
            response = self._generate(obs)
            last_response = response
            if not response.strip():
                if attempt + 1 < attempts:
                    continue
                raise RuntimeError("model.generate() produced an empty blind-rollout response")
            reasoning, action = _parse_response(response, obs)
            if not self.strict_output or (reasoning and action is not None):
                break
        else:
            raise RuntimeError(
                "blind model did not return non-empty reasoning and a valid <action> "
                f"after {attempts} attempts; raw response={last_response[:500]!r}"
            )

        if action is None:
            action = _fallback_action(obs)
            reasoning = reasoning or f"fallback: {action}"
        return reasoning, action


class OllamaAgent:
    """Frontier eval via local Ollama HTTP API (no HF / bitsandbytes)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = OLLAMA_BASE_URL,
        max_new_tokens: int = 192,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens

    def decide(self, obs: dict[str, Any]) -> tuple[str, Any]:
        prompt = obs["prompt"]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": 0,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request failed ({self.base_url}, model={self.model}). "
                f"Is the server running? "
                f"Try: /scratch/workspace/eunbiyoon_umass_edu-paper/ollama-local/scripts/start.sh"
            ) from exc

        response = body.get("message", {}).get("content", "") or ""
        reasoning, action = _parse_response(response, obs)
        if action is None:
            action = _fallback_action(obs)
            reasoning = reasoning or f"fallback: {action}"
        return reasoning, action


_REASONING_FORMATS = ("concise", "slots")

_CONCISE_CONTRACT = (
    "\n\nThe opponent's current-round action is not observable. Reason only from "
    "the game description and completed-round history shown above. "
    "Keep the reasoning concise: at most 80 words. "
    "Respond exactly as <think>your non-empty reasoning</think>"
    "<action>one legal action</action>."
)

# Slot form elicits the [Prior][Update][EV][Decision] structure the coupling
# metric parses (eval/metric/coupling.py). A+beta ignores the blind reasoning
# format (it is the rejected side; the chosen side is re-written by the teacher);
# Hypothesis B's coupling filter needs the [EV] slot to exist.
_SLOTS_CONTRACT = (
    "\n\nThe opponent's current-round action is not observable. Reason only from "
    "the game description and completed-round history above — never state the "
    "opponent's move this round as observed.\n"
    "Write one <think> block with exactly these four sections, in order:\n"
    "[Prior] your belief about the opponent's action this round, as rough probabilities.\n"
    "[Update] how the completed-round history (if any) shifts that belief.\n"
    "[EV] for EACH legal action, the probability-weighted expected value as "
    "arithmetic, e.g. EV(C) = 0.7*3 + 0.3*0 = 2.1\n"
    "[Decision] the legal action with the highest EV.\n"
    "Keep each section to one short sentence. Then one <action> block with that action.\n"
    "Respond exactly as <think>[Prior] ... [Update] ... [EV] ... [Decision] ...</think>"
    "<action>one legal action</action>."
)


def _format_llm_prompt(tokenizer, prompt: str, reasoning_format: str = "concise") -> str:
    """Use chat template when present (Instruct); else raw prompt (pretrained base)."""
    output_contract = _SLOTS_CONTRACT if reasoning_format == "slots" else _CONCISE_CONTRACT
    messages = [{"role": "user", "content": prompt + output_contract}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt + output_contract + "\n"


_REASONING_TAGS = r"think|p|reasoning|thinking|thought|analysis"


def _parse_response(text: str, obs: dict[str, Any]) -> tuple[str, Any]:
    thinking = ""
    m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m_think:
        thinking = m_think.group(1).strip()
    else:
        # Sampling sometimes wraps the reasoning in <p>...</p> (or mismatches the
        # open/close tag). Recover the reasoning as the text before <action>,
        # tolerating any single wrapper tag on either side.
        m_alt = re.search(
            rf"(?:<(?:{_REASONING_TAGS})>)?\s*(.+?)\s*(?:</(?:{_REASONING_TAGS})>)?\s*<action>",
            text,
            re.DOTALL,
        )
        if m_alt:
            thinking = m_alt.group(1).strip()
        else:
            pre = text.split("<action>", 1)[0]
            thinking = re.sub(r"^\s*<[a-z]+>|</[a-z]+>\s*$", "", pre.strip(), flags=re.I).strip()

    action_raw = None
    m_act = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if m_act:
        action_raw = m_act.group(1).strip()
    else:
        for line in reversed(text.strip().splitlines()):
            line = line.strip()
            if line and not line.startswith("<"):
                action_raw = line
                break

    action = _coerce_action(action_raw, obs) if action_raw else None
    return thinking, action


def _coerce_action(raw: str, obs: dict[str, Any]) -> Any:
    legal = obs.get("legal_actions") or []
    game = obs.get("game", "")
    raw = raw.strip()

    if game == "negotiation":
        nums = [int(x) for x in re.findall(r"-?\d+", raw)][:3]
        if len(nums) == 3:
            return tuple(nums)
        return (1, 1, 1)

    if game == "tic-tac-toe":
        nums = re.findall(r"\d+", raw)
        if nums:
            val = int(nums[0])
            if val in legal:
                return val
        return legal[0] if legal else 0

    if game == "auction":
        nums = re.findall(r"\d+", raw)
        return int(nums[0]) if nums else obs.get("private_value", 100)

    if game == "divide-dollar":
        nums = re.findall(r"\d*\.?\d+", raw)
        return float(nums[0]) if nums else 0.5

    if game == "p-beauty":
        nums = re.findall(r"\d+", raw)
        return int(nums[0]) if nums else 33

    for candidate in legal:
        if str(candidate).lower() == raw.lower():
            return candidate
    for candidate in legal:
        if str(candidate).lower() in raw.lower():
            return candidate
    return legal[0] if legal else raw


def _fallback_action(obs: dict[str, Any]) -> Any:
    legal = obs.get("legal_actions") or []
    return legal[0] if legal else None
