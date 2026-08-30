from __future__ import annotations

import unittest

from eval.rollout.agents import (
    _EV_SHORTLIST_THRESHOLD,
    _format_llm_prompt,
    _slots_contract,
)


class _NoChatTemplateTokenizer:
    chat_template = None

    def apply_chat_template(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("chat template path should not be taken in this test")


class SlotsContractTests(unittest.TestCase):
    def test_small_action_space_keeps_exhaustive_ev(self):
        contract = _slots_contract(2)
        self.assertIn("for EACH legal action", contract)
        self.assertNotIn("3-5 most promising", contract)
        self.assertIn("[Decision]", contract)

    def test_large_action_space_switches_to_shortlist_ev(self):
        # negotiation (~125) / auction (~201): enumerating every EV overran the
        # token budget and the model dropped [Decision] entirely.
        contract = _slots_contract(125)
        self.assertIn("3-5 most promising", contract)
        self.assertNotIn("for EACH legal action", contract)
        self.assertIn("[Decision]", contract)

    def test_threshold_boundary(self):
        self.assertIn("for EACH legal action", _slots_contract(_EV_SHORTLIST_THRESHOLD))
        self.assertIn("3-5 most promising", _slots_contract(_EV_SHORTLIST_THRESHOLD + 1))

    def test_none_legal_count_defaults_to_exhaustive(self):
        self.assertIn("for EACH legal action", _slots_contract(None))

    def test_format_prompt_threads_legal_count_into_slots_contract(self):
        tok = _NoChatTemplateTokenizer()
        small = _format_llm_prompt(tok, "PROMPT", "slots", n_legal=2)
        big = _format_llm_prompt(tok, "PROMPT", "slots", n_legal=200)
        self.assertIn("for EACH legal action", small)
        self.assertIn("3-5 most promising", big)

    def test_concise_format_is_unaffected_by_legal_count(self):
        tok = _NoChatTemplateTokenizer()
        self.assertEqual(
            _format_llm_prompt(tok, "PROMPT", "concise", n_legal=2),
            _format_llm_prompt(tok, "PROMPT", "concise", n_legal=200),
        )


if __name__ == "__main__":
    unittest.main()
