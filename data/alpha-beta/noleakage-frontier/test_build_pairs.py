from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("build_pairs.py")
SPEC = importlib.util.spec_from_file_location("build_pairs", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
PAIR_PIPELINE = module.PAIR_PIPELINE
build_pairs = module.build_pairs


PAYOFFS = [
    {"own": "C", "opponent": "C", "payoff": [3, 3]},
    {"own": "D", "opponent": "C", "payoff": [5, 0]},
    {"own": "C", "opponent": "D", "payoff": [0, 5]},
    {"own": "D", "opponent": "D", "payoff": [1, 1]},
]


class TrainingPairPipelineTests(unittest.TestCase):
    def test_counterfactual_flag_selects_paper_eq2_or_eq3(self):
        row = {
            "game_family": "matrix",
            "opponent": "grim_trigger",
            "legal_actions": ["C", "D"],
            "payoffs": PAYOFFS,
            "own_actions": ["C", "C"],
            "opponent_actions": ["C", "C"],
        }
        with patch.object(module.optimal_continuation, "horizon_aware_return", return_value=7) as eq3, \
             patch.object(module.fixed_continuation, "fixed_continuation_return", return_value=6) as eq2:
            module._accepted_rounds(row, counterfactual_mode="horizon-aware")
            self.assertTrue(eq3.called)
            self.assertFalse(eq2.called)
        with patch.object(module.optimal_continuation, "horizon_aware_return", return_value=7) as eq3, \
             patch.object(module.fixed_continuation, "fixed_continuation_return", return_value=6) as eq2:
            module._accepted_rounds(row, counterfactual_mode="fixed")
            self.assertFalse(eq3.called)
            self.assertTrue(eq2.called)

    def test_emits_frontier_reasoning_and_blind_round_only(self):
        row = {
            "id": "pd-last-round",
            "game_family": "matrix",
            "opponent": "grim_trigger",
            "legal_actions": ["C", "D"],
            "payoffs": PAYOFFS,
            "own_actions": ["C", "C"],
            "opponent_actions": ["C", "C"],
            "round_prompts": ["round one visible state", "round two visible history"],
            "round_completions": [
                "<think>blind one</think><action>C</action>",
                "<think>blind two</think><action>C</action>",
            ],
            "frontier_completions": [
                "<think>[Prior] cooperative type probability 0.5\n[Update] posterior remains uncertain\n[EV] EV(D)=0.5 * 5 + 0.5 * 1; EV(C)=0.5 * 3 + 0.5 * 0\n[DECISION] D</think><action>D</action>",
                "<think>[Prior] cooperative type probability 0.8\n[Update] prior cooperation supports this posterior\n[EV] EV(D)=0.8 * 5 + 0.2 * 1; EV(C)=0.8 * 3 + 0.2 * 0\n[DECISION] D</think><action>D</action>",
            ],
        }
        pairs = build_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["prompt"], "round two visible history")
        self.assertEqual(pairs[0]["rejected"], row["round_completions"][1])
        self.assertEqual(pairs[0]["chosen"], row["frontier_completions"][1])
        self.assertEqual(pairs[0]["provenance"]["pipeline"], PAIR_PIPELINE)

    def test_leakage_flag_is_recorded_not_fatal(self):
        row = {
            "id": "pd-leak-audit",
            "game_family": "matrix",
            "opponent": "grim_trigger",
            "legal_actions": ["C", "D"],
            "payoffs": PAYOFFS,
            "own_actions": ["C", "C"],
            "opponent_actions": ["C", "C"],
            "round_prompts": ["round one visible state", "round two visible history"],
            "round_completions": [
                "<think>blind one</think><action>C</action>",
                "<think>blind two</think><action>C</action>",
            ],
            "frontier_completions": [
                "<think>[Prior] p=0.5\n[Update] unchanged\n[EV] EV(D)=0.5 * 5 + 0.5 * 1\n[DECISION] D</think><action>D</action>",
                "<think>[Prior] cooperative type probability 0.8\n"
                "[Update] the opponent cooperated in round 1, a prior round\n"
                "[EV] EV(D)=0.8 * 5 + 0.2 * 1\n[DECISION] D</think><action>D</action>",
            ],
        }
        pairs = build_pairs(row)
        self.assertEqual(len(pairs), 1)
        audit = pairs[0]["provenance"]["leakage_audit"]
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["rule"], "current-round-opponent-action")

    def test_one_shot_flip_without_strict_improvement_is_skipped(self):
        # Opponent claims the whole capacity: take-the-rest is (0,0,0) and the
        # recorded infeasible claim also scores 0, so there is no improving flip.
        row = {
            "id": "neg-no-improvement",
            "game_family": "bargaining",
            "capacity": 4,
            "own_weights": [3, 4, 3],
            "own_actions": [[2, 2, 2]],
            "opponent_actions": [[4, 4, 4]],
            "round_prompts": ["negotiation state"],
            "round_completions": ["<think>blind</think><action>[2,2,2]</action>"],
            "frontier_completions": ["<think>x</think><action>[0,0,0]</action>"],
        }
        self.assertEqual(build_pairs(row), [])

    def test_malformed_teacher_round_raises_by_default_but_skips_on_request(self):
        row = {
            "id": "pd-malformed-frontier",
            "game_family": "matrix",
            "opponent": "grim_trigger",
            "legal_actions": ["C", "D"],
            "payoffs": PAYOFFS,
            "own_actions": ["C", "C"],
            "opponent_actions": ["C", "C"],
            "round_prompts": ["round one visible state", "round two visible history"],
            "round_completions": [
                "<think>blind one</think><action>C</action>",
                "<think>blind two</think><action>C</action>",
            ],
            "frontier_completions": [
                "<think>[Prior] p=0.5\n[Update] unchanged\n[EV] EV(D)=0.5 * 5 + 0.5 * 1\n[DECISION] D</think><action>D</action>",
                # accepted round, but the [EV] slot has no probability-weighted
                # arithmetic and there is no <action> block
                "<think>[Prior] p=0.8\n[Update] unchanged\n[EV] D just looks better\n[DECISION] D</think>",
            ],
        }
        with self.assertRaisesRegex(ValueError, "invalid frontier completion"):
            build_pairs(row)
        skipped: list = []
        pairs = build_pairs(row, on_invalid="skip", skipped=skipped)
        self.assertEqual(pairs, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["round"], 2)

    def test_rejects_multi_round_input_without_round_prompts(self):
        row = {
            "id": "missing-prompts",
            "game_family": "matrix",
            "opponent": "always_cooperate",
            "legal_actions": ["C", "D"],
            "payoffs": PAYOFFS,
            "own_actions": ["C", "C"],
            "opponent_actions": ["C", "C"],
            "response": "<think>a</think><action>C</action><think>b</think><action>C</action>",
            "frontier_completions": [],
        }
        with self.assertRaisesRegex(ValueError, "round_prompts"):
            build_pairs(row)


if __name__ == "__main__":
    unittest.main()
