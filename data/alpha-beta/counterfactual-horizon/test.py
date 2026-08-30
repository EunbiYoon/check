from __future__ import annotations

import unittest

from build_pairs import build_pairs
from fixed_continuation import (
    fixed_continuation_return, opponent_action, recorded_return,
)
from counterfactual_utils import is_reconstructible
from optimal_continuation import horizon_aware_return


LEGAL = ["C", "D"]
PAYOFFS = {
    ("C", "C"): 3.0,
    ("D", "C"): 5.0,
    ("C", "D"): 0.0,
    ("D", "D"): 1.0,
}


class CounterfactualHorizonTests(unittest.TestCase):
    def test_reconstructs_reciprocity_policies(self):
        self.assertEqual(opponent_action("tit_for_tat", (), LEGAL), "C")
        self.assertEqual(opponent_action("tit_for_tat", (("D", "C"),), LEGAL), "D")
        self.assertEqual(opponent_action("grim_trigger", (("D", "C"), ("C", "D")), LEGAL), "D")
        self.assertEqual(
            opponent_action("tit_for_two_tats", (("D", "C"), ("D", "D")), LEGAL),
            "D",
        )

    def test_fixed_return_replays_opponent_but_keeps_student_actions(self):
        own = ["C", "C", "C"]
        opp = ["C", "C", "C"]
        self.assertEqual(recorded_return(own, opp, PAYOFFS), 9.0)
        # Flip D, TFT retaliates once, then recorded C restores cooperation.
        self.assertEqual(
            fixed_continuation_return(own, opp, 0, "D", "tit_for_tat", LEGAL, PAYOFFS),
            8.0,
        )

    def test_horizon_dp_optimizes_after_forced_flip(self):
        own = ["C", "C", "C"]
        opp = ["C", "C", "C"]
        # D now gives 5; optimal continuation restores C, then defects for 0+5.
        self.assertEqual(
            horizon_aware_return(own, opp, 0, "D", "tit_for_tat", LEGAL, PAYOFFS),
            10.0,
        )

    def test_filter_rejects_early_grim_defection_and_accepts_last_round(self):
        horizon = 10
        response = " ".join(
            f"<think>r{i}</think><action>C</action>" for i in range(horizon)
        )
        row = {
            "id": "grim-cooperation",
            "game_family": "matrix",
            "prompt": "repeated PD",
            "response": response,
            "own_actions": ["C"] * horizon,
            "opponent_actions": ["C"] * horizon,
            "opponent": "grim_trigger",
            "legal_actions": LEGAL,
            "payoffs": [
                {"own": own, "opponent": opp, "payoff": payoff}
                for (own, opp), payoff in PAYOFFS.items()
            ],
        }
        pairs = build_pairs(row)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["meta"]["flip_round"], 9)
        self.assertEqual(pairs[0]["meta"]["recorded_return"], 30.0)
        self.assertEqual(pairs[0]["meta"]["counterfactual_return"], 32.0)
        self.assertEqual(pairs[0]["chosen"].count("<action>D</action>"), 1)

    def test_non_reconstructible_opponent_is_skipped_not_errored(self):
        self.assertTrue(is_reconstructible("tit_for_tat"))
        self.assertFalse(is_reconstructible("generous_tft"))
        self.assertFalse(is_reconstructible("random"))
        horizon = 4
        response = " ".join(
            f"<think>r{i}</think><action>C</action>" for i in range(horizon)
        )
        row = {
            "id": "vs-epsilon-greedy",
            "game_family": "matrix",
            "prompt": "repeated PD",
            "response": response,
            "own_actions": ["C"] * horizon,
            "opponent_actions": ["C"] * horizon,
            "opponent": "epsilon_greedy",
            "legal_actions": LEGAL,
            "payoffs": [
                {"own": own, "opponent": opp, "payoff": payoff}
                for (own, opp), payoff in PAYOFFS.items()
            ],
        }
        self.assertEqual(build_pairs(row), [])

    def test_rejects_stochastic_and_inconsistent_opponents(self):
        with self.assertRaises(ValueError):
            opponent_action("random", (), LEGAL)
        row = {
            "game_family": "matrix",
            "prompt": "x",
            "response": "<action>C</action>",
            "own_actions": ["C"],
            "opponent_actions": ["D"],
            "opponent": "tit_for_tat",
            "legal_actions": LEGAL,
            "payoffs": [
                {"own": own, "opponent": opp, "payoff": payoff}
                for (own, opp), payoff in PAYOFFS.items()
            ],
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_pairs(row)


if __name__ == "__main__":
    unittest.main()
