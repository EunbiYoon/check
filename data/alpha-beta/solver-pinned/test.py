from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from update_pairs import build_file, build_pair, pin_actions
from best_action import (
    bargaining_best_action,
    auction_best_action,
    matrix_best_action,
    stage_utility,
)


class SolverPinnedTests(unittest.TestCase):
    def test_matrix_enumerates_and_uses_legal_order_for_ties(self):
        payoffs = {
            ("C", "C"): 3,
            ("D", "C"): 5,
            ("C", "D"): 0,
            ("D", "D"): 1,
        }
        self.assertEqual(matrix_best_action("C", ["C", "D"], payoffs), "D")
        self.assertEqual(matrix_best_action("D", ["C", "D"], payoffs), "D")
        tied = {("left", "x"): 1, ("right", "x"): 1}
        self.assertEqual(matrix_best_action("x", ["right", "left"], tied), "right")

    def test_first_price_auction_uses_smallest_strictly_higher_legal_bid(self):
        bids = list(range(21))
        self.assertEqual(auction_best_action(10, 20, bids), 11)
        self.assertEqual(auction_best_action(10, 11, bids), 0)
        self.assertEqual(auction_best_action(20, 100, bids), 0)

    def test_bargaining_takes_the_rest(self):
        self.assertEqual(bargaining_best_action([1, 3, 2], 4, [3, 4, 3]), (3, 1, 2))
        self.assertEqual(
            bargaining_best_action([1, 3, 2], [2, 4, 6], [1, 1, 1]),
            (1, 1, 4),
        )

    def test_main_auction_reads_recorded_trajectory_schema(self):
        from best_action import main

        row = {
            "game_family": "auction",
            "private_value": 100,          # recorded engine key
            "legal_actions": list(range(0, 201)),  # recorded bid menu
            "own_actions": [30],           # recorded bid sequence, not the menu
            "opponent_actions": [40],
        }
        self.assertEqual(main(row, 40), 41)   # smallest profitable bid above 40
        row["private_value"] = 20
        self.assertEqual(main(row, 40), 0)    # winning bid 41 > value -> deliberate loss

    def test_stage_utility_matches_game_payoffs(self):
        matrix_spec = {
            "game_family": "pd-classic",
            "payoffs": [
                {"own": "C", "opponent": "C", "payoff": [3, 3]},
                {"own": "D", "opponent": "C", "payoff": [5, 0]},
                {"own": "C", "opponent": "D", "payoff": [0, 5]},
                {"own": "D", "opponent": "D", "payoff": [1, 1]},
            ],
        }
        self.assertEqual(stage_utility(matrix_spec, "D", "C"), 5.0)
        auction_spec = {"game_family": "auction", "own_value": 100}
        self.assertEqual(stage_utility(auction_spec, 40, 30), 60.0)  # win
        self.assertEqual(stage_utility(auction_spec, 20, 30), 0.0)   # lose
        self.assertEqual(stage_utility(auction_spec, 0, 30), 0.0)    # deliberate loss
        bargaining_spec = {
            "game_family": "bargaining",
            "capacity": 4,
            "own_weights": [3, 4, 3],
        }
        self.assertEqual(stage_utility(bargaining_spec, (3, 1, 2), (1, 3, 2)), 19.0)
        self.assertEqual(stage_utility(bargaining_spec, (4, 4, 4), (1, 3, 2)), 0.0)  # infeasible

    def test_action_substitution_is_round_aligned(self):
        response = "<think>a</think><action>C</action> context <action>D</action>"
        self.assertEqual(
            pin_actions(response, ["D", "C"]),
            "<think>a</think><action>D</action> context <action>C</action>",
        )

    def test_build_pair_preserves_reasoning_and_rejects_recorded_actions(self):
        row = {
            "id": "pd-1",
            "game_family": "pd-classic",
            "prompt": "state visible at decision time",
            "response": "<think>recorded reasoning</think>\n<action>C</action>",
            "opponent_actions": ["C"],
            "legal_actions": ["C", "D"],
            "payoffs": [
                {"own": "C", "opponent": "C", "payoff": [3, 3]},
                {"own": "D", "opponent": "C", "payoff": [5, 0]},
                {"own": "C", "opponent": "D", "payoff": [0, 5]},
                {"own": "D", "opponent": "D", "payoff": [1, 1]},
            ],
        }
        pair = build_pair(row)
        self.assertEqual(pair["rejected"], row["response"])
        self.assertIn("<think>recorded reasoning</think>", pair["chosen"])
        self.assertIn("<action>D</action>", pair["chosen"])
        self.assertNotIn("opponent_actions", pair)

    def test_jsonl_cli_core_writes_standard_pair_schema(self):
        row = {
            "game_family": "bargaining",
            "prompt": "prompt",
            "response": "<action>[0,0,0]</action>",
            "opponent_actions": [[1, 3, 2]],
            "capacity": 4,
            "own_weights": [3, 4, 3],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trajectories.jsonl"
            output = Path(directory) / "pairs.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(build_file(source, output), 1)
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(set(result), {"prompt", "chosen", "rejected"})
        self.assertEqual(result["chosen"], "<action>[3,1,2]</action>")


if __name__ == "__main__":
    unittest.main()
