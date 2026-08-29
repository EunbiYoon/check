from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_pairs import build_file, build_pairs
from coupling_pool import build_pool, expand_row, trajectory_coupled
from nash import ne_own_actions
from strategies import strategy_consensus

PD_PAYOFFS = [
    {"own": "C", "opponent": "C", "payoff": [3, 3]},
    {"own": "C", "opponent": "D", "payoff": [0, 5]},
    {"own": "D", "opponent": "C", "payoff": [5, 0]},
    {"own": "D", "opponent": "D", "payoff": [1, 1]},
]
STAG_PAYOFFS = [
    {"own": "Stag", "opponent": "Stag", "payoff": [4, 4]},
    {"own": "Stag", "opponent": "Hare", "payoff": [0, 3]},
    {"own": "Hare", "opponent": "Stag", "payoff": [3, 0]},
    {"own": "Hare", "opponent": "Hare", "payoff": [1, 1]},
]
MP_PAYOFFS = [
    {"own": "H", "opponent": "H", "payoff": [1, -1]},
    {"own": "H", "opponent": "T", "payoff": [-1, 1]},
    {"own": "T", "opponent": "H", "payoff": [-1, 1]},
    {"own": "T", "opponent": "T", "payoff": [1, -1]},
]

PROMPT = "PD one-shot: choose C or D."


def _completion(action: str, evs: dict[str, float]) -> str:
    body = "[EV] " + "; ".join(f"EV({a}) = {v} + 0 = {v}" for a, v in evs.items())
    return f"<think>{body} [Decision] {action}</think><action>{action}</action>"


def _one_shot(identifier: str, action: str, opp: str, evs: dict[str, float]) -> dict:
    return {
        "id": identifier,
        "game": "pd-classic",
        "opponent": "always_cooperate",
        "frontier_model": "test",
        "round_prompts": [PROMPT],
        "round_completions": [_completion(action, evs)],
        "own_actions": [action],
        "opponent_actions": [opp],
        "game_family": "matrix",
        "legal_actions": ["C", "D"],
        "payoffs": PD_PAYOFFS,
    }


class NashTests(unittest.TestCase):
    def test_prisoners_dilemma_has_a_single_pure_equilibrium(self):
        self.assertEqual(ne_own_actions(PD_PAYOFFS, ["C", "D"]), {"D"})

    def test_stag_hunt_has_two_pure_equilibria(self):
        self.assertEqual(ne_own_actions(STAG_PAYOFFS, ["Stag", "Hare"]), {"Stag", "Hare"})

    def test_matching_pennies_falls_back_to_full_mixed_support(self):
        self.assertEqual(ne_own_actions(MP_PAYOFFS, ["H", "T"]), {"H", "T"})


class PoolTests(unittest.TestCase):
    def test_expand_row_tags_coupling_and_cumulative_reward(self):
        row = {
            "id": "pd-tft-1",
            "game": "pd-classic",
            "opponent": "tit_for_tat",
            "round_prompts": ["r1", "r2"],
            "round_completions": [
                _completion("D", {"C": 1.0, "D": 2.0}),
                _completion("C", {"C": 1.0, "D": 2.0}),
            ],
            "own_actions": ["D", "C"],
            "opponent_actions": ["C", "D"],
            "game_family": "matrix",
            "legal_actions": ["C", "D"],
            "payoffs": PD_PAYOFFS,
        }
        first, second = expand_row(row)
        self.assertEqual(first.trajectory_return, 5.0)  # u0(D,C)=5 + u0(C,D)=0
        self.assertTrue(first.coupled)          # played D == argmax stated EV
        self.assertFalse(second.coupled)        # played C != argmax stated EV
        # Final parsable round is uncoupled -> trajectory is uncoupled.
        self.assertFalse(trajectory_coupled([first, second]))

    def test_unparsable_rows_are_skipped_not_fatal(self):
        good = _one_shot("ok", "D", "C", {"C": 1.0, "D": 2.0})
        bad = {"id": "bad", "round_completions": ["x"], "own_actions": ["D"]}
        pool = build_pool([good, bad])
        self.assertEqual({o.trajectory_id for o in pool.observations}, {"ok"})
        self.assertEqual([identifier for identifier, _ in pool.skipped], ["bad"])


class StrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        # Same prompt, three frontier completions: D beats C on every axis.
        self.rows = [
            _one_shot("d1", "D", "C", {"C": 1.0, "D": 2.0}),   # coupled, return 5
            _one_shot("d2", "D", "C", {"C": 1.0, "D": 2.0}),   # coupled, return 5
            _one_shot("c1", "C", "C", {"C": 1.0, "D": 2.0}),   # uncoupled, return 3
        ]

    def test_every_strategy_pins_chosen_to_recorded_and_swaps_the_action(self):
        pairs = build_pairs(self.rows, coupling_filter=False)
        self.assertTrue(pairs)
        for pair in pairs:
            self.assertEqual(pair["provenance"]["chosen_action_label"], "D")
            self.assertEqual(pair["provenance"]["rejected_action_label"], "C")
            self.assertIn("<action>D</action>", pair["chosen"])
            self.assertIn("<action>C</action>", pair["rejected"])
            # Reasoning is kept verbatim; only the action token changes.
            self.assertEqual(
                pair["chosen"].split("<action>")[0],
                pair["rejected"].split("<action>")[0],
            )
        emitted = {name for pair in pairs for name in pair["provenance"]["strategies"]}
        self.assertEqual(
            emitted, {"S1-outcome", "S2-ne", "S3-consensus", "S4-l2-l0"}
        )

    def test_consensus_strategy_uses_the_modal_action(self):
        pool = build_pool(self.rows)
        candidates = strategy_consensus(pool)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate.chosen.action_label, "D")
            self.assertEqual(candidate.rejected_label, "C")

    def test_coupling_filter_keeps_only_coupled_chosen_trajectories(self):
        # Add an uncoupled defect trajectory: its argmax stated EV is C, so a
        # coupling-filter-on build must drop any pair whose chosen side is it.
        rows = self.rows + [_one_shot("d3", "D", "C", {"C": 9.0, "D": 1.0})]
        unfiltered = build_pairs(rows, coupling_filter=False)
        filtered = build_pairs(rows, coupling_filter=True)
        self.assertTrue(filtered)
        self.assertLess(len(filtered), len(unfiltered))
        for pair in filtered:
            self.assertTrue(pair["provenance"]["chosen_trajectory_coupled"])

    def test_build_file_writes_dpo_pair_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "traj.jsonl"
            output = Path(directory) / "pairs.jsonl"
            source.write_text(
                "\n".join(json.dumps(row) for row in self.rows) + "\n",
                encoding="utf-8",
            )
            count = build_file(source, output, coupling_filter=False)
            written = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(count, len(written))
        for pair in written:
            self.assertEqual({"prompt", "chosen", "rejected"} & set(pair),
                             {"prompt", "chosen", "rejected"})
            self.assertIn("<action>D</action>", pair["chosen"])


if __name__ == "__main__":
    unittest.main()
