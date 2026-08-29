from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("offline_trajectory.py")
SPEC = importlib.util.spec_from_file_location("blind_rollout_offline", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
rollout_game = module.rollout_game
write_trajectories = module.write_trajectories
shard_unit_total = module.shard_unit_total


class FixedAgent:
    def decide(self, obs):
        return "history-only blind reasoning", obs["legal_actions"][0]


class BlindRolloutTests(unittest.TestCase):
    def test_rollout_emits_lossless_algorithm1_schema(self):
        rows = list(rollout_game(
            game="pd-classic",
            agent=FixedAgent(),
            episodes_per_combination=1,
            seed=42,
            STUDENT_MODEL="Qwen/Qwen2.5-7B-Instruct",
        ))
        self.assertEqual(len(rows), 6)
        row = rows[0]
        self.assertEqual(row["frontier_model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(len(row["round_prompts"]), 10)
        self.assertEqual(len(row["round_completions"]), 10)
        self.assertEqual(len(row["own_actions"]), 10)
        self.assertEqual(len(row["opponent_actions"]), 10)
        self.assertIn("History: (none yet", row["round_prompts"][0])
        self.assertIn("<think>history-only blind reasoning</think>", row["round_completions"][0])

    def test_shards_partition_the_work_units_without_overlap(self):
        kwargs = dict(
            game="pd-classic", agent=FixedAgent(),
            episodes_per_combination=3, seed=42,
            STUDENT_MODEL="Qwen/Qwen2.5-7B-Instruct",
        )
        whole = [r["id"] for r in rollout_game(**kwargs)]
        self.assertEqual(len(whole), 18)  # 6 opponents x 3 episodes
        pieces: list[str] = []
        for shard_index in range(3):
            part = [
                r["id"] for r in rollout_game(
                    **kwargs, num_shards=3, shard_index=shard_index
                )
            ]
            self.assertEqual(
                len(part),
                shard_unit_total(["pd-classic"], 3, num_shards=3, shard_index=shard_index),
            )
            pieces.extend(part)
        self.assertEqual(sorted(pieces), sorted(whole))

    def test_shard_unit_total_handles_fewer_units_than_shards(self):
        # negotiation has 4 opponents; at 1 episode/combination only 4 units.
        totals = [
            shard_unit_total(["negotiation"], 1, num_shards=6, shard_index=i)
            for i in range(6)
        ]
        self.assertEqual(sorted(totals, reverse=True), [1, 1, 1, 1, 0, 0])


class WriteTrajectoriesTests(unittest.TestCase):
    def test_streams_rows(self):
        seen: list[int] = []

        def rows():
            for i in range(3):
                # File is readable mid-stream.
                if i > 0:
                    self.assertEqual(
                        len(output.read_text(encoding="utf-8").splitlines()), i
                    )
                seen.append(i)
                yield {"i": i}

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            output = Path(directory) / "shard.jsonl"
            count = write_trajectories(rows(), output, progress_every=0)
            self.assertEqual(count, 3)
            self.assertEqual(seen, [0, 1, 2])

    def test_empty_rollout_raises_and_removes_output(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            output = Path(directory) / "shard.jsonl"
            with self.assertRaises(ValueError):
                write_trajectories(iter([]), output)
            self.assertFalse(output.exists())

    def test_empty_reasoning_is_rejected(self):
        row = {
            "trajectory_id": "broken-1",
            "round_completions": ["<think></think><action>C</action>"],
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            output = Path(directory) / "shard.jsonl"
            with self.assertRaisesRegex(RuntimeError, "empty or fallback reasoning"):
                write_trajectories(iter([row]), output, total=1)

    def test_fallback_reasoning_is_rejected(self):
        row = {
            "trajectory_id": "fallback-1",
            "round_completions": [
                "<think>fallback: C</think><action>C</action>"
            ],
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            output = Path(directory) / "shard.jsonl"
            with self.assertRaisesRegex(RuntimeError, "empty or fallback reasoning"):
                write_trajectories(iter([row]), output, total=1)


if __name__ == "__main__":
    unittest.main()
