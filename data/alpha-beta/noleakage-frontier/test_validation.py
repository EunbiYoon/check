import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validation.py")
SPEC = importlib.util.spec_from_file_location("noleakage_validation", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class NoLeakageTest(unittest.TestCase):
    def test_prompt_separates_inference_and_oracle_information(self):
        prompt = module.build_user_prompt(
            history_prompt="History through round 2",
            solver_action="D",
            opponent_action="C",
        )
        self.assertIn("INFERENCE-TIME INFORMATION", prompt)
        self.assertIn("OUT-OF-BAND REFERENCE", prompt)
        self.assertIn("solver-pinned best response: D", prompt)
        # paper §2.3: the realised opponent move is given out of band
        self.assertIn("opponent's actual current-round move: C", prompt)

    def test_prompt_withholds_opponent_move_when_unknown(self):
        prompt = module.build_user_prompt(
            history_prompt="Round 1 history", solver_action="D"
        )
        self.assertIn("withheld", prompt)

    def test_system_prompt_reproduces_paper_sentences(self):
        # Paper §2.3 (page 4) printed sentences, outside the two ". . ." gaps.
        for sentence in (
            "You are documenting strategic decisions",
            "The other player's current-round move is NOT observable at decision time.",
            "We will tell you (out-of-band, for your reference) the opponent's "
            "actual move that round and the solver-computed best response",
            "your written reasoning must NEVER reference the current-round "
            "opponent move as observed.",
            "Treat it as belief-state reasoning under uncertainty.",
            'NEVER write phrases like "opponent played C this round"',
            "The [EV] arithmetic must average over your posterior types",
            "Your [DECISION] must match the solver-pinned best response we provide",
        ):
            self.assertIn(sentence, module.SYSTEM_PROMPT)

    def test_flags_current_round_observation(self):
        flags = module.audit_reasoning(
            "The opponent defected this round, so I defect.", round_number=3
        )
        self.assertTrue(flags)

    def test_prior_round_reference_is_flagged_for_manual_review(self):
        # Appendix C's regex deliberately also matches legitimate prior-round
        # prose; such candidates are kept after manual inspection, not deleted.
        flags = module.audit_reasoning(
            "The opponent defected in round 2, so my posterior favors a defector type.",
            round_number=3,
        )
        self.assertTrue(flags)

    def test_round_one_past_tense_claim_is_flagged(self):
        flags = module.audit_reasoning(
            "The opponent defected, so defection is safest.", round_number=1
        )
        self.assertTrue(flags)

    def test_pinned_action_validation(self):
        valid = "<think>[DECISION] Play D</think>\n<action>D</action>"
        self.assertEqual(module.validate_pinned_action(valid, "D"), [])
        self.assertTrue(module.validate_pinned_action(valid, "C"))

    def test_validates_posterior_ev_structure(self):
        valid = (
            "[Prior] Two opponent types each have probability 0.5.\n"
            "[Update] The history leaves this posterior unchanged.\n"
            "[EV] EV(C) = 0.5 * 3 + 0.5 * 0 = 1.5; "
            "EV(D) = 0.5 * 5 + 0.5 * 1 = 3.\n"
            "[DECISION] D"
        )
        self.assertEqual(module.validate_reasoning_structure(valid), [])
        self.assertTrue(module.validate_reasoning_structure("[DECISION] D"))
        no_weighting = (
            "[Prior] opponent type is unknown\n[Update] no history\n"
            "[EV] D is best\n[DECISION] D"
        )
        self.assertIn(
            "[EV] must show probability-weighted arithmetic",
            module.validate_reasoning_structure(no_weighting),
        )
        # Terminal round: the solver is the stage-game best response and the [EV]
        # slot may state raw payoffs, so the arithmetic requirement is waived.
        final_round = (
            "[Prior] opponent type is unknown, several posterior types possible\n"
            "[Update] this is the last round, no future to consider\n"
            "[EV] C: 3.000 (mutual cooperation). D: 5.000 (defection payoff).\n"
            "[DECISION] D"
        )
        self.assertEqual(
            module.validate_reasoning_structure(final_round, is_final_round=True), []
        )
        self.assertTrue(module.validate_reasoning_structure(final_round))

    def test_688_round_audit_and_manual_prior_round_review_record(self):
        # Seven of 688 rounds (1.017%) are heuristic candidates. Each candidate
        # explicitly says round 1 and is recorded here as manually reviewed,
        # legitimate prior-round discussion rather than current-round leakage.
        clean = (
            "[Prior] Two types have probability 0.5. "
            "[Update] The posterior follows the available history. "
            "[EV] EV(C)=0.5 * 3 + 0.5 * 0; EV(D)=0.5 * 5 + 0.5 * 1. "
            "[DECISION] D"
        )
        reviewed_prior = (
            "[Prior] Two types have probability 0.5. "
            "[Update] The opponent cooperated in round 1, a prior round, so my "
            "posterior leans cooperative. "
            "[EV] EV(C)=0.5 * 3 + 0.5 * 0; EV(D)=0.5 * 5 + 0.5 * 1. "
            "[DECISION] D"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "oracle_rounds.jsonl"
            rows = []
            for index in range(688):
                manually_reviewed = 1 <= index <= 7
                rows.append({
                    "id": f"oracle-{index + 1}",
                    "round": index + 1,
                    "reasoning": reviewed_prior if manually_reviewed else clean,
                    "manual_review": (
                        "legitimate-prior-round-discussion" if manually_reviewed else None
                    ),
                })
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            counts = module.audit_file(source)
        self.assertEqual(counts["records"], 688)
        self.assertEqual(counts["flagged_records"], 7)
        self.assertEqual(counts["round_1_flagged"], 0)
        self.assertAlmostEqual(counts["flagged_records"] / counts["records"], 0.01, places=2)


if __name__ == "__main__":
    unittest.main()
