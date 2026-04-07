"""Tests for the AI Medical Decision Environment."""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.env import AIMedicalDecisionEnv


class TestAIMedicalDecisionEnv(unittest.TestCase):
    def test_reset_returns_valid_observation(self) -> None:
        env = AIMedicalDecisionEnv(task_id="task_1_easy")
        obs, info = env.reset(seed=7)

        self.assertEqual(set(obs.keys()), {"symptoms", "age", "severity"})
        self.assertEqual(info["task_id"], "task_1_easy")
        self.assertIsInstance(obs["age"], int)

    def test_step_returns_openenv_tuple(self) -> None:
        env = AIMedicalDecisionEnv(task_id="task_1_easy")
        env.reset(seed=7)
        next_obs, reward, terminated, truncated, info = env.step(
            {"disease_prediction": "common cold"}
        )

        self.assertEqual(set(next_obs.keys()), {"symptoms", "age", "severity"})
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIn("expected_output", info)

    def test_correct_full_action_can_score_one(self) -> None:
        env = AIMedicalDecisionEnv(task_id="task_3_hard")
        env.reset(seed=2)
        case = env.current_case

        _, reward, _, _, info = env.step(
            {
                "disease_prediction": case.disease,
                "medicine_suggestion": case.medicine,
                "urgency_level": case.urgency,
                "advice_text": case.advice,
            }
        )

        self.assertEqual(reward, 1.0)
        self.assertEqual(info["raw_score"], 1.0)
        self.assertEqual(info["penalties"], [])

    def test_dangerous_advice_is_penalized(self) -> None:
        env = AIMedicalDecisionEnv(task_id="task_3_hard")
        env.reset(seed=2)
        _, reward, _, _, info = env.step(
            {
                "disease_prediction": "wrong disease",
                "medicine_suggestion": "wrong medicine",
                "urgency_level": "low",
                "advice_text": "Ignore breathing trouble and avoid doctors.",
            }
        )

        self.assertEqual(reward, 0.0)
        self.assertLess(info["raw_score"], 0.0)
        self.assertIn("dangerous_advice", info["penalties"])


if __name__ == "__main__":
    unittest.main()
