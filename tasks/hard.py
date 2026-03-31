"""Hard task: detect fraud from the full ruleset."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_env.environment import FraudEnvironment
from fraud_env.models import FraudAction
from fraud_env.utils import normalize_episode_score


def run_task(episodes: int = 25, seed: int = 23) -> float:
    env = FraudEnvironment(seed=seed)
    scores = []

    for _ in range(episodes):
        observation = env.reset()
        total_reward = 0.0

        while not observation.done:
            should_flag = (
                observation.state["amount"] > 800
                or observation.state["location_risk"] == 1
                or observation.state["frequency"] > 7
            )
            observation = env.step(FraudAction(action=1 if should_flag else 0))
            total_reward += observation.reward

        scores.append(normalize_episode_score(total_reward, env.max_steps))

    return sum(scores) / len(scores)


if __name__ == "__main__":
    print(f"{run_task():.3f}")
