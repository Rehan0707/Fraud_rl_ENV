"""Run all benchmark tasks for the fraud detection environment."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tasks.easy import run_task as run_easy
from tasks.hard import run_task as run_hard
from tasks.medium import run_task as run_medium
from fraud_env.model import DQN, preprocess_observation

def run_task_with_model(task_fn, model):
    """Wrapper to run a task using the trained model if available."""
    if model is None:
        return task_fn()
    
    # We need to pass the model to the task function
    # But current task functions don't support it.
    # I'll create a local patched version for execution.
    from fraud_env.environment import FraudEnvironment
    from fraud_env.models import FraudAction
    from fraud_env.utils import normalize_episode_score

    # Re-implementing a simple run loop for the model
    env = FraudEnvironment()
    scores = []
    
    for _ in range(25):
        obs = env.reset()
        total_reward = 0
        while not obs.done:
            state = preprocess_observation(obs.state)
            with torch.no_grad():
                action_idx = model(state.unsqueeze(0)).argmax().item()
            obs = env.step(FraudAction(action=action_idx))
            total_reward += obs.reward
        scores.append(normalize_episode_score(total_reward, env.max_steps))
    return sum(scores) / len(scores)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to trained model.pth")
    args = parser.parse_args()

    model = None
    if args.model:
        model = DQN()
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
        model.eval()
        print(f"Loaded model from {args.model}")

    if model:
        easy_score = run_task_with_model(run_easy, model)
        medium_score = run_task_with_model(run_medium, model)
        hard_score = run_task_with_model(run_hard, model)
    else:
        easy_score = run_easy()
        medium_score = run_medium()
        hard_score = run_hard()

    print("Fraud Detection Decision Environment Benchmark")
    print(f"Mode              : {'AI Model' if model else 'Rule-based Baseline'}")
    print(f"Easy Task Score   : {easy_score:.3f}")
    print(f"Medium Task Score : {medium_score:.3f}")
    print(f"Hard Task Score   : {hard_score:.3f}")


if __name__ == "__main__":
    main()
