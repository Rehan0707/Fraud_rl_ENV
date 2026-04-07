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

def run_task_with_model(task_fn, model, task_name: str, seed: int) -> float:
    """Wrapper to run a task using the trained model with structured logging."""
    from fraud_env.environment import FraudEnvironment
    from fraud_env.models import FraudAction
    from fraud_env.utils import normalize_episode_score

    env = FraudEnvironment(seed=seed)
    scores = []
    
    # Run multiple episodes to get a stable score
    episodes = 25
    for _ in range(episodes):
        obs = env.reset()
        total_reward = 0
        step_idx = 0
        
        print(f"[START] task={task_name}", flush=True)
        
        while not obs.done:
            state = preprocess_observation(obs.state)
            with torch.no_grad():
                # Get the action with the highest probability
                action_idx = model(state.unsqueeze(0)).argmax().item()
            
            obs = env.step(FraudAction(action=action_idx))
            total_reward += obs.reward
            
            print(f"[STEP] step={step_idx} reward={obs.reward:.2f}", flush=True)
            step_idx += 1
            
        score = normalize_episode_score(total_reward, env.max_steps)
        scores.append(score)
        
        print(f"[END] task={task_name} score={score:.3f} steps={step_idx}", flush=True)
        
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
        # Run tasks with appropriate seeds from task scripts
        easy_score = run_task_with_model(run_easy, model, "Easy", seed=11)
        medium_score = run_task_with_model(run_medium, model, "Medium", seed=17)
        hard_score = run_task_with_model(run_hard, model, "Hard", seed=23)
    else:
        # Fallback to rules-based tasks (baseline) - no structured logging here
        # but we could add it if necessary.
        easy_score = run_easy()
        medium_score = run_medium()
        hard_score = run_hard()

    print("\nFraud Detection Decision Environment Benchmark")
    print(f"Mode              : {'AI Model' if model else 'Rule-based Baseline'}")
    print(f"Easy Task Score   : {easy_score:.3f}")
    print(f"Medium Task Score : {medium_score:.3f}")
    print(f"Hard Task Score   : {hard_score:.3f}")


if __name__ == "__main__":
    main()
