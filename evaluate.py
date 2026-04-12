"""Evaluation script for the Fraud Investigator Simulator."""

import os
import sys
import torch
import json
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_env.environment import FraudEnvironment
from fraud_env.model import DQN, preprocess_observation
from fraud_env.models import FraudAction

def evaluate(episodes: int = 1, model_path: str = "model.pth") -> None:
    # Load Model
    model = DQN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    env = FraudEnvironment()

    print("--- START ---")
    
    total_metrics = {
        "accuracy": 0.0,
        "false_positives": 0,
        "missed_fraud": 0,
        "total_reward": 0.0,
        "trust_score": 0.0
    }

    for ep in range(episodes):
        obs = env.reset()
        done = False
        step_idx = 0
        
        while not done:
            # Plan Action
            state_tensor = preprocess_observation(obs)
            with torch.no_grad():
                action_idx = model(state_tensor.unsqueeze(0)).argmax().item()
            
            # Take Step
            obs, reward, done, info = env.step(action_idx)
            
            # Log Step (Structured for OpenEnv Grader)
            print(f"--- STEP {step_idx} ---")
            print(f"Action: {action_idx} ({info['decision'].upper()})")
            print(f"Fraud: {info['fraud']}")
            print(f"Correct: {info['correct']}")
            print(f"Trust: {info['trust_score']}")
            print(f"Reward: {reward}")
            
            step_idx += 1

        # Collect final metrics
        m = env.get_metrics()
        total_metrics["accuracy"] += m.accuracy
        total_metrics["false_positives"] += m.false_positives
        total_metrics["missed_fraud"] += m.missed_fraud
        total_metrics["total_reward"] += m.total_reward
        total_metrics["trust_score"] += m.trust_score

    print("--- END ---")

    # Print Final Summary Metrics
    avg_accuracy = total_metrics["accuracy"] / episodes
    avg_trust = total_metrics["trust_score"] / episodes
    print(f"\nFinal Metrics ({episodes} episodes):")
    print(f"Accuracy: {avg_accuracy:.2%}")
    print(f"Average Final Trust: {avg_trust:.1f}/100")
    print(f"Total False Positives: {total_metrics['false_positives']}")
    print(f"Total Missed Fraud: {total_metrics['missed_fraud']}")
    print(f"Average Reward: {total_metrics['total_reward'] / episodes:.2f}")


if __name__ == "__main__":
    evaluate(episodes=1, model_path="model.pth")

