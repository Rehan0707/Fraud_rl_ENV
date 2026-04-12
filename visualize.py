"""Visualization script for Fraud Investigator Simulator results."""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Setup paths
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_env.environment import FraudEnvironment
from fraud_env.model import DQN, preprocess_observation

def run_evaluation(episodes: int = 20, model_path: str = "model.pth"):
    # Load Model
    model = DQN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    env = FraudEnvironment()
    
    rewards = []
    trust_scores = []
    accuracies = []
    false_positives = []
    missed_frauds = []

    print(f"Running {episodes} evaluation episodes...")
    for _ in tqdm(range(episodes)):
        obs = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            state_tensor = preprocess_observation(obs)
            with torch.no_grad():
                action_idx = model(state_tensor.unsqueeze(0)).argmax().item()
            
            obs, reward, done, info = env.step(action_idx)
            ep_reward += reward
            
        m = env.get_metrics()
        rewards.append(ep_reward)
        trust_scores.append(m.trust_score)
        accuracies.append(m.accuracy)
        false_positives.append(m.false_positives)
        missed_frauds.append(m.missed_fraud)

    return {
        "rewards": rewards,
        "trust_scores": trust_scores,
        "accuracies": accuracies,
        "false_positives": false_positives,
        "missed_frauds": missed_frauds
    }

def plot_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fraud Investigator Simulator - Performance Analysis', fontsize=20, fontweight='bold', color='#1a237e')

    # 1. Rewards Distribution
    axes[0, 0].hist(results["rewards"], bins=10, color='#3f51b5', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Episode Rewards Distribution', fontsize=14)
    axes[0, 0].set_xlabel('Total Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Trust Score
    axes[0, 1].boxplot(results["trust_scores"], patch_artist=True, 
                     boxprops=dict(facecolor='#4caf50', color='black'),
                     medianprops=dict(color='white'))
    axes[0, 1].set_title('Final Customer Trust Score', fontsize=14)
    axes[0, 1].set_ylabel('Trust Score (%)')
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Accuracy vs False Positives
    axes[1, 0].scatter(results["false_positives"], results["accuracies"], color='#e91e63', s=100, alpha=0.6)
    axes[1, 0].set_title('Accuracy vs False Positives', fontsize=14)
    axes[1, 0].set_xlabel('Total False Positives')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].grid(linestyle='--', alpha=0.7)

    # 4. Error Metrics
    labels = ['False Positives', 'Missed Frauds']
    counts = [np.mean(results["false_positives"]), np.mean(results["missed_frauds"])]
    axes[1, 1].bar(labels, counts, color=['#ff9800', '#f44336'], alpha=0.8, edgecolor='black')
    axes[1, 1].set_title('Average errors per Episode', fontsize=14)
    axes[1, 1].set_ylabel('Average Count')
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('performance_report.png', dpi=300)
    print("\nPerformance report saved as performance_report.png")

if __name__ == "__main__":
    results = run_evaluation(episodes=50)
    plot_results(results)
