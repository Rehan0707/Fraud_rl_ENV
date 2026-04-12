"""Training script for the Fraud Investigator Simulator DQN agent."""

import random
import sys
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fraud_env.environment import FraudEnvironment
from fraud_env.model import DQN, preprocess_observation

# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 64
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
EPISODES = 500

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def train():
    env = FraudEnvironment()
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    epsilon = EPS_START

    for episode in range(EPISODES):
        obs = env.reset()
        state = preprocess_observation(obs)
        total_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randint(0, 1)
            else:
                with torch.no_grad():
                    action_idx = policy_net(state.unsqueeze(0)).argmax().item()

            # Take Step
            next_obs, reward, done, info = env.step(action_idx)
            
            next_state = preprocess_observation(next_obs) if not done else None
            
            memory.push(state, action_idx, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Perform optimization step
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                b_state, b_action, b_reward, b_next_state, b_done = zip(*transitions)
                
                b_state = torch.stack(b_state)
                b_action = torch.tensor(b_action).unsqueeze(1)
                b_reward = torch.tensor(b_reward, dtype=torch.float32)
                b_done = torch.tensor(b_done, dtype=torch.float32)
                
                current_q = policy_net(b_state).gather(1, b_action).squeeze(1)
                
                max_next_q = torch.zeros(BATCH_SIZE)
                non_final_mask = torch.tensor([s is not None for s in b_next_state], dtype=torch.bool)
                if non_final_mask.any():
                    non_final_next_states = torch.stack([s for s in b_next_state if s is not None])
                    max_next_q[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
                
                expected_q = b_reward + (GAMMA * max_next_q * (1 - b_done))
                loss = nn.MSELoss()(current_q, expected_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{EPISODES}, Epsilon: {epsilon:.2f}, Reward: {total_reward:.2f}")

    # Save the model
    torch.save(policy_net.state_dict(), "model.pth")
    print("Training complete. Model saved to model.pth")

if __name__ == "__main__":
    train()

