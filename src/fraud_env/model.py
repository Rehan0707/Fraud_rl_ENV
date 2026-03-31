import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-Network for fraud detection."""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def preprocess_observation(obs: dict) -> torch.Tensor:
    """Convert observation dictionary to normalized tensor."""
    # amount: 10-1000 -> normalize to [0, 1]
    amount = (obs.get("amount", 0) - 10) / 990.0
    # location_risk: 0 or 1
    location_risk = float(obs.get("location_risk", 0))
    # frequency: 1-10 -> normalize to [0, 1]
    frequency = (obs.get("frequency", 1) - 1) / 9.0
    
    return torch.tensor([amount, location_risk, frequency], dtype=torch.float32)
