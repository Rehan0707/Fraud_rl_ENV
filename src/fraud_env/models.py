"""Typed models used by the fraud detection environment."""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class FraudAction:
    action: int  # 0: APPROVE, 1: FLAG


@dataclass
class FraudObservation:
    transaction: Dict[str, Any]
    step: int
    max_steps: int
    trust_score: float


@dataclass
class FraudMetrics:
    accuracy: float
    false_positives: int
    missed_fraud: int
    total_reward: float
    trust_score: float


