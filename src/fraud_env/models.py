"""Typed models used by the fraud detection environment."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class FraudAction:
    action: int


@dataclass
class FraudObservation:
    state: Dict[str, int]
    reward: float
    done: bool


@dataclass
class FraudState:
    step_count: int
