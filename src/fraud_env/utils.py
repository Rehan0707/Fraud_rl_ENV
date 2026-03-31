"""Utility helpers for transaction generation and score normalization."""

from __future__ import annotations

import random
from typing import Dict, Optional

Transaction = Dict[str, int]


def build_rng(seed: Optional[int] = None) -> random.Random:
    return random.Random(seed)


def sample_transaction(rng: random.Random) -> Transaction:
    """Generate realistic but bounded transaction features."""
    amount_band = rng.choices(
        population=["low", "normal", "high", "spike"],
        weights=[0.25, 0.45, 0.20, 0.10],
        k=1,
    )[0]

    if amount_band == "low":
        amount = rng.randint(10, 120)
    elif amount_band == "normal":
        amount = rng.randint(121, 620)
    elif amount_band == "high":
        amount = rng.randint(621, 850)
    else:
        amount = rng.randint(851, 1000)

    frequency = rng.choices(
        population=list(range(1, 11)),
        weights=[12, 11, 10, 10, 9, 8, 7, 6, 5, 4],
        k=1,
    )[0]

    location_risk = 1 if rng.random() < 0.28 else 0

    return {
        "amount": amount,
        "location_risk": location_risk,
        "frequency": frequency,
    }


def is_fraud(transaction: Transaction) -> bool:
    return (
        transaction["amount"] > 800
        or transaction["location_risk"] == 1
        or transaction["frequency"] > 7
    )


def normalize_episode_score(total_reward: float, max_steps: int) -> float:
    """Map the possible reward range to 0.0..1.0 for task scoring."""
    minimum = -1.0 * max_steps
    maximum = 1.2 * max_steps
    normalized = (total_reward - minimum) / (maximum - minimum)
    return max(0.0, min(1.0, normalized))
