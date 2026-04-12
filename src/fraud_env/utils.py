"""Utility helpers for transaction generation and score normalization."""

from __future__ import annotations

import random
from typing import Dict, Optional, Any

Transaction = Dict[str, Any]


def build_rng(seed: Optional[int] = None) -> random.Random:
    return random.Random(seed)


def sample_transaction(rng: random.Random, step_idx: int = 0) -> Transaction:
    """
    Generate professional transaction features with progressive narrative scaling.
    As step_idx increases (0-20), the 'pressure' increases: 
    Amounts get higher, and risk markers become more ambiguous.
    """
    
    # Narrative Scaling Factor (0.0 to 1.0)
    pressure = min(1.0, step_idx / 20.0)
    
    # Base Categorical data
    merchant_categories = ["electronics", "grocery", "restaurant", "travel", "luxury", "utilities"]
    merchant = rng.choice(merchant_categories)
    
    # Amount logic based on category + pressure
    # As pressure increases, transaction amounts spike (simulating large-scale theft)
    amount_multiplier = 1.0 + (pressure * 0.5)
    if merchant == "luxury":
        amount = rng.randint(800, int(5000 * amount_multiplier))
    elif merchant == "travel":
        amount = rng.randint(200, int(3000 * amount_multiplier))
    elif merchant == "electronics":
        amount = rng.randint(50, int(1500 * amount_multiplier))
    else:
        amount = rng.randint(5, int(200 * amount_multiplier))

    # Risk markers - Probability increases slightly with pressure
    location_risk_prob = 0.15 + (pressure * 0.1)
    location_risk = 1 if rng.random() < location_risk_prob else 0
    
    frequency_boost = int(pressure * 10)
    frequency_24h = rng.randint(1, 15 + frequency_boost)
    
    # New features
    is_new_device = 1 if rng.random() < (0.1 + pressure * 0.15) else 0
    user_age = rng.randint(18, 85)
    hour_of_day = rng.randint(0, 23)

    return {
        "amount": amount,
        "merchant_category": merchant,
        "location_risk": location_risk,
        "frequency_24h": frequency_24h,
        "is_new_device": is_new_device,
        "user_age": user_age,
        "hour_of_day": hour_of_day,
    }


def is_fraud(transaction: Transaction) -> bool:
    """Hidden fraud logic (the pattern the agent needs to learn)."""
    # High-risk combinations
    if transaction["location_risk"] == 1 and transaction["amount"] > 1000:
        return True
    if transaction["is_new_device"] == 1 and transaction["amount"] > 500:
        return True
    if transaction["frequency_24h"] > 10 and transaction["merchant_category"] in ["electronics", "luxury"]:
        return True
    if 1 <= transaction["hour_of_day"] <= 4 and transaction["amount"] > 2000:
        return True
    
    # Base random fraud (noise)
    return random.random() < 0.02


def normalize_episode_score(total_reward: float, max_steps: int) -> float:
    """Map the possible reward range to 0.0..1.0 for task scoring."""
    # Adjusted maximum to account for streak bonuses
    minimum = -1.0 * max_steps
    maximum = 1.5 * max_steps 
    normalized = (total_reward - minimum) / (maximum - minimum)
    return max(0.0, min(1.0, normalized))


