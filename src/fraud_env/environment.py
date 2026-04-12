"""Fraud Investigator Simulator - High-Stakes Decision Environment."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

from .models import FraudAction, FraudObservation, FraudMetrics
from .utils import Transaction, build_rng, is_fraud, sample_transaction

class FraudEnvironment:
    """
    Fraud Investigator Simulator.
    
    The agent acts as an AI fraud investigator in a high-stakes simulation.
    Every decision has consequences:
    - Missed Fraud costs the company money.
    - False Positives (Flagging safe users) damages Customer Trust.
    """

    def __init__(self, max_steps: int = 20, seed: Optional[int] = None) -> None:
        self.max_steps = max_steps
        self._base_seed = seed
        self._rng = build_rng(seed)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Resets the environment to the initial state."""
        if seed is not None:
            self._rng = build_rng(seed)
        elif self._base_seed is not None:
            self._rng = build_rng(self._base_seed)

        self._step_count = 0
        self._total_reward = 0.0
        self._correct_decisions = 0
        self._false_positives = 0
        self._missed_fraud = 0
        
        # Game Mechanics
        self._trust_score = 100.0
        self._consecutive_fraud_flags = 0  # For streak bonus
        
        self._current_transaction: Transaction = sample_transaction(self._rng, step_idx=self._step_count)
        
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        """Standardized observation dictionary."""
        return {
            "transaction": self._current_transaction,
            "step": self._step_count,
            "max_steps": self.max_steps,
            "trust_score": self._trust_score
        }

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Processes an investigator's decision and returns (obs, reward, done, info).
        
        Actions:
            0: APPROVE - Allow the transaction. Increases trust if correct.
            1: FLAG - Flag for investigation. Damages trust if incorrect.
        """
        action_val = action.action if hasattr(action, 'action') else action
        if isinstance(action_val, dict):
            action_val = action_val.get("action", 0)

        if action_val not in (0, 1):
            raise ValueError(f"Invalid action {action_val}. Must be 0 (APPROVE) or 1 (FLAG).")

        actual_fraud = is_fraud(self._current_transaction)
        predicted_fraud = (action_val == 1)
        
        # Base Reward Logic
        reward = 0.0
        if predicted_fraud == actual_fraud:
            reward = 1.0  # Correct decision
            self._correct_decisions += 1
            
            # Streak Bonus for Fraud Detection
            if actual_fraud:
                self._consecutive_fraud_flags += 1
                if self._consecutive_fraud_flags >= 2:
                    reward += 0.2  # Winning streak bonus
            else:
                self._consecutive_fraud_flags = 0
                self._trust_score = min(100.0, self._trust_score + 1.0) # Passive trust recovery
                
        elif predicted_fraud and not actual_fraud:
            reward = -0.5  # False Positive (User Friction)
            self._false_positives += 1
            self._trust_score = max(0.0, self._trust_score - 10.0) # Heavy trust penalty
            self._consecutive_fraud_flags = 0
        else:
            reward = -1.0  # Missed Fraud (Financial Loss)
            self._missed_fraud += 1
            self._trust_score = max(0.0, self._trust_score - 5.0) # Moderate trust penalty (theft impacts UX)
            self._consecutive_fraud_flags = 0

        self._total_reward += reward
        self._step_count += 1
        done = self._step_count >= self.max_steps or self._trust_score <= 0

        # High-Impact Info Dictionary
        info = {
            "correct": predicted_fraud == actual_fraud,
            "fraud": actual_fraud,
            "decision": "flag" if predicted_fraud else "approve",
            "trust_score": self._trust_score,
            "accuracy": self._correct_decisions / self._step_count if self._step_count > 0 else 0.0,
            "metrics": {
                "false_positives": self._false_positives,
                "missed_fraud": self._missed_fraud,
                "streak": self._consecutive_fraud_flags
            }
        }

        if not done:
            # Difficulty scales with step count (Narrative Shift)
            self._current_transaction = sample_transaction(self._rng, step_idx=self._step_count)

        return self._get_obs(), reward, done, info

    def get_metrics(self) -> FraudMetrics:
        """Returns consolidated metrics for the session."""
        accuracy = (self._correct_decisions / self._step_count) if self._step_count > 0 else 0.0
        return FraudMetrics(
            accuracy=accuracy,
            false_positives=self._false_positives,
            missed_fraud=self._missed_fraud,
            total_reward=self._total_reward,
            trust_score=self._trust_score
        )


