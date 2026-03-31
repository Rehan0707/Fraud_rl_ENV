"""Production-ready fraud detection environment compatible with OpenEnv."""

from __future__ import annotations

from dataclasses import asdict
from typing import Generic, Optional, TypeVar

from .models import FraudAction, FraudObservation, FraudState
from .utils import Transaction, build_rng, is_fraud, sample_transaction

try:
    from openenv import Environment as OpenEnvEnvironment  # type: ignore
except ImportError:
    try:
        from openenv_core import Environment as OpenEnvEnvironment  # type: ignore
    except ImportError:
        ActionT = TypeVar("ActionT")
        ObservationT = TypeVar("ObservationT")
        StateT = TypeVar("StateT")

        class OpenEnvEnvironment(Generic[ActionT, ObservationT, StateT]):
            """Local fallback to keep the project runnable without OpenEnv installed."""

            def reset(self) -> ObservationT:
                raise NotImplementedError

            def step(self, action: ActionT) -> ObservationT:
                raise NotImplementedError

            def state(self) -> StateT:
                raise NotImplementedError


class FraudEnvironment(OpenEnvEnvironment[FraudAction, FraudObservation, FraudState]):
    """Environment for learning fraud-vs-approval decisions on transactions."""

    def __init__(self, max_steps: int = 20, seed: Optional[int] = None) -> None:
        self.max_steps = max_steps
        self._base_seed = seed
        self._rng = build_rng(seed)
        self._step_count = 0
        self._streak = 0
        self._current_transaction: Transaction = sample_transaction(self._rng)

    def reset(self) -> FraudObservation:
        if self._base_seed is not None:
            self._rng = build_rng(self._base_seed)

        self._step_count = 0
        self._streak = 0
        self._current_transaction = sample_transaction(self._rng)
        return FraudObservation(
            state=dict(self._current_transaction),
            reward=0.0,
            done=False,
        )

    def step(self, action: FraudAction) -> FraudObservation:
        if action.action not in (0, 1):
            raise ValueError("FraudAction.action must be 0 (approve) or 1 (flag).")

        transaction = dict(self._current_transaction)
        fraud = is_fraud(transaction)
        predicted_fraud = action.action == 1

        if predicted_fraud and fraud:
            reward = 1.0
            self._streak += 1
            if self._step_count < 5 and self._streak >= 2:
                reward += 0.2
        elif not predicted_fraud and not fraud:
            reward = 1.0
            self._streak = 0
        elif predicted_fraud and not fraud:
            reward = -0.5
            self._streak = 0
        else:
            reward = -1.0
            self._streak = 0

        self._step_count += 1
        done = self._step_count >= self.max_steps

        if not done:
            self._current_transaction = sample_transaction(self._rng)

        next_state = {} if done else dict(self._current_transaction)
        return FraudObservation(state=next_state, reward=reward, done=done)

    def state(self) -> FraudState:
        return FraudState(step_count=self._step_count)

    def snapshot(self) -> dict:
        """Convenience helper for debugging or custom API serialization."""
        return {
            "state": asdict(self.state()),
            "transaction": dict(self._current_transaction),
        }
