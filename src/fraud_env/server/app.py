"""FastAPI application exposing the fraud environment."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI

from fraud_env.environment import FraudEnvironment
from fraud_env.models import FraudAction

app = FastAPI(
    title="Fraud Detection Decision Environment",
    description="OpenEnv-compatible RL environment for financial fraud detection.",
    version="1.0.0",
)

environment = FraudEnvironment()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset")
def reset() -> dict:
    return asdict(environment.reset())


@app.post("/step")
def step(action: FraudAction) -> dict:
    return asdict(environment.step(action))


def main() -> None:
    """Entry point for the fraud environment server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
