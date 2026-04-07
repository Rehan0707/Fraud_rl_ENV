"""FastAPI application exposing the fraud environment."""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Setup paths to find internal modules
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


@app.get("/")
def root() -> dict:
    return {
        "message": "Welcome to the Fraud Detection Decision Environment",
        "status": "Running",
        "docs": "/docs",
        "endpoints": ["/health", "/reset", "/step"],
        "openenv_compliant": True,
    }


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
