"""FastAPI application exposing the Fraud Investigator Simulator."""

from __future__ import annotations

from dataclasses import asdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fraud_env.environment import FraudEnvironment
from fraud_env.models import FraudAction

app = FastAPI(
    title="Fraud Investigator Simulator",
    description="Standardized RL environment for financial fraud detection.",
    version="1.1.0",
)

# Enable CORS for local testing with dashboard.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

environment = FraudEnvironment()


@app.get("/")
def root() -> dict:
    return {
        "message": "Welcome to the Fraud Investigator Simulator",
        "status": "Active",
        "docs": "/docs",
        "openenv_compliant": True,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset")
def reset() -> dict:
    """Resets the environment and returns the initial observation."""
    return environment.reset()


@app.post("/step")
def step(action: FraudAction) -> dict:
    """Processes an action and returns (observation, reward, done, info)."""
    obs, reward, done, info = environment.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/metrics")
def metrics() -> dict:
    """Returns current session metrics including Customer Trust."""
    return asdict(environment.get_metrics())


def main() -> None:
    """Entry point for the fraud environment server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

