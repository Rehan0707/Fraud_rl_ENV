---
title: Fraud Detection Decision Environment
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Fraud Detection Decision Environment

## Overview

This project provides a production-ready OpenEnv-compatible reinforcement learning environment for fintech fraud detection. An agent reviews simulated transaction features and decides whether to approve the transaction or flag it as fraudulent. The environment is packaged as a FastAPI service, Dockerized for deployment, and structured for Hugging Face Spaces. It also includes integration for proxy model evaluation and Hackathon auto-graders.

## Problem Explanation

Financial platforms need low-latency fraud decisions under uncertainty. This environment simulates that workflow with realistic bounded transaction signals:

- `amount`: transaction amount from 10 to 1000
- `location_risk`: binary high-risk location indicator
- `frequency`: recent transaction frequency from 1 to 10

Fraud is defined by the domain rule:

- `amount > 800`, or
- `location_risk == 1`, or
- `frequency > 7`

## Action Space

- `0`: approve transaction
- `1`: flag as fraud

## Observation Space

Each environment response returns:

- `state`: dictionary containing `amount`, `location_risk`, and `frequency`
- `reward`: float reward for the last action
- `done`: whether the episode has ended

Internal environment state is also tracked through the typed `FraudState(step_count: int)` model.

## Reward Logic

- Correct fraud detection: `+1.0`
- Correct approval: `+1.0`
- False positive: `-0.5`
- Missed fraud: `-1.0`
- Early fraud streak bonus: `+0.2` for consecutive correct fraud flags in the first five steps

Each episode ends after `20` decisions.

## APIs and Compliance

To comply with hackathon automated "Agentic Evaluation" processes, this project utilizes and exposes the following APIs:
- **FastAPI / REST Environment API**: Serves the reinforcement learning environment endpoints (`/reset`, `/step`).
- **OpenAI Proxy Client**: Built-in support to route LLM decisions utilizing `API_BASE_URL` and `API_KEY` for evaluation scenarios.
- **OpenEnv Sync Client**: Ensures standard interaction with remote URL instances.
- **Structured Debug Logging**: Scripts print formatted `--- START ---`, `--- STEP n ---`, and `--- END ---` blocks so external graders can correctly trace episodes.

## Project Structure

```text
fraud_rl_env/
├── src/fraud_env/
│   ├── environment.py
│   ├── model.py           # DQN Model definition
│   ├── models.py          # Data classes and Schemas
│   └── server/
│       └── app.py         # FastAPI Service
├── train.py               # DQN reinforcement learning training script
├── evaluate.py            # Environment evaluation & structured logging
├── inference.py           # Agentic inference and proxy evaluation logic
├── openenv.yaml           # Environment metadata config
├── Dockerfile             # Container configuration
├── requirements.txt       
└── README.md
```

## Setup Instructions

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run Locally

**1. Run the API server:**

```bash
uvicorn fraud_env.server.app:app --host 0.0.0.0 --port 7860
```

Available endpoints:
- `GET /health`
- `POST /reset`
- `POST /step`

Example local HTTP request:

```bash
curl -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": 1}'
```

**2. Train the RL Agent:**
```bash
python train.py
```
This will train the DQN and generate a `model.pth` weights file.

**3. Run the Hackathon Evaluation:**
```bash
python evaluate.py
```
This script respects the `API_BASE_URL` if testing a remote instance, or falls back to initializing the local Python environment wrapper.

## Docker

Build and run the system locally:

```bash
docker build -t fraud-rl-env .
docker run -p 7860:7860 fraud-rl-env
```

## Hugging Face Spaces Deployment

This project is explicitly ready for a Docker Hugging Face Space:

1. Create a new Hugging Face Space and choose `Docker` as the SDK.
2. Upload this repository's contents.
3. Ensure the Space exposes port `7860` (as defined in the Dockerfile).
4. Hugging Face will build the container image automatically.

## Production Notes

- Typed models are defined with dataclasses/Pydantic for clean environment contracts.
- A local OpenEnv fallback base class is included so the project still runs if `openenv-core` is temporarily unreachable.
- Transaction generation is randomized with weighted sampling to realistically simulate production class imbalances (fraud is rarer than regular transactions).
