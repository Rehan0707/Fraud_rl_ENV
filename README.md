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

This project provides a production-ready OpenEnv-compatible reinforcement learning environment for fintech fraud detection. An agent reviews simulated transaction features and decides whether to approve the transaction or flag it as fraudulent. The environment is packaged as a FastAPI service, Dockerized for deployment, and structured for Hugging Face Spaces.

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

## Project Structure

```text
fraud_rl_env/
├── src/fraud_env/
│   ├── __init__.py
│   ├── environment.py
│   ├── models.py
│   ├── utils.py
│   └── server/
│       └── app.py
├── tasks/
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup Instructions

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run Locally

Run the benchmark tasks:

```bash
PYTHONPATH=src python inference.py
```

Run the API server:

```bash
PYTHONPATH=src uvicorn fraud_env.server.app:app --host 0.0.0.0 --port 7860
```

Available endpoints:

- `GET /health`
- `POST /reset`
- `POST /step`

Example request:

```bash
curl -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": 1}'
```

## Task Benchmarks

- `tasks/easy.py`: amount-only fraud detection
- `tasks/medium.py`: amount plus location-based detection
- `tasks/hard.py`: full fraud ruleset

Each task returns a normalized score between `0.0` and `1.0`.

## Docker

Build and run:

```bash
docker build -t fraud-rl-env .
docker run -p 7860:7860 fraud-rl-env
```

## Hugging Face Spaces Deployment

This project is ready for a Docker Space:

1. Create a new Hugging Face Space and choose `Docker`.
2. Upload this repository.
3. Ensure the Space exposes port `7860`.
4. Hugging Face will build the `Dockerfile` automatically.

## Production Notes

- Typed models are defined with dataclasses for clean environment contracts.
- A local OpenEnv fallback base class is included so the project still runs if `openenv-core` is temporarily unavailable.
- Transaction generation is randomized with weighted sampling to better simulate production transaction mix.
- The codebase is modular, container-ready, and benchmarked through the included inference script.
# Fraud_rl_ENV
