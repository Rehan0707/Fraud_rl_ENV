# AI Medical Decision Environment

## Problem Description

This project is a production-ready OpenEnv environment for a medical triage hackathon. An AI agent receives patient symptoms, age, and severity, then decides:

- Disease classification
- Medicine suggestion
- Urgency level
- Health advice

The environment is designed for quick benchmarking, offline demos, and API-backed experiments with OpenAI models.

## Folder Structure

```text
project/
├── inference.py
├── openenv.yaml
├── requirements.txt
├── README.md
├── Dockerfile
├── env/
│   ├── __init__.py
│   ├── env.py
│   ├── graders.py
│   └── utils.py
└── tests/
    └── test_env.py
```

## Environment Design

### Observation Space

- `symptoms`: `string`
- `age`: `int`
- `severity`: `low | medium | high`

### Action Space

- `disease_prediction`
- `medicine_suggestion`
- `urgency_level`
- `advice_text`

### API Surface

The environment class is implemented in [`env/env.py`](/Users/rehansanadi/Downloads/fraud_rl_env/project/env/env.py) and provides:

- `reset()`
- `step(action)`
- `state()`

`step()` returns the required tuple:

```python
(next_obs, reward, terminated, truncated, info)
```

## Tasks

Exactly three tasks are provided:

1. `task_1_easy`: disease classification only
2. `task_2_medium`: disease classification plus medicine suggestion
3. `task_3_hard`: full pipeline with disease, medicine, urgency, and advice

Each task includes:

- An example input
- An expected output example
- A dedicated grader function

Task metadata is declared in [`env/utils.py`](/Users/rehansanadi/Downloads/fraud_rl_env/project/env/utils.py) and referenced by [`openenv.yaml`](/Users/rehansanadi/Downloads/fraud_rl_env/project/openenv.yaml).

## Reward System

Scoring is implemented in [`env/graders.py`](/Users/rehansanadi/Downloads/fraud_rl_env/project/env/graders.py):

- Correct disease: `+0.3`
- Correct medicine: `+0.3`
- Correct urgency: `+0.2`
- Good advice quality: `+0.2`
- Dangerous advice: `-0.5`
- Completely wrong: `-1.0`

The grader computes a raw score using the hackathon rules, then clips the final environment reward into `0.0` to `1.0` so evaluation stays stable and RL-friendly. Raw penalties remain visible in `info`.

## Sample Dataset

A built-in sample dataset of patient cases is included in [`env/utils.py`](/Users/rehansanadi/Downloads/fraud_rl_env/project/env/utils.py). It covers:

- Common cold
- Influenza
- Gastritis
- Migraine
- Asthma exacerbation
- Hypertension

## Inference Runner

[`inference.py`](/Users/rehansanadi/Downloads/fraud_rl_env/project/inference.py) supports two modes:

- `mock`: offline rule-based inference for guaranteed hackathon demos
- `openai`: live inference using the OpenAI API when `OPENAI_API_KEY` is available

It runs 50 to 100 episodes comfortably, prints per-episode logs, and reports the average score.

### Run Batch Inference

```bash
cd /Users/rehansanadi/Downloads/fraud_rl_env/project
python inference.py --task task_3_hard --episodes 60 --backend mock
```

### Run with OpenAI API

```bash
cd /Users/rehansanadi/Downloads/fraud_rl_env/project
export OPENAI_API_KEY="your_api_key_here"
python inference.py --task task_3_hard --episodes 60 --backend openai --model gpt-4.1-mini
```

## Gradio UI

A simple Gradio demo is included in [`inference.py`](/Users/rehansanadi/Downloads/fraud_rl_env/project/inference.py).

```bash
cd /Users/rehansanadi/Downloads/fraud_rl_env/project
python inference.py --ui --host 0.0.0.0 --port 7860
```

Then open [http://localhost:7860](http://localhost:7860).

## Setup Instructions

```bash
cd /Users/rehansanadi/Downloads/fraud_rl_env/project
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Docker

Build:

```bash
cd /Users/rehansanadi/Downloads/fraud_rl_env/project
docker build -t ai-medical-decision-env .
```

Run:

```bash
docker run -p 7860:7860 ai-medical-decision-env
```

This starts the Gradio UI on port `7860`.
