"""Inference runner for the AI Medical Decision Environment."""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional at runtime
    OpenAI = None  # type: ignore[assignment]

from env.env import AIMedicalDecisionEnv
from env.utils import TASK_SPECS


SYSTEM_PROMPT = """You are a careful AI triage assistant.
Return valid JSON with keys:
- disease_prediction
- medicine_suggestion
- urgency_level
- advice_text

Rules:
- Be concise and safe.
- Never suggest dangerous or illegal advice.
- If unsure, choose a conservative urgency level.
"""


def mock_agent(observation: dict[str, Any]) -> dict[str, str]:
    """Rule-based fallback agent for offline or hackathon demos."""

    symptoms = observation["symptoms"].lower()
    severity = observation["severity"].lower()

    if "wheezing" in symptoms or "shortness of breath" in symptoms:
        return {
            "disease_prediction": "asthma exacerbation",
            "medicine_suggestion": "albuterol inhaler",
            "urgency_level": "high",
            "advice_text": "Use the rescue inhaler immediately and seek urgent care if breathing remains difficult.",
        }
    if "fever" in symptoms and "body aches" in symptoms:
        return {
            "disease_prediction": "influenza",
            "medicine_suggestion": "oseltamivir",
            "urgency_level": "medium",
            "advice_text": "Rest, stay hydrated, isolate from others, and contact a doctor if symptoms worsen.",
        }
    if "runny nose" in symptoms or "sneezing" in symptoms:
        return {
            "disease_prediction": "common cold",
            "medicine_suggestion": "paracetamol",
            "urgency_level": "low",
            "advice_text": "Rest, drink warm fluids, monitor your symptoms, and seek care if breathing worsens.",
        }
    if "burning stomach pain" in symptoms or "acid reflux" in symptoms:
        return {
            "disease_prediction": "gastritis",
            "medicine_suggestion": "omeprazole",
            "urgency_level": "medium",
            "advice_text": "Avoid spicy food, eat light meals, and visit a clinic if the pain persists.",
        }
    if "one-sided headache" in symptoms or "sensitivity to light" in symptoms:
        return {
            "disease_prediction": "migraine",
            "medicine_suggestion": "ibuprofen",
            "urgency_level": "medium",
            "advice_text": "Rest in a dark room, hydrate well, and seek urgent care if weakness or confusion appears.",
        }
    if "blood pressure" in symptoms or severity == "high":
        return {
            "disease_prediction": "hypertension",
            "medicine_suggestion": "amlodipine",
            "urgency_level": "high",
            "advice_text": "Recheck blood pressure and arrange same-day medical evaluation.",
        }

    return {
        "disease_prediction": "common cold",
        "medicine_suggestion": "paracetamol",
        "urgency_level": "low",
        "advice_text": "Rest, drink fluids, and consult a doctor if symptoms get worse.",
    }


def openai_agent(observation: dict[str, Any], model: str) -> dict[str, str]:
    """Call the OpenAI API if credentials are available."""

    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Use the mock agent or install dependencies.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Use the mock agent or export your API key.")

    client = OpenAI(api_key=api_key)
    user_prompt = (
        "Patient observation:\n"
        f"{json.dumps(observation, indent=2)}\n\n"
        "Respond with JSON only."
    )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.output_text
    return json.loads(content)


def agent_predict(observation: dict[str, Any], backend: str, model: str) -> dict[str, str]:
    """Dispatch to the selected inference backend."""

    if backend == "openai":
        return openai_agent(observation, model)
    return mock_agent(observation)


def trim_action_for_task(action: dict[str, str], task_id: str) -> dict[str, str]:
    """Keep only the required fields for the current task."""

    required = TASK_SPECS[task_id].required_fields
    return {field: action.get(field, "") for field in required}


def run_episodes(task_id: str, episodes: int, backend: str, model: str, verbose: bool = True) -> list[dict[str, Any]]:
    """Run multiple episodes and return structured logs."""

    env = AIMedicalDecisionEnv(task_id=task_id)
    results: list[dict[str, Any]] = []

    for episode in range(1, episodes + 1):
        observation, reset_info = env.reset()
        action = trim_action_for_task(agent_predict(observation, backend=backend, model=model), task_id)
        _, reward, terminated, truncated, info = env.step(action)
        row = {
            "episode": episode,
            "task_id": task_id,
            "case_id": reset_info["case_id"],
            "observation": observation,
            "action": action,
            "reward": reward,
            "raw_score": info["raw_score"],
            "penalties": info["penalties"],
            "terminated": terminated,
            "truncated": truncated,
        }
        results.append(row)
        if verbose:
            print(
                f"[Episode {episode:03d}] task={task_id} reward={reward:.2f} "
                f"raw={info['raw_score']:.2f} penalties={info['penalties'] or 'none'}"
            )

    average = mean(item["reward"] for item in results)
    raw_average = mean(item["raw_score"] for item in results)
    if verbose:
        print("\nSummary")
        print(f"- Episodes run: {episodes}")
        print(f"- Task: {task_id}")
        print(f"- Backend: {backend}")
        print(f"- Average clipped score: {average:.3f}")
        print(f"- Average raw score: {raw_average:.3f}")
    return results


def build_gradio_app():
    """Create a lightweight Gradio demo UI."""

    import gradio as gr

    def predict(symptoms: str, age: int, severity: str, task_id: str) -> tuple[str, str]:
        observation = {"symptoms": symptoms, "age": age, "severity": severity}
        action = trim_action_for_task(mock_agent(observation), task_id)
        env = AIMedicalDecisionEnv(task_id=task_id)
        env.current_case = type("InlineCase", (), {
            "case_id": "manual_case",
            "symptoms": symptoms,
            "age": age,
            "severity": severity,
            "disease": action.get("disease_prediction", ""),
            "medicine": action.get("medicine_suggestion", ""),
            "urgency": action.get("urgency_level", ""),
            "advice": action.get("advice_text", ""),
        })()
        env.terminated = False
        _, reward, _, _, info = env.step(action)
        return json.dumps(action, indent=2), json.dumps(
            {"reward": reward, "raw_score": info["raw_score"], "breakdown": info["breakdown"]},
            indent=2,
        )

    with gr.Blocks(title="AI Medical Decision Environment") as demo:
        gr.Markdown("# AI Medical Decision Environment")
        gr.Markdown("Enter symptoms and get a model prediction for the selected task.")
        with gr.Row():
            symptoms = gr.Textbox(label="Symptoms", lines=4, value="High fever, body aches, chills, dry cough.")
            with gr.Column():
                age = gr.Number(label="Age", value=35, precision=0)
                severity = gr.Dropdown(label="Severity", choices=["low", "medium", "high"], value="medium")
                task_id = gr.Dropdown(
                    label="Task",
                    choices=list(TASK_SPECS.keys()),
                    value="task_3_hard",
                )
                submit = gr.Button("Run Decision")
        output = gr.Code(label="Predicted Action", language="json")
        score = gr.Code(label="Evaluation Snapshot", language="json")
        submit.click(fn=predict, inputs=[symptoms, age, severity, task_id], outputs=[output, score])
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the AI Medical Decision Environment.")
    parser.add_argument("--task", default="task_3_hard", choices=list(TASK_SPECS.keys()))
    parser.add_argument("--episodes", type=int, default=60, help="Number of episodes to run (50-100 recommended).")
    parser.add_argument("--backend", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model to use when backend=openai.")
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio demo instead of batch inference.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ui:
        app = build_gradio_app()
        app.launch(server_name=args.host, server_port=args.port)
        return

    run_episodes(task_id=args.task, episodes=args.episodes, backend=args.backend, model=args.model, verbose=True)


if __name__ == "__main__":
    main()
