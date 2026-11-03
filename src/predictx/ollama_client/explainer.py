"""
explainer.py
============
Calls a locally-running Ollama LLM to explain anomalies detected by the
Isolation Forest model and suggest maintenance actions.

If Ollama is not running, returns a canned fallback explanation so the
rest of the dashboard still works.

Requirements:
    - Ollama installed and running: https://ollama.com/download
    - Model pulled: `ollama pull llama3.2:3b`
    - Environment vars: OLLAMA_URL, OLLAMA_MODEL
"""

from __future__ import annotations

import os
import textwrap

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
REQUEST_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(row: dict, score: float) -> str:
    """
    Build a concise prompt for the LLM given sensor readings and anomaly score.
    """
    severity = "HIGH" if score < -0.10 else "MEDIUM" if score < -0.05 else "LOW"
    return textwrap.dedent(f"""\
        You are an industrial predictive maintenance engineer AI assistant.
        A machine sensor reading has been flagged as an anomaly by an Isolation Forest model.

        Sensor readings:
          - Air temperature:     {row.get('Air temperature [K]', 'N/A')} K
          - Process temperature: {row.get('Process temperature [K]', 'N/A')} K
          - Rotational speed:    {row.get('Rotational speed [rpm]', 'N/A')} rpm
          - Torque:              {row.get('Torque [Nm]', 'N/A')} Nm
          - Tool wear:           {row.get('Tool wear [min]', 'N/A')} min

        Anomaly score: {score:.4f}  (lower = more anomalous)
        Severity:      {severity}

        In 3–5 sentences:
        1. Identify which sensor reading(s) look abnormal and why.
        2. Suggest the most likely failure mode.
        3. Recommend an immediate maintenance action.

        Be concise and practical.
    """)


# ---------------------------------------------------------------------------
# Ollama HTTP call
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 256,
        },
    }
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# Fallback explanation (used when Ollama is unavailable)
# ---------------------------------------------------------------------------

def _fallback_explanation(row: dict, score: float) -> str:
    """Rule-based canned explanation when Ollama is offline."""
    lines = [
        "⚠️  Ollama is not running — showing rule-based explanation.",
        "",
        f"Anomaly score: **{score:.4f}** (lower = more severe)",
        "",
        "Possible causes based on sensor values:",
    ]

    air_temp = float(row.get("Air temperature [K]", 300))
    proc_temp = float(row.get("Process temperature [K]", 310))
    rpm = float(row.get("Rotational speed [rpm]", 1500))
    torque = float(row.get("Torque [Nm]", 40))
    wear = float(row.get("Tool wear [min]", 100))

    if rpm > 2500:
        lines.append("- **High rotational speed** — risk of bearing overload or imbalance.")
    if torque < 10:
        lines.append("- **Very low torque** — possible tool breakage or spindle fault.")
    if wear > 240:
        lines.append("- **Excessive tool wear** — immediate tool replacement recommended.")
    if air_temp > 303:
        lines.append("- **Elevated air temperature** — check cooling system.")
    if proc_temp - air_temp > 12:
        lines.append("- **Large temperature gradient** — inspect heat dissipation.")

    if len(lines) == 5:
        lines.append("- No single dominant cause — review full sensor history.")

    lines += ["", "**Action:** Schedule inspection within 24 hours."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_anomaly(row: dict, score: float) -> str:
    """
    Return a human-readable explanation for a flagged sensor reading.

    Parameters
    ----------
    row   : dict of sensor column → value
    score : anomaly_score from the Isolation Forest (float, lower = worse)

    Returns
    -------
    str — Markdown-formatted explanation
    """
    try:
        prompt = _build_prompt(row, score)
        response = _call_ollama(prompt)
        return response
    except requests.exceptions.ConnectionError:
        return _fallback_explanation(row, score)
    except requests.exceptions.Timeout:
        return (
            "⏱️  Ollama request timed out. "
            "The model may be loading — try again in a moment.\n\n"
            + _fallback_explanation(row, score)
        )
    except Exception as exc:
        return (
            f"❌  Ollama error: {exc}\n\n"
            + _fallback_explanation(row, score)
        )


def is_ollama_available() -> bool:
    """Return True if Ollama is reachable at OLLAMA_URL."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """Return list of model names available in the local Ollama instance."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []
