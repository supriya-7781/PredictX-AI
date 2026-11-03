"""
Tests for src/predictx/ollama_client/explainer.py

Uses unittest.mock to simulate Ollama HTTP responses
so tests pass without a running Ollama instance.
"""
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictx.ollama_client.explainer import (
    explain_anomaly,
    _build_prompt,
    _fallback_explanation,
    is_ollama_available,
)

SAMPLE_ROW = {
    "Air temperature [K]": 304.5,
    "Process temperature [K]": 313.8,
    "Rotational speed [rpm]": 2940.0,
    "Torque [Nm]": 5.2,
    "Tool wear [min]": 290.0,
}
SAMPLE_SCORE = -0.168


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def test_prompt_contains_sensor_values():
    prompt = _build_prompt(SAMPLE_ROW, SAMPLE_SCORE)
    assert "304.5" in prompt
    assert "2940.0" in prompt
    assert "290.0" in prompt


def test_prompt_contains_anomaly_score():
    prompt = _build_prompt(SAMPLE_ROW, SAMPLE_SCORE)
    assert str(round(SAMPLE_SCORE, 4)) in prompt


def test_prompt_marks_high_severity():
    prompt = _build_prompt(SAMPLE_ROW, -0.15)
    assert "HIGH" in prompt


def test_prompt_marks_medium_severity():
    prompt = _build_prompt(SAMPLE_ROW, -0.07)
    assert "MEDIUM" in prompt


def test_prompt_marks_low_severity():
    prompt = _build_prompt(SAMPLE_ROW, -0.02)
    assert "LOW" in prompt


# ---------------------------------------------------------------------------
# Fallback explanation
# ---------------------------------------------------------------------------

def test_fallback_contains_warning():
    result = _fallback_explanation(SAMPLE_ROW, SAMPLE_SCORE)
    assert "Ollama" in result


def test_fallback_mentions_high_rpm():
    row = {**SAMPLE_ROW, "Rotational speed [rpm]": 2600}
    result = _fallback_explanation(row, -0.12)
    assert "rotational speed" in result.lower() or "rpm" in result.lower() or "Rotational" in result


def test_fallback_mentions_tool_wear():
    row = {**SAMPLE_ROW, "Tool wear [min]": 260}
    result = _fallback_explanation(row, -0.12)
    assert "wear" in result.lower()


def test_fallback_is_str():
    result = _fallback_explanation(SAMPLE_ROW, SAMPLE_SCORE)
    assert isinstance(result, str)
    assert len(result) > 20


# ---------------------------------------------------------------------------
# explain_anomaly — mock HTTP
# ---------------------------------------------------------------------------

def test_explain_anomaly_uses_ollama_when_available():
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "The tool wear is critically high."}
    mock_response.raise_for_status = MagicMock()

    with patch("predictx.ollama_client.explainer.requests.post", return_value=mock_response):
        result = explain_anomaly(SAMPLE_ROW, SAMPLE_SCORE)

    assert "tool wear" in result.lower()


def test_explain_anomaly_falls_back_on_connection_error():
    import requests as real_requests

    with patch(
        "predictx.ollama_client.explainer.requests.post",
        side_effect=real_requests.exceptions.ConnectionError("offline"),
    ):
        result = explain_anomaly(SAMPLE_ROW, SAMPLE_SCORE)

    # Should get the fallback, not an exception
    assert isinstance(result, str)
    assert "Ollama" in result


def test_explain_anomaly_falls_back_on_timeout():
    import requests as real_requests

    with patch(
        "predictx.ollama_client.explainer.requests.post",
        side_effect=real_requests.exceptions.Timeout(),
    ):
        result = explain_anomaly(SAMPLE_ROW, SAMPLE_SCORE)

    assert "timed out" in result.lower()


# ---------------------------------------------------------------------------
# is_ollama_available — mock HTTP
# ---------------------------------------------------------------------------

def test_is_ollama_available_true():
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch("predictx.ollama_client.explainer.requests.get", return_value=mock_response):
        assert is_ollama_available() is True


def test_is_ollama_available_false_on_connection_error():
    import requests as real_requests
    with patch(
        "predictx.ollama_client.explainer.requests.get",
        side_effect=real_requests.exceptions.ConnectionError(),
    ):
        assert is_ollama_available() is False
