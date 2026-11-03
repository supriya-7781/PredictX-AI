"""
Tests for src/predictx/model/predictor.py
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictx.model.predictor import Predictor, FEATURE_COLS

MODEL_PATH = ROOT / "models" / "isolation_forest.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"


@pytest.fixture(scope="module")
def predictor():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        pytest.skip("Model not trained yet — run training/train_model.py first.")
    return Predictor(MODEL_PATH, SCALER_PATH)


def _normal_row() -> dict:
    return {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0,
    }


def _anomaly_row() -> dict:
    """Extreme values known to trigger anomaly flag."""
    return {
        "Air temperature [K]": 304.5,
        "Process temperature [K]": 313.8,
        "Rotational speed [rpm]": 2940.0,
        "Torque [Nm]": 5.2,
        "Tool wear [min]": 290.0,
    }


# ---------------------------------------------------------------------------
# Basic shape / dtype checks
# ---------------------------------------------------------------------------

def test_predict_returns_correct_columns(predictor):
    df = pd.DataFrame([_normal_row()])
    result = predictor.predict(df)
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns


def test_predict_preserves_row_count(predictor):
    rows = [_normal_row() for _ in range(10)]
    df = pd.DataFrame(rows)
    result = predictor.predict(df)
    assert len(result) == 10


def test_is_anomaly_is_binary(predictor):
    df = pd.DataFrame([_normal_row(), _anomaly_row()])
    result = predictor.predict(df)
    unique_vals = set(result["is_anomaly"].unique())
    assert unique_vals <= {0, 1}


def test_anomaly_score_is_float(predictor):
    df = pd.DataFrame([_normal_row()])
    result = predictor.predict(df)
    assert result["anomaly_score"].dtype == float


# ---------------------------------------------------------------------------
# Functional checks
# ---------------------------------------------------------------------------

def test_normal_reading_is_not_anomaly(predictor):
    df = pd.DataFrame([_normal_row()])
    result = predictor.predict(df)
    assert result["is_anomaly"].iloc[0] == 0


def test_extreme_reading_is_anomaly(predictor):
    df = pd.DataFrame([_anomaly_row()])
    result = predictor.predict(df)
    assert result["is_anomaly"].iloc[0] == 1


def test_anomaly_score_lower_for_anomaly(predictor):
    df = pd.DataFrame([_normal_row(), _anomaly_row()])
    result = predictor.predict(df)
    normal_score = result.iloc[0]["anomaly_score"]
    anomaly_score = result.iloc[1]["anomaly_score"]
    assert anomaly_score < normal_score


def test_predict_row_dict(predictor):
    result = predictor.predict_row(_normal_row())
    assert "anomaly_score" in result
    assert "is_anomaly" in result
    assert isinstance(result["anomaly_score"], float)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_column_raises(predictor):
    df = pd.DataFrame([{"Air temperature [K]": 298.1}])
    with pytest.raises(ValueError, match="missing required columns"):
        predictor.predict(df)


def test_model_not_found_raises():
    p = Predictor(
        model_path="/nonexistent/model.pkl",
        scaler_path="/nonexistent/scaler.pkl",
    )
    with pytest.raises(FileNotFoundError):
        _ = p.model


def test_is_ready_true(predictor):
    assert predictor.is_ready is True
