"""
Tests for src/predictx/health/health_scorer.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictx.xai.feature_explainer import FEATURE_COLS
from predictx.health.health_scorer import (
    DEFAULT_WEIGHTS,
    Z_CRITICAL,
    HealthScorer,
    HealthResult,
    classify_band,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MEANS = np.array([298.3, 309.1, 1538.8, 39.6, 107.9], dtype=float)
_STDS  = np.array([2.0,   1.5,   179.3,  9.8,  63.7], dtype=float)


def _make_scaler() -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_  = _MEANS.copy()
    scaler.scale_ = _STDS.copy()
    scaler.n_features_in_ = len(FEATURE_COLS)
    return scaler


@pytest.fixture
def scorer() -> HealthScorer:
    return HealthScorer(_make_scaler())


def _normal_row() -> dict:
    return {col: float(_MEANS[i]) for i, col in enumerate(FEATURE_COLS)}


def _extreme_row() -> dict:
    """All sensors at 3σ above baseline — worst possible."""
    return {col: float(_MEANS[i] + 3.0 * _STDS[i]) for i, col in enumerate(FEATURE_COLS)}


def _borderline_row() -> dict:
    """Two sensors at 2σ — should land in Warning band."""
    row = _normal_row()
    row["Rotational speed [rpm]"] = _MEANS[2] + 2.0 * _STDS[2]
    row["Torque [Nm]"]            = _MEANS[3] + 2.0 * _STDS[3]
    return row


# ---------------------------------------------------------------------------
# Score validation
# ---------------------------------------------------------------------------

def test_normal_row_score_near_100(scorer):
    result = scorer.score_row(_normal_row())
    assert result.score >= 99


def test_extreme_row_score_is_critical(scorer):
    result = scorer.score_row(_extreme_row())
    assert result.score <= 39


def test_borderline_row_in_warning_band(scorer):
    result = scorer.score_row(_borderline_row())
    # Two features at exactly 2σ: penalty per feature = (2/3)² = 0.444
    # weighted: RPM 0.25 * 0.444 + Torque 0.25 * 0.444 = 0.222
    # score = round(100 * (1 - 0.222)) = 78 → Warning
    assert result.band == "Warning"


def test_score_always_in_0_to_100(scorer):
    for row in [_normal_row(), _extreme_row(), _borderline_row()]:
        result = scorer.score_row(row)
        assert 0 <= result.score <= 100


def test_score_is_int(scorer):
    result = scorer.score_row(_normal_row())
    assert isinstance(result.score, int)


def test_same_input_same_output(scorer):
    row = _borderline_row()
    r1 = scorer.score_row(row)
    r2 = scorer.score_row(row)
    assert r1.score == r2.score


# ---------------------------------------------------------------------------
# Health bands
# ---------------------------------------------------------------------------

def test_healthy_band_at_95(scorer):
    assert classify_band(95) == ("Healthy", "normal")


def test_warning_band_at_80(scorer):
    assert classify_band(80) == ("Warning", "warning")


def test_poor_band_at_55(scorer):
    assert classify_band(55) == ("Poor", "anomaly")


def test_critical_band_at_20(scorer):
    assert classify_band(20) == ("Critical", "anomaly")


def test_boundary_90_is_healthy(scorer):
    assert classify_band(90)[0] == "Healthy"


def test_boundary_70_is_warning(scorer):
    assert classify_band(70)[0] == "Warning"


def test_boundary_40_is_poor(scorer):
    assert classify_band(40)[0] == "Poor"


def test_boundary_39_is_critical(scorer):
    assert classify_band(39)[0] == "Critical"


# ---------------------------------------------------------------------------
# HealthResult attributes
# ---------------------------------------------------------------------------

def test_result_has_feature_penalties(scorer):
    result = scorer.score_row(_borderline_row())
    assert set(result.feature_penalties.keys()) == set(FEATURE_COLS)


def test_result_has_feature_z_scores(scorer):
    result = scorer.score_row(_borderline_row())
    assert set(result.feature_z_scores.keys()) == set(FEATURE_COLS)


def test_normal_row_is_healthy_property(scorer):
    result = scorer.score_row(_normal_row())
    assert result.is_healthy


def test_extreme_row_is_critical_property(scorer):
    result = scorer.score_row(_extreme_row())
    assert result.is_critical


def test_extreme_row_needs_attention(scorer):
    result = scorer.score_row(_extreme_row())
    assert result.needs_attention


# ---------------------------------------------------------------------------
# Missing / invalid values
# ---------------------------------------------------------------------------

def test_missing_sensor_raises(scorer):
    row = _normal_row()
    del row["Torque [Nm]"]
    with pytest.raises(ValueError, match="Torque"):
        scorer.score_row(row)


def test_nan_sensor_raises(scorer):
    row = _normal_row()
    row["Tool wear [min]"] = float("nan")
    with pytest.raises(ValueError, match="Tool wear"):
        scorer.score_row(row)


# ---------------------------------------------------------------------------
# score_dataframe
# ---------------------------------------------------------------------------

def test_score_dataframe_adds_columns(scorer):
    import pandas as pd
    df = pd.DataFrame([_normal_row(), _borderline_row(), _extreme_row()])
    out = scorer.score_dataframe(df)
    assert "health_score" in out.columns
    assert "health_band"  in out.columns


def test_score_dataframe_preserves_rows(scorer):
    import pandas as pd
    df = pd.DataFrame([_normal_row(), _borderline_row()])
    out = scorer.score_dataframe(df)
    assert len(out) == 2


def test_score_dataframe_does_not_modify_input(scorer):
    import pandas as pd
    df = pd.DataFrame([_normal_row()])
    _ = scorer.score_dataframe(df)
    assert "health_score" not in df.columns


# ---------------------------------------------------------------------------
# Weights validation
# ---------------------------------------------------------------------------

def test_default_weights_sum_to_1():
    total = sum(DEFAULT_WEIGHTS.values())
    assert abs(total - 1.0) < 0.001


def test_invalid_weights_raise():
    bad_weights = {col: 0.3 for col in FEATURE_COLS}   # sums to 1.5
    with pytest.raises(ValueError, match="sum to 1.0"):
        HealthScorer(_make_scaler(), weights=bad_weights)


def test_missing_weight_raises():
    partial_weights = {col: 0.25 for col in FEATURE_COLS[:4]}   # missing one
    with pytest.raises(ValueError):
        HealthScorer(_make_scaler(), weights=partial_weights)


# ---------------------------------------------------------------------------
# Baseline property
# ---------------------------------------------------------------------------

def test_baseline_keys_match_feature_cols(scorer):
    b = scorer.baseline
    assert set(b.keys()) == set(FEATURE_COLS)
