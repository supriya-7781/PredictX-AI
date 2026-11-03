"""
Tests for src/predictx/xai/feature_explainer.py

All tests use a synthetic scaler so they never depend on model files
being present on disk.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictx.xai.feature_explainer import (
    FEATURE_COLS,
    Z_SIGNIFICANT,
    Z_CRITICAL,
    FeatureContribution,
    ExplanationResult,
    FeatureExplainer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Representative baseline stats (close to the real training dataset values)
_MEANS = np.array([298.3, 309.1, 1538.8, 39.6, 107.9], dtype=float)
_STDS  = np.array([2.0,   1.5,   179.3,  9.8,  63.7], dtype=float)


def _make_scaler() -> StandardScaler:
    """Return a fitted scaler with known mean_ and scale_ arrays."""
    scaler = StandardScaler()
    # Manually inject the fitted attributes so we don't need a training dataset
    scaler.mean_  = _MEANS.copy()
    scaler.scale_ = _STDS.copy()
    scaler.n_features_in_ = len(FEATURE_COLS)
    return scaler


@pytest.fixture
def explainer() -> FeatureExplainer:
    return FeatureExplainer(_make_scaler())


def _normal_row() -> dict:
    """A row exactly at the baseline mean — perfectly normal."""
    return {col: float(_MEANS[i]) for i, col in enumerate(FEATURE_COLS)}


def _high_rpm_row() -> dict:
    """RPM is 3σ above normal; all other sensors at baseline."""
    row = _normal_row()
    row["Rotational speed [rpm]"] = _MEANS[2] + 3.0 * _STDS[2]  # +3σ
    return row


def _multi_anomaly_row() -> dict:
    """High air temp (+2.5σ) and very high RPM (+4σ)."""
    row = _normal_row()
    row["Air temperature [K]"]      = _MEANS[0] + 2.5 * _STDS[0]
    row["Rotational speed [rpm]"]   = _MEANS[2] + 4.0 * _STDS[2]
    return row


# ---------------------------------------------------------------------------
# Scaler validation
# ---------------------------------------------------------------------------

def test_unfitted_scaler_raises():
    raw = StandardScaler()
    with pytest.raises(ValueError, match="not been fitted"):
        FeatureExplainer(raw)


def test_wrong_feature_count_raises():
    scaler = _make_scaler()
    scaler.mean_ = np.array([1.0, 2.0])  # wrong length
    scaler.scale_ = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="FEATURE_COLS"):
        FeatureExplainer(scaler)


# ---------------------------------------------------------------------------
# Normal reading
# ---------------------------------------------------------------------------

def test_normal_row_all_z_scores_zero(explainer):
    result = explainer.explain(_normal_row())
    for c in result.contributions:
        assert abs(c.z_score) < 0.001, f"Expected ~0 z-score for {c.feature}"


def test_normal_row_no_significant_features(explainer):
    result = explainer.explain(_normal_row())
    assert result.n_significant == 0


def test_normal_row_overall_deviation_near_zero(explainer):
    result = explainer.explain(_normal_row())
    assert result.overall_deviation < 0.01


def test_normal_row_narrative_is_non_empty_string(explainer):
    result = explainer.explain(_normal_row())
    assert isinstance(result.narrative, str)
    assert len(result.narrative) > 20


# ---------------------------------------------------------------------------
# Single extreme feature
# ---------------------------------------------------------------------------

def test_high_rpm_is_rank_1(explainer):
    result = explainer.explain(_high_rpm_row())
    top = result.contributions[0]   # sorted by abs_z descending
    assert top.feature == "Rotational speed [rpm]"
    assert top.rank == 1


def test_high_rpm_is_significant(explainer):
    result = explainer.explain(_high_rpm_row())
    rpm_contrib = next(c for c in result.contributions if "rpm" in c.feature)
    assert rpm_contrib.is_significant


def test_high_rpm_direction_above(explainer):
    result = explainer.explain(_high_rpm_row())
    rpm_contrib = next(c for c in result.contributions if "rpm" in c.feature)
    assert rpm_contrib.direction == "above normal"


def test_high_rpm_z_score_matches_expectation(explainer):
    result = explainer.explain(_high_rpm_row())
    rpm_contrib = next(c for c in result.contributions if "rpm" in c.feature)
    assert abs(rpm_contrib.z_score - 3.0) < 0.01


def test_high_rpm_n_significant_is_1(explainer):
    result = explainer.explain(_high_rpm_row())
    assert result.n_significant == 1


# ---------------------------------------------------------------------------
# Multiple anomalous features
# ---------------------------------------------------------------------------

def test_multi_anomaly_rpm_outranks_air_temp(explainer):
    result = explainer.explain(_multi_anomaly_row())
    rpm_c   = next(c for c in result.contributions if "rpm" in c.feature)
    air_c   = next(c for c in result.contributions if "Air temp" in c.short_name)
    assert rpm_c.rank < air_c.rank    # smaller rank = more important


def test_multi_anomaly_n_significant_is_2(explainer):
    result = explainer.explain(_multi_anomaly_row())
    assert result.n_significant == 2


def test_multi_anomaly_primary_driver_is_rpm(explainer):
    result = explainer.explain(_multi_anomaly_row())
    assert "speed" in result.primary_driver.lower()


# ---------------------------------------------------------------------------
# Missing sensor value
# ---------------------------------------------------------------------------

def test_missing_sensor_raises(explainer):
    row = _normal_row()
    del row["Torque [Nm]"]
    with pytest.raises(ValueError, match="Torque"):
        explainer.explain(row)


def test_nan_sensor_raises(explainer):
    row = _normal_row()
    row["Tool wear [min]"] = float("nan")
    with pytest.raises(ValueError, match="Tool wear"):
        explainer.explain(row)


# ---------------------------------------------------------------------------
# Direction labels
# ---------------------------------------------------------------------------

def test_below_normal_direction(explainer):
    row = _normal_row()
    row["Torque [Nm]"] = _MEANS[3] - 3.0 * _STDS[3]   # 3σ below
    result = explainer.explain(row)
    torque_c = next(c for c in result.contributions if "Torque" in c.feature)
    assert torque_c.direction == "below normal"


def test_within_normal_direction(explainer):
    row = _normal_row()
    # Tiny deviation — should be "within normal range"
    row["Air temperature [K]"] = _MEANS[0] + 0.1 * _STDS[0]
    result = explainer.explain(row)
    air_c = next(c for c in result.contributions if "Air temp" in c.short_name)
    assert air_c.direction == "within normal range"


# ---------------------------------------------------------------------------
# Batch explain
# ---------------------------------------------------------------------------

def test_explain_batch_length(explainer):
    df = pd.DataFrame([_normal_row(), _high_rpm_row(), _multi_anomaly_row()])
    results = explainer.explain_batch(df)
    assert len(results) == 3


def test_explain_batch_each_is_explanation_result(explainer):
    df = pd.DataFrame([_normal_row()])
    results = explainer.explain_batch(df)
    assert isinstance(results[0], ExplanationResult)


# ---------------------------------------------------------------------------
# Baseline property
# ---------------------------------------------------------------------------

def test_baseline_keys_match_feature_cols(explainer):
    b = explainer.baseline
    assert set(b.keys()) == set(FEATURE_COLS)


def test_baseline_normal_range_spans_two_sigma(explainer):
    b = explainer.baseline
    for col in FEATURE_COLS:
        entry = b[col]
        expected_width = 4 * entry["std"]
        actual_width   = entry["normal_high"] - entry["normal_low"]
        assert abs(actual_width - expected_width) < 0.01


# ---------------------------------------------------------------------------
# to_dict serialisation
# ---------------------------------------------------------------------------

def test_to_dict_contains_required_keys(explainer):
    result = explainer.explain(_high_rpm_row(), anomaly_score=-0.15)
    d = result.to_dict()
    for key in ["primary_driver", "n_significant", "overall_deviation",
                "anomaly_score", "narrative", "contributions"]:
        assert key in d


def test_to_dict_contributions_count(explainer):
    result = explainer.explain(_normal_row())
    d = result.to_dict()
    assert len(d["contributions"]) == len(FEATURE_COLS)
