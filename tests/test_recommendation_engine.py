"""
Tests for src/predictx/recommendations/recommendation_engine.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictx.xai.feature_explainer import FEATURE_COLS, FeatureExplainer
from predictx.recommendations.recommendation_engine import (
    Recommendation,
    RecommendationSet,
    RecommendationEngine,
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
def explainer() -> FeatureExplainer:
    return FeatureExplainer(_make_scaler())


@pytest.fixture
def engine() -> RecommendationEngine:
    return RecommendationEngine()


def _explain(explainer, row, score=-0.15):
    return explainer.explain(row, anomaly_score=score)


def _normal_row():
    return {col: float(_MEANS[i]) for i, col in enumerate(FEATURE_COLS)}


def _high_rpm_row():
    row = _normal_row()
    row["Rotational speed [rpm]"] = _MEANS[2] + 3.5 * _STDS[2]
    return row


def _low_torque_row():
    row = _normal_row()
    row["Torque [Nm]"] = _MEANS[3] - 3.0 * _STDS[3]
    return row


def _high_wear_row():
    row = _normal_row()
    row["Tool wear [min]"] = _MEANS[4] + 3.5 * _STDS[4]   # critical wear
    return row


def _high_proc_temp_row():
    row = _normal_row()
    row["Process temperature [K]"] = _MEANS[1] + 2.5 * _STDS[1]
    return row


def _multi_anomaly_row():
    row = _normal_row()
    row["Rotational speed [rpm]"]   = _MEANS[2] + 2.5 * _STDS[2]
    row["Air temperature [K]"]      = _MEANS[0] + 2.2 * _STDS[0]
    row["Process temperature [K]"]  = _MEANS[1] + 2.1 * _STDS[1]
    return row


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_returns_recommendation_set(explainer, engine):
    result = engine.recommend(_explain(explainer, _high_rpm_row()), "M001")
    assert isinstance(result, RecommendationSet)


def test_recommendation_set_has_list(explainer, engine):
    result = engine.recommend(_explain(explainer, _high_rpm_row()))
    assert isinstance(result.recommendations, list)
    assert len(result.recommendations) >= 1


def test_each_recommendation_has_required_fields(explainer, engine):
    result = engine.recommend(_explain(explainer, _high_rpm_row()))
    for r in result.recommendations:
        assert r.action         != ""
        assert r.evidence       != ""
        assert r.confidence     in ("Suspected", "Possible", "Inconclusive")
        assert r.sensor_trigger != ""
        assert isinstance(r.priority, int)


# ---------------------------------------------------------------------------
# Rule triggers
# ---------------------------------------------------------------------------

def test_high_rpm_triggers_bearing_recommendation(explainer, engine):
    result = engine.recommend(_explain(explainer, _high_rpm_row()))
    actions = " ".join(r.action.lower() for r in result.recommendations)
    assert "bearing" in actions


def test_low_torque_triggers_tooling_recommendation(explainer, engine):
    result = engine.recommend(_explain(explainer, _low_torque_row()))
    actions = " ".join(r.action.lower() for r in result.recommendations)
    assert "tool" in actions or "spindle" in actions


def test_high_wear_triggers_replacement_recommendation(explainer, engine):
    result = engine.recommend(_explain(explainer, _high_wear_row()))
    actions = " ".join(r.action.lower() for r in result.recommendations)
    assert "replacement" in actions or "replace" in actions


def test_high_proc_temp_triggers_coolant_recommendation(explainer, engine):
    result = engine.recommend(_explain(explainer, _high_proc_temp_row()))
    actions = " ".join(r.action.lower() for r in result.recommendations)
    assert "coolant" in actions or "heat" in actions or "cool" in actions


def test_multi_anomaly_multiple_recommendations(explainer, engine):
    result = engine.recommend(_explain(explainer, _multi_anomaly_row()))
    assert len(result.recommendations) >= 2


# ---------------------------------------------------------------------------
# Inconclusive result
# ---------------------------------------------------------------------------

def test_normal_row_returns_inconclusive(explainer, engine):
    result = engine.recommend(_explain(explainer, _normal_row(), score=-0.01))
    # Normal row has no significant deviations — should be inconclusive
    assert all(r.confidence == "Inconclusive" for r in result.recommendations)


def test_inconclusive_has_actionable_false(explainer, engine):
    result = engine.recommend(_explain(explainer, _normal_row(), score=-0.01))
    assert result.has_actionable is False


def test_anomaly_has_actionable_true(explainer, engine):
    result = engine.recommend(_explain(explainer, _high_rpm_row()))
    assert result.has_actionable is True


# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------

def test_very_high_rpm_is_suspected(explainer, engine):
    """3.5σ deviation should be 'Suspected' (≥3σ threshold)."""
    result = engine.recommend(_explain(explainer, _high_rpm_row()))
    bearing_rec = next(
        (r for r in result.recommendations if "bearing" in r.action.lower()),
        None,
    )
    assert bearing_rec is not None
    assert bearing_rec.confidence == "Suspected"


def test_moderate_wear_is_possible(explainer, engine):
    """2σ wear should be 'Possible' (<3σ threshold)."""
    row = _normal_row()
    row["Tool wear [min]"] = _MEANS[4] + 2.1 * _STDS[4]  # 2.1σ — below 3
    result = engine.recommend(_explain(explainer, row))
    wear_recs = [r for r in result.recommendations if "tool" in r.action.lower() or "replace" in r.action.lower()]
    if wear_recs:
        assert any(r.confidence == "Possible" for r in wear_recs)


# ---------------------------------------------------------------------------
# Confidence badge kind
# ---------------------------------------------------------------------------

def test_suspected_badge_kind(explainer, engine):
    result = engine.recommend(_explain(explainer, _high_rpm_row()))
    suspected = [r for r in result.recommendations if r.confidence == "Suspected"]
    if suspected:
        assert suspected[0].confidence_kind == "warning"


def test_inconclusive_badge_kind(explainer, engine):
    result = engine.recommend(_explain(explainer, _normal_row(), score=-0.01))
    inc = result.recommendations[0]
    assert inc.confidence_kind == "muted"


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

def test_by_priority_is_sorted(explainer, engine):
    result = engine.recommend(_explain(explainer, _multi_anomaly_row()))
    ordered = result.by_priority()
    priorities = [r.priority for r in ordered]
    assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# Machine ID
# ---------------------------------------------------------------------------

def test_machine_id_in_summary(explainer, engine):
    result = engine.recommend(_explain(explainer, _high_rpm_row()), machine_id="M-007")
    assert "M-007" in result.summary


def test_disclaimer_is_non_empty(explainer, engine):
    result = engine.recommend(_explain(explainer, _high_rpm_row()))
    assert len(result.disclaimer) > 20


# ---------------------------------------------------------------------------
# No duplicate actions
# ---------------------------------------------------------------------------

def test_no_duplicate_actions(explainer, engine):
    result = engine.recommend(_explain(explainer, _multi_anomaly_row()))
    actions = [r.action for r in result.recommendations]
    assert len(actions) == len(set(actions))
