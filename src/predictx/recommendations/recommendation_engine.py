"""
recommendation_engine.py
========================
Rule-based maintenance recommendation engine for PredictX.

When the Isolation Forest detects an anomaly, this module converts the
structured XAI output (``ExplanationResult``) into specific, evidence-backed
inspection recommendations.

Design principles
-----------------
1.  Every recommendation is triggered by a specific, named sensor reading
    that exceeds a defined threshold — no hidden logic.
2.  Confidence levels distinguish suspected causes from possibilities:
      "Suspected"     — sensor deviation ≥ 3σ, strong statistical evidence.
      "Possible"      — sensor deviation 2–3σ, moderate evidence.
      "Inconclusive"  — no individual sensor is statistically significant;
                       collective mild deviation only.
3.  No claim is made that a specific component HAS failed — only that
    inspection is warranted based on the observed readings.
4.  When available sensor evidence is insufficient, an "Inconclusive" result
    is returned rather than inventing a recommendation.

Rule table
----------
Each rule is a dict with keys:
  sensor         : which FEATURE_COLS key triggers this rule
  direction      : "above" | "below" | "any"
  z_threshold    : minimum |z-score| to fire this rule
  action         : the maintenance inspection action to recommend
  evidence_tmpl  : format string describing the evidence
  confidence_fn  : callable(abs_z) -> "Suspected" | "Possible"
  priority       : int, 1 = highest (used for ordering)

Usage
-----
    from predictx.xai.feature_explainer import FeatureExplainer, get_explainer
    from predictx.recommendations.recommendation_engine import RecommendationEngine

    explainer = get_explainer()
    result    = explainer.explain(row_dict, anomaly_score)

    engine    = RecommendationEngine()
    recs      = engine.recommend(result)
    for r in recs:
        print(r.action, r.confidence, r.evidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from predictx.xai.feature_explainer import ExplanationResult, FeatureContribution, Z_SIGNIFICANT

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """
    A single maintenance inspection recommendation.

    All fields are populated from measured sensor evidence — no guesses.
    """
    action:         str    # What to inspect / check
    evidence:       str    # Which sensor reading triggered this, and how
    confidence:     str    # "Suspected" | "Possible" | "Inconclusive"
    sensor_trigger: str    # Short name of the triggering sensor
    priority:       int    # 1 = highest; used for ordering in the UI
    z_score:        float  # The Z-score that fired this rule

    @property
    def confidence_kind(self) -> str:
        """Map confidence level to dashboard badge kind."""
        return {
            "Suspected":    "warning",
            "Possible":     "info",
            "Inconclusive": "muted",
        }.get(self.confidence, "muted")


@dataclass
class RecommendationSet:
    """
    The complete set of recommendations for one anomaly event.

    Attributes
    ----------
    recommendations :
        Ordered list (highest priority first).
    has_actionable :
        True if at least one recommendation is not "Inconclusive".
    machine_id :
        Passed-through for display — not used in logic.
    summary :
        One-sentence summary for dashboard cards.
    disclaimer :
        Mandatory legal/safety disclaimer.
    """
    recommendations: list[Recommendation]
    has_actionable:  bool
    machine_id:      str
    summary:         str
    disclaimer:      str = (
        "These recommendations are based on statistical sensor deviations "
        "and are not confirmed fault diagnoses.  Always verify with a "
        "qualified maintenance engineer before taking action."
    )

    def by_priority(self) -> list[Recommendation]:
        """Return recommendations sorted by priority (1 = first)."""
        return sorted(self.recommendations, key=lambda r: r.priority)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------
# Each rule is evaluated against a FeatureContribution object.
# Firing condition: direction matches AND abs_z >= z_threshold.

def _confidence(abs_z: float, suspected_at: float = 3.0) -> str:
    return "Suspected" if abs_z >= suspected_at else "Possible"


# (sensor_short_name, direction, z_threshold, action, evidence_template, priority)
_RULES: list[dict] = [

    # ── Rotational speed ────────────────────────────────────────────────────
    {
        "sensor":     "Rotational speed",
        "direction":  "above",
        "z_threshold": Z_SIGNIFICANT,
        "action":     "Inspect bearing condition and check for mechanical imbalance or misalignment.",
        "evidence_tmpl": (
            "Rotational speed is {val:.0f} rpm ({z:+.2f}σ above baseline). "
            "Elevated speed increases bearing stress and vibration."
        ),
        "priority": 1,
    },
    {
        "sensor":     "Rotational speed",
        "direction":  "below",
        "z_threshold": Z_SIGNIFICANT,
        "action":     "Inspect drive system — check belt, coupling, or transmission for slippage or wear.",
        "evidence_tmpl": (
            "Rotational speed is {val:.0f} rpm ({z:+.2f}σ below baseline). "
            "Reduced speed may indicate drive-train resistance or slippage."
        ),
        "priority": 2,
    },

    # ── Torque ──────────────────────────────────────────────────────────────
    {
        "sensor":     "Torque",
        "direction":  "above",
        "z_threshold": Z_SIGNIFICANT,
        "action":     "Check motor load and shaft coupling for overload conditions.",
        "evidence_tmpl": (
            "Torque is {val:.1f} Nm ({z:+.2f}σ above baseline). "
            "Elevated torque may indicate mechanical resistance or overloading."
        ),
        "priority": 2,
    },
    {
        "sensor":     "Torque",
        "direction":  "below",
        "z_threshold": Z_SIGNIFICANT,
        "action":     "Inspect tooling and spindle — possible tool breakage or loss of cutting contact.",
        "evidence_tmpl": (
            "Torque is {val:.1f} Nm ({z:+.2f}σ below baseline). "
            "Abnormally low torque may indicate tool breakage or a spindle fault."
        ),
        "priority": 1,
    },

    # ── Tool wear ────────────────────────────────────────────────────────────
    {
        "sensor":     "Tool wear",
        "direction":  "above",
        "z_threshold": Z_SIGNIFICANT,
        "action":     "Schedule tool inspection and replacement evaluation.",
        "evidence_tmpl": (
            "Tool wear is {val:.0f} min ({z:+.2f}σ above baseline). "
            "Excessive wear reduces cutting precision and may precede tool failure."
        ),
        "priority": 2,
    },
    {
        "sensor":     "Tool wear",
        "direction":  "above",
        "z_threshold": 3.0,   # override — emergency at 3σ
        "action":     "Stop machine and perform emergency tool replacement before resuming operation.",
        "evidence_tmpl": (
            "Tool wear is {val:.0f} min ({z:+.2f}σ above baseline — critical level). "
            "Immediate replacement required to prevent damage."
        ),
        "priority": 0,   # highest possible priority
    },

    # ── Air temperature ──────────────────────────────────────────────────────
    {
        "sensor":     "Air temperature",
        "direction":  "above",
        "z_threshold": Z_SIGNIFICANT,
        "action":     "Check cooling system, air circulation, and ambient temperature around the machine.",
        "evidence_tmpl": (
            "Air temperature is {val:.1f} K ({z:+.2f}σ above baseline). "
            "Elevated ambient temperature reduces machine cooling efficiency."
        ),
        "priority": 3,
    },

    # ── Process temperature ──────────────────────────────────────────────────
    {
        "sensor":     "Process temperature",
        "direction":  "above",
        "z_threshold": Z_SIGNIFICANT,
        "action":     "Inspect coolant flow, cutting fluid levels, and heat dissipation pathways.",
        "evidence_tmpl": (
            "Process temperature is {val:.1f} K ({z:+.2f}σ above baseline). "
            "High process temperature is a direct precursor to thermal failure modes."
        ),
        "priority": 1,
    },

]


# ---------------------------------------------------------------------------
# RecommendationEngine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """
    Converts an ExplanationResult into a RecommendationSet.

    The engine evaluates every rule against the per-feature contributions
    in the ExplanationResult.  Rules fire independently — multiple rules
    can fire for the same reading.

    Parameters
    ----------
    rules : list[dict], optional
        Override the default rule table.  Useful for testing.
    """

    def __init__(self, rules: list[dict] | None = None):
        self._rules = rules if rules is not None else _RULES

    def recommend(
        self,
        explanation: ExplanationResult,
        machine_id: str = "Unknown",
    ) -> RecommendationSet:
        """
        Generate maintenance recommendations from an ExplanationResult.

        Parameters
        ----------
        explanation : ExplanationResult
            Output from FeatureExplainer.explain().
        machine_id : str
            Machine identifier for display (not used in logic).

        Returns
        -------
        RecommendationSet
        """
        # Build a lookup by short name for fast rule matching
        contrib_by_short = {c.short_name: c for c in explanation.contributions}

        fired: list[Recommendation] = []
        seen_actions: set[str] = set()   # deduplicate exact-same action strings

        for rule in self._rules:
            sensor_name = rule["sensor"]
            contrib = contrib_by_short.get(sensor_name)
            if contrib is None:
                continue  # sensor not in this reading (shouldn't happen)

            z     = contrib.z_score
            abs_z = contrib.abs_z
            val   = contrib.current_value

            # Check direction
            direction = rule["direction"]
            if direction == "above" and z <= 0:
                continue
            if direction == "below" and z >= 0:
                continue
            # direction == "any" always passes

            # Check threshold
            if abs_z < rule["z_threshold"]:
                continue

            # Rule fired — build recommendation
            evidence = rule["evidence_tmpl"].format(val=val, z=z)
            action   = rule["action"]

            # Skip if we've already added the exact same action
            if action in seen_actions:
                continue
            seen_actions.add(action)

            confidence = _confidence(abs_z, suspected_at=3.0)

            fired.append(Recommendation(
                action         = action,
                evidence       = evidence,
                confidence     = confidence,
                sensor_trigger = sensor_name,
                priority       = rule["priority"],
                z_score        = round(z, 4),
            ))

        # If no rules fired, return an inconclusive result
        if not fired:
            fired = [_inconclusive_recommendation(explanation)]

        # Sort by priority (lower number = higher priority)
        fired.sort(key=lambda r: (r.priority, -abs(r.z_score)))

        has_actionable = any(r.confidence != "Inconclusive" for r in fired)
        summary        = _build_summary(fired, machine_id)

        return RecommendationSet(
            recommendations = fired,
            has_actionable  = has_actionable,
            machine_id      = machine_id,
            summary         = summary,
        )

    def recommend_from_row(
        self,
        row: dict,
        anomaly_score: float,
        machine_id: str = "Unknown",
    ) -> RecommendationSet:
        """
        Convenience method — builds an ExplanationResult internally and then
        generates recommendations.

        Parameters
        ----------
        row : dict
            Sensor reading dict with all FEATURE_COLS.
        anomaly_score : float
            Pre-computed anomaly score from the Isolation Forest.
        machine_id : str
            Machine identifier for display.
        """
        from predictx.xai.feature_explainer import get_explainer
        explainer = get_explainer()
        explanation = explainer.explain(row, anomaly_score)
        return self.recommend(explanation, machine_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _inconclusive_recommendation(explanation: ExplanationResult) -> Recommendation:
    """Return a generic recommendation when no rule fires."""
    if explanation.n_significant == 0 and explanation.overall_deviation < 1.0:
        evidence = (
            f"No individual sensor reading exceeds the significance threshold. "
            f"Overall deviation is {explanation.overall_deviation:.2f}σ (mean absolute). "
            f"The anomaly may reflect a subtle multi-sensor pattern."
        )
        action = (
            "Monitor the next 5–10 readings before scheduling maintenance. "
            "No urgent action is indicated at this time."
        )
    else:
        evidence = (
            f"Sensor readings show collective mild deviation "
            f"(overall {explanation.overall_deviation:.2f}σ) "
            f"without a single dominant cause."
        )
        action = (
            "Schedule a routine inspection and review recent operating conditions. "
            "No specific component fault is indicated."
        )

    return Recommendation(
        action         = action,
        evidence       = evidence,
        confidence     = "Inconclusive",
        sensor_trigger = "multiple",
        priority       = 5,
        z_score        = 0.0,
    )


def _build_summary(recs: list[Recommendation], machine_id: str) -> str:
    """Build a one-sentence summary for the dashboard card header."""
    actionable = [r for r in recs if r.confidence != "Inconclusive"]
    if not actionable:
        return f"Machine {machine_id}: no specific fault indicated — monitor readings."

    top = actionable[0]
    if top.confidence == "Suspected":
        return (
            f"Machine {machine_id}: {top.sensor_trigger} deviation suggests "
            f"inspection is required. {len(actionable)} action(s) recommended."
        )
    return (
        f"Machine {machine_id}: {len(actionable)} possible issue(s) identified. "
        f"Routine inspection recommended."
    )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_engine: RecommendationEngine | None = None


def get_recommendation_engine() -> RecommendationEngine:
    """Return a singleton RecommendationEngine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = RecommendationEngine()
    return _default_engine
