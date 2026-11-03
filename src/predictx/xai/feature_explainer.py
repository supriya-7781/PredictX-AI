"""
feature_explainer.py
====================
Explainable AI module for PredictX.

Answers the question: *why* did the Isolation Forest flag this reading?

Method — Standardised Deviation Analysis
-----------------------------------------
The ``StandardScaler`` fitted during training stores two arrays:
  - ``scaler.mean_``  : the mean of each feature across the 10 000 training rows
  - ``scaler.scale_`` : the standard deviation of each feature

A Z-score  ``z = (current_value - mean) / std``  tells us, in a statistically
grounded way, how far each sensor reading lies from the model's learned
"normal" operating centre.

Why not SHAP?
  ``IsolationForest.decision_function`` is not reliably supported by
  ``shap.TreeExplainer`` — the SHAP values for path-length-based anomaly
  scores are not mathematically stable across tree ensembles.  Z-score
  analysis uses the exact same numerical space the model was fitted in, so
  it is both honest and reproducible.

Significance threshold: |z| > 2.0  (2 standard deviations from normal —
a standard engineering criterion for "noteworthy deviation").

Usage
-----
    from predictx.xai.feature_explainer import FeatureExplainer
    explainer = FeatureExplainer(scaler)                 # pass loaded scaler
    result    = explainer.explain(row_dict)              # single reading
    results   = explainer.explain_batch(df)              # DataFrame of readings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

# Must match the order used during training (StandardScaler was fitted on
# these columns in this exact order).
FEATURE_COLS: list[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Short display names — used in charts and narrative text
FEATURE_SHORT: dict[str, str] = {
    "Air temperature [K]":      "Air temperature",
    "Process temperature [K]":  "Process temperature",
    "Rotational speed [rpm]":   "Rotational speed",
    "Torque [Nm]":              "Torque",
    "Tool wear [min]":          "Tool wear",
}

# Units for display
FEATURE_UNITS: dict[str, str] = {
    "Air temperature [K]":      "K",
    "Process temperature [K]":  "K",
    "Rotational speed [rpm]":   "rpm",
    "Torque [Nm]":              "Nm",
    "Tool wear [min]":          "min",
}

# Significance threshold — readings beyond this Z-score are flagged
Z_SIGNIFICANT: float = 2.0

# Critical threshold — readings beyond this are highlighted as severe
Z_CRITICAL: float = 3.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    """
    The deviation profile for a single sensor feature.

    All values are computed from the trained scaler — no arbitrary numbers.
    """
    feature: str          # Full column name, e.g. "Air temperature [K]"
    short_name: str        # Display name, e.g. "Air temperature"
    unit: str              # Physical unit, e.g. "K"
    current_value: float   # The actual sensor reading
    baseline_mean: float   # scaler.mean_ for this feature (training average)
    baseline_std: float    # scaler.scale_ for this feature (training std dev)
    z_score: float         # (current - mean) / std  — signed deviation
    abs_z: float           # |z_score| — used for ranking
    rank: int              # 1 = largest contributor to anomaly
    direction: str         # "above normal" | "below normal" | "within normal range"
    is_significant: bool   # True if abs_z > Z_SIGNIFICANT

    @property
    def deviation_pct(self) -> float:
        """Percentage deviation from the baseline mean (signed)."""
        if self.baseline_mean == 0:
            return 0.0
        return (self.current_value - self.baseline_mean) / abs(self.baseline_mean) * 100

    @property
    def severity_label(self) -> str:
        """Human-readable severity based on Z-score magnitude."""
        if self.abs_z >= Z_CRITICAL:
            return "severe"
        if self.abs_z >= Z_SIGNIFICANT:
            return "notable"
        if self.abs_z >= 1.0:
            return "mild"
        return "within normal range"


@dataclass
class ExplanationResult:
    """
    Complete XAI explanation for a single sensor reading.

    Attributes
    ----------
    contributions :
        List of FeatureContribution objects, sorted by abs_z descending
        (most anomalous feature first).
    primary_driver :
        Short name of the feature with the highest absolute Z-score.
    n_significant :
        Number of features with abs_z > Z_SIGNIFICANT.
    overall_deviation :
        Mean absolute Z-score across all 5 features — a summary statistic.
    narrative :
        Human-readable paragraph built from the contributions.
        Facts only — no claims about specific component failures.
    anomaly_score :
        The original Isolation Forest anomaly_score (passed through for
        reference — not recomputed here).
    """
    contributions: list[FeatureContribution]
    primary_driver: str
    n_significant: int
    overall_deviation: float
    narrative: str
    anomaly_score: float = 0.0

    def top_contributors(self, n: int = 3) -> list[FeatureContribution]:
        """Return the top-n contributors by absolute Z-score."""
        return sorted(self.contributions, key=lambda c: c.abs_z, reverse=True)[:n]

    def significant_contributors(self) -> list[FeatureContribution]:
        """Return only features that exceed the significance threshold."""
        return [c for c in self.contributions if c.is_significant]

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for storage or logging)."""
        return {
            "primary_driver": self.primary_driver,
            "n_significant": self.n_significant,
            "overall_deviation": round(self.overall_deviation, 4),
            "anomaly_score": round(self.anomaly_score, 6),
            "narrative": self.narrative,
            "contributions": [
                {
                    "feature": c.feature,
                    "current_value": round(c.current_value, 3),
                    "baseline_mean": round(c.baseline_mean, 3),
                    "baseline_std": round(c.baseline_std, 3),
                    "z_score": round(c.z_score, 4),
                    "direction": c.direction,
                    "rank": c.rank,
                    "is_significant": c.is_significant,
                }
                for c in self.contributions
            ],
        }


# ---------------------------------------------------------------------------
# FeatureExplainer
# ---------------------------------------------------------------------------

class FeatureExplainer:
    """
    Computes feature-level deviation analysis for Isolation Forest anomalies.

    Parameters
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler from training.  Its ``mean_`` and ``scale_``
        arrays define the "normal operating baseline" for each sensor.
    """

    def __init__(self, scaler: StandardScaler):
        if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
            raise ValueError(
                "Scaler has not been fitted yet.  "
                "Load it from models/scaler.pkl."
            )
        if len(scaler.mean_) != len(FEATURE_COLS):
            raise ValueError(
                f"Scaler has {len(scaler.mean_)} features but "
                f"FEATURE_COLS has {len(FEATURE_COLS)}.  "
                "Retrain the model with the current feature set."
            )
        self._scaler = scaler
        # Cache baseline arrays for speed
        self._means: np.ndarray  = scaler.mean_.copy()
        self._stds:  np.ndarray  = scaler.scale_.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        row: dict,
        anomaly_score: float = 0.0,
    ) -> ExplanationResult:
        """
        Explain a single sensor reading.

        Parameters
        ----------
        row : dict
            Must contain all keys in FEATURE_COLS.  Extra keys are ignored.
        anomaly_score : float
            The Isolation Forest anomaly_score for this row (for reference).

        Returns
        -------
        ExplanationResult
        """
        # Extract and validate feature values
        values = self._extract_values(row)

        # Compute Z-scores
        z_scores = (values - self._means) / np.where(self._stds > 0, self._stds, 1.0)

        # Build per-feature contributions
        contributions = self._build_contributions(values, z_scores)

        # Assemble result
        return self._assemble_result(contributions, anomaly_score)

    def explain_batch(
        self,
        df: pd.DataFrame,
        anomaly_score_col: str = "anomaly_score",
    ) -> list[ExplanationResult]:
        """
        Explain every row in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all FEATURE_COLS.
        anomaly_score_col : str
            Column name holding the pre-computed anomaly score (optional).

        Returns
        -------
        list of ExplanationResult — one per row, in DataFrame row order.
        """
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        results = []
        for _, row in df.iterrows():
            score = float(row.get(anomaly_score_col, 0.0))
            results.append(self.explain(row.to_dict(), anomaly_score=score))
        return results

    @property
    def baseline(self) -> dict[str, dict]:
        """
        Return the learned baseline statistics for each feature.

        Useful for display in the dashboard ("Normal range: X ± Y").
        """
        return {
            col: {
                "mean": round(float(self._means[i]), 3),
                "std":  round(float(self._stds[i]), 3),
                "normal_low":  round(float(self._means[i] - 2 * self._stds[i]), 3),
                "normal_high": round(float(self._means[i] + 2 * self._stds[i]), 3),
            }
            for i, col in enumerate(FEATURE_COLS)
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_values(self, row: dict) -> np.ndarray:
        """Extract feature values in FEATURE_COLS order, raising on missing."""
        values = []
        for col in FEATURE_COLS:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                raise ValueError(
                    f"Missing or NaN value for required feature '{col}'. "
                    "Ensure the row has been validated before explaining."
                )
            values.append(float(val))
        return np.array(values, dtype=float)

    def _build_contributions(
        self,
        values: np.ndarray,
        z_scores: np.ndarray,
    ) -> list[FeatureContribution]:
        """Build FeatureContribution objects sorted by abs_z descending."""
        abs_zs = np.abs(z_scores)

        # Rank by absolute Z-score (1 = largest)
        # argsort gives ascending order — reverse for descending rank
        order = np.argsort(abs_zs)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)

        contributions = []
        for i, col in enumerate(FEATURE_COLS):
            z  = float(z_scores[i])
            az = float(abs_zs[i])

            if z > 0.2:
                direction = "above normal"
            elif z < -0.2:
                direction = "below normal"
            else:
                direction = "within normal range"

            contributions.append(FeatureContribution(
                feature        = col,
                short_name     = FEATURE_SHORT[col],
                unit           = FEATURE_UNITS[col],
                current_value  = float(values[i]),
                baseline_mean  = float(self._means[i]),
                baseline_std   = float(self._stds[i]),
                z_score        = round(z, 4),
                abs_z          = round(az, 4),
                rank           = int(ranks[i]),
                direction      = direction,
                is_significant = az >= Z_SIGNIFICANT,
            ))

        # Sort by abs_z descending so index 0 is always the top contributor
        contributions.sort(key=lambda c: c.abs_z, reverse=True)
        return contributions

    def _assemble_result(
        self,
        contributions: list[FeatureContribution],
        anomaly_score: float,
    ) -> ExplanationResult:
        """Assemble the final ExplanationResult including the narrative."""
        significant = [c for c in contributions if c.is_significant]
        overall_dev = float(np.mean([c.abs_z for c in contributions]))
        primary     = contributions[0].short_name if contributions else "unknown"

        narrative = _build_narrative(contributions, significant)

        return ExplanationResult(
            contributions    = contributions,
            primary_driver   = primary,
            n_significant    = len(significant),
            overall_deviation= round(overall_dev, 4),
            narrative        = narrative,
            anomaly_score    = round(anomaly_score, 6),
        )


# ---------------------------------------------------------------------------
# Narrative builder
# ---------------------------------------------------------------------------

def _build_narrative(
    contributions: list[FeatureContribution],
    significant: list[FeatureContribution],
) -> str:
    """
    Build a plain-English explanation from the contribution list.

    Rules:
    - Only state measured facts and their Z-score deviations.
    - Do not claim a specific component has failed.
    - Use "above/below its normal operating range" rather than "too high/low".
    """
    if not significant:
        # All features within 2σ — collective mild deviation
        mild = [c for c in contributions if c.abs_z >= 1.0]
        if not mild:
            return (
                "All sensor readings are within their normal operating ranges "
                "(all within 2 standard deviations of the training baseline). "
                "The anomaly classification may reflect a subtle combination of "
                "mildly elevated readings rather than a single dominant fault."
            )
        mild_names = ", ".join(c.short_name for c in mild[:3])
        return (
            f"No individual sensor reading is statistically exceptional. "
            f"{mild_names} show mild deviations (1–2 standard deviations) "
            f"that, in combination, contributed to the anomaly classification. "
            f"The pattern is borderline — monitor the next several readings "
            f"before concluding that maintenance is required."
        )

    parts = []

    for c in significant:
        z_str    = f"{c.z_score:+.2f}"
        val_str  = f"{c.current_value:.1f} {c.unit}"
        base_str = f"{c.baseline_mean:.1f} {c.unit}"
        dev_str  = f"{abs(c.deviation_pct):.0f}% {c.direction}"

        severity = c.severity_label
        sentence = (
            f"{c.short_name} is {val_str}, which is {dev_str} "
            f"({z_str}σ from the training baseline of {base_str}). "
            f"This is a {severity} deviation."
        )
        parts.append(sentence)

    # Add context about non-significant features if they have mild elevation
    mild_non_sig = [c for c in contributions if not c.is_significant and c.abs_z >= 1.0]
    if mild_non_sig:
        names = " and ".join(c.short_name for c in mild_non_sig[:2])
        parts.append(
            f"{names} also show mild deviation (1–2σ) but are not individually significant."
        )

    # Combine and add attribution statement
    body = "  ".join(parts)

    n = len(significant)
    if n == 1:
        summary = (
            f"  {significant[0].short_name} is the primary contributor "
            f"to this anomaly classification."
        )
    else:
        names = " and ".join(c.short_name for c in significant[:2])
        summary = (
            f"  {names} are the primary contributors to this anomaly classification."
        )

    return body + summary


# ---------------------------------------------------------------------------
# Module-level factory (reuses the Predictor's already-loaded scaler)
# ---------------------------------------------------------------------------

_default_explainer: FeatureExplainer | None = None


def get_explainer() -> FeatureExplainer:
    """
    Return a singleton FeatureExplainer, loading the scaler from disk
    via the shared Predictor singleton if needed.

    This avoids loading the scaler twice — the Predictor already holds it.
    """
    global _default_explainer
    if _default_explainer is None:
        # Import here to avoid circular imports at module load time
        from predictx.model.predictor import get_predictor
        predictor = get_predictor()
        # Access the scaler property (loads from disk on first access)
        _default_explainer = FeatureExplainer(predictor.scaler)
    return _default_explainer
