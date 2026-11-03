"""
health_scorer.py
================
Dynamic machine health scoring for PredictX.

Produces a score from 0 to 100 for each sensor reading by measuring how
far the reading sits from the model's learned normal baseline.

Formula (transparent — no hidden arithmetic)
---------------------------------------------
Step 1  Per-feature penalty
    penalty(z) = min(1.0, (|z| / Z_CRITICAL) ** 2)

    At 0σ  → penalty = 0.00  (perfectly normal)
    At 1σ  → penalty = 0.11
    At 2σ  → penalty = 0.44
    At 3σ+ → penalty = 1.00  (capped at maximum)

    The quadratic shape rewards staying well within the normal range and
    accelerates as readings approach the critical threshold (3σ).

Step 2  Weighted combination
    raw_penalty = Σ weight[i] * penalty(z[i])

    Default weights reflect standard industrial engineering priorities:
        Rotational speed : 0.25   primary mechanical stress
        Torque           : 0.25   primary mechanical stress
        Air temperature  : 0.20   process health indicator
        Process temp     : 0.20   direct failure driver
        Tool wear        : 0.10   gradual wear — less acute

Step 3  Score conversion
    health_score = round(100 * (1 - raw_penalty))
    health_score = max(0, min(100, health_score))

Health bands (configurable)
    90–100  Healthy
    70–89   Warning
    40–69   Poor
    0–39    Critical

Usage
-----
    from predictx.health.health_scorer import HealthScorer
    scorer  = HealthScorer(scaler)
    score   = scorer.score_row(row_dict)          # single reading
    df_out  = scorer.score_dataframe(df)          # adds "health_score" column
    history = scorer.health_trend(df, machine_id) # per-machine trend
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from predictx.xai.feature_explainer import FEATURE_COLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Z-score at which penalty reaches 1.0 (maximum).
# 3σ is a standard engineering "critical deviation" threshold.
Z_CRITICAL: float = 3.0

# Default feature weights.  Must sum to 1.0.
DEFAULT_WEIGHTS: dict[str, float] = {
    "Air temperature [K]":      0.20,
    "Process temperature [K]":  0.20,
    "Rotational speed [rpm]":   0.25,
    "Torque [Nm]":              0.25,
    "Tool wear [min]":          0.10,
}

# Health bands (lower_bound, label, css_kind for dashboard)
HEALTH_BANDS: list[tuple[int, str, str]] = [
    (90, "Healthy",  "normal"),
    (70, "Warning",  "warning"),
    (40, "Poor",     "anomaly"),
    (0,  "Critical", "anomaly"),
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class HealthResult:
    """
    Health assessment for a single sensor reading.

    Attributes
    ----------
    score : int
        0–100 health score (100 = perfect, 0 = critical).
    band : str
        "Healthy" | "Warning" | "Poor" | "Critical"
    band_kind : str
        Design system kind for the dashboard badge.
    raw_penalty : float
        Weighted average penalty before conversion (0–1).
    feature_penalties : dict[str, float]
        Per-feature penalty values (0–1) — shows which sensor contributed
        most to the score reduction.
    feature_z_scores : dict[str, float]
        Per-feature Z-scores (signed) — for display alongside the score.
    """
    score:            int
    band:             str
    band_kind:        str
    raw_penalty:      float
    feature_penalties: dict[str, float]
    feature_z_scores:  dict[str, float]

    @property
    def is_healthy(self) -> bool:
        return self.score >= 90

    @property
    def needs_attention(self) -> bool:
        return self.score < 70

    @property
    def is_critical(self) -> bool:
        return self.score < 40


# ---------------------------------------------------------------------------
# HealthScorer
# ---------------------------------------------------------------------------

class HealthScorer:
    """
    Compute a 0–100 health score from sensor readings.

    Parameters
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler from training — provides baseline mean and std.
    weights : dict[str, float], optional
        Override the default feature weights.  Must cover all FEATURE_COLS
        and sum to 1.0 (±0.001 tolerance).
    """

    def __init__(
        self,
        scaler: StandardScaler,
        weights: dict[str, float] | None = None,
    ):
        if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
            raise ValueError("Scaler has not been fitted. Load from models/scaler.pkl.")

        self._scaler = scaler
        self._means: np.ndarray = scaler.mean_.copy()
        self._stds:  np.ndarray = scaler.scale_.copy()

        # Validate and store weights
        self._weights = self._validate_weights(weights or DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_row(self, row: dict) -> HealthResult:
        """
        Score a single sensor reading.

        Parameters
        ----------
        row : dict
            Must contain all FEATURE_COLS.  Extra keys are ignored.

        Returns
        -------
        HealthResult
        """
        values = self._extract_values(row)
        return self._compute_score(values)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a ``health_score`` column and a ``health_band`` column to a copy
        of the input DataFrame.  Missing feature values raise ValueError.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all FEATURE_COLS.

        Returns
        -------
        pd.DataFrame — copy of df with two new columns appended.
        """
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        scores = []
        bands  = []
        for _, row in df.iterrows():
            try:
                result = self.score_row(row.to_dict())
                scores.append(result.score)
                bands.append(result.band)
            except ValueError:
                # Row has a NaN sensor value — assign neutral score
                scores.append(50)
                bands.append("Unknown")

        out = df.copy()
        out["health_score"] = scores
        out["health_band"]  = bands
        return out

    def health_trend(
        self,
        df: pd.DataFrame,
        machine_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of health scores suitable for plotting as a
        time-series trend.

        If ``machine_col`` is provided (e.g. "Product ID"), the scores are
        grouped per machine and returned with a "machine_id" column.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all FEATURE_COLS (and optionally machine_col).
        machine_col : str or None
            Column identifying the machine.  If None, all rows are treated
            as a single machine named "M-001".

        Returns
        -------
        pd.DataFrame with columns:
            reading_index, machine_id, health_score, health_band
        """
        scored = self.score_dataframe(df)

        rows = []
        for idx, row in scored.iterrows():
            machine_id = str(row[machine_col]) if machine_col and machine_col in scored.columns else "M-001"
            rows.append({
                "reading_index": idx,
                "machine_id":    machine_id,
                "health_score":  int(row["health_score"]),
                "health_band":   row["health_band"],
            })

        return pd.DataFrame(rows)

    @property
    def weights(self) -> dict[str, float]:
        """Return the active feature weights."""
        return dict(self._weights)

    @property
    def baseline(self) -> dict[str, dict]:
        """Return per-feature baseline statistics (mean, std, normal range)."""
        return {
            col: {
                "mean":        round(float(self._means[i]), 3),
                "std":         round(float(self._stds[i]), 3),
                "normal_low":  round(float(self._means[i] - 2 * self._stds[i]), 3),
                "normal_high": round(float(self._means[i] + 2 * self._stds[i]), 3),
            }
            for i, col in enumerate(FEATURE_COLS)
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_values(self, row: dict) -> np.ndarray:
        values = []
        for col in FEATURE_COLS:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                raise ValueError(f"Missing or NaN value for feature '{col}'.")
            values.append(float(val))
        return np.array(values, dtype=float)

    def _compute_score(self, values: np.ndarray) -> HealthResult:
        """Core scoring arithmetic."""
        # Step 1 — per-feature Z-scores
        stds_safe = np.where(self._stds > 0, self._stds, 1.0)
        z_scores  = (values - self._means) / stds_safe
        abs_z     = np.abs(z_scores)

        # Step 2 — per-feature quadratic penalties (capped at 1.0)
        penalties = np.minimum(1.0, (abs_z / Z_CRITICAL) ** 2)

        # Step 3 — weighted sum
        weight_arr = np.array([self._weights[col] for col in FEATURE_COLS])
        raw_penalty = float(np.dot(weight_arr, penalties))

        # Step 4 — convert to 0–100 score
        score = max(0, min(100, round(100 * (1.0 - raw_penalty))))

        # Determine band
        band, band_kind = _classify_band(score)

        return HealthResult(
            score             = score,
            band              = band,
            band_kind         = band_kind,
            raw_penalty       = round(raw_penalty, 6),
            feature_penalties = {
                col: round(float(penalties[i]), 4)
                for i, col in enumerate(FEATURE_COLS)
            },
            feature_z_scores  = {
                col: round(float(z_scores[i]), 4)
                for i, col in enumerate(FEATURE_COLS)
            },
        )

    @staticmethod
    def _validate_weights(weights: dict[str, float]) -> dict[str, float]:
        missing = [c for c in FEATURE_COLS if c not in weights]
        if missing:
            raise ValueError(f"Weights missing for features: {missing}")
        total = sum(weights[c] for c in FEATURE_COLS)
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"Feature weights must sum to 1.0, got {total:.4f}. "
                "Adjust the weights so they total exactly 1.0."
            )
        return {col: weights[col] for col in FEATURE_COLS}  # canonical order


# ---------------------------------------------------------------------------
# Band classifier
# ---------------------------------------------------------------------------

def _classify_band(score: int) -> tuple[str, str]:
    """Return (band_label, badge_kind) for a health score."""
    for threshold, label, kind in HEALTH_BANDS:
        if score >= threshold:
            return label, kind
    return "Critical", "anomaly"


def classify_band(score: int) -> tuple[str, str]:
    """Public helper — classify any integer score into a health band."""
    return _classify_band(score)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_scorer: HealthScorer | None = None


def get_health_scorer() -> HealthScorer:
    """Return a singleton HealthScorer using the shared Predictor scaler."""
    global _default_scorer
    if _default_scorer is None:
        from predictx.model.predictor import get_predictor
        _default_scorer = HealthScorer(get_predictor().scaler)
    return _default_scorer
