"""
predictor.py
============
Loads the trained Isolation Forest model and runs inference.

Usage
-----
    from predictx.model.predictor import Predictor
    p = Predictor()
    result_df = p.predict(df)
"""

from __future__ import annotations

import pathlib
from functools import cached_property

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


class Predictor:
    """Wrapper around the trained Isolation Forest for inference."""

    def __init__(
        self,
        model_path: pathlib.Path | str = MODEL_PATH,
        scaler_path: pathlib.Path | str = SCALER_PATH,
    ):
        self._model_path = pathlib.Path(model_path)
        self._scaler_path = pathlib.Path(scaler_path)
        self._model = None
        self._scaler = None

    # ------------------------------------------------------------------
    # Lazy loading — models are loaded on first use
    # ------------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self._model_path}. "
                    "Run `python training/train_model.py` first."
                )
            self._model = joblib.load(self._model_path)
        return self._model

    @property
    def scaler(self):
        if self._scaler is None:
            if not self._scaler_path.exists():
                raise FileNotFoundError(
                    f"Scaler not found at {self._scaler_path}. "
                    "Run `python training/train_model.py` first."
                )
            self._scaler = joblib.load(self._scaler_path)
        return self._scaler

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run anomaly detection on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all FEATURE_COLS columns.

        Returns
        -------
        pd.DataFrame
            Original df plus two new columns:
            - ``anomaly_score``  : float  (lower = more anomalous)
            - ``is_anomaly``     : int    (1 = anomaly, 0 = normal)
        """
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        X = self.scaler.transform(df[FEATURE_COLS].values)
        scores = self.model.decision_function(X)   # lower → more anomalous
        raw_preds = self.model.predict(X)          # -1 = anomaly, 1 = normal
        is_anomaly = np.where(raw_preds == -1, 1, 0)

        result = df.copy()
        result["anomaly_score"] = scores.round(6)
        result["is_anomaly"] = is_anomaly
        return result

    def predict_row(self, row: dict) -> dict:
        """
        Predict a single row given as a dict of feature values.

        Returns the row dict with ``anomaly_score`` and ``is_anomaly`` added.
        """
        df_row = pd.DataFrame([row])
        result = self.predict(df_row)
        return result.iloc[0].to_dict()

    @property
    def is_ready(self) -> bool:
        """Return True if model files exist and can be loaded."""
        return self._model_path.exists() and self._scaler_path.exists()


# Module-level singleton — reused across the application
_default_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Return a module-level singleton Predictor instance."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = Predictor()
    return _default_predictor
