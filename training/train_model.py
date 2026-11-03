"""
train_model.py
==============
Trains an Isolation Forest anomaly-detection model on the AI4I 2020
Predictive Maintenance Dataset and saves the artefacts to models/.

Run from the project root:
    python training/train_model.py
"""

import os
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# ---------------------------------------------------------------------------
# Feature columns used for training (sensor inputs only)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Binary failure label — column name differs between UCI and Kaggle variants
# UCI:    "Machine failure"
# Kaggle: "Target"
# Training uses whichever is present; if neither exists evaluation is skipped.
_LABEL_CANDIDATES = ["Machine failure", "Target"]


def _find_label_col(df: pd.DataFrame) -> str | None:
    """Return the name of the binary failure label column, or None."""
    for col in _LABEL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load the CSV and return a clean DataFrame."""
    df = pd.read_csv(path)
    print(f"[data]   Loaded {len(df):,} rows from {path.name}")
    print(f"[data]   Columns: {list(df.columns)}")
    label_col = _find_label_col(df)
    if label_col:
        print(f"[data]   Label column: '{label_col}'  |  failure rate: {df[label_col].mean():.2%}")
    else:
        print("[data]   No label column found — evaluation step will be skipped.")
    return df


def train(df: pd.DataFrame):
    """Fit scaler + Isolation Forest; return (scaler, model)."""
    X = df[FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("[train]  StandardScaler fitted.")

    # contamination ≈ observed failure rate (~3 %) — slightly relaxed to 0.05
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    print("[train]  IsolationForest fitted.")
    return scaler, model


def evaluate(df: pd.DataFrame, scaler: StandardScaler, model: IsolationForest):
    """
    Evaluate against the ground-truth failure label when available.

    Works with both the UCI label ('Machine failure') and the Kaggle label
    ('Target').  If neither is present the function still returns a scored
    DataFrame — evaluation metrics are simply omitted.

    IsolationForest returns  1 = normal, -1 = anomaly.
    We map -1 → 1 and 1 → 0 to match the binary failure convention.
    """
    X = scaler.transform(df[FEATURE_COLS].values)
    raw_preds = model.predict(X)          # 1 or -1
    y_pred    = np.where(raw_preds == -1, 1, 0)
    scores    = model.decision_function(X)

    label_col = _find_label_col(df)
    if label_col:
        y_true = df[label_col].values
        print(f"\n[eval]   Classification report (vs. '{label_col}'):")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Failure"]))
        print("[eval]   Confusion matrix (rows=actual, cols=predicted):")
        cm = confusion_matrix(y_true, y_pred)
        print(f"         TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"         FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    else:
        print("\n[eval]   No label column — skipping classification report.")

    df = df.copy()
    df["anomaly_score"] = scores
    df["is_anomaly"]    = y_pred
    return df, label_col


def save_artefacts(scaler: StandardScaler, model: IsolationForest):
    joblib.dump(scaler, SCALER_PATH)
    print(f"[save]   Scaler  → {SCALER_PATH}")
    joblib.dump(model, MODEL_PATH)
    print(f"[save]   Model   → {MODEL_PATH}")


def main():
    print("=" * 60)
    print("PredictX — Isolation Forest Training")
    print("=" * 60)

    df = load_data(DATA_PATH)
    scaler, model = train(df)
    result_df, label_col = evaluate(df, scaler, model)

    save_artefacts(scaler, model)

    # Quick sanity check: top 5 anomalies
    extra_cols = [label_col] if label_col else []
    top_anomalies = (
        result_df[result_df["is_anomaly"] == 1]
        .sort_values("anomaly_score")
        .head(5)[FEATURE_COLS + ["anomaly_score"] + extra_cols]
    )
    print("\n[info]   Top 5 most anomalous rows:")
    print(top_anomalies.to_string(index=False))
    print("\n[done]   Training complete.")


if __name__ == "__main__":
    main()
