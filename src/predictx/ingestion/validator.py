"""
validator.py
============
CSV schema validation and cleaning for PredictX sensor data.

Accepts a CSV file path or an existing DataFrame.
Returns a cleaned DataFrame and a validation report dict.

Supported CSV layouts
---------------------
Both the original UCI download and the Kaggle mirror are accepted.

UCI / ai4i2020.csv layout (columns 0-13):
  UDI, Product ID, Type,
  Air temperature [K], Process temperature [K],
  Rotational speed [rpm], Torque [Nm], Tool wear [min],
  Machine failure, TWF, HDF, PWF, OSF, RNF

Kaggle mirror layout:
  UDI, Product ID, Type,
  Air temperature [K], Process temperature [K],
  Rotational speed [rpm], Torque [Nm], Tool wear [min],
  Target, Failure Type

Minimal upload layout (no label columns at all):
  Air temperature [K], Process temperature [K],
  Rotational speed [rpm], Torque [Nm], Tool wear [min]

The five sensor feature columns are identical across all layouts and are
the only mandatory columns.  Label columns (``Machine failure``, ``Target``,
``Failure Type``, individual failure flags) are treated as optional
evaluation metadata — they are passed through but never fed to the model.
"""

from __future__ import annotations

import io
import pathlib
from typing import Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# The five sensor columns that the Isolation Forest is trained on.
# These are the ONLY mandatory columns — identical in every CSV variant.
REQUIRED_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Optional metadata columns present in the UCI original.
_OPTIONAL_UCI = [
    "UDI",
    "Product ID",
    "Type",
    "Machine failure",   # binary label (UCI)
    "TWF", "HDF", "PWF", "OSF", "RNF",  # individual failure-mode flags
]

# Optional metadata columns present in the Kaggle mirror.
_OPTIONAL_KAGGLE = [
    "UDI",
    "Product ID",
    "Type",
    "Target",            # binary label (Kaggle, equivalent to Machine failure)
    "Failure Type",      # multi-class failure label (Kaggle)
]

# All optional columns across every known variant — passed through if present.
OPTIONAL_COLS: list[str] = list(
    dict.fromkeys(_OPTIONAL_UCI + _OPTIONAL_KAGGLE)  # deduplicated, order-stable
)

# Reasonable sensor bounds (min, max) — used for clipping / warning only.
COL_BOUNDS: dict[str, tuple[float, float]] = {
    "Air temperature [K]":      (290.0, 310.0),
    "Process temperature [K]":  (300.0, 320.0),
    "Rotational speed [rpm]":   (1000.0, 3000.0),
    "Torque [Nm]":              (0.0, 100.0),
    "Tool wear [min]":          (0.0, 300.0),
}


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def detect_layout(df: pd.DataFrame) -> str:
    """
    Return a string describing which CSV variant was detected.

    Returns one of:
      'uci'     – has Machine failure / individual failure flags
      'kaggle'  – has Target / Failure Type
      'minimal' – only the five sensor feature columns
    """
    if "Machine failure" in df.columns:
        return "uci"
    if "Target" in df.columns or "Failure Type" in df.columns:
        return "kaggle"
    return "minimal"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_and_clean(
    source: Union[str, pathlib.Path, pd.DataFrame, io.IOBase],
    clip_outliers: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Validate sensor data, clean it, and return (clean_df, report).

    Parameters
    ----------
    source :
        A CSV file path, a file-like object, or an existing DataFrame.
        Accepts the UCI original, the Kaggle mirror, or any CSV that
        contains at minimum the five REQUIRED_COLS sensor columns.
    clip_outliers :
        If True, clip values that exceed COL_BOUNDS to the boundary
        rather than dropping the row.

    Returns
    -------
    clean_df : pd.DataFrame
        DataFrame with REQUIRED_COLS first, then any optional metadata
        columns that were present in the source (label columns kept as-is).
    report : dict
        Keys: ok (bool), errors (list[str]), warnings (list[str]),
              rows_in (int), rows_out (int), clipped (int),
              layout (str: 'uci' | 'kaggle' | 'minimal').
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        try:
            df = pd.read_csv(source)
        except Exception as exc:
            errors.append(f"Cannot read CSV: {exc}")
            return pd.DataFrame(), _report(errors, warnings, 0, 0, 0, "unknown")

    rows_in = len(df)
    layout  = detect_layout(df)

    # ------------------------------------------------------------------
    # 2. Required column check
    # ------------------------------------------------------------------
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return pd.DataFrame(), _report(errors, warnings, rows_in, 0, 0, layout)

    # ------------------------------------------------------------------
    # 3. Drop fully-empty rows
    # ------------------------------------------------------------------
    df = df.dropna(subset=REQUIRED_COLS, how="all")
    if len(df) == 0:
        errors.append("All rows are empty after dropping nulls.")
        return pd.DataFrame(), _report(errors, warnings, rows_in, 0, 0, layout)

    # ------------------------------------------------------------------
    # 4. Type coercion — convert feature columns to numeric
    # ------------------------------------------------------------------
    for col in REQUIRED_COLS:
        original_nulls = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        new_nulls = df[col].isna().sum() - original_nulls
        if new_nulls > 0:
            warnings.append(f"Column '{col}': {new_nulls} non-numeric value(s) → NaN")

    # ------------------------------------------------------------------
    # 5. Fill remaining NaNs in feature columns with column median
    # ------------------------------------------------------------------
    null_counts = df[REQUIRED_COLS].isna().sum()
    for col, cnt in null_counts.items():
        if cnt > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            warnings.append(
                f"Column '{col}': {cnt} NaN(s) filled with median={median_val:.2f}"
            )

    # ------------------------------------------------------------------
    # 6. Clip / flag out-of-range values in feature columns
    # ------------------------------------------------------------------
    clipped = 0
    for col, (lo, hi) in COL_BOUNDS.items():
        if col not in df.columns:
            continue
        out_mask = (df[col] < lo) | (df[col] > hi)
        n_out = int(out_mask.sum())
        if n_out > 0:
            warnings.append(
                f"Column '{col}': {n_out} value(s) outside [{lo}, {hi}]"
                + (" — clipped" if clip_outliers else " — kept as-is")
            )
            if clip_outliers:
                df[col] = df[col].clip(lower=lo, upper=hi)
                clipped += n_out

    # ------------------------------------------------------------------
    # 7. Keep feature columns + any optional metadata columns present.
    #    Label columns (Machine failure, Target, Failure Type, TWF, …)
    #    are passed through unchanged — they are NEVER used as model input.
    # ------------------------------------------------------------------
    present_optional = [c for c in OPTIONAL_COLS if c in df.columns]
    keep = REQUIRED_COLS + present_optional
    df = df[keep].reset_index(drop=True)

    # Report which label column is available (for UI display)
    if "Machine failure" in df.columns:
        warnings.append("Label column 'Machine failure' detected (UCI layout) — used for evaluation only, not model input.")
    elif "Target" in df.columns:
        warnings.append("Label column 'Target' detected (Kaggle layout) — used for evaluation only, not model input.")

    rows_out = len(df)
    return df, _report(errors, warnings, rows_in, rows_out, clipped, layout)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report(
    errors: list[str],
    warnings: list[str],
    rows_in: int,
    rows_out: int,
    clipped: int,
    layout: str = "unknown",
) -> dict:
    return {
        "ok":       len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
        "rows_in":  rows_in,
        "rows_out": rows_out,
        "clipped":  clipped,
        "layout":   layout,
    }


def generate_sample_csv(out_path: Union[str, pathlib.Path], n: int = 20) -> pathlib.Path:
    """Write a small sample CSV for dashboard upload testing (minimal layout)."""
    rng = np.random.default_rng(42)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "Air temperature [K]":      rng.uniform(295, 305, n).round(1),
        "Process temperature [K]":  rng.uniform(306, 315, n).round(1),
        "Rotational speed [rpm]":   rng.integers(1200, 2800, n).astype(float),
        "Torque [Nm]":              rng.uniform(10, 70, n).round(1),
        "Tool wear [min]":          rng.integers(0, 250, n).astype(float),
    })
    # Inject a couple of anomalous rows (only if the dataset is large enough)
    if n > 5:
        df.loc[5, "Rotational speed [rpm]"] = 2950
        df.loc[5, "Torque [Nm]"] = 5.0
    if n > 12:
        df.loc[12, "Tool wear [min]"] = 295
        df.loc[12, "Air temperature [K]"] = 304.5

    df.to_csv(out_path, index=False)
    return out_path
