"""
Tests for src/predictx/ingestion/validator.py
"""
import io
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predictx.ingestion.validator import (
    REQUIRED_COLS,
    OPTIONAL_COLS,
    COL_BOUNDS,
    validate_and_clean,
    detect_layout,
    generate_sample_csv,
)


def _make_df(**overrides) -> pd.DataFrame:
    """Create a minimal valid DataFrame with one row (no label columns)."""
    defaults = {
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 10.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_dataframe_passes():
    df = _make_df()
    clean, report = validate_and_clean(df)
    assert report["ok"] is True
    assert report["rows_out"] == 1
    # Feature columns must come first
    assert list(clean.columns[:len(REQUIRED_COLS)]) == REQUIRED_COLS


def test_valid_csv_string():
    csv_io = io.StringIO(
        "Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]\n"
        "298.1,308.6,1551,42.8,10\n"
        "299.0,309.0,1600,40.0,20\n"
    )
    clean, report = validate_and_clean(csv_io)
    assert report["ok"] is True
    assert report["rows_out"] == 2


def test_report_contains_layout_key():
    df = _make_df()
    _, report = validate_and_clean(df)
    assert "layout" in report
    assert report["layout"] == "minimal"


# ---------------------------------------------------------------------------
# Layout detection — UCI vs Kaggle vs minimal
# ---------------------------------------------------------------------------

def test_detect_layout_uci():
    df = _make_df()
    df["Machine failure"] = 0
    assert detect_layout(df) == "uci"


def test_detect_layout_kaggle():
    df = _make_df()
    df["Target"] = 0
    df["Failure Type"] = "No Failure"
    assert detect_layout(df) == "kaggle"


def test_detect_layout_minimal():
    df = _make_df()
    assert detect_layout(df) == "minimal"


def test_uci_layout_accepted():
    """Full UCI column set is accepted; Machine failure is passed through unchanged."""
    df = _make_df()
    df["Machine failure"] = 0
    df["TWF"] = 0
    df["HDF"] = 0
    clean, report = validate_and_clean(df)
    assert report["ok"] is True
    assert report["layout"] == "uci"
    assert "Machine failure" in clean.columns
    # Feature columns must NOT include Machine failure
    for col in ["Machine failure", "TWF", "HDF"]:
        assert col not in REQUIRED_COLS


def test_kaggle_layout_accepted():
    """Kaggle variant with Target + Failure Type is accepted."""
    df = _make_df()
    df["Target"] = 1
    df["Failure Type"] = "Tool Wear Failure"
    clean, report = validate_and_clean(df)
    assert report["ok"] is True
    assert report["layout"] == "kaggle"
    assert "Target" in clean.columns
    assert "Failure Type" in clean.columns
    # Target must NOT be in REQUIRED_COLS (never a model input)
    assert "Target" not in REQUIRED_COLS


def test_uci_label_not_used_as_feature():
    """Machine failure column must not appear in REQUIRED_COLS."""
    assert "Machine failure" not in REQUIRED_COLS


def test_kaggle_label_not_used_as_feature():
    """Target and Failure Type must not appear in REQUIRED_COLS."""
    assert "Target" not in REQUIRED_COLS
    assert "Failure Type" not in REQUIRED_COLS


def test_optional_cols_includes_both_label_variants():
    assert "Machine failure" in OPTIONAL_COLS
    assert "Target" in OPTIONAL_COLS
    assert "Failure Type" in OPTIONAL_COLS


def test_uci_full_dataset_columns():
    """Simulates loading the full ai4i2020.csv column set."""
    df = _make_df()
    for col in ["UDI", "Product ID", "Type", "Machine failure",
                "TWF", "HDF", "PWF", "OSF", "RNF"]:
        df[col] = 0
    clean, report = validate_and_clean(df)
    assert report["ok"] is True
    assert report["layout"] == "uci"
    # All five feature cols present
    for col in REQUIRED_COLS:
        assert col in clean.columns
    # Label cols present but not in REQUIRED_COLS
    assert "Machine failure" in clean.columns
    assert "Machine failure" not in REQUIRED_COLS


# ---------------------------------------------------------------------------
# Optional columns preserved
# ---------------------------------------------------------------------------

def test_optional_columns_preserved_uci():
    df = _make_df()
    df["Machine failure"] = 0
    df["Product ID"] = "M001"
    clean, report = validate_and_clean(df)
    assert "Machine failure" in clean.columns
    assert "Product ID" in clean.columns


def test_optional_columns_preserved_kaggle():
    df = _make_df()
    df["Target"] = 0
    df["Failure Type"] = "No Failure"
    clean, report = validate_and_clean(df)
    assert "Target" in clean.columns
    assert "Failure Type" in clean.columns


# ---------------------------------------------------------------------------
# Missing columns
# ---------------------------------------------------------------------------

def test_missing_required_column_fails():
    df = _make_df()
    df = df.drop(columns=["Torque [Nm]"])
    clean, report = validate_and_clean(df)
    assert report["ok"] is False
    assert any("Torque" in e for e in report["errors"])
    assert clean.empty


def test_missing_multiple_columns_fails():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    clean, report = validate_and_clean(df)
    assert report["ok"] is False
    assert len(report["errors"]) > 0


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

def test_string_values_coerced():
    df = _make_df(**{"Rotational speed [rpm]": "1500"})
    clean, report = validate_and_clean(df)
    assert report["ok"] is True
    # Accept any numeric dtype (int64 or float64 depending on numpy version)
    assert np.issubdtype(clean["Rotational speed [rpm]"].dtype, np.number)


def test_bad_string_becomes_nan_then_median():
    df = pd.DataFrame([
        {c: 298.1 if c == "Air temperature [K]" else
            308.6 if c == "Process temperature [K]" else
            "NOT_A_NUMBER" if c == "Rotational speed [rpm]" else
            42.8 if c == "Torque [Nm]" else 10.0
         for c in REQUIRED_COLS},
        {c: 299.0 if c == "Air temperature [K]" else
            309.0 if c == "Process temperature [K]" else
            1600.0 if c == "Rotational speed [rpm]" else
            40.0 if c == "Torque [Nm]" else 20.0
         for c in REQUIRED_COLS},
    ])
    clean, report = validate_and_clean(df)
    assert report["ok"] is True
    assert any("Rotational speed" in w for w in report["warnings"])
    assert clean["Rotational speed [rpm]"].notna().all()


# ---------------------------------------------------------------------------
# Out-of-range clipping
# ---------------------------------------------------------------------------

def test_out_of_range_values_clipped():
    df = _make_df(**{"Rotational speed [rpm]": 9999.0})
    clean, report = validate_and_clean(df, clip_outliers=True)
    assert report["ok"] is True
    assert report["clipped"] > 0
    hi = COL_BOUNDS["Rotational speed [rpm]"][1]
    assert clean["Rotational speed [rpm]"].iloc[0] == hi


def test_out_of_range_not_clipped_when_disabled():
    df = _make_df(**{"Rotational speed [rpm]": 9999.0})
    clean, report = validate_and_clean(df, clip_outliers=False)
    assert report["clipped"] == 0
    assert clean["Rotational speed [rpm]"].iloc[0] == 9999.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_null_rows_returns_error():
    df = pd.DataFrame([{c: None for c in REQUIRED_COLS}])
    clean, report = validate_and_clean(df)
    assert report["ok"] is False


def test_empty_dataframe():
    df = pd.DataFrame(columns=REQUIRED_COLS)
    clean, report = validate_and_clean(df)
    assert report["rows_in"] == 0


def test_generate_sample_csv(tmp_path):
    out = tmp_path / "sample.csv"
    result = generate_sample_csv(out, n=10)
    assert result.exists()
    df = pd.read_csv(result)
    assert len(df) == 10
    for col in REQUIRED_COLS:
        assert col in df.columns
