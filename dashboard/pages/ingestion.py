"""
Data Ingestion page
====================
CSV upload with validation feedback, data preview, and inference trigger.
Results are stored in session_state for use by the Anomaly Detection and
AI Maintenance pages.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from design import (
    C, inject_css, page_header, divider, info_card,
    section_label, apply_plotly_theme, status_badge,
)
from predictx.ingestion.validator import validate_and_clean, REQUIRED_COLS, COL_BOUNDS
from predictx.model.predictor import get_predictor
from predictx.storage import store

inject_css()

page_header(
    "Data Ingestion",
    "Upload a sensor CSV, validate schema, and run anomaly detection",
    "📂",
)

predictor = get_predictor()

# ── Source selection ────────────────────────────────────────────────────────
section_label("Data Source")

tab_upload, tab_sample = st.tabs(["Upload CSV", "Use Sample Dataset"])

uploaded_file = None

with tab_upload:
    st.markdown("<br>", unsafe_allow_html=True)

    # Required columns info
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};border-radius:8px;
            padding:14px 18px;margin-bottom:16px">
  <div style="color:{C['text_muted']};font-size:0.7rem;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:8px">Required CSV Columns</div>
  <div style="display:flex;flex-wrap:wrap;gap:6px">
    {"".join(f'<code style="background:{C["surface2"]};color:{C["accent"]};padding:3px 8px;border-radius:4px;font-size:0.78rem">{c}</code>' for c in REQUIRED_COLS)}
  </div>
</div>
""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your CSV here or click to browse",
        type=["csv"],
        label_visibility="collapsed",
    )

with tab_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    sample_path = ROOT / "data" / "sample" / "sample_upload.csv"
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};border-radius:8px;
            padding:20px">
  <div style="color:{C['text']};font-weight:600;margin-bottom:6px">Sample Sensor Dataset</div>
  <div style="color:{C['text_muted']};font-size:0.82rem;margin-bottom:16px">
    30-row excerpt from the AI4I 2020 Predictive Maintenance Dataset with two
    injected anomalous readings — ready to explore without uploading a file.
  </div>
""", unsafe_allow_html=True)
    if st.button("Load Sample Dataset", type="primary"):
        if sample_path.exists():
            st.session_state["sample_bytes"] = sample_path.read_bytes()
            st.session_state["sample_loaded"] = True
        else:
            st.error("Sample CSV not found. Run `python training/train_model.py` first.")
    st.markdown("</div>", unsafe_allow_html=True)

# Allow the sample to persist across reruns
import io as _io
if st.session_state.get("sample_loaded") and uploaded_file is None:
    _raw = st.session_state.get("sample_bytes")
    if _raw is None and sample_path.exists():
        _raw = sample_path.read_bytes()
        st.session_state["sample_bytes"] = _raw
    if _raw is not None:
        uploaded_file = _io.BytesIO(_raw)

# ── Clear button ─────────────────────────────────────────────────────────────
if st.session_state.get("ingest_result") is not None:
    if st.button("Clear results", type="secondary"):
        for k in ["ingest_result", "ingest_report", "sample_loaded", "sample_bytes"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── Process uploaded / sample file ──────────────────────────────────────────
if uploaded_file is not None:
    divider()
    section_label("Validation")

    with st.spinner("Validating data..."):
        clean_df, report = validate_and_clean(uploaded_file)

    # Validation summary row
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Rows received",   report["rows_in"])
    v2.metric("Rows accepted",   report["rows_out"])
    v3.metric("Rows rejected",   report["rows_in"] - report["rows_out"])
    v4.metric("Values clipped",  report["clipped"])

    if report["errors"]:
        for e in report["errors"]:
            info_card(f"<strong>Error:</strong> {e}", kind="error")
        st.stop()

    if report["warnings"]:
        with st.expander(f"Validation warnings ({len(report['warnings'])})", expanded=False):
            for w in report["warnings"]:
                st.markdown(f"""
<div style="color:{C['warning']};font-size:0.82rem;padding:4px 0;
            border-bottom:1px solid {C['border']}">◦ {w}</div>
""", unsafe_allow_html=True)
    else:
        info_card("All rows passed validation with no warnings.", kind="success")

    divider()
    section_label("Data Preview")

    # Compact preview table
    preview = clean_df.head(10).copy()
    st.dataframe(
        preview.style.format(precision=2),
        use_container_width=True,
        height=280,
    )
    st.caption(f"Showing first 10 of {len(clean_df):,} rows · {len(clean_df.columns)} columns")

    divider()
    section_label("Inference")

    if not predictor.is_ready:
        info_card(
            "ML model is not trained. Run <code>python training/train_model.py</code> first.",
            kind="error",
        )
        st.stop()

    run_btn = st.button("Run Anomaly Detection", type="primary")

    # Auto-run if coming from sample or previous upload
    auto_run = "ingest_result" not in st.session_state

    if run_btn or (auto_run and uploaded_file is not None):
        with st.spinner("Running Isolation Forest inference..."):
            try:
                result_df = predictor.predict(clean_df)
                st.session_state["ingest_result"] = result_df
                st.session_state["ingest_report"] = report
            except Exception as exc:
                info_card(f"Inference failed: {exc}", kind="error")
                st.stop()

    result_df = st.session_state.get("ingest_result")

    if result_df is not None:
        n_total    = len(result_df)
        n_anomaly  = int(result_df["is_anomaly"].sum())
        n_normal   = n_total - n_anomaly
        anom_rate  = n_anomaly / n_total * 100

        divider()
        section_label("Results Summary")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total readings",    f"{n_total:,}")
        r2.metric("Normal",            f"{n_normal:,}")
        r3.metric("Anomalies flagged", f"{n_anomaly}",
                  delta=f"{anom_rate:.1f}% rate", delta_color="inverse")
        r4.metric("Lowest score",      f"{result_df['anomaly_score'].min():.4f}")

        divider()
        section_label("Sensor Charts")

        ch_left, ch_right = st.columns(2)

        # Score distribution
        with ch_left:
            score_map = result_df["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
            fig_hist = px.histogram(
                result_df,
                x="anomaly_score",
                color=score_map,
                nbins=35,
                color_discrete_map={"Normal": C["blue"], "Anomaly": C["anomaly"]},
                labels={"anomaly_score": "Anomaly Score", "color": "Status"},
                title="Anomaly Score Distribution",
            )
            fig_hist.add_vline(
                x=0, line_dash="dot", line_color=C["warning"],
                annotation_text="Threshold", annotation_font_color=C["warning"],
                annotation_font_size=11,
            )
            apply_plotly_theme(fig_hist, height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

        # RPM vs Torque scatter
        with ch_right:
            fig_scatter = px.scatter(
                result_df,
                x="Rotational speed [rpm]",
                y="Torque [Nm]",
                color=score_map,
                color_discrete_map={"Normal": C["blue"], "Anomaly": C["anomaly"]},
                hover_data=["Tool wear [min]", "anomaly_score"],
                title="Rotational Speed vs Torque",
                opacity=0.75,
            )
            apply_plotly_theme(fig_scatter, height=300)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Tool wear timeline
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=result_df.index,
            y=result_df["Tool wear [min]"],
            mode="lines",
            name="Tool wear",
            line=dict(color=C["blue"], width=1.5),
        ))
        anom_rows = result_df[result_df["is_anomaly"] == 1]
        fig_line.add_trace(go.Scatter(
            x=anom_rows.index,
            y=anom_rows["Tool wear [min]"],
            mode="markers",
            name="Anomaly",
            marker=dict(color=C["anomaly"], size=7, symbol="x", line=dict(width=2)),
        ))
        fig_line.update_layout(title="Tool Wear Over Readings")
        apply_plotly_theme(fig_line, height=260)
        st.plotly_chart(fig_line, use_container_width=True)

        divider()
        section_label("Full Results Table")

        display_df = result_df.copy()
        display_df["Status"] = display_df["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
        display_df = display_df.drop(columns=["is_anomaly"])
        display_df["anomaly_score"] = display_df["anomaly_score"].round(5)

        st.dataframe(
            display_df.style.apply(
                lambda col: [
                    f"color:{C['anomaly']};font-weight:600" if v == "Anomaly" else ""
                    for v in col
                ] if col.name == "Status" else [""] * len(col),
                axis=0,
            ),
            use_container_width=True,
            height=380,
        )

        divider()
        save_col, _ = st.columns([1, 3])
        with save_col:
            if st.button("Save results to storage", type="primary", use_container_width=True):
                with st.spinner("Saving..."):
                    loc = store.save_result(result_df)
                info_card(f"Saved to: <code>{loc}</code>", kind="success")

        info_card(
            "Results are now available in <strong>Anomaly Detection</strong> "
            "and <strong>AI Maintenance</strong> pages.",
            kind="info",
        )

elif not st.session_state.get("sample_loaded"):
    # Empty state
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="text-align:center;padding:60px 20px;color:{C['text_muted']}">
  <div style="font-size:2.5rem;margin-bottom:16px;opacity:0.4">↑</div>
  <div style="font-size:1rem;font-weight:500;color:{C['text']};margin-bottom:8px">
    No data loaded
  </div>
  <div style="font-size:0.85rem">
    Upload a CSV file above or use the sample dataset to get started.
  </div>
</div>
""", unsafe_allow_html=True)
