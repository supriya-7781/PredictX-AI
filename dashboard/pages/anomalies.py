"""
Anomaly Detection page
=======================
Detailed view of flagged anomalies from the most recent analysis session,
plus loading from saved historical results.
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

# New Feature 1 — XAI
from predictx.xai.feature_explainer import get_explainer

load_dotenv(ROOT / ".env")

from design import (
    C, inject_css, page_header, divider, info_card,
    section_label, apply_plotly_theme, status_badge, severity_from_score,
)
from predictx.storage import store

inject_css()

page_header(
    "Anomaly Detection",
    "Inspect flagged sensor readings and anomaly scores",
    "🔍",
)

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# ── Data source ───────────────────────────────────────────────────────────────
result_df: pd.DataFrame | None = st.session_state.get("ingest_result")

src_tab1, src_tab2 = st.tabs(["Current Session", "Load from History"])

with src_tab1:
    if result_df is None:
        info_card(
            "No data in the current session. "
            "Go to <strong>Data Ingestion</strong> or <strong>Live Monitoring</strong> "
            "to analyse sensor data first.",
            kind="info",
        )
    else:
        n = len(result_df)
        na = int(result_df["is_anomaly"].sum())
        st.markdown(f"""
<div style="color:{C['text_muted']};font-size:0.82rem;padding:8px 0">
  Session data: <strong style="color:{C['text']}">{n:,} readings</strong> ·
  <strong style="color:{C['anomaly'] if na > 0 else C['normal']}">{na} anomalies</strong>
</div>
""", unsafe_allow_html=True)

with src_tab2:
    saved_keys = store.list_results()
    if not saved_keys:
        info_card("No saved results found. Save a session from Data Ingestion first.", kind="info")
    else:
        sel_key = st.selectbox("Select a saved result", saved_keys, label_visibility="collapsed")
        if st.button("Load selected result", type="secondary"):
            with st.spinner("Loading..."):
                result_df = store.load_result(sel_key)
                st.session_state["ingest_result"] = result_df
            st.rerun()

if result_df is None:
    st.stop()

# Ensure required columns exist
if "is_anomaly" not in result_df.columns or "anomaly_score" not in result_df.columns:
    info_card("The loaded data does not contain inference results. Re-run anomaly detection.", kind="error")
    st.stop()

divider()

# ── Summary metrics ───────────────────────────────────────────────────────────
n_total   = len(result_df)
n_anomaly = int(result_df["is_anomaly"].sum())
n_normal  = n_total - n_anomaly
anom_rate = n_anomaly / n_total * 100 if n_total > 0 else 0
min_score = result_df["anomaly_score"].min()
avg_score = result_df["anomaly_score"].mean()

section_label("Detection Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total readings",  f"{n_total:,}")
m2.metric("Normal",          f"{n_normal:,}")
m3.metric("Anomalies",       f"{n_anomaly}",
          delta=f"{anom_rate:.1f}% of total", delta_color="inverse")
m4.metric("Worst score",     f"{min_score:.4f}")

divider()

# ── Charts row ────────────────────────────────────────────────────────────────
section_label("Score Analysis")
c_left, c_right = st.columns(2)

with c_left:
    status_col = result_df["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
    fig_hist = px.histogram(
        result_df,
        x="anomaly_score",
        color=status_col,
        nbins=40,
        color_discrete_map={"Normal": C["blue"], "Anomaly": C["anomaly"]},
        labels={"anomaly_score": "Anomaly Score", "color": "Status"},
        title="Score Distribution",
        opacity=0.85,
    )
    fig_hist.add_vline(x=0, line_dash="dot", line_color=C["warning"],
                       annotation_text="Decision boundary",
                       annotation_font_color=C["warning"], annotation_font_size=10)
    apply_plotly_theme(fig_hist, height=300)
    st.plotly_chart(fig_hist, use_container_width=True)

with c_right:
    fig_scatter = px.scatter(
        result_df,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        color=status_col,
        color_discrete_map={"Normal": C["blue"], "Anomaly": C["anomaly"]},
        hover_data=["Tool wear [min]", "anomaly_score"],
        title="RPM vs Torque",
        opacity=0.7,
    )
    apply_plotly_theme(fig_scatter, height=300)
    st.plotly_chart(fig_scatter, use_container_width=True)

divider()

# ── Anomaly list ──────────────────────────────────────────────────────────────
anomaly_df = result_df[result_df["is_anomaly"] == 1].copy()
anomaly_df = anomaly_df.sort_values("anomaly_score").reset_index(drop=True)

section_label(f"Flagged Events — {len(anomaly_df)} anomalies")

if anomaly_df.empty:
    info_card(
        "No anomalies detected in this dataset. All readings are within normal parameters.",
        kind="success",
    )
    st.stop()

# Severity filter
sev_filter = st.radio(
    "Filter by severity",
    ["All", "High (score < −0.10)", "Medium (−0.10 to −0.05)", "Low (> −0.05)"],
    horizontal=True,
    label_visibility="collapsed",
)

def _sev_label(score: float) -> str:
    if score < -0.10: return "High"
    if score < -0.05: return "Medium"
    return "Low"

anomaly_df["Severity"] = anomaly_df["anomaly_score"].apply(_sev_label)

if sev_filter.startswith("High"):
    anomaly_df = anomaly_df[anomaly_df["Severity"] == "High"]
elif sev_filter.startswith("Medium"):
    anomaly_df = anomaly_df[anomaly_df["Severity"] == "Medium"]
elif sev_filter.startswith("Low"):
    anomaly_df = anomaly_df[anomaly_df["Severity"] == "Low"]

st.markdown(f"<div style='color:{C['text_muted']};font-size:0.8rem;margin:8px 0'>"
            f"Showing {len(anomaly_df)} event(s)</div>", unsafe_allow_html=True)

# Anomaly cards
for rank, (_, row) in enumerate(anomaly_df.iterrows(), 1):
    score    = float(row["anomaly_score"])
    sev_lbl, sev_kind = severity_from_score(score)
    sev_col  = {"HIGH": C["anomaly"], "MEDIUM": C["warning"], "LOW": C["blue"]}[sev_lbl]
    badge_html = status_badge(sev_lbl, sev_kind)

    product_id  = row.get("Product ID", "—")
    machine_type = row.get("Type", "—")
    header_label = f"Event #{rank}"
    if product_id != "—":
        header_label += f"  ·  {product_id}"

    with st.expander(header_label, expanded=(rank <= 3)):
        # Severity + score header
        st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:14px">
  <div style="display:flex;align-items:center;gap:10px">
    {badge_html}
    <span style="color:{C['text_muted']};font-size:0.8rem">
      Score: <strong style="color:{sev_col}">{score:.5f}</strong>
    </span>
  </div>
  <div style="color:{C['text_muted']};font-size:0.75rem">
    Rank {rank} / {len(anomaly_df)} most severe
  </div>
</div>
""", unsafe_allow_html=True)

        # Sensor readings grid
        sv1, sv2, sv3, sv4, sv5 = st.columns(5)
        def _sensor_cell(col_el, label, value, unit, hi_thresh=None):
            hi = hi_thresh is not None and value > hi_thresh
            val_col = C["anomaly"] if hi else C["text"]
            col_el.markdown(f"""
<div style="background:{C['surface2']};border:1px solid {C['border']};border-radius:6px;
            padding:12px;text-align:center">
  <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
              letter-spacing:0.07em;margin-bottom:4px">{label}</div>
  <div style="color:{val_col};font-size:1.15rem;font-weight:700">{value:.1f}</div>
  <div style="color:{C['text_dim']};font-size:0.7rem">{unit}</div>
</div>
""", unsafe_allow_html=True)

        _sensor_cell(sv1, "Air Temp",  row["Air temperature [K]"],      "K",   303.0)
        _sensor_cell(sv2, "Proc Temp", row["Process temperature [K]"],   "K",   313.0)
        _sensor_cell(sv3, "RPM",       row["Rotational speed [rpm]"],    "rpm", 2500.0)
        _sensor_cell(sv4, "Torque",    row["Torque [Nm]"],               "Nm")
        _sensor_cell(sv5, "Tool Wear", row["Tool wear [min]"],           "min", 240.0)

        st.markdown("<br>", unsafe_allow_html=True)

        # Context bar
        feat_cols = [c for c in FEATURE_COLS if c in result_df.columns]
        if len(feat_cols) >= 2:
            # Mini bar chart of this row's values vs dataset mean
            means = result_df[feat_cols].mean()
            vals  = [row[c] for c in feat_cols]
            short_names = ["Air T", "Proc T", "RPM", "Torque", "Wear"]
            fig_bar = go.Figure(go.Bar(
                x=short_names[:len(feat_cols)],
                y=vals,
                marker_color=[
                    C["anomaly"] if v > m * 1.15 or v < m * 0.85 else C["blue"]
                    for v, m in zip(vals, means)
                ],
                name="This reading",
            ))
            fig_bar.add_trace(go.Scatter(
                x=short_names[:len(feat_cols)],
                y=means.tolist(),
                mode="markers",
                marker=dict(color=C["accent"], size=8, symbol="diamond"),
                name="Dataset mean",
            ))
            fig_bar.update_layout(title="Sensor Values vs Dataset Mean", showlegend=True)
            apply_plotly_theme(fig_bar, height=220)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── XAI Feature Deviation Panel (Feature 1) ──────────────────────
        try:
            explainer    = get_explainer()
            xai_result   = explainer.explain(row.to_dict(), anomaly_score=score)
            short_names_z = [c.short_name for c in xai_result.contributions]
            z_vals        = [c.z_score    for c in xai_result.contributions]
            z_colours     = [
                C["anomaly"] if abs(z) >= 3.0
                else C["warning"] if abs(z) >= 2.0
                else C["blue"]
                for z in z_vals
            ]
            fig_z = go.Figure(go.Bar(
                x=short_names_z, y=z_vals,
                marker_color=z_colours,
                text=[f"{z:+.2f}σ" for z in z_vals],
                textposition="outside",
            ))
            fig_z.add_hline(y=2.0,  line_dash="dot", line_color=C["warning"],
                            annotation_text="+2σ", annotation_font_size=9,
                            annotation_font_color=C["warning"])
            fig_z.add_hline(y=-2.0, line_dash="dot", line_color=C["warning"])
            fig_z.update_layout(title="Feature Deviation from Normal Baseline (Z-scores)",
                                yaxis_title="Std deviations (σ)")
            apply_plotly_theme(fig_z, height=220)
            st.plotly_chart(fig_z, use_container_width=True)

            st.markdown(f"""
<div style="background:{C['surface2']};border-left:3px solid {C['accent']};
            padding:10px 14px;border-radius:0 6px 6px 0;margin-bottom:8px">
  <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
              letter-spacing:0.07em;margin-bottom:4px">Why was this flagged?</div>
  <div style="color:{C['text']};font-size:0.8rem;line-height:1.6">
    {xai_result.narrative}
  </div>
</div>
""", unsafe_allow_html=True)
        except Exception:
            pass  # XAI panel is non-critical — page still works if it fails
