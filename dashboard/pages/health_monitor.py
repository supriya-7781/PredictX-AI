"""
Machine Health Monitor page
============================
Displays:
  1. Fleet health summary (Healthy / Warning / Poor / Critical counts)
  2. Health score timeline — Plotly line chart across the session
  3. Health band breakdown — stacked bar
  4. Feature deviation heatmap — Z-scores for every reading
  5. XAI detail panel — select any reading, see per-feature explanation
  6. Structured maintenance recommendations for the selected reading

Reads from session_state["ingest_result"] — the same DataFrame shared with
all other pages.  Calls the three new backend modules.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from io import StringIO

load_dotenv(ROOT / ".env")

from design import (
    C, inject_css, page_header, divider, info_card,
    section_label, apply_plotly_theme, status_badge, severity_from_score,
)
from predictx.health.health_scorer import get_health_scorer, HEALTH_BANDS, classify_band
from predictx.xai.feature_explainer import get_explainer, FEATURE_COLS
from predictx.recommendations.recommendation_engine import get_recommendation_engine

inject_css()

page_header(
    "Machine Health Monitor",
    "Dynamic health scores, feature deviation analysis, and maintenance recommendations",
    "🩺",
)

# ── Data check ────────────────────────────────────────────────────────────────
result_df: pd.DataFrame | None = st.session_state.get("ingest_result")

if result_df is None or not all(c in result_df.columns for c in FEATURE_COLS):
    info_card(
        "No sensor data available. Go to <strong>Data Ingestion</strong> or "
        "<strong>Live Monitoring</strong> to load readings first.",
        kind="info",
    )
    st.markdown(f"""
<div style="text-align:center;padding:60px 20px;color:{C['text_muted']}">
  <div style="font-size:2.5rem;margin-bottom:16px;opacity:0.4">🩺</div>
  <div style="font-size:1rem;font-weight:500;color:{C['text']};margin-bottom:8px">
    No data loaded
  </div>
  <div style="font-size:0.85rem">Upload a CSV or run the simulator, then return here.</div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Compute health scores (cache per session DataFrame identity) ──────────────
@st.cache_data(show_spinner=False)
def _compute_scores(df_json: str) -> pd.DataFrame:
    """Cached wrapper — recomputes only when the DataFrame changes."""
    df = pd.read_json(StringIO(df_json), orient="split")
    scorer = get_health_scorer()
    return scorer.score_dataframe(df)


# Use a stable hash of the data so cache invalidates when new data arrives
_df_hash = str(hash(result_df.to_json(orient="split", date_format="iso")))
scored_df = _compute_scores(result_df.to_json(orient="split", date_format="iso"))

# ── Fleet summary ─────────────────────────────────────────────────────────────
section_label("Fleet Health Summary")

band_counts = scored_df["health_band"].value_counts().to_dict()
total        = len(scored_df)
n_healthy    = band_counts.get("Healthy",  0)
n_warning    = band_counts.get("Warning",  0)
n_poor       = band_counts.get("Poor",     0)
n_critical   = band_counts.get("Critical", 0)
avg_score    = int(scored_df["health_score"].mean())
min_score    = int(scored_df["health_score"].min())

band_col, _, min_col = st.columns([3, 0.2, 1])

with band_col:
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)

    def _fleet_card(col, label, value, colour):
        col.markdown(f"""
<div style="background:{C['surface']};border:1px solid {colour}44;
            border-radius:8px;padding:16px;text-align:center">
  <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:6px">{label}</div>
  <div style="color:{colour};font-size:1.8rem;font-weight:700">{value}</div>
</div>
""", unsafe_allow_html=True)

    _fleet_card(fc1, "Healthy",  n_healthy,  C["normal"])
    _fleet_card(fc2, "Warning",  n_warning,  C["warning"])
    _fleet_card(fc3, "Poor",     n_poor,     C["anomaly"])
    _fleet_card(fc4, "Critical", n_critical, C["anomaly"])
    _fleet_card(fc5, "Avg Score", avg_score, C["accent"])

with min_col:
    lowest_band, lowest_kind = classify_band(min_score)
    band_colour = {"Healthy": C["normal"], "Warning": C["warning"],
                   "Poor": C["anomaly"], "Critical": C["anomaly"]}.get(lowest_band, C["text_muted"])
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {band_colour}66;
            border-radius:8px;padding:16px;text-align:center;height:100%">
  <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:6px">Lowest Reading</div>
  <div style="color:{band_colour};font-size:2.2rem;font-weight:700">{min_score}</div>
  <div style="margin-top:6px">{status_badge(lowest_band, lowest_kind)}</div>
</div>
""", unsafe_allow_html=True)

divider()

# ── Health score timeline ─────────────────────────────────────────────────────
section_label("Health Score Timeline")

fig_timeline = go.Figure()

# Shade bands
band_colours = {
    "Healthy":  ("rgba(34,197,94,0.06)",  90),
    "Warning":  ("rgba(245,158,11,0.06)", 70),
    "Poor":     ("rgba(239,68,68,0.06)",  40),
    "Critical": ("rgba(239,68,68,0.10)",   0),
}
band_thresholds = [(90, 100, "rgba(34,197,94,0.06)"),
                   (70, 90,  "rgba(245,158,11,0.06)"),
                   (40, 70,  "rgba(239,68,68,0.06)"),
                   (0,  40,  "rgba(239,68,68,0.10)")]
for lo, hi, colour in band_thresholds:
    fig_timeline.add_hrect(y0=lo, y1=hi, fillcolor=colour,
                           line_width=0, layer="below")

# Colour each point by its band
point_colours = scored_df["health_score"].apply(
    lambda s: C["normal"] if s >= 90 else C["warning"] if s >= 70 else C["anomaly"]
)

fig_timeline.add_trace(go.Scatter(
    x=scored_df.index,
    y=scored_df["health_score"],
    mode="lines+markers",
    name="Health Score",
    line=dict(color=C["accent"], width=2),
    marker=dict(color=point_colours.tolist(), size=5),
    hovertemplate="Reading %{x}<br>Score: %{y}<br>",
))

# Band label annotations
for lo, label, _ in HEALTH_BANDS:
    fig_timeline.add_annotation(
        x=0, y=lo + 3, xref="paper", yref="y",
        text=label, font=dict(size=9, color=C["text_dim"]),
        showarrow=False, xanchor="left",
    )

fig_timeline.update_layout(
    title="Health Score per Reading (100 = perfect, 0 = critical)",
    yaxis=dict(range=[0, 105]),
    xaxis_title="Reading index",
    yaxis_title="Health Score",
)
apply_plotly_theme(fig_timeline, height=280)
st.plotly_chart(fig_timeline, use_container_width=True)

divider()

# ── Feature deviation heatmap ─────────────────────────────────────────────────
section_label("Feature Deviation Heatmap  (Z-scores)")
st.markdown(f"""
<div style="color:{C['text_muted']};font-size:0.8rem;margin-bottom:8px">
  Each cell shows how many standard deviations (σ) a sensor reading lies
  from the model's learned training baseline.  Values beyond ±2σ are
  statistically noteworthy.
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def _compute_zscores(df_json: str) -> pd.DataFrame:
    df   = pd.read_json(StringIO(df_json), orient="split")
    exp  = get_explainer()
    rows = []
    for _, row in df.iterrows():
        try:
            expl = exp.explain(row.to_dict())
            z_row = {c.short_name: round(c.z_score, 3) for c in expl.contributions}
            rows.append(z_row)
        except Exception:
            rows.append({})
    return pd.DataFrame(rows)

zscore_df = _compute_zscores(result_df.to_json(orient="split", date_format="iso"))

if not zscore_df.empty:
    # Show at most 200 rows in heatmap for performance
    display_z = zscore_df.head(200)
    fig_heat = px.imshow(
        display_z.T,
        color_continuous_scale=[
            [0.0,  "#1a3a6b"],   # deep below normal
            [0.25, "#1e3a5f"],
            [0.4,  "#1a1f2e"],   # near-zero (normal)
            [0.6,  "#1a1f2e"],
            [0.75, "#7f2020"],
            [1.0,  "#ef4444"],   # deep above normal
        ],
        zmin=-4, zmax=4,
        aspect="auto",
        title=f"Z-score heatmap ({min(200, len(zscore_df))} readings × 5 features)",
        labels=dict(x="Reading index", y="Sensor", color="Z-score (σ)"),
    )
    fig_heat.update_layout(coloraxis_colorbar=dict(title="σ"))
    apply_plotly_theme(fig_heat, height=220)
    st.plotly_chart(fig_heat, use_container_width=True)

divider()

# ── Per-reading XAI + Recommendation detail ───────────────────────────────────
section_label("Reading Detail — XAI & Recommendations")

# Build selector: show all readings, highlight anomalies
sel_options = list(scored_df.index)
anom_col = "is_anomaly" in result_df.columns

def _reading_label(idx):
    score = int(scored_df.loc[idx, "health_score"])
    band  = scored_df.loc[idx, "health_band"]
    flag  = ""
    if anom_col and result_df.loc[idx, "is_anomaly"] == 1:
        flag = "  ⚠ Anomaly"
    return f"Reading {idx}  |  Health: {score}/100  |  {band}{flag}"

selected_reading = st.selectbox(
    "Select a reading to analyse",
    sel_options,
    format_func=_reading_label,
    label_visibility="collapsed",
)

row_data    = result_df.loc[selected_reading].to_dict()
anom_score  = float(row_data.get("anomaly_score", 0.0))
health_res  = get_health_scorer().score_row(row_data)
expl_result = get_explainer().explain(row_data, anom_score)
rec_set     = get_recommendation_engine().recommend(
    expl_result,
    machine_id=str(row_data.get("Product ID", f"R-{selected_reading}")),
)

# ── Reading header row ────────────────────────────────────────────────────────
hdr1, hdr2, hdr3 = st.columns(3)

band_c = {"Healthy": C["normal"], "Warning": C["warning"],
          "Poor": C["anomaly"], "Critical": C["anomaly"]}.get(health_res.band, C["text_muted"])

with hdr1:
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {band_c}55;
            border-radius:8px;padding:20px;text-align:center">
  <div style="color:{C['text_muted']};font-size:0.68rem;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:8px">Health Score</div>
  <div style="color:{band_c};font-size:2.8rem;font-weight:700;line-height:1">
    {health_res.score}
  </div>
  <div style="margin-top:8px">{status_badge(health_res.band, health_res.band_kind)}</div>
</div>
""", unsafe_allow_html=True)

with hdr2:
    sev_lbl, sev_kind = severity_from_score(anom_score)
    sev_c = {"HIGH": C["anomaly"], "MEDIUM": C["warning"], "LOW": C["blue"]}.get(sev_lbl, C["text_muted"])
    is_anom = anom_col and result_df.loc[selected_reading, "is_anomaly"] == 1
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};
            border-radius:8px;padding:20px;text-align:center">
  <div style="color:{C['text_muted']};font-size:0.68rem;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:8px">Anomaly Status</div>
  <div style="color:{C['anomaly'] if is_anom else C['normal']};font-size:1.1rem;font-weight:600">
    {"⚠ Anomaly" if is_anom else "✓ Normal"}
  </div>
  <div style="color:{sev_c};font-size:0.85rem;margin-top:4px">
    Score: {anom_score:.5f}
  </div>
</div>
""", unsafe_allow_html=True)

with hdr3:
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};
            border-radius:8px;padding:20px;text-align:center">
  <div style="color:{C['text_muted']};font-size:0.68rem;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:8px">Primary Deviation</div>
  <div style="color:{C['text']};font-size:1rem;font-weight:600">
    {expl_result.primary_driver}
  </div>
  <div style="color:{C['text_muted']};font-size:0.78rem;margin-top:4px">
    {expl_result.n_significant} significant feature(s)
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── XAI panel ────────────────────────────────────────────────────────────────
xai_col, rec_col = st.columns([1, 1], gap="large")

with xai_col:
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};
            border-radius:8px;padding:20px">
  <div style="font-weight:600;color:{C['text']};margin-bottom:14px;
              font-size:0.9rem">Feature Deviation Analysis</div>
""", unsafe_allow_html=True)

    # Z-score bar chart
    short_names = [c.short_name for c in expl_result.contributions]
    z_values    = [c.z_score    for c in expl_result.contributions]
    bar_colours = [
        C["anomaly"] if abs(z) >= 3.0
        else C["warning"] if abs(z) >= 2.0
        else C["blue"]
        for z in z_values
    ]
    fig_z = go.Figure(go.Bar(
        x=short_names,
        y=z_values,
        marker_color=bar_colours,
        text=[f"{z:+.2f}σ" for z in z_values],
        textposition="outside",
    ))
    fig_z.add_hline(y=2.0,  line_dash="dot", line_color=C["warning"],
                    annotation_text="+2σ threshold", annotation_font_size=9,
                    annotation_font_color=C["warning"])
    fig_z.add_hline(y=-2.0, line_dash="dot", line_color=C["warning"])
    fig_z.update_layout(title="Z-scores (deviation from training baseline)",
                        yaxis_title="Standard deviations (σ)")
    apply_plotly_theme(fig_z, height=260)
    st.plotly_chart(fig_z, use_container_width=True)

    # Narrative
    st.markdown(f"""
<div style="background:{C['surface2']};border-left:3px solid {C['accent']};
            padding:12px 16px;border-radius:0 6px 6px 0;margin-top:8px">
  <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:6px">Explanation</div>
  <div style="color:{C['text']};font-size:0.82rem;line-height:1.6">
    {expl_result.narrative}
  </div>
</div>
""", unsafe_allow_html=True)

    # Baseline table
    with st.expander("Normal operating ranges (training baseline)", expanded=False):
        baseline = get_explainer().baseline
        rows_html = ""
        for col in FEATURE_COLS:
            b = baseline[col]
            contrib = next(c for c in expl_result.contributions if c.feature == col)
            z_colour = (C["anomaly"] if contrib.abs_z >= 3.0
                        else C["warning"] if contrib.abs_z >= 2.0
                        else C["normal"])
            rows_html += f"""
<tr style="border-bottom:1px solid {C['border']}">
  <td style="padding:7px 10px;color:{C['text']};font-size:0.78rem">{contrib.short_name}</td>
  <td style="padding:7px 10px;color:{C['text_muted']};font-size:0.78rem;text-align:right">
    {contrib.current_value:.1f} {contrib.unit}</td>
  <td style="padding:7px 10px;color:{C['text_muted']};font-size:0.78rem;text-align:right">
    {b['normal_low']:.1f} – {b['normal_high']:.1f}</td>
  <td style="padding:7px 10px;color:{z_colour};font-size:0.78rem;
             text-align:right;font-family:monospace">
    {contrib.z_score:+.2f}σ</td>
</tr>
"""
        st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};
            border-radius:6px;overflow:hidden">
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="background:{C['surface2']}">
        <th style="padding:7px 10px;color:{C['text_muted']};font-size:0.65rem;
                   text-transform:uppercase;letter-spacing:0.07em;text-align:left">Sensor</th>
        <th style="padding:7px 10px;color:{C['text_muted']};font-size:0.65rem;
                   text-transform:uppercase;letter-spacing:0.07em;text-align:right">Current</th>
        <th style="padding:7px 10px;color:{C['text_muted']};font-size:0.65rem;
                   text-transform:uppercase;letter-spacing:0.07em;text-align:right">Normal range (±2σ)</th>
        <th style="padding:7px 10px;color:{C['text_muted']};font-size:0.65rem;
                   text-transform:uppercase;letter-spacing:0.07em;text-align:right">Z-score</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close XAI panel div

with rec_col:
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};
            border-radius:8px;padding:20px">
  <div style="font-weight:600;color:{C['text']};margin-bottom:4px;font-size:0.9rem">
    Maintenance Recommendations
  </div>
  <div style="color:{C['text_muted']};font-size:0.75rem;margin-bottom:14px">
    Based on sensor deviations — not confirmed fault diagnoses
  </div>
""", unsafe_allow_html=True)

    if not rec_set.has_actionable:
        st.markdown(f"""
<div style="background:rgba(34,197,94,0.06);border:1px solid {C['normal']}44;
            border-radius:6px;padding:14px;color:{C['text']};font-size:0.82rem;
            margin-bottom:12px">
  No urgent maintenance action indicated. Monitor next readings before intervening.
</div>
""", unsafe_allow_html=True)
    else:
        for r in rec_set.by_priority():
            badge     = status_badge(r.confidence, r.confidence_kind)
            pri_label = "Priority" if r.priority <= 1 else ""
            pri_html  = (f'<span style="color:{C["anomaly"]};font-size:0.65rem;'
                         f'font-weight:700;text-transform:uppercase;margin-left:8px">'
                         f'HIGH PRIORITY</span>' if pri_label else "")

            st.markdown(f"""
<div style="background:{C['surface2']};border:1px solid {C['border']};
            border-radius:6px;padding:14px;margin-bottom:10px">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
    {badge}{pri_html}
    <span style="color:{C['text_muted']};font-size:0.72rem">
      Trigger: {r.sensor_trigger}
    </span>
  </div>
  <div style="color:{C['text']};font-size:0.82rem;font-weight:500;
              margin-bottom:6px">{r.action}</div>
  <div style="color:{C['text_muted']};font-size:0.76rem;line-height:1.5;
              border-left:2px solid {C['border']};padding-left:8px">
    {r.evidence}
  </div>
</div>
""", unsafe_allow_html=True)

    # Disclaimer
    st.markdown(f"""
<div style="color:{C['text_dim']};font-size:0.7rem;margin-top:12px;
            border-top:1px solid {C['border']};padding-top:10px;line-height:1.5">
  {rec_set.disclaimer}
</div>
</div>
""", unsafe_allow_html=True)
