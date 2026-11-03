"""
Live Monitoring page
=====================
Streams readings from the sensor simulator, runs inference on each,
and updates charts in real time.
"""

from __future__ import annotations

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulator"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from design import (
    C, inject_css, page_header, divider, info_card,
    section_label, apply_plotly_theme, status_badge,
)
from predictx.ingestion.validator import REQUIRED_COLS
from predictx.model.predictor import get_predictor
from predictx.storage import store

inject_css()

page_header(
    "Live Monitoring",
    "Stream sensor readings and detect anomalies in real time",
    "📡",
)

predictor = get_predictor()

if not predictor.is_ready:
    info_card(
        "ML model is not trained. Run <code>python training/train_model.py</code> first.",
        kind="error",
    )
    st.stop()

# ── Stream configuration ─────────────────────────────────────────────────────
section_label("Simulator Configuration")

cfg1, cfg2, cfg3 = st.columns(3)
with cfg1:
    sim_mode = st.selectbox(
        "Source",
        ["Mock (local CSV)", "MQTT (AWS IoT Core)"],
        help="Mock mode streams from the local dataset without AWS.",
    )
with cfg2:
    sim_rows = st.slider("Readings to stream", 20, 500, 100, 10)
with cfg3:
    sim_delay = st.slider("Interval between readings (s)", 0.0, 2.0, 0.3, 0.1)

use_mock = "Mock" in sim_mode

if not use_mock:
    import os
    if not os.getenv("IOT_ENDPOINT"):
        info_card(
            "AWS IoT Core is not configured. Switch to Mock mode or "
            "follow <code>docs/AWS_SETUP.md</code>.",
            kind="warning",
        )

divider()
section_label("Live Feed")

# ── Placeholders ─────────────────────────────────────────────────────────────
status_bar    = st.empty()
progress_bar  = st.empty()

col_chart, col_table = st.columns([3, 2], gap="medium")
with col_chart:
    chart_ph = st.empty()
with col_table:
    table_ph = st.empty()

sensor_charts_ph = st.empty()

start_col, _ = st.columns([1, 4])
with start_col:
    start_btn = st.button("Start Streaming", type="primary", use_container_width=True)

# ── Stream loop ───────────────────────────────────────────────────────────────
if start_btn:
    from sensor_simulator import SensorSimulator

    sim = SensorSimulator(
        mode="mock" if use_mock else "mqtt",
        delay=0.0,
        rows=sim_rows,
    )
    sim.mqtt_client.connect()

    live_rows: list[dict] = []
    source_label = "Demo / Local CSV" if use_mock else "MQTT · AWS IoT Core"

    status_bar.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['accent']}33;border-radius:8px;
            padding:12px 18px;display:flex;align-items:center;gap:12px">
  <div style="width:8px;height:8px;border-radius:50%;background:{C['accent']};
              animation:pulse 1.5s infinite"></div>
  <span style="color:{C['text']};font-size:0.875rem">
    Streaming from <strong>{source_label}</strong>
  </span>
</div>
<style>@keyframes pulse {{0%{{opacity:1}}50%{{opacity:0.3}}100%{{opacity:1}}}}</style>
""", unsafe_allow_html=True)

    for i, (_, row) in enumerate(sim.df.head(sim_rows).iterrows(), 1):
        payload = sim._row_to_payload(row)
        sim.mqtt_client.publish(payload)

        row_df = pd.DataFrame([payload])
        row_df = row_df[[c for c in REQUIRED_COLS if c in row_df.columns]]

        try:
            pred = predictor.predict(row_df)
            new_row = pred.iloc[0].to_dict()
            new_row["timestamp"] = payload.get("timestamp", "")
            live_rows.append(new_row)
        except Exception:
            pass

        df_live = pd.DataFrame(live_rows)
        progress_bar.progress(i / sim_rows, text=f"Reading {i} / {sim_rows}")

        is_anom = new_row.get("is_anomaly", 0)
        score   = new_row.get("anomaly_score", 0)

        # ── Live anomaly score chart ─────────────────────────────────────
        with col_chart:
            if len(df_live) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(df_live))),
                    y=df_live["anomaly_score"].tolist(),
                    mode="lines",
                    name="Score",
                    line=dict(color=C["blue"], width=2),
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.07)",
                ))
                anom_idx = df_live[df_live["is_anomaly"] == 1].index.tolist()
                if anom_idx:
                    fig.add_trace(go.Scatter(
                        x=anom_idx,
                        y=df_live.loc[anom_idx, "anomaly_score"].tolist(),
                        mode="markers",
                        name="Anomaly",
                        marker=dict(color=C["anomaly"], size=8, symbol="x",
                                    line=dict(width=2)),
                    ))
                fig.add_hline(y=0, line_dash="dot", line_color=C["warning"],
                              annotation_text="Threshold",
                              annotation_font_color=C["warning"],
                              annotation_font_size=10)
                fig.update_layout(title="Live Anomaly Score", showlegend=True)
                apply_plotly_theme(fig, height=280)
                chart_ph.plotly_chart(fig, use_container_width=True)

        # ── Recent readings table ────────────────────────────────────────
        with col_table:
            recent = df_live.tail(15).copy()
            recent["Status"] = recent["is_anomaly"].map({0: "Normal", 1: "ANOMALY"})
            recent = recent[["Rotational speed [rpm]", "Torque [Nm]",
                              "Tool wear [min]", "anomaly_score", "Status"]].copy()
            recent.columns = ["RPM", "Torque", "Wear", "Score", "Status"]
            recent["Score"] = recent["Score"].round(4)
            table_ph.dataframe(
                recent.style.apply(
                    lambda col: [
                        f"color:{C['anomaly']};font-weight:600" if v == "ANOMALY" else
                        f"color:{C['normal']}"
                        for v in col
                    ] if col.name == "Status" else [""] * len(col),
                    axis=0,
                ),
                use_container_width=True,
                height=300,
            )

        time.sleep(sim_delay)

    # ── Final summary ────────────────────────────────────────────────────
    df_final  = pd.DataFrame(live_rows)
    n_anom    = int(df_final["is_anomaly"].sum()) if "is_anomaly" in df_final.columns else 0
    n_total   = len(df_final)

    progress_bar.empty()
    status_bar.markdown(f"""
<div style="background:rgba(34,197,94,0.08);border:1px solid {C['normal']}44;
            border-radius:8px;padding:12px 18px">
  <span style="color:{C['normal']};font-weight:600">Stream complete</span>
  <span style="color:{C['text_muted']};font-size:0.875rem;margin-left:12px">
    {n_total} readings · {n_anom} anomalies detected
    ({n_anom/n_total*100:.1f}%)
  </span>
</div>
""", unsafe_allow_html=True)

    # Store for other pages
    st.session_state["ingest_result"] = df_final

    divider()
    section_label("Sensor Breakdown")

    if len(df_final) > 0:
        sb1, sb2 = st.columns(2)

        with sb1:
            fig_rpm = go.Figure()
            fig_rpm.add_trace(go.Scatter(
                y=df_final["Rotational speed [rpm]"], mode="lines",
                name="RPM", line=dict(color=C["accent"], width=1.5),
            ))
            fig_rpm.update_layout(title="Rotational Speed")
            apply_plotly_theme(fig_rpm, height=220)
            st.plotly_chart(fig_rpm, use_container_width=True)

        with sb2:
            fig_torque = go.Figure()
            fig_torque.add_trace(go.Scatter(
                y=df_final["Torque [Nm]"], mode="lines",
                name="Torque", line=dict(color=C["purple"], width=1.5),
            ))
            fig_torque.update_layout(title="Torque")
            apply_plotly_theme(fig_torque, height=220)
            st.plotly_chart(fig_torque, use_container_width=True)

    save_col2, _ = st.columns([1, 4])
    with save_col2:
        if st.button("Save stream results", type="primary", use_container_width=True):
            store.save_result(df_final)
            info_card("Results saved to storage.", kind="success")
        info_card(
            "Results are also available on the <strong>Anomaly Detection</strong> page.",
            kind="info",
        )

else:
    # Idle state
    st.markdown(f"""
<div style="text-align:center;padding:60px 20px;color:{C['text_muted']}">
  <div style="font-size:2.5rem;margin-bottom:16px;opacity:0.4">📡</div>
  <div style="font-size:1rem;font-weight:500;color:{C['text']};margin-bottom:8px">
    Simulator is idle
  </div>
  <div style="font-size:0.85rem">
    Configure the parameters above and click <strong>Start Streaming</strong>.
  </div>
</div>
""", unsafe_allow_html=True)
