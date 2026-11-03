"""
Overview page
=============
Project introduction, data pipeline diagram, system status,
and quick-action entry points.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from design import C, inject_css, page_header, divider, info_card, section_label, status_badge
from predictx.model.predictor import get_predictor
from predictx.storage import store
from predictx.ollama_client.explainer import is_ollama_available

inject_css()

# ── Status checks ───────────────────────────────────────────────────────────
predictor      = get_predictor()
ollama_ok      = is_ollama_available()
model_ok       = predictor.is_ready
iot_configured = bool(os.getenv("IOT_ENDPOINT", ""))
storage_mode   = store.STORAGE_MODE.upper()
is_demo        = not iot_configured

# ── Page header ─────────────────────────────────────────────────────────────
page_header(
    "PredictX Overview",
    "AWS-Based Sensor Data Ingestion and Predictive Maintenance Platform",
    "🏠",
)

# ── Mode banner ─────────────────────────────────────────────────────────────
if is_demo:
    info_card(
        "Running in <strong>Demo / Local Mode</strong> — "
        "AWS IoT Core is not configured. All processing runs locally. "
        "See <code>docs/AWS_SETUP.md</code> to enable cloud connectivity.",
        kind="warning",
    )
else:
    info_card(
        "AWS IoT Core is configured. Real-time MQTT ingestion is available.",
        kind="success",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Quick actions ────────────────────────────────────────────────────────────
section_label("Quick Actions")
qa1, qa2, qa3 = st.columns(3)

with qa1:
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};border-radius:10px;
            padding:20px;text-align:center;height:140px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:10px">
  <div style="font-size:1.6rem">📂</div>
  <div style="font-weight:600;color:{C['text']}">Upload Sensor Data</div>
  <div style="font-size:0.78rem;color:{C['text_muted']}">Validate and analyse a CSV file</div>
</div>
""", unsafe_allow_html=True)
    if st.button("Go to Data Ingestion", key="qa_ingest", use_container_width=True):
        st.switch_page("pages/ingestion.py")

with qa2:
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};border-radius:10px;
            padding:20px;text-align:center;height:140px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:10px">
  <div style="font-size:1.6rem">📡</div>
  <div style="font-weight:600;color:{C['text']}">Live Monitoring</div>
  <div style="font-size:0.78rem;color:{C['text_muted']}">Stream readings from the simulator</div>
</div>
""", unsafe_allow_html=True)
    if st.button("Go to Live Monitoring", key="qa_monitor", use_container_width=True):
        st.switch_page("pages/monitoring.py")

with qa3:
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};border-radius:10px;
            padding:20px;text-align:center;height:140px;display:flex;flex-direction:column;
            align-items:center;justify-content:center;gap:10px">
  <div style="font-size:1.6rem">🔍</div>
  <div style="font-weight:600;color:{C['text']}">View Anomalies</div>
  <div style="font-size:0.78rem;color:{C['text_muted']}">Inspect flagged sensor events</div>
</div>
""", unsafe_allow_html=True)
    if st.button("Go to Anomaly Detection", key="qa_anomaly", use_container_width=True):
        st.switch_page("pages/anomalies.py")

divider()

# ── Data pipeline ────────────────────────────────────────────────────────────
section_label("Data Pipeline")

stages = [
    ("Data Source",       "CSV upload or sensor simulator",            True),
    ("Data Ingestion",    "Schema validation, type coercion, cleaning", True),
    ("AWS IoT Core",      "MQTT publish to predictx/sensors/+",        iot_configured),
    ("IoT Rules Engine",  "SQL rule routes messages to Lambda",         iot_configured),
    ("Lambda Processing", "Serverless inference on each message",       iot_configured),
    ("ML Detection",      "Isolation Forest anomaly scoring",           model_ok),
    ("AI Explanation",    "Ollama LLM maintenance suggestions",         ollama_ok),
    ("Storage & UI",      "Results in S3 / local store + dashboard",   True),
]

# Build pipeline HTML
pipeline_html = f"""
<div style="overflow-x:auto;padding:4px 0 12px">
<div style="display:flex;align-items:stretch;gap:0;min-width:700px">
"""

for i, (name, desc, active) in enumerate(stages):
    is_last = i == len(stages) - 1
    border_col = C["accent"] if active else C["border"]
    text_col   = C["text"]   if active else C["text_muted"]
    badge_col  = C["accent"] if active else C["text_dim"]
    badge_bg   = "rgba(0,212,170,0.10)" if active else C["surface2"]
    num_col    = C["accent"] if active else C["text_dim"]

    pipeline_html += f"""
  <div style="flex:1;min-width:80px">
    <div style="background:{C['surface']};border:1px solid {border_col};border-radius:8px;
                padding:14px 12px;height:100%;box-sizing:border-box">
      <div style="color:{num_col};font-size:0.65rem;font-weight:700;
                  letter-spacing:0.1em;margin-bottom:6px">STAGE {i+1}</div>
      <div style="color:{text_col};font-weight:600;font-size:0.82rem;
                  margin-bottom:4px;line-height:1.3">{name}</div>
      <div style="color:{C['text_muted']};font-size:0.7rem;line-height:1.4">{desc}</div>
      <div style="margin-top:8px">
        <span style="background:{badge_bg};color:{badge_col};font-size:0.6rem;
                     font-weight:600;padding:2px 7px;border-radius:10px;
                     letter-spacing:0.05em">
          {"ACTIVE" if active else "DEMO"}
        </span>
      </div>
    </div>
  </div>
"""
    if not is_last:
        arrow_col = C["accent"] if active else C["border"]
        pipeline_html += f"""
  <div style="display:flex;align-items:center;padding:0 2px;flex-shrink:0">
    <div style="color:{arrow_col};font-size:1rem;font-weight:300">›</div>
  </div>
"""

pipeline_html += "</div></div>"
st.markdown(pipeline_html, unsafe_allow_html=True)

divider()

# ── System status cards ──────────────────────────────────────────────────────
section_label("System Status")
sc1, sc2, sc3, sc4 = st.columns(4)

def _status_card(col, label, value, sub, ok, warn=False):
    dot_col = C["normal"] if ok else (C["warning"] if warn else C["anomaly"])
    col.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};border-radius:8px;
            padding:16px 18px">
  <div style="color:{C['text_muted']};font-size:0.68rem;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:8px">{label}</div>
  <div style="display:flex;align-items:center;gap:8px">
    <div style="width:8px;height:8px;border-radius:50%;background:{dot_col};
                flex-shrink:0"></div>
    <div style="color:{C['text']};font-weight:600;font-size:0.95rem">{value}</div>
  </div>
  <div style="color:{C['text_muted']};font-size:0.73rem;margin-top:4px">{sub}</div>
</div>
""", unsafe_allow_html=True)

_status_card(sc1, "ML Model",    "Ready" if model_ok else "Not trained",
             "Isolation Forest" if model_ok else "Run train_model.py", model_ok)
_status_card(sc2, "Ollama LLM",  "Online" if ollama_ok else "Offline",
             "llama3.2:3b" if ollama_ok else "Fallback explanations active",
             ollama_ok, warn=True)
_status_card(sc3, "Storage",     storage_mode,
             "local_store/" if storage_mode == "LOCAL" else store.S3_BUCKET, True)
_status_card(sc4, "AWS IoT",     "Configured" if iot_configured else "Demo Mode",
             "MQTT active" if iot_configured else "See docs/AWS_SETUP.md",
             iot_configured, warn=True)

divider()

# ── Recent results summary ───────────────────────────────────────────────────
section_label("Recent Activity")

saved_keys = store.list_results()

if not saved_keys:
    info_card(
        "No saved results yet. Upload a CSV file or run the simulator to get started.",
        kind="info",
    )
else:
    recent = saved_keys[-5:][::-1]  # last 5, newest first
    total_anomalies = 0
    total_readings  = 0

    rows_html = ""
    for key in recent:
        try:
            df = store.load_result(key)
            n  = len(df)
            na = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
            total_readings  += n
            total_anomalies += na
            rate = f"{na/n*100:.1f}%" if n > 0 else "—"
            badge = status_badge("Anomalies" if na > 0 else "Normal", "anomaly" if na > 0 else "normal")
            short_key = key.replace("results/", "").replace(".csv", "")
            rows_html += f"""
<tr>
  <td style="padding:8px 12px;color:{C['text']};font-size:0.82rem">{short_key}</td>
  <td style="padding:8px 12px;color:{C['text_muted']};font-size:0.82rem;text-align:right">{n:,}</td>
  <td style="padding:8px 12px;color:{C['anomaly'] if na > 0 else C['normal']};
             font-size:0.82rem;text-align:right">{na}</td>
  <td style="padding:8px 12px;text-align:right">{badge}</td>
</tr>
"""
        except Exception:
            pass

    if rows_html:
        ra1, ra2 = st.columns([3, 1])
        with ra1:
            st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};border-radius:8px;overflow:hidden">
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="background:{C['surface2']}">
        <th style="padding:8px 12px;color:{C['text_muted']};font-size:0.68rem;
                   text-transform:uppercase;letter-spacing:0.08em;text-align:left">Session</th>
        <th style="padding:8px 12px;color:{C['text_muted']};font-size:0.68rem;
                   text-transform:uppercase;letter-spacing:0.08em;text-align:right">Readings</th>
        <th style="padding:8px 12px;color:{C['text_muted']};font-size:0.68rem;
                   text-transform:uppercase;letter-spacing:0.08em;text-align:right">Anomalies</th>
        <th style="padding:8px 12px;color:{C['text_muted']};font-size:0.68rem;
                   text-transform:uppercase;letter-spacing:0.08em;text-align:right">Status</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

        with ra2:
            anom_rate = f"{total_anomalies/total_readings*100:.1f}%" if total_readings > 0 else "—"
            st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};border-radius:8px;
            padding:20px;text-align:center">
  <div style="color:{C['text_muted']};font-size:0.68rem;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:8px">Overall Anomaly Rate</div>
  <div style="color:{C['anomaly'] if total_anomalies > 0 else C['normal']};
              font-size:2.2rem;font-weight:700">{anom_rate}</div>
  <div style="color:{C['text_muted']};font-size:0.75rem;margin-top:6px">
    {total_anomalies} of {total_readings:,} readings
  </div>
</div>
""", unsafe_allow_html=True)

# ── About strip ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};border-radius:8px;
            padding:16px 20px;display:flex;gap:32px;flex-wrap:wrap;align-items:center">
  <div>
    <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
                letter-spacing:0.08em">Dataset</div>
    <div style="color:{C['text']};font-size:0.82rem;margin-top:2px">AI4I 2020 · 10,000 readings</div>
  </div>
  <div>
    <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
                letter-spacing:0.08em">Model</div>
    <div style="color:{C['text']};font-size:0.82rem;margin-top:2px">Isolation Forest · Scikit-learn</div>
  </div>
  <div>
    <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
                letter-spacing:0.08em">LLM</div>
    <div style="color:{C['text']};font-size:0.82rem;margin-top:2px">Ollama · llama3.2:3b</div>
  </div>
  <div>
    <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
                letter-spacing:0.08em">Transport</div>
    <div style="color:{C['text']};font-size:0.82rem;margin-top:2px">AWS IoT Core · MQTT TLS</div>
  </div>
  <div>
    <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
                letter-spacing:0.08em">Storage</div>
    <div style="color:{C['text']};font-size:0.82rem;margin-top:2px">Amazon S3 · Local fallback</div>
  </div>
</div>
""", unsafe_allow_html=True)
