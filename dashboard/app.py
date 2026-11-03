"""
app.py — PredictX Dashboard entry point
=========================================
Run:
    streamlit run dashboard/app.py

Uses Streamlit's multi-page navigation (st.navigation / st.Page).
Each page is a separate file under dashboard/pages/.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulator"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# ── Shared state & services (loaded once, cached) ──────────────────────────
from predictx.model.predictor import get_predictor
from predictx.storage import store
from predictx.ollama_client.explainer import is_ollama_available
from design import inject_css, C

st.set_page_config(
    page_title="PredictX",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


# ── Sidebar brand + status ──────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown(f"""
<div style="padding:20px 16px 12px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
    <div style="width:32px;height:32px;background:{C['accent']};border-radius:6px;
                display:flex;align-items:center;justify-content:center;
                font-size:1rem;font-weight:700;color:#0f1117">P</div>
    <span style="font-size:1.1rem;font-weight:700;color:{C['text']};
                 letter-spacing:-0.02em">PredictX</span>
  </div>
  <div style="color:{C['text_muted']};font-size:0.72rem;padding-left:42px;
              letter-spacing:0.05em;text-transform:uppercase">
    Predictive Maintenance
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:{C["border"]};margin:0 0 8px"/>', unsafe_allow_html=True)

    # ── Pages ───────────────────────────────────────────────────────────────
    pages = [
        st.Page("pages/overview.py",       title="Overview",            icon="🏠", default=True),
        st.Page("pages/ingestion.py",      title="Data Ingestion",      icon="📂"),
        st.Page("pages/monitoring.py",     title="Live Monitoring",     icon="📡"),
        st.Page("pages/health_monitor.py", title="Machine Health",      icon="🩺"),
        st.Page("pages/anomalies.py",      title="Anomaly Detection",   icon="🔍"),
        st.Page("pages/assistant.py",      title="AI Maintenance",      icon="🤖"),
        st.Page("pages/history.py",        title="History",             icon="📜"),
    ]

    nav = st.navigation(pages, position="sidebar")

    st.markdown(f'<hr style="border-color:{C["border"]};margin:8px 0"/>', unsafe_allow_html=True)

    # ── System status strip ─────────────────────────────────────────────────
    predictor       = get_predictor()
    ollama_ok       = is_ollama_available()
    storage_mode    = store.STORAGE_MODE.upper()
    iot_configured  = bool(__import__("os").getenv("IOT_ENDPOINT", ""))

    def _dot(ok: bool, warn: bool = False) -> str:
        col = C["normal"] if ok else (C["warning"] if warn else C["anomaly"])
        return f'<span style="color:{col};font-size:0.65rem">●</span>'

    model_ok = predictor.is_ready
    st.markdown(f"""
<div style="padding:0 16px 16px;font-size:0.75rem;color:{C['text_muted']}">
  <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;
              margin-bottom:8px;color:{C['text_dim']}">System Status</div>
  <div style="display:flex;flex-direction:column;gap:5px">
    <div style="display:flex;justify-content:space-between">
      <span>ML Model</span>
      <span>{_dot(model_ok)} {"Ready" if model_ok else "Not trained"}</span>
    </div>
    <div style="display:flex;justify-content:space-between">
      <span>Ollama LLM</span>
      <span>{_dot(ollama_ok, warn=True)} {"Online" if ollama_ok else "Offline"}</span>
    </div>
    <div style="display:flex;justify-content:space-between">
      <span>Storage</span>
      <span>{_dot(True)} {storage_mode}</span>
    </div>
    <div style="display:flex;justify-content:space-between">
      <span>AWS IoT</span>
      <span>{_dot(iot_configured, warn=True)} {"Configured" if iot_configured else "Demo mode"}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Run selected page ───────────────────────────────────────────────────────
nav.run()
