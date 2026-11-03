"""
AI Maintenance Assistant page
==============================
Structured LLM explanation panel. Shows sensor context, calls Ollama
(or falls back to rule-based), and presents a clean maintenance summary.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from design import (
    C, inject_css, page_header, divider, info_card,
    section_label, status_badge, severity_from_score,
)
from predictx.ollama_client.explainer import (
    explain_anomaly,
    is_ollama_available,
    list_ollama_models,
    OLLAMA_MODEL,
)
# Feature 1 — XAI
from predictx.xai.feature_explainer import get_explainer
# Feature 3 — Recommendations
from predictx.recommendations.recommendation_engine import get_recommendation_engine

inject_css()

page_header(
    "AI Maintenance Assistant",
    "LLM-generated explanations and recommended actions for detected anomalies",
    "🤖",
)

ollama_ok = is_ollama_available()

# ── Ollama status banner ──────────────────────────────────────────────────────
if ollama_ok:
    models = list_ollama_models()
    model_str = ", ".join(models) if models else OLLAMA_MODEL
    info_card(
        f"Ollama is <strong>online</strong> · Model: <code>{model_str}</code> · "
        "Explanations are AI-generated.",
        kind="success",
    )
else:
    info_card(
        "Ollama is <strong>offline</strong>. Rule-based fallback explanations will be shown. "
        "Install Ollama and run <code>ollama pull llama3.2:3b</code> to enable AI explanations.",
        kind="warning",
    )

# ── Load anomaly data ────────────────────────────────────────────────────────
result_df: pd.DataFrame | None = st.session_state.get("ingest_result")

if result_df is None or "is_anomaly" not in result_df.columns:
    divider()
    info_card(
        "No anomaly data available. Go to <strong>Data Ingestion</strong> or "
        "<strong>Live Monitoring</strong> to analyse sensor data first.",
        kind="info",
    )
    st.stop()

anomaly_df = result_df[result_df["is_anomaly"] == 1].copy()
anomaly_df = anomaly_df.sort_values("anomaly_score").reset_index(drop=True)

if anomaly_df.empty:
    divider()
    info_card(
        "No anomalies were detected in the current session — the machines appear healthy.",
        kind="success",
    )
    st.stop()

divider()

# ── Anomaly selector ─────────────────────────────────────────────────────────
section_label("Select Anomaly to Explain")

# Build selector labels
def _make_label(i: int, row: pd.Series) -> str:
    score = float(row["anomaly_score"])
    sev, _ = severity_from_score(score)
    pid = row.get("Product ID", "")
    pid_str = f" · {pid}" if pid and pid != "nan" else ""
    return f"Event #{i+1}{pid_str}  |  Score: {score:.4f}  |  Severity: {sev}"

labels = [_make_label(i, row) for i, (_, row) in enumerate(anomaly_df.iterrows())]

selected_idx = st.selectbox(
    "Anomaly event",
    range(len(labels)),
    format_func=lambda i: labels[i],
    label_visibility="collapsed",
)

selected_row   = anomaly_df.iloc[selected_idx]
selected_score = float(selected_row["anomaly_score"])
sev_label, sev_kind = severity_from_score(selected_score)
sev_col = {"HIGH": C["anomaly"], "MEDIUM": C["warning"], "LOW": C["blue"]}[sev_label]

divider()

# ── Event context panel ───────────────────────────────────────────────────────
section_label("Anomaly Context")

left_panel, right_panel = st.columns([2, 1], gap="large")

with left_panel:
    # Sensor readings
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};
            border-radius:10px;padding:20px;margin-bottom:16px">
  <div style="display:flex;align-items:center;justify-content:space-between;
              margin-bottom:16px">
    <span style="font-weight:600;color:{C['text']}">Sensor Readings</span>
    {status_badge(sev_label, sev_kind)}
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px">
""", unsafe_allow_html=True)

    sensors = [
        ("Air Temp",   "Air temperature [K]",      "K",   290, 310),
        ("Proc Temp",  "Process temperature [K]",   "K",   300, 320),
        ("RPM",        "Rotational speed [rpm]",    "rpm", 1000, 3000),
        ("Torque",     "Torque [Nm]",               "Nm",  0,   100),
        ("Tool Wear",  "Tool wear [min]",           "min", 0,   300),
    ]

    for s_label, col_name, unit, lo, hi in sensors:
        val = float(selected_row.get(col_name, 0))
        pct = max(0, min(100, (val - lo) / (hi - lo) * 100))
        is_extreme = val > hi * 0.90 or val < lo * 1.10
        bar_col = C["anomaly"] if is_extreme else C["accent"]
        st.markdown(f"""
<div style="background:{C['surface2']};border:1px solid {C['border']};
            border-radius:6px;padding:12px">
  <div style="color:{C['text_muted']};font-size:0.65rem;text-transform:uppercase;
              letter-spacing:0.07em;margin-bottom:6px">{s_label}</div>
  <div style="color:{C['text']};font-size:1.15rem;font-weight:700;margin-bottom:8px">
    {val:.1f} <span style="color:{C['text_dim']};font-size:0.7rem">{unit}</span>
  </div>
  <div style="background:{C['border']};border-radius:2px;height:3px">
    <div style="background:{bar_col};width:{pct:.0f}%;height:3px;border-radius:2px"></div>
  </div>
  <div style="display:flex;justify-content:space-between;color:{C['text_dim']};
              font-size:0.62rem;margin-top:3px">
    <span>{lo}</span><span>{hi}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

with right_panel:
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {sev_col}44;
            border-radius:10px;padding:20px">
  <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;
              color:{C['text_muted']};margin-bottom:12px">Detection Details</div>
  <div style="display:flex;flex-direction:column;gap:12px">
    <div>
      <div style="color:{C['text_muted']};font-size:0.72rem">Anomaly Score</div>
      <div style="color:{sev_col};font-size:1.6rem;font-weight:700;
                  font-family:monospace">{selected_score:.5f}</div>
      <div style="color:{C['text_dim']};font-size:0.7rem">Lower = more anomalous</div>
    </div>
    <div style="border-top:1px solid {C['border']};padding-top:12px">
      <div style="color:{C['text_muted']};font-size:0.72rem">Severity</div>
      <div style="margin-top:4px">{status_badge(sev_label, sev_kind)}</div>
    </div>
    <div style="border-top:1px solid {C['border']};padding-top:12px">
      <div style="color:{C['text_muted']};font-size:0.72rem">Event rank</div>
      <div style="color:{C['text']};font-size:0.9rem;font-weight:600">
        #{selected_idx+1} of {len(anomaly_df)} anomalies
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

divider()

# ── AI Explanation ────────────────────────────────────────────────────────────
section_label("AI-Generated Maintenance Explanation")

explain_btn = st.button(
    "Generate Explanation" if ollama_ok else "Generate Rule-Based Explanation",
    type="primary",
)

cache_key = f"explanation_{selected_idx}"

if explain_btn or cache_key in st.session_state:
    if explain_btn:
        with st.spinner("Generating explanation..." if ollama_ok else "Building explanation..."):
            explanation = explain_anomaly(selected_row.to_dict(), selected_score)
            st.session_state[cache_key] = explanation
    else:
        explanation = st.session_state[cache_key]

    source_label = "AI-generated (Ollama)" if ollama_ok else "Rule-based (Ollama offline)"
    source_col   = C["accent"] if ollama_ok else C["warning"]

    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};
            border-radius:10px;padding:24px;margin-top:8px">
  <div style="display:flex;align-items:center;justify-content:space-between;
              margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid {C['border']}">
    <span style="font-weight:600;color:{C['text']}">Maintenance Analysis</span>
    <span style="color:{source_col};font-size:0.72rem;font-style:italic">{source_label}</span>
  </div>
  <div style="color:{C['text']};font-size:0.875rem;line-height:1.7">
""", unsafe_allow_html=True)

    st.markdown(explanation)

    st.markdown(f"""
  </div>
  <div style="margin-top:16px;padding-top:12px;border-top:1px solid {C['border']};
              color:{C['text_muted']};font-size:0.72rem">
    This analysis is generated automatically and is not a confirmed engineering diagnosis.
    Always verify with qualified maintenance personnel before taking action.
  </div>
</div>
""", unsafe_allow_html=True)

else:
    st.markdown(f"""
<div style="background:{C['surface']};border:1px dashed {C['border']};border-radius:10px;
            padding:40px;text-align:center;color:{C['text_muted']}">
  <div style="font-size:1.5rem;margin-bottom:12px;opacity:0.4">🤖</div>
  <div style="font-size:0.9rem">
    Click <strong style="color:{C['text']}">Generate Explanation</strong> to get a
    maintenance analysis for the selected anomaly.
  </div>
</div>
""", unsafe_allow_html=True)

divider()

# ── Batch explain ─────────────────────────────────────────────────────────────
section_label("Batch Analysis")
st.markdown(f"""
<div style="color:{C['text_muted']};font-size:0.82rem;margin-bottom:12px">
  Generate explanations for the top anomalies automatically.
  This may take a moment if Ollama is processing.
</div>
""", unsafe_allow_html=True)

top_n = st.slider("Number of anomalies to explain", 1, min(10, len(anomaly_df)), 3)

if st.button("Run Batch Explanation", type="secondary"):
    batch_top = anomaly_df.head(top_n)
    for i, (_, row) in enumerate(batch_top.iterrows(), 1):
        score = float(row["anomaly_score"])
        slbl, skind = severity_from_score(score)
        with st.expander(f"Event #{i} — Score {score:.4f} — {slbl}", expanded=(i == 1)):
            with st.spinner(f"Explaining event #{i}..."):
                expl = explain_anomaly(row.to_dict(), score)
            st.markdown(expl)
            st.caption(
                "AI-generated (Ollama)" if ollama_ok else "Rule-based (Ollama offline)"
            )

# ── Structured Recommendations (Feature 3) ───────────────────────────────────
divider()
section_label("Structured Maintenance Recommendations")
st.markdown(f"""
<div style="color:{C['text_muted']};font-size:0.82rem;margin-bottom:12px">
  Rule-based recommendations derived from sensor deviations.
  Works without Ollama — always available offline.
</div>
""", unsafe_allow_html=True)

rec_engine = get_recommendation_engine()
explainer  = get_explainer()

rec_top_n = st.slider("Events to analyse", 1, min(5, len(anomaly_df)), 1,
                      key="rec_slider")

for i, (_, row) in enumerate(anomaly_df.head(rec_top_n).iterrows(), 1):
    score     = float(row["anomaly_score"])
    machine   = str(row.get("Product ID", f"Event-{i}"))
    slbl, sk  = severity_from_score(score)

    with st.expander(f"Event #{i}  ·  {machine}  ·  Score {score:.4f}", expanded=(i == 1)):
        try:
            xai_res = explainer.explain(row.to_dict(), score)
            rec_set = rec_engine.recommend(xai_res, machine_id=machine)

            # Summary banner
            banner_col = C["anomaly"] if rec_set.has_actionable else C["normal"]
            st.markdown(f"""
<div style="background:{banner_col}11;border:1px solid {banner_col}44;
            border-radius:6px;padding:10px 14px;margin-bottom:12px;
            color:{C['text']};font-size:0.82rem">
  {rec_set.summary}
</div>
""", unsafe_allow_html=True)

            # Individual recommendations
            for r in rec_set.by_priority():
                badge = status_badge(r.confidence, r.confidence_kind)
                st.markdown(f"""
<div style="background:{C['surface2']};border:1px solid {C['border']};
            border-radius:6px;padding:12px;margin-bottom:8px">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
    {badge}
    <span style="color:{C['text_muted']};font-size:0.72rem">
      Trigger: {r.sensor_trigger}
    </span>
  </div>
  <div style="color:{C['text']};font-size:0.82rem;font-weight:500;
              margin-bottom:5px">{r.action}</div>
  <div style="color:{C['text_muted']};font-size:0.76rem;line-height:1.5;
              border-left:2px solid {C['border']};padding-left:8px">
    {r.evidence}
  </div>
</div>
""", unsafe_allow_html=True)

            # Disclaimer
            st.markdown(f"""
<div style="color:{C['text_dim']};font-size:0.7rem;margin-top:8px;
            border-top:1px solid {C['border']};padding-top:8px">
  {rec_set.disclaimer}
</div>
""", unsafe_allow_html=True)

        except Exception as exc:
            st.markdown(
                f'<div style="color:{C["text_muted"]};font-size:0.8rem">'
                f'Could not generate recommendations: {exc}</div>',
                unsafe_allow_html=True,
            )
