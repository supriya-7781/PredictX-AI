"""
History page
=============
Browse, search, filter, and inspect previously saved analysis sessions.
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from design import (
    C, inject_css, page_header, divider, info_card,
    section_label, apply_plotly_theme, status_badge,
)
from predictx.storage import store

inject_css()

page_header(
    "History",
    "Browse and inspect previously saved anomaly detection sessions",
    "📜",
)

# ── Refresh ───────────────────────────────────────────────────────────────────
refresh_col, _ = st.columns([1, 5])
with refresh_col:
    if st.button("Refresh", type="secondary", use_container_width=True):
        st.rerun()

saved_keys = store.list_results()

if not saved_keys:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="text-align:center;padding:60px 20px;color:{C['text_muted']}">
  <div style="font-size:2.5rem;margin-bottom:16px;opacity:0.4">📜</div>
  <div style="font-size:1rem;font-weight:500;color:{C['text']};margin-bottom:8px">
    No saved sessions
  </div>
  <div style="font-size:0.85rem">
    Save results from <strong>Data Ingestion</strong> or <strong>Live Monitoring</strong>
    to build a history.
  </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Build index ───────────────────────────────────────────────────────────────
# Collect summary info without loading full DataFrames (just check for cached)
@st.cache_data(ttl=60, show_spinner=False)
def _build_index(keys: tuple) -> pd.DataFrame:
    rows = []
    for key in keys:
        try:
            df = store.load_result(key)
            n  = len(df)
            na = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
            short = key.replace("results/", "").replace(".csv", "")
            # Try to parse timestamp from filename
            try:
                dt = datetime.strptime(short, "%Y-%m-%dT%H-%M-%S")
                ts_str = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                ts_str = short
            rows.append({
                "key":       key,
                "session":   short,
                "timestamp": ts_str,
                "readings":  n,
                "anomalies": na,
                "rate":      round(na / n * 100, 1) if n > 0 else 0.0,
                "status":    "Anomalies" if na > 0 else "Normal",
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

index_df = _build_index(tuple(saved_keys))

if index_df.empty:
    info_card("Could not load any saved sessions.", kind="warning")
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
divider()
section_label("Filters")

f1, f2, f3 = st.columns(3)

with f1:
    search_q = st.text_input("Search by session name", placeholder="e.g. 2024-01-15")

with f2:
    status_opts = ["All", "Normal", "Anomalies"]
    status_filter = st.selectbox("Status", status_opts)

with f3:
    sort_by = st.selectbox("Sort by", ["Newest first", "Oldest first", "Most anomalies", "Highest rate"])

# Apply filters
filtered = index_df.copy()

if search_q:
    filtered = filtered[filtered["session"].str.contains(search_q, case=False, na=False)]

if status_filter != "All":
    filtered = filtered[filtered["status"] == status_filter]

if sort_by == "Newest first":
    filtered = filtered.iloc[::-1]
elif sort_by == "Oldest first":
    pass  # already ascending
elif sort_by == "Most anomalies":
    filtered = filtered.sort_values("anomalies", ascending=False)
elif sort_by == "Highest rate":
    filtered = filtered.sort_values("rate", ascending=False)

st.markdown(
    f'<div style="color:{C["text_muted"]};font-size:0.78rem;margin:8px 0">'
    f'Showing {len(filtered)} of {len(index_df)} sessions</div>',
    unsafe_allow_html=True,
)

# ── Summary metrics ───────────────────────────────────────────────────────────
divider()
section_label("Aggregate Summary")

hm1, hm2, hm3, hm4 = st.columns(4)
hm1.metric("Sessions",         len(index_df))
hm2.metric("Total readings",   f"{index_df['readings'].sum():,}")
hm3.metric("Total anomalies",  int(index_df["anomalies"].sum()))
total_r = index_df["readings"].sum()
total_a = index_df["anomalies"].sum()
hm4.metric("Overall rate",
           f"{total_a/total_r*100:.1f}%" if total_r > 0 else "—")

# ── Sessions table ────────────────────────────────────────────────────────────
divider()
section_label("Session List")

if filtered.empty:
    info_card("No sessions match the current filters.", kind="info")
else:
    # Build an HTML table for precise styling
    rows_html = ""
    for _, r in filtered.iterrows():
        status_html = status_badge(r["status"], "anomaly" if r["status"] == "Anomalies" else "normal")
        rate_col = C["anomaly"] if r["rate"] > 5 else (C["warning"] if r["rate"] > 0 else C["normal"])
        rows_html += f"""
<tr style="border-bottom:1px solid {C['border']}">
  <td style="padding:10px 14px;color:{C['text']};font-size:0.82rem;
             font-family:monospace">{r['timestamp']}</td>
  <td style="padding:10px 14px;color:{C['text_muted']};font-size:0.82rem;
             text-align:right">{r['readings']:,}</td>
  <td style="padding:10px 14px;font-size:0.82rem;text-align:right;
             color:{C['anomaly'] if r['anomalies'] > 0 else C['normal']};font-weight:600">
    {r['anomalies']}
  </td>
  <td style="padding:10px 14px;text-align:right;color:{rate_col};
             font-size:0.82rem;font-weight:600">{r['rate']}%</td>
  <td style="padding:10px 14px;text-align:right">{status_html}</td>
</tr>
"""

    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {C['border']};
            border-radius:8px;overflow:hidden">
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="background:{C['surface2']}">
        <th style="padding:10px 14px;color:{C['text_muted']};font-size:0.68rem;
                   text-transform:uppercase;letter-spacing:0.08em;text-align:left">Session</th>
        <th style="padding:10px 14px;color:{C['text_muted']};font-size:0.68rem;
                   text-transform:uppercase;letter-spacing:0.08em;text-align:right">Readings</th>
        <th style="padding:10px 14px;color:{C['text_muted']};font-size:0.68rem;
                   text-transform:uppercase;letter-spacing:0.08em;text-align:right">Anomalies</th>
        <th style="padding:10px 14px;color:{C['text_muted']};font-size:0.68rem;
                   text-transform:uppercase;letter-spacing:0.08em;text-align:right">Rate</th>
        <th style="padding:10px 14px;color:{C['text_muted']};font-size:0.68rem;
                   text-transform:uppercase;letter-spacing:0.08em;text-align:right">Status</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

# ── Detail viewer ─────────────────────────────────────────────────────────────
divider()
section_label("Session Detail")

sel_options = filtered["key"].tolist()
if not sel_options:
    info_card("No sessions available to view.", kind="info")
    st.stop()

sel_key = st.selectbox(
    "Select a session to inspect",
    sel_options,
    format_func=lambda k: k.replace("results/", "").replace(".csv", ""),
    label_visibility="collapsed",
)

if st.button("Load session details", type="primary"):
    with st.spinner("Loading session..."):
        detail_df = store.load_result(sel_key)
        st.session_state["history_detail"] = detail_df
        st.session_state["history_detail_key"] = sel_key

detail_df: pd.DataFrame | None = st.session_state.get("history_detail")
detail_key: str = st.session_state.get("history_detail_key", "")

if detail_df is not None and detail_key == sel_key:
    n_d  = len(detail_df)
    na_d = int(detail_df["is_anomaly"].sum()) if "is_anomaly" in detail_df.columns else 0

    info_card(
        f"Session <code>{sel_key.replace('results/', '').replace('.csv','')}</code>: "
        f"<strong>{n_d:,} readings</strong>, "
        f"<strong style='color:{C['anomaly'] if na_d > 0 else C['normal']}'>"
        f"{na_d} anomalies</strong>",
        kind="success",
    )

    d1, d2 = st.columns(2)
    d1.metric("Readings", f"{n_d:,}")
    d2.metric("Anomalies", na_d, delta_color="inverse")

    if "anomaly_score" in detail_df.columns:
        status_col = detail_df["is_anomaly"].map({0: "Normal", 1: "Anomaly"}) \
                     if "is_anomaly" in detail_df.columns else None
        fig_h = px.histogram(
            detail_df,
            x="anomaly_score",
            color=status_col,
            nbins=30,
            color_discrete_map={"Normal": C["blue"], "Anomaly": C["anomaly"]},
            title="Anomaly Score Distribution",
        )
        fig_h.add_vline(x=0, line_dash="dot", line_color=C["warning"])
        apply_plotly_theme(fig_h, height=250)
        st.plotly_chart(fig_h, use_container_width=True)

    disp = detail_df.copy()
    if "is_anomaly" in disp.columns:
        disp["Status"] = disp["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
        disp = disp.drop(columns=["is_anomaly"])

    st.dataframe(
        disp.style.apply(
            lambda col: [
                f"color:{C['anomaly']};font-weight:600" if v == "Anomaly" else ""
                for v in col
            ] if col.name == "Status" else [""] * len(col),
            axis=0,
        ) if "Status" in disp.columns else disp.style,
        use_container_width=True,
        height=350,
    )

    use_for_analysis = st.button("Use this session for Anomaly Detection", type="secondary")
    if use_for_analysis:
        st.session_state["ingest_result"] = detail_df
        st.switch_page("pages/anomalies.py")
