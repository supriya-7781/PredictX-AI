"""
design.py
=========
Shared design system for the PredictX dashboard.

Injects global CSS, exposes colour tokens, and provides reusable
UI helper functions used across all pages.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Colour tokens
# ---------------------------------------------------------------------------
C = {
    "bg":           "#0f1117",   # page background
    "surface":      "#1a1f2e",   # card / panel surface
    "surface2":     "#242938",   # elevated surface
    "border":       "#2d3348",   # subtle border
    "accent":       "#00d4aa",   # primary cyan-teal accent
    "accent_dim":   "#00a882",   # darker accent
    "normal":       "#22c55e",   # green — normal status
    "warning":      "#f59e0b",   # amber — warning
    "anomaly":      "#ef4444",   # red — anomaly / error
    "anomaly_dim":  "#7f1d1d",   # dark red bg tint
    "text":         "#e2e8f0",   # primary text
    "text_muted":   "#8892a4",   # secondary text
    "text_dim":     "#4a5568",   # very dim text
    "blue":         "#3b82f6",   # informational blue
    "purple":       "#8b5cf6",   # purple accent (charts)
}

# Plotly theme shared across all charts
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_PAPER_BG  = "rgba(0,0,0,0)"
PLOTLY_PLOT_BG   = "rgba(26,31,46,0.6)"
PLOTLY_GRID_COL  = "rgba(45,51,72,0.8)"
PLOTLY_FONT_COL  = C["text_muted"]


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Inject the full PredictX design system CSS into the current page."""
    st.markdown(f"""
<style>
/* ── Reset & base ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {C['bg']} !important;
    color: {C['text']};
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}

/* Main content area */
[data-testid="stMain"] {{
    background-color: {C['bg']};
}}

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: {C['surface']} !important;
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{
    color: {C['text']} !important;
}}
[data-testid="stSidebarNavLink"] {{
    border-radius: 6px;
    margin: 2px 8px;
    padding: 8px 12px;
    transition: background 0.15s;
    color: {C['text_muted']} !important;
    font-size: 0.875rem;
    font-weight: 400;
}}
[data-testid="stSidebarNavLink"]:hover {{
    background-color: {C['surface2']} !important;
    color: {C['text']} !important;
}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
    background-color: rgba(0,212,170,0.12) !important;
    color: {C['accent']} !important;
    border-left: 3px solid {C['accent']};
    font-weight: 500;
}}

/* ── Headings ─────────────────────────────────────────────── */
h1 {{ color: {C['text']} !important; font-size: 1.5rem !important; font-weight: 600 !important; letter-spacing: -0.02em; }}
h2 {{ color: {C['text']} !important; font-size: 1.2rem !important; font-weight: 600 !important; margin-top: 1.5rem !important; }}
h3 {{ color: {C['text_muted']} !important; font-size: 1rem !important; font-weight: 500 !important; }}

/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 14px 18px !important;
}}
[data-testid="stMetricLabel"] {{ color: {C['text_muted']} !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }}
[data-testid="stMetricValue"] {{ color: {C['text']} !important; font-size: 1.6rem !important; font-weight: 700 !important; }}
[data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}

/* ── Buttons ──────────────────────────────────────────────── */
[data-testid="stBaseButton-primary"] > button,
button[kind="primary"] {{
    background: {C['accent']} !important;
    color: #0f1117 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
    transition: opacity 0.15s;
}}
[data-testid="stBaseButton-primary"] > button:hover {{
    opacity: 0.85 !important;
}}
[data-testid="stBaseButton-secondary"] > button,
button[kind="secondary"] {{
    background: {C['surface2']} !important;
    color: {C['text']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}}

/* ── Dataframes / tables ──────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C['border']};
    border-radius: 8px;
    overflow: hidden;
}}
.stDataFrame thead tr th {{
    background: {C['surface2']} !important;
    color: {C['text_muted']} !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.stDataFrame tbody tr:nth-child(even) {{
    background: rgba(36,41,56,0.5) !important;
}}

/* ── Expanders ────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    margin-bottom: 8px;
}}
[data-testid="stExpanderToggleIcon"] {{ color: {C['accent']} !important; }}

/* ── Alerts ───────────────────────────────────────────────── */
[data-testid="stAlert"] {{
    border-radius: 8px !important;
    border-left-width: 3px !important;
}}

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: {C['surface']};
    border: 2px dashed {C['border']};
    border-radius: 10px;
    padding: 8px;
    transition: border-color 0.2s;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: {C['accent']};
}}

/* ── Select / inputs ──────────────────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] > div {{
    background: {C['surface']} !important;
    border-color: {C['border']} !important;
    border-radius: 6px !important;
}}

/* ── Progress bar ─────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {{
    background: {C['accent']} !important;
}}

/* ── Tabs ─────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {C['surface']} !important;
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent !important;
    color: {C['text_muted']} !important;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background: {C['surface2']} !important;
    color: {C['accent']} !important;
}}

/* ── Dividers ─────────────────────────────────────────────── */
hr {{ border-color: {C['border']} !important; margin: 1rem 0 !important; }}

/* ── Spinner text ─────────────────────────────────────────── */
[data-testid="stSpinner"] {{ color: {C['text_muted']} !important; }}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C['surface']}; }}
::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C['text_dim']}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------

def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a consistent page header with title, icon, and optional subtitle."""
    icon_html = f'<span style="color:{C["accent"]};margin-right:10px;font-size:1.3rem">{icon}</span>' if icon else ""
    st.markdown(f"""
<div style="margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid {C['border']}">
  <h1 style="margin:0;display:flex;align-items:center">{icon_html}{title}</h1>
  {"<p style='margin:4px 0 0;color:" + C['text_muted'] + ";font-size:0.875rem'>" + subtitle + "</p>" if subtitle else ""}
</div>
""", unsafe_allow_html=True)


def status_badge(label: str, kind: str = "normal") -> str:
    """
    Return an HTML status badge string.
    kind: 'normal' | 'warning' | 'anomaly' | 'info' | 'muted'
    """
    colours = {
        "normal":  (C["normal"],  "rgba(34,197,94,0.12)"),
        "warning": (C["warning"], "rgba(245,158,11,0.12)"),
        "anomaly": (C["anomaly"], "rgba(239,68,68,0.12)"),
        "info":    (C["blue"],    "rgba(59,130,246,0.12)"),
        "muted":   (C["text_muted"], C["surface2"]),
    }
    fg, bg = colours.get(kind, colours["muted"])
    return (
        f'<span style="background:{bg};color:{fg};border:1px solid {fg}33;'
        f'padding:2px 10px;border-radius:20px;font-size:0.75rem;'
        f'font-weight:600;letter-spacing:0.04em;white-space:nowrap">'
        f'{label}</span>'
    )


def metric_card(label: str, value: str, sub: str = "", accent: bool = False) -> None:
    """Render a styled metric card using HTML (more control than st.metric)."""
    border_col = C["accent"] if accent else C["border"]
    st.markdown(f"""
<div style="background:{C['surface']};border:1px solid {border_col};
            border-radius:8px;padding:16px 20px;height:100%">
  <div style="color:{C['text_muted']};font-size:0.7rem;text-transform:uppercase;
              letter-spacing:0.07em;margin-bottom:6px">{label}</div>
  <div style="color:{C['text']};font-size:1.5rem;font-weight:700;
              line-height:1.2">{value}</div>
  {"<div style='color:" + C['text_muted'] + ";font-size:0.75rem;margin-top:4px'>" + sub + "</div>" if sub else ""}
</div>
""", unsafe_allow_html=True)


def info_card(text: str, kind: str = "info") -> None:
    """Inline info / warning / error banner."""
    colours = {
        "info":    (C["blue"],    "rgba(59,130,246,0.08)"),
        "warning": (C["warning"], "rgba(245,158,11,0.08)"),
        "error":   (C["anomaly"], "rgba(239,68,68,0.08)"),
        "success": (C["normal"],  "rgba(34,197,94,0.08)"),
    }
    fg, bg = colours.get(kind, colours["info"])
    st.markdown(f"""
<div style="background:{bg};border-left:3px solid {fg};border-radius:0 6px 6px 0;
            padding:10px 16px;margin:8px 0;color:{C['text']};font-size:0.875rem">
  {text}
</div>
""", unsafe_allow_html=True)


def section_label(text: str) -> None:
    """Small uppercase section label."""
    st.markdown(
        f'<div style="color:{C["text_muted"]};font-size:0.7rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.1em;margin:1.2rem 0 0.5rem">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown(f'<hr style="border-color:{C["border"]};margin:1.2rem 0"/>', unsafe_allow_html=True)


def apply_plotly_theme(fig, height: int = 360) -> None:
    """Apply the shared dark theme to a Plotly figure in-place."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_PLOT_BG,
        font=dict(color=PLOTLY_FONT_COL, family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=0, r=0, t=36, b=0),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=PLOTLY_GRID_COL,
            borderwidth=1,
        ),
        xaxis=dict(gridcolor=PLOTLY_GRID_COL, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=PLOTLY_GRID_COL, showgrid=True, zeroline=False),
    )


def severity_from_score(score: float) -> tuple[str, str]:
    """Return (label, kind) for a given anomaly score."""
    if score < -0.10:
        return "HIGH", "anomaly"
    if score < -0.05:
        return "MEDIUM", "warning"
    return "LOW", "info"
