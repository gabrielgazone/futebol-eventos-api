# -*- coding: utf-8 -*-
"""Tema e helpers de design (P4 — extraído do monólito).

CSS global do app + helpers visuais (_hr, _badge). Streamlit-aware (injeta
markup), mas sem acoplamento com a lógica/estado do app.
"""
from __future__ import annotations

import streamlit as st


_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─ Tipografia global ─────────────────────────────────────────── */
html, body, [class*="css"], .stApp, .stMarkdown, .stCaption,
button, input, select, textarea, label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ─ Fade-in na área principal ──────────────────────────────────── */
@keyframes _fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.main .block-container { animation: _fadeUp 0.35s ease-out; }

/* ─ Tabs — navegação em 2 níveis (robusto a versões do Streamlit) ─ */
/* Mira role="tab" (ARIA, estável) + data-baseweb (legado). Neutraliza  */
/* o indicador/borda padrão do BaseWeb p/ não conflitar com o desenho.  */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    background: transparent !important; height: 0 !important;
}

/* NÍVEL 1 — barra principal: "segmented control" de vidro */
.stTabs [data-baseweb="tab-list"] {
    gap: 5px; padding: 6px; margin-bottom: 8px;
    border: 1px solid rgba(255,255,255,0.06); border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%,
                rgba(255,255,255,0.015) 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05),
                0 8px 24px rgba(0,0,0,0.30);
    backdrop-filter: blur(8px);
    flex-wrap: wrap;
    animation: _tabsIn 0.45s cubic-bezier(.2,.7,.3,1);
}
@keyframes _tabsIn {
    from { opacity: 0; transform: translateY(-5px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.stTabs button[role="tab"], .stTabs [data-baseweb="tab"] {
    position: relative;
    border-radius: 10px !important;
    padding: 8px 17px !important;
    background: transparent !important;
    color: rgba(255,255,255,0.52) !important;
    border: 1px solid transparent !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    white-space: nowrap;
    transition: color .2s, background .2s, box-shadow .25s, transform .12s !important;
}
.stTabs button[role="tab"]:hover {
    color: rgba(255,255,255,0.92) !important;
    background: rgba(46,134,193,0.14) !important;
    transform: translateY(-1px);
}
.stTabs button[role="tab"]:focus-visible {
    outline: 2px solid rgba(93,173,226,0.7) !important; outline-offset: 2px;
}
.stTabs button[aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #1a5276 0%, #2471a3 55%, #2e86c1 100%) !important;
    border-color: rgba(93,173,226,0.45) !important;
    box-shadow: 0 4px 18px rgba(36,113,163,0.45),
                inset 0 1px 0 rgba(255,255,255,0.16) !important;
}

/* NÍVEL 2 — sub-abas: mais leves, hierarquia claramente secundária */
.stTabs .stTabs [data-baseweb="tab-list"] {
    gap: 3px; padding: 0; margin: 2px 0 12px 0;
    border: none; border-radius: 0; background: transparent;
    box-shadow: none; backdrop-filter: none;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    animation: none;
}
.stTabs .stTabs button[role="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 6px 14px !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.45) !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs .stTabs button[role="tab"]:hover {
    background: rgba(46,134,193,0.10) !important;
    color: rgba(255,255,255,0.85) !important;
    transform: none;
}
.stTabs .stTabs button[aria-selected="true"] {
    color: #7cc4ef !important;
    background: rgba(46,134,193,0.12) !important;
    border-color: transparent !important;
    border-bottom: 2px solid #2e86c1 !important;
    box-shadow: none !important;
}

/* ─ Metric cards ───────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(46,134,193,0.35) !important;
    box-shadow: 0 0 14px rgba(46,134,193,0.13) !important;
}

/* ─ Expanders ──────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.015) !important;
    overflow: hidden;
}
details[data-testid="stExpander"] summary { font-weight: 500; }
details[data-testid="stExpander"] summary:hover { color: #5dade2 !important; }

/* ─ Divisores temáticos ────────────────────────────────────────── */
.app-divider {
    display: flex; align-items: center; gap: 10px;
    margin: 18px 0 14px 0; color: rgba(255,255,255,0.22);
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 1.8px; text-transform: uppercase;
}
.app-divider::before, .app-divider::after {
    content: ''; flex: 1;
    border-top: 1px solid rgba(255,255,255,0.07);
}

/* ─ Badges de posição ──────────────────────────────────────────── */
.badge-gk  { display:inline-block;padding:2px 9px;border-radius:12px;font-size:0.7rem;font-weight:600;
             background:#1a237e;border:1px solid #5c6bc0;color:white; }
.badge-def { display:inline-block;padding:2px 9px;border-radius:12px;font-size:0.7rem;font-weight:600;
             background:#1b5e20;border:1px solid #43a047;color:white; }
.badge-mid { display:inline-block;padding:2px 9px;border-radius:12px;font-size:0.7rem;font-weight:600;
             background:#e65100;border:1px solid #fb8c00;color:white; }
.badge-fwd { display:inline-block;padding:2px 9px;border-radius:12px;font-size:0.7rem;font-weight:600;
             background:#880e4f;border:1px solid #e91e8c;color:white; }
.badge-gen { display:inline-block;padding:2px 9px;border-radius:12px;font-size:0.7rem;font-weight:600;
             background:#263238;border:1px solid #546e7a;color:white; }

/* ─ Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141921 0%, #1b2436 100%) !important;
    border-right: 2px solid rgba(255,255,255,0.11) !important;
    box-shadow: 4px 0 32px rgba(0,0,0,0.55) !important;
}
[data-testid="stSidebarContent"] {
    padding-top: 1rem !important;
}

/* ─ Botões ─────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35) !important;
}

/* ─ Scrollbar ──────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.11); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
</style>
"""


def inject_global_css() -> None:
    """Injeta o CSS global uma vez por render (chamado no topo de main)."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ─── Design helpers ────────────────────────────────────────────────────────────
def _hr(label: str = "", icon: str = ""):
    """Divisor de seção temático com ícone e label."""
    _inner = f"{icon}&nbsp;&nbsp;{label}" if (icon or label) else ""
    st.markdown(
        f'<div class="app-divider">{_inner}</div>',
        unsafe_allow_html=True,
    )


def _badge(pos: str) -> str:
    """Retorna HTML de badge colorido por posição."""
    _pos_l = (pos or "").lower()
    if any(k in _pos_l for k in ('goleiro', 'goalkeeper', 'gk', 'portero', 'gardien')):
        return f'<span class="badge-pos badge-gk">{pos}</span>'
    if any(k in _pos_l for k in ('defens', 'zagueiro', 'lateral', 'defender', 'back')):
        return f'<span class="badge-pos badge-def">{pos}</span>'
    if any(k in _pos_l for k in ('meia', 'meio', 'midfield', 'volante', 'centrocampista')):
        return f'<span class="badge-pos badge-mid">{pos}</span>'
    if any(k in _pos_l for k in ('atacante', 'forward', 'striker', 'winger', 'delantero')):
        return f'<span class="badge-pos badge-fwd">{pos}</span>'
    return f'<span class="badge-pos badge-gen">{pos}</span>'
