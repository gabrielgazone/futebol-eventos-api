# -*- coding: utf-8 -*-
"""Diagnóstico da sessão e proveniência do dado (P4/P5 — extraído do monólito).

Selo de fonte (_selo_fonte) e o log de eventos visíveis na aba de diagnóstico
(_diag_log/_diag_reset). Streamlit-aware (sessão + caption); loga em applog.
"""
from __future__ import annotations

import streamlit as st
import applog as _applog


# ── P4: selo de proveniência do dado ─────────────────────────────────────────
# Toda métrica sensível mostra DE ONDE veio o número (o app escolhe a fonte
# automaticamente; o usuário precisa saber qual foi usada).
_PROV_LABELS = {
    'efforts':  ('🟢', "Ações oficiais da API Catapult (*acceleration_efforts*, caixa Gen2)"),
    'sensor':   ('🔵', "Sinal nativo do sensor (10 Hz)"),
    'derivado': ('🟡', "Derivado do sinal de velocidade (dv/dt) — menor confiança"),
    'gps':      ('🟣', "Velocidade da trajetória GPS (field-filtered)"),
    'summary':  ('⚪', "Resumo pré-calculado OpenField"),
}


def _selo_fonte(fonte, extra: str = ""):
    """(P4) Selo de proveniência: informa a fonte do número exibido."""
    icone, desc = _PROV_LABELS.get(fonte, ('⚪', str(fonte)))
    st.caption(f"{icone} **Fonte do dado:** {desc}"
               + (f" · {extra}" if extra else ""))


# ── P5: diagnóstico da sessão (falhas silenciosas viram eventos visíveis) ───
def _diag_log(categoria: str, msg: str):
    """(P5) Registra um evento de diagnóstico (dado descartado / fallback)."""
    try:
        _lst = st.session_state.setdefault('_diag_eventos', [])
        _item = f"**[{categoria}]** {msg}"
        if _item not in _lst:
            _lst.append(_item)
            if len(_lst) > 200:
                del _lst[:-200]
    except Exception:
        _applog.log_debug_exc()


def _diag_reset():
    """(P5) Limpa o diagnóstico no início de um novo carregamento."""
    try:
        st.session_state['_diag_eventos'] = []
        st.session_state.pop('_api_last_err', None)
    except Exception:
        _applog.log_debug_exc()
