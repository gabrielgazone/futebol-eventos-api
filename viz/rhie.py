# -*- coding: utf-8 -*-
"""Seção RHIE (Repeated High-Intensity Efforts) reutilizável nas abas Esforços,
WCS e Janelas Temporais.

Mostra, por atleta, os blocos RHIE (≥N ações de alta intensidade — Vel/Acc/Dec —
separadas por menos que a recuperação), com composição por tipo e duração, em
tabela selecionável; ao selecionar, anima o bloco no campo (play/pause).

Cortes padronizados e EDITÁVEIS pelo usuário (Vel≥B4, Acc≥A2, Dec≥D2, ≥3 ações,
21 s de recuperação — padrão da literatura).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import _CHAVE_COMBINADO
from analysis import (
    calcular_blocos_rhie, combinar_periodos_continuo, get_min_dur_s)
from viz.campo_anim import (
    _build_period_maps, _periodo_label_de_t, _animar_esforco_campo)

# Defaults dos cortes (correspondem a B4 / A2 / D2 das bandas padrão).
_RHIE_VEL_THR = 19.8   # km/h  (piso da banda B4)
_RHIE_ACC_THR = 3.0    # m/s²  (piso da banda A2)
_RHIE_DEC_THR = 3.0    # m/s²  magnitude (entrada em D2 → desacel ≤ -3.0)
_RHIE_MIN_ACOES = 3
_RHIE_RECUP_S = 21.0


def _fmt_mmss(seg: float) -> str:
    seg = max(0.0, float(seg))
    return f"{int(seg // 60):02d}:{int(seg % 60):02d}"


def _rhie_config(key_pref: str) -> dict:
    """UI editável dos cortes do RHIE (num expander). Retorna o dict de config."""
    with st.expander("⚙️ Cortes do RHIE (editável)", expanded=False):
        st.caption(
            "Padrão da literatura (Buchheit; Spencer): ≥3 ações de alta "
            "intensidade separadas por <21 s. Ações = corridas (Vel≥B4), "
            "acelerações (≥A2) e desacelerações (≥D2). Ajuste os cortes:")
        c1, c2, c3 = st.columns(3)
        vel_thr = c1.number_input(
            "Vel mín (km/h) · B4", value=_RHIE_VEL_THR, min_value=0.0,
            step=0.5, key=f"rhie_velthr_{key_pref}")
        acc_thr = c2.number_input(
            "Acc mín (m/s²) · A2", value=_RHIE_ACC_THR, min_value=0.0,
            step=0.5, key=f"rhie_accthr_{key_pref}")
        dec_thr = c3.number_input(
            "Dec mín (m/s², mag) · D2", value=_RHIE_DEC_THR, min_value=0.0,
            step=0.5, key=f"rhie_decthr_{key_pref}")
        c4, c5 = st.columns(2)
        min_acoes = c4.number_input(
            "Mín. de ações no bloco", value=_RHIE_MIN_ACOES, min_value=2,
            max_value=12, step=1, key=f"rhie_minac_{key_pref}")
        recup = c5.number_input(
            "Recuperação máx entre ações (s)", value=_RHIE_RECUP_S,
            min_value=1.0, step=1.0, key=f"rhie_recup_{key_pref}")
    return {
        'vel_thr': float(vel_thr), 'acc_thr': float(acc_thr),
        'dec_thr': float(dec_thr), 'min_acoes': int(min_acoes),
        'recuperacao_s': float(recup), 'min_dur': get_min_dur_s(),
    }


def _atletas_do_sensor(dados_sensor: dict) -> list:
    _a = []
    for _pd in dados_sensor.values():
        for _nm in _pd:
            if _nm not in _a:
                _a.append(_nm)
    return sorted(_a)


def render_rhie_secao(dados_sensor, dados_posicao, hz, key_pref):
    """Seção RHIE completa e autossuficiente (seletores próprios de período,
    atleta e cortes). Chamável de qualquer aba: só precisa dos dados de sensor
    e de posição (para a animação no campo) + a frequência (hz)."""
    st.markdown("---")
    st.markdown("### ⚡ RHIE — Esforços Repetidos de Alta Intensidade")
    st.caption(
        "Blocos com ≥3 ações de alta intensidade (corridas, acelerações e "
        "desacelerações) próximas no tempo. Veja a composição e a duração de "
        "cada bloco e selecione uma linha para ver o esforço no campo.")

    if not dados_sensor:
        st.info("Carregue uma atividade para ver o RHIE.")
        return

    _cfg = _rhie_config(key_pref)

    _periodos = list(dados_sensor.keys())
    _opts_per = ([_CHAVE_COMBINADO] + _periodos) if len(_periodos) > 1 else _periodos
    _col_p, _col_a = st.columns(2)
    _per_sel = _col_p.selectbox(
        "Período:", _opts_per, key=f"rhie_per_{key_pref}")
    _modo_todos = (_per_sel == _CHAVE_COMBINADO)

    _atletas = _atletas_do_sensor(dados_sensor)
    if not _atletas:
        st.info("Sem atletas com dados de sensor.")
        return
    _atl = _col_a.selectbox("Atleta:", _atletas, key=f"rhie_atl_{key_pref}")

    # ── Linha do tempo do atleta (combinado ou período único) ────────────────
    if _modo_todos:
        _pts = combinar_periodos_continuo(dados_sensor, _atl)
    else:
        _pts = dados_sensor.get(_per_sel, {}).get(_atl, [])
    if not _pts:
        st.info("Sem dados de sensor para este atleta no período.")
        return

    _vel = [float(_p.get('v') or 0.0) * 3.6 for _p in _pts]   # m/s → km/h
    _acc = [float(_p.get('a') or 0.0) for _p in _pts]          # m/s²

    _blocos = calcular_blocos_rhie(
        _vel, _acc, hz,
        vel_thr_kmh=_cfg['vel_thr'], acc_thr_ms2=_cfg['acc_thr'],
        dec_thr_ms2=_cfg['dec_thr'], min_dur_acc_s=_cfg['min_dur'],
        min_acoes=_cfg['min_acoes'], recuperacao_s=_cfg['recuperacao_s'])

    _c1, _c2, _c3 = st.columns(3)
    _c1.metric(f"Blocos RHIE — {_atl}", len(_blocos))
    _c2.metric("Ações totais nos blocos",
               int(sum(_b['n_total'] for _b in _blocos)))
    _c3.metric("Duração somada (s)",
               f"{sum(_b['dur_s'] for _b in _blocos):.0f}")

    if not _blocos:
        st.info("Nenhum bloco RHIE com os cortes atuais.")
        return

    # ── Mapa de tempo (para localizar o bloco no campo em match-time) ────────
    _maps = _build_period_maps(dados_sensor, _modo_todos, _per_sel, dados_posicao)
    _off = _maps['atl_offset'](_atl)

    _rows = []
    for _i, _b in enumerate(_blocos, 1):
        _t0 = _b['frame_ini'] / hz
        _rows.append({
            '#': _i,
            'Início': _fmt_mmss(_t0),
            'Duração (s)': round(_b['dur_s'], 1),
            'Ações': _b['n_total'],
            'Vel (B4+)': _b['n_vel'],
            'Acc (A2+)': _b['n_acc'],
            'Dec (D2+)': _b['n_dec'],
        })
    _df = pd.DataFrame(_rows)
    _sel = st.dataframe(
        _df, use_container_width=True, hide_index=True,
        height=min(430, 44 + len(_rows) * 36),
        on_select="rerun", selection_mode="single-row",
        key=f"rhie_tbl_{key_pref}")
    _idx = (_sel.selection.rows[0]
            if (hasattr(_sel, 'selection') and _sel.selection.rows) else 0)
    _b = _blocos[_idx]

    # Resumo do bloco no formato pedido (Atleta / composição / duração)
    st.markdown(
        f"**{_atl}** — bloco {_idx + 1}: "
        f"**{_b['n_vel']}** Vel B4+ · **{_b['n_acc']}** Acc A2+ · "
        f"**{_b['n_dec']}** Dec D2+ · **{_b['dur_s']:.0f} s** de duração "
        f"(⏱️ {_fmt_mmss(_b['frame_ini'] / hz)}→{_fmt_mmss(_b['frame_fim'] / hz)})")
    st.caption("Clique numa linha da tabela para plotar o bloco no campo.")

    _t_ini = _b['frame_ini'] / hz / 60.0 + _off       # match-time (min)
    _win = max(_b['dur_s'] / 60.0, 1.0 / 60.0)
    _per_lbl = _periodo_label_de_t(_maps, _t_ini)
    _animar_esforco_campo(
        [_atl], _t_ini, _win, dados_posicao, _maps, "ações",
        _fmt_mmss(_b['frame_ini'] / hz), _fmt_mmss(_b['frame_fim'] / hz),
        _b['n_total'], _per_lbl, min_atl=1)
