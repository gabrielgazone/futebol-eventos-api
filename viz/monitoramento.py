# -*- coding: utf-8 -*-
"""Aba de Monitoramento longitudinal (P4 — 1a render_* extraida para viz/).

Carga longitudinal via /stats: ACWR (EWMA), monotonia/strain (Foster) e
semaforo de risco. Depende so de modulos ja extraidos (metrics, diagnostics,
i18n) + streamlit/pandas/numpy.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import metrics as _mtr
from diagnostics import _diag_log
from i18n import t


_ACWR_EMOJI = {'ideal': '🟢', 'subcarga': '🟡', 'atenção': '🟠',
               'alto risco': '🔴', '—': '⚪'}


def render_monitoramento():
    """Aba 📈 Monitoramento — carga longitudinal da conta via POST /stats."""
    import datetime as _dtm
    import time as _tm

    st.markdown("### 📈 Monitoramento Longitudinal")
    st.caption(
        "Consome o histórico da conta (POST /stats) para monitorar a carga ao "
        "longo das semanas: **ACWR** agudo:crônico (EWMA 7:28 — Williams et al., "
        "2017; semanal acoplado quando o histórico é semanal), **monotonia** e "
        "**strain** de Foster, com semáforo de risco (zona ideal: **0,8–1,3**)."
    )
    _api = st.session_state.get('api')
    if _api is None:
        st.info("Conecte-se e carregue uma atividade primeiro — o token da "
                "sessão é usado para consultar o histórico.")
        return

    _hoje = _dtm.date.today()
    _cm1, _cm2, _cm3 = st.columns(3)
    with _cm1:
        _d_ini = st.date_input("De:", value=_hoje - _dtm.timedelta(days=56),
                               key="mon_d_ini")
    with _cm2:
        _d_fim = st.date_input("Até:", value=_hoje, key="mon_d_fim")
    with _cm3:
        _metrica = st.selectbox(
            "Métrica de carga", ["player_load", "total_distance"],
            format_func=lambda v: {"player_load": "PlayerLoad",
                                   "total_distance": "Distância (m)"}[v],
            key="mon_metrica")

    if st.button("🔄 Consultar histórico (/stats)", type="primary",
                 key="btn_mon_stats"):
        _base_payload = {
            "parameters": ["total_distance", "player_load"],
            "source": "cached_stats",
            "start_time": int(_tm.mktime(_d_ini.timetuple())),
            "end_time": int(_tm.mktime(
                (_d_fim + _dtm.timedelta(days=1)).timetuple())),
        }
        with st.spinner("Consultando /stats..."):
            _resp, _gran = None, None
            # Tenta granularidade diária; cai para semanal se a conta não expõe.
            for _dims, _g in ((["athlete", "date"], 'date'),
                              (["athlete", "day"], 'date'),
                              (["athlete", "week"], 'week')):
                try:
                    _r = _api.get_stats(dict(_base_payload, group_by=_dims))
                except Exception:
                    _r = None
                _rows_r = (_r if isinstance(_r, list)
                           else (_r or {}).get('data', []))
                if _rows_r:
                    _resp, _gran = _rows_r, _g
                    break
            st.session_state['mon_resp'] = (_resp, _gran)
            if _resp is None:
                _diag_log('Monitoramento', "POST /stats sem dados para o "
                                           "intervalo/agrupamentos testados")

    _saved = st.session_state.get('mon_resp')
    if not _saved or not _saved[0]:
        if _saved is not None:
            st.warning("O /stats não retornou dados para o intervalo — "
                       "verifique permissões do token ou se há histórico "
                       "suficiente na plataforma.")
        return
    _rows, _gran = _saved

    # ── Normaliza a resposta (achata 'parameters' aninhado) ─────────────────
    _flat = []
    for _r in _rows:
        if not isinstance(_r, dict):
            continue
        _d = dict(_r)
        _p = _d.pop('parameters', None)
        if isinstance(_p, dict):
            _d.update(_p)
        _flat.append(_d)
    _dfm = pd.DataFrame(_flat)
    _col_atl = next((c for c in ('athlete_name', 'athlete', 'name')
                     if c in _dfm.columns), None)
    _col_t = next((c for c in ('date', 'day', 'week', 'week_start',
                               'start_time') if c in _dfm.columns), None)
    if _dfm.empty or _col_atl is None or _col_t is None \
            or _metrica not in _dfm.columns:
        st.warning("Formato inesperado da resposta do /stats — veja a "
                   "resposta crua abaixo e me reporte o formato.")
        with st.expander("Resposta crua (3 primeiras linhas)"):
            st.json(_rows[:3])
        return
    _dfm[_metrica] = pd.to_numeric(_dfm[_metrica], errors='coerce').fillna(0.0)

    # Coluna temporal → datetime (epoch ou string); semanas-rótulo viram ordem.
    if pd.api.types.is_numeric_dtype(_dfm[_col_t]) and \
            float(_dfm[_col_t].max()) > 1e8:
        _dfm['_t'] = pd.to_datetime(_dfm[_col_t], unit='s', errors='coerce')
    else:
        _dfm['_t'] = pd.to_datetime(_dfm[_col_t].astype(str), errors='coerce')
    if _dfm['_t'].isna().all():
        _dfm['_t'] = pd.factorize(_dfm[_col_t].astype(str))[0]
        _gran = 'week'
    _dfm = _dfm.dropna(subset=['_t'])

    _lbl_metr = {"player_load": "PlayerLoad",
                 "total_distance": "Distância (m)"}[_metrica]
    st.caption(f"📡 {len(_dfm)} registros · granularidade **"
               f"{'diária' if _gran == 'date' else 'semanal'}** · "
               f"métrica **{_lbl_metr}**")

    # ── Métricas por atleta ──────────────────────────────────────────────────
    _tab_rows = []
    _series_por_atleta = {}
    for _atl_m, _g in _dfm.groupby(_col_atl):
        _s = _g.groupby('_t')[_metrica].sum().sort_index()
        if _gran == 'date' and isinstance(_s.index, pd.DatetimeIndex):
            _idx_full = pd.date_range(_s.index.min(), _s.index.max(), freq='D')
            _s = _s.reindex(_idx_full, fill_value=0.0)
            _acwr_serie = _mtr.acwr_ewma(list(_s.values))
            _carga_ag = float(np.sum(_s.values[-7:]))
            _mono, _strain = _mtr.monotonia_strain(list(_s.values[-7:]))
            _wk = _s.resample('W').sum()
            _z = None
            if len(_wk) >= 5:
                _prev4 = _wk.values[-5:-1].astype(float)
                _sd4 = float(np.std(_prev4, ddof=1))
                if _sd4 > 1e-9:
                    _z = float((_wk.values[-1] - _prev4.mean()) / _sd4)
        else:
            _vals = [float(x) for x in _s.values]
            _acwr_serie = _mtr.acwr_semanal(_vals)
            _carga_ag = _vals[-1] if _vals else 0.0
            _mono = _strain = None
            _z = None
            if len(_vals) >= 5:
                _prev4 = np.asarray(_vals[-5:-1], dtype=float)
                _sd4 = float(np.std(_prev4, ddof=1))
                if _sd4 > 1e-9:
                    _z = float((_vals[-1] - _prev4.mean()) / _sd4)
        _acwr_last = next((a for a in reversed(_acwr_serie)
                           if a is not None), None)
        _zona = _mtr.classificar_acwr(_acwr_last)
        _series_por_atleta[str(_atl_m)] = (_s, _acwr_serie)
        _tab_rows.append({
            'Zona': _ACWR_EMOJI.get(_zona, '⚪'),
            'Atleta': str(_atl_m),
            'ACWR': round(_acwr_last, 2) if _acwr_last is not None else None,
            'Classificação': _zona,
            f'Carga aguda ({_lbl_metr})': round(_carga_ag, 0),
            'Monotonia (Foster)': (round(_mono, 2)
                                   if _mono is not None else None),
            'Strain (Foster)': (round(_strain, 0)
                                if _strain is not None else None),
            'Z-score vs 4 sem.': round(_z, 2) if _z is not None else None,
        })

    _df_tab = pd.DataFrame(_tab_rows).sort_values(
        'ACWR', ascending=False, na_position='last')
    st.markdown("##### 🚦 Semáforo de risco por atleta")
    st.dataframe(_df_tab, use_container_width=True, hide_index=True)
    st.caption("🟢 0,8–1,3 ideal · 🟡 <0,8 subcarga · 🟠 1,3–1,5 atenção · "
               "🔴 >1,5 alto risco. **Monotonia** ≥ 2,0 indica carga pouco "
               "variada (maior risco); **strain** = carga semanal × monotonia. "
               "Monotonia/strain exigem granularidade diária.")
    st.download_button("📥 Exportar monitoramento (CSV)",
                       _df_tab.to_csv(index=False).encode('utf-8'),
                       "monitoramento_acwr.csv", mime='text/csv',
                       key="dl_mon_csv")

    # ── Gráfico individual: carga + ACWR ────────────────────────────────────
    _atl_plot = st.selectbox("Atleta para o gráfico",
                             sorted(_series_por_atleta.keys()),
                             key="mon_atleta_plot")
    _s_pl, _acwr_pl = _series_por_atleta[_atl_plot]
    _x_pl = list(_s_pl.index)
    _figm = go.Figure()
    _figm.add_trace(go.Bar(x=_x_pl, y=list(_s_pl.values),
                           name=_lbl_metr,
                           marker_color='#42A5F5', opacity=0.75))
    _figm.add_trace(go.Scatter(
        x=_x_pl, y=[a if a is not None else None for a in _acwr_pl],
        name='ACWR', yaxis='y2', mode='lines+markers',
        line=dict(color='#FFD740', width=2.5)))
    _figm.add_hrect(y0=0.8, y1=1.3, yref='y2', fillcolor='rgba(102,187,106,0.15)',
                    line_width=0, annotation_text='zona ideal',
                    annotation_font_color='#66BB6A')
    _figm.update_layout(
        title=f"Carga e ACWR — {_atl_plot}",
        height=430, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#1f2937'),
        yaxis=dict(title=_lbl_metr, gridcolor='#1f2937'),
        yaxis2=dict(title='ACWR', overlaying='y', side='right',
                    showgrid=False, range=[0, 2.5]),
        legend=dict(orientation='h', y=1.12),
    )
    st.plotly_chart(_figm, use_container_width=True)
    st.caption("Barras = carga por "
               f"{'dia' if _gran == 'date' else 'semana'}; linha = ACWR "
               "(eixo direito) com a faixa verde 0,8–1,3. Referências: "
               "Gabbett (2016); Williams et al. (2017); Foster (1998).")


