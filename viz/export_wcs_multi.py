# -*- coding: utf-8 -*-
"""Export WCS multi-atividade (artigo científico).

Seção da aba Exportação que permite escolher VÁRIAS atividades de uma vez
(independente do filtro da barra lateral), carregar cada uma via API e exportar
os picos de pior cenário (worst-case scenario) por atleta × variável × janela
(1/3/5 min), em formato longo/tidy pronto para jamovi/R/SPSS.

O cálculo delega para `wcs_export` (mesmo método canônico da aba WCS).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import applog as _applog
import wcs_export as _wx
from analysis import combinar_periodos_continuo
from data_loader import carregar_dados

_SS_RES = '_wcs_multi_resultado'      # DataFrame tidy calculado
_SS_LOG = '_wcs_multi_log'            # avisos por atividade


def _nome_atleta(_m):
    """Nome do atleta num dict de métricas (tolerante à chave)."""
    for _k in ('Atleta', 'atleta', 'Nome', 'nome', 'Athlete'):
        if _m.get(_k):
            return str(_m[_k])
    return ''


def _mapa_atleta_info(resultados_por_periodo):
    """{atleta: (posição, equipe)} a partir dos resultados carregados."""
    _out = {}
    for _lst in (resultados_por_periodo or {}).values():
        for _m in (_lst or []):
            _nm = _nome_atleta(_m)
            if _nm and _nm not in _out:
                _out[_nm] = (_m.get('Posição') or '', _m.get('Equipe') or '')
    return _out


def _linhas_wcs_atividade(act_nome, act_data, dados_sensor, resultados,
                          variaveis, janelas, escopos, hz, cortes):
    """Linhas tidy de WCS para UMA atividade já carregada."""
    _rows = []
    _info = _mapa_atleta_info(resultados)
    _atletas = sorted({_a for _p in dados_sensor.values() for _a in _p.keys()})

    for _atl in _atletas:
        _pos, _eq = _info.get(_atl, ('', ''))

        _escopo_series = []
        if 'Partida inteira' in escopos:
            _sp = (combinar_periodos_continuo(dados_sensor, _atl)
                   if len(dados_sensor) > 1
                   else next(iter(dados_sensor.values()), {}).get(_atl, []))
            if _sp:
                _escopo_series.append(('Partida inteira', _sp))
        if 'Por período' in escopos:
            for _pnm, _pdados in dados_sensor.items():
                _sp_p = _pdados.get(_atl, [])
                if _sp_p:
                    _escopo_series.append((_pnm, _sp_p))

        for _escopo, _sp in _escopo_series:
            _picos = _wx.calcular_wcs(_sp, variaveis, janelas, hz, **cortes)
            for (_var, _wmin), _val in _picos.items():
                _rows.append({
                    'Atividade': act_nome,
                    'Data': act_data,
                    'Equipe': _eq,
                    'Atleta': _atl,
                    'Posicao': _pos,
                    'Escopo': _escopo,
                    'Variavel': _var,
                    'Janela_min': _wmin,
                    'Valor': _val,
                })
    return _rows


def render_export_wcs_multi(api):
    """Seção de export WCS de múltiplas atividades (tidy, para jamovi)."""
    st.markdown("---")
    st.markdown("### 🔬 Export WCS multi-atividade (artigo científico)")
    st.caption(
        "Escolha **várias atividades** (independente do filtro da barra lateral) "
        "e exporte os picos de **pior cenário** por atleta, variável e janela "
        "(1/3/5 min) em formato **longo/tidy** — pronto para jamovi, R ou SPSS. "
        "O cálculo usa o **mesmo método da aba WCS**.")

    if api is None:
        st.info("Conecte-se à API (barra lateral) para usar o export multi-atividade.")
        return

    _dfa = st.session_state.get('df_activities')
    if _dfa is None or getattr(_dfa, 'empty', True):
        st.info("Carregue os dados na barra lateral uma vez para listar as atividades.")
        return

    # ── Seleção de atividades (independente da sidebar) ──────────────────────
    _opts, _meta = [], {}
    for _, _r in _dfa.iterrows():
        _nm = str(_r.get('nome') or '(sem nome)')
        _dt = str(_r.get('data') or '')
        _lbl = f"{_nm} — {_dt[:10]}" if _dt else _nm
        _opts.append(_lbl)
        _meta[_lbl] = (_r.get('id'), _nm, _dt[:10])

    _sel = st.multiselect(
        "Atividades a exportar (busque por nome/data):", _opts,
        key='wcs_multi_acts',
        help="Digite para filtrar. Selecione todas as partidas do estudo.")

    _c1, _c2 = st.columns(2)
    with _c1:
        _vars_sel = st.multiselect(
            "Variáveis:", _wx.VARIAVEIS,
            default=[_wx.VAR_DIST, _wx.VAR_DIST_REL, _wx.VAR_HSR,
                     _wx.VAR_SPRINT, _wx.VAR_VMAX],
            key='wcs_multi_vars')
        _jan_sel = st.multiselect(
            "Janelas (min):", [1, 3, 5, 10], default=[1, 3, 5],
            key='wcs_multi_jan')
    with _c2:
        _esc_sel = st.multiselect(
            "Escopo do pico:", ['Partida inteira', 'Por período'],
            default=['Partida inteira', 'Por período'], key='wcs_multi_esc',
            help="Partida inteira = períodos encadeados (pior janela do jogo). "
                 "Por período = pico separado de cada tempo.")
        with st.expander("⚙️ Cortes das variáveis"):
            _hsr = st.number_input("HSR ≥ (km/h)", 10.0, 30.0,
                                   _wx.DEFAULT_HSR_KMH, 0.1, key='wcs_multi_hsr')
            _spr = st.number_input("Sprint ≥ (km/h)", 15.0, 40.0,
                                   _wx.DEFAULT_SPRINT_KMH, 0.1, key='wcs_multi_spr')
            _acc = st.number_input("Aceleração ≥ (m/s²)", 0.5, 10.0,
                                   _wx.DEFAULT_ACC_MS2, 0.1, key='wcs_multi_acc')
            _dec = st.number_input("Desaceleração ≤ -(m/s²)", 0.5, 10.0,
                                   _wx.DEFAULT_DEC_MS2, 0.1, key='wcs_multi_dec')

    if not _sel:
        st.info("Selecione ao menos uma atividade acima.")
        return
    if not _vars_sel or not _jan_sel or not _esc_sel:
        st.warning("Escolha ao menos uma variável, uma janela e um escopo.")
        return

    st.caption(f"**{len(_sel)}** atividade(s) · {len(_vars_sel)} variável(is) · "
               f"janelas {_jan_sel} · escopo(s) {len(_esc_sel)}")

    # ── Cálculo (sob demanda — não roda a cada rerun) ────────────────────────
    if st.button("🚀 Carregar atividades e calcular WCS", type="primary",
                 key='wcs_multi_run'):
        _cortes = {'hsr_kmh': _hsr, 'sprint_kmh': _spr,
                   'acc_ms2': _acc, 'dec_ms2': _dec}
        _rows, _log = [], []
        _prog = st.progress(0.0, text="Carregando atividades...")
        for _i, _lbl in enumerate(_sel, 1):
            _aid, _nm, _dt = _meta[_lbl]
            _prog.progress((_i - 1) / len(_sel),
                           text=f"({_i}/{len(_sel)}) {_nm}")
            try:
                _praw = api.get_activity_periods(_aid) or []
                _pids = {}
                for _p in (_praw if isinstance(_praw, list) else []):
                    if _p.get('id'):
                        _pids[_p.get('name') or f"Período {len(_pids)+1}"] = _p['id']
                if not _pids:
                    _pids = {'Atividade Completa': None}
                _carga = carregar_dados(api, _aid, _pids, list(_pids.keys()))
                _res, _sensor = _carga[0], _carga[1]
                if not _sensor:
                    _log.append(f"⚠️ {_nm}: sem dados de sensor — ignorada.")
                    continue
                _novas = _linhas_wcs_atividade(
                    _nm, _dt, _sensor, _res, _vars_sel, _jan_sel, _esc_sel,
                    10.0, _cortes)
                _rows += _novas
                _log.append(f"✅ {_nm}: {len(_novas)} linha(s).")
            except Exception as _e:
                _applog.log_exc(f"export WCS multi — atividade {_nm}")
                _log.append(f"❌ {_nm}: falhou ({type(_e).__name__}).")
        _prog.progress(1.0, text="Concluído.")
        st.session_state[_SS_RES] = (pd.DataFrame(_rows) if _rows
                                     else pd.DataFrame())
        st.session_state[_SS_LOG] = _log

    # ── Resultado ────────────────────────────────────────────────────────────
    _df = st.session_state.get(_SS_RES)
    _log = st.session_state.get(_SS_LOG) or []
    if _log:
        with st.expander(f"📋 Log da carga ({len(_log)} atividade(s))"):
            for _l in _log:
                st.write(_l)

    if _df is None:
        return
    if getattr(_df, 'empty', True):
        st.warning("Nenhuma linha gerada. Verifique se as atividades têm dados "
                   "de sensor e se as janelas cabem na duração.")
        return

    st.success(f"**{len(_df)}** linhas · {_df['Atleta'].nunique()} atleta(s) · "
               f"{_df['Atividade'].nunique()} atividade(s)")
    st.dataframe(_df, use_container_width=True, height=380, hide_index=True)

    st.download_button(
        "📥 Baixar CSV (tidy — jamovi/R/SPSS)",
        _df.to_csv(index=False).encode('utf-8'),
        "wcs_multi_atividades_tidy.csv", mime='text/csv',
        key='wcs_multi_dl', type="primary")
    st.caption(
        "Formato longo: 1 linha por atividade × atleta × escopo × variável × "
        "janela. No jamovi, use **Variavel**, **Janela_min**, **Escopo**, "
        "**Equipe** e **Posicao** como fatores e **Valor** como variável dependente.")
