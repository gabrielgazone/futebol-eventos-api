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
_SS_ACTS = '_wcs_multi_acts_calc'     # atividades que geraram o resultado atual

# Colunas que identificam uma linha unicamente (usadas para deduplicar ao
# acumular lotes de atividades).
_CHAVE_LINHA = ['Atividade', 'Atleta', 'Escopo', 'Variavel', 'Janela_min']

# Colunas que o resultado ATUAL precisa ter. Um resultado guardado na sessão de
# antes de um deploy pode não ter as colunas novas (ex.: minutos) — nesse caso é
# descartado, em vez de ser exibido incompleto e parecer um bug.
_COLS_ESPERADAS = ['Atividade', 'Data', 'Equipe', 'Atleta', 'Posicao',
                   'Minutos_OpenField', 'Minutos_sensor', 'Escopo',
                   'Variavel', 'Janela_min', 'Valor']


def _fmt_data_br(valor) -> str:
    """Formata a data da atividade para DD/MM/AAAA (aceita epoch, ISO ou str)."""
    from datetime import datetime as _dt
    if valor is None or valor == '':
        return ''
    try:
        fv = float(valor)
        if fv > 1e8:
            return _dt.fromtimestamp(fv).strftime('%d/%m/%Y')
    except (TypeError, ValueError):
        pass
    s = str(valor)
    try:
        return _dt.fromisoformat(s.replace('Z', '').split('.')[0]).strftime('%d/%m/%Y')
    except Exception:
        _applog.log_debug_exc()
    for _f in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'):
        try:
            return _dt.strptime(s[:10], _f).strftime('%d/%m/%Y')
        except ValueError:
            continue
    return s[:10]


# Nomes candidatos do parâmetro de duração no /stats (varia por conta/versão).
_DUR_PARAMS = ('total_duration', 'duration', 'total_time', 'time_on_field',
               'athlete_duration', 'total_session_time')


def _para_minutos(valor):
    """Converte a duração do /stats para MINUTOS.

    A API pode devolver segundos ou minutos dependendo do parâmetro; usa uma
    heurística conservadora: > 300 é tratado como segundos (300 min = 5 h seria
    implausível para uma partida), senão já está em minutos.
    """
    try:
        _v = float(valor)
    except (TypeError, ValueError):
        return None
    if _v <= 0:
        return None
    return round(_v / 60.0, 1) if _v > 300 else round(_v, 1)


def _minutos_openfield(api, epoch_ini, epoch_fim):
    """{atleta: minutos} OFICIAIS do OpenField via POST /stats para a janela da
    atividade. Retorna também o nome do parâmetro que respondeu (para o log).

    O nome do parâmetro de duração varia por conta, então tentamos os candidatos
    de `_DUR_PARAMS`. Se nenhum responder, devolve ({}, None) e o export cai na
    duração derivada do sensor (coluna separada), sem inventar valor.
    """
    if not epoch_ini:
        return {}, None
    _payload_base = {
        "group_by": ["athlete"],
        "source": "cached_stats",
        "start_time": int(epoch_ini),
        "end_time": int(epoch_fim or (epoch_ini + 86400)),
    }
    for _par in _DUR_PARAMS:
        try:
            _r = api.get_stats(dict(_payload_base, parameters=[_par]))
        except Exception:
            _applog.log_debug_exc()
            continue
        _rows = _r if isinstance(_r, list) else (_r or {}).get('data', [])
        _out = {}
        for _d in (_rows or []):
            if not isinstance(_d, dict):
                continue
            _nm = str(_d.get('athlete') or _d.get('athlete_name')
                      or _d.get('name') or '')
            _p = _d.get('parameters') or {}
            _val = (_p.get(_par) if isinstance(_p, dict) else None)
            if _val is None:
                _val = _d.get(_par)
            _mm = _para_minutos(_val)
            if _nm and _mm:
                _out[_nm] = _mm
        if _out:
            return _out, _par
    return {}, None


def _mapa_posicoes(api):
    """{position_id: nome} direto da API (/positions). Cacheado no _api_fetch."""
    _out = {}
    try:
        for _p in (api.get_positions() or []):
            if _p.get('id'):
                _out[_p['id']] = _p.get('name') or ''
    except Exception:
        _applog.log_debug_exc()
    return _out


def _mapa_equipes(api):
    """{athlete_id: nome da equipe} direto da API (/teams + /teams/x/athletes).
    Independente do filtro da barra lateral."""
    _out = {}
    try:
        for _t in (api.get_teams() or []):
            _tid, _tnm = _t.get('id'), _t.get('name') or ''
            if not _tid:
                continue
            try:
                for _a in (api.get_team_athletes(_tid) or []):
                    if _a.get('id'):
                        _out[_a['id']] = _tnm
            except Exception:
                _applog.log_debug_exc()
    except Exception:
        _applog.log_debug_exc()
    return _out


def _atletas_da_atividade(api, activity_id, pos_map, eq_map):
    """Elenco da atividade DIRETO da API, com posição resolvida via /positions.

    Retorna DataFrame no mesmo formato de `atletas_filtrados` (id, nome, posicao,
    equipe) — é o que `carregar_dados` consome. Buscar por atividade (em vez de
    usar a seleção da barra lateral) garante o elenco correto de CADA jogo e a
    posição vinda da API, mesmo em clubes diferentes.
    """
    try:
        _raw = api.get_activity_athletes(activity_id) or []
    except Exception:
        _applog.log_exc(f"elenco da atividade {activity_id}")
        return pd.DataFrame()
    _rows = []
    for _a in (_raw if isinstance(_raw, list) else []):
        _nm = f"{_a.get('first_name', '')} {_a.get('last_name', '')}".strip()
        if not _nm:
            _nm = _a.get('name') or ''
        if not _nm:
            continue
        _aid = _a.get('id')
        _rows.append({
            'id': _aid,
            'nome': _nm,
            'posicao': pos_map.get(_a.get('position_id'), '') or '',
            'equipe': eq_map.get(_aid, '') or '',
        })
    return pd.DataFrame(_rows)


def pivotar_variaveis(df):
    """Converte o formato longo em VARIÁVEIS COMO COLUNAS (atletas nas linhas).

    Índice: Atividade, Data, Equipe, Atleta, Posicao, Minutos*, Escopo,
    Janela_min. Uma coluna por variável, na ordem de `wcs_export.VARIAVEIS`.
    """
    if df is None or getattr(df, 'empty', True):
        return df
    _idx = [_c for _c in ('Atividade', 'Data', 'Equipe', 'Atleta', 'Posicao',
                          'Minutos_OpenField', 'Minutos_sensor', 'Escopo',
                          'Janela_min') if _c in df.columns]
    _p = df.pivot_table(index=_idx, columns='Variavel', values='Valor',
                        aggfunc='first').reset_index()
    _p.columns.name = None
    # Ordena as colunas de variáveis como em VARIAVEIS (o resto vem antes)
    _vars_ord = [_v for _v in _wx.VARIAVEIS if _v in _p.columns]
    _outras = [_c for _c in _p.columns if _c not in _vars_ord]
    return _p[_outras + _vars_ord]


def _linhas_wcs_atividade(act_nome, act_data, dados_sensor, info_atl,
                          variaveis, janelas, escopos, hz, cortes,
                          min_map=None):
    """Linhas tidy de WCS para UMA atividade já carregada.

    info_atl: {nome: (posicao, equipe)} vindo da API (ver _atletas_da_atividade).
    min_map: {nome: minutos} oficiais do OpenField (/stats). A duração derivada
    do sensor entra numa coluna separada, para conferência.
    """
    _rows = []
    _info = info_atl or {}
    _mins = min_map or {}
    _atletas = sorted({_a for _p in dados_sensor.values() for _a in _p.keys()})

    for _atl in _atletas:
        _pos, _eq = _info.get(_atl, ('', ''))
        # Duração pelo sinal do sensor (10 Hz): amostras únicas de todos os
        # períodos — referência para conferir o Minutos oficial.
        _n_amostras = sum(len(_p.get(_atl, [])) for _p in dados_sensor.values())
        _min_sensor = round(_n_amostras / hz / 60.0, 1) if hz > 0 else 0.0
        _min_of = _mins.get(_atl)

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
                    'Minutos_OpenField': _min_of if _min_of else '',
                    'Minutos_sensor': _min_sensor,
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
        "(1/3/5 min) — pronto para jamovi, R ou SPSS. O cálculo usa o **mesmo "
        "método da aba WCS**. **Você define os limiares** de HSR, Sprint e das "
        "bandas de aceleração/desaceleração (B2/B3) em *Cortes das variáveis*.")

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
        # A API devolve start_time como epoch Unix — formata para DD/MM/AAAA
        # (antes o corte [:10] deixava o número cru na coluna Data).
        _dt = _fmt_data_br(_r.get('data'))
        _lbl = f"{_nm} — {_dt}" if _dt else _nm
        _opts.append(_lbl)
        _meta[_lbl] = (_r.get('id'), _nm, _dt, _r.get('data'))

    _sel = st.multiselect(
        "Atividades a exportar (busque por nome/data):", _opts,
        key='wcs_multi_acts',
        help="Digite para filtrar. Selecione todas as partidas do estudo.")

    _c1, _c2 = st.columns(2)
    with _c1:
        _vars_sel = st.multiselect(
            "Variáveis:", _wx.VARIAVEIS, default=list(_wx.VARIAVEIS),
            key='wcs_multi_vars',
            help="Acelerações/desacelerações são contadas por banda "
                 "(B2, B3 e B2+ = B2+B3), classificadas pelo pico da ação.")
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
            st.caption("Bandas de aceleração (m/s²) — B2+ soma B2 e B3:")
            _a2i = st.number_input("Acc B2: de", 0.5, 10.0,
                                   _wx.DEFAULT_ACC_B2[0], 0.1, key='wcs_m_a2i')
            _a2f = st.number_input("Acc B2: até (= início do B3)", 0.5, 12.0,
                                   _wx.DEFAULT_ACC_B2[1], 0.1, key='wcs_m_a2f')
            _a3f = st.number_input("Acc B3: até", 1.0, 20.0,
                                   _wx.DEFAULT_ACC_B3[1], 0.5, key='wcs_m_a3f')
            st.caption("Desaceleração — informe a MAGNITUDE (positiva):")
            _d2i = st.number_input("Dec B2: de", 0.5, 10.0,
                                   abs(_wx.DEFAULT_DEC_B2[1]), 0.1, key='wcs_m_d2i')
            _d2f = st.number_input("Dec B2: até (= início do B3)", 0.5, 12.0,
                                   abs(_wx.DEFAULT_DEC_B2[0]), 0.1, key='wcs_m_d2f')
            _d3f = st.number_input("Dec B3: até", 1.0, 20.0,
                                   abs(_wx.DEFAULT_DEC_B3[0]), 0.5, key='wcs_m_d3f')

    if not _sel:
        st.info("Selecione ao menos uma atividade acima.")
        return
    if not _vars_sel or not _jan_sel or not _esc_sel:
        st.warning("Escolha ao menos uma variável, uma janela e um escopo.")
        return

    st.caption(f"**{len(_sel)}** atividade(s) selecionada(s) · {len(_vars_sel)} "
               f"variável(is) · janelas {_jan_sel} · escopo(s) {len(_esc_sel)}")

    _acumular = st.checkbox(
        "➕ Acumular com o resultado já calculado", value=False,
        key='wcs_multi_acum',
        help="Marque para SOMAR estas atividades ao que já foi calculado, em vez "
             "de substituir. Útil para exportar muitas atividades em lotes "
             "(ex.: 30 partidas em 3 lotes de 10) e baixar um CSV único no fim. "
             "Linhas repetidas da mesma atividade são substituídas, não duplicadas.")

    # ── Cálculo (sob demanda — não roda a cada rerun) ────────────────────────
    if st.button("🚀 Carregar atividades e calcular WCS", type="primary",
                 key='wcs_multi_run'):
        # Desaceleração: a UI pede magnitude; o motor espera bandas negativas
        # ordenadas (min, max), ex.: B2 = (-4, -3) e B3 = (-10, -4).
        _cortes = {'hsr_kmh': _hsr, 'sprint_kmh': _spr,
                   'acc_b2': (_a2i, _a2f), 'acc_b3': (_a2f, _a3f),
                   'dec_b2': (-_d2f, -_d2i), 'dec_b3': (-_d3f, -_d2f)}
        _rows, _log = [], []
        _prog = st.progress(0.0, text="Buscando posições e equipes na API...")
        _pos_map = _mapa_posicoes(api)
        _eq_map = _mapa_equipes(api)

        # `carregar_dados` lê o elenco destas duas chaves (definidas pela barra
        # lateral). Trocamos por atividade — para carregar o elenco CORRETO de
        # cada jogo, com posição da API — e restauramos ao final, para não
        # perturbar o resto do app.
        _bkp_sel = st.session_state.get('atletas_sel')
        _bkp_filt = st.session_state.get('atletas_filtrados')
        try:
            for _i, _lbl in enumerate(_sel, 1):
                _aid, _nm, _dt, _epoch = _meta[_lbl]
                _prog.progress((_i - 1) / len(_sel),
                               text=f"({_i}/{len(_sel)}) {_nm}")
                try:
                    _df_atl = _atletas_da_atividade(api, _aid, _pos_map, _eq_map)
                    if _df_atl.empty:
                        _log.append(f"⚠️ {_nm}: API não retornou elenco — ignorada.")
                        continue
                    _info_atl = {_r['nome']: (_r['posicao'], _r['equipe'])
                                 for _, _r in _df_atl.iterrows()}
                    st.session_state['atletas_filtrados'] = _df_atl
                    st.session_state['atletas_sel'] = _df_atl['nome'].tolist()

                    _praw = api.get_activity_periods(_aid) or []
                    _pids = {}
                    for _p in (_praw if isinstance(_praw, list) else []):
                        if _p.get('id'):
                            _pids[_p.get('name') or f"Período {len(_pids)+1}"] = _p['id']
                    if not _pids:
                        _pids = {'Atividade Completa': None}
                    _carga = carregar_dados(api, _aid, _pids, list(_pids.keys()))
                    _sensor = _carga[1]
                    if not _sensor:
                        _log.append(f"⚠️ {_nm}: sem dados de sensor — ignorada.")
                        continue
                    # Minutos OFICIAIS do OpenField (/stats) para esta atividade
                    _ep = None
                    try:
                        _ep = float(_epoch) if _epoch else None
                    except (TypeError, ValueError):
                        _ep = None
                    _min_map, _par_dur = _minutos_openfield(
                        api, _ep, (_ep + 86400) if _ep else None)
                    _novas = _linhas_wcs_atividade(
                        _nm, _dt, _sensor, _info_atl, _vars_sel, _jan_sel,
                        _esc_sel, 10.0, _cortes, min_map=_min_map)
                    _rows += _novas
                    _n_sem_pos = sum(1 for _v in _info_atl.values() if not _v[0])
                    _log.append(
                        f"✅ {_nm}: {len(_novas)} linha(s), "
                        f"{len(_info_atl)} atleta(s) no elenco"
                        + (f" — ⚠️ {_n_sem_pos} sem posição na API"
                           if _n_sem_pos else "")
                        + (f" · Minutos via /stats ('{_par_dur}'): "
                           f"{len(_min_map)} atleta(s)" if _par_dur else
                           " · ⚠️ /stats não retornou duração — use "
                           "Minutos_sensor"))
                except Exception as _e:
                    _applog.log_exc(f"export WCS multi — atividade {_nm}")
                    _log.append(f"❌ {_nm}: falhou ({type(_e).__name__}).")
        finally:
            # Restaura a seleção da barra lateral (sempre, mesmo em erro)
            if _bkp_sel is not None:
                st.session_state['atletas_sel'] = _bkp_sel
            if _bkp_filt is not None:
                st.session_state['atletas_filtrados'] = _bkp_filt
        _prog.progress(1.0, text="Concluído.")
        _novo_df = pd.DataFrame(_rows) if _rows else pd.DataFrame()
        _ant_df = st.session_state.get(_SS_RES)
        _ant_acts = list(st.session_state.get(_SS_ACTS) or [])
        if (_acumular and _ant_df is not None
                and not getattr(_ant_df, 'empty', True) and not _novo_df.empty):
            # Junta lotes e remove repetições da MESMA atividade (recálculo)
            _novo_df = (pd.concat([_ant_df, _novo_df], ignore_index=True)
                        .drop_duplicates(subset=_CHAVE_LINHA, keep='last')
                        .reset_index(drop=True))
            _acts_final = _ant_acts + [_a for _a in _sel if _a not in _ant_acts]
        else:
            _acts_final = list(_sel)
        st.session_state[_SS_RES] = _novo_df
        st.session_state[_SS_LOG] = _log
        st.session_state[_SS_ACTS] = _acts_final

    # ── Resultado ────────────────────────────────────────────────────────────
    _df = st.session_state.get(_SS_RES)
    _log = st.session_state.get(_SS_LOG) or []
    if _log:
        with st.expander(f"📋 Log da carga ({len(_log)} atividade(s))"):
            for _l in _log:
                st.write(_l)

    if _df is None:
        return

    # Resultado de ANTES de um deploy pode não ter as colunas novas (minutos,
    # bandas). Nesse caso descarta — melhor recalcular do que exibir incompleto.
    if (not getattr(_df, 'empty', True)
            and any(_c not in _df.columns for _c in _COLS_ESPERADAS)):
        _falta = [_c for _c in _COLS_ESPERADAS if _c not in _df.columns]
        for _k in (_SS_RES, _SS_LOG, _SS_ACTS):
            st.session_state.pop(_k, None)
        st.warning(
            "O resultado guardado é de uma versão anterior do app "
            f"(sem: {', '.join(_falta)}) e foi descartado. Clique em "
            "**🚀 Carregar atividades e calcular WCS** para recalcular.")
        return

    if getattr(_df, 'empty', True):
        st.warning("Nenhuma linha gerada. Verifique se as atividades têm dados "
                   "de sensor e se as janelas cabem na duração.")
        return

    # Aviso de resultado DESATUALIZADO: a seleção mudou depois do cálculo.
    _acts_calc = list(st.session_state.get(_SS_ACTS) or [])
    if set(_acts_calc) != set(_sel):
        _faltam = [_a for _a in _sel if _a not in _acts_calc]
        st.warning(
            f"⚠️ Este resultado é de **{len(_acts_calc)}** atividade(s), mas você "
            f"tem **{len(_sel)}** selecionada(s)"
            + (f" — falta calcular: {', '.join(_faltam[:5])}"
               + ("…" if len(_faltam) > 5 else "") if _faltam else "")
            + ". Clique em **🚀 Carregar atividades e calcular WCS** para atualizar.")

    st.success(f"**{len(_df)}** linhas · {_df['Atleta'].nunique()} atleta(s) · "
               f"{_df['Atividade'].nunique()} atividade(s) calculada(s)")

    _fmt_out = st.radio(
        "Formato da tabela:",
        ["Variáveis em colunas", "Longo (tidy)"],
        horizontal=True, key='wcs_multi_fmt',
        help="Variáveis em colunas: 1 linha por atleta (× atividade, escopo e "
             "janela), uma coluna por variável. Longo: 1 linha por variável.")

    if _fmt_out == "Variáveis em colunas":
        _dfx = pivotar_variaveis(_df)
        _nome_csv = "wcs_multi_atividades_variaveis_em_colunas.csv"
        _legenda = (
            "Cada linha é um **atleta** numa atividade/escopo/janela; cada "
            "**variável é uma coluna**. Filtre `Janela_min` (1/3/5) e `Escopo` "
            "no jamovi para a análise desejada.")
    else:
        _dfx = _df
        _nome_csv = "wcs_multi_atividades_tidy.csv"
        _legenda = (
            "Formato longo: 1 linha por atividade × atleta × escopo × variável × "
            "janela. Use **Variavel**, **Janela_min**, **Escopo**, **Equipe** e "
            "**Posicao** como fatores e **Valor** como variável dependente.")

    st.dataframe(_dfx, use_container_width=True, height=380, hide_index=True)
    st.download_button(
        "📥 Baixar CSV (jamovi/R/SPSS)",
        _dfx.to_csv(index=False).encode('utf-8'),
        _nome_csv, mime='text/csv', key='wcs_multi_dl', type="primary")
    st.caption(_legenda)
