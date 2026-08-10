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
                   'Minutos', 'Periodos_jogados', 'Escopo', 'Variavel',
                   'Janela_min', 'Valor', 'Ocorrencias_pct']


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


def _nome_do_atleta(_a):
    """Nome como o app monta a chave do atleta (first + last, senão name)."""
    _nm = f"{_a.get('first_name', '')} {_a.get('last_name', '')}".strip()
    return _nm or str(_a.get('name') or '')


def ler_atividade_profunda(resp):
    """Extrai duração e participação de GET /activities/{id}?include=all.

    Fonte AUTORITATIVA (docs Catapult v6): cada período traz `start_time`/
    `end_time` em epoch (+ `start_centiseconds`/`end_centiseconds`) e a lista de
    atletas do período — exatamente o que o OpenField usa para o export de cada
    tempo. Duração = end − start; participação = presença na lista do período.

    Retorna (duracoes, participantes, info_atl):
      duracoes     {nome_do_periodo: minutos}
      participantes{nome_do_periodo: set(nomes) | None}
      info_atl     {nome: (posicao, equipe)}
    """
    _obj = resp
    if isinstance(_obj, list):
        _obj = _obj[0] if _obj else {}
    if not isinstance(_obj, dict):
        return {}, {}, {}
    _obj = _obj.get('data', _obj) if isinstance(_obj.get('data'), dict) else _obj

    # Posição/equipe dos atletas da atividade
    _info = {}
    _equipe_ativ = ''
    _teams = _obj.get('teams') or []
    if isinstance(_teams, list) and _teams:
        _t0 = _teams[0]
        if isinstance(_t0, dict):
            _equipe_ativ = str(_t0.get('name') or '')
    for _a in (_obj.get('athletes') or []):
        if not isinstance(_a, dict):
            continue
        _nm = _nome_do_atleta(_a)
        if not _nm:
            continue
        _pos = str(_a.get('position_name') or _a.get('position') or '')
        if isinstance(_a.get('position'), dict):
            _pos = str(_a['position'].get('name') or '')
        _info[_nm] = (_pos, _equipe_ativ)

    # id → nome, para resolver listas de período que só trazem athlete_id
    _id2nome = {}
    for _a in (_obj.get('athletes') or []):
        if isinstance(_a, dict) and _a.get('id'):
            _id2nome[str(_a['id'])] = _nome_do_atleta(_a)

    _duracoes, _partic = {}, {}
    for _p in (_obj.get('periods') or []):
        if not isinstance(_p, dict):
            continue
        _pnm = str(_p.get('name') or f"Período {len(_duracoes) + 1}")
        try:
            _ini = (float(_p.get('start_time') or 0)
                    + float(_p.get('start_centiseconds') or 0) / 100.0)
            _fim = (float(_p.get('end_time') or 0)
                    + float(_p.get('end_centiseconds') or 0) / 100.0)
        except (TypeError, ValueError):
            _ini = _fim = 0.0
        _duracoes[_pnm] = (round((_fim - _ini) / 60.0, 5)
                           if _fim > _ini > 0 else 0.0)

        _nomes = set()
        for _chave in ('athletes', 'period_athletes'):
            for _pa in (_p.get(_chave) or []):
                if not isinstance(_pa, dict):
                    continue
                _nm = _nome_do_atleta(_pa)
                if not _nm:
                    _aid = _pa.get('athlete_id') or _pa.get('id')
                    _nm = _id2nome.get(str(_aid), '') if _aid else ''
                if _nm:
                    _nomes.add(_nm)
        _partic[_pnm] = _nomes or None
    return _duracoes, _partic, _info


def duracoes_periodos(dados_sensor, hz=10.0):
    """{período: duração em minutos} pela janela de tempo do período.

    Replica o "Minutos" do OpenField, que é UNIFORME por período (ex.: 51,40233
    no 1º tempo e 55,7 no 2º para todos os atletas) — ou seja, é a duração do
    período, não o tempo de bola rolando de cada atleta. Usa a união dos
    timestamps de todos os atletas do período.
    """
    _out = {}
    for _pnm, _pdados in (dados_sensor or {}).items():
        _mn = _mx = None
        for _pts in (_pdados or {}).values():
            if not _pts:
                continue
            for _p in (_pts[0], _pts[-1]):
                try:
                    _t = float(_p.get('ts') or 0) + float(_p.get('cs') or 0) / 100.0
                except (TypeError, ValueError):
                    continue
                if _t <= 0:
                    continue
                if _mn is None or _t < _mn:
                    _mn = _t
                if _mx is None or _t > _mx:
                    _mx = _t
        _out[_pnm] = (round((_mx - _mn) / 60.0, 5)
                      if (_mn is not None and _mx is not None and _mx > _mn)
                      else 0.0)
    return _out


def _participantes_por_periodo(api, period_ids, id_para_nome):
    """{período: set(nomes)} pela lista OFICIAL de participantes da API
    (/periods/{id}/athletes) — a mesma fonte que o OpenField usa para decidir
    quem entra no export de cada tempo.

    Valor None para um período = a API não informou; nesse caso o chamador NÃO
    filtra por lista (e avisa), para não descartar dado por falha de rede.
    """
    _out = {}
    for _pnm, _pid in (period_ids or {}).items():
        if not _pid:
            _out[_pnm] = None
            continue
        try:
            _resp = api.get_athletes_in_period(_pid)
        except Exception:
            _applog.log_debug_exc()
            _out[_pnm] = None
            continue
        _lst = (_resp if isinstance(_resp, list)
                else (_resp or {}).get('data', (_resp or {}).get('items', [])))
        if not _lst:
            _out[_pnm] = None
            continue
        _nomes = set()
        for _a in _lst:
            if not isinstance(_a, dict):
                continue
            _aid = _a.get('id') or _a.get('athlete_id')
            _nm = id_para_nome.get(str(_aid)) if _aid else None
            if not _nm:                        # nome direto, se vier no payload
                _nm = (f"{_a.get('first_name', '')} "
                       f"{_a.get('last_name', '')}").strip() or _a.get('name')
            if _nm:
                _nomes.add(str(_nm))
        _out[_pnm] = _nomes or None
    return _out


# Piso de participação (m/min): usado SÓ quando a API não informa o elenco do
# período. No export do OpenField, reservas com o dispositivo ligado ficam em
# ~2–3 m/min (ex.: 136 m em 51 min), enquanto quem entra em campo passa de
# 60 m/min (ex.: 5187 m em 55,7 min). 25 m/min separa os dois casos com folga e
# é interpretável — melhor que uma fração da mediana, que o próprio reserva
# distorce quando há muitos no banco.
_PISO_M_POR_MIN = 25.0


def _dist_total_m(sensor_points, hz=10.0):
    """Distância total (m) de uma série do sensor — para o piso de participação."""
    _tot = 0.0
    _prev = None
    for _p in (sensor_points or []):
        try:
            _v = float(_p.get('v') or 0.0)
        except (TypeError, ValueError):
            _v = 0.0
        if _prev is not None:
            _tot += ((_prev + _v) / 2.0) / hz
        _prev = _v
    return _tot


def participou_do_periodo(nome, periodo, dados_sensor, participantes, hz=10.0,
                          duracoes=None):
    """Se o atleta PARTICIPOU do período (critério do OpenField).

    1) Lista oficial da API, quando disponível — critério primário.
    2) Senão, piso de intensidade em m/min (_PISO_M_POR_MIN): o dispositivo
       ligado no banco fica uma ordem de grandeza abaixo de quem entra em campo.
    Evita gerar linhas que sugerem presença num tempo em que o atleta não jogou.
    """
    _sp = (dados_sensor.get(periodo) or {}).get(nome) or []
    if not _sp:
        return False
    _oficial = (participantes or {}).get(periodo)
    if _oficial is not None:
        return nome in _oficial
    _dur_min = (duracoes or {}).get(periodo)
    if not _dur_min:
        _dur_min = len(_sp) / hz / 60.0 if hz > 0 else 0.0
    if _dur_min <= 0:
        return False
    return (_dist_total_m(_sp, hz) / _dur_min) >= _PISO_M_POR_MIN


def _cols_id(df):
    """Colunas identificadoras presentes (na ordem de exibição)."""
    return [_c for _c in ('Atividade', 'Data', 'Equipe', 'Atleta', 'Posicao',
                          'Minutos', 'Periodos_jogados',
                          'Escopo') if _c in df.columns]


def pivotar_variaveis_x_janelas(df, pct=90):
    """UMA linha por atleta (× atividade/escopo) e uma coluna por
    VARIÁVEL × JANELA — ex.: "Distância (m) 1min", "Distância (m) 3min", ...

    Evita as 3 linhas por atleta (uma por janela): as janelas viram colunas,
    agrupadas por variável e em ordem crescente de janela.
    """
    if df is None or getattr(df, 'empty', True):
        return df
    _idx = _cols_id(df)
    _tem_ocor = 'Ocorrencias_pct' in df.columns
    _vals = ['Valor'] + (['Ocorrencias_pct'] if _tem_ocor else [])
    _p = df.pivot_table(index=_idx, columns=['Variavel', 'Janela_min'],
                        values=_vals, aggfunc='first').reset_index()
    # Achata o cabeçalho: ('Valor', 'Distância (m)', 1) -> 'Distância (m) 1min';
    # ('Ocorrencias_pct', ...) -> 'Distância (m) ≥90% 1min'
    _novos = []
    for _c in _p.columns:
        if isinstance(_c, tuple):
            _partes = [_x for _x in _c if _x != '']
            if len(_partes) >= 3:
                _tipo, _var, _jan = _partes[0], _partes[1], _partes[2]
                _suf = f' ≥{int(pct)}%' if _tipo == 'Ocorrencias_pct' else ''
                _novos.append(f"{_var}{_suf} {int(_jan)}min")
            else:
                _novos.append(str(_partes[-1] if _partes else _c[0]))
        else:
            _novos.append(str(_c))
    _p.columns = _novos
    # Ordena: identificadores; depois, por variável, os picos 1/3/5 e em
    # seguida as ocorrências ≥90% 1/3/5.
    _janelas = sorted({int(_j) for _j in df['Janela_min'].unique()})
    _ordem_vars = []
    for _v in _wx.VARIAVEIS:
        for _suf in ('', f' ≥{int(pct)}%'):
            for _j in _janelas:
                _nome = f"{_v}{_suf} {_j}min"
                if _nome in _p.columns:
                    _ordem_vars.append(_nome)
    _outras = [_c for _c in _p.columns if _c not in _ordem_vars]
    return _p[_outras + _ordem_vars]


def pivotar_variaveis(df, pct=90):
    """Converte o formato longo em VARIÁVEIS COMO COLUNAS (atletas nas linhas).

    Índice: Atividade, Data, Equipe, Atleta, Posicao, Minutos*, Escopo,
    Janela_min. Uma coluna por variável, na ordem de `wcs_export.VARIAVEIS`.
    """
    if df is None or getattr(df, 'empty', True):
        return df
    _idx = [_c for _c in ('Atividade', 'Data', 'Equipe', 'Atleta', 'Posicao',
                          'Minutos', 'Periodos_jogados',
                          'Escopo', 'Janela_min') if _c in df.columns]
    _tem_ocor = 'Ocorrencias_pct' in df.columns
    _vals = ['Valor'] + (['Ocorrencias_pct'] if _tem_ocor else [])
    _p = df.pivot_table(index=_idx, columns='Variavel', values=_vals,
                        aggfunc='first').reset_index()
    _novos = []
    for _c in _p.columns:
        if isinstance(_c, tuple):
            _partes = [_x for _x in _c if _x != '']
            if len(_partes) >= 2:
                _tipo, _var = _partes[0], _partes[1]
                _novos.append(f"{_var} ≥{int(pct)}%"
                              if _tipo == 'Ocorrencias_pct' else str(_var))
            else:
                _novos.append(str(_partes[-1] if _partes else _c[0]))
        else:
            _novos.append(str(_c))
    _p.columns = _novos
    # Ordena: identificadores; por variável, o pico e depois as ocorrências
    _vars_ord = []
    for _v in _wx.VARIAVEIS:
        for _suf in ('', f' ≥{int(pct)}%'):
            if f"{_v}{_suf}" in _p.columns:
                _vars_ord.append(f"{_v}{_suf}")
    _outras = [_c for _c in _p.columns if _c not in _vars_ord]
    return _p[_outras + _vars_ord]


def _linhas_wcs_atividade(act_nome, act_data, dados_sensor, info_atl,
                          variaveis, janelas, escopos, hz, cortes,
                          participantes=None, duracoes=None, pct=0.90):
    """Linhas tidy de WCS para UMA atividade já carregada.

    info_atl: {nome: (posicao, equipe)} vindo da API (ver _atletas_da_atividade).
    participantes: {período: set(nomes)} da API (None por período = desconhecido).
    duracoes: {período: minutos} (ver duracoes_periodos).

    Só gera linhas dos períodos em que o atleta PARTICIPOU — sem isso, um atleta
    que ficou no banco no 1º tempo (dispositivo ligado) aparecia como se tivesse
    jogado. `Minutos` replica o OpenField: soma das durações dos períodos
    participados (no escopo de período, a duração daquele período).
    Retorna (linhas, n_pares_excluidos).
    """
    _rows = []
    _info = info_atl or {}
    _dur = duracoes or {}
    _atletas = sorted({_a for _p in dados_sensor.values() for _a in _p.keys()})
    _excluidos = 0

    for _atl in _atletas:
        _pos, _eq = _info.get(_atl, ('', ''))

        # Períodos em que ESTE atleta participou (critério do OpenField)
        _peri_ok = []
        for _pnm in dados_sensor.keys():
            if participou_do_periodo(_atl, _pnm, dados_sensor, participantes,
                                     hz, _dur):
                _peri_ok.append(_pnm)
            elif (dados_sensor.get(_pnm) or {}).get(_atl):
                _excluidos += 1
        if not _peri_ok:
            continue                       # não jogou nada nesta atividade

        # Minutos do atleta = soma das durações dos períodos participados
        _min_total = round(sum(_dur.get(_p, 0.0) for _p in _peri_ok), 5)

        _escopo_series = []
        if 'Partida inteira' in escopos:
            # Encadeia SÓ os períodos participados
            _ds_ok = {_p: dados_sensor[_p] for _p in _peri_ok}
            _sp = (combinar_periodos_continuo(_ds_ok, _atl)
                   if len(_ds_ok) > 1
                   else next(iter(_ds_ok.values()), {}).get(_atl, []))
            if _sp:
                _escopo_series.append(('Partida inteira', _sp, _min_total))
        if 'Por período' in escopos:
            for _pnm in _peri_ok:
                _sp_p = (dados_sensor.get(_pnm) or {}).get(_atl) or []
                if _sp_p:
                    _escopo_series.append((_pnm, _sp_p, _dur.get(_pnm, 0.0)))

        for _escopo, _sp, _min_escopo in _escopo_series:
            _picos = _wx.calcular_wcs(_sp, variaveis, janelas, hz, **cortes)
            _ocor = _wx.calcular_ocorrencias(_sp, variaveis, janelas, hz,
                                             pct, **cortes)
            for (_var, _wmin), _val in _picos.items():
                _rows.append({
                    'Atividade': act_nome,
                    'Data': act_data,
                    'Equipe': _eq,
                    'Atleta': _atl,
                    'Posicao': _pos,
                    'Minutos': _min_escopo,
                    'Periodos_jogados': len(_peri_ok),
                    'Escopo': _escopo,
                    'Variavel': _var,
                    'Janela_min': _wmin,
                    'Valor': _val,
                    'Ocorrencias_pct': _ocor.get((_var, _wmin), 0),
                })
    return _rows, _excluidos


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
        _pct_ocor = st.number_input(
            "Limiar de ocorrências (% do máximo do atleta)", 50, 100, 90, 5,
            key='wcs_multi_pct',
            help="Para cada variável e janela, conta quantos esforços DISTINTOS "
                 "(não-sobrepostos) o atleta fez acima desta fração do SEU "
                 "próprio pico. 90% = repetições do próprio pior cenário.")
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
                    # Duração e participação AUTORITATIVAS: 1 chamada
                    # /activities/{id}?include=all (períodos com start/end_time
                    # + atletas de cada período). Fallback: /periods/{id}/
                    # athletes + timestamps do sensor.
                    _duracs, _partic, _info_deep = {}, {}, {}
                    _fonte_min = 'sensor'
                    try:
                        _deep = api.get_deep_activity(_aid)
                        if _deep:
                            _duracs, _partic, _info_deep = ler_atividade_profunda(_deep)
                            if _duracs:
                                _fonte_min = 'API (deep activity)'
                    except Exception:
                        _applog.log_debug_exc()
                    if _info_deep:                 # posição/equipe da mesma fonte
                        for _k, _v in _info_deep.items():
                            _info_atl.setdefault(_k, _v)
                    if not _duracs:
                        _id2nome = {str(_r['id']): _r['nome']
                                    for _, _r in _df_atl.iterrows() if _r.get('id')}
                        _partic = _participantes_por_periodo(api, _pids, _id2nome)
                        _duracs = duracoes_periodos(_sensor, 10.0)
                    else:
                        # Preenche períodos que o sensor tem e a API não nomeou
                        for _pn_s in _sensor.keys():
                            _duracs.setdefault(
                                _pn_s, duracoes_periodos(
                                    {_pn_s: _sensor[_pn_s]}, 10.0)[_pn_s])
                            _partic.setdefault(_pn_s, None)
                    _sem_lista = [_p for _p in _sensor.keys()
                                  if _partic.get(_p) is None]
                    _novas, _excl = _linhas_wcs_atividade(
                        _nm, _dt, _sensor, _info_atl, _vars_sel, _jan_sel,
                        _esc_sel, 10.0, _cortes,
                        participantes=_partic, duracoes=_duracs,
                        pct=_pct_ocor / 100.0)
                    _rows += _novas
                    _n_sem_pos = sum(1 for _v in _info_atl.values() if not _v[0])
                    _log.append(
                        f"✅ {_nm}: {len(_novas)} linha(s), "
                        f"{len(_info_atl)} atleta(s) no elenco"
                        + (f" — ⚠️ {_n_sem_pos} sem posição na API"
                           if _n_sem_pos else "")
                        + f" · Minutos: {_fonte_min}"
                        + (f" · {_excl} par(es) atleta×período excluído(s) "
                           "por não participação" if _excl else "")
                        + (f" · ⚠️ sem lista oficial de participantes em: "
                           f"{', '.join(_sem_lista)} (usado piso de "
                           "participação)" if _sem_lista else ""))
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
        ["Variáveis × janelas em colunas", "Variáveis em colunas",
         "Longo (tidy)"],
        horizontal=True, key='wcs_multi_fmt',
        help="Variáveis × janelas em colunas: 1 linha por atleta, com 3 colunas "
             "por variável (1/3/5 min) — sem repetir o atleta em 3 linhas. "
             "Variáveis em colunas: 1 linha por atleta E janela. "
             "Longo: 1 linha por variável (para modelos mistos).")

    if _fmt_out == "Variáveis × janelas em colunas":
        _dfx = pivotar_variaveis_x_janelas(_df, _pct_ocor)
        _nome_csv = "wcs_multi_atividades_var_x_janelas.csv"
        _legenda = (
            "Cada linha é um **atleta** numa atividade/escopo; cada variável tem "
            "**uma coluna por janela** (ex.: `Distância (m) 1min`, "
            "`Distância (m) 3min`, `Distância (m) 5min`). Formato direto para "
            "comparar janelas lado a lado no jamovi.")
    elif _fmt_out == "Variáveis em colunas":
        _dfx = pivotar_variaveis(_df, _pct_ocor)
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
