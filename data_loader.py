# -*- coding: utf-8 -*-
"""Camada de carga de dados (P9 — performance).

Pré-busca EM PARALELO o sinal 10 Hz + efforts de cada (período × atleta),
aquecendo o cache `@st.cache_data` do cliente. O loop de processamento de
`main()` — que ainda roda sequencialmente para atualizar a barra de progresso —
passa a acertar o cache em cada atleta (chamadas instantâneas), trocando N
idas-e-voltas de rede sequenciais por N concorrentes.

Best-effort: exceções são logadas em DEBUG e ignoradas (o loop principal lida
com dados faltantes). As chamadas usam os métodos do cliente (com retry/backoff
do P7); os writes de erro em session_state dentro do cliente são protegidos, o
que torna seguro chamá-los de threads.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import numpy as np

import applog as _applog
from config import FUTEBOL_EVENTS_CONFIG
from diagnostics import _diag_log
from analysis import calcular_metricas, get_zones_for_athlete
from field import (extrair_dados_sensor, extrair_efforts_data,
                   extrair_eventos_futebol, enriquecer_eventos_com_posicao)


def prefetch_sensores(api, activity_id, period_ids, periodos, atletas_ids,
                      effort_types: str = "velocity,acceleration",
                      max_workers: int = 8) -> int:
    """Aquece o cache do sinal+efforts de todos os (período × atleta) em
    paralelo. Retorna o nº de tarefas disparadas."""
    tarefas = []
    for pnome in (periodos or []):
        pid = (period_ids or {}).get(pnome)
        for aid in (atletas_ids or []):
            tarefas.append((pid, aid))
    if not tarefas:
        return 0

    def _um(t):
        pid, aid = t
        try:
            if pid:
                api.get_period_sensor_data(pid, aid)
                api.get_period_efforts(pid, aid, effort_types)
            else:
                api.get_sensor_data(activity_id, aid)
                api.get_activity_efforts(activity_id, aid, effort_types)
        except Exception:
            _applog.log_debug_exc()

    try:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tarefas))) as ex:
            list(ex.map(_um, tarefas))
    except Exception:
        _applog.log_debug_exc()          # sem paralelismo, o loop ainda funciona
    return len(tarefas)


def carregar_dados(api, activity_id, period_ids, periodos_selecionados):
        """(P4/P9) Carrega e processa o sinal 10 Hz + efforts de todos os
        períodos × atletas (pré-busca paralela + loop com barra de progresso).
        Devolve os 9 dicionários por período + contadores de carga."""
        # Dicionários para armazenar dados por período
        resultados_por_periodo = {}
        dados_sensor_por_atleta_por_periodo = {}
        dados_efforts_vel_por_periodo = {}
        dados_efforts_acc_por_periodo = {}
        dados_hr_efforts_por_periodo = {}
        dados_jump_efforts_por_periodo = {}
        dados_step_efforts_por_periodo = {}
        dados_posicao_por_periodo = {}
        dados_eventos_por_periodo = {}   # ← eventos futebol

        # Tipos de eventos futebol selecionados na sidebar
        eventos_futebol_sel = st.session_state.get('eventos_futebol_sel', list(FUTEBOL_EVENTS_CONFIG.keys()))
        eventos_futebol_str = ','.join(eventos_futebol_sel) if eventos_futebol_sel else ''

        # ── Container único de carregamento (substituído a cada atleta, apagado no fim) ──
        _n_per_ld   = len(periodos_selecionados)
        _n_atl_ld   = len(st.session_state.atletas_sel)
        _total_ld   = max(1, _n_per_ld * _n_atl_ld)
        _done_ld    = 0
        _ok_ld      = 0
        _warn_ld    = []
        _ld_box     = st.empty()

        # (P9) Pré-busca PARALELA do sinal 10 Hz + efforts de todos os atletas,
        # aquecendo o cache do cliente. O loop abaixo (sequencial p/ a barra de
        # progresso) passa a acertar o cache — N fetches concorrentes em vez de
        # N sequenciais.
        try:
            _af_pf = st.session_state.get('atletas_filtrados')
            _ids_pf = []
            if _af_pf is not None and not _af_pf.empty:
                for _an_pf in st.session_state.atletas_sel:
                    _row_pf = _af_pf[_af_pf['nome'] == _an_pf]
                    if not _row_pf.empty:
                        _ids_pf.append(_row_pf['id'].values[0])
            if _ids_pf:
                with _ld_box.container():
                    st.caption(f"⚡ Pré-carregando {len(_ids_pf)} atletas em paralelo…")
                prefetch_sensores(
                    api, activity_id, period_ids, periodos_selecionados, _ids_pf,
                    effort_types="velocity,acceleration")
        except Exception:
            _applog.log_debug_exc()      # sem prefetch, o loop ainda carrega tudo

        for periodo_nome in periodos_selecionados:
            period_id = period_ids.get(periodo_nome)

            # ── (Validação/Minutos) participantes oficiais do período ────────
            # O OpenField só atribui o período aos atletas que participaram;
            # dispositivos ligados no banco inflavam Minutos (+30%), m/min e a
            # cauda do PlayerLoad. Carrega apenas quem está na lista oficial.
            _part_ids = set()
            if period_id:
                try:
                    _resp_part = api.get_athletes_in_period(period_id)
                    _lst_part = (_resp_part if isinstance(_resp_part, list)
                                 else (_resp_part or {}).get(
                                     'data', (_resp_part or {}).get('items', [])))
                    for _a_p in (_lst_part or []):
                        if isinstance(_a_p, dict) and _a_p.get('id'):
                            _part_ids.add(str(_a_p['id']))
                except Exception:
                    _part_ids = set()
                # Diagnóstico: quantos participantes a API declara p/ o período?
                # Se == nº total de atletas, o endpoint não distingue banco de
                # campo e o 'Minutos' NÃO baterá com o OpenField (limitação API).
                if _part_ids:
                    _diag_log('Carga', f"Período '{periodo_nome}': "
                                       f"{len(_part_ids)} participantes oficiais "
                                       "na API (filtro de Minutos ativo)")
                else:
                    _diag_log('Carga', f"Período '{periodo_nome}': API sem lista "
                                       "de participantes — todos os atletas "
                                       "carregados (Minutos pode divergir do OF)")

            resultados = []
            dados_sensor_por_atleta = {}
            dados_efforts_vel = {}
            dados_efforts_acc = {}
            dados_hr_efforts = {}
            dados_jump_efforts = {}
            dados_step_efforts = {}
            dados_posicao = {}
            dados_eventos = {}   # ← eventos futebol deste período

            _idx_per_ld = periodos_selecionados.index(periodo_nome) + 1

            for i, atleta_nome in enumerate(st.session_state.atletas_sel):
                _done_ld += 1
                _pct_ld   = _done_ld / _total_ld
                with _ld_box.container():
                    st.markdown(
                        f"<div style='padding:14px 20px;background:#111827;border-radius:10px;"
                        f"border:1px solid #1f2937'>"
                        f"<div style='color:#6b7280;font-size:12px;margin-bottom:6px'>"
                        f"⏳ &nbsp;Período &nbsp;<b style='color:#93c5fd'>{periodo_nome}</b>"
                        f"&nbsp; <span style='opacity:.6'>({_idx_per_ld}/{_n_per_ld})</span>"
                        f"</div>"
                        f"<div style='color:#f9fafb;font-size:15px;font-weight:600'>{atleta_nome}</div>"
                        f"<div style='color:#6b7280;font-size:12px;margin-top:4px'>"
                        f"{_done_ld} / {_total_ld} &nbsp;atletas</div></div>",
                        unsafe_allow_html=True
                    )
                    st.progress(_pct_ld)
                
                athlete_row = st.session_state.atletas_filtrados[st.session_state.atletas_filtrados['nome'] == atleta_nome]
                if athlete_row.empty:
                    continue
                    
                athlete_id = athlete_row['id'].values[0]

                # (Validação/Minutos) atleta fora da lista oficial do período →
                # não participou (dispositivo no banco); não carrega este período.
                if _part_ids and str(athlete_id) not in _part_ids:
                    _diag_log('Carga', f"{atleta_nome}: não participante do período "
                                       f"'{periodo_nome}' — excluído (espelha o "
                                       "Minutos do OpenField)")
                    continue

                athlete_posicao = athlete_row['posicao'].values[0] if 'posicao' in athlete_row.columns else ''
                athlete_equipe = athlete_row['equipe'].values[0] if 'equipe' in athlete_row.columns else ''

                # ── Item 15: limiares individuais do atleta ───────────────────
                # Tenta buscar limiares específicos deste atleta via API.
                # Se disponíveis, usa-os para a análise (armazenado em session_state).
                _thr_key = f"thresholds_{athlete_id}"
                if _thr_key not in st.session_state:
                    try:
                        _thr_raw = api.get_athlete_thresholds(athlete_id)
                        _thr_data = {}
                        if _thr_raw:
                            _thr_items = _thr_raw if isinstance(_thr_raw, list) else _thr_raw.get('data', [])
                            for _thr in (_thr_items if isinstance(_thr_items, list) else []):
                                _tname = _thr.get('name') or _thr.get('type', '')
                                _tval  = _thr.get('value') or _thr.get('threshold')
                                if _tname and _tval is not None:
                                    _thr_data[_tname] = float(_tval)
                        st.session_state[_thr_key] = _thr_data
                    except Exception:
                        st.session_state[_thr_key] = {}

                # ── Auto-popular Vmax histórico a partir do cadastro ──────────
                # Só sobrescreve se o novo valor for MAIOR que o atual
                # (preserva o máximo histórico entre sessões e não destrói
                # valores já captados pelo POST /stats ou openfield_summary).
                # Nunca sobrescreve override manual do usuário.
                _hvm_global = st.session_state.get('hist_vmax', {})
                _src_global = st.session_state.get('hist_vmax_source', {})
                _src_atual  = _src_global.get(atleta_nome, '')
                if _src_atual != 'manual':
                    _vmax_encontrado = 0.0
                    _vmax_fonte      = ''
                    # — Fonte 1: limiares cadastrados (thresholds endpoint) ——
                    _thr_data_now = st.session_state.get(_thr_key, {})
                    _vmax_keys_thr = [
                        'max_velocity', 'velocity_max', 'peak_speed',
                        'max_speed', 'v_max', 'maximum_velocity',
                        'MaxVelocity', 'MaxSpeed', 'velocidade_maxima',
                        'vmax',
                    ]
                    for _vk in _vmax_keys_thr:
                        _vv = _thr_data_now.get(_vk, 0)
                        if _vv:
                            _vvf = float(_vv)
                            # Converte km/h → m/s se plausível
                            if _vvf > 15:
                                _vvf /= 3.6
                            if 0.5 < _vvf < 15.0 and _vvf > _vmax_encontrado:
                                _vmax_encontrado = _vvf
                                _vmax_fonte = 'thresholds'
                    # — Fonte 2: perfil do atleta (GET /athletes/{id}) ————
                    try:
                        _prof_key = f"profile_{athlete_id}"
                        if _prof_key not in st.session_state:
                            _prof_raw = api.get_athlete(athlete_id)
                            st.session_state[_prof_key] = _prof_raw or {}
                        _prof_outer = st.session_state.get(_prof_key, {})
                        _prof = (
                            _prof_outer.get('data', _prof_outer)
                            if isinstance(_prof_outer, dict)
                            else (_prof_outer[0] if isinstance(_prof_outer, list)
                                  and _prof_outer else {})
                        )
                        _vmax_prof_keys = [
                            'max_speed', 'max_velocity', 'maximum_velocity',
                            'maximum_speed', 'peak_speed', 'v_max',
                        ]
                        for _pk in _vmax_prof_keys:
                            _pv = _prof.get(_pk, 0)
                            if _pv:
                                _pvf = float(_pv)
                                if _pvf > 15:
                                    _pvf /= 3.6
                                if 0.5 < _pvf < 15.0 and _pvf > _vmax_encontrado:
                                    _vmax_encontrado = _pvf
                                    _vmax_fonte = 'profile'
                    except Exception:
                        _applog.log_debug_exc()
                    # — Persiste apenas se o valor é MAIOR que o já armazenado ─
                    # (POST /stats já pode ter um valor melhor de outra sessão)
                    _cur_best = _hvm_global.get(atleta_nome, 0.0)
                    if _vmax_encontrado > _cur_best:
                        _hvm_global[atleta_nome] = _vmax_encontrado
                        _src_global[atleta_nome] = _vmax_fonte
                st.session_state['hist_vmax']        = _hvm_global
                st.session_state['hist_vmax_source'] = _src_global

                if period_id:
                    response         = api.get_period_sensor_data(period_id, athlete_id)
                    efforts_response = api.get_period_efforts(period_id, athlete_id,
                                                             "velocity,acceleration,heart_rate,jump,step_balance")
                    events_response  = api.get_period_events(period_id, athlete_id, eventos_futebol_str) if eventos_futebol_str else None
                else:
                    response         = api.get_sensor_data(activity_id, athlete_id)
                    efforts_response = api.get_activity_efforts(activity_id, athlete_id,
                                                               "velocity,acceleration,heart_rate,jump,step_balance")
                    events_response  = api.get_activity_events(activity_id, athlete_id, eventos_futebol_str) if eventos_futebol_str else None
                
                sensor_points = extrair_dados_sensor(response)
                
                if sensor_points:
                    dados_sensor_por_atleta[atleta_nome] = sensor_points

                    _atleta_zones = get_zones_for_athlete(atleta_nome)
                    metricas = calcular_metricas(sensor_points, atleta_nome,
                                                 zones=_atleta_zones)
                    if metricas:
                        metricas['Posição'] = athlete_posicao
                        metricas['Equipe'] = athlete_equipe
                        resultados.append(metricas)

                    if efforts_response:
                        _vel_eff, _acc_eff, _hr_eff, _jmp_eff, _step_eff = extrair_efforts_data(efforts_response)
                        if _vel_eff:
                            dados_efforts_vel[atleta_nome] = _vel_eff
                        if _acc_eff:
                            dados_efforts_acc[atleta_nome] = _acc_eff
                        if _hr_eff:
                            dados_hr_efforts[atleta_nome] = _hr_eff
                        if _jmp_eff:
                            dados_jump_efforts[atleta_nome] = _jmp_eff
                        if _step_eff:
                            dados_step_efforts[atleta_nome] = _step_eff

                    # ── OpenField pre-computed summary ────────────────────────
                    try:
                        if period_id:
                            _of_sum = api.get_athlete_period_summary(period_id, athlete_id)
                        else:
                            _of_sum = api.get_athlete_activity_summary(activity_id, athlete_id)
                        if _of_sum:
                            if atleta_nome not in dados_posicao:
                                dados_posicao[atleta_nome] = {
                                    'vel': [], 'xs': [], 'ys': [], 'acc': [], 'ts_pos': [],
                                    'posicao': athlete_posicao, 'equipe': athlete_equipe,
                                    'n_pontos': 0,
                                }
                            dados_posicao[atleta_nome]['openfield_summary'] = _of_sum
                            # ── Extrai max_velocity da summary para hist_vmax ──
                            # A summary retorna max_velocity em m/s (confirmado no
                            # código de comparação OpenField, linha ~9955).
                            # Guarda o maior valor observado entre os períodos carregados.
                            try:
                                _sum_d = (_of_sum if isinstance(_of_sum, dict)
                                          else (_of_sum[0] if isinstance(_of_sum, list) and _of_sum else {}))
                                _sum_p = _sum_d.get('parameters', _sum_d)
                                _sum_vmax_ms = float(_sum_p.get('max_velocity') or 0)
                                # Sanity check: valores plausíveis para velocidade humana
                                # (0.5 a 15 m/s = ~2 a 54 km/h)
                                if 0.5 < _sum_vmax_ms < 15.0:
                                    _hvm_now = st.session_state.get('hist_vmax', {})
                                    _src_now = st.session_state.get('hist_vmax_source', {})
                                    # Mantém o máximo histórico entre períodos;
                                    # nunca sobrescreve override manual do usuário
                                    if (_src_now.get(atleta_nome, '') != 'manual'
                                            and _sum_vmax_ms > _hvm_now.get(atleta_nome, 0)):
                                        _hvm_now[atleta_nome] = _sum_vmax_ms
                                        _src_now[atleta_nome] = 'summary'
                                        st.session_state['hist_vmax']        = _hvm_now
                                        st.session_state['hist_vmax_source'] = _src_now
                            except Exception:
                                _applog.log_debug_exc()
                    except Exception:
                        _applog.log_debug_exc()
                    
                    # A API devolve x,y com origem no canto inferior esquerdo (0,0).
                    # Filtra nulos e artefactos de projeção GPS (valores absurdamente altos).
                    _venue   = st.session_state.get('venue', {})
                    _fl_v    = float(_venue.get('length') or 105)
                    _fw_v    = float(_venue.get('width')  or 68)
                    pontos_pos = [
                        (float(p['x']), float(p['y']),
                         (p.get('v') or 0) * 3.6,
                         float(p.get('a') or 0),
                         float(p.get('ts') or 0))
                        for p in sensor_points
                        if p.get('x') is not None and p.get('y') is not None
                        and float(p['x']) > -15 and float(p['x']) < _fl_v + 15
                        and float(p['y']) > -15 and float(p['y']) < _fw_v + 15
                    ]
                    if pontos_pos:
                        xs          = [pt[0] for pt in pontos_pos]
                        ys          = [pt[1] for pt in pontos_pos]
                        velocidades = [pt[2] for pt in pontos_pos]
                        aceleracoes = [pt[3] for pt in pontos_pos]
                        ts_pos      = [pt[4] for pt in pontos_pos]

                        # ── Fallback de aceleração (dv/dt) ───────────────────
                        # Muitos dispositivos/exports NÃO trazem o parâmetro 'a'
                        # (aceleração). Sem ele, a WCS por bandas de aceleração
                        # ficaria zerada. Quando 'a' está ausente, derivamos a
                        # aceleração (m/s²) da série de velocidade usando os ts.
                        if not any(abs(_a) > 0.05 for _a in aceleracoes):
                            import statistics as _stacc
                            _vms = [float(v) / 3.6 for v in velocidades]  # km/h→m/s
                            _dts = []
                            for _i in range(1, len(ts_pos)):
                                _d = ts_pos[_i] - ts_pos[_i - 1]
                                _dts.append(_d if (_d and 0 < _d < 2) else None)
                            _valid_dt = [_d for _d in _dts if _d]
                            _dt_med = (_stacc.median(_valid_dt)
                                       if _valid_dt else 0.1)
                            _acc_calc = [0.0] * len(_vms)
                            for _i in range(1, len(_vms)):
                                _dt = (_dts[_i - 1] if _dts[_i - 1] else _dt_med)
                                if _dt and _dt > 0:
                                    _acc_calc[_i] = (_vms[_i] - _vms[_i - 1]) / _dt
                            # Suaviza (média móvel 3) e satura em ±10 m/s².
                            _acc_sm = []
                            for _i in range(len(_acc_calc)):
                                _lo = max(0, _i - 1)
                                _hi = min(len(_acc_calc), _i + 2)
                                _mv = sum(_acc_calc[_lo:_hi]) / (_hi - _lo)
                                _acc_sm.append(max(-10.0, min(10.0, _mv)))
                            aceleracoes = _acc_sm

                        dados_posicao[atleta_nome] = {
                            'vel': velocidades, 'xs': xs, 'ys': ys,
                            'acc': aceleracoes, 'ts_pos': ts_pos,
                            'posicao': athlete_posicao, 'equipe': athlete_equipe,
                            'n_pontos': len(pontos_pos)
                        }

                    # Coleta lat/lon reais (GPS) para o mapa satélite.
                    # Filtra zeros (sem lock de GPS) e valores geograficamente inválidos.
                    # Armazena ts (Unix timestamp) para filtrar pontos por esforço.
                    pontos_gps = [
                        (float(p['lat']), float(p['long']),
                         (p.get('v') or 0) * 3.6,
                         float(p.get('ts') or 0))
                        for p in sensor_points
                        if p.get('lat') is not None and p.get('long') is not None
                        and abs(float(p['lat'])) > 1e-6 and abs(float(p['long'])) > 1e-6
                        and -90 < float(p['lat']) < 90
                        and -180 < float(p['long']) < 180
                    ]
                    if pontos_gps:
                        step_gps = max(1, len(pontos_gps) // 30000)
                        gps_sub = pontos_gps[::step_gps]
                        if atleta_nome not in dados_posicao:
                            dados_posicao[atleta_nome] = {
                                'vel': [], 'xs': [], 'ys': [], 'acc': [], 'ts_pos': [],
                                'posicao': athlete_posicao, 'equipe': athlete_equipe,
                                'n_pontos': 0
                            }
                        dados_posicao[atleta_nome]['lats'] = [pt[0] for pt in gps_sub]
                        dados_posicao[atleta_nome]['lons'] = [pt[1] for pt in gps_sub]
                        dados_posicao[atleta_nome]['vels_gps'] = [pt[2] for pt in gps_sub]
                        dados_posicao[atleta_nome]['ts_gps']  = [pt[3] for pt in gps_sub]

                    # ── GPS Quality (pq, hdop, ref) e Odômetro (o) ────────────
                    # Item 8: qualidade do sinal GPS; Item 12: distância pelo odômetro nativo
                    _pq_vals   = [float(p['pq'])   for p in sensor_points if p.get('pq')   is not None and float(p.get('pq') or 0) > 0]
                    _hdop_vals = [float(p['hdop'])  for p in sensor_points if p.get('hdop') is not None]
                    _ref_vals  = [float(p['ref'])   for p in sensor_points if p.get('ref')  is not None and float(p.get('ref') or 0) > 0]
                    _o_vals    = [float(p['o'])     for p in sensor_points if p.get('o')    is not None]
                    if atleta_nome in dados_posicao:
                        dados_posicao[atleta_nome]['pq_mean']   = round(float(np.mean(_pq_vals)),   1) if _pq_vals   else None
                        dados_posicao[atleta_nome]['hdop_mean'] = round(float(np.mean(_hdop_vals)), 2) if _hdop_vals else None
                        dados_posicao[atleta_nome]['ref_mean']  = round(float(np.mean(_ref_vals)),  1) if _ref_vals  else None
                        # Odometer: distância acumulada nativa do dispositivo (mais preciso que integrar v)
                        if len(_o_vals) >= 2:
                            _o_start = min(_o_vals[0], _o_vals[-1])
                            _o_end   = max(_o_vals[0], _o_vals[-1])
                            dados_posicao[atleta_nome]['odometro_m'] = round(_o_end - _o_start, 1)
                        else:
                            dados_posicao[atleta_nome]['odometro_m'] = None
                        # Série temporal do odômetro para gráfico de evolução
                        if _o_vals:
                            _o_base = _o_vals[0]
                            dados_posicao[atleta_nome]['o_series'] = [v - _o_base for v in _o_vals]
                        else:
                            dados_posicao[atleta_nome]['o_series'] = []

                    # ── Processar eventos futebol ─────────────────────────────
                    if events_response:
                        ev_raw = extrair_eventos_futebol(events_response)
                        if ev_raw:
                            ts_g   = dados_posicao.get(atleta_nome, {}).get('ts_gps', [])
                            lats_g = dados_posicao.get(atleta_nome, {}).get('lats', [])
                            lons_g = dados_posicao.get(atleta_nome, {}).get('lons', [])
                            vels_g = dados_posicao.get(atleta_nome, {}).get('vels_gps', [])
                            dados_eventos[atleta_nome] = enriquecer_eventos_com_posicao(
                                ev_raw, ts_g, lats_g, lons_g, vels_g
                                # campo_config será enriquecido depois, no momento da visualização
                            )
                            n_ev = sum(len(v) for v in ev_raw.values())
                            _ok_ld += 1
                        else:
                            _ok_ld += 1
                    else:
                        _ok_ld += 1
                
            resultados_por_periodo[periodo_nome] = resultados
            dados_sensor_por_atleta_por_periodo[periodo_nome] = dados_sensor_por_atleta
            dados_efforts_vel_por_periodo[periodo_nome] = dados_efforts_vel
            dados_efforts_acc_por_periodo[periodo_nome] = dados_efforts_acc
            dados_hr_efforts_por_periodo[periodo_nome] = dados_hr_efforts
            dados_jump_efforts_por_periodo[periodo_nome] = dados_jump_efforts
            dados_step_efforts_por_periodo[periodo_nome] = dados_step_efforts
            dados_posicao_por_periodo[periodo_nome] = dados_posicao
            dados_eventos_por_periodo[periodo_nome] = dados_eventos

        # (Removido) Calibração automática das bandas de velocidade. Confirmou-se
        # que a Connect API v6 não expõe os limiares nem as distâncias por banda
        # (nem /velocity_zones, nem summary, nem /stats). O app usa limiares
        # FIXOS e documentados (_DEFAULT_VELOCITY_ZONES, padrão da literatura) —
        # instrumento determinístico, requisito para a validação científica.

        # (Removido) Derivação dos cortes a partir dos efforts. Inferir os
        # limiares dos dados é uma forma de auto-calibração — inadequada para um
        # instrumento de validação. O app usa os limiares FIXOS documentados
        # (_DEFAULT_VELOCITY_ZONES / _DEFAULT_ACCELERATION_ZONES); quando a conta
        # expõe as zonas por API (leitura limpa da configuração), essas têm
        # prioridade — ver _zonas_conta_via_api na conexão.

        # Apagar container de loading e mostrar resumo compacto
        _ld_box.empty()
        return (resultados_por_periodo, dados_sensor_por_atleta_por_periodo,
                dados_efforts_vel_por_periodo, dados_efforts_acc_por_periodo,
                dados_hr_efforts_por_periodo, dados_jump_efforts_por_periodo,
                dados_step_efforts_por_periodo, dados_posicao_por_periodo,
                dados_eventos_por_periodo, _ok_ld, _n_atl_ld)
