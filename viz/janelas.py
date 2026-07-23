# -*- coding: utf-8 -*-
"""Aba extraída de main() (P4 — page-decomposition)."""
from __future__ import annotations

from bands import _ACC_KEY_TO_NUM
from config import _CHAVE_COMBINADO
import applog as _applog
from bands import _bandas_acc_ativas
from bands import _bandas_vel_ativas
from diagnostics import _diag_log
from bands import _fmt_num_banda
from field import _get_pos_grupo
import metrics as _mtr
from diagnostics import _selo_fonte
from field import _vmax_individual_kmh
from analysis import acc_series_from_vel
from analysis import calcular_distancia_janelas_discretas_10s
from analysis import calcular_distancia_janelas_por_vel_posicao
from analysis import calcular_janelas_discretas_10s
from analysis import combinar_periodos_continuo
from analysis import combinar_periodos_continuo_posicao
from field import desenhar_campo_futebol_bonito
from analysis import detectar_acoes_acc_idx
from analysis import encontrar_eventos_nao_sobrepostos
from analysis import exibir_resultados_janela
from analysis import get_min_dur_s
import plotly.graph_objects as go
from field import gps_para_campo_coords
import numpy as np
from analysis import obter_limites_periodos_posicao
import pandas as pd
import streamlit as st


def render_janelas(REFERENCIAS, _REL_VEL_BANDAS, _SENSOR_HZ, dados_efforts_acc_por_periodo, dados_posicao_por_periodo, dados_sensor_por_atleta_por_periodo):
                st.subheader("📊 Análise de Intensidade - Janelas Temporais (Rolling Window)")
                st.markdown("""
                Esta análise usa uma **janela deslizante** (*rolling window*): a janela avança **10 s por passo**
                ao longo de toda a sessão, capturando o pico real de intensidade independentemente de onde ele
                começa — sem o erro de corte das janelas discretas fixas.
                No modo **combinado**, todos os períodos são encadeados em uma linha do tempo contínua
                (sem o gap do intervalo).
                """)

                if dados_sensor_por_atleta_por_periodo:

                    # ── Seletor de modo ────────────────────────────────────────────
                    _jan_modo = st.radio(
                        "Modo de análise:",
                        ["🔵 Individual", "🟡 Por Posição", "🔴 Time Completo"],
                        horizontal=True, key="jan_modo_analise",
                        help="Individual: análise detalhada de um atleta · "
                             "Por Posição: curva média por grupo tático · "
                             "Time Completo: heatmap de intensidade de todos os atletas"
                    )
                    st.divider()

                    # ── Controles comuns: período + janela + métrica ───────────────
                    _JAN_TODOS = "🔀 Todos os períodos (combinado)"
                    _jan_opcoes = [_JAN_TODOS] + list(dados_sensor_por_atleta_por_periodo.keys())
                    periodo_janela = st.selectbox("Selecione o período:", _jan_opcoes, key="periodo_janela")
                    _jan_modo_todos = (periodo_janela == _JAN_TODOS)

                    if _jan_modo_todos:
                        _jan_ats_set = set()
                        for _pv in dados_sensor_por_atleta_por_periodo.values():
                            _jan_ats_set.update(_pv.keys())
                        _jan_atletas = sorted(_jan_ats_set)
                    else:
                        _jan_atletas = list(dados_sensor_por_atleta_por_periodo.get(periodo_janela, {}).keys())

                    bandas_vel = [3, 4, 5, 6, 7, 8]
                    bandas_acc = [1, 2, 3]
                    _col_w, _col_m, _col_extra = st.columns(3)
                    with _col_w:
                        window_minutes = st.slider(
                            "Janela temporal (minutos):",
                            min_value=0.5, max_value=10.0, value=1.0, step=0.5,
                            key="jan_window"
                        )
                    _MET_ACOES = '💥 Ações Acel/Desacel'
                    _MET_VEL_BANDAS = '🏃 Velocidade (bandas)'
                    with _col_m:
                        tipo_metrica = st.selectbox(
                            "Métrica:",
                            ['Distância', 'PlayerLoad', _MET_VEL_BANDAS, _MET_ACOES],
                            key="jan_metrica",
                            help="🏃 Velocidade (bandas) = distância (m) percorrida nas bandas "
                                 "de velocidade selecionadas, por janela (igual ao WCS). "
                                 "💥 Ações Acel/Desacel = nº de ações de acel/desacel nas "
                                 "bandas, detectadas no sinal de aceleração do sensor (mesma "
                                 "fonte da aba Neuromuscular)."
                        )
                    # ── Bandas de VELOCIDADE (para a métrica '🏃 Velocidade (bandas)') ──
                    sel_vel_bands = []   # dicts {min,max} absolutos (km/h)
                    sel_vel_pct = []     # (P9) faixas relativas (fração da Vmáx)
                    jan_vel_rel = False
                    with _col_extra:
                        if tipo_metrica == _MET_VEL_BANDAS:
                            jan_vel_rel = st.checkbox(
                                "% da Vmáx individual", value=False, key="jan_vel_rel",
                                help="(P9) Em vez das bandas absolutas da conta, usa faixas "
                                     "relativas à velocidade máxima de CADA atleta "
                                     "(histórico da conta, com fallback no pico da sessão).")
                            if jan_vel_rel:
                                _rel_pick_j = st.multiselect(
                                    "🎚️ Faixas (% da Vmáx)",
                                    list(_REL_VEL_BANDAS.keys()),
                                    default=[_k for _k in _REL_VEL_BANDAS
                                             if _REL_VEL_BANDAS[_k][0] >= 0.7],
                                    key="jan_vel_rel_bands",
                                    help="A distância (m) é acumulada quando a velocidade "
                                         "está dentro da faixa relativa do próprio atleta.")
                                sel_vel_pct = [_REL_VEL_BANDAS[_s] for _s in _rel_pick_j]
                                if not sel_vel_pct:
                                    st.info("Selecione ao menos uma faixa.")
                            else:
                                _bv_act_j = _bandas_vel_ativas()
                                _bv_lbl_j = {}
                                for _bk, _bd in _bv_act_j.items():
                                    _mx = float(_bd.get('max', 9999))
                                    _faixa = (f">{_fmt_num_banda(_bd.get('min', 0))}"
                                              if _mx >= 9000
                                              else f"{_fmt_num_banda(_bd.get('min', 0))}-"
                                                   f"{_fmt_num_banda(_mx)}")
                                    _bv_lbl_j[f"B{_bk} — {_faixa} km/h"] = _bk
                                _bv_pick_j = st.multiselect(
                                    "🎚️ Bandas de velocidade",
                                    list(_bv_lbl_j.keys()),
                                    default=list(_bv_lbl_j.keys()),
                                    key="jan_vel_bands",
                                    help="A distância (m) é acumulada apenas enquanto a velocidade "
                                         "está dentro das bandas selecionadas — igual ao WCS."
                                )
                                sel_vel_bands = [_bv_act_j[_bv_lbl_j[_s]] for _s in _bv_pick_j]
                                if not sel_vel_bands:
                                    st.info("Selecione ao menos uma banda de velocidade.")

                    _unidade_jan = {
                        'Distância': 'm/min', 'PlayerLoad': 'PL/min',
                        _MET_VEL_BANDAS: 'm', _MET_ACOES: 'ações',
                    }.get(tipo_metrica, '')

                    # ── Bandas de AÇÕES (efforts) — duas caixas accel/decel ────────
                    # Mesma seleção da aba WCS para que os valores batam.
                    sel_acc_bands = []
                    sel_acc_boxes = set()   # caixas Gen2 (1..8) selecionadas
                    if tipo_metrica == _MET_ACOES:
                        _ba_act_j = _bandas_acc_ativas()
                        _acc_lbl_j = {_ba_act_j[k]['label']: k
                                      for k in _ba_act_j if str(k).startswith('A')}
                        _dec_lbl_j = {_ba_act_j[k]['label']: k
                                      for k in _ba_act_j if str(k).startswith('D')}
                        _cja, _cjd = st.columns(2)
                        with _cja:
                            _acc_pick_j = st.multiselect(
                                "🚀 Aceleração",
                                list(_acc_lbl_j.keys()),
                                default=list(_acc_lbl_j.keys()),
                                key="jan_acc_bands_pos",
                                help="Ações de aceleração (Gen2Acceleration · caixas 6,7,8)."
                            )
                        with _cjd:
                            _dec_pick_j = st.multiselect(
                                "🛑 Desaceleração",
                                list(_dec_lbl_j.keys()),
                                default=list(_dec_lbl_j.keys()),
                                key="jan_acc_bands_neg",
                                help="Ações de desaceleração (Gen2Acceleration · caixas 3,2,1)."
                            )
                        sel_acc_bands = (
                            [_ba_act_j[_acc_lbl_j[_s]] for _s in _acc_pick_j]
                            + [_ba_act_j[_dec_lbl_j[_s]] for _s in _dec_pick_j]
                        )
                        # Caixas Gen2 oficiais das bandas escolhidas (A1..A3→6,7,8;
                        # D1..D3→3,2,1). Contar pela caixa é robusto — não depende do
                        # valor médio do effort cair no intervalo derivado.
                        sel_acc_boxes = (
                            {_ACC_KEY_TO_NUM[_acc_lbl_j[_s]] for _s in _acc_pick_j
                             if _acc_lbl_j[_s] in _ACC_KEY_TO_NUM}
                            | {_ACC_KEY_TO_NUM[_dec_lbl_j[_s]] for _s in _dec_pick_j
                               if _dec_lbl_j[_s] in _ACC_KEY_TO_NUM}
                        )
                        st.caption(
                            "Conta o **nº de ações** de acel/desacel nas bandas selecionadas "
                            "por janela — detectadas no **sinal de aceleração do sensor** "
                            "(mesma fonte da aba **Neuromuscular**), sustentadas pela duração "
                            "mínima da sidebar. O pior minuto é a janela com mais ações. "
                            "Quando a API traz *acceleration_efforts* (modo por período), usa "
                            "a contagem oficial por caixa Gen2."
                        )
                        if not sel_acc_bands:
                            st.info("Selecione ao menos uma banda de aceleração ou desaceleração.")

                    # Detecta Hz uma vez — (P1) canônico: metrics.estimate_hz
                    # (nativo usa ts_pos, GPS-only usa ts_gps).
                    _hz_jan = 10.0
                    if dados_posicao_por_periodo:
                        _series_hz = []
                        for _pn_hz in list(dados_posicao_por_periodo.keys())[:5]:
                            for _an_hz in list(dados_posicao_por_periodo[_pn_hz].values())[:5]:
                                _tss_hz = _an_hz.get('ts_pos', []) or _an_hz.get('ts_gps', [])
                                if len(_tss_hz) > 20:
                                    _series_hz.append(_tss_hz)
                        _hz_jan = _mtr.estimate_hz(_series_hz, default=10.0)

                    # ── Helper: AÇÕES (efforts) — espelha exatamente o cálculo WCS ─
                    def _calc_rolling_acoes(_atl):
                        """
                        Conta ações (efforts) de acel/desacel por janela rolante,
                        usando EXATAMENTE a mesma lógica da aba 'Pior Cenário (WCS)':
                        timeline = ts_pos concatenado dos períodos (de
                        dados_posicao_por_periodo); cada effort soma +1 na amostra
                        mais próxima do seu start_time; soma rolante de N amostras.
                        Assim o pico individual coincide com o WCS.
                        Retorna (tempos_min, valores) — valores = nº de ações na janela.
                        """
                        # Períodos reais (para efforts oficiais da API, quando existirem).
                        _ps = ([k for k in dados_posicao_por_periodo
                                if k != _CHAVE_COMBINADO]
                               if _jan_modo_todos else [periodo_janela])

                        # ── FONTE: sinal de aceleração do SENSOR (nativo 'a', 10 Hz) ──
                        # Mesma fonte da aba Neuromuscular (que conta 200+ ações). Antes
                        # usava a trajetória GPS + derivação por velocidade, que zerava
                        # em dispositivos só-GPS. O sinal do sensor é sempre confiável.
                        if _jan_modo_todos:
                            _sp = combinar_periodos_continuo(
                                dados_sensor_por_atleta_por_periodo, _atl)
                        else:
                            _sp = dados_sensor_por_atleta_por_periodo.get(
                                periodo_janela, {}).get(_atl, [])
                        if not _sp or not sel_acc_bands:
                            return [], []

                        _Hz = float(_SENSOR_HZ)                 # sensor uniforme 10 Hz
                        _n = max(2, int(window_minutes * 60 * _Hz))
                        _nsp = len(_sp)
                        if _nsp < _n:
                            return [], []

                        # Sinal de aceleração (m/s²): nativo 'a'; se ausente (só-GPS sem
                        # IMU), deriva de dv/dt da velocidade do próprio sensor.
                        _acc_sig = [float(_p.get('a') or 0.0) for _p in _sp]
                        _ts_raw = [float(_p.get('ts') or 0.0) for _p in _sp]
                        st.session_state['_prov_jan_acoes'] = 'sensor'   # (P4)
                        if not any(abs(_a) > 0.05 for _a in _acc_sig):
                            _vel_sig = [float(_p.get('v') or 0.0) * 3.6 for _p in _sp]
                            _acc_sig = acc_series_from_vel(_vel_sig, _ts_raw, _Hz)
                            st.session_state['_prov_jan_acoes'] = 'derivado'
                            _diag_log('Janelas', f"{_atl}: sem aceleração nativa — "
                                                 "ações derivadas por dv/dt da velocidade")

                        _sv = [0.0] * _nsp
                        _ts_np = np.array(_ts_raw, dtype=float)
                        _ts_unix_ok = (_ts_np.size > 0
                                       and float(np.median(_ts_np)) > 1e6)
                        _has_api_eff = any(
                            len(dados_efforts_acc_por_periodo
                                .get(_pn, {}).get(_atl, []) or []) > 0
                            for _pn in _ps)
                        # Efforts oficiais da API (contagem por caixa Gen2) só quando há
                        # timestamps Unix — o modo combinado reescreve os ts, então usa o
                        # sinal. Ambos contam AÇÕES; o sinal garante que nunca zere.
                        if _has_api_eff and _ts_unix_ok and not _jan_modo_todos:
                            st.session_state['_prov_jan_acoes'] = 'efforts'   # (P4)
                            for _pn in _ps:
                                for _ef in (dados_efforts_acc_por_periodo
                                            .get(_pn, {}).get(_atl, []) or []):
                                    try:
                                        _bx = int(round(float(_ef.get('band'))))
                                        _stt = float(_ef.get('start_time') or 0)
                                    except (TypeError, ValueError):
                                        continue
                                    if _stt <= 0 or _bx not in sel_acc_boxes:
                                        continue
                                    _idx = int(np.argmin(np.abs(_ts_np - _stt)))
                                    if 0 <= _idx < _nsp:
                                        _sv[_idx] += 1.0
                        else:
                            _idxs_acc = detectar_acoes_acc_idx(
                                _acc_sig, sel_acc_bands, freq_hz=_Hz)
                            for _ix in _idxs_acc:
                                if 0 <= _ix < _nsp:
                                    _sv[_ix] += 1.0

                        # Soma rolante — (P1) canônico: metrics.rolling_sum
                        _roll = _mtr.rolling_sum(_sv, _n)
                        if not _roll:
                            return [], []

                        # Downsample (~1 ponto/s) garantindo o pico real (= WCS)
                        _stepd = max(1, int(round(_Hz)))
                        _t_out, _v_out = [], []
                        for _i in range(0, len(_roll), _stepd):
                            _t_out.append(_i / (_Hz * 60.0))
                            _v_out.append(float(_roll[_i]))
                        _imax = int(np.argmax(_roll))
                        if _imax % _stepd != 0:
                            import bisect as _bis
                            _tp = _imax / (_Hz * 60.0)
                            _pos = _bis.bisect_left(_t_out, _tp)
                            _t_out.insert(_pos, _tp)
                            _v_out.insert(_pos, float(_roll[_imax]))
                        return _t_out, _v_out

                    # ── Helper: VELOCIDADE (bandas) — distância (m) nas bandas por janela ─
                    def _calc_rolling_vel_bandas(_atl):
                        """Distância (m) percorrida nas bandas de velocidade selecionadas,
                        por janela rolante — MESMA lógica do WCS '🏃 Velocidade (bandas)':
                        soma v/(3.6·Hz) por amostra quando a velocidade cai nas bandas."""
                        if _jan_modo_todos:
                            _ps = [k for k in dados_posicao_por_periodo
                                   if k != _CHAVE_COMBINADO]
                        else:
                            _ps = [periodo_janela]
                        _wts, _wv = [], []
                        for _pn in _ps:
                            _da = dados_posicao_por_periodo.get(_pn, {}).get(_atl, {})
                            # Nativo: ts_pos/vel · GPS-only: ts_gps/vels_gps
                            _ts = _da.get('ts_pos', []) or _da.get('ts_gps', [])
                            _vl = _da.get('vel', []) or _da.get('vels_gps', [])   # km/h
                            _nn = min(len(_ts), len(_vl))
                            if _nn == 0:
                                continue
                            _wts += list(_ts[:_nn])
                            _wv += list(_vl[:_nn])
                        _Hz = _hz_jan
                        _n = max(2, int(window_minutes * 60 * _Hz))
                        if len(_wv) < _n:
                            return [], []
                        if jan_vel_rel:
                            # (P9) faixas relativas à Vmáx individual do atleta
                            _vmx = _vmax_individual_kmh(_atl, _wv)
                            if _vmx <= 0 or not sel_vel_pct:
                                _diag_log('Janelas', f"{_atl}: sem Vmáx individual "
                                                     "confiável — excluído do modo % Vmáx")
                                return [], []
                            _faixas_v = [(lo * _vmx, hi * _vmx) for lo, hi in sel_vel_pct]
                        else:
                            _faixas_v = [(float(b.get('min', 0)), float(b.get('max', 9999)))
                                         for b in sel_vel_bands]
                        if not _faixas_v:
                            return [], []

                        # (P1) canônico: metrics.per_sample_distance_in_bands + rolling_sum
                        _sv = _mtr.per_sample_distance_in_bands(_wv, _faixas_v, _Hz)
                        _roll = _mtr.rolling_sum(_sv, _n)
                        if not _roll:
                            return [], []
                        _stepd = max(1, int(round(_Hz)))
                        _t_out, _v_out = [], []
                        for _i in range(0, len(_roll), _stepd):
                            _t_out.append(_i / (_Hz * 60.0))
                            _v_out.append(float(_roll[_i]))
                        _imax = int(np.argmax(_roll))
                        if _imax % _stepd != 0:
                            import bisect as _bis
                            _tp = _imax / (_Hz * 60.0)
                            _pos = _bis.bisect_left(_t_out, _tp)
                            _t_out.insert(_pos, _tp)
                            _v_out.insert(_pos, float(_roll[_imax]))
                        return _t_out, _v_out

                    # ── Helper: rolling window para um atleta ──────────────────────
                    def _calc_rolling(_atl):
                        """Retorna (tempos_min, valores) para o atleta e configuração atual."""
                        if tipo_metrica == _MET_ACOES:
                            return _calc_rolling_acoes(_atl)
                        if tipo_metrica == _MET_VEL_BANDAS:
                            return _calc_rolling_vel_bandas(_atl)
                        # ── Sensor helper (reutilizado no fallback de Distância) ────
                        def _get_sp():
                            if _jan_modo_todos:
                                return combinar_periodos_continuo(
                                    dados_sensor_por_atleta_por_periodo, _atl)
                            return dados_sensor_por_atleta_por_periodo.get(
                                periodo_janela, {}).get(_atl, [])

                        if tipo_metrica == 'Distância':
                            # 1ª tentativa: GPS field-filtered (mais preciso)
                            if dados_posicao_por_periodo:
                                if _jan_modo_todos:
                                    _vj, _tj = combinar_periodos_continuo_posicao(
                                        dados_posicao_por_periodo, _atl)
                                else:
                                    _daj = dados_posicao_por_periodo.get(
                                        periodo_janela, {}).get(_atl, {})
                                    _vj = _daj.get('vel', [])
                                    _tj = _daj.get('ts_pos', [])
                                if _vj:
                                    _res_gps = calcular_distancia_janelas_por_vel_posicao(
                                        _vj, _tj, window_minutes, _hz_jan)
                                    if _res_gps[0]:   # GPS devolveu dados válidos
                                        st.session_state['_prov_jan_dist'] = 'gps'   # (P4)
                                        return _res_gps
                            # 2ª tentativa: sensor IMU (fallback)
                            _sp = _get_sp()
                            if _sp:
                                st.session_state['_prov_jan_dist'] = 'sensor'   # (P4)
                                return calcular_distancia_janelas_discretas_10s(
                                    _sp, window_minutes)
                            return [], []

                        # Métricas baseadas em sensor
                        _sp = _get_sp()
                        if not _sp:
                            return [], []
                        if tipo_metrica == 'PlayerLoad':
                            return calcular_janelas_discretas_10s(_sp, window_minutes, 'pl', None)
                        if tipo_metrica == 'Velocidade':
                            return calcular_janelas_discretas_10s(
                                _sp, window_minutes, 'v', {'velocity_bands': bandas_vel})
                        if tipo_metrica == 'Aceleração':
                            return calcular_janelas_discretas_10s(
                                _sp, window_minutes, 'a', {'acceleration_bands': bandas_acc})
                        return [], []

                    # ── Helper: posição do atleta ──────────────────────────────────
                    def _get_pos_atl(_atl):
                        for _pd in dados_posicao_por_periodo.values():
                            if _atl in _pd:
                                return _pd[_atl].get('posicao') or 'Outro'
                        return 'Outro'

                    # ── Paleta por posição ─────────────────────────────────────────
                    _POS_COR = {
                        'Goleiro': '#5dade2',      'Zagueiro': '#2ecc71',
                        'Lateral': '#1abc9c',      'Volante': '#f39c12',
                        'Meia': '#e67e22',         'Meia-atacante': '#d4ac0d',
                        'Atacante': '#e74c3c',     'Extremo': '#c0392b',
                        'Centroavante': '#9b59b6', 'Outro': '#95a5a6',
                    }
                    _POS_RGBA_FILL = {
                        k: f"rgba({int(v[1:3],16)},{int(v[3:5],16)},{int(v[5:7],16)},0.13)"
                        for k, v in _POS_COR.items()
                    }

                    # ══════════════════════════════════════════════════════════════
                    # MODO 1 — INDIVIDUAL
                    # ══════════════════════════════════════════════════════════════
                    if _jan_modo == "🔵 Individual":
                        if _jan_atletas:
                            atleta_janela = st.selectbox(
                                "Selecione o atleta:", _jan_atletas, key="atleta_janela")

                            if _jan_modo_todos:
                                sensor_points = combinar_periodos_continuo(
                                    dados_sensor_por_atleta_por_periodo, atleta_janela)
                                st.caption(
                                    f"📊 Combinando **{len(dados_sensor_por_atleta_por_periodo)} períodos** "
                                    f"em linha do tempo contínua → {len(sensor_points):,} amostras "
                                    f"para **{atleta_janela}**.")
                            else:
                                sensor_points = dados_sensor_por_atleta_por_periodo[
                                    periodo_janela].get(atleta_janela, [])

                            if sensor_points:
                                _dur_s_aba4 = get_min_dur_s()
                                st.caption(
                                    f"⚙️ Duração mínima de acc/dec: **{_dur_s_aba4:.1f} s** "
                                    f"({max(1, round(_dur_s_aba4 * _SENSOR_HZ))} frames) — "
                                    "ajuste na sidebar.")

                                if _jan_modo_todos and dados_posicao_por_periodo:
                                    _period_boundaries = obter_limites_periodos_posicao(
                                        dados_posicao_por_periodo, atleta_janela)
                                elif not _jan_modo_todos:
                                    _period_boundaries = [(0.0, float('inf'), periodo_janela)]
                                else:
                                    _period_boundaries = None

                                with st.spinner("Calculando janelas temporais..."):
                                    _tj, _vj = _calc_rolling(atleta_janela)
                                    if _tj:
                                        # (P4) selo de proveniência da métrica exibida
                                        if tipo_metrica == _MET_ACOES:
                                            _selo_fonte(st.session_state.get(
                                                '_prov_jan_acoes', 'sensor'))
                                        elif tipo_metrica == 'Distância':
                                            _selo_fonte(st.session_state.get(
                                                '_prov_jan_dist', 'gps'))
                                        elif tipo_metrica == _MET_VEL_BANDAS:
                                            _vmx_ref = (_vmax_individual_kmh(atleta_janela)
                                                        if jan_vel_rel else 0.0)
                                            _selo_fonte('gps',
                                                        (f"faixas em **% da Vmáx individual** "
                                                         f"(ref.: {_vmx_ref:.1f} km/h)"
                                                         if jan_vel_rel and _vmx_ref > 0 else
                                                         ("faixas em % da Vmáx individual"
                                                          if jan_vel_rel else "")))
                                        elif tipo_metrica == 'PlayerLoad':
                                            _selo_fonte('sensor')
                                        exibir_resultados_janela(
                                            _tj, _vj, tipo_metrica, atleta_janela,
                                            window_minutes, _unidade_jan, _period_boundaries)
                                    else:
                                        st.warning("Dados insuficientes para calcular janelas.")

                                if not st.session_state.get('modo_apresentacao'):
                                    st.markdown(REFERENCIAS["janelas"])
                            else:
                                st.info("Dados de sensor não disponíveis")
                        else:
                            st.info("Selecione um atleta para análise")

                    # ══════════════════════════════════════════════════════════════
                    # MODO 2 — POR POSIÇÃO
                    # ══════════════════════════════════════════════════════════════
                    elif _jan_modo == "🟡 Por Posição":
                        if not _jan_atletas:
                            st.info("Sem atletas disponíveis.")
                        else:
                            # Agrupa atletas por posição
                            _pos_atls: dict = {}
                            for _a in _jan_atletas:
                                _p = _get_pos_atl(_a)
                                _pos_atls.setdefault(_p, []).append(_a)

                            _pos_sel = st.multiselect(
                                "Posições a comparar:",
                                options=sorted(_pos_atls.keys()),
                                default=sorted(_pos_atls.keys()),
                                key="jan_pos_sel"
                            )
                            if not _pos_sel:
                                st.info("Selecione ao menos uma posição.")
                            else:
                                with st.spinner("Calculando por posição..."):
                                    import plotly.graph_objects as _go_pos

                                    # Rolling window para cada atleta das posições selecionadas
                                    _atl_res: dict = {}
                                    for _ps in _pos_sel:
                                        for _a in _pos_atls.get(_ps, []):
                                            if _a not in _atl_res:
                                                _t_a, _v_a = _calc_rolling(_a)
                                                if _t_a and _v_a:
                                                    _atl_res[_a] = (
                                                        np.array(_t_a), np.array(_v_a))

                                    if not _atl_res:
                                        st.warning(
                                            "Sem dados suficientes para as posições selecionadas.")
                                    else:
                                        _max_t_pos = max(
                                            float(_t[-1]) + window_minutes
                                            for _t, _ in _atl_res.values())
                                        _t_grid_pos = np.arange(0, _max_t_pos + 1/60, 1/60)

                                        fig_pos = _go_pos.Figure()
                                        _pos_summ: list = []

                                        for _ps in _pos_sel:
                                            _atls_ps = [
                                                _a for _a in _pos_atls.get(_ps, [])
                                                if _a in _atl_res]
                                            if not _atls_ps:
                                                continue

                                            _v_mat = np.array([
                                                np.interp(_t_grid_pos,
                                                          _atl_res[_a][0],
                                                          _atl_res[_a][1],
                                                          left=np.nan, right=np.nan)
                                                for _a in _atls_ps
                                            ])
                                            _v_mean = np.nanmean(_v_mat, axis=0)
                                            _cor = _POS_COR.get(_ps, '#95a5a6')
                                            _fil = _POS_RGBA_FILL.get(_ps,
                                                                       'rgba(149,165,166,0.13)')

                                            # Área ± std (se mais de 1 atleta)
                                            if len(_atls_ps) > 1:
                                                _v_std = np.nanstd(_v_mat, axis=0)
                                                _yu = _v_mean + _v_std
                                                _yd = _v_mean - _v_std
                                                fig_pos.add_trace(_go_pos.Scatter(
                                                    x=np.concatenate(
                                                        [_t_grid_pos, _t_grid_pos[::-1]]),
                                                    y=np.concatenate([_yu, _yd[::-1]]),
                                                    fill='toself', fillcolor=_fil,
                                                    line=dict(color='rgba(0,0,0,0)'),
                                                    showlegend=False, hoverinfo='skip',
                                                ))

                                            fig_pos.add_trace(_go_pos.Scatter(
                                                x=_t_grid_pos, y=_v_mean,
                                                name=f"{_ps} (n={len(_atls_ps)})",
                                                line=dict(color=_cor, width=2.5),
                                                mode='lines',
                                                hovertemplate=(
                                                    f"<b>{_ps}</b><br>"
                                                    "Tempo: %{x:.1f} min<br>"
                                                    f"{tipo_metrica}: %{{y:.1f}} {_unidade_jan}"
                                                    "<extra></extra>"
                                                ),
                                            ))

                                            _pk = round(float(np.nanmax(_v_mean)), 1)
                                            _av = round(float(np.nanmean(_v_mean)), 1)
                                            _pos_summ.append({
                                                'Posição': _ps,
                                                'N Atletas': len(_atls_ps),
                                                f'Pico Médio ({_unidade_jan})': _pk,
                                                f'Média Geral ({_unidade_jan})': _av,
                                            })

                                        # Limiares globais
                                        _all_v_pos = np.concatenate(
                                            [v for _, v in _atl_res.values()])
                                        _gmax_pos = float(np.nanmax(_all_v_pos))
                                        _la_pos = round(_gmax_pos * 0.90, 1)
                                        _lm_pos = round(_gmax_pos * 0.75, 1)
                                        fig_pos.add_hline(
                                            y=_la_pos, line_dash='dash',
                                            line_color='rgba(239,68,68,0.50)',
                                            annotation_text=f"Alta ≥{_la_pos} {_unidade_jan}",
                                            annotation_position="right")
                                        fig_pos.add_hline(
                                            y=_lm_pos, line_dash='dot',
                                            line_color='rgba(245,158,11,0.50)',
                                            annotation_text=f"Média-Alta ≥{_lm_pos} {_unidade_jan}",
                                            annotation_position="right")

                                        fig_pos.update_layout(
                                            title=dict(
                                                text=(f"Intensidade de {tipo_metrica} por Posição"
                                                      f" — Rolling Window {window_minutes} min"),
                                                font=dict(color='white', size=14)),
                                            xaxis=dict(
                                                title='Tempo (minutos)',
                                                color='rgba(255,255,255,0.6)',
                                                gridcolor='rgba(255,255,255,0.07)'),
                                            yaxis=dict(
                                                title=f'{tipo_metrica} ({_unidade_jan})',
                                                color='rgba(255,255,255,0.6)',
                                                gridcolor='rgba(255,255,255,0.07)'),
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            legend=dict(
                                                font=dict(color='white'),
                                                bgcolor='rgba(0,0,0,0)'),
                                            hovermode='x unified',
                                            height=440,
                                        )
                                        st.plotly_chart(fig_pos, use_container_width=True)

                                        if _pos_summ:
                                            st.markdown("##### 📊 Resumo por Posição")
                                            _df_pos = (
                                                pd.DataFrame(_pos_summ)
                                                .sort_values(f'Pico Médio ({_unidade_jan})',
                                                             ascending=False)
                                                .reset_index(drop=True)
                                            )
                                            st.dataframe(
                                                _df_pos, use_container_width=True,
                                                hide_index=True,
                                                height=38 * len(_df_pos) + 60)

                    # ══════════════════════════════════════════════════════════════
                    # MODO 3 — TIME COMPLETO
                    # ══════════════════════════════════════════════════════════════
                    elif _jan_modo == "🔴 Time Completo":
                        if not _jan_atletas:
                            st.info("Sem atletas disponíveis.")
                        else:
                            with st.spinner(
                                    f"Calculando rolling window para {len(_jan_atletas)} atletas..."):
                                import plotly.graph_objects as _go_tm

                                # Rolling window para cada atleta
                                _team_res: dict = {}
                                for _a in _jan_atletas:
                                    _ta, _va = _calc_rolling(_a)
                                    if _ta and _va:
                                        _team_res[_a] = (
                                            np.array(_ta), np.array(_va))

                                if not _team_res:
                                    st.warning("Dados insuficientes.")
                                else:
                                    # ── Offset absoluto por atleta ─────────────────
                                    # Lógica: cada "período" é uma actividade gravada
                                    # separadamente. Atletas que continuam no jogo
                                    # aparecem em múltiplos períodos consecutivos.
                                    # O substituto entra apenas a partir do período
                                    # em que foi inserido.
                                    #
                                    # Usamos SEMPRE duração acumulada via sensor IMU
                                    # (ts_last − ts_first dentro do mesmo período),
                                    # que funciona tanto para ts relativo (0-based)
                                    # quanto Unix — porque o intervalo interno cancela
                                    # qualquer origem absoluta.
                                    # NÃO usamos Unix ts direto: combinar_periodos_
                                    # continuo remove os intervalos entre períodos,
                                    # enquanto Unix ts os inclui → dessincronização.
                                    _period_order_tm = (
                                        list(dados_sensor_por_atleta_por_periodo.keys())
                                        if _jan_modo_todos
                                        else [periodo_janela]
                                    )

                                    # ── Timestamps absolutos de cada período ──────────
                                    # Sensor IMU (ts + cs/100) usa Unix absoluto →
                                    # períodos sobrepostos têm ts que se intersectam.
                                    def _period_abs_ts_tm(_pnm: str):
                                        """(first_ts, last_ts) em segundos Unix via sensor IMU."""
                                        _mn, _mx = None, None
                                        for _spl in dados_sensor_por_atleta_por_periodo.get(
                                                _pnm, {}).values():
                                            for _pp in _spl:
                                                _tt = (float(_pp.get('ts') or 0)
                                                       + float(_pp.get('cs') or 0) / 100.0)
                                                if _tt <= 0:
                                                    continue
                                                if _mn is None or _tt < _mn: _mn = _tt
                                                if _mx is None or _tt > _mx: _mx = _tt
                                        return (_mn or 0.0, _mx or 0.0)

                                    _period_abs_tm = {
                                        _pn: _period_abs_ts_tm(_pn)
                                        for _pn in _period_order_tm}

                                    # ── Posição de cada período no tempo de jogo ───────
                                    # Lógica de sobreposição:
                                    #   "2tempo" registra os 10 atletas que continuam
                                    #   por TODO o 2º tempo (ex: 45-95 min).
                                    #   "2tempo1" registra apenas o substituto, que
                                    #   COMEÇA DENTRO do "2tempo" (ex: 65-95 min).
                                    #   → "2tempo1" é sub-período de "2tempo", não
                                    #   um período sequencial após ele.
                                    #
                                    # Detecção: se first_ts(P) está ENTRE first_ts(Q) e
                                    # last_ts(Q) de outro período Q já ativo → P é
                                    # sub-período de Q.
                                    # Offset de P = match_start(Q) + (first_ts(P) - first_ts(Q)) / 60
                                    #
                                    # Períodos principais (sem sobreposição) acumulam
                                    # _cum_min_tm normalmente.
                                    _sorted_by_ts_tm = sorted(
                                        _period_order_tm,
                                        key=lambda _p: _period_abs_tm[_p][0])

                                    _period_start_min_tm: dict = {}
                                    _cum_min_tm = 0.0        # só cresce em períodos principais
                                    _active_mains_tm: list = []  # (nm, ft, lt, match_start)

                                    for _pn_s in _sorted_by_ts_tm:
                                        _ft_s, _lt_s = _period_abs_tm[_pn_s]
                                        _dur_s = (
                                            (_lt_s - _ft_s) / 60.0
                                            if _lt_s > _ft_s else 0.0)

                                        # Descarta períodos principais já encerrados
                                        _active_mains_tm = [
                                            _m for _m in _active_mains_tm
                                            if _m[2] > _ft_s]

                                        # Este período começa DENTRO de algum ativo?
                                        _par_tm = next(
                                            (_m for _m in _active_mains_tm
                                             if _ft_s > _m[1] and _ft_s < _m[2]),
                                            None)

                                        if _par_tm is None:
                                            # Período principal — sem sobreposição
                                            _period_start_min_tm[_pn_s] = _cum_min_tm
                                            _active_mains_tm.append(
                                                (_pn_s, _ft_s, _lt_s, _cum_min_tm))
                                            _cum_min_tm += _dur_s
                                        else:
                                            # Sub-período — entra no meio do pai
                                            _, _par_ft, _, _par_ms = _par_tm
                                            _period_start_min_tm[_pn_s] = (
                                                _par_ms + (_ft_s - _par_ft) / 60.0)

                                    # Ordem cronológica de períodos (por match-time start)
                                    _sorted_period_order_tm = sorted(
                                        _period_order_tm,
                                        key=lambda _p: _period_start_min_tm.get(_p, 0.0))

                                    def _atl_offset_min(_atl_nm: str) -> float:
                                        """Offset = match-time do 1º período (ordem ts) com dados do atleta."""
                                        for _pn_ao in _sorted_by_ts_tm:
                                            if (dados_posicao_por_periodo.get(
                                                    _pn_ao, {}).get(_atl_nm, {}).get('vel')
                                                    or dados_sensor_por_atleta_por_periodo.get(
                                                    _pn_ao, {}).get(_atl_nm)):
                                                return _period_start_min_tm.get(_pn_ao, 0.0)
                                        return 0.0

                                    # Ordena atletas por posição → nome
                                    _atls_ord = sorted(
                                        _team_res.keys(),
                                        key=lambda _a: (_get_pos_atl(_a), _a))

                                    # Pré-calcula offsets uma vez
                                    _offsets_tm = {
                                        _a: _atl_offset_min(_a) for _a in _atls_ord}

                                    # Grade temporal comum no tempo absoluto do jogo
                                    _max_t_tm = max(
                                        float(_t[-1]) + _offsets_tm[_a] + window_minutes
                                        for _a, (_t, _) in _team_res.items()
                                        if _a in _offsets_tm)
                                    _tg = np.arange(0, _max_t_tm + 1/60, 1/60)

                                    _z_mat:   list = []   # normalizado (% máx coletivo)
                                    _raw_mat: list = []   # bruto (para média)
                                    _y_lbl:   list = []

                                    # 1ª passagem — séries brutas no tempo absoluto
                                    for _a in _atls_ord:
                                        _ta, _va = _team_res[_a]
                                        # ← desloca para a linha do tempo real do jogo
                                        _ta_abs = _ta + _offsets_tm[_a]
                                        _vr = np.interp(_tg, _ta_abs, _va,
                                                        left=np.nan, right=np.nan)
                                        _raw_mat.append(_vr)
                                        _y_lbl.append(
                                            f"{_a}  [{_get_pos_atl(_a)}]")

                                    # Máximo coletivo — referência única de todo o time
                                    _col_max = max(
                                        (float(np.nanmax(_vr))
                                         for _vr in _raw_mat
                                         if not np.all(np.isnan(_vr))),
                                        default=1.0,
                                    )
                                    if _col_max <= 0:
                                        _col_max = 1.0

                                    # 2ª passagem — normaliza pelo máximo coletivo
                                    # Linhas inteiramente NaN → mantém NaN (não zeros),
                                    # assim "fora de campo" fica transparente no heatmap
                                    # e distingue-se visualmente de 0% de intensidade.
                                    for _vr in _raw_mat:
                                        if not np.all(np.isnan(_vr)):
                                            # Posições NaN dentro da linha ficam NaN;
                                            # posições ativas → 0–100 % do máx coletivo
                                            _vn = np.where(
                                                np.isnan(_vr),
                                                np.nan,
                                                _vr / _col_max * 100,
                                            )
                                        else:
                                            _vn = np.full_like(_vr, np.nan)
                                        _z_mat.append(_vn)

                                    # ── Pré-calcula bandas de período ──────────────
                                    # Usa _sorted_period_order_tm (ordem por match-time)
                                    # para que as bandas sejam sempre sequenciais.
                                    # Cada banda vai do início do período até o início
                                    # do próximo período (ou até o fim do jogo).
                                    _period_bands_tm = []
                                    for _i_pb, _pn_pb in enumerate(_sorted_period_order_tm):
                                        _ps_pb = _period_start_min_tm[_pn_pb]
                                        _pe_pb = (
                                            _period_start_min_tm[
                                                _sorted_period_order_tm[_i_pb + 1]]
                                            if _i_pb + 1 < len(_sorted_period_order_tm)
                                            else _cum_min_tm
                                        )
                                        _period_bands_tm.append((_pn_pb, _ps_pb, _pe_pb))

                                    def _add_period_bands_tm(_fig, show_labels: bool = True):
                                        """Adiciona bandas alternadas + linhas divisórias + rótulos de período."""
                                        _fc = ['rgba(255,255,255,0.045)', 'rgba(0,0,0,0)']
                                        for _ii, (_nm, _ps, _pe) in enumerate(_period_bands_tm):
                                            _fig.add_vrect(
                                                x0=_ps, x1=_pe,
                                                fillcolor=_fc[_ii % 2],
                                                layer='below', line_width=0,
                                            )
                                            if _ii > 0:  # linha divisória entre períodos
                                                _fig.add_vline(
                                                    x=_ps,
                                                    line_dash='dot',
                                                    line_color='rgba(255,255,255,0.20)',
                                                    line_width=1,
                                                )
                                            if show_labels:
                                                _fig.add_annotation(
                                                    x=(_ps + _pe) / 2,
                                                    y=1.01,
                                                    yref='paper',
                                                    text=_nm,
                                                    showarrow=False,
                                                    font=dict(
                                                        color='rgba(255,255,255,0.40)',
                                                        size=9),
                                                    xanchor='center',
                                                    yanchor='bottom',
                                                )

                                    # ── Paleta compartilhada heatmap / swimlane ─────
                                    # NaN = transparente → mostra o plot_bgcolor (cinza)
                                    # 0 %  = azul-marinho escuro  (ativo, baixa intensidade)
                                    # 50 % = verde médio
                                    # 75 % = âmbar
                                    # 87 % = laranja
                                    # 100% = vermelho
                                    _HT_CS = [
                                        [0.000, 'rgba(15,25,90,1)'],
                                        [0.250, 'rgba(12,90,45,1)'],
                                        [0.500, 'rgba(22,163,74,1)'],
                                        [0.750, 'rgba(234,179,8,1)'],
                                        [0.875, 'rgba(234,88,12,1)'],
                                        [1.000, 'rgba(220,38,38,1)'],
                                    ]
                                    # Fundo cinza-índigo: NaN (transparente) destacado
                                    _HT_BG  = 'rgba(28,28,44,1)'

                                    # ── Heatmap (% do máx coletivo) ────────────────
                                    _fig_ht = _go_tm.Figure(_go_tm.Heatmap(
                                        z=_z_mat,
                                        x=_tg,
                                        y=_y_lbl,
                                        colorscale=_HT_CS,
                                        zmin=0, zmax=100,
                                        colorbar=dict(
                                            title=dict(
                                                text='% do Máx<br>Coletivo',
                                                font=dict(color='white'),
                                            ),
                                            tickfont=dict(color='white'),
                                            tickvals=[0, 25, 50, 75, 100],
                                            ticktext=['0%', '25%', '50%', '75%', '100%'],
                                        ),
                                        hovertemplate=(
                                            "<b>%{y}</b><br>"
                                            "Tempo: %{x:.1f} min<br>"
                                            "Intensidade: %{z:.0f}% do máx coletivo"
                                            "<extra></extra>"
                                        ),
                                    ))
                                    _fig_ht.update_layout(
                                        title=dict(
                                            text=(f"Heatmap de Intensidade — {tipo_metrica}"
                                                  f" (% do Máx Coletivo)"
                                                  f" | Janela {window_minutes} min"),
                                            font=dict(color='white', size=13)),
                                        xaxis=dict(
                                            title='Tempo (minutos)',
                                            color='rgba(255,255,255,0.6)',
                                            gridcolor='rgba(255,255,255,0.05)'),
                                        yaxis=dict(
                                            color='rgba(255,255,255,0.75)',
                                            tickfont=dict(size=10)),
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor=_HT_BG,
                                        height=max(300, 36 * len(_atls_ord) + 120),
                                        margin=dict(l=200, r=100, t=40),
                                    )
                                    _add_period_bands_tm(_fig_ht)
                                    st.plotly_chart(_fig_ht, use_container_width=True)

                                    # ── Swimlane (visualização alternativa opcional) ─
                                    with st.expander(
                                            "🔀 Visualização alternativa — Swimlane por Atleta",
                                            expanded=False):
                                        st.caption(
                                            "Cada faixa horizontal representa um atleta. "
                                            "A cor indica a intensidade relativa ao pico coletivo. "
                                            "Cinza = fora de campo.")

                                        # Reamostrar em bins de 1 min para visual "tile"
                                        _sw_bin   = 1.0
                                        _sw_edges = np.arange(
                                            0, _cum_min_tm + _sw_bin, _sw_bin)
                                        _sw_ctrs  = (_sw_edges[:-1] + _sw_edges[1:]) / 2

                                        # Para cada atleta, média de intensidade por bin.
                                        # Bins sem dado → sentinel -10 (cinza explícito).
                                        _sw_z = []
                                        for _vr_sw in _raw_mat:
                                            _row_sw = []
                                            for _bi in range(len(_sw_ctrs)):
                                                _m = ((_tg >= _sw_edges[_bi]) &
                                                      (_tg <  _sw_edges[_bi + 1]))
                                                _vals = _vr_sw[_m]
                                                _ok   = _vals[~np.isnan(_vals)]
                                                _row_sw.append(
                                                    float(np.mean(_ok))
                                                    if len(_ok) > 0
                                                    else -10.0)
                                            _sw_z.append(_row_sw)

                                        # Colorscale com cinza explícito para -10
                                        # Normalizado: (v+10)/110
                                        # -10 → 0.000  |  0 → 0.091  |  50 → 0.545
                                        # 75  → 0.773  |  100→ 1.000
                                        _sw_cs = [
                                            [0.000, 'rgba(45,45,65,1)'],   # -10: fora
                                            [0.090, 'rgba(45,45,65,1)'],   # limite cinza
                                            [0.091, 'rgba(15,25,90,1)'],   # 0%: azul
                                            [0.364, 'rgba(12,90,45,1)'],   # 30%: verde
                                            [0.545, 'rgba(22,163,74,1)'],  # 50%: verde
                                            [0.773, 'rgba(234,179,8,1)'],  # 75%: âmbar
                                            [0.864, 'rgba(234,88,12,1)'],  # 87%: laranja
                                            [1.000, 'rgba(220,38,38,1)'],  # 100%: vermelho
                                        ]

                                        _fig_sw = _go_tm.Figure(_go_tm.Heatmap(
                                            z=_sw_z,
                                            x=list(_sw_ctrs),
                                            y=_y_lbl,
                                            colorscale=_sw_cs,
                                            zmin=-10, zmax=100,
                                            ygap=4,
                                            xgap=1,
                                            colorbar=dict(
                                                title=dict(
                                                    text='% do Máx<br>Coletivo',
                                                    font=dict(color='white'),
                                                ),
                                                tickfont=dict(color='white'),
                                                tickvals=[0, 25, 50, 75, 100],
                                                ticktext=['0%', '25%', '50%',
                                                          '75%', '100%'],
                                            ),
                                            hovertemplate=(
                                                "<b>%{y}</b><br>"
                                                "Tempo: %{x:.0f}–%{x:.0f} min<br>"
                                                "Intensidade: %{z:.0f}% do máx coletivo"
                                                "<extra></extra>"
                                            ),
                                        ))
                                        _fig_sw.update_layout(
                                            title=dict(
                                                text=(f"Swimlane — {tipo_metrica}"
                                                      f" | bins 1 min"),
                                                font=dict(color='white', size=12)),
                                            xaxis=dict(
                                                title='Tempo (minutos)',
                                                color='rgba(255,255,255,0.6)',
                                                gridcolor='rgba(255,255,255,0.04)'),
                                            yaxis=dict(
                                                color='rgba(255,255,255,0.75)',
                                                tickfont=dict(size=10)),
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor=_HT_BG,
                                            height=max(280,
                                                       28 * len(_atls_ord) + 100),
                                            margin=dict(l=200, r=100, t=36),
                                        )
                                        _add_period_bands_tm(_fig_sw)
                                        st.plotly_chart(
                                            _fig_sw, use_container_width=True)

                                    # ── Média do time (valores brutos, colorida) ────
                                    _tm_mean = np.nanmean(_raw_mat, axis=0)
                                    if not np.all(np.isnan(_tm_mean)):
                                        _tmx = float(np.nanmax(_tm_mean))
                                        _tla = round(_tmx * 0.90, 1)
                                        _tlm = round(_tmx * 0.75, 1)
                                        _tm_cores = [
                                            '#ef4444' if v >= _tla
                                            else '#f59e0b' if v >= _tlm
                                            else '#22c55e'
                                            for v in _tm_mean
                                        ]
                                        _fig_avg = _go_tm.Figure()
                                        _fig_avg.add_trace(_go_tm.Scatter(
                                            x=_tg, y=_tm_mean,
                                            mode='markers',
                                            marker=dict(color=_tm_cores, size=3),
                                            name='Média do Time',
                                            hovertemplate=(
                                                "Tempo: %{x:.1f} min<br>"
                                                f"Média Time: %{{y:.1f}} {_unidade_jan}"
                                                "<extra></extra>"
                                            ),
                                        ))
                                        _fig_avg.add_hline(
                                            y=_tla, line_dash='dash',
                                            line_color='rgba(239,68,68,0.50)',
                                            annotation_text=f"Alta ≥{_tla}")
                                        _fig_avg.add_hline(
                                            y=_tlm, line_dash='dot',
                                            line_color='rgba(245,158,11,0.50)',
                                            annotation_text=f"Média-Alta ≥{_tlm}")
                                        _add_period_bands_tm(_fig_avg)
                                        _fig_avg.update_layout(
                                            title=dict(
                                                text=(f"Intensidade Média do Time — "
                                                      f"{tipo_metrica} ({_unidade_jan})"
                                                      f" | Rolling {window_minutes} min"),
                                                font=dict(color='white', size=13)),
                                            xaxis=dict(
                                                title='Tempo (minutos)',
                                                color='rgba(255,255,255,0.6)',
                                                gridcolor='rgba(255,255,255,0.07)'),
                                            yaxis=dict(
                                                title=f'{tipo_metrica} ({_unidade_jan})',
                                                color='rgba(255,255,255,0.6)',
                                                gridcolor='rgba(255,255,255,0.07)'),
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            height=300,
                                        )
                                        st.plotly_chart(_fig_avg, use_container_width=True)

                                        # ── Cards de esforços coletivos ────────────
                                        _tg_v   = _tg[~np.isnan(_tm_mean)]
                                        _tmm_v  = _tm_mean[~np.isnan(_tm_mean)]
                                        _alta_ev_tm, _media_ev_tm = (
                                            encontrar_eventos_nao_sobrepostos(
                                                list(_tg_v), list(_tmm_v),
                                                window_minutes, _tla, _tlm, _tmx,
                                            ) if len(_tg_v) > 1 else ([], [])
                                        )
                                        _alta_cnt_tm  = len(_alta_ev_tm)
                                        _media_cnt_tm = len(_media_ev_tm)

                                        _card_alta_tm = f"""
    <div style="
        background: linear-gradient(135deg,rgba(220,38,38,0.18) 0%,rgba(153,27,27,0.08) 100%);
        border: 1px solid rgba(239,68,68,0.55); border-radius: 18px;
        padding: 32px 24px 26px; text-align: center;
        box-shadow: 0 0 32px rgba(220,38,38,0.22),0 2px 8px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(255,255,255,0.07);
        position: relative; overflow: hidden;">
      <div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;
                  border-radius:50%;background:rgba(220,38,38,0.10);pointer-events:none;"></div>
      <div style="font-size:11px;font-weight:600;letter-spacing:2px;
                  color:rgba(255,255,255,0.5);text-transform:uppercase;margin-bottom:10px;">
        Alta Intensidade — Time</div>
      <div style="font-size:72px;font-weight:800;color:#f87171;line-height:1;
                  text-shadow:0 0 24px rgba(248,113,113,0.5);">{_alta_cnt_tm}</div>
      <div style="font-size:12px;color:rgba(255,255,255,0.38);margin-top:14px;line-height:1.6;">
        janelas coletivas com <strong style="color:rgba(248,113,113,0.8);">
        {tipo_metrica} ≥ {_tla} {_unidade_jan}</strong><br>
        &gt; 90% do pico coletivo ({_tmx:.1f} {_unidade_jan})
      </div>
    </div>"""

                                        _card_media_tm = f"""
    <div style="
        background: linear-gradient(135deg,rgba(202,138,4,0.18) 0%,rgba(133,77,14,0.08) 100%);
        border: 1px solid rgba(234,179,8,0.50); border-radius: 18px;
        padding: 32px 24px 26px; text-align: center;
        box-shadow: 0 0 32px rgba(202,138,4,0.22),0 2px 8px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(255,255,255,0.07);
        position: relative; overflow: hidden;">
      <div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;
                  border-radius:50%;background:rgba(202,138,4,0.10);pointer-events:none;"></div>
      <div style="font-size:11px;font-weight:600;letter-spacing:2px;
                  color:rgba(255,255,255,0.5);text-transform:uppercase;margin-bottom:10px;">
        Média-Alta Intensidade — Time</div>
      <div style="font-size:72px;font-weight:800;color:#fbbf24;line-height:1;
                  text-shadow:0 0 24px rgba(251,191,36,0.5);">{_media_cnt_tm}</div>
      <div style="font-size:12px;color:rgba(255,255,255,0.38);margin-top:14px;line-height:1.6;">
        janelas coletivas com <strong style="color:rgba(251,191,36,0.8);">
        {_tlm} ≤ {tipo_metrica} &lt; {_tla} {_unidade_jan}</strong><br>
        75–90% do pico coletivo ({_tmx:.1f} {_unidade_jan})
      </div>
    </div>"""

                                        _ctm1, _ctm2 = st.columns(2)
                                        with _ctm1:
                                            st.markdown(_card_alta_tm,
                                                        unsafe_allow_html=True)
                                        with _ctm2:
                                            st.markdown(_card_media_tm,
                                                        unsafe_allow_html=True)

                                        # ── Feedback coletivo ───────────────────────
                                        _n_tot_tm  = _alta_cnt_tm + _media_cnt_tm
                                        _s_tot_tm  = "s" if _n_tot_tm    != 1 else ""
                                        _s_alt_tm  = "s" if _alta_cnt_tm != 1 else ""
                                        _s_med_tm  = "s" if _media_cnt_tm!= 1 else ""
                                        _dur_tot_tm = (
                                            float(_tg_v[-1]) + window_minutes
                                            if len(_tg_v) else 0.0)
                                        _dh = int(_dur_tot_tm // 60)
                                        _dm = int(_dur_tot_tm % 60)
                                        _dur_str_tm = (
                                            f"{_dh}h {_dm:02d}min" if _dh
                                            else f"{_dm} min")
                                        st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(25,35,55,0.65) 0%,rgba(15,25,45,0.45) 100%);
     border:1px solid rgba(255,255,255,0.09);
     border-left:3px solid rgba(93,173,226,0.55);
     border-radius:10px;padding:14px 20px;margin:20px 0 10px 0;
     font-size:0.875rem;line-height:1.75;color:rgba(255,255,255,0.72);">
  💬 &nbsp;<strong style="color:white">O time</strong> apresentou
  <strong style="color:#f87171">{_alta_cnt_tm}</strong> período{_s_alt_tm}
  de <span style="color:#f87171">alta intensidade coletiva</span> e
  <strong style="color:#fbbf24">{_media_cnt_tm}</strong>
  de <span style="color:#fbbf24">média-alta</span> —
  totalizando <strong style="color:white">{_n_tot_tm} janela{_s_tot_tm}
  distinta{_s_tot_tm}</strong> de {window_minutes} min com
  <em>{tipo_metrica}</em> médio ≥
  <strong>{_tlm:.1f} {_unidade_jan}</strong>,
  ao longo de <strong style="color:#5dade2">{_dur_str_tm}</strong>
  analisados. Pico coletivo máximo:
  <strong style="color:white">{_tmx:.1f} {_unidade_jan}</strong>.
</div>""", unsafe_allow_html=True)

                                    # ── Tabela de esforços coletivos ────────────────
                                    st.markdown(
                                        "#### 📋 Esforços Coletivos — Média-Alta e Alta Intensidade")
                                    st.caption(
                                        f"Cada linha é uma janela de **{window_minutes} min** "
                                        "distinta e não-sobreposta da **média do time**, "
                                        "selecionada pelo pico máximo coletivo.")

                                    def _periodo_para_t_tm(_t_m):
                                        for (_nm_p, _ps_p, _pe_p) in _period_bands_tm:
                                            if _ps_p <= _t_m <= _pe_p + 0.1:
                                                return _nm_p
                                        if _period_bands_tm:
                                            return min(
                                                _period_bands_tm,
                                                key=lambda _b: abs(_b[1] - _t_m))[0]
                                        return None

                                    _todos_ev_tm = (
                                        [dict(_e, _cat='alta')  for _e in _alta_ev_tm] +
                                        [dict(_e, _cat='media') for _e in _media_ev_tm]
                                    )
                                    _todos_ev_tm.sort(
                                        key=lambda _e: _e['valor'], reverse=True)

                                    if _todos_ev_tm:
                                        _rows_tm = []
                                        for _rk_tm, _ev_tm in enumerate(_todos_ev_tm, 1):
                                            _per_nm = _periodo_para_t_tm(
                                                _ev_tm.get('t_ini_min', 0.0))
                                            _row_tm = {
                                                '#': _rk_tm,
                                                'Início': _ev_tm['inicio'],
                                                'Fim':    _ev_tm['fim'],
                                            }
                                            if _per_nm:
                                                _row_tm['Período'] = _per_nm
                                            _row_tm.update({
                                                f'{tipo_metrica} Médio ({_unidade_jan})':
                                                    _ev_tm['valor'],
                                                '↓ % do Pico Coletivo':
                                                    _ev_tm['pct_max'],
                                                'Intensidade':
                                                    _ev_tm['intensidade'],
                                            })
                                            _rows_tm.append(_row_tm)

                                        _df_ev_tm = pd.DataFrame(_rows_tm)

                                        def _style_ev_tm(row):
                                            if ('Alta Intensidade' in str(
                                                    row.get('Intensidade', ''))
                                                    and 'Média' not in str(
                                                    row.get('Intensidade', ''))):
                                                return ['background-color:rgba(239,68,68,0.12)'] * len(row)
                                            elif 'Média-Alta' in str(row.get('Intensidade', '')):
                                                return ['background-color:rgba(245,158,11,0.10)'] * len(row)
                                            return [''] * len(row)

                                        _fmt_tm = {
                                            f'{tipo_metrica} Médio ({_unidade_jan})': '{:.1f}',
                                            '↓ % do Pico Coletivo': '{:.1f}%',
                                        }
                                        # ── Tabela com seleção de linha ────────────
                                        # Clicar na linha aciona a animação abaixo.
                                        _ev_tbl_ev = st.dataframe(
                                            _df_ev_tm.style.apply(
                                                _style_ev_tm, axis=1).format(_fmt_tm),
                                            use_container_width=True,
                                            height=min(600, 40 + len(_rows_tm) * 36),
                                            on_select="rerun",
                                            selection_mode="single-row",
                                            key="ev_tm_table_sel",
                                        )
                                        if not st.session_state.get('modo_apresentacao'):
                                            st.download_button(
                                                "📥 Exportar Esforços Coletivos (CSV)",
                                                _df_ev_tm.to_csv(index=False).encode('utf-8'),
                                                f"esforcos_coletivos_{tipo_metrica}"
                                                f"_{window_minutes}min.csv",
                                                mime='text/csv',
                                                key="dl_ef_team",
                                            )

                                        # ── Animação do esforço selecionado ────────
                                        if dados_posicao_por_periodo:
                                            # Índice da linha selecionada (padrão: 0)
                                            _sel_ao_idx = (
                                                _ev_tbl_ev.selection.rows[0]
                                                if (hasattr(_ev_tbl_ev, 'selection')
                                                    and _ev_tbl_ev.selection.rows)
                                                else 0
                                            )
                                            _ev_anim = _todos_ev_tm[_sel_ao_idx]
                                            _t_ini_ao = _ev_anim.get('t_ini_min', 0.0)
                                            _per_ao_lbl = (
                                                _periodo_para_t_tm(_t_ini_ao) or '?')

                                            st.markdown("---")
                                            st.markdown(
                                                "#### 🎬 Animar Esforço Coletivo no Campo")
                                            st.caption(
                                                f"**Clique em uma linha** da tabela para "
                                                f"selecionar o esforço. Exibindo: "
                                                f"**{_ev_anim['inicio']}→"
                                                f"{_ev_anim['fim']}** "
                                                f"({_per_ao_lbl})")

                                            # Config do campo (necessária antes do loop
                                            # para o fallback lats/lons → xs/ys)
                                            _anim_cfg_ao = None
                                            for _hk_ao in list(st.session_state.keys()):
                                                if (_hk_ao.startswith("campo_cfg__")
                                                        and isinstance(
                                                        st.session_state[_hk_ao], dict)):
                                                    _anim_cfg_ao = st.session_state[_hk_ao]
                                                    break
                                            _fl_ao = float(
                                                _anim_cfg_ao.get('fl', 105)
                                                if _anim_cfg_ao else 105)
                                            _fw_ao = float(
                                                _anim_cfg_ao.get('fw', 68)
                                                if _anim_cfg_ao else 68)

                                            # Fim de cada período em match-time
                                            _pend_ao = {
                                                _pn_ao2: (
                                                    _period_start_min_tm[_pn_ao2]
                                                    + (_period_abs_tm[_pn_ao2][1]
                                                       - _period_abs_tm[_pn_ao2][0])
                                                    / 60.0)
                                                for _pn_ao2 in _period_order_tm
                                            }

                                            # Paleta de cores
                                            _pal_ao = [
                                                '#FF6B6B','#4ECDC4','#45B7D1','#96CEB4',
                                                '#FFEAA7','#DDA0DD','#98D8C8','#F7DC6F',
                                                '#BB8FCE','#76D7C4','#F1948A','#85C1E9',
                                                '#82E0AA','#F8C471','#AED6F1',
                                            ]

                                            # Para cada atleta: acha período que cobre
                                            # t_ini_ao e extrai o segmento GPS correto.
                                            # Hz calculado por len(xs)/dur_s — robusto
                                            # mesmo com ts_pos = 0 (não confiável).
                                            # Fallback: lats/lons → gps_para_campo_coords
                                            # Exclui goleiro: animação mostra
                                            # apenas os jogadores de linha
                                            _jan_atls_linha = [
                                                _a for _a in _jan_atletas
                                                if _get_pos_grupo(
                                                    _get_pos_atl(_a))[0] != 'Goleiro'
                                            ]

                                            _anim_map = {}
                                            _ao_hz_ref = 10.0  # Hz de ref para frames
                                            for _ci_ao, _atl_ao in enumerate(
                                                    _jan_atls_linha):
                                                for _pn_ao3 in _sorted_by_ts_tm:
                                                    _pos_ao = dados_posicao_por_periodo.get(
                                                        _pn_ao3, {}).get(_atl_ao, {})
                                                    _xs_ao = list(
                                                        _pos_ao.get('xs', []))
                                                    _ys_ao = list(
                                                        _pos_ao.get('ys', []))

                                                    # Fallback: coordenadas GPS → campo
                                                    if not _xs_ao and _anim_cfg_ao:
                                                        _lts_ao = _pos_ao.get('lats', [])
                                                        _lns_ao = _pos_ao.get('lons', [])
                                                        if _lts_ao and _lns_ao:
                                                            try:
                                                                _xs_ao, _ys_ao = (
                                                                    gps_para_campo_coords(
                                                                        _lts_ao, _lns_ao,
                                                                        _anim_cfg_ao))
                                                            except Exception:
                                                                _applog.log_debug_exc()

                                                    if not _xs_ao:
                                                        continue

                                                    _ps_ao = _period_start_min_tm.get(
                                                        _pn_ao3, 0.0)
                                                    _pe_ao = _pend_ao.get(_pn_ao3, _ps_ao)
                                                    if not (_ps_ao <= _t_ini_ao <= _pe_ao):
                                                        continue

                                                    # Hz por comprimento do array
                                                    _dur_s_ao = (
                                                        (_period_abs_tm[_pn_ao3][1]
                                                         - _period_abs_tm[_pn_ao3][0])
                                                        if _period_abs_tm.get(_pn_ao3)
                                                        else 0.0)
                                                    _hz_ao = (
                                                        len(_xs_ao) / _dur_s_ao
                                                        if _dur_s_ao > 0
                                                        else 10.0)
                                                    _ao_hz_ref = _hz_ao

                                                    _off_s_ao = (
                                                        (_t_ini_ao - _ps_ao) * 60.0)
                                                    _n_smp_ao = max(
                                                        2,
                                                        int(window_minutes * 60 * _hz_ao))
                                                    _is_ao = int(_off_s_ao * _hz_ao)
                                                    _ie_ao = min(
                                                        _is_ao + _n_smp_ao,
                                                        len(_xs_ao))

                                                    if 0 <= _is_ao < len(_xs_ao):
                                                        _vs_ao = list(
                                                            _pos_ao.get('vel', []))
                                                        _anim_map[_atl_ao] = {
                                                            'xs': _xs_ao[_is_ao:_ie_ao],
                                                            'ys': (
                                                                _ys_ao[_is_ao:_ie_ao]
                                                                if _ys_ao
                                                                else [0]*(_ie_ao-_is_ao)),
                                                            'vel': (
                                                                _vs_ao[_is_ao:_ie_ao]
                                                                if _vs_ao
                                                                else [0]*(_ie_ao-_is_ao)),
                                                            'color': _pal_ao[
                                                                _ci_ao % len(_pal_ao)],
                                                            'label': (
                                                                _atl_ao.split()[-1][:10]
                                                                if _atl_ao.split()
                                                                else _atl_ao[:10]),
                                                        }
                                                    break

                                            if len(_anim_map) < 2:
                                                st.info(
                                                    "GPS insuficiente para este esforço "
                                                    f"({len(_anim_map)} atleta(s) com "
                                                    "dados de posição). Verifique se o "
                                                    "campo foi configurado e se os dados "
                                                    "GPS foram importados.")
                                            else:
                                                _per_ao_lbl = _periodo_para_t_tm(
                                                    _t_ini_ao) or '?'
                                                _fig_ao = desenhar_campo_futebol_bonito(
                                                    field_length=_fl_ao,
                                                    field_width=_fw_ao,
                                                    title=(
                                                        f"🎬 {_ev_anim['inicio']}"
                                                        f"→{_ev_anim['fim']}"
                                                        f" | {_per_ao_lbl}"
                                                        f" | {_ev_anim['valor']:.1f}"
                                                        f" {_unidade_jan}"),
                                                )

                                                _atls_ao = list(_anim_map.keys())
                                                _tidxs_ao = []
                                                for _pa_ao in _atls_ao:
                                                    _wd_ao = _anim_map[_pa_ao]
                                                    _fig_ao.add_trace(go.Scatter(
                                                        x=[_wd_ao['xs'][0]]
                                                            if _wd_ao['xs'] else [0],
                                                        y=[_wd_ao['ys'][0]]
                                                            if _wd_ao['ys'] else [0],
                                                        mode='markers+text',
                                                        marker=dict(
                                                            size=20,
                                                            color=_wd_ao['color'],
                                                            symbol='circle',
                                                            line=dict(color='white',
                                                                       width=2)),
                                                        text=[_wd_ao['label']],
                                                        textposition='top center',
                                                        textfont=dict(color='white',
                                                                       size=8),
                                                        name=_pa_ao, showlegend=True,
                                                    ))
                                                    _tidxs_ao.append(
                                                        len(_fig_ao.data) - 1)

                                                _wl_ao = max(
                                                    len(_anim_map[a]['xs'])
                                                    for a in _atls_ao)
                                                _step_ao = max(1, _wl_ao // 80)
                                                _fr_ao = list(
                                                    range(0, _wl_ao, _step_ao))
                                                if (_fr_ao and
                                                        _fr_ao[-1] != _wl_ao - 1):
                                                    _fr_ao.append(_wl_ao - 1)

                                                _frames_ao = []
                                                for _fi_ao in _fr_ao:
                                                    _ts_ao = _fi_ao / _ao_hz_ref
                                                    _mm_ao = int(_ts_ao // 60)
                                                    _ss_ao = int(_ts_ao % 60)
                                                    _fd_ao = []
                                                    for _pa_ao in _atls_ao:
                                                        _wd_ao = _anim_map[_pa_ao]
                                                        _xi_ao = (
                                                            _wd_ao['xs'][_fi_ao]
                                                            if _fi_ao < len(_wd_ao['xs'])
                                                            else (_wd_ao['xs'][-1]
                                                                  if _wd_ao['xs'] else 0))
                                                        _yi_ao = (
                                                            _wd_ao['ys'][_fi_ao]
                                                            if _fi_ao < len(_wd_ao['ys'])
                                                            else (_wd_ao['ys'][-1]
                                                                  if _wd_ao['ys'] else 0))
                                                        _fd_ao.append(go.Scatter(
                                                            x=[_xi_ao], y=[_yi_ao],
                                                            mode='markers+text',
                                                            marker=dict(
                                                                size=20,
                                                                color=_wd_ao['color'],
                                                                symbol='circle',
                                                                line=dict(color='white',
                                                                           width=2)),
                                                            text=[_wd_ao['label']],
                                                            textposition='top center',
                                                            textfont=dict(color='white',
                                                                           size=8),
                                                        ))
                                                    _frames_ao.append(go.Frame(
                                                        data=_fd_ao,
                                                        traces=_tidxs_ao,
                                                        name=str(_fi_ao),
                                                        layout=go.Layout(title=dict(
                                                            text=(
                                                                f"🎬 "
                                                                f"{_ev_anim['inicio']}"
                                                                f"→{_ev_anim['fim']}"
                                                                f" | ⏱️ +"
                                                                f"{_mm_ao}:{_ss_ao:02d}"
                                                                f" min"
                                                            ),
                                                            font=dict(color='white',
                                                                       size=12),
                                                        )),
                                                    ))

                                                _fig_ao.frames = _frames_ao
                                                _fig_ao.update_layout(
                                                    height=580,
                                                    updatemenus=[dict(
                                                        type='buttons',
                                                        showactive=False,
                                                        y=0, x=0.5,
                                                        xanchor='center',
                                                        buttons=[
                                                            dict(
                                                                label='▶ Play',
                                                                method='animate',
                                                                args=[None, dict(
                                                                    frame=dict(
                                                                        duration=100,
                                                                        redraw=True),
                                                                    fromcurrent=True,
                                                                    transition=dict(
                                                                        duration=100,
                                                                        easing='linear'),
                                                                    mode='immediate')]),
                                                            dict(
                                                                label='⏸ Pause',
                                                                method='animate',
                                                                args=[[None], dict(
                                                                    frame=dict(
                                                                        duration=0,
                                                                        redraw=False),
                                                                    mode='immediate')]),
                                                        ],
                                                    )],
                                                    sliders=[dict(
                                                        steps=[
                                                            dict(
                                                                args=[[f.name],
                                                                      dict(
                                                                          frame=dict(
                                                                              duration=0,
                                                                              redraw=True),
                                                                          mode='immediate')],
                                                                method='animate',
                                                                label='',
                                                            )
                                                            for f in _frames_ao
                                                        ],
                                                        x=0.0, y=-0.05, len=1.0,
                                                        currentvalue=dict(visible=False),
                                                    )],
                                                    legend=dict(
                                                        orientation='h',
                                                        yanchor='bottom', y=-0.30,
                                                        xanchor='center', x=0.5,
                                                        font=dict(color='white', size=8),
                                                    ),
                                                )
                                                st.plotly_chart(
                                                    _fig_ao,
                                                    use_container_width=True)
                                    else:
                                        st.info(
                                            "Nenhum esforço coletivo de média-alta "
                                            "ou alta intensidade encontrado.")
