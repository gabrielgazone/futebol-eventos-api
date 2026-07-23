# -*- coding: utf-8 -*-
"""Aba extraída de main() (P4 — page-decomposition)."""
from __future__ import annotations

from bands import _ACC_KEY_TO_NUM
from config import _CHAVE_COMBINADO
from diagnostics import _PROV_LABELS
from bands import _bandas_acc_ativas
from bands import _bandas_vel_ativas
from diagnostics import _diag_log
from bands import _fmt_num_banda
import metrics as _mtr
from field import _vmax_individual_kmh
from analysis import acc_series_from_vel
from analysis import combinar_periodos_continuo
from field import desenhar_campo_futebol_bonito
from analysis import detectar_acoes_acc_idx
from analysis import get_min_dur_s
import plotly.graph_objects as go
from field import gps_para_campo_coords
import numpy as np
import pandas as pd
import streamlit as st


def render_wcs(_REL_VEL_BANDAS, _SENSOR_HZ, _ok_ld, dados_efforts_acc_por_periodo, dados_posicao_por_periodo, dados_sensor_por_atleta_por_periodo, resultados_por_periodo):
                st.subheader("⚡ Worst-Case Scenario")
                st.caption(
                    "Identifica a **janela temporal de maior exigência física** de cada atleta "
                    "buscando o pior cenário em todos os períodos da atividade. "
                    "Base para prescrição de cargas de treino acima do jogo. "
                    "_(Delaney et al., 2018; Martín-García et al., 2018)_"
                )

                if _ok_ld == 0 or not dados_posicao_por_periodo:
                    st.info("Carregue os dados para usar a análise de Worst-Case Scenario.")
                else:
                    # ── Controles ──────────────────────────────────────────────────
                    _wcs2_c1, _wcs2_c2 = st.columns([1, 2])
                    with _wcs2_c1:
                        _wcs2_min = st.slider(
                            "⏱️ Janela temporal (min)", 1, 15, 5,
                            key="wcs2_janela",
                            help="Duração da janela rolante para identificar o pior cenário"
                        )
                    with _wcs2_c2:
                        _wcs2_metric_opts = [
                            "Distância (m)",
                            "🏃 Velocidade (bandas)",
                            "💥 Ações Acel/Desacel (efforts)",
                            "Velocidade Máx (km/h)",
                            "PlayerLoad",
                        ]
                        _wcs2_metric = st.selectbox(
                            "📊 Variável", _wcs2_metric_opts,
                            key="wcs2_metric_sel",
                            help="Métrica para identificar a janela de maior exigência. "
                                 "🏃 Velocidade = distância nas bandas escolhidas. "
                                 "💥 Ações Acel/Desacel = nº de esforços (ações reais da "
                                 "Catapult) de aceleração/desaceleração no pior minuto."
                        )

                        # ── Seleção de bandas (aparece ao escolher Velocidade/Aceleração) ──
                        _sel_vel_bands = []   # lista de dicts {min,max} das bandas de velocidade marcadas
                        _sel_acc_bands = []   # idem para aceleração
                        _sel_acc_boxes = set()  # caixas Gen2 (1..8) selecionadas
                        _sel_vel_pct = []       # (P9) faixas relativas (fração da Vmáx)
                        _wcs2_vel_rel = False   # (P9) modo % Vmáx individual
                        if _wcs2_metric == "🏃 Velocidade (bandas)":
                            _wcs2_vel_rel = st.checkbox(
                                "% da Vmáx individual", value=False, key="wcs2_vel_rel",
                                help="(P9) Em vez das bandas absolutas da conta, usa faixas "
                                     "relativas à Vmáx de CADA atleta (histórico da conta, "
                                     "fallback no pico da sessão).")
                            if _wcs2_vel_rel:
                                _rel_pick_w = st.multiselect(
                                    "🎚️ Faixas (% da Vmáx)",
                                    list(_REL_VEL_BANDAS.keys()),
                                    default=[_k for _k in _REL_VEL_BANDAS
                                             if _REL_VEL_BANDAS[_k][0] >= 0.7],
                                    key="wcs2_vel_rel_bands",
                                    help="A distância (m) é acumulada quando a velocidade está "
                                         "dentro da faixa relativa do próprio atleta.")
                                _sel_vel_pct = [_REL_VEL_BANDAS[_s] for _s in _rel_pick_w]
                                if not _sel_vel_pct:
                                    st.info("Selecione ao menos uma faixa.")
                            else:
                                _bv_act = _bandas_vel_ativas()
                                _bv_lbl = {}
                                for _bk, _bd in _bv_act.items():
                                    _mx = float(_bd.get('max', 9999))
                                    _faixa = (f"&gt;{_fmt_num_banda(_bd.get('min', 0))}"
                                              if _mx >= 9000
                                              else f"{_fmt_num_banda(_bd.get('min', 0))}-{_fmt_num_banda(_mx)}")
                                    _bv_lbl[f"B{_bk} — {_faixa} km/h"] = _bk
                                _bv_pick = st.multiselect(
                                    "🎚️ Bandas de velocidade a visualizar",
                                    list(_bv_lbl.keys()),
                                    default=list(_bv_lbl.keys()),
                                    key="wcs2_vel_bands",
                                    help="A distância percorrida (m) é acumulada apenas enquanto a "
                                         "velocidade está dentro das bandas selecionadas."
                                )
                                _sel_vel_bands = [_bv_act[_bv_lbl[_s]] for _s in _bv_pick]
                                if not _sel_vel_bands:
                                    st.info("Selecione ao menos uma banda de velocidade.")
                        elif _wcs2_metric == "💥 Ações Acel/Desacel (efforts)":
                            _ba_act = _bandas_acc_ativas()
                            # Duas caixas separadas: Aceleração (A*) e Desaceleração (D*).
                            _acc_lbl = {_ba_act[k]['label']: k
                                        for k in _ba_act if str(k).startswith('A')}
                            _dec_lbl = {_ba_act[k]['label']: k
                                        for k in _ba_act if str(k).startswith('D')}
                            _cwa, _cwd = st.columns(2)
                            with _cwa:
                                _acc_pick = st.multiselect(
                                    "🚀 Aceleração",
                                    list(_acc_lbl.keys()),
                                    default=list(_acc_lbl.keys()),
                                    key="wcs2_acc_bands_pos",
                                    help="Ações de aceleração (Gen2Acceleration · caixas 6,7,8)."
                                )
                            with _cwd:
                                _dec_pick = st.multiselect(
                                    "🛑 Desaceleração",
                                    list(_dec_lbl.keys()),
                                    default=list(_dec_lbl.keys()),
                                    key="wcs2_acc_bands_neg",
                                    help="Ações de desaceleração (Gen2Acceleration · caixas 3,2,1)."
                                )
                            _sel_acc_bands = (
                                [_ba_act[_acc_lbl[_s]] for _s in _acc_pick]
                                + [_ba_act[_dec_lbl[_s]] for _s in _dec_pick]
                            )
                            _sel_acc_boxes = (
                                {_ACC_KEY_TO_NUM[_acc_lbl[_s]] for _s in _acc_pick
                                 if _acc_lbl[_s] in _ACC_KEY_TO_NUM}
                                | {_ACC_KEY_TO_NUM[_dec_lbl[_s]] for _s in _dec_pick
                                   if _dec_lbl[_s] in _ACC_KEY_TO_NUM}
                            )
                            st.caption(
                                "Conta o **nº de ações** de acel/desacel na janela — dos "
                                "*acceleration_efforts* da Catapult quando disponíveis, senão "
                                "detectadas no **sinal de aceleração do sensor** (mesma fonte da "
                                "aba Neuromuscular). O pior minuto é a janela com mais ações."
                            )
                            if not _sel_acc_bands:
                                st.info("Selecione ao menos uma banda de aceleração ou desaceleração.")

                    # ── Detecta Hz real a partir dos timestamps ─────────────────────
                    def _detect_hz(_periodos_list, _dppp):
                        """Estima a frequência de amostragem (Hz) como nº de amostras ÷ duração.

                        Usar contagem/duração (em vez da mediana das diferenças) é robusto
                        quando os timestamps vêm arredondados para segundos inteiros mas há
                        vários pontos por segundo — caso em que a mediana das diferenças daria
                        1 Hz erroneamente e a integração de distância superestimaria ~Nx."""
                        # (P1) Delegado ao motor único: metrics.estimate_hz.
                        _series = []
                        for _pnn in _periodos_list[:5]:
                            for _adat in list(_dppp.get(_pnn, {}).values())[:5]:
                                # Nativo: ts_pos · GPS-only: ts_gps
                                _tss = _adat.get('ts_pos', []) or _adat.get('ts_gps', [])
                                if len(_tss) > 20:
                                    _series.append(_tss)
                        return _mtr.estimate_hz(_series, default=10.0)

                    _wcs2_periodos_tmp = [
                        k for k in dados_posicao_por_periodo
                        if k != _CHAVE_COMBINADO
                    ]
                    _wcs2_hz = _detect_hz(_wcs2_periodos_tmp, dados_posicao_por_periodo)
                    _wcs2_n  = max(2, int(_wcs2_min * 60 * _wcs2_hz))
                    st.caption(f"📡 Frequência GPS detectada: **{_wcs2_hz} Hz** — janela = {_wcs2_n} amostras")

                    # ── Config do campo (necessário para fallback GPS) ──────────────
                    _wcs2_cfg = None
                    for _hkw in list(st.session_state.keys()):
                        if (_hkw.startswith("campo_cfg__")
                                and isinstance(st.session_state[_hkw], dict)
                                and 'fl' in st.session_state[_hkw]):
                            _wcs2_cfg = st.session_state[_hkw]
                            break

                    # ── Cálculo do WCS por atleta ───────────────────────────────────
                    _wcs2_rows = []
                    _wcs2_segs = {}  # atleta → {xn, yn, vel} para animação
                    _wcs2_prov = {}  # (P4) fonte das ações → nº de atletas

                    _wcs2_periodos = _wcs2_periodos_tmp
                    _wcs2_athletes = sorted(set(
                        a for _pn in _wcs2_periodos
                        for a in dados_posicao_por_periodo.get(_pn, {}).keys()
                    ))

                    # ── Diagnóstico: quantos efforts de aceleração existem? ─────────
                    if _wcs2_metric == "💥 Ações Acel/Desacel (efforts)":
                        _tot_acc_eff = sum(
                            len(dados_efforts_acc_por_periodo.get(_pn, {}).get(_a, []) or [])
                            for _pn in _wcs2_periodos for _a in _wcs2_athletes
                        )
                        if _tot_acc_eff > 0:
                            st.caption(
                                f"💥 **{_tot_acc_eff} ações** de acel/desacel encontradas "
                                f"nos efforts da conta — contadas por janela para achar o pior minuto."
                            )
                        else:
                            _dur_fb = get_min_dur_s()
                            st.info(
                                "ℹ️ A API não retornou *acceleration_efforts* para estes "
                                "atletas/períodos (dispositivo sem aceleração nativa). "
                                f"Contando **ações discretas** detectadas no sinal de "
                                f"aceleração (dv/dt): cada entrada sustentada por "
                                f"≥ **{_dur_fb:.1f} s** numa banda conta como 1 ação "
                                "(ajuste a duração mínima na sidebar)."
                            )

                    def _wcalc_wcs(_sv, _n, _is_vm):
                        if len(_sv) < max(_n, 2):
                            return 0.0
                        if _is_vm:
                            from collections import deque as _DqW
                            _dqW = _DqW(); _bvW = -1.0
                            for _iW in range(len(_sv)):
                                while _dqW and _sv[_dqW[-1]] <= _sv[_iW]:
                                    _dqW.pop()
                                _dqW.append(_iW)
                                if _dqW[0] <= _iW - _n:
                                    _dqW.popleft()
                                if _iW >= _n - 1:
                                    _cW = _sv[_dqW[0]]
                                    if _cW > _bvW:
                                        _bvW = _cW
                            return _bvW
                        else:
                            # (P1) canônico: metrics.rolling_sum
                            _rw = _mtr.rolling_sum(_sv, _n)
                            return max(_rw) if _rw else 0.0

                    for _wa in _wcs2_athletes:
                        _wx, _wy, _wv, _wac, _wts, _wper = [], [], [], [], [], []
                        for _pn in _wcs2_periodos:
                            _da  = dados_posicao_por_periodo.get(_pn, {}).get(_wa, {})
                            _xs  = _da.get('xs', [])
                            _ys  = _da.get('ys', [])
                            _vl  = _da.get('vel', [])         # km/h
                            _ac  = _da.get('acc', [])
                            _ts  = _da.get('ts_pos', [])
                            # _nn usa xs/ys como referência — vel pode estar vazia
                            _nn  = min(len(_xs), len(_ys))

                            if _nn == 0 and _wcs2_cfg:
                                # Fallback GPS: converte lats/lons para coordenadas de campo
                                _lp = _da.get('lats', [])
                                _lo = _da.get('lons', [])
                                if _lp and _lo:
                                    try:
                                        _gx2, _gy2 = gps_para_campo_coords(_lp, _lo, _wcs2_cfg)
                                        _xs = _gx2;  _ys = _gy2
                                        _vl = _da.get('vels_gps', [0.0] * len(_gx2))
                                        _ts = _da.get('ts_gps', []) or _ts   # timestamps Unix reais (GPS)
                                        _nn = min(len(_xs), len(_ys))
                                    except Exception:
                                        _nn = 0
                                        _diag_log('WCS', f"{_wa}: falha ao projetar "
                                                         f"GPS→campo no período {_pn}")

                            if _nn > 0:
                                # Preenche vel/acc/ts com zeros se ausentes ou mais curtos
                                _vl_pad  = list(_vl[:_nn])  + [0.0] * max(0, _nn - len(_vl))
                                _ac_pad  = list(_ac[:_nn])  + [0.0] * max(0, _nn - len(_ac))
                                _ts_pad  = list(_ts[:_nn])  + [0.0] * max(0, _nn - len(_ts))
                                _wx  += list(_xs[:_nn])
                                _wy  += list(_ys[:_nn])
                                _wv  += _vl_pad
                                _wac += _ac_pad
                                _wts += _ts_pad
                                _wper += [_pn] * _nn

                        if len(_wx) < max(_wcs2_n, 2):
                            continue

                        # Valor por amostra para a métrica selecionada
                        _Hz = _wcs2_hz
                        _m  = _wcs2_metric
                        if _m == "Distância (m)":
                            _sv = [v / (3.6 * _Hz) for v in _wv]
                        elif _m == "🏃 Velocidade (bandas)":
                            # Distância (m) acumulada apenas nas bandas marcadas.
                            if _wcs2_vel_rel:
                                # (P9) faixas relativas à Vmáx individual do atleta
                                _vmx_w = _vmax_individual_kmh(_wa, _wv)
                                _faixas_v = ([(lo * _vmx_w, hi * _vmx_w)
                                              for lo, hi in _sel_vel_pct]
                                             if _vmx_w > 0 else [])
                                if not _faixas_v:
                                    _diag_log('WCS', f"{_wa}: sem Vmáx individual "
                                                     "confiável — excluído do modo % Vmáx")
                            else:
                                _faixas_v = [(float(b.get('min', 0)), float(b.get('max', 9999)))
                                             for b in _sel_vel_bands]
                            # (P1) canônico: metrics.per_sample_distance_in_bands
                            _sv = (_mtr.per_sample_distance_in_bands(_wv, _faixas_v, _Hz)
                                   if _faixas_v else [0.0] * len(_wv))
                        elif _m == "💥 Ações Acel/Desacel (efforts)":
                            # Nº de AÇÕES (efforts da Catapult) de aceleração/desaceleração
                            # na janela: cada effort cuja aceleração cai nas bandas marcadas
                            # soma +1 na amostra mais próxima do seu start_time. A janela
                            # rolante soma as ações → identifica o pior minuto.
                            _faixas_a = [(float(b.get('min', -9999)), float(b.get('max', 9999)))
                                         for b in _sel_acc_bands]
                            def _in_aband(_aa, _ff=_faixas_a):
                                for _lo, _hi in _ff:
                                    if _lo <= _aa < _hi:
                                        return True
                                return False
                            _sv = [0.0] * len(_wx)
                            _wts_np = np.array(_wts, dtype=float)
                            # Timestamps Unix utilizáveis? (efforts usam start_time Unix)
                            _ts_unix_ok = (_wts_np.size > 0
                                           and float(np.median(_wts_np)) > 1e6)
                            # Existem efforts reais da API para este atleta?
                            _has_api_eff = any(
                                len(dados_efforts_acc_por_periodo
                                    .get(_pn, {}).get(_wa, []) or []) > 0
                                for _pn in _wcs2_periodos)
                            if _faixas_a and _ts_unix_ok and _has_api_eff:
                                # Caminho preferido: AÇÕES reais (efforts da Catapult).
                                _wcs2_prov['efforts'] = _wcs2_prov.get('efforts', 0) + 1  # (P4)
                                for _pn in _wcs2_periodos:
                                    _effs = (dados_efforts_acc_por_periodo
                                             .get(_pn, {}).get(_wa, []) or [])
                                    for _ef in _effs:
                                        try:
                                            _bx  = int(round(float(_ef.get('band'))))
                                            _stt = float(_ef.get('start_time') or 0)
                                        except (TypeError, ValueError):
                                            continue
                                        if _stt <= 0 or _bx not in _sel_acc_boxes:
                                            continue
                                        _idx = int(np.argmin(np.abs(_wts_np - _stt)))
                                        if 0 <= _idx < len(_sv):
                                            _sv[_idx] += 1.0
                            elif _faixas_a:
                                # Fallback (API sem efforts): detecta AÇÕES no SINAL DE
                                # ACELERAÇÃO do sensor (nativo 'a', 10 Hz) — mesma fonte da
                                # aba Neuromuscular/Janelas — e mapeia proporcionalmente para
                                # a timeline de posição do WCS. Nunca zera por falta de 'acc'.
                                _sp_w = (combinar_periodos_continuo(
                                            dados_sensor_por_atleta_por_periodo, _wa)
                                         if len(_wcs2_periodos) > 1 else
                                         dados_sensor_por_atleta_por_periodo
                                            .get(_wcs2_periodos[0], {}).get(_wa, [])) \
                                        if _wcs2_periodos else []
                                _acc_w = [float(_p.get('a') or 0.0) for _p in _sp_w]
                                if _acc_w and not any(abs(_a) > 0.05 for _a in _acc_w):
                                    _vw = [float(_p.get('v') or 0.0) * 3.6 for _p in _sp_w]
                                    _tw = [float(_p.get('ts') or 0.0) for _p in _sp_w]
                                    _acc_w = acc_series_from_vel(_vw, _tw, _SENSOR_HZ)
                                    _wcs2_prov['derivado'] = _wcs2_prov.get('derivado', 0) + 1  # (P4)
                                    _diag_log('WCS', f"{_wa}: sem aceleração nativa — "
                                                     "ações derivadas por dv/dt da velocidade")
                                elif _acc_w:
                                    _wcs2_prov['sensor'] = _wcs2_prov.get('sensor', 0) + 1  # (P4)
                                if _acc_w:
                                    _idxs_acc = detectar_acoes_acc_idx(
                                        _acc_w, _sel_acc_bands, freq_hz=_SENSOR_HZ)
                                    _Ls, _Lp = len(_acc_w), len(_sv)
                                    for _ix in _idxs_acc:
                                        _pi = int(_ix / _Ls * _Lp) if _Ls > 0 else 0
                                        if 0 <= _pi < _Lp:
                                            _sv[_pi] += 1.0
                                else:
                                    # Sem sensor (raro): mantém derivação posicional (dv/dt).
                                    _wac_fb = _wac
                                    if any(abs(_v) > 0.1 for _v in _wv):
                                        _wac_fb = acc_series_from_vel(_wv, _wts, _Hz)
                                    _idxs_acc = detectar_acoes_acc_idx(
                                        _wac_fb, _sel_acc_bands, freq_hz=_Hz)
                                    for _ix in _idxs_acc:
                                        if 0 <= _ix < len(_sv):
                                            _sv[_ix] += 1.0
                        elif _m == "Velocidade Máx (km/h)":
                            _sv = list(_wv)   # rolling max — tratado abaixo
                        elif _m == "PlayerLoad":
                            _pl_raw = []
                            for _ppn in _wcs2_periodos:
                                _pl_raw += dados_sensor_por_atleta_por_periodo.get(_ppn, {}).get(_wa, [])
                            if len(_pl_raw) >= len(_wx):
                                _sv = [float(p.get('pl') or 0) for p in _pl_raw[:len(_wx)]]
                            else:
                                _sv = [float(p.get('pl') or 0) for p in _pl_raw] + [0.0] * (len(_wx) - len(_pl_raw))
                        else:
                            _sv = [v / (3.6 * _Hz) for v in _wv]

                        # Multi-janela (1, 3, 5 min) para comparativo na tabela
                        _is_vm2 = (_m == "Velocidade Máx (km/h)")
                        _mw_vals = {}
                        for _mwname, _mwmin in [('1 min', 1), ('3 min', 3), ('5 min', 5)]:
                            _mwn = int(_mwmin * 60 * _wcs2_hz)
                            if _mwn != _wcs2_n and len(_sv) >= max(_mwn, 2):
                                _mw_vals[_mwname] = round(_wcalc_wcs(_sv, _mwn, _is_vm2), 1)

                        # Janela rolante
                        if _m == "Velocidade Máx (km/h)":
                            from collections import deque as _Dq
                            _dq3 = _Dq()
                            _bv3, _bsi3, _bei3 = -1.0, 0, _wcs2_n
                            for _i3 in range(len(_sv)):
                                while _dq3 and _sv[_dq3[-1]] <= _sv[_i3]:
                                    _dq3.pop()
                                _dq3.append(_i3)
                                if _dq3[0] <= _i3 - _wcs2_n:
                                    _dq3.popleft()
                                if _i3 >= _wcs2_n - 1:
                                    _c3 = _sv[_dq3[0]]
                                    if _c3 > _bv3:
                                        _bv3  = _c3
                                        _bei3 = _i3 + 1
                                        _bsi3 = _bei3 - _wcs2_n
                            _best_val2, _best_si2, _best_ei2 = _bv3, _bsi3, _bei3
                        else:
                            # (P1) canônico: metrics.rolling_sum + argmax
                            _rw2 = _mtr.rolling_sum(_sv, _wcs2_n)
                            if _rw2:
                                _bi2 = int(np.argmax(_rw2))
                                _best_val2 = _rw2[_bi2]
                                _best_si2  = _bi2
                                _best_ei2  = _bi2 + _wcs2_n
                            else:
                                _best_val2, _best_si2, _best_ei2 = 0.0, 0, _wcs2_n

                        # Timestamps
                        _ts0 = _wts[_best_si2] if _best_si2 < len(_wts) else 0
                        _ts1 = _wts[min(_best_ei2 - 1, len(_wts) - 1)] if _wts else 0
                        try:
                            from datetime import datetime as _dtc
                            _ini_str = _dtc.fromtimestamp(float(_ts0)).strftime('%H:%M:%S') if float(_ts0) > 1e6 else f"{int(_best_si2/_Hz//60):02d}:{int(_best_si2/_Hz%60):02d}"
                            _fim_str = _dtc.fromtimestamp(float(_ts1)).strftime('%H:%M:%S') if float(_ts1) > 1e6 else f"{int(_best_ei2/_Hz//60):02d}:{int(_best_ei2/_Hz%60):02d}"
                        except Exception:
                            _ini_str = f"{int(_best_si2/_Hz//60):02d}:{int(_best_si2/_Hz%60):02d}"
                            _fim_str = f"{int(_best_ei2/_Hz//60):02d}:{int(_best_ei2/_Hz%60):02d}"

                        # Posição do atleta
                        _posicao2 = '—'
                        for _rpn2, _rpl2 in resultados_por_periodo.items():
                            for _rrow2 in _rpl2:
                                if str(_rrow2.get('Atleta', '')) == _wa:
                                    _posicao2 = str(_rrow2.get('Posição', '—'))
                                    break
                            if _posicao2 != '—':
                                break

                        # ── Série rolling completa (para timeline) ─────────────
                        _is_vm3 = (_m == "Velocidade Máx (km/h)")
                        if _is_vm3:
                            from collections import deque as _DqTL
                            _dqTL = _DqTL(); _roll_full = []
                            for _iRL in range(len(_sv)):
                                while _dqTL and _sv[_dqTL[-1]] <= _sv[_iRL]:
                                    _dqTL.pop()
                                _dqTL.append(_iRL)
                                if _dqTL[0] <= _iRL - _wcs2_n:
                                    _dqTL.popleft()
                                if _iRL >= _wcs2_n - 1:
                                    _roll_full.append(_sv[_dqTL[0]])
                        else:
                            # (P1) canônico: metrics.rolling_sum
                            _roll_full = _mtr.rolling_sum(_sv, _wcs2_n)

                        # ── Densidade de Pico (janelas ≥ 90% do WCS) ───────────
                        _density_90 = (
                            sum(1 for _rv in _roll_full if _rv >= 0.9 * _best_val2)
                            if _best_val2 > 0 and _roll_full else 0
                        )

                        _row_d = {
                            '_atl_orig':   _wa,
                            'Atleta':      _wa,
                            'Posição':     _posicao2,
                            'Período':     _wper[_best_si2] if _best_si2 < len(_wper) else '—',
                            _wcs2_metric:  round(_best_val2, 1),
                            'Picos ≥90%':  _density_90,
                            'Início':      _ini_str,
                            'Fim':         _fim_str,
                        }
                        _row_d.update(_mw_vals)
                        _wcs2_rows.append(_row_d)
                        # Série de aceleração (m/s²) p/ colorir a trilha por banda
                        # de acel/desacel quando a métrica de AÇÕES está ativa.
                        _acc_full_anim = []
                        if _m == "💥 Ações Acel/Desacel (efforts)":
                            if any(abs(_v) > 0.1 for _v in _wv):
                                _acc_full_anim = acc_series_from_vel(_wv, _wts, _Hz)
                            else:
                                _acc_full_anim = list(_wac)
                        _wcs2_segs[_wa] = {
                            'xn':     _wx[_best_si2:_best_ei2],
                            'yn':     _wy[_best_si2:_best_ei2],
                            'vel':    _wv[_best_si2:_best_ei2],
                            'acc':    (_acc_full_anim[_best_si2:_best_ei2]
                                       if _acc_full_anim else []),
                            'rolling': _roll_full,
                            'vel_all': _wv,        # velocidade de toda a série (trilha colorida)
                            'acc_all': _acc_full_anim,
                            'xn_all':  _wx,
                            'yn_all':  _wy,
                        }

                    _wcs2_rows.sort(key=lambda r: r.get(_wcs2_metric, 0), reverse=True)
                    _rank_icons = ['🔴', '🟠', '🟡']   # vermelho = maior carga/fadiga
                    for _ri2, _wr2 in enumerate(_wcs2_rows):
                        _wr2['#'] = _rank_icons[_ri2] if _ri2 < 3 else f'#{_ri2 + 1}'

                    if not _wcs2_rows:
                        st.warning(
                            "Dados insuficientes para calcular WCS com essa janela temporal. "
                            "Reduza a janela ou carregue mais períodos."
                        )
                        # Diagnóstico para debug
                        with st.expander("🛠️ Diagnóstico de dados", expanded=True):
                            _diag = []
                            for _wa_d in _wcs2_athletes[:8]:
                                _tot_xs, _tot_vel = 0, 0
                                for _pn_d in _wcs2_periodos:
                                    _da_d = dados_posicao_por_periodo.get(_pn_d, {}).get(_wa_d, {})
                                    _tot_xs  += len(_da_d.get('xs', []))
                                    _tot_vel += len(_da_d.get('vel', []))
                                _diag.append({'Atleta': _wa_d,
                                              'Amostras XY': _tot_xs,
                                              'Amostras vel': _tot_vel,
                                              'Necessário': _wcs2_n})
                            if _diag:
                                st.dataframe(pd.DataFrame(_diag), hide_index=True,
                                             use_container_width=True)
                            st.caption(f"Hz detectado: {_wcs2_hz} | Janela: {_wcs2_min} min = {_wcs2_n} amostras | Períodos: {len(_wcs2_periodos)}")
                    else:
                        # % do Máximo do grupo
                        _wcs2_top = _wcs2_rows[0].get(_wcs2_metric, 0) or 1.0
                        _wcs2_avg = sum(r.get(_wcs2_metric, 0) for r in _wcs2_rows) / len(_wcs2_rows)
                        for _wr in _wcs2_rows:
                            _wr['% Máx Grupo'] = round(_wr.get(_wcs2_metric, 0) / _wcs2_top * 100, 1)

                        # KPIs resumo
                        _wk1, _wk2, _wk3, _wk4 = st.columns(4)
                        _wk1.metric(
                            "🔴 Maior Fadiga",
                            _wcs2_rows[0]['Atleta'],
                            f"{_wcs2_top:.1f} — {_wcs2_rows[0].get('Período', '—')}",
                        )
                        _wk2.metric(
                            "📊 Média Grupo",
                            f"{_wcs2_avg:.1f}",
                            f"Δ {_wcs2_top - _wcs2_avg:.1f} vs líder",
                        )
                        _wk3.metric("👥 Atletas", str(len(_wcs2_rows)))
                        _wk4.metric(
                            "⏱️ Janela",
                            f"{_wcs2_min} min",
                            f"{_wcs2_rows[0].get('Início','—')} → {_wcs2_rows[0].get('Fim','—')}",
                        )

                        st.markdown("---")

                        # Tabela
                        # (P4) selo de proveniência das ações (pode variar por atleta)
                        if (_wcs2_metric == "💥 Ações Acel/Desacel (efforts)"
                                and _wcs2_prov):
                            _prov_txt = " · ".join(
                                f"{_PROV_LABELS.get(_k, ('⚪', _k))[0]} "
                                f"{_PROV_LABELS.get(_k, ('', str(_k)))[1]}: "
                                f"**{_v} atleta(s)**"
                                for _k, _v in sorted(_wcs2_prov.items()))
                            st.caption(f"**Fonte das ações** — {_prov_txt}")

                        # (P9) aviso do modo relativo (cortes diferentes por atleta)
                        if _wcs2_metric == "🏃 Velocidade (bandas)" and _wcs2_vel_rel:
                            st.caption("🎚️ **Modo % da Vmáx individual** — os cortes das "
                                       "faixas são calculados por atleta (referência: Vmáx "
                                       "histórica da conta, com fallback no pico da sessão).")

                        _df_all_tmp = pd.DataFrame(_wcs2_rows)
                        _mw_avail   = [c for c in ['1 min', '3 min', '5 min']
                                       if c in _df_all_tmp.columns
                                       and _df_all_tmp[c].notna().any()]
                        _wcs2_col_order = (
                            ['#', 'Atleta', 'Posição', 'Período',
                             _wcs2_metric, '% Máx Grupo', 'Picos ≥90%']
                            + _mw_avail
                            + ['Início', 'Fim']
                        )
                        _wcs2_col_order = [c for c in _wcs2_col_order if c in _df_all_tmp.columns]
                        _df_wcs2 = _df_all_tmp[_wcs2_col_order]

                        _col_cfg_wcs = {
                            '% Máx Grupo': st.column_config.ProgressColumn(
                                '% Máx', min_value=0, max_value=100, format='%.1f%%'
                            ),
                        }
                        _wcs2_evt = st.dataframe(
                            _df_wcs2,
                            use_container_width=True,
                            hide_index=True,
                            on_select='rerun',
                            selection_mode='single-row',
                            key='wcs2_table_sel',
                            column_config=_col_cfg_wcs,
                        )
                        st.caption(
                            "💡 Clique em uma linha para visualizar o percurso animado no campo abaixo. "
                            "As colunas 1/3/5 min mostram o valor WCS para janelas fixas de comparação. "
                            "**Picos ≥90%** = nº de janelas onde o atleta atingiu ≥90% do seu WCS."
                        )

                        # ── Timeline do Rolling Window — removido (P8) ──

                        # ── WCS por Período ─────────────────────────────────────────
                        with st.expander("📊 WCS por Período", expanded=False):
                            st.caption(
                                "WCS de cada atleta calculado **separadamente por período**. "
                                "🔴 Vermelho = maior demanda. Detecta queda de desempenho por fadiga entre tempos."
                            )
                            _ppw_data = {}
                            for _pn_pp in _wcs2_periodos:
                                _ppw_data[_pn_pp] = {}
                                for _wa_pp in _wcs2_athletes:
                                    _da_pp  = dados_posicao_por_periodo.get(_pn_pp, {}).get(_wa_pp, {})
                                    _xs_pp  = list(_da_pp.get('xs', []))
                                    _ys_pp  = list(_da_pp.get('ys', []))
                                    _vl_pp  = list(_da_pp.get('vel', []))
                                    _ac_pp  = list(_da_pp.get('acc', []))
                                    _nn_pp  = min(len(_xs_pp), len(_ys_pp))
                                    if _nn_pp == 0 and _wcs2_cfg:
                                        _lp_pp = _da_pp.get('lats', [])
                                        _lo_pp = _da_pp.get('lons', [])
                                        if _lp_pp and _lo_pp:
                                            try:
                                                _gx_pp, _gy_pp = gps_para_campo_coords(
                                                    _lp_pp, _lo_pp, _wcs2_cfg
                                                )
                                                _xs_pp = _gx_pp; _ys_pp = _gy_pp
                                                _vl_pp = _da_pp.get('vels_gps', [0.0]*len(_gx_pp))
                                                _nn_pp = min(len(_xs_pp), len(_ys_pp))
                                            except Exception:
                                                _nn_pp = 0
                                    if _nn_pp < _wcs2_n:
                                        _ppw_data[_pn_pp][_wa_pp] = None
                                        continue
                                    _vl_pp_p = list(_vl_pp[:_nn_pp]) + [0.0]*max(0,_nn_pp-len(_vl_pp))
                                    _ac_pp_p = list(_ac_pp[:_nn_pp]) + [0.0]*max(0,_nn_pp-len(_ac_pp))
                                    _m_pp = _wcs2_metric
                                    if _m_pp == "Distância (m)":
                                        _sv_pp = [v/(3.6*_wcs2_hz) for v in _vl_pp_p]
                                    elif ">14" in _m_pp:
                                        _sv_pp = [v/(3.6*_wcs2_hz) if v>14 else 0.0 for v in _vl_pp_p]
                                    elif "19" in _m_pp:
                                        _sv_pp = [v/(3.6*_wcs2_hz) if v>19 else 0.0 for v in _vl_pp_p]
                                    elif "24" in _m_pp:
                                        _sv_pp = [v/(3.6*_wcs2_hz) if v>24 else 0.0 for v in _vl_pp_p]
                                    elif "Velocidade Máx" in _m_pp:
                                        _sv_pp = _vl_pp_p
                                    elif "PlayerLoad" in _m_pp:
                                        _pl_pp = dados_sensor_por_atleta_por_periodo.get(_pn_pp,{}).get(_wa_pp,[])
                                        _sv_pp = ([float(p.get('pl') or 0) for p in _pl_pp[:_nn_pp]]
                                                  + [0.0]*max(0,_nn_pp-len(_pl_pp)))
                                    elif ">2" in _m_pp and "Acel" in _m_pp:
                                        _sv_pp = [1.0 if a>2 else 0.0 for a in _ac_pp_p]
                                    elif ">3" in _m_pp and "Acel" in _m_pp:
                                        _sv_pp = [1.0 if a>3 else 0.0 for a in _ac_pp_p]
                                    elif "<-2" in _m_pp:
                                        _sv_pp = [1.0 if a<-2 else 0.0 for a in _ac_pp_p]
                                    elif "<-3" in _m_pp:
                                        _sv_pp = [1.0 if a<-3 else 0.0 for a in _ac_pp_p]
                                    else:
                                        _sv_pp = [v/(3.6*_wcs2_hz) for v in _vl_pp_p]
                                    _is_vm_pp = ("Velocidade Máx" in _m_pp)
                                    _wcs_pp   = _wcalc_wcs(_sv_pp, _wcs2_n, _is_vm_pp)
                                    _ppw_data[_pn_pp][_wa_pp] = round(_wcs_pp, 1) if _wcs_pp > 0 else None

                            _df_ppw = pd.DataFrame(_ppw_data).T
                            _df_ppw.index.name = 'Período'
                            if not _df_ppw.empty and _df_ppw.notna().any().any():
                                _z_ppw      = _df_ppw.values.tolist()
                                _aths_ppw   = _df_ppw.columns.tolist()
                                _pers_ppw   = _df_ppw.index.tolist()

                                # Limites globais para escala
                                _all_ppw  = [v for row in _z_ppw for v in row if v is not None]
                                _zmin_ppw = min(_all_ppw) if _all_ppw else 0
                                _zmax_ppw = max(_all_ppw) if _all_ppw else 1

                                # ── Heatmap principal (sem texttemplate) ──────────────
                                _fig_ppw = go.Figure(data=go.Heatmap(
                                    z=_z_ppw,
                                    x=_aths_ppw,
                                    y=_pers_ppw,
                                    colorscale='RdYlGn_r',
                                    zmin=_zmin_ppw,
                                    zmax=_zmax_ppw,
                                    hovertemplate='%{y} — %{x}<br>WCS: %{z:.1f}<extra></extra>',
                                    showscale=True,
                                    colorbar=dict(
                                        title=dict(text=_wcs2_metric, font=dict(color='white')),
                                        tickfont=dict(color='white'),
                                    ),
                                ))

                                # Anotações por célula com cor de texto adaptativa:
                                # RdYlGn_r: baixo→verde(escuro), meio→amarelo(claro), alto→vermelho(escuro)
                                # Zona amarela (norm 0.25–0.75) → texto preto; demais → branco
                                _rng_ppw = _zmax_ppw - _zmin_ppw if _zmax_ppw > _zmin_ppw else 1
                                for _ri_a, _py_a in enumerate(_pers_ppw):
                                    for _ci_a, _px_a in enumerate(_aths_ppw):
                                        _v_a = _z_ppw[_ri_a][_ci_a]
                                        if _v_a is None:
                                            _lbl_a = '—'
                                            _tc_a  = '#888888'
                                        else:
                                            _lbl_a = f'{_v_a:.1f}'
                                            _norm_a = (_v_a - _zmin_ppw) / _rng_ppw
                                            _tc_a   = ('#111111'
                                                        if 0.25 < _norm_a < 0.75
                                                        else 'white')
                                        _fig_ppw.add_annotation(
                                            x=_px_a, y=_py_a,
                                            text=_lbl_a,
                                            showarrow=False,
                                            font=dict(size=11, color=_tc_a,
                                                      family='monospace'),
                                            xanchor='center', yanchor='middle',
                                        )

                                _fig_ppw.update_layout(
                                    title=dict(
                                        text=f"WCS por Período — {_wcs2_metric} ({_wcs2_min} min)",
                                        font=dict(color='white', size=13)
                                    ),
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    height=max(260, len(_pers_ppw) * 70 + 120),
                                    margin=dict(t=50, b=100, l=10, r=10),
                                    xaxis=dict(tickangle=-35,
                                               tickfont=dict(size=9, color='white'),
                                               color='white'),
                                    yaxis=dict(color='white'),
                                )
                                st.plotly_chart(_fig_ppw, use_container_width=True)

                                # ── Heatmap de variação % entre períodos consecutivos ─
                                if len(_pers_ppw) >= 2:
                                    _diff_z    = []
                                    _diff_lbls = []
                                    _diff_ys   = []
                                    for _pi_d in range(1, len(_pers_ppw)):
                                        _p1d = _pers_ppw[_pi_d - 1]
                                        _p2d = _pers_ppw[_pi_d]
                                        _diff_ys.append(f'Δ% {_p1d}→{_p2d}')
                                        _row_dz = []; _row_dl = []
                                        for _ath_d in _aths_ppw:
                                            _v1d = _ppw_data.get(_p1d, {}).get(_ath_d)
                                            _v2d = _ppw_data.get(_p2d, {}).get(_ath_d)
                                            if (_v1d is not None and _v2d is not None
                                                    and _v1d > 0):
                                                _dv = round((_v2d - _v1d) / _v1d * 100, 1)
                                                _row_dz.append(_dv)
                                                _row_dl.append(
                                                    f'+{_dv:.1f}%' if _dv >= 0
                                                    else f'{_dv:.1f}%'
                                                )
                                            else:
                                                _row_dz.append(None)
                                                _row_dl.append('—')
                                        _diff_z.append(_row_dz)
                                        _diff_lbls.append(_row_dl)

                                    _all_dv = [v for row in _diff_z
                                               for v in row if v is not None]
                                    if _all_dv:
                                        _absmax_d = max(abs(v) for v in _all_dv) or 10
                                        _fig_diff = go.Figure(data=go.Heatmap(
                                            z=_diff_z,
                                            x=_aths_ppw,
                                            y=_diff_ys,
                                            colorscale='RdYlGn_r',
                                            zmid=0,
                                            zmin=-_absmax_d,
                                            zmax=_absmax_d,
                                            hovertemplate='%{y}<br>%{x}: %{z:+.1f}%<extra></extra>',
                                            showscale=True,
                                            colorbar=dict(
                                                title=dict(text='Δ%',
                                                           font=dict(color='white')),
                                                tickfont=dict(color='white'),
                                                ticksuffix='%',
                                            ),
                                        ))
                                        # Anotações adaptativas para diff
                                        for _ri_d2, _yd2 in enumerate(_diff_ys):
                                            for _ci_d2, _xd2 in enumerate(_aths_ppw):
                                                _vd2  = _diff_z[_ri_d2][_ci_d2]
                                                _ld2  = _diff_lbls[_ri_d2][_ci_d2]
                                                if _vd2 is None:
                                                    _tcd2 = '#888888'
                                                else:
                                                    # norm em [-absmax, +absmax] → [0, 1]
                                                    _nd2  = (_vd2 + _absmax_d) / (2 * _absmax_d)
                                                    _tcd2 = ('#111111'
                                                              if 0.25 < _nd2 < 0.75
                                                              else 'white')
                                                _fig_diff.add_annotation(
                                                    x=_xd2, y=_yd2,
                                                    text=_ld2,
                                                    showarrow=False,
                                                    font=dict(size=11, color=_tcd2,
                                                              family='monospace'),
                                                    xanchor='center', yanchor='middle',
                                                )
                                        # altura: cada linha precisa de ~80px + 160px de margens
                                        _h_diff = max(260, len(_diff_ys) * 80 + 160)
                                        _fig_diff.update_layout(
                                            title=dict(
                                                text='Variação % entre Períodos',
                                                font=dict(color='white', size=13)
                                            ),
                                            plot_bgcolor='#0e1117',
                                            paper_bgcolor='#0e1117',
                                            font=dict(color='white'),
                                            height=_h_diff,
                                            margin=dict(t=45, b=110, l=10, r=10),
                                            xaxis=dict(
                                                tickangle=-35,
                                                tickfont=dict(size=9, color='white'),
                                                color='white',
                                            ),
                                            yaxis=dict(
                                                color='white',
                                                tickfont=dict(size=10),
                                            ),
                                        )
                                        st.plotly_chart(_fig_diff, use_container_width=True)
                                        st.caption(
                                            "🟢 Verde = queda de demanda no período seguinte  "
                                            "🔴 Vermelho = aumento de demanda (atenção à fadiga)"
                                        )
                            else:
                                st.info("Dados insuficientes para comparar períodos com esta janela.")

                        st.markdown("---")

                        # ── Animação no campo ao selecionar linha ──────────────────
                        _wcs2_sel = (_wcs2_evt.selection.rows
                                     if _wcs2_evt.selection else [])
                        if _wcs2_sel:
                            _wsel_atl = _wcs2_rows[_wcs2_sel[0]].get(
                                '_atl_orig', _wcs2_rows[_wcs2_sel[0]]['Atleta']
                            )
                            _wsel_row = _wcs2_rows[_wcs2_sel[0]]
                            _wseg     = _wcs2_segs.get(_wsel_atl, {})
                            _wcs2_xn  = _wseg.get('xn', [])
                            _wcs2_yn  = _wseg.get('yn', [])
                            _wcs2_vel = _wseg.get('vel', [])
                            _wcs2_acc = _wseg.get('acc', [])
                            # Quando a métrica é AÇÕES, a trilha/legenda usam bandas de
                            # acel/desacel (m/s²) em vez de velocidade.
                            _is_acoes_anim = (
                                _wcs2_metric == "💥 Ações Acel/Desacel (efforts)"
                                and len(_wcs2_acc) == len(_wcs2_xn)
                            )

                            if len(_wcs2_xn) >= 2:
                                st.markdown(f"### 🏃 {_wsel_atl} — WCS {_wcs2_min} min")
                                _wval_str = f"{_wsel_row.get(_wcs2_metric, 0):.1f}"
                                _wm1, _wm2, _wm3 = st.columns(3)
                                _wm1.metric(_wcs2_metric, _wval_str)
                                _wm2.metric("⏰ Início",  _wsel_row.get('Início', '—'))
                                _wm3.metric("🏁 Fim",     _wsel_row.get('Fim',    '—'))

                                # Campo config
                                _wcs2_cfg = None
                                for _hk2 in list(st.session_state.keys()):
                                    if (_hk2.startswith("campo_cfg__")
                                            and isinstance(st.session_state[_hk2], dict)
                                            and 'fl' in st.session_state[_hk2]):
                                        _wcs2_cfg = st.session_state[_hk2]
                                        break
                                _wcs2_fl = float(_wcs2_cfg.get('fl', 105)) if _wcs2_cfg else 105.0
                                _wcs2_fw = float(_wcs2_cfg.get('fw', 68))  if _wcs2_cfg else 68.0

                                def _vc_wcs2(v):
                                    if v < 7:  return '#2196F3'
                                    if v < 14: return '#4CAF50'
                                    if v < 19: return '#FFEB3B'
                                    if v < 24: return '#FF9800'
                                    return '#F44336'

                                # ── Cor por banda de aceleração/desaceleração ───────
                                _acc_bands_anim = list(_bandas_acc_ativas().values())
                                _NEUTRO_COR_AC = '#546E7A'   # zona leve/neutra (−2..2)
                                def _ac_color(a):
                                    for _b in _acc_bands_anim:
                                        try:
                                            if float(_b['min']) <= a < float(_b['max']):
                                                return _b.get('color', _NEUTRO_COR_AC)
                                        except (TypeError, ValueError, KeyError):
                                            continue
                                    # satura no topo da banda extrema de aceleração
                                    try:
                                        _tops = [float(_b['max']) for _b in _acc_bands_anim
                                                 if float(_b['min']) >= 0]
                                        if _tops and a >= max(_tops):
                                            return next(_b.get('color', _NEUTRO_COR_AC)
                                                        for _b in _acc_bands_anim
                                                        if float(_b['max']) == max(_tops))
                                    except (TypeError, ValueError, KeyError, StopIteration):
                                        pass
                                    return _NEUTRO_COR_AC

                                _fig_wcs2 = desenhar_campo_futebol_bonito(
                                    field_length=_wcs2_fl,
                                    field_width=_wcs2_fw,
                                    title=(
                                        f"WCS {_wcs2_min} min — {_wsel_atl}  |  "
                                        f"{_wsel_row.get('Início','—')} → {_wsel_row.get('Fim','—')}"
                                    )
                                )
                                _n_base_wcs2 = len(_fig_wcs2.data)

                                # Downsampling
                                _nf2   = len(_wcs2_xn)
                                _step2 = max(1, _nf2 // 120)
                                _fr2   = list(range(0, _nf2, _step2))
                                if _fr2[-1] != _nf2 - 1:
                                    _fr2.append(_nf2 - 1)

                                # ── Trilha completa colorida (estática) ──
                                # Métrica de AÇÕES → cor por banda de acel/desacel;
                                # caso contrário → faixa de velocidade.
                                if _is_acoes_anim:
                                    _trail_colors = [_ac_color(a) for a in _wcs2_acc]
                                else:
                                    _trail_colors = [_vc_wcs2(v) for v in _wcs2_vel]
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=_wcs2_xn, y=_wcs2_yn,
                                    mode='markers',
                                    marker=dict(
                                        color=_trail_colors,
                                        size=7, opacity=0.55,
                                    ),
                                    name='Trilha', showlegend=False, hoverinfo='skip',
                                ))
                                # ── Legenda inline ──
                                if _is_acoes_anim:
                                    # Bandas de acel/desacel (rótulo curto B1/B2/B3)
                                    import re as _re_anim
                                    for _bk, _bd in _bandas_acc_ativas().items():
                                        _lbl_full = _bd.get('label', _bk)
                                        # encurta: "Aceleração B1 — 2 a 3 m/s²"
                                        _emoji = '🚀' if str(_bk).startswith('A') else '🛑'
                                        _fig_wcs2.add_trace(go.Scatter(
                                            x=[None], y=[None], mode='markers',
                                            marker=dict(size=10,
                                                        color=_bd.get('color', '#888')),
                                            name=f"{_emoji} {_lbl_full}",
                                        ))
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color=_NEUTRO_COR_AC),
                                        name='• Neutro (−2 a 2 m/s²)',
                                    ))
                                else:
                                    # Legenda de velocidade inline
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color='#2196F3'), name='< 7 km/h',
                                    ))
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color='#4CAF50'), name='7–14 km/h',
                                    ))
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color='#FFEB3B'), name='14–19 km/h',
                                    ))
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color='#FF9800'), name='19–24 km/h',
                                    ))
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color='#F44336'), name='> 24 km/h',
                                    ))
                                # Marcadores início / fim
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=[_wcs2_xn[0]], y=[_wcs2_yn[0]],
                                    mode='markers+text',
                                    marker=dict(size=14, color='#4CAF50', symbol='circle',
                                                line=dict(color='white', width=2)),
                                    text=['▶'], textposition='top center',
                                    textfont=dict(color='#4CAF50', size=11),
                                    name='Início', showlegend=False, hoverinfo='skip',
                                ))
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=[_wcs2_xn[-1]], y=[_wcs2_yn[-1]],
                                    mode='markers+text',
                                    marker=dict(size=14, color='#F44336', symbol='x',
                                                line=dict(color='white', width=2)),
                                    text=['■'], textposition='top center',
                                    textfont=dict(color='#F44336', size=11),
                                    name='Fim', showlegend=False, hoverinfo='skip',
                                ))
                                # Dot animado (único trace que muda nos frames)
                                _dot_c0 = (_ac_color(_wcs2_acc[0] if _wcs2_acc else 0)
                                           if _is_acoes_anim
                                           else _vc_wcs2(_wcs2_vel[0] if _wcs2_vel else 0))
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=[_wcs2_xn[0]], y=[_wcs2_yn[0]], mode='markers',
                                    marker=dict(
                                        size=20,
                                        color=_dot_c0,
                                        symbol='circle',
                                        line=dict(color='white', width=3),
                                    ),
                                    name='Posição atual', showlegend=False,
                                ))

                                # Índice: dot é o último trace adicionado
                                _idx_d2 = len(_fig_wcs2.data) - 1

                                # Frames — só atualiza o dot
                                _frames2 = []
                                for _fk2 in _fr2:
                                    _ds3  = (_fk2 / max(_nf2 - 1, 1)) * _wcs2_min * 60
                                    _dm3  = int(_ds3 // 60)
                                    _dsr3 = int(_ds3 % 60)
                                    _v3   = float(_wcs2_vel[_fk2]) if _fk2 < len(_wcs2_vel) else 0.0
                                    if _is_acoes_anim:
                                        _a3  = float(_wcs2_acc[_fk2]) if _fk2 < len(_wcs2_acc) else 0.0
                                        _c3  = _ac_color(_a3)
                                        _hud = f'   |   ⚡ {_a3:+.1f} m/s²'
                                    else:
                                        _c3  = _vc_wcs2(_v3)
                                        _hud = f'   |   💨 {_v3:.1f} km/h'
                                    _frames2.append(go.Frame(
                                        data=[go.Scatter(
                                            x=[_wcs2_xn[_fk2]],
                                            y=[_wcs2_yn[_fk2]],
                                            mode='markers',
                                            marker=dict(
                                                size=20, color=_c3,
                                                symbol='circle',
                                                line=dict(color='white', width=3),
                                            ),
                                        )],
                                        traces=[_idx_d2],
                                        name=str(_fk2),
                                        layout=go.Layout(title=dict(
                                            text=(
                                                f'WCS {_wcs2_min} min — {_wsel_atl} | '
                                                f'⏱️ {_dm3}:{_dsr3:02d} / {_wcs2_min}:00'
                                                f'{_hud}'
                                            ),
                                            font=dict(color='white', size=12),
                                        )),
                                    ))

                                _fig_wcs2.frames = _frames2

                                _sliders_wcs2 = [{
                                    'steps': [
                                        {
                                            'args': [[str(_fk2)],
                                                     {'frame': {'duration': 0, 'redraw': True},
                                                      'mode': 'immediate'}],
                                            'label': '',
                                            'method': 'animate',
                                        }
                                        for _fk2 in _fr2
                                    ],
                                    'transition': {'duration': 0},
                                    'x': 0.05, 'len': 0.90,
                                    'currentvalue': {'visible': False},
                                    'bgcolor': '#1e3a5f',
                                    'bordercolor': '#2196F3',
                                    'tickcolor': 'white',
                                    'font': {'color': 'white', 'size': 9},
                                }]
                                _fig_wcs2.update_layout(
                                    height=560,
                                    updatemenus=[{
                                        'type': 'buttons',
                                        'showactive': False,
                                        'y': -0.08, 'x': 0.5,
                                        'xanchor': 'center', 'yanchor': 'top',
                                        'buttons': [
                                            {
                                                'label': '▶ Play',
                                                'method': 'animate',
                                                'args': [None, {
                                                    'frame': {'duration': 60, 'redraw': True},
                                                    'fromcurrent': True,
                                                    'transition': {'duration': 60, 'easing': 'linear'},
                                                }],
                                            },
                                            {
                                                'label': '⏸ Pause',
                                                'method': 'animate',
                                                'args': [[None], {
                                                    'frame': {'duration': 0, 'redraw': False},
                                                    'mode': 'immediate',
                                                    'transition': {'duration': 0},
                                                }],
                                            },
                                        ],
                                        'font': {'color': 'white'},
                                        'bgcolor': '#1e3a5f',
                                        'bordercolor': '#2196F3',
                                    }],
                                    sliders=_sliders_wcs2,
                                    margin=dict(b=80),
                                )
                                st.plotly_chart(_fig_wcs2, use_container_width=True)
                            else:
                                st.info("Dados GPS insuficientes para animação deste atleta.")
