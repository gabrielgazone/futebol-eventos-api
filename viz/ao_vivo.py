# -*- coding: utf-8 -*-
"""Aba extraída de main() (P4 — page-decomposition)."""
from __future__ import annotations

import applog as _applog
import pandas as pd
import streamlit as st


def render_ao_vivo():
                st.subheader("📡 Monitoramento em Tempo Real")
                st.caption(
                    "Conecta ao endpoint `/live` da API Catapult para acompanhar métricas "
                    "de atletas durante uma sessão ativa. Requer uma sessão aberta no OpenField."
                )

                # ── Inicializar session state do live ─────────────────────
                if 'live_active' not in st.session_state:
                    st.session_state['live_active'] = False
                if 'live_alert_log' not in st.session_state:
                    st.session_state['live_alert_log'] = []
                if 'live_snapshot' not in st.session_state:
                    st.session_state['live_snapshot'] = None
                if 'live_info_snapshot' not in st.session_state:
                    st.session_state['live_info_snapshot'] = None

                # ── Layout de controles ────────────────────────────────────
                _lv_c1, _lv_c2, _lv_c3 = st.columns([2, 2, 1])
                with _lv_c1:
                    _lv_interval = st.select_slider(
                        "⏱️ Intervalo de atualização:",
                        options=[5, 10, 15, 30, 60],
                        value=10,
                        format_func=lambda x: f"{x}s",
                        key="live_interval",
                    )
                with _lv_c2:
                    _lv_cols = st.multiselect(
                        "📊 Métricas a exibir:",
                        options=["velocity", "heart_rate", "player_load",
                                 "acceleration", "total_distance", "odometer"],
                        default=["velocity", "heart_rate", "player_load"],
                        key="live_metrics_sel",
                    )
                with _lv_c3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    _lv_refresh_now = st.button(
                        "🔄 Agora", key="live_refresh_now",
                        help="Atualizar imediatamente uma vez"
                    )

                # ── Configuração de limiares (alertas) ────────────────────
                with st.expander("🚨 Configurar Limiares de Alerta", expanded=True):
                    st.caption(
                        "O app dispara um alerta visual quando o atleta **atingir ou ultrapassar** "
                        "o valor configurado. Deixe 0 para desativar."
                    )
                    _lv_th_c1, _lv_th_c2, _lv_th_c3 = st.columns(3)
                    with _lv_th_c1:
                        _lv_th_vel = st.number_input(
                            "🚀 Vel. máx (km/h)", min_value=0.0,
                            max_value=40.0, value=25.0, step=0.5,
                            key="live_th_vel",
                        )
                        _lv_th_dist = st.number_input(
                            "📏 Distância total (m)", min_value=0.0,
                            max_value=15000.0, value=0.0, step=100.0,
                            key="live_th_dist",
                            help="Alerta quando atleta passar desta distância acumulada. 0 = desativado."
                        )
                    with _lv_th_c2:
                        _lv_th_hr = st.number_input(
                            "❤️ FC máx (bpm)", min_value=0.0,
                            max_value=220.0, value=180.0, step=1.0,
                            key="live_th_hr",
                        )
                        _lv_th_pl = st.number_input(
                            "⚡ PlayerLoad", min_value=0.0,
                            max_value=1500.0, value=0.0, step=10.0,
                            key="live_th_pl",
                            help="Alerta quando PlayerLoad acumulado ultrapassar este valor. 0 = desativado."
                        )
                    with _lv_th_c3:
                        _lv_th_acc = st.number_input(
                            "💥 Aceleração (m/s²)", min_value=0.0,
                            max_value=10.0, value=3.5, step=0.1,
                            key="live_th_acc",
                        )
                        _lv_sound = st.checkbox(
                            "🔔 Exibir badge de alerta",
                            value=True, key="live_sound",
                        )

                    _lv_thresholds = {
                        "velocity":       (_lv_th_vel,  "km/h", "🚀"),
                        "heart_rate":     (_lv_th_hr,   "bpm",  "❤️"),
                        "acceleration":   (_lv_th_acc,  "m/s²", "💥"),
                        "total_distance": (_lv_th_dist, "m",    "📏"),
                        "player_load":    (_lv_th_pl,   "UA",   "⚡"),
                    }

                # ── Botões de controle ─────────────────────────────────────
                _lv_btn_c1, _lv_btn_c2, _lv_btn_c3 = st.columns([2, 2, 3])
                with _lv_btn_c1:
                    if not st.session_state['live_active']:
                        if st.button("▶ Iniciar Monitoramento", type="primary",
                                     key="live_start"):
                            st.session_state['live_active'] = True
                            st.session_state['live_alert_log'] = []
                            st.rerun()
                    else:
                        if st.button("⏹ Parar Monitoramento", type="secondary",
                                     key="live_stop"):
                            st.session_state['live_active'] = False
                            st.rerun()
                with _lv_btn_c2:
                    if st.button("🗑️ Limpar Alertas", key="live_clear_alerts"):
                        st.session_state['live_alert_log'] = []
                        st.rerun()
                with _lv_btn_c3:
                    _lv_status_ph = st.empty()

                # ── Indicador de status ────────────────────────────────────
                if st.session_state['live_active']:
                    _lv_status_ph.markdown(
                        "<span style='display:inline-flex;align-items:center;gap:6px;"
                        "background:#064e3b;border:1px solid #10b981;border-radius:20px;"
                        "padding:4px 12px;font-size:13px;color:#34d399'>"
                        "<span style='width:8px;height:8px;border-radius:50%;"
                        "background:#10b981;animation:pulse 1s infinite'></span>"
                        " AO VIVO</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    _lv_status_ph.markdown(
                        "<span style='display:inline-flex;align-items:center;gap:6px;"
                        "background:#1f2937;border:1px solid #4b5563;border-radius:20px;"
                        "padding:4px 12px;font-size:13px;color:#9ca3af'>"
                        "⏸ PAUSADO</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # ── Containers de exibição ─────────────────────────────────
                _lv_session_ph  = st.empty()   # info da sessão
                _lv_athletes_ph = st.empty()   # cards dos atletas
                _lv_alerts_ph   = st.empty()   # log de alertas

                # ── Função para renderizar os dados ───────────────────────
                def _render_live(info_data, athletes_data):
                    """Renderiza info da sessão + cards de atletas."""
                    import time as _tm

                    # ── Painel de sessão ──────────────────────────────────
                    with _lv_session_ph.container():
                        if info_data and isinstance(info_data, dict):
                            _si_c1, _si_c2, _si_c3, _si_c4 = st.columns(4)
                            _act_name = (
                                info_data.get('activity_name') or
                                info_data.get('name') or
                                info_data.get('id', '—')
                            )
                            _start_ts = info_data.get('start_time') or info_data.get('started_at') or 0
                            _elapsed  = ""
                            if _start_ts:
                                try:
                                    _el_s = int(_tm.time() - float(_start_ts))
                                    _elapsed = f"{_el_s//3600:02d}:{(_el_s%3600)//60:02d}:{_el_s%60:02d}"
                                except Exception:
                                    _elapsed = "—"
                            _n_atl = len(athletes_data) if isinstance(athletes_data, list) else 0
                            with _si_c1:
                                st.metric("🎯 Sessão", str(_act_name)[:28] or "Ativa")
                            with _si_c2:
                                st.metric("⏱️ Tempo decorrido", _elapsed or "—")
                            with _si_c3:
                                st.metric("👥 Atletas ativos", str(_n_atl))
                            with _si_c4:
                                st.metric("🔁 Última atualização",
                                          _tm.strftime('%H:%M:%S'))
                        elif athletes_data:
                            st.metric("👥 Atletas ativos",
                                      len(athletes_data) if isinstance(athletes_data, list) else "—")

                    # ── Cards dos atletas ──────────────────────────────────
                    with _lv_athletes_ph.container():
                        if not athletes_data:
                            st.info(
                                "📭 Nenhuma sessão ao vivo detectada.\n\n"
                                "**Verifique:**\n"
                                "- Existe uma atividade aberta no OpenField agora\n"
                                "- Os dispositivos estão transmitindo dados\n"
                                "- O token tem permissão para dados ao vivo"
                            )
                            return

                        # Normalizar resposta (pode ser lista ou dict com key 'athletes')
                        _atl_list = athletes_data
                        if isinstance(athletes_data, dict):
                            _atl_list = (
                                athletes_data.get('athletes') or
                                athletes_data.get('data') or
                                list(athletes_data.values())
                            )
                        if not isinstance(_atl_list, list):
                            st.warning("⚠️ Formato de resposta inesperado da API /live.")
                            st.json(athletes_data)
                            return

                        # Paleta de cores por atleta
                        _LV_COLORS = [
                            '#FF6B6B','#4ECDC4','#45B7D1','#96CEB4',
                            '#FFEAA7','#DDA0DD','#98FB98','#FFB347',
                        ]

                        # ── Mapeamento de campos (diferentes versões da API) ──
                        def _get_field(d, *keys, default=None):
                            for k in keys:
                                if k in d:
                                    return d[k]
                            return default

                        # ── Grid de cards (3 por linha) ───────────────────
                        _n_cols_lv = min(3, len(_atl_list))
                        _lv_rows   = [
                            _atl_list[i:i+_n_cols_lv]
                            for i in range(0, len(_atl_list), _n_cols_lv)
                        ]

                        _new_alerts = []
                        import time as _tm2

                        for _row in _lv_rows:
                            _cols = st.columns(len(_row))
                            for _ci, (_col, _atl) in enumerate(zip(_cols, _row)):
                                if not isinstance(_atl, dict):
                                    continue
                                _color = _LV_COLORS[_ci % len(_LV_COLORS)]
                                _name  = (
                                    _get_field(_atl, 'name', 'athlete_name',
                                               'display_name', default='Atleta')
                                )

                                # Extrair métricas (tentativa em vários campos)
                                _vel  = _get_field(_atl, 'velocity', 'current_velocity',
                                                   'speed', 'v', default=None)
                                _hr   = _get_field(_atl, 'heart_rate', 'hr',
                                                   'current_heart_rate', default=None)
                                _pl   = _get_field(_atl, 'player_load', 'playerload',
                                                   'total_player_load', 'pl', default=None)
                                _acc  = _get_field(_atl, 'acceleration', 'acc',
                                                   'current_acceleration', 'a', default=None)
                                _dist = _get_field(_atl, 'total_distance', 'distance',
                                                   'odometer', 'o', default=None)

                                # Verificar limiares
                                _card_alerts = []
                                _metric_map = {
                                    "velocity":       (_vel,  _lv_thresholds["velocity"]),
                                    "heart_rate":     (_hr,   _lv_thresholds["heart_rate"]),
                                    "acceleration":   (_acc,  _lv_thresholds["acceleration"]),
                                    "total_distance": (_dist, _lv_thresholds["total_distance"]),
                                    "player_load":    (_pl,   _lv_thresholds["player_load"]),
                                }
                                for _mk, (_mv, (_thr, _unit, _ico)) in _metric_map.items():
                                    if _mv is not None and _thr > 0:
                                        try:
                                            if float(_mv) >= float(_thr):
                                                _card_alerts.append(
                                                    f"{_ico} {_mk.replace('_',' ').title()}: "
                                                    f"**{float(_mv):.1f}** {_unit} "
                                                    f"(limiar: {_thr})"
                                                )
                                                _new_alerts.append({
                                                    'ts': _tm2.strftime('%H:%M:%S'),
                                                    'atleta': _name,
                                                    'metrica': _mk,
                                                    'valor': float(_mv),
                                                    'limiar': _thr,
                                                    'unidade': _unit,
                                                    'ico': _ico,
                                                })
                                        except Exception:
                                            _applog.log_debug_exc()

                                _has_alert = bool(_card_alerts)

                                # Cor do card por velocidade
                                try:
                                    _v_num = float(_vel) if _vel is not None else 0
                                    _vel_zone_color = (
                                        '#1e3a5f' if _v_num < 7 else
                                        '#1a3d2b' if _v_num < 14 else
                                        '#3d2b00' if _v_num < 19 else
                                        '#3d1a1a'
                                    )
                                except Exception:
                                    _vel_zone_color = '#1e293b'

                                _border_color = '#ef4444' if _has_alert else _color

                                with _col:
                                    # Card HTML
                                    st.markdown(
                                        f"<div style='background:{_vel_zone_color};"
                                        f"border:2px solid {_border_color};"
                                        f"border-radius:10px;padding:14px 12px;"
                                        f"margin-bottom:8px;min-height:160px'>"
                                        f"<div style='display:flex;justify-content:space-between;"
                                        f"align-items:center;margin-bottom:8px'>"
                                        f"<span style='font-weight:700;font-size:14px;"
                                        f"color:{_color}'>{_name[:18]}</span>"
                                        + (
                                            "<span style='background:#ef4444;color:white;"
                                            "font-size:10px;padding:2px 6px;border-radius:10px'>"
                                            "⚠️ ALERTA</span>"
                                            if _has_alert else
                                            "<span style='background:#10b981;color:white;"
                                            "font-size:10px;padding:2px 6px;border-radius:10px'>"
                                            "✅ OK</span>"
                                        ) +
                                        "</div>"
                                        # Métricas
                                        + (
                                            f"<div style='font-size:22px;font-weight:800;"
                                            f"color:white;margin:4px 0'>"
                                            f"🚀 {float(_vel):.1f} <span style='font-size:11px;"
                                            f"color:#94a3b8'>km/h</span></div>"
                                            if _vel is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"❤️ {float(_hr):.0f} bpm</div>"
                                            if _hr is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"⚡ PL {float(_pl):.1f}</div>"
                                            if _pl is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"💥 {float(_acc):.2f} m/s²</div>"
                                            if _acc is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"📏 {float(_dist):.0f} m</div>"
                                            if _dist is not None else ""
                                        )
                                        + "</div>",
                                        unsafe_allow_html=True,
                                    )
                                    # Detalhes do alerta abaixo do card
                                    if _card_alerts and _lv_sound:
                                        for _al in _card_alerts:
                                            st.markdown(
                                                f"<div style='background:#450a0a;border-left:"
                                                f"3px solid #ef4444;border-radius:4px;"
                                                f"padding:4px 8px;font-size:11px;color:#fca5a5;"
                                                f"margin-bottom:3px'>{_al}</div>",
                                                unsafe_allow_html=True,
                                            )

                        # Adicionar novos alertas ao log (sem duplicar os últimos 5s)
                        if _new_alerts:
                            _existing_keys = {
                                (a['atleta'], a['metrica'], a['ts'])
                                for a in st.session_state['live_alert_log'][-20:]
                            }
                            for _na in _new_alerts:
                                _key = (_na['atleta'], _na['metrica'], _na['ts'])
                                if _key not in _existing_keys:
                                    st.session_state['live_alert_log'].append(_na)

                    # ── Log de alertas ────────────────────────────────────
                    with _lv_alerts_ph.container():
                        _log = st.session_state['live_alert_log']
                        if _log:
                            st.markdown("#### 🔴 Histórico de Alertas")
                            _df_log = pd.DataFrame(list(reversed(_log[-50:])))
                            _df_log.columns = [
                                'Hora', 'Atleta', 'Métrica',
                                'Valor', 'Limiar', 'Unidade', '—'
                            ]
                            st.dataframe(
                                _df_log[['Hora','Atleta','Métrica','Valor','Limiar','Unidade']],
                                use_container_width=True,
                                hide_index=True,
                            )

                # ── Loop de polling ────────────────────────────────────────
                _lv_api = st.session_state.get('api')

                if _lv_api is None:
                    st.warning(
                        "⚠️ API não inicializada. Carregue os dados na sidebar primeiro."
                    )
                elif st.session_state['live_active'] or _lv_refresh_now:
                    import time as _lv_time

                    # Buscar dados
                    _lv_info  = _lv_api.get_live_info()
                    _lv_data  = _lv_api.get_live_athletes()

                    # Guardar snapshot para exibição após parar
                    st.session_state['live_snapshot']      = _lv_data
                    st.session_state['live_info_snapshot'] = _lv_info

                    _render_live(_lv_info, _lv_data)

                    # Continuar polling se ativo
                    if st.session_state['live_active'] and not _lv_refresh_now:
                        _lv_time.sleep(int(st.session_state.get('live_interval', 10)))
                        st.rerun()

                elif st.session_state.get('live_snapshot') is not None:
                    # Mostrar último snapshot após pausar
                    st.caption("📸 Último snapshot (monitoramento pausado)")
                    _render_live(
                        st.session_state['live_info_snapshot'],
                        st.session_state['live_snapshot'],
                    )
                else:
                    # Estado inicial
                    st.markdown(
                        "<div style='text-align:center;padding:48px 0;"
                        "color:#64748b'>"
                        "<div style='font-size:48px;margin-bottom:12px'>📡</div>"
                        "<div style='font-size:18px;font-weight:600;color:#94a3b8'>"
                        "Monitoramento ao vivo pronto</div>"
                        "<div style='font-size:14px;margin-top:8px'>"
                        "Configure os limiares acima e clique em "
                        "<b style='color:#10b981'>▶ Iniciar Monitoramento</b><br>"
                        "para começar a acompanhar a sessão em tempo real."
                        "</div></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "> **Pré-requisito:** uma sessão deve estar aberta e transmitindo "
                        "dados no OpenField Cloud agora. O endpoint `/live` só retorna dados "
                        "durante sessões ativas."
                    )
