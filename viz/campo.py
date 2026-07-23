# -*- coding: utf-8 -*-
"""Aba extraída de main() (P4 — page-decomposition)."""
from __future__ import annotations

from config import FUTEBOL_EVENTS_CONFIG
from config import _DEFAULT_VELOCITY_ZONES
import applog as _applog
from bands import _bandas_acc_ativas
from bands import _bandas_vel_ativas
from persistence import _carregar_venues
from persistence import _excluir_venue
from scipy.ndimage import gaussian_filter as _gf
from bands import _legenda_vel_items
from persistence import _salvar_venue
from field import adicionar_convex_hull
from field import adicionar_eventos_campo
from field import adicionar_grade_quadrantes
from field import adicionar_pontos_aceleracao_bandas
from field import adicionar_pontos_velocidade_bandas
from field import adicionar_setas_direcao
from field import adicionar_tercos_campo
from field import adicionar_trajetoria_campo
from analysis import calcular_efforts_aceleracao_sensor
from analysis import calcular_efforts_velocidade_sensor
from field import calcular_voronoi_campo
from field import criar_html_campo_fixo
from datetime import datetime
from field import desenhar_campo_futebol_bonito
from field import enriquecer_esforcos_taticos
from field import enriquecer_eventos_com_posicao
from analysis import get_min_dur_s
from analysis import get_min_dur_vel_s
import plotly.graph_objects as go
from field import gps_para_campo_coords
import numpy as np
import pandas as pd
from analysis import processar_efforts_aceleracao
from analysis import processar_efforts_velocidade
import streamlit as st
from field import stats_quadrante


def render_campo(REFERENCIAS, _campo_component, dados_efforts_acc_por_periodo, dados_efforts_vel_por_periodo, dados_eventos_por_periodo, dados_posicao_por_periodo, resultados_por_periodo):
                st.subheader("🗺️ Campo de Futebol — Análise de Movimentação")
                st.caption(REFERENCIAS["campo"])

                # ── Split View toggle ──────────────────────────────────────────
                _split_mode = st.toggle("⚖️ Modo Comparação (Split View)", value=False, key="split_view_mode")

                if _split_mode and dados_posicao_por_periodo:
                    _split_col_A, _split_col_B = st.columns(2)
                    for _split_idx, _split_col in enumerate([_split_col_A, _split_col_B]):
                        with _split_col:
                            _lbl = "A" if _split_idx == 0 else "B"
                            st.markdown(f"**Lado {_lbl}**")
                            _sp_per = st.selectbox(f"Período {_lbl}:", list(dados_posicao_por_periodo.keys()), key=f"split_per_{_lbl}")
                            _sp_ats = list(dados_posicao_por_periodo.get(_sp_per, {}).keys())
                            _sp_atl = st.selectbox(f"Atleta {_lbl}:", _sp_ats, key=f"split_atl_{_lbl}") if _sp_ats else None
                            if _sp_atl:
                                _sp_d = dados_posicao_por_periodo.get(_sp_per, {}).get(_sp_atl, {})
                                _sp_xs = list(_sp_d.get('xs', []))
                                _sp_ys = list(_sp_d.get('ys', []))
                                _sp_vs = list(_sp_d.get('vel', _sp_d.get('vels_gps', [])))

                                # ── Fallback GPS: converte lat/lon se não há x/y ────
                                _sp_gps_used = False
                                if not _sp_xs:
                                    _sp_lats = _sp_d.get('lats', [])
                                    _sp_lons = _sp_d.get('lons', [])
                                    if _sp_lats and _sp_lons:
                                        # Busca qualquer cfg de campo salvo em session_state
                                        _sp_cfg = None
                                        for _ssk, _ssv in st.session_state.items():
                                            if (isinstance(_ssk, str)
                                                    and _ssk.startswith('campo_cfg__')
                                                    and isinstance(_ssv, dict)
                                                    and _ssv.get('fl')):
                                                _sp_cfg = _ssv
                                                break
                                        if _sp_cfg:
                                            try:
                                                _sp_xs, _sp_ys = gps_para_campo_coords(
                                                    _sp_lats, _sp_lons, _sp_cfg)
                                                _sp_vs = list(_sp_d.get('vels_gps', [0.0]*len(_sp_xs)))
                                                _sp_gps_used = True
                                            except Exception:
                                                _applog.log_debug_exc()

                                if _sp_xs:
                                    # Direção de ataque para este lado do split view
                                    _sp_atk_opts = ["— Não definido", "➡️  Esq → Dir", "⬅️  Dir → Esq"]
                                    _sp_atk_key = f"attack_dir_split_{_lbl}"
                                    _sp_atk_default = 0
                                    if st.session_state.get(_sp_atk_key) == 'left_to_right':
                                        _sp_atk_default = 1
                                    elif st.session_state.get(_sp_atk_key) == 'right_to_left':
                                        _sp_atk_default = 2
                                    _sp_atk_sel = st.radio(
                                        "⚽ Ataque:",
                                        _sp_atk_opts,
                                        index=_sp_atk_default,
                                        horizontal=True,
                                        key=f"_sp_atk_radio_{_lbl}",
                                    )
                                    if "➡️" in _sp_atk_sel:
                                        st.session_state[_sp_atk_key] = 'left_to_right'
                                    elif "⬅️" in _sp_atk_sel:
                                        st.session_state[_sp_atk_key] = 'right_to_left'
                                    else:
                                        st.session_state[_sp_atk_key] = None
                                    _sp_fig = desenhar_campo_futebol_bonito(
                                        title=f"{_sp_atl} — {_sp_per}"
                                              + (" (GPS)" if _sp_gps_used else ""),
                                        attack_direction=st.session_state[_sp_atk_key],
                                    )
                                    _sp_step = max(1, len(_sp_xs) // 5000)
                                    _sp_vs_sub = ([_sp_vs[i] for i in range(0, len(_sp_vs), _sp_step)]
                                                  if _sp_vs else [0] * len(_sp_xs[::_sp_step]))
                                    _sp_fig.add_trace(go.Scatter(
                                        x=_sp_xs[::_sp_step], y=_sp_ys[::_sp_step],
                                        mode='markers',
                                        marker=dict(
                                            size=3,
                                            color=_sp_vs_sub,
                                            colorscale=[[0,'#2196F3'],[0.4,'#4CAF50'],[0.7,'#FF9800'],[1,'#F44336']],
                                            showscale=False,
                                        ),
                                        hoverinfo='skip', showlegend=False,
                                    ))
                                    _sp_fig.update_layout(height=400, margin=dict(t=30,b=10,l=10,r=10))
                                    st.plotly_chart(_sp_fig, use_container_width=True)
                                else:
                                    _sp_has_gps = bool(_sp_d.get('lats'))
                                    if _sp_has_gps:
                                        st.warning(
                                            "📡 Dados GPS disponíveis mas campo não calibrado.\n\n"
                                            "Vá até a aba **Campo & GPS** no modo normal, aplique o campo "
                                            "no mapa e retorne ao Split View."
                                        )
                                    else:
                                        st.info("Sem dados de posição (x/y ou GPS) para este atleta/período.")
                elif dados_posicao_por_periodo:
                    _todos_periodos = list(dados_posicao_por_periodo.keys())
                    col1, col2 = st.columns(2)
                    with col1:
                        periodos_mapa_sel = st.multiselect(
                            "Selecione o(s) período(s):",
                            options=_todos_periodos,
                            default=_todos_periodos[:1],
                            key="periodos_mapa_sel"
                        )
                    atleta_mapa = None
                    # Período de referência (primeiro selecionado) — usado para config do campo
                    periodo_mapa = periodos_mapa_sel[0] if periodos_mapa_sel else _todos_periodos[0]
                    with col2:
                        # União de atletas disponíveis em todos os períodos selecionados
                        _ats_disponiveis = []
                        for _pm in (periodos_mapa_sel or [periodo_mapa]):
                            for _a in dados_posicao_por_periodo.get(_pm, {}).keys():
                                if _a not in _ats_disponiveis:
                                    _ats_disponiveis.append(_a)
                        if _ats_disponiveis:
                            atletas_mapa = st.multiselect(
                                "Selecione o(s) atleta(s):",
                                _ats_disponiveis,
                                default=_ats_disponiveis[:1],
                                key="atletas_mapa"
                            )
                        else:
                            atletas_mapa = []
                    # Atleta primário: usado para GPS, calibração do campo e ETAPA 2
                    atleta_mapa = atletas_mapa[0] if atletas_mapa else None

                    if not periodos_mapa_sel:
                        st.info("Selecione pelo menos um período.")
                    elif atleta_mapa:
                        # Combina dados de todos os períodos selecionados
                        dados = dados_posicao_por_periodo.get(periodo_mapa, {}).get(atleta_mapa, {})

                        # GPS combinado (todos os períodos)
                        lats_gps, lons_gps, vels_gps, ts_gps = [], [], [], []
                        for _pm in periodos_mapa_sel:
                            _d = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                            lats_gps  += _d.get('lats', [])
                            lons_gps  += _d.get('lons', [])
                            vels_gps  += _d.get('vels_gps', [])
                            ts_gps    += _d.get('ts_gps', [])

                        n_xy  = sum(dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {}).get('n_pontos', 0) for _pm in periodos_mapa_sel)
                        n_gps = len(lats_gps)
                        _label_periodos = " + ".join(periodos_mapa_sel)
                        st.caption(f"📡 Pontos campo (x/y): **{n_xy}** &nbsp;|&nbsp; 🌍 Pontos GPS: **{n_gps}** &nbsp;|&nbsp; 📅 **{_label_periodos}**")

                        # ── Item 8: Badge de qualidade GPS ──────────────────────
                        st.caption("ℹ️ **HDOP** — Horizontal Dilution of Precision: qualidade do sinal GPS. <1.5 = excelente, 1.5–3 = bom, >3 = ruim", unsafe_allow_html=False)
                        _gps_q_parts = []
                        for _pm in periodos_mapa_sel:
                            _dq = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                            _pq_m  = _dq.get('pq_mean')
                            _hdop_m = _dq.get('hdop_mean')
                            _ref_m = _dq.get('ref_mean')
                            if _pq_m is not None or _hdop_m is not None:
                                _gps_q_parts.append((_pm, _pq_m, _hdop_m, _ref_m))
                        if _gps_q_parts:
                            _gps_badge_html = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px'>"
                            for _pm, _pq, _hdop, _ref in _gps_q_parts:
                                # pq: >80% good, 60-80% ok, <60% poor; hdop: <1.5 excellent, <3 good, >3 poor
                                _pq_color  = ('#4CAF50' if (_pq  or 0) >= 80 else '#FF9800' if (_pq  or 0) >= 60 else '#F44336') if _pq  is not None else '#888'
                                _hdop_color= ('#4CAF50' if (_hdop or 0) <= 1.5 else '#FF9800' if (_hdop or 0) <= 3.0 else '#F44336') if _hdop is not None else '#888'
                                _pq_txt    = f"PQ: <b style='color:{_pq_color}'>{_pq:.0f}%</b>" if _pq  is not None else ""
                                _hdop_txt  = f" HDOP: <b style='color:{_hdop_color}'>{_hdop:.2f}</b>" if _hdop is not None else ""
                                _ref_txt   = f" Satélites: <b>{_ref:.0f}</b>" if _ref is not None else ""
                                _gps_badge_html += (
                                    f"<span style='background:#1e2d3d;border:1px solid #2d4a6a;border-radius:8px;"
                                    f"padding:4px 10px;font-size:11px;color:#ccc'>"
                                    f"📡 {_pm} — {_pq_txt}{_hdop_txt}{_ref_txt}</span>"
                                )
                            _gps_badge_html += "</div>"
                            st.markdown(_gps_badge_html, unsafe_allow_html=True)

                        # ── Item 12: Odômetro vs distância integrada ─────────────
                        _odo_vals, _dist_vals = [], []
                        for _pm in periodos_mapa_sel:
                            _dq = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                            _odo = _dq.get('odometro_m')
                            if _odo is not None:
                                _odo_vals.append(_odo)
                        _res_rows = resultados_por_periodo.get(periodos_mapa_sel[0], []) if periodos_mapa_sel else []
                        _dist_integrada = next((r.get('Distância (m)', 0) for r in _res_rows if r.get('Atleta') == atleta_mapa), None)
                        if _odo_vals and _dist_integrada:
                            _odo_total = sum(_odo_vals)
                            _div = abs(_odo_total - _dist_integrada)
                            _div_pct = _div / max(_dist_integrada, 1) * 100
                            _odo_col = '#4CAF50' if _div_pct < 5 else '#FF9800' if _div_pct < 15 else '#F44336'
                            st.markdown(
                                f"<div style='display:flex;gap:12px;align-items:center;margin-bottom:6px;"
                                f"background:#0d1f0d;border-left:3px solid #4CAF50;padding:6px 12px;border-radius:0 6px 6px 0'>"
                                f"<span style='color:#86efac;font-size:12px'>🛣️ <b>Odômetro Catapult</b></span>"
                                f"<span style='color:white;font-size:13px'><b>{_odo_total:,.0f} m</b></span>"
                                f"<span style='color:#aaa;font-size:11px'>vs. GPS integrado: {_dist_integrada:,.0f} m</span>"
                                f"<span style='color:{_odo_col};font-size:11px'>Δ {_div_pct:.1f}%</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        # Chave por atleta (campo físico não muda entre períodos)
                        campo_key = f"campo_cfg__{atleta_mapa}"

                        # ── Auto-carga do banco compartilhado de venues ──────────
                        # Se o venue da atividade já foi configurado por algum usuário,
                        # aplica automaticamente — sem precisar passar pela FASE 1.
                        _venues_db   = _carregar_venues()
                        _venue_name  = st.session_state.get('venue', {}).get('name', '')
                        if (_venue_name and _venue_name in _venues_db
                                and campo_key not in st.session_state):
                            _vs = _venues_db[_venue_name]
                            st.session_state[campo_key] = {
                                'lat':          float(_vs.get('lat') or 0),
                                'lon':          float(_vs.get('lon') or 0),
                                'rot':          int(_vs.get('rot',   0)),
                                'fl':           int(_vs.get('fl',  105)),
                                'fw':           int(_vs.get('fw',   68)),
                                'ig':           int(_vs.get('ig',    1)),
                                '_from_venues': _venue_name,
                                '_saved_at':    _vs.get('saved_at', ''),
                            }

                        campo_aplicado = campo_key in st.session_state

                        # ══════════════════════════════════════════════════════
                        # FASE 1 — POSICIONAMENTO INTERATIVO NO SATÉLITE
                        # ══════════════════════════════════════════════════════
                        if not campo_aplicado:
                            st.markdown("### 1️⃣ Posicionamento no Campo Físico")
                            st.markdown(
                                "Ajuste o campo de futebol sobre a imagem de satélite e clique "
                                "**✅ Aplicar Campo** no painel inferior do mapa."
                            )

                            if lats_gps and lons_gps:
                                st.info(
                                    "📌 **Edite Lat/Lon** (ou use ↑↓) para mover o ⊙ amarelo · "
                                    "⚽ **Mostrar Campo** para ativar o overlay · "
                                    "Sliders ajustam rotação e dimensões **em tempo real** · "
                                    "**✅ Aplicar Campo** quando estiver satisfeito"
                                )

                                # Subamostrar GPS para o componente (máx 3000 pts)
                                _n = len(lats_gps)
                                _step = max(1, _n // 3000)
                                _pts = [
                                    {"lat": round(lats_gps[i], 7),
                                     "lon": round(lons_gps[i], 7),
                                     "v":   round(float(vels_gps[i]) if i < len(vels_gps) else 0.0, 1)}
                                    for i in range(0, _n, _step)
                                ]

                                # ── Venue da atividade → pré-popula campo ────────────
                                # Centro do mapa: sempre usa mediana GPS real dos atletas
                                # (o venue.lat/lng da API pode estar cadastrado num local errado).
                                # Dimensões e rotação do venue são usadas quando disponíveis.
                                _venue_info = st.session_state.get('venue', {})
                                _venue_rot  = int(_venue_info.get('rotation') or 0)
                                # Dimensões padrão fixas (105×68m). O usuário ainda pode
                                # ajustar comprimento/largura no painel do mapa; apenas os
                                # valores de ENTRADA padrão são 105×68 (não os do venue).
                                _venue_fl   = 105.0
                                _venue_fw   = 68.0

                                # Posição do mapa: mediana GPS > venue lat/lng > 0,0
                                if lats_gps:
                                    _lat_c = round(float(np.median(lats_gps)), 7)
                                    _lon_c = round(float(np.median(lons_gps)), 7)
                                else:
                                    _vl = _venue_info.get('lat')
                                    _vg = _venue_info.get('lng')
                                    _lat_c = round(float(_vl), 7) if _vl else 0.0
                                    _lon_c = round(float(_vg), 7) if _vg else 0.0

                                if _venue_info:
                                    st.caption(
                                        f"📡 Campo padrão: **{_venue_fl:.0f}×{_venue_fw:.0f}m** · "
                                        f"rot **{_venue_rot}°** · "
                                        f"centro **{_lat_c:.5f}, {_lon_c:.5f}** "
                                        f"_(ajustável no painel do mapa)_"
                                    )

                                # Componente bidirecional: retorna {lat,lon,rot,fl,fw,ig}
                                # quando o usuário clica "✅ Aplicar Campo" no painel do mapa
                                resultado_campo = _campo_component(
                                    pts=_pts,
                                    lat_c=_lat_c,
                                    lon_c=_lon_c,
                                    rot=_venue_rot,
                                    fl=_venue_fl,
                                    fw=_venue_fw,
                                    legend=_legenda_vel_items(),
                                    key=f"campo_mapa_{atleta_mapa}",
                                    default=None
                                )

                                if resultado_campo is not None:
                                    _novo_cfg = {
                                        'lat': float(resultado_campo['lat']),
                                        'lon': float(resultado_campo['lon']),
                                        'rot': int(resultado_campo['rot']),
                                        'fl':  int(resultado_campo['fl']),
                                        'fw':  int(resultado_campo['fw']),
                                        'ig':  int(resultado_campo['ig']),
                                    }
                                    st.session_state[campo_key] = _novo_cfg
                                    st.success("✅ Campo aplicado com sucesso!")

                                    # ── Oferecer salvar no banco compartilhado ──────
                                    _vdb_atual  = _carregar_venues()
                                    _def_vname  = (_venue_name
                                                   or st.session_state.get('venue', {}).get('name', '')
                                                   or '')
                                    st.markdown("---")
                                    st.markdown("#### 💾 Deseja salvar no banco compartilhado de venues?")
                                    st.caption(
                                        "Outros usuários que abrirem uma atividade **neste mesmo venue** "
                                        "terão o campo configurado automaticamente."
                                    )
                                    with st.form("_form_save_venue"):
                                        _vname_inp = st.text_input(
                                            "Nome do venue:",
                                            value=_def_vname,
                                            placeholder="Ex: Estádio Municipal / CT do Clube",
                                        )
                                        if _vname_inp and _vname_inp.strip() in _vdb_atual:
                                            st.warning(
                                                f"⚠️ Já existe um venue salvo com este nome "
                                                f"(salvo em {_vdb_atual[_vname_inp.strip()].get('saved_at','?')}). "
                                                f"Confirme para sobrescrever."
                                            )
                                        _c1, _c2 = st.columns(2)
                                        with _c1:
                                            _btn_salvar = st.form_submit_button(
                                                "💾 Salvar para todos", type="primary")
                                        with _c2:
                                            _btn_pular  = st.form_submit_button("⏭ Pular")

                                    if _btn_salvar and _vname_inp.strip():
                                        _salvar_venue(_vname_inp.strip(), _novo_cfg)
                                        st.success(
                                            f"✅ Venue **{_vname_inp.strip()}** salvo! "
                                            f"Outros usuários serão beneficiados automaticamente."
                                        )
                                        st.session_state[campo_key]['_from_venues'] = _vname_inp.strip()
                                        st.session_state[campo_key]['_saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                                    if _btn_salvar or _btn_pular:
                                        st.rerun()
                            else:
                                st.warning(
                                    "⚠️ Nenhum ponto GPS real (lat/lon) encontrado para este atleta.\n\n"
                                    "Isso pode ocorrer se o sensor não obteve lock GPS durante a sessão."
                                )

                        # ══════════════════════════════════════════════════════
                        # FASE 2 — CAMPO APLICADO + ANÁLISE DE ESFORÇOS
                        # ══════════════════════════════════════════════════════
                        else:
                            cfg = st.session_state[campo_key]
                            _cfg_from_vn  = cfg.get('_from_venues', '')
                            _cfg_saved_at = cfg.get('_saved_at', '')

                            # ── Banner de origem ─────────────────────────────────
                            if _cfg_from_vn:
                                st.success(
                                    f"📦 Campo carregado automaticamente do banco compartilhado: "
                                    f"**{_cfg_from_vn}**"
                                    + (f" · salvo em {_cfg_saved_at}" if _cfg_saved_at else "")
                                )

                            # Botão para reajustar (volta à Fase 1)
                            col_hdr, col_btn, col_mgr = st.columns([4, 1, 1])
                            with col_hdr:
                                st.markdown(
                                    f"### 1️⃣ Campo Aplicado  "
                                    f"<span style='font-size:13px;color:#90CAF9'>"
                                    f"📍 {cfg['lat']:.5f}, {cfg['lon']:.5f} &nbsp; "
                                    f"🧭 {cfg['rot']}° &nbsp; "
                                    f"📏 {cfg['fl']}×{cfg['fw']}m</span>",
                                    unsafe_allow_html=True
                                )
                            with col_btn:
                                if st.button("🔄 Reajustar", key="btn_reajustar"):
                                    del st.session_state[campo_key]
                                    st.rerun()
                            with col_mgr:
                                if st.button("🗄️ Venues", key="btn_mgr_venues"):
                                    st.session_state['_show_venue_mgr'] = not st.session_state.get('_show_venue_mgr', False)

                            # ── Gerenciador de venues ────────────────────────────
                            if st.session_state.get('_show_venue_mgr', False):
                                with st.expander("🗄️ Banco Compartilhado de Venues", expanded=True):
                                    _vdb = _carregar_venues()
                                    if not _vdb:
                                        st.info("Nenhum venue salvo ainda. Configure um campo e clique em 💾 Salvar.")
                                    else:
                                        for _vn, _vc in _vdb.items():
                                            _v1, _v2, _v3, _v4 = st.columns([3, 2, 2, 1])
                                            with _v1:
                                                st.markdown(f"**{_vn}**")
                                            with _v2:
                                                st.caption(f"📏 {_vc.get('fl',105)}×{_vc.get('fw',68)}m · 🧭 {_vc.get('rot',0)}°")
                                            with _v3:
                                                st.caption(f"💾 {_vc.get('saved_at','?')}")
                                            with _v4:
                                                if st.button("🗑️", key=f"_del_vn_{_vn}",
                                                             help=f"Excluir {_vn}"):
                                                    _excluir_venue(_vn)
                                                    st.rerun()
                                    # Salvar config atual com novo nome
                                    st.markdown("---")
                                    st.markdown("**Salvar configuração atual:**")
                                    with st.form("_form_mgr_save"):
                                        _mgr_nome = st.text_input(
                                            "Nome:", value=_venue_name or '',
                                            key="_mgr_vname")
                                        if st.form_submit_button("💾 Salvar / Atualizar", type="primary"):
                                            if _mgr_nome.strip():
                                                _salvar_venue(_mgr_nome.strip(), cfg)
                                                st.success(f"✅ **{_mgr_nome.strip()}** salvo!")
                                                st.rerun()

                            # ── Esforço selecionado (para filtrar GPS no mapa e animação) ──
                            lats_eff, lons_eff, vels_eff, eff_desc = [], [], [], ""
                            _anim_start_ts  = 0.0
                            _anim_end_ts    = 0.0
                            _anim_effort_row = None

                            # ── Arrays de coordenadas (fonte única para ETAPA 2 e ETAPA 3) ──
                            # Construídos aqui para que detecção de esforços e visualização
                            # do campo usem exatamente os mesmos dados — garantindo que
                            # _seg_start_idx dos esforços indexe corretamente xn/yn.
                            xn, yn, vel_raw_campo, acc_raw_campo, ts_pos_campo = [], [], [], [], []
                            _fonte_xy = False
                            for _pm in periodos_mapa_sel:
                                _dc = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                                if _dc.get('xs') and _dc.get('ys') and _dc.get('vel'):
                                    xn  += list(_dc['xs'])
                                    yn  += list(_dc['ys'])
                                    vel_raw_campo += list(_dc['vel'])
                                    acc_raw_campo += list(_dc.get('acc', [0.0]*len(_dc['xs'])))
                                    ts_pos_campo  += list(_dc.get('ts_pos', []))
                                    _fonte_xy = True
                                elif lats_gps and cfg:
                                    _lats_pm = _dc.get('lats', [])
                                    _lons_pm = _dc.get('lons', [])
                                    if _lats_pm:
                                        _gx, _gy = gps_para_campo_coords(_lats_pm, _lons_pm, cfg)
                                        xn  += _gx
                                        yn  += _gy
                                        vel_raw_campo += _dc.get('vels_gps', [0.0]*len(_gx))
                                        acc_raw_campo += [0.0]*len(_gx)
                            _has_xy  = bool(xn and yn)
                            _has_gps = bool(lats_gps and lons_gps and cfg)

                            # ── Coordenadas de campo + GPS por atleta (ETAPA 2) ──
                            # Necessário para calcular esforços de todos os atletas e
                            # recuperar o segmento correto ao selecionar uma linha.
                            _coords_atletas = {}
                            for _atl in atletas_mapa:
                                _xna, _yna, _vela, _acca, _tsa = [], [], [], [], []
                                _latsa, _lonsa, _vgpsa, _tgpsa = [], [], [], []
                                for _pm in periodos_mapa_sel:
                                    _dca = dados_posicao_por_periodo.get(_pm, {}).get(_atl, {})
                                    if _dca.get('xs') and _dca.get('ys') and _dca.get('vel'):
                                        _xna  += list(_dca['xs'])
                                        _yna  += list(_dca['ys'])
                                        _vela += list(_dca['vel'])
                                        _acca += list(_dca.get('acc', [0.0]*len(_dca['xs'])))
                                        _tsa  += list(_dca.get('ts_pos', []))
                                    elif cfg:
                                        _lp = _dca.get('lats', [])
                                        _lo = _dca.get('lons', [])
                                        if _lp:
                                            _gx, _gy = gps_para_campo_coords(_lp, _lo, cfg)
                                            _xna  += _gx;  _yna  += _gy
                                            _vela += _dca.get('vels_gps', [0.0]*len(_gx))
                                            _acca += [0.0]*len(_gx)
                                    _latsa  += _dca.get('lats', [])
                                    _lonsa  += _dca.get('lons', [])
                                    _vgpsa  += _dca.get('vels_gps', [])
                                    _tgpsa  += _dca.get('ts_gps', [])
                                _coords_atletas[_atl] = dict(
                                    xn=_xna, yn=_yna, vel=_vela, acc=_acca, ts=_tsa,
                                    lats=_latsa, lons=_lonsa, vels_gps=_vgpsa, ts_gps=_tgpsa,
                                )

                            # Atleta do esforço selecionado (atualizado na seleção de linha)
                            _sel_atleta = atleta_mapa

                            # Mapa fixo (atualizado abaixo se houver esforço selecionado)
                            mapa_placeholder = st.empty()

                            st.divider()

                            # ── ETAPA 2: Análise de esforços ─────────────────────
                            st.markdown("### 2️⃣ Análise de Esforços no Campo")

                            _min_dur_vel_s = get_min_dur_vel_s()
                            _min_dur_acc_s = get_min_dur_s()

                            tipo_esf = st.radio(
                                "Tipo de esforço:",
                                ["⚡ Velocidade", "🔁 Aceleração"],
                                horizontal=True, key="tipo_esf_campo"
                            )

                            # ── Detecção de esforços para TODOS os atletas ──────────
                            efforts_df_full = pd.DataFrame()
                            _esf_usou_api = False
                            _frames_esf = []

                            for _atl in atletas_mapa:
                                _c = _coords_atletas[_atl]
                                _df_atl = pd.DataFrame()

                                # ── Tentativa 1: Esforços da API Catapult (fonte primária) ──
                                # Mais precisos — calculados pelos algoritmos oficiais
                                # da Catapult com os mesmos limiares do OpenField.
                                _raw_api: list = []
                                for _pm in periodos_mapa_sel:
                                    if tipo_esf == "⚡ Velocidade":
                                        _raw_api += dados_efforts_vel_por_periodo.get(
                                            _pm, {}).get(_atl, [])
                                    else:
                                        _raw_api += dados_efforts_acc_por_periodo.get(
                                            _pm, {}).get(_atl, [])
                                if _raw_api:
                                    _esf_usou_api = True
                                    _df_atl = (processar_efforts_velocidade(_raw_api)
                                               if tipo_esf == "⚡ Velocidade"
                                               else processar_efforts_aceleracao(_raw_api))

                                # ── Fallback: cálculo local com dados do sensor ──────
                                # Usado apenas se a API não retornou esforços.
                                if _df_atl.empty and _c['xn']:
                                    if tipo_esf == "⚡ Velocidade":
                                        _df_atl = calcular_efforts_velocidade_sensor(
                                            _c['xn'], _c['yn'], _c['vel'], _c['ts'],
                                            min_dur_s=_min_dur_vel_s)
                                    else:
                                        _df_atl = calcular_efforts_aceleracao_sensor(
                                            _c['xn'], _c['yn'], _c['acc'], _c['vel'], _c['ts'],
                                            min_dur_s=_min_dur_acc_s)

                                if not _df_atl.empty:
                                    _df_atl['Atleta'] = _atl
                                    _frames_esf.append(_df_atl)

                            if _frames_esf:
                                efforts_df_full = pd.concat(_frames_esf, ignore_index=True)
                                if '_start_ts' in efforts_df_full.columns and \
                                        efforts_df_full['_start_ts'].sum() > 0:
                                    efforts_df_full = efforts_df_full.sort_values(
                                        '_start_ts').reset_index(drop=True)
                                efforts_df_full['Esforço'] = range(1, len(efforts_df_full) + 1)

                                # ── FEATURE 13: Zona e Direção tática ────────────
                                _fl_tac = float(
                                    st.session_state.get('venue', {}).get('length') or 105)
                                efforts_df_full = enriquecer_esforcos_taticos(
                                    efforts_df_full, _coords_atletas, field_length=_fl_tac)

                            if not efforts_df_full.empty:
                                # Indicador de fonte de dados
                                if _esf_usou_api:
                                    st.caption(
                                        "✅ Esforços calculados pela **API Catapult** (algoritmos oficiais OpenField). "
                                        "Atletas sem dados de API usam cálculo local como fallback."
                                    )
                                else:
                                    st.caption(
                                        "⚙️ Esforços calculados **localmente** a partir dos dados do sensor "
                                        "(API Catapult não retornou esforços para estes atletas)."
                                    )

                                # Filtro de bandas
                                if 'Banda' in efforts_df_full.columns:
                                    bandas_disp = sorted(efforts_df_full['Banda'].dropna().unique())
                                    if bandas_disp:
                                        bandas_sel = st.multiselect(
                                            "Filtrar por bandas:", bandas_disp,
                                            default=bandas_disp, key="bandas_campo"
                                        )
                                        efforts_df_full = efforts_df_full[
                                            efforts_df_full['Banda'].isin(bandas_sel)
                                        ] if bandas_sel else efforts_df_full

                                if efforts_df_full.empty:
                                    st.info("Nenhum esforço encontrado após aplicar os filtros.")
                                else:
                                    # Colunas visíveis — "Atleta" primeiro, depois as demais
                                    _cols_ocultas = {c for c in efforts_df_full.columns
                                                     if c.startswith('_')}
                                    _cols_base = [c for c in efforts_df_full.columns
                                                  if c not in _cols_ocultas and c != 'Atleta']
                                    cols_show = (['Atleta'] + _cols_base
                                                 if 'Atleta' in efforts_df_full.columns
                                                 else _cols_base)
                                    efforts_df_show = efforts_df_full[cols_show]

                                    # Métricas resumidas
                                    mc1, mc2, mc3, mc4 = st.columns(4)
                                    with mc1: st.metric("Total de esforços", len(efforts_df_full))
                                    with mc2: st.metric("Duração total (s)", round(efforts_df_full['Duração (s)'].sum(), 1))
                                    with mc3:
                                        if 'Distância (m)' in efforts_df_full.columns:
                                            st.metric("Distância total (m)", round(efforts_df_full['Distância (m)'].sum(), 1))
                                    with mc4: st.metric("Média % máximo", round(efforts_df_full['% do Máximo'].mean(), 1))

                                    st.markdown(
                                        "**Selecione uma linha** para visualizar esse esforço no mapa de satélite acima. "
                                        "Clique novamente na linha selecionada para desfazer a seleção."
                                    )

                                    # Tabela interativa com seleção de linha
                                    evt = st.dataframe(
                                        efforts_df_show,
                                        use_container_width=True,
                                        height=360,
                                        on_select="rerun",
                                        selection_mode="single-row",
                                        key="tbl_esforcos_campo"
                                    )

                                    sel_rows = evt.selection.rows if evt.selection else []

                                    # Captura esforço selecionado + atleta correto
                                    if sel_rows:
                                        _ae_row = efforts_df_full.iloc[sel_rows[0]]
                                        _anim_start_ts   = float(_ae_row.get('_start_ts') or 0)
                                        _anim_end_ts     = float(_ae_row.get('_end_ts')   or 0)
                                        _anim_effort_row = _ae_row
                                        # Atleta dono do esforço selecionado
                                        _sel_atleta = str(_ae_row.get('Atleta', atleta_mapa))

                                    # Highlight GPS no mapa satélite (usa GPS do atleta correto)
                                    if sel_rows:
                                        sel_idx  = sel_rows[0]
                                        row_full = efforts_df_full.iloc[sel_idx]
                                        start_ts = row_full.get('_start_ts', 0)
                                        end_ts   = row_full.get('_end_ts', 0)
                                        _c_sel   = _coords_atletas.get(_sel_atleta,
                                                       _coords_atletas[atleta_mapa])
                                        _ts_gps_sel   = _c_sel['ts_gps']
                                        _lats_gps_sel = _c_sel['lats']
                                        _lons_gps_sel = _c_sel['lons']
                                        _vels_gps_sel = _c_sel['vels_gps']

                                        if start_ts and end_ts and \
                                                len(_ts_gps_sel) == len(_lats_gps_sel):
                                            filtered = [
                                                (la, lo, ve)
                                                for la, lo, ve, ts in zip(
                                                    _lats_gps_sel, _lons_gps_sel,
                                                    _vels_gps_sel, _ts_gps_sel)
                                                if start_ts <= ts <= end_ts
                                            ]
                                            if filtered:
                                                lats_eff = [p[0] for p in filtered]
                                                lons_eff = [p[1] for p in filtered]
                                                vels_eff = [p[2] for p in filtered]
                                                inicio_str = row_full.get('Início', '')
                                                dur_str    = row_full.get('Duração (s)', '')
                                                eff_desc   = (
                                                    f"Esforço #{row_full['Esforço']} "
                                                    f"({_sel_atleta}) — {inicio_str} — {dur_str}s"
                                                )
                                            else:
                                                st.warning("⚠️ Nenhum ponto GPS encontrado na janela de tempo deste esforço.")
                                        elif not _ts_gps_sel:
                                            st.info("ℹ️ Timestamps GPS não disponíveis.")

                                    # Download da tabela
                                    _atls_str = "_".join(a.replace(' ', '') for a in atletas_mapa)
                                    st.download_button(
                                        "📥 Exportar esforços",
                                        efforts_df_show.to_csv(index=False),
                                        file_name=f"esforcos_{_atls_str}_{_label_periodos.replace(' + ','_')}.csv"
                                    )
                            else:
                                st.info("ℹ️ Sem dados de posição (x/y) para calcular esforços. "
                                        "Os dados de sensor precisam incluir coordenadas de campo.")

                            # Renderiza o mapa fixo (com ou sem esforço destacado)
                            with mapa_placeholder:
                                if lats_gps and lons_gps:
                                    html_fixo = criar_html_campo_fixo(
                                        lats_gps, lons_gps, vels_gps, cfg,
                                        lats_eff=lats_eff or None,
                                        lons_eff=lons_eff or None,
                                        vels_eff=vels_eff or None,
                                        atleta_nome=atleta_mapa,
                                        esforco_desc=eff_desc,
                                        height=580
                                    )
                                    st.components.v1.html(html_fixo, height=580, scrolling=False)
                                else:
                                    st.warning("⚠️ Dados GPS não disponíveis para exibição do mapa.")

                            st.divider()

                            # ── ETAPA 3: Campo bonito + análise avançada ─────────
                            st.markdown("### 3️⃣ Análise de Movimentação no Campo")
                            # xn, yn, vel_raw_campo, etc. já foram construídos antes
                            # de ETAPA 2 para garantir consistência de índices.

                            # ── Computar coords do esforço ANTECIPADO ──────────────
                            # (usado no highlight do fig_campo E na animação abaixo)
                            # Usa arrays do atleta dono do esforço selecionado.
                            _c_eff  = _coords_atletas.get(_sel_atleta, _coords_atletas[atleta_mapa])
                            _xn_eff = _c_eff['xn'] or xn
                            _yn_eff = _c_eff['yn'] or yn
                            _vel_eff = _c_eff['vel'] or vel_raw_campo
                            _acc_eff = _c_eff['acc'] or acc_raw_campo
                            _ts_eff  = _c_eff['ts']  or ts_pos_campo

                            xs_a, ys_a, vel_a, acc_a = [], [], [], []
                            if _anim_effort_row is not None and _xn_eff:
                                # ── Tier 0: índices de segmento armazenados pelas
                                #    funções de sensor — indexam o array do atleta correto.
                                _seg_si = int(_anim_effort_row.get('_seg_start_idx', -1))
                                _seg_ei = int(_anim_effort_row.get('_seg_end_idx',   -1))
                                if 0 <= _seg_si < _seg_ei <= len(_xn_eff):
                                    xs_a  = _xn_eff[_seg_si:_seg_ei]
                                    ys_a  = _yn_eff[_seg_si:_seg_ei]
                                    vel_a = (_vel_eff[_seg_si:_seg_ei]
                                             if len(_vel_eff) == len(_xn_eff)
                                             else [0.0] * (_seg_ei - _seg_si))
                                    acc_a = (_acc_eff[_seg_si:_seg_ei]
                                             if len(_acc_eff) == len(_xn_eff)
                                             else [0.0] * (_seg_ei - _seg_si))

                                # ── Tier 1: matching por timestamp exato
                                # Tenta timestamps de campo primeiro; cai para GPS
                                # (campo x/y convertido de GPS → mesma indexação de ts_gps)
                                if len(xs_a) < 2 and _anim_start_ts > 0 and _anim_end_ts > 0:
                                    _ts1_src = []
                                    if _ts_eff and len(_ts_eff) == len(_xn_eff):
                                        _ts1_src = _ts_eff
                                    elif (_c_eff.get('ts_gps')
                                          and len(_c_eff['ts_gps']) == len(_xn_eff)):
                                        _ts1_src = _c_eff['ts_gps']
                                    if _ts1_src:
                                        _ts_c = np.array(_ts1_src, dtype=float)
                                        _m    = (_ts_c >= _anim_start_ts) & (_ts_c <= _anim_end_ts)
                                        if _m.any():
                                            xs_a  = np.array(_xn_eff)[_m].tolist()
                                            ys_a  = np.array(_yn_eff)[_m].tolist()
                                            vel_a = (np.array(_vel_eff)[_m].tolist()
                                                     if len(_vel_eff) == len(_xn_eff) else [0]*int(_m.sum()))
                                            acc_a = (np.array(_acc_eff)[_m].tolist()
                                                     if len(_acc_eff) == len(_xn_eff) else [0]*int(_m.sum()))

                                # ── Tier 2: fallback proporcional por timestamp
                                # Mesma lógica de source: campo → GPS
                                if len(xs_a) < 2 and _anim_start_ts > 0 and _anim_end_ts > 0:
                                    _ts2_src = (_ts_eff
                                                or _c_eff.get('ts_gps', []))
                                    if _ts2_src:
                                        _ts_min = min(_ts2_src)
                                        _ts_max = max(_ts2_src)
                                        if _ts_max > _ts_min:
                                            _sp = max(0.0, (_anim_start_ts - _ts_min) / (_ts_max - _ts_min))
                                            _ep = min(1.0, (_anim_end_ts   - _ts_min) / (_ts_max - _ts_min))
                                            _si = int(_sp * len(_xn_eff))
                                            _ei = min(len(_xn_eff), int(_ep * len(_xn_eff)) + 1)
                                            if _ei > _si + 1:
                                                xs_a  = _xn_eff[_si:_ei]
                                                ys_a  = _yn_eff[_si:_ei]
                                                vel_a = (_vel_eff[_si:_ei]
                                                         if len(_vel_eff) == len(_xn_eff) else [0]*(_ei-_si))
                                                acc_a = (_acc_eff[_si:_ei]
                                                         if len(_acc_eff) == len(_xn_eff) else [0]*(_ei-_si))

                                # ── Tier 3: fallback por string de início + duração
                                if len(xs_a) < 2:
                                    try:
                                        _dur_s   = float(_anim_effort_row.get('Duração (s)', 5))
                                        _ini_s   = 0.0
                                        _ini_str = str(_anim_effort_row.get('Início', ''))
                                        if ':' in _ini_str:
                                            _parts = _ini_str.replace('s', '').split(':')
                                            if len(_parts) == 3:
                                                _ini_s = int(_parts[0])*3600 + int(_parts[1])*60 + float(_parts[2])
                                            else:
                                                _ini_s = int(_parts[0])*60 + float(_parts[1])
                                        elif _ini_str.endswith('s'):
                                            _ini_s = float(_ini_str[:-1])
                                        _total_s = len(_xn_eff) / 10.0
                                        if _total_s > 0:
                                            _sp = max(0.0, min(1.0, _ini_s / _total_s))
                                            _ep = max(0.0, min(1.0, (_ini_s + _dur_s) / _total_s))
                                            _si = int(_sp * len(_xn_eff))
                                            _ei = min(len(_xn_eff), int(_ep * len(_xn_eff)) + 1)
                                            if _ei > _si + 1:
                                                xs_a  = _xn_eff[_si:_ei]
                                                ys_a  = _yn_eff[_si:_ei]
                                                vel_a = (_vel_eff[_si:_ei]
                                                         if len(_vel_eff) == len(_xn_eff)
                                                         else [0]*(_ei-_si))
                                                acc_a = (_acc_eff[_si:_ei]
                                                         if len(_acc_eff) == len(_xn_eff)
                                                         else [0]*(_ei-_si))
                                    except Exception:
                                        _applog.log_debug_exc()

                            if _fonte_xy:
                                st.caption("📡 Coordenadas x/y Catapult OpenField")
                            elif not _fonte_xy and _has_gps:
                                st.caption("🌍 Coordenadas derivadas do GPS + campo aplicado.")

                            vel_raw = vel_raw_campo
                            acc_raw = acc_raw_campo

                            if _has_xy or _has_gps:

                                if len(xn) > 0:
                                    # ── Linha 1: modo de visualização ────────────────
                                    col_modo, col_ov = st.columns([2, 3])
                                    with col_modo:
                                        modo_viz = st.radio(
                                            "🎨 Modo de visualização",
                                            ["🗺️ Trajetória",
                                             "⚡ Bandas de Velocidade",
                                             "🔁 Bandas de Aceleração"],
                                            key="modo_campo_v3"
                                        )
                                    with col_ov:
                                        st.markdown("**Overlays opcionais:**")
                                        oa, ob = st.columns(2)
                                        with oa:
                                            ov_setas   = st.checkbox("🏹 Setas de direção",    key="ov_setas")
                                            ov_hull    = st.checkbox("📐 Área de atuação",     key="ov_hull")
                                            ov_eventos = st.checkbox("⚽ Eventos Futebol",      key="ov_eventos")
                                        with ob:
                                            ov_tercos  = st.checkbox("📊 Terços do campo",     key="ov_tercos")
                                            ov_grade   = st.checkbox("🔲 Grade de quadrantes", key="ov_grade")
                                            ov_heatmap_comp = st.checkbox("🔥 Heatmap por Período",  key="ov_hcmp")
                                            ov_voronoi_comp = st.checkbox("🔷 Voronoi por Período",  key="ov_vcmp")

                                    # ── Direção de ataque por período ─────────────────
                                    _atk_key = f"attack_dir_{periodo_mapa}"
                                    _atk_opts = [
                                        "— Não definido",
                                        "➡️  Esq → Dir  (ataque para a direita)",
                                        "⬅️  Dir → Esq  (ataque para a esquerda)",
                                    ]
                                    _atk_default_idx = 0
                                    if st.session_state.get(_atk_key) == 'left_to_right':
                                        _atk_default_idx = 1
                                    elif st.session_state.get(_atk_key) == 'right_to_left':
                                        _atk_default_idx = 2
                                    _atk_sel = st.radio(
                                        f"⚽ Direção de ataque — {periodo_mapa}:",
                                        options=_atk_opts,
                                        index=_atk_default_idx,
                                        horizontal=True,
                                        key=f"_atk_radio_{periodo_mapa}",
                                    )
                                    if "➡️" in _atk_sel:
                                        st.session_state[_atk_key] = 'left_to_right'
                                    elif "⬅️" in _atk_sel:
                                        st.session_state[_atk_key] = 'right_to_left'
                                    else:
                                        st.session_state[_atk_key] = None
                                    _attack_dir_val = st.session_state[_atk_key]

                                    # ── Seletores de bandas (dependem do modo) ────────
                                    _bv_ui = _bandas_vel_ativas()
                                    _ba_ui = _bandas_acc_ativas()
                                    bandas_vel_sel = list(_bv_ui.keys())
                                    bandas_acc_sel = list(_ba_ui.keys())

                                    if modo_viz == "⚡ Bandas de Velocidade":
                                        bandas_vel_sel = st.multiselect(
                                            "Bandas de velocidade a exibir:",
                                            options=list(_bv_ui.keys()),
                                            default=list(_bv_ui.keys()),
                                            format_func=lambda k: _bv_ui[k]['label'],
                                            key="ms_bv"
                                        )
                                    elif modo_viz == "🔁 Bandas de Aceleração":
                                        # Duas caixas separadas: Aceleração (A*) e Desaceleração (D*).
                                        _acc_k = [k for k in _ba_ui if str(k).startswith('A')]
                                        _dec_k = [k for k in _ba_ui if str(k).startswith('D')]
                                        _cba, _cbd = st.columns(2)
                                        with _cba:
                                            _sel_a = st.multiselect(
                                                "🚀 Aceleração:",
                                                options=_acc_k,
                                                default=_acc_k,
                                                format_func=lambda k: _ba_ui[k]['label'],
                                                key="ms_ba_pos"
                                            )
                                        with _cbd:
                                            _sel_d = st.multiselect(
                                                "🛑 Desaceleração:",
                                                options=_dec_k,
                                                default=_dec_k,
                                                format_func=lambda k: _ba_ui[k]['label'],
                                                key="ms_ba_neg"
                                            )
                                        bandas_acc_sel = list(_sel_a) + list(_sel_d)

                                    # ── Configuração da grade ─────────────────────────
                                    # ── Seletor de eventos para o campo ──────────────
                                    _dados_ev_campo = {}
                                    _ev_tipos_sel   = []
                                    if ov_eventos:
                                        # Combina eventos de todos os períodos selecionados
                                        _ev_raw = {}
                                        for _pm in periodos_mapa_sel:
                                            _ev_pm = dados_eventos_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                                            for _et, _evlist in _ev_pm.items():
                                                _ev_raw.setdefault(_et, [])
                                                _ev_raw[_et] += _evlist
                                        if _ev_raw:
                                            # Enriquecer com posição no campo agora que temos cfg
                                            _ev_rich = enriquecer_eventos_com_posicao(
                                                _ev_raw,
                                                dados.get('ts_gps', []),
                                                dados.get('lats', []),
                                                dados.get('lons', []),
                                                dados.get('vels_gps', []),
                                                campo_config=cfg,
                                            )
                                            _ev_tipos_disp = [
                                                k for k in _ev_rich if _ev_rich[k]]
                                            if _ev_tipos_disp:
                                                _ev_tipos_sel = st.multiselect(
                                                    "Tipos de evento no campo:",
                                                    options=_ev_tipos_disp,
                                                    default=_ev_tipos_disp,
                                                    format_func=lambda k: FUTEBOL_EVENTS_CONFIG[k]['label'],
                                                    key="ev_campo_tipos"
                                                )
                                                _dados_ev_campo = _ev_rich
                                            else:
                                                st.info("Nenhum evento de futebol carregado para este atleta/período.")
                                        else:
                                            st.info("Nenhum evento de futebol carregado. Recarregue os dados com eventos ativados na sidebar.")

                                    n_cols_g, n_rows_g, zona_sel = 4, 3, None
                                    if ov_grade:
                                        gc1, gc2, gc3 = st.columns(3)
                                        with gc1:
                                            n_cols_g = st.slider("Colunas", 2, 10, 4, key="g_cols")
                                        with gc2:
                                            n_rows_g = st.slider("Linhas",  2, 8,  3, key="g_rows")
                                        with gc3:
                                            zonas_disp = ["— Nenhuma —"] + [
                                                f"{chr(65+r)}{c+1}"
                                                for r in range(n_rows_g)
                                                for c in range(n_cols_g)
                                            ]
                                            zona_sel = st.selectbox(
                                                "🔍 Detalhar zona:", zonas_disp, key="zona_det")
                                            if zona_sel == "— Nenhuma —":
                                                zona_sel = None

                                    # ── Construir figura ──────────────────────────────
                                    _titulo_campo = (
                                        f"📍 {' + '.join(atletas_mapa)} — {_label_periodos}"
                                        if len(atletas_mapa) > 1
                                        else f"📍 {atleta_mapa} — {_label_periodos}"
                                    )
                                    fig_campo = desenhar_campo_futebol_bonito(
                                        title=_titulo_campo,
                                        attack_direction=_attack_dir_val,
                                    )

                                    # ── Plota TODOS os atletas com o mesmo modo ───────
                                    # Usa _coords_atletas para garantir que cada atleta
                                    # receba exatamente o mesmo tratamento visual.
                                    # Prefixo do nome = atleta (visível na legenda).
                                    _n_atls = len(atletas_mapa)
                                    for _atl_i, _atl_viz in enumerate(atletas_mapa):
                                        _cv = _coords_atletas.get(_atl_viz, {})
                                        _xv = _cv.get('xn', [])
                                        _yv = _cv.get('yn', [])
                                        _velv = _cv.get('vel', [])
                                        _accv = _cv.get('acc', [])
                                        if not _xv:
                                            continue
                                        # Prefixo apenas quando há mais de 1 atleta
                                        _pfx = _atl_viz if _n_atls > 1 else ''
                                        if modo_viz == "🗺️ Trajetória":
                                            adicionar_trajetoria_campo(
                                                fig_campo, _xv, _yv, _velv, _atl_viz)
                                        elif modo_viz == "⚡ Bandas de Velocidade" and bandas_vel_sel:
                                            adicionar_pontos_velocidade_bandas(
                                                fig_campo, _xv, _yv, _velv, bandas_vel_sel,
                                                mostrar_setas=ov_setas,
                                                atleta_prefix=_pfx)
                                        elif modo_viz == "🔁 Bandas de Aceleração" and bandas_acc_sel:
                                            adicionar_pontos_aceleracao_bandas(
                                                fig_campo, _xv, _yv, _accv, bandas_acc_sel,
                                                mostrar_setas=ov_setas,
                                                atleta_prefix=_pfx)
                                        # Seta de trajetória por atleta
                                        if ov_setas and modo_viz == "🗺️ Trajetória":
                                            adicionar_setas_direcao(fig_campo, _xv, _yv)

                                    # Seta do esforço destacado (laranja, sobre tudo)
                                    if ov_setas and xs_a and len(xs_a) >= 2:
                                        adicionar_setas_direcao(
                                            fig_campo, xn, yn,
                                            xs_effort=xs_a, ys_effort=ys_a,
                                        )
                                    if ov_hull:
                                        adicionar_convex_hull(fig_campo, xn, yn)
                                    if ov_tercos:
                                        adicionar_tercos_campo(fig_campo, xn, yn)
                                    if ov_grade:
                                        adicionar_grade_quadrantes(
                                            fig_campo, xn, yn, n_cols_g, n_rows_g)
                                    if ov_eventos and _dados_ev_campo and _ev_tipos_sel:
                                        adicionar_eventos_campo(
                                            fig_campo, _dados_ev_campo, _ev_tipos_sel)

                                    # ── Highlight do esforço selecionado ─────────────
                                    if xs_a and len(xs_a) >= 2:
                                        # Halo laranja pulsante (linha mais grossa por baixo)
                                        fig_campo.add_trace(go.Scatter(
                                            x=xs_a, y=ys_a, mode='lines',
                                            line=dict(color='rgba(255,152,0,0.35)', width=14),
                                            name='_halo', showlegend=False, hoverinfo='skip'
                                        ))
                                        # Trajetória do esforço em laranja sólido
                                        fig_campo.add_trace(go.Scatter(
                                            x=xs_a, y=ys_a, mode='lines+markers',
                                            line=dict(color='#FF9800', width=4),
                                            marker=dict(size=3, color='#FF9800'),
                                            name=f"Esforço #{int(_anim_effort_row['Esforço'])}",
                                            showlegend=True, hoverinfo='skip'
                                        ))
                                        # Marcador de início (verde)
                                        fig_campo.add_trace(go.Scatter(
                                            x=[xs_a[0]], y=[ys_a[0]], mode='markers+text',
                                            marker=dict(size=14, color='#4CAF50', symbol='circle',
                                                        line=dict(color='white', width=2)),
                                            text=['▶'], textposition='top center',
                                            textfont=dict(color='white', size=10),
                                            name='Início', showlegend=False, hoverinfo='skip'
                                        ))
                                        # Marcador de fim (vermelho)
                                        fig_campo.add_trace(go.Scatter(
                                            x=[xs_a[-1]], y=[ys_a[-1]], mode='markers+text',
                                            marker=dict(size=14, color='#F44336', symbol='x',
                                                        line=dict(color='white', width=2)),
                                            text=['■'], textposition='top center',
                                            textfont=dict(color='white', size=10),
                                            name='Fim', showlegend=False, hoverinfo='skip'
                                        ))
                                        fig_campo.update_layout(
                                            title=dict(
                                                text=(
                                                    f"📍 {atleta_mapa} — {_label_periodos} &nbsp;|&nbsp; "
                                                    f"🟠 Esforço #{int(_anim_effort_row['Esforço'])} destacado"
                                                )
                                            )
                                        )

                                    st.plotly_chart(fig_campo, use_container_width=True)

                                    # ── FEATURE 12: Heatmap comparativo por período ──
                                    if ov_heatmap_comp and len(periodos_mapa_sel) >= 1:
                                        st.markdown("#### 🔥 Heatmap de Posicionamento por Período")
                                        _fl_hm = float(st.session_state.get('venue', {}).get('length') or 105)
                                        _fw_hm = float(st.session_state.get('venue', {}).get('width')  or 68)
                                        _hm_cols = st.columns(min(len(periodos_mapa_sel), 3))
                                        for _hm_i, _hm_pm in enumerate(periodos_mapa_sel[:3]):
                                            with _hm_cols[_hm_i]:
                                                _hm_xs, _hm_ys = [], []
                                                for _hm_atl in atletas_mapa:
                                                    _hm_d = dados_posicao_por_periodo.get(_hm_pm, {}).get(_hm_atl, {})
                                                    _hm_xs += _hm_d.get('xs', [])
                                                    _hm_ys += _hm_d.get('ys', [])
                                                if _hm_xs:
                                                    _fig_hm = desenhar_campo_futebol_bonito(
                                                        _fl_hm, _fw_hm, title=_hm_pm)
                                                    _xhm = np.array(_hm_xs)
                                                    _yhm = np.array(_hm_ys)
                                                    _H, _xe, _ye = np.histogram2d(
                                                        _xhm, _yhm, bins=[52, 34],
                                                        range=[[0, _fl_hm], [0, _fw_hm]])
                                                    _H = _gf(_H, sigma=2.5)
                                                    _xc = (_xe[:-1] + _xe[1:]) / 2
                                                    _yc = (_ye[:-1] + _ye[1:]) / 2
                                                    _fig_hm.add_trace(go.Heatmap(
                                                        x=_xc, y=_yc, z=_H.T,
                                                        colorscale='Hot', opacity=0.60,
                                                        showscale=False, hoverinfo='skip'))
                                                    _fig_hm.update_layout(
                                                        height=320,
                                                        margin=dict(l=5, r=5, t=35, b=5))
                                                    st.plotly_chart(_fig_hm, use_container_width=True)
                                                else:
                                                    st.info(f"Sem dados para {_hm_pm}")

                                    # ── FEATURE 14: Voronoi comparativo por período ──
                                    if ov_voronoi_comp and len(periodos_mapa_sel) >= 1:
                                        st.markdown("#### 🔷 Voronoi — Raio de Ação por Período")
                                        _vc_cols = st.columns(min(len(periodos_mapa_sel), 3))
                                        for _vc_i, _vc_pm in enumerate(periodos_mapa_sel[:3]):
                                            with _vc_cols[_vc_i]:
                                                _vc_pos = {}
                                                for _vc_atl in atletas_mapa:
                                                    _vc_d = dados_posicao_por_periodo.get(_vc_pm, {}).get(_vc_atl, {})
                                                    if _vc_d.get('xs'):
                                                        _vc_pos[_vc_atl] = {'xs': _vc_d['xs'], 'ys': _vc_d['ys']}
                                                if len(_vc_pos) >= 2:
                                                    _fig_vc = calcular_voronoi_campo(_vc_pos)
                                                    if _fig_vc:
                                                        _fig_vc.update_layout(
                                                            title=dict(text=_vc_pm, font=dict(size=13)),
                                                            height=320,
                                                            margin=dict(l=5, r=5, t=35, b=5),
                                                            showlegend=False)
                                                        st.plotly_chart(_fig_vc, use_container_width=True)
                                                    else:
                                                        st.info(f"Sem dados Voronoi para {_vc_pm}")
                                                else:
                                                    st.info(f"Voronoi precisa de ≥2 atletas com dados em {_vc_pm}")

                                    # ══════════════════════════════════════════════
                                    # ANIMAÇÃO DO ESFORÇO SELECIONADO
                                    # ══════════════════════════════════════════════
                                    if _anim_effort_row is not None:
                                        st.markdown("---")
                                        _vel_max_hdr = _anim_effort_row.get(
                                            'Vel. Máx (km/h)',
                                            _anim_effort_row.get('Acc. Máx (m/s²)', '—')
                                        )
                                        _vel_unit_hdr = (
                                            'km/h' if 'Vel. Máx (km/h)' in _anim_effort_row.index
                                            else 'm/s²'
                                        )
                                        st.markdown(
                                            f"### 🎬 Animação — Esforço **#{int(_anim_effort_row['Esforço'])}** &nbsp;|&nbsp; "
                                            f"Início: **{_anim_effort_row['Início']}** &nbsp;|&nbsp; "
                                            f"Duração: **{_anim_effort_row['Duração (s)']}s** &nbsp;|&nbsp; "
                                            f"Vel. Máx: **{_vel_max_hdr} {_vel_unit_hdr}**"
                                        )

                                        # xs_a / ys_a já foram calculados acima com fallbacks
                                        def _vc_anim(v):
                                            if v < 7:  return '#2196F3'
                                            if v < 14: return '#4CAF50'
                                            if v < 19: return '#FFEB3B'
                                            if v < 24: return '#FF9800'
                                            return '#F44336'

                                        if len(xs_a) >= 2:
                                            _n_pts = len(xs_a)
                                            _step_a = max(1, _n_pts // 80)  # máx 80 frames

                                            # ── Slider de velocidade ──────────────────
                                            _vel_opcoes = {
                                                "🐢 0.25×": 320,
                                                "🚶 0.5×":  160,
                                                "🏃 1× (real)": 80,
                                                "⚡ 2×":    40,
                                                "🚀 4×":    20,
                                            }
                                            _vel_sel = st.select_slider(
                                                "⏩ Velocidade da animação",
                                                options=list(_vel_opcoes.keys()),
                                                value="🏃 1× (real)",
                                                key="anim_speed_slider"
                                            )
                                            _frame_dur = _vel_opcoes[_vel_sel]

                                            # ── Campo base (estático) ─────────────────
                                            _fig_a = desenhar_campo_futebol_bonito(
                                                title=(
                                                    f"Esforço #{int(_anim_effort_row['Esforço'])} | "
                                                    f"{_anim_effort_row['Início']} | "
                                                    f"Vel. Máx {_anim_effort_row['Vel. Máx (km/h)']} km/h"
                                                )
                                            )
                                            _n_base_a = len(_fig_a.data)

                                            # Trajetória completa (fundo desbotado)
                                            _sbg = max(1, len(xn) // 5000)
                                            _fig_a.add_trace(go.Scatter(
                                                x=xn[::_sbg], y=yn[::_sbg], mode='markers',
                                                marker=dict(size=1.5, color='rgba(255,255,255,0.10)'),
                                                name='Trajetória', showlegend=False, hoverinfo='skip'
                                            ))
                                            # Marcadores início / fim do esforço (estáticos)
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[0]], y=[ys_a[0]], mode='markers+text',
                                                marker=dict(size=13, color='#4CAF50', symbol='circle',
                                                            line=dict(color='white', width=2)),
                                                text=['▶'], textposition='top center',
                                                textfont=dict(color='#4CAF50', size=11),
                                                name='Início', showlegend=False, hoverinfo='skip'
                                            ))
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[-1]], y=[ys_a[-1]], mode='markers+text',
                                                marker=dict(size=13, color='#F44336', symbol='x',
                                                            line=dict(color='white', width=2)),
                                                text=['■'], textposition='top center',
                                                textfont=dict(color='#F44336', size=11),
                                                name='Fim', showlegend=False, hoverinfo='skip'
                                            ))
                                            # Traço do esforço (animado)
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[0]], y=[ys_a[0]], mode='lines',
                                                line=dict(color='#FF9800', width=4),
                                                name='Esforço', showlegend=False
                                            ))
                                            # Marcador de posição atual (animado)
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[0]], y=[ys_a[0]], mode='markers',
                                                marker=dict(
                                                    size=18, color=_vc_anim(vel_a[0]),
                                                    symbol='circle',
                                                    line=dict(color='white', width=3)
                                                ),
                                                name='Posição', showlegend=False
                                            ))

                                            _idx_trace  = _n_base_a + 3  # traço do esforço
                                            _idx_dot    = _n_base_a + 4  # marcador posição

                                            # ── Frames de animação ────────────────────
                                            _frames_a = []
                                            for _k in range(0, _n_pts, _step_a):
                                                _t  = _k * 0.1
                                                _v  = vel_a[_k] if _k < len(vel_a) else 0
                                                _ac = acc_a[_k] if _k < len(acc_a) else 0
                                                _col = _vc_anim(_v)
                                                _frames_a.append(go.Frame(
                                                    data=[
                                                        go.Scatter(
                                                            x=xs_a[:_k+1], y=ys_a[:_k+1],
                                                            mode='lines',
                                                            line=dict(color=_col, width=4),
                                                        ),
                                                        go.Scatter(
                                                            x=[xs_a[_k]], y=[ys_a[_k]],
                                                            mode='markers',
                                                            marker=dict(
                                                                size=18, color=_col, symbol='circle',
                                                                line=dict(color='white', width=3)
                                                            ),
                                                        ),
                                                    ],
                                                    traces=[_idx_trace, _idx_dot],
                                                    name=str(_k),
                                                    layout=go.Layout(title=dict(
                                                        text=(
                                                            f"⏱️ {_t:.1f}s / {_anim_effort_row['Duração (s)']}s"
                                                            f"   |   💨 {_v:.1f} km/h"
                                                            f"   |   ⚡ {_ac:+.2f} m/s²"
                                                            f"   |   📍 x={xs_a[_k]:.1f}m y={ys_a[_k]:.1f}m"
                                                        ),
                                                        font=dict(color='white', size=12)
                                                    ))
                                                ))
                                            _fig_a.frames = _frames_a

                                            # ── Controles Play/Pause + Slider ─────────
                                            _slider_steps = [
                                                {
                                                    'args': [[f.name],
                                                             {'frame': {'duration': 0, 'redraw': True},
                                                              'mode': 'immediate',
                                                              'transition': {'duration': 0}}],
                                                    'label': f"{int(f.name)*0.1:.0f}s",
                                                    'method': 'animate',
                                                }
                                                for f in _frames_a[::max(1, len(_frames_a)//15)]
                                            ]
                                            _fig_a.update_layout(
                                                height=560,
                                                margin=dict(t=55, b=110, l=10, r=10),
                                                updatemenus=[{
                                                    'buttons': [
                                                        {
                                                            'args': [None, {
                                                                'frame': {'duration': _frame_dur, 'redraw': True},
                                                                'fromcurrent': True,
                                                                'transition': {'duration': _frame_dur, 'easing': 'linear'},
                                                            }],
                                                            'label': '▶  Play',
                                                            'method': 'animate',
                                                        },
                                                        {
                                                            'args': [[None], {
                                                                'frame': {'duration': 0, 'redraw': False},
                                                                'mode': 'immediate',
                                                                'transition': {'duration': 0},
                                                            }],
                                                            'label': '⏸  Pause',
                                                            'method': 'animate',
                                                        },
                                                    ],
                                                    'direction': 'left',
                                                    'pad': {'r': 10, 't': 10},
                                                    'showactive': True,
                                                    'type': 'buttons',
                                                    'x': 0.0, 'y': -0.07,
                                                    'bgcolor': '#1565C0',
                                                    'font': {'color': 'white', 'size': 13},
                                                    'bordercolor': '#0D47A1',
                                                }],
                                                sliders=[{
                                                    'active': 0,
                                                    'steps': _slider_steps,
                                                    'currentvalue': {
                                                        'prefix': '⏱️ ',
                                                        'visible': True,
                                                        'font': {'color': 'white', 'size': 12},
                                                    },
                                                    'tickcolor': 'white',
                                                    'font': {'color': 'white'},
                                                    'y': -0.02,
                                                    'len': 0.88,
                                                    'x': 0.1,
                                                    'bgcolor': '#1a1a2e',
                                                    'bordercolor': '#555',
                                                }],
                                            )
                                            st.plotly_chart(_fig_a, use_container_width=True)

                                            # Painel de info abaixo
                                            _ai1, _ai2, _ai3, _ai4, _ai5 = st.columns(5)
                                            _ai1.metric("⏱️ Duração",       f"{_anim_effort_row['Duração (s)']}s")
                                            _ai2.metric("💨 Vel. Máx",      f"{_anim_effort_row['Vel. Máx (km/h)']} km/h")
                                            _ai3.metric("🏁 Vel. Inicial",  f"{_anim_effort_row['Vel. Inicial (km/h)']} km/h")
                                            _ai4.metric("📏 Distância",     f"{_anim_effort_row['Distância (m)']} m")
                                            _ai5.metric("📊 % do Máximo",   f"{_anim_effort_row['% do Máximo']}%")

                                            # ──────────────────────────────────────────────
                                            # PERFIL DE SPRINT — FASES E EFICIÊNCIA (item 6)
                                            # ──────────────────────────────────────────────
                                            st.markdown("---")
                                            st.markdown("#### 🏃 Perfil do Sprint — Fases e Eficiência")
                                            st.caption(
                                                "Decomposição em 3 fases: 🟠 Aceleração (derivada >+0.5 m/s²) · "
                                                "🔴 Pico (≥95% da vel. máxima) · 🔵 Desaceleração (derivada <−0.5 m/s²)"
                                            )

                                            _sp_vel = np.array(vel_a, dtype=float)
                                            _sp_acc = np.array(acc_a, dtype=float)
                                            _sp_t   = np.arange(len(_sp_vel)) * 0.1

                                            # Suavização para detecção de fases
                                            _sp_wl = min(11, len(_sp_vel) - (1 - len(_sp_vel) % 2))
                                            if len(_sp_vel) >= 5:
                                                from scipy.signal import savgol_filter as _svgf
                                                _sp_sm = np.clip(_svgf(_sp_vel, max(5, _sp_wl), 2), 0, None)
                                            else:
                                                _sp_sm = np.clip(_sp_vel, 0, None)

                                            _sp_grad = np.gradient(_sp_sm, 0.1)
                                            _sp_vmax = float(_sp_sm.max()) if _sp_sm.max() > 0 else 1.0

                                            _sp_phases = np.where(
                                                _sp_sm >= _sp_vmax * 0.95, 2,           # pico
                                                np.where(_sp_grad >= 0.5, 1,            # aceleração
                                                np.where(_sp_grad <= -0.5, 3, 2))       # desaceleração / pico
                                            )

                                            _ph_clrs = {1: '#FFA726', 2: '#EF5350', 3: '#42A5F5'}
                                            _ph_lbls = {1: 'Aceleração', 2: 'Pico', 3: 'Desaceleração'}

                                            _fig_sp = go.Figure()
                                            # Linha de velocidade (fundo)
                                            _fig_sp.add_trace(go.Scatter(
                                                x=_sp_t, y=_sp_sm, mode='lines',
                                                line=dict(color='rgba(255,255,255,0.25)', width=1.5),
                                                showlegend=False, hoverinfo='skip'
                                            ))
                                            # Pontos coloridos por fase
                                            for _ph in [1, 2, 3]:
                                                _msk_ph = _sp_phases == _ph
                                                if _msk_ph.any():
                                                    _fig_sp.add_trace(go.Scatter(
                                                        x=_sp_t[_msk_ph],
                                                        y=_sp_sm[_msk_ph],
                                                        mode='markers',
                                                        name=_ph_lbls[_ph],
                                                        marker=dict(size=6,
                                                                    color=_ph_clrs[_ph],
                                                                    line=dict(width=0)),
                                                    ))
                                            # Linha de vel. máxima
                                            _fig_sp.add_hline(
                                                y=_sp_vmax, line_dash='dot',
                                                line_color='rgba(255,255,255,0.5)', line_width=1,
                                                annotation_text=f"Vmáx {_sp_vmax:.1f} km/h",
                                                annotation_font_color='white',
                                                annotation_font_size=10,
                                            )
                                            _fig_sp.update_layout(
                                                xaxis=dict(title='Tempo (s)', color='white',
                                                           gridcolor='rgba(255,255,255,0.1)'),
                                                yaxis=dict(title='Velocidade (km/h)', color='white',
                                                           gridcolor='rgba(255,255,255,0.1)'),
                                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                                font=dict(color='white'), height=270,
                                                legend=dict(font=dict(color='white'),
                                                            orientation='h', y=1.10),
                                                margin=dict(t=10, b=40, l=55, r=10),
                                            )
                                            st.plotly_chart(_fig_sp, use_container_width=True)

                                            # Métricas por fase
                                            _sp_mc = st.columns(3)
                                            _ph_icons = {1: '🟠', 2: '🔴', 3: '🔵'}
                                            for _phi in [1, 2, 3]:
                                                _msk_ph = _sp_phases == _phi
                                                _ph_dur  = float(_msk_ph.sum()) * 0.1
                                                _ph_vavg = (float(_sp_sm[_msk_ph].mean())
                                                            if _msk_ph.any() else 0.0)
                                                _ph_dist = (float((_sp_sm[_msk_ph] / 3.6 / 10).sum())
                                                            if _msk_ph.any() else 0.0)
                                                _ph_gacc = (float(_sp_acc[_msk_ph].mean())
                                                            if _msk_ph.any() and len(_sp_acc) == len(_sp_vel)
                                                            else 0.0)
                                                with _sp_mc[_phi - 1]:
                                                    st.markdown(
                                                        f"**{_ph_icons[_phi]} {_ph_lbls[_phi]}**"
                                                    )
                                                    st.metric("Duração", f"{_ph_dur:.1f} s")
                                                    st.metric("Vel. Média", f"{_ph_vavg:.1f} km/h")
                                                    st.metric("Distância", f"{_ph_dist:.1f} m")
                                                    if abs(_ph_gacc) > 0.01:
                                                        st.metric(
                                                            "Acc. Média",
                                                            f"{_ph_gacc:+.2f} m/s²"
                                                        )
                                        else:
                                            st.warning(
                                                "⚠️ Não foi possível localizar os pontos de campo para este esforço. "
                                                "Verifique se o campo está configurado corretamente na aba **🗺️ Campo de Futebol** "
                                                "e se os timestamps do esforço correspondem aos dados de posição carregados."
                                            )
                                    else:
                                        st.info("💡 Selecione um esforço na tabela acima para ver a **animação no campo**.")

                                    # ── Detalhes de zona selecionada ──────────────────
                                    if zona_sel and ov_grade:
                                        r_idx = ord(zona_sel[0]) - 65
                                        c_idx = int(zona_sel[1:]) - 1
                                        cw_g  = 105.0 / n_cols_g
                                        rh_g  = 68.0  / n_rows_g
                                        st_z  = stats_quadrante(
                                            xn, yn, vel_raw, acc_raw,
                                            c_idx*cw_g, (c_idx+1)*cw_g,
                                            r_idx*rh_g, (r_idx+1)*rh_g)
                                        st.markdown(f"#### 🔍 Zona **{zona_sel}** — Detalhes")
                                        z1,z2,z3,z4,z5,z6 = st.columns(6)
                                        with z1: st.metric("% do Tempo",     f"{st_z['pct']:.1f}%")
                                        with z2: st.metric("Pontos",          st_z['n_pontos'])
                                        with z3: st.metric("Vel. Média",     f"{st_z['vel_media']:.1f} km/h")
                                        with z4: st.metric("Vel. Máx",       f"{st_z['vel_max']:.1f} km/h")
                                        with z5: st.metric("Acc. Média",     f"{st_z['acc_media']:.2f} m/s²")
                                        with z6: st.metric("|Acc| Máx",      f"{st_z['acc_max']:.2f} m/s²")

                                    # ── Estatísticas gerais ───────────────────────────
                                    st.markdown("#### 📊 Estatísticas de Movimentação")
                                    sc1, sc2, sc3, sc4 = st.columns(4)
                                    with sc1:
                                        dist_km = sum(vel_raw) * 0.1 / 3600
                                        st.metric("Distância Total", f"{dist_km:.2f} km")
                                    with sc2:
                                        xr = max(xn) - min(xn)
                                        st.metric("Largura Atuação", f"{xr:.0f} m")
                                    with sc3:
                                        yr = max(yn) - min(yn)
                                        st.metric("Profundidade", f"{yr:.0f} m")
                                    with sc4:
                                        try:
                                            from scipy.spatial import ConvexHull as _CH
                                            _area = _CH(np.column_stack([xn, yn])).volume
                                            st.metric("Área de Atuação", f"{_area:.0f} m²")
                                        except Exception:
                                            st.metric("Área (bbox)", f"{xr*yr:.0f} m²")

                                    # ── Comparação (Períodos ou Atletas) ─────────────
                                    st.markdown("---")
                                    st.markdown("#### 🔄 Comparação")
                                    _cmp_modo = st.radio(
                                        "Comparar:",
                                        ["📅 Períodos", "👤 Atletas"],
                                        horizontal=True, key="cmp_modo"
                                    )

                                    if _cmp_modo == "📅 Períodos":
                                        periodos_lista = list(dados_posicao_por_periodo.keys())
                                        if len(periodos_lista) < 2:
                                            st.info("Carregue pelo menos 2 períodos para usar esta comparação.")
                                        else:
                                            cp1, cp2 = st.columns(2)
                                            with cp1:
                                                per1 = st.selectbox("Período A:", periodos_lista,
                                                                     index=0, key="cmp_p1")
                                            with cp2:
                                                per2 = st.selectbox("Período B:", periodos_lista,
                                                                     index=min(1, len(periodos_lista)-1),
                                                                     key="cmp_p2")

                                            if per1 != per2:
                                                d1c = dados_posicao_por_periodo.get(per1, {}).get(atleta_mapa, {})
                                                d2c = dados_posicao_por_periodo.get(per2, {}).get(atleta_mapa, {})
                                                cfg_ref = st.session_state.get(f"campo_cfg__{atleta_mapa}", cfg)
                                                if d1c.get('xs') and d2c.get('xs'):
                                                    x1c, y1c = list(d1c['xs']), list(d1c['ys'])
                                                    x2c, y2c = list(d2c['xs']), list(d2c['ys'])
                                                elif d1c.get('lats') and d2c.get('lats') and cfg_ref:
                                                    x1c, y1c = gps_para_campo_coords(d1c['lats'], d1c['lons'], cfg_ref)
                                                    x2c, y2c = gps_para_campo_coords(d2c['lats'], d2c['lons'], cfg_ref)
                                                else:
                                                    x1c = x2c = []
                                                if x1c and x2c:
                                                    fig_cmp = desenhar_campo_futebol_bonito(
                                                        title=f"Comparação: {per1} (azul) vs {per2} (rosa)")
                                                    _s1 = max(1, len(x1c)//3000)
                                                    _s2 = max(1, len(x2c)//3000)
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=x1c[::_s1], y=y1c[::_s1], mode='markers', name=per1,
                                                        marker=dict(size=2, color='#00E5FF', opacity=0.5)))
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=x2c[::_s2], y=y2c[::_s2], mode='markers', name=per2,
                                                        marker=dict(size=2, color='#FF4081', opacity=0.5)))
                                                    st.plotly_chart(fig_cmp, use_container_width=True)
                                                else:
                                                    st.info("Dados de posição não disponíveis em um dos períodos.")
                                            else:
                                                st.info("Selecione dois períodos diferentes para comparar.")

                                    else:  # Comparar Atletas
                                        _ats_cmp = _ats_disponiveis if _ats_disponiveis else [atleta_mapa]
                                        if len(_ats_cmp) < 2:
                                            st.info("Carregue pelo menos 2 atletas para usar esta comparação.")
                                        else:
                                            ca1, ca2 = st.columns(2)
                                            with ca1:
                                                atl_A = st.selectbox("Atleta A:", _ats_cmp,
                                                                      index=0, key="cmp_a1")
                                            with ca2:
                                                atl_B = st.selectbox("Atleta B:", _ats_cmp,
                                                                      index=min(1, len(_ats_cmp)-1),
                                                                      key="cmp_a2")

                                            if atl_A != atl_B:
                                                # Combina todos os períodos selecionados para cada atleta
                                                xa, ya, xb, yb = [], [], [], []
                                                for _pm in periodos_mapa_sel:
                                                    _dA = dados_posicao_por_periodo.get(_pm, {}).get(atl_A, {})
                                                    _dB = dados_posicao_por_periodo.get(_pm, {}).get(atl_B, {})
                                                    cfg_ref = st.session_state.get(f"campo_cfg__{atleta_mapa}", cfg)
                                                    if _dA.get('xs'):
                                                        xa += list(_dA['xs']); ya += list(_dA['ys'])
                                                    elif _dA.get('lats') and cfg_ref:
                                                        _gxa, _gya = gps_para_campo_coords(_dA['lats'], _dA['lons'], cfg_ref)
                                                        xa += _gxa; ya += _gya
                                                    if _dB.get('xs'):
                                                        xb += list(_dB['xs']); yb += list(_dB['ys'])
                                                    elif _dB.get('lats') and cfg_ref:
                                                        _gxb, _gyb = gps_para_campo_coords(_dB['lats'], _dB['lons'], cfg_ref)
                                                        xb += _gxb; yb += _gyb

                                                if xa and xb:
                                                    fig_cmp = desenhar_campo_futebol_bonito(
                                                        title=f"Comparação: {atl_A} (azul) vs {atl_B} (rosa)")
                                                    _sa = max(1, len(xa)//3000)
                                                    _sb = max(1, len(xb)//3000)
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=xa[::_sa], y=ya[::_sa], mode='markers', name=atl_A,
                                                        marker=dict(size=2, color='#00E5FF', opacity=0.5)))
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=xb[::_sb], y=yb[::_sb], mode='markers', name=atl_B,
                                                        marker=dict(size=2, color='#FF4081', opacity=0.5)))
                                                    st.plotly_chart(fig_cmp, use_container_width=True)
                                                else:
                                                    st.info("Dados de posição não disponíveis para um dos atletas selecionados.")
                                            else:
                                                st.info("Selecione dois atletas diferentes para comparar.")

                                    # ══════════════════════════════════════════════
                                    # ══════════════════════════════════════════════
                                    # ANÁLISE — MAPA DE ALTA INTENSIDADE POSICIONAL
                                    # ══════════════════════════════════════════════
                                    st.markdown("---")
                                    with st.expander("🗺️ Mapa de Alta Intensidade Posicional (HSR Zone Map)", expanded=False):
                                        st.markdown(
                                            "Mostra **onde no campo** o atleta realiza ações de alta intensidade. "
                                            "Vai além do heatmap global — filtra apenas os momentos acima do "
                                            "limiar configurado, revelando corredores de sprint, zonas de "
                                            "pressing e rotas de recuperação."
                                        )
                                        # ── Bandas de velocidade da conta Catapult ──────
                                        _hsr_zones_acc = (
                                            st.session_state.get('velocity_zones_account')
                                            or _DEFAULT_VELOCITY_ZONES
                                        )
                                        # Monta lista de opções: apenas bandas com min >= 5 km/h
                                        _hsr_band_opts = []
                                        _hsr_band_map  = {}  # label → min_kmh
                                        for _z in _hsr_zones_acc:
                                            _z_min_kmh = round(float(_z.get('min_ms') or 0) * 3.6, 1)
                                            _z_max_ms  = float(_z.get('max_ms', 9999))
                                            _z_max_str = (
                                                '∞'
                                                if _z_max_ms >= 9000
                                                else f"{round(_z_max_ms*3.6,1)} km/h"
                                            )
                                            if _z_min_kmh >= 5:
                                                _lbl = f"{_z['name']}  ({_z_min_kmh}–{_z_max_str})"
                                                _hsr_band_opts.append(_lbl)
                                                _hsr_band_map[_lbl] = _z_min_kmh

                                        # Default: seleciona automaticamente zonas ≥ 14 km/h
                                        _hsr_default = [
                                            l for l, v in _hsr_band_map.items() if v >= 14.0
                                        ] or (_hsr_band_opts[-2:] if len(_hsr_band_opts) >= 2 else _hsr_band_opts)

                                        _hsr_sels = st.multiselect(
                                            "Bandas HSR (High Speed Running):",
                                            _hsr_band_opts,
                                            default=_hsr_default,
                                            key="hsr_zones_sel",
                                            help=(
                                                "Selecione as bandas que representam alta intensidade. "
                                                "O limiar será o **menor valor** entre as bandas escolhidas."
                                            ),
                                        )

                                        # Limiar = mínimo das bandas selecionadas
                                        if _hsr_sels:
                                            _hsr_vel_thr = min(_hsr_band_map[l] for l in _hsr_sels)
                                        else:
                                            _hsr_vel_thr = 14.0   # fallback

                                        st.caption(
                                            f"🎯 Limiar calculado: **>{_hsr_vel_thr} km/h** "
                                            f"(mínimo das bandas selecionadas)"
                                        )
                                        _hsr_vel_arr = np.array(vel_raw_campo, dtype=float)
                                        _hsr_xn = np.array(xn, dtype=float)
                                        _hsr_yn = np.array(yn, dtype=float)
                                        _hsr_mask = _hsr_vel_arr >= _hsr_vel_thr

                                        _hm1, _hm2, _hm3 = st.columns(3)
                                        _hm1.metric(f"Pontos >{_hsr_vel_thr} km/h", f"{int(_hsr_mask.sum()):,}")
                                        _hm2.metric("% tempo em HSR",
                                                    f"{float(_hsr_mask.mean())*100:.1f}%")
                                        _hm3.metric("Distância HSR (m)",
                                                    f"{float((_hsr_vel_arr[_hsr_mask]/3.6/10).sum()):.0f}")

                                        if _hsr_mask.sum() >= 10:
                                            _hsr_fl = float(st.session_state.get('venue', {}).get('length') or 105)
                                            _hsr_fw = float(st.session_state.get('venue', {}).get('width')  or 68)
                                            _n_lng, _n_lat = 5, 3
                                            _z_lng = ['Def. Prof.', 'Defensivo', 'Meio-Campo',
                                                      'Ofensivo', 'Ataque Prof.']
                                            _z_lat = ['Flanco Esq.', 'Centro', 'Flanco Dir.']

                                            _hsr_xz = _hsr_xn[_hsr_mask]
                                            _hsr_yz = _hsr_yn[_hsr_mask]
                                            _hsr_vz = _hsr_vel_arr[_hsr_mask]

                                            _zone_cnt = np.zeros((_n_lat, _n_lng))
                                            _zone_vel = np.zeros((_n_lat, _n_lng))
                                            for _px, _py, _pv in zip(_hsr_xz, _hsr_yz, _hsr_vz):
                                                _ci = min(int(_px / _hsr_fl * _n_lng), _n_lng - 1)
                                                _ri = min(int(_py / _hsr_fw * _n_lat), _n_lat - 1)
                                                _zone_cnt[_ri, _ci] += 1
                                                _zone_vel[_ri, _ci] += _pv
                                            _zone_vavg = np.where(
                                                _zone_cnt > 0, _zone_vel / _zone_cnt, 0)

                                            _hc1, _hc2 = st.columns([2, 1])
                                            with _hc1:
                                                _fig_hsr = desenhar_campo_futebol_bonito(
                                                    title=(f"HSR >{_hsr_vel_thr} km/h "
                                                           f"— {atleta_mapa}")
                                                )
                                                _sbg_h = max(1, len(_hsr_xn) // 3000)
                                                _fig_hsr.add_trace(go.Scatter(
                                                    x=_hsr_xn[::_sbg_h], y=_hsr_yn[::_sbg_h],
                                                    mode='markers',
                                                    marker=dict(size=1.5,
                                                                color='rgba(255,255,255,0.06)'),
                                                    showlegend=False, hoverinfo='skip'
                                                ))
                                                _sbg_hs = max(1, int(_hsr_mask.sum()) // 3000)
                                                _fig_hsr.add_trace(go.Scatter(
                                                    x=_hsr_xz[::_sbg_hs],
                                                    y=_hsr_yz[::_sbg_hs],
                                                    mode='markers',
                                                    marker=dict(
                                                        size=4,
                                                        color=_hsr_vz[::_sbg_hs],
                                                        colorscale=[
                                                            [0, '#66BB6A'], [0.4, '#FFA726'],
                                                            [0.7, '#EF5350'], [1, '#B71C1C']],
                                                        cmin=float(_hsr_vel_thr),
                                                        cmax=float(min(_hsr_vz.max(), 40)),
                                                        showscale=True,
                                                        colorbar=dict(
                                                            title=dict(text='km/h',
                                                                       font=dict(color='white')),
                                                            tickfont=dict(color='white'), len=0.55,
                                                        ),
                                                        opacity=0.78,
                                                    ),
                                                    showlegend=False,
                                                ))
                                                st.plotly_chart(_fig_hsr, use_container_width=True)

                                            with _hc2:
                                                _fig_zmap = go.Figure(go.Heatmap(
                                                    z=_zone_cnt.tolist(),
                                                    x=_z_lng, y=_z_lat,
                                                    colorscale='YlOrRd',
                                                    text=[[f"{int(_zone_cnt[r, c])}"
                                                           for c in range(_n_lng)]
                                                          for r in range(_n_lat)],
                                                    texttemplate='%{text}',
                                                    textfont=dict(size=10, color='black'),
                                                    showscale=False,
                                                    hovertemplate=(
                                                        '%{x} / %{y}<br>'
                                                        'Pontos HSR: %{z}<extra></extra>'
                                                    ),
                                                ))
                                                _fig_zmap.update_layout(
                                                    title=dict(text='Freq. por Zona',
                                                               font=dict(color='white', size=12)),
                                                    plot_bgcolor='#0e1117',
                                                    paper_bgcolor='#0e1117',
                                                    font=dict(color='white'), height=290,
                                                    margin=dict(t=40, b=60, l=90, r=10),
                                                    xaxis=dict(tickfont=dict(color='white', size=8),
                                                               tickangle=-30),
                                                    yaxis=dict(tickfont=dict(color='white', size=8)),
                                                )
                                                st.plotly_chart(_fig_zmap, use_container_width=True)
                                                _max_ri2, _max_ci2 = np.unravel_index(
                                                    _zone_cnt.argmax(), _zone_cnt.shape)
                                                st.caption(
                                                    f"📍 Zona mais ativa: **{_z_lat[_max_ri2]}** × "
                                                    f"**{_z_lng[_max_ci2]}** "
                                                    f"({int(_zone_cnt[_max_ri2, _max_ci2])} ações)"
                                                )
                                        else:
                                            st.info(
                                                f"Nenhum ponto com velocidade >{_hsr_vel_thr} km/h. "
                                                "Tente reduzir o limiar."
                                            )

                                    # ══════════════════════════════════════════════
                                    # ANÁLISE 5 — DISTÂNCIA ENTRE ATLETAS
                                    # ══════════════════════════════════════════════
                                    if len(atletas_mapa) >= 2:
                                        st.markdown("---")
                                        with st.expander(
                                            f"📏 Distância Entre Atletas "
                                            f"({len(atletas_mapa)} selecionados)",
                                            expanded=False
                                        ):
                                            st.markdown(
                                                "Distância Euclidiana entre os atletas selecionados "
                                                "ao longo do tempo, a partir das coordenadas de campo. "
                                                "Útil para analisar **compactação defensiva**, "
                                                "**pressing em dupla** e **cobertura de espaço**."
                                            )
                                            _cfg_dist = st.session_state.get(
                                                f"campo_cfg__{atleta_mapa}", cfg)
                                            _DCORES = ['#00E5FF','#FF4081','#FFEB3B',
                                                       '#69F0AE','#FF9800','#CE93D8']

                                            # Coleta arrays por atleta
                                            _dist_dados = {}
                                            for _atl_d in atletas_mapa:
                                                _xd, _yd, _tsd = [], [], []
                                                for _pm in periodos_mapa_sel:
                                                    _dd = dados_posicao_por_periodo.get(
                                                        _pm, {}).get(_atl_d, {})
                                                    if _dd.get('xs') and _dd.get('ys'):
                                                        _xd  += list(_dd['xs'])
                                                        _yd  += list(_dd['ys'])
                                                        _tsd += list(_dd.get('ts_pos', []))
                                                    elif _dd.get('lats') and _cfg_dist:
                                                        _gxd, _gyd = gps_para_campo_coords(
                                                            _dd['lats'], _dd['lons'], _cfg_dist)
                                                        _xd  += _gxd
                                                        _yd  += _gyd
                                                        _tsd += [0.0] * len(_gxd)
                                                if _xd:
                                                    _dist_dados[_atl_d] = {
                                                        'x': np.array(_xd, dtype=float),
                                                        'y': np.array(_yd, dtype=float),
                                                        'ts': np.array(_tsd, dtype=float)
                                                    }

                                            _pares_d = [
                                                (atletas_mapa[_ii], atletas_mapa[_jj])
                                                for _ii in range(len(atletas_mapa))
                                                for _jj in range(_ii + 1, len(atletas_mapa))
                                            ]

                                            if len(_dist_dados) >= 2:
                                                _fig_dist = go.Figure()
                                                _resumo_d = []

                                                for _pi_d, (_na, _nb) in enumerate(_pares_d):
                                                    if _na not in _dist_dados or _nb not in _dist_dados:
                                                        continue
                                                    _dA = _dist_dados[_na]
                                                    _dB = _dist_dados[_nb]
                                                    _tsA, _tsB = _dA['ts'], _dB['ts']
                                                    _has_ts_d = (
                                                        _tsA.sum() > 0 and _tsB.sum() > 0)

                                                    if _has_ts_d:
                                                        _tsA_v = _tsA[_tsA > 0]
                                                        _tsB_v = _tsB[_tsB > 0]
                                                        _tc = np.arange(
                                                            max(_tsA_v.min(), _tsB_v.min()),
                                                            min(_tsA_v.max(), _tsB_v.max()),
                                                            0.1
                                                        )
                                                        if len(_tc) < 10:
                                                            _has_ts_d = False

                                                    if _has_ts_d:
                                                        _xA_i = np.interp(_tc, _tsA_v, _dA['x'][_tsA > 0])
                                                        _yA_i = np.interp(_tc, _tsA_v, _dA['y'][_tsA > 0])
                                                        _xB_i = np.interp(_tc, _tsB_v, _dB['x'][_tsB > 0])
                                                        _yB_i = np.interp(_tc, _tsB_v, _dB['y'][_tsB > 0])
                                                        _t_ax = _tc - _tc[0]
                                                    else:
                                                        _nn = min(len(_dA['x']), len(_dB['x']))
                                                        _xA_i = _dA['x'][:_nn]
                                                        _yA_i = _dA['y'][:_nn]
                                                        _xB_i = _dB['x'][:_nn]
                                                        _yB_i = _dB['y'][:_nn]
                                                        _t_ax = np.arange(_nn) / 10.0

                                                    _dist_v = np.sqrt(
                                                        (_xA_i - _xB_i)**2 +
                                                        (_yA_i - _yB_i)**2
                                                    )
                                                    _sbgd = max(1, len(_dist_v) // 3000)
                                                    _lbl = f"{_na.split()[0]} ↔ {_nb.split()[0]}"
                                                    _cor_d = _DCORES[_pi_d % len(_DCORES)]

                                                    _fig_dist.add_trace(go.Scatter(
                                                        x=_t_ax[::_sbgd] / 60,
                                                        y=_dist_v[::_sbgd],
                                                        mode='lines',
                                                        name=_lbl,
                                                        line=dict(color=_cor_d, width=1.5),
                                                        hovertemplate=(
                                                            '%{x:.1f} min | '
                                                            '%{y:.1f} m<extra>'
                                                            + _lbl + '</extra>'
                                                        )
                                                    ))
                                                    _resumo_d.append({
                                                        'Par': _lbl,
                                                        'Dist. Média (m)':
                                                            round(float(_dist_v.mean()), 1),
                                                        'Mediana (m)':
                                                            round(float(np.median(_dist_v)), 1),
                                                        'Mín (m)':
                                                            round(float(_dist_v.min()), 1),
                                                        'Máx (m)':
                                                            round(float(_dist_v.max()), 1),
                                                        '% Tempo < 5 m':
                                                            round(100 * ((_dist_v < 5).sum()
                                                                         / len(_dist_v)), 1),
                                                        '% Tempo < 10 m':
                                                            round(100 * ((_dist_v < 10).sum()
                                                                         / len(_dist_v)), 1),
                                                    })

                                                    # Campo de proximidade (apenas 1º par)
                                                    if _pi_d == 0 and _dist_v.min() < 15:
                                                        _prox_mk = _dist_v < 10
                                                        if _prox_mk.sum() > 5:
                                                            _st_prox = _dist_dados
                                                            _xmid = ((_xA_i[_prox_mk] +
                                                                       _xB_i[_prox_mk]) / 2)
                                                            _ymid = ((_yA_i[_prox_mk] +
                                                                       _yB_i[_prox_mk]) / 2)
                                                            _fig_prox = desenhar_campo_futebol_bonito(
                                                                title=(f"Zonas de Proximidade "
                                                                       f"(< 10 m) — {_lbl}")
                                                            )
                                                            _sbgp = max(1, len(_xA_i) // 3000)
                                                            _fig_prox.add_trace(go.Scatter(
                                                                x=_xA_i[::_sbgp],
                                                                y=_yA_i[::_sbgp],
                                                                mode='markers',
                                                                marker=dict(size=1.5,
                                                                    color='rgba(0,229,255,0.18)'),
                                                                name=_na, showlegend=True
                                                            ))
                                                            _fig_prox.add_trace(go.Scatter(
                                                                x=_xB_i[::_sbgp],
                                                                y=_yB_i[::_sbgp],
                                                                mode='markers',
                                                                marker=dict(size=1.5,
                                                                    color='rgba(255,64,129,0.18)'),
                                                                name=_nb, showlegend=True
                                                            ))
                                                            _sbgpr = max(1, len(_xmid) // 1500)
                                                            _fig_prox.add_trace(go.Scatter(
                                                                x=_xmid[::_sbgpr],
                                                                y=_ymid[::_sbgpr],
                                                                mode='markers',
                                                                marker=dict(
                                                                    size=5,
                                                                    color='rgba(255,82,82,0.65)',
                                                                    symbol='circle'
                                                                ),
                                                                name='Proximidade < 10 m',
                                                                showlegend=True
                                                            ))
                                                            _fig_prox.update_layout(height=430)
                                                            _prox_fig_final = _fig_prox

                                                # Linhas de referência
                                                _fig_dist.add_hline(
                                                    y=5, line_dash='dot',
                                                    line_color='rgba(255,235,59,0.45)',
                                                    annotation_text='5 m',
                                                    annotation_font_color='#FFEB3B',
                                                    annotation_position='right'
                                                )
                                                _fig_dist.add_hline(
                                                    y=10, line_dash='dot',
                                                    line_color='rgba(255,152,0,0.45)',
                                                    annotation_text='10 m',
                                                    annotation_font_color='#FF9800',
                                                    annotation_position='right'
                                                )
                                                _fig_dist.update_layout(
                                                    title=dict(
                                                        text="Distância Entre Atletas ao Longo do Tempo",
                                                        font=dict(color='white', size=14)
                                                    ),
                                                    xaxis=dict(
                                                        title='Tempo (min)',
                                                        gridcolor='rgba(255,255,255,0.1)',
                                                        color='white'
                                                    ),
                                                    yaxis=dict(
                                                        title='Distância (m)',
                                                        gridcolor='rgba(255,255,255,0.1)',
                                                        color='white'
                                                    ),
                                                    paper_bgcolor='rgba(0,0,0,0)',
                                                    plot_bgcolor='rgba(20,20,30,0.85)',
                                                    legend=dict(
                                                        font=dict(color='white'),
                                                        bgcolor='rgba(0,0,0,0.45)'
                                                    ),
                                                    height=390
                                                )
                                                st.plotly_chart(_fig_dist,
                                                                use_container_width=True)

                                                if _resumo_d:
                                                    st.markdown("**📋 Resumo por Par**")
                                                    st.dataframe(
                                                        pd.DataFrame(_resumo_d).set_index('Par'),
                                                        use_container_width=True
                                                    )

                                                if 'prox_fig_final' in dir() and _prox_fig_final:
                                                    st.markdown("---")
                                                    st.plotly_chart(_prox_fig_final,
                                                                    use_container_width=True)
                                                    st.caption(
                                                        "🔴 Pontos vermelhos = posição média "
                                                        "entre os atletas quando distância < 10 m"
                                                    )
                                            else:
                                                st.warning(
                                                    "Dados de posição insuficientes "
                                                    "para os atletas selecionados.")
                                    else:
                                        pass  # análise de distância requer 2+ atletas

                                else:
                                    st.error("❌ Não foi possível calcular as coordenadas de campo.")
                            else:
                                st.warning(
                                    "⚠️ Sem dados de posição disponíveis para análise de campo.\n\n"
                                    "**Para ativar esta seção:**\n"
                                    "- Certifique-se de que o campo GPS está **aplicado** "
                                    "(clique ✅ Aplicar Campo no mapa acima)\n"
                                    "- Verifique se o sensor GPS estava ativo durante a sessão"
                                )

                    elif periodos_mapa_sel:
                        st.info("Selecione pelo menos um atleta para continuar.")

                else:
                    st.info("Dados de posição não disponíveis. Verifique se o sensor GPS estava ativo durante a sessão.")

                # ── FEATURE 5: Anotações Táticas ─────────────────────────────
                st.markdown("---")
                st.markdown("### 📌 Anotações Táticas")
                _ann_api = st.session_state.get('api')
                _ann_act_id = st.session_state.get('activity_id')

                if _ann_api and _ann_act_id:
                    # Carregar anotações existentes
                    if st.button("🔄 Carregar anotações", key="btn_load_ann"):
                        try:
                            _ann_raw = _ann_api.get_activity_annotations(_ann_act_id)
                            st.session_state['annotations'] = _ann_raw
                        except Exception as _ae:
                            st.error(f"Erro: {_ae}")

                    _annotations = st.session_state.get('annotations')
                    if _annotations:
                        try:
                            _ann_list = (_annotations if isinstance(_annotations, list)
                                         else _annotations.get('data', []))
                            if _ann_list:
                                st.markdown("#### Anotações existentes")
                                # Timeline visual
                                _ann_ts0 = min(
                                    (float(a.get('start_time') or 0) for a in _ann_list if a.get('start_time')),
                                    default=0
                                )
                                _fig_ann = go.Figure()
                                for _ia, _ann in enumerate(_ann_list):
                                    _t0 = float(_ann.get('start_time') or 0) - _ann_ts0
                                    _t1 = float(_ann.get('end_time', _ann.get('start_time', 0))) - _ann_ts0
                                    _ann_name = _ann.get('name', f'Anotação {_ia+1}')
                                    _fig_ann.add_shape(
                                        type='rect',
                                        x0=_t0, x1=max(_t1, _t0 + 1),
                                        y0=_ia, y1=_ia + 0.8,
                                        fillcolor='#2196F3', opacity=0.6,
                                        line_width=0,
                                    )
                                    _fig_ann.add_annotation(
                                        x=(_t0 + _t1) / 2, y=_ia + 0.4,
                                        text=_ann_name, showarrow=False,
                                        font=dict(color='white', size=10),
                                    )
                                _fig_ann.update_layout(
                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(title='Tempo (s)', gridcolor='#333'),
                                    yaxis=dict(visible=False),
                                    height=max(120, len(_ann_list) * 40 + 40),
                                    margin=dict(t=10, b=30),
                                )
                                st.plotly_chart(_fig_ann, use_container_width=True)

                                # Lista com botão de deletar
                                for _ann in _ann_list:
                                    _ann_id = _ann.get('id', '')
                                    _ann_nm = _ann.get('name', '—')
                                    _c1, _c2 = st.columns([4, 1])
                                    _c1.write(f"**{_ann_nm}** (id: {_ann_id})")
                                    if _c2.button("🗑️", key=f"del_ann_{_ann_id}"):
                                        try:
                                            _del_ok = _ann_api.delete_annotation(_ann_id)
                                            if _del_ok:
                                                st.success("Anotação removida.")
                                                st.session_state.pop('annotations', None)
                                                st.rerun()
                                        except Exception as _de:
                                            st.error(f"Erro: {_de}")
                        except Exception as _ep:
                            st.error(f"Erro ao processar anotações: {_ep}")

                    # Criar nova anotação
                    with st.expander("➕ Nova Anotação", expanded=False):
                        _ann_c1, _ann_c2 = st.columns(2)
                        with _ann_c1:
                            _ann_new_name = st.text_input("Nome:", key="ann_new_name")
                            _ann_new_start = st.number_input("Início (s Unix):", value=0, key="ann_new_start")
                        with _ann_c2:
                            _ann_new_type = st.selectbox(
                                "Tipo:", ["phase", "event", "highlight"],
                                key="ann_new_type"
                            )
                            _ann_new_end = st.number_input("Fim (s Unix):", value=0, key="ann_new_end")
                        if st.button("✅ Criar Anotação", key="btn_create_ann"):
                            if _ann_new_name:
                                try:
                                    _cr_resp = _ann_api.create_activity_annotation(
                                        _ann_act_id, _ann_new_name,
                                        _ann_new_start, _ann_new_end,
                                        _ann_new_type,
                                    )
                                    if _cr_resp:
                                        st.success("Anotação criada!")
                                        st.session_state.pop('annotations', None)
                                    else:
                                        st.warning("Endpoint de anotações pode não estar disponível nesta licença.")
                                except Exception as _cae:
                                    st.error(f"Erro: {_cae}")
                            else:
                                st.warning("Informe um nome para a anotação.")
                else:
                    st.info("Carregue os dados de uma atividade para usar anotações.")
