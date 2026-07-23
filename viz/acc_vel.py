# -*- coding: utf-8 -*-
"""Aba extraída de main() (P4 — page-decomposition)."""
from __future__ import annotations

from scipy.ndimage import gaussian_filter as _gf
from field import cor_atleta
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st


def render_acc_vel(dados_sensor_por_atleta_por_periodo, resultados_por_periodo):
                st.subheader("🏎️ Perfil Aceleração × Velocidade")
                st.caption(
                    "Baseado no modelo de Samozino & Morin (2016) — relação linear entre aceleração e velocidade "
                    "para extrair o perfil mecânico individual de sprint."
                )
                st.markdown("---")

                _av_periodos = list(dados_sensor_por_atleta_por_periodo.keys())
                if _av_periodos and resultados_por_periodo:
                    _AV_TODOS = "🔀 Todos os períodos (combinado)"
                    _av_opcoes_per = [_AV_TODOS] + _av_periodos
                    _av_col1, _av_col2 = st.columns([2, 1])
                    with _av_col1:
                        _av_per = st.selectbox("Período:", _av_opcoes_per, key="av_periodo")
                    with _av_col2:
                        if _av_per == _AV_TODOS:
                            _av_ats_set = set()
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _av_ats_set.update(_pv.keys())
                            _av_atletas_disp = sorted(_av_ats_set)
                        else:
                            _av_atletas_disp = list(dados_sensor_por_atleta_por_periodo.get(_av_per, {}).keys())
                        _av_atls_sel = st.multiselect(
                            "Atletas (até 6):", _av_atletas_disp,
                            default=_av_atletas_disp[:min(3, len(_av_atletas_disp))],
                            key="av_atletas_sel"
                        )

                    if _av_atls_sel:
                        # ── Paleta de cores por atleta (usa paleta global persistente) ─────
                        _AV_PALETTE = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4']
                        _av_cores = {
                            a: st.session_state.get('athlete_colors', {}).get(a, _AV_PALETTE[i % len(_AV_PALETTE)])
                            for i, a in enumerate(_av_atls_sel)
                        }

                        # ── Extrai (v, a) de todos os atletas selecionados ────
                        _av_dados = {}
                        for _av_atl in _av_atls_sel:
                            if _av_per == _AV_TODOS:
                                _spts = []
                                for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                    _spts += _pv2.get(_av_atl, [])
                            else:
                                _spts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_av_atl, [])
                            if not _spts:
                                continue
                            _vels, _accs = [], []
                            for _p in _spts:
                                _v = _p.get('v')
                                _a = _p.get('a')
                                if _v is not None and _a is not None:
                                    _vels.append(float(_v) * 3.6)
                                    _accs.append(float(_a))
                            if _vels:
                                _av_dados[_av_atl] = {
                                    'vel': np.array(_vels),
                                    'acc': np.array(_accs),
                                }

                        if not _av_dados:
                            st.warning("Dados de aceleração não disponíveis para os atletas selecionados.")
                        else:
                            # ── FEATURE 3: toggle para velocidade bruta (rv) ─
                            _av_use_rv = st.toggle(
                                "Usar velocidade bruta (rv) para F-V profiling",
                                value=False, key="av_use_rv",
                                help="rv = velocidade bruta do sensor (mais preciso para F-V). "
                                     "Requer que o dispositivo suporte o parâmetro rv.",
                            )
                            if _av_use_rv:
                                st.caption("📍 **Perfil F-V (velocidade bruta — mais preciso)**")
                                # Recalcula usando rv ao invés de v
                                for _av_atl in list(_av_dados.keys()):
                                    if _av_per == _AV_TODOS:
                                        _spts_rv = []
                                        for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                            _spts_rv += _pv2.get(_av_atl, [])
                                    else:
                                        _spts_rv = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_av_atl, [])
                                    _vels_rv, _accs_rv = [], []
                                    for _p in _spts_rv:
                                        _rv = _p.get('rv')
                                        _a = _p.get('a')
                                        if _rv is not None and _a is not None:
                                            _vels_rv.append(float(_rv) * 3.6)
                                            _accs_rv.append(float(_a))
                                    if _vels_rv:
                                        _av_dados[_av_atl]['vel'] = np.array(_vels_rv)
                                        _av_dados[_av_atl]['acc'] = np.array(_accs_rv)

                            # ════════════════════════════════════════════════
                            # SEÇÃO 1 — SCATTER ACC × VEL (multi-atleta)
                            # ════════════════════════════════════════════════
                            _sc_title = ("📍 Perfil F-V (velocidade bruta — mais preciso)"
                                         if _av_use_rv else "📍 Scatter Aceleração × Velocidade")
                            st.markdown(f"### {_sc_title}")
                            _fig_sc = go.Figure()

                            for _av_atl, _d in _av_dados.items():
                                # Sub-amostra para não sobrecarregar o gráfico
                                _step = max(1, len(_d['vel']) // 3000)
                                _v_s  = _d['vel'][::_step]
                                _a_s  = _d['acc'][::_step]
                                _fig_sc.add_trace(go.Scatter(
                                    x=_v_s, y=_a_s, mode='markers',
                                    marker=dict(color=_av_cores[_av_atl], size=3, opacity=0.45),
                                    name=_av_atl,
                                    hovertemplate=f'{_av_atl}<br>Vel: %{{x:.1f}} km/h<br>Acc: %{{y:.2f}} m/s²<extra></extra>',
                                ))

                            # Linhas de limiar
                            for _lim, _cor, _txt in [(3.0,'#F44336','Acc >3 m/s²'),
                                                      (2.0,'#FF9800','Acc >2 m/s²'),
                                                      (-2.0,'#FF9800','Dcc <-2 m/s²'),
                                                      (-3.0,'#F44336','Dcc <-3 m/s²')]:
                                _fig_sc.add_hline(y=_lim, line=dict(color=_cor, dash='dash', width=1),
                                                  annotation_text=_txt,
                                                  annotation_font=dict(color=_cor, size=10))

                            _fig_sc.add_hline(y=0, line=dict(color='white', width=1, dash='dot'))
                            _fig_sc.update_layout(
                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                font=dict(color='white'),
                                xaxis=dict(title='Velocidade (km/h)', gridcolor='#333', range=[-1, None]),
                                yaxis=dict(title='Aceleração (m/s²)', gridcolor='#333'),
                                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)'),
                                height=420, margin=dict(t=20, b=10),
                            )
                            st.plotly_chart(_fig_sc, use_container_width=True)

                            st.markdown("---")

                            # ════════════════════════════════════════════════
                            # SEÇÃO 2 — HISTOGRAMA DE ACELERAÇÕES POR ZONA
                            # ════════════════════════════════════════════════
                            st.markdown("### 📊 Distribuição de Acelerações por Zona")
                            _av_c3, _av_c4 = st.columns(2)

                            _ZONAS_ACC = [
                                ('Muito intensa (>3)', 3.0, 99, '#F44336'),
                                ('Intensa (2–3)',       2.0, 3.0,'#FF9800'),
                                ('Moderada (1–2)',      1.0, 2.0,'#FFEB3B'),
                                ('Leve (0–1)',          0.0, 1.0,'#4CAF50'),
                                ('Desac. leve (0–-1)', -1.0, 0.0,'#26C6DA'),
                                ('Desac. mod. (-1–-2)',-2.0,-1.0,'#1565C0'),
                                ('Desac. int. (-2–-3)',-3.0,-2.0,'#7B1FA2'),
                                ('Desac. muito int. (<-3)',-99,-3.0,'#880E4F'),
                            ]

                            with _av_c3:
                                _fig_hist = go.Figure()
                                for _av_atl, _d in _av_dados.items():
                                    _fig_hist.add_trace(go.Histogram(
                                        x=_d['acc'], name=_av_atl,
                                        marker_color=_av_cores[_av_atl],
                                        opacity=0.65, nbinsx=60,
                                        hovertemplate='Acc: %{x:.2f} m/s²<br>Freq: %{y}<extra></extra>',
                                    ))
                                _fig_hist.update_layout(
                                    title=dict(text='Histograma de Aceleração', font=dict(color='white',size=13)),
                                    barmode='overlay',
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(title='Aceleração (m/s²)', gridcolor='#333'),
                                    yaxis=dict(title='Frequência', gridcolor='#333'),
                                    legend=dict(font=dict(color='white')),
                                    height=320, margin=dict(t=45,b=10,l=10,r=10),
                                )
                                st.plotly_chart(_fig_hist, use_container_width=True)

                            with _av_c4:
                                # Proporção por zona para o primeiro atleta selecionado
                                _av_atl_zona = _av_atls_sel[0]
                                if _av_atl_zona in _av_dados:
                                    _acc_z = _av_dados[_av_atl_zona]['acc']
                                    _zona_counts, _zona_labels, _zona_cores = [], [], []
                                    for _zl, _zmin, _zmax, _zc in _ZONAS_ACC:
                                        _n = int(np.sum((_acc_z >= _zmin) & (_acc_z < _zmax)))
                                        _zona_counts.append(_n)
                                        _zona_labels.append(_zl)
                                        _zona_cores.append(_zc)
                                    _fig_pie = go.Figure(go.Pie(
                                        labels=_zona_labels, values=_zona_counts,
                                        marker=dict(colors=_zona_cores),
                                        textfont=dict(color='white', size=10),
                                        hole=0.4,
                                    ))
                                    _fig_pie.update_layout(
                                        title=dict(text=f'Distribuição por Zona — {_av_atl_zona}',
                                                   font=dict(color='white',size=13)),
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        legend=dict(font=dict(color='white',size=9)),
                                        height=320, margin=dict(t=45,b=10,l=10,r=10),
                                    )
                                    st.plotly_chart(_fig_pie, use_container_width=True)

                            st.markdown("---")

                            # ════════════════════════════════════════════════
                            # SEÇÃO 5 — DENSIDADE NO ESPAÇO ACC × VEL (heatmap 2D)
                            # ════════════════════════════════════════════════
                            st.markdown("### 🌡️ Mapa de Densidade Acc × Vel")
                            st.caption(
                                "Mostra onde o atleta passa a maior parte do tempo no espaço aceleração-velocidade. "
                                "Regiões quentes = maior acúmulo de esforço."
                            )
                            _av_atl_hm = st.selectbox("Atleta para mapa de densidade:",
                                                       _av_atls_sel, key="av_hm_atl")
                            if _av_atl_hm in _av_dados:
                                _v_hm = _av_dados[_av_atl_hm]['vel']
                                _a_hm = _av_dados[_av_atl_hm]['acc']
                                _H2d, _xe2, _ye2 = np.histogram2d(
                                    _v_hm, _a_hm, bins=[50, 40],
                                    range=[[0, max(35, float(_v_hm.max()))], [-6, 6]]
                                )
                                _H2d = _gf(_H2d, sigma=1.5)
                                _xc2 = (_xe2[:-1] + _xe2[1:]) / 2
                                _yc2 = (_ye2[:-1] + _ye2[1:]) / 2
                                _fig_hm2 = go.Figure(go.Heatmap(
                                    x=_xc2, y=_yc2, z=_H2d.T,
                                    colorscale=[[0,'rgba(0,0,0,0)'],[0.0001,'#0D47A1'],
                                                [0.3,'#1565C0'],[0.6,'#FFEB3B'],
                                                [0.85,'#FF9800'],[1,'#F44336']],
                                    opacity=0.85, showscale=True,
                                    colorbar=dict(
                                        title=dict(text='Densidade', font=dict(color='white')),
                                        tickfont=dict(color='white'),
                                    ),
                                    hovertemplate='Vel: %{x:.1f} km/h<br>Acc: %{y:.2f} m/s²<br>Freq: %{z:.0f}<extra></extra>',
                                ))
                                _fig_hm2.add_hline(y=0, line=dict(color='white', width=1, dash='dot'))
                                _fig_hm2.add_hline(y=3,  line=dict(color='#F44336', width=1, dash='dash'))
                                _fig_hm2.add_hline(y=-3, line=dict(color='#F44336', width=1, dash='dash'))
                                _fig_hm2.update_layout(
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(title='Velocidade (km/h)', gridcolor='#333'),
                                    yaxis=dict(title='Aceleração (m/s²)', gridcolor='#333'),
                                    height=400, margin=dict(t=20,b=10),
                                )
                                st.plotly_chart(_fig_hm2, use_container_width=True)

                    else:
                        st.info("Selecione pelo menos 1 atleta para ver o perfil Acc × Vel.")

                    # ════════════════════════════════════════════════════════
                    # #14 — QUALIDADE DE ACELERAÇÃO + #11 — PL DIRECIONAL
                    # ════════════════════════════════════════════════════════
                    if _av_atletas_disp:
                        st.markdown("---")

                        # ── #14: Qualidade dos Eventos de Aceleração ──────────
                        st.markdown("### 🔬 Qualidade de Aceleração")
                        st.caption(
                            "Cada evento de aceleração (cruzar +2 m/s² por ≥0.3 s) é "
                            "caracterizado por: pico, impulso (área sob a curva), taxa de "
                            "desenvolvimento e velocidade de entrada. Mostra SE os esforços "
                            "mantêm qualidade ao longo do jogo."
                        )
                        _qa_atl = st.selectbox("Atleta:", _av_atletas_disp, key="qa_atleta")
                        if _av_per == _AV_TODOS:
                            _qa_pts: list = []
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _qa_pts += _pv.get(_qa_atl, [])
                        else:
                            _qa_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_qa_atl, [])

                        _qa_vs = np.array([float(p.get('v') or 0) * 3.6 for p in _qa_pts])
                        _qa_as = np.array([float(p.get('a') or 0) for p in _qa_pts])
                        _qa_ts = np.array([float(p.get('ts') or 0) for p in _qa_pts])

                        if len(_qa_as) > 10:
                            # Detectar eventos de aceleração: a > 2 m/s² por ≥ 3 amostras (0.3 s a 10 Hz)
                            _qa_eventos = []
                            _qa_in_event = False
                            _qa_start = 0
                            _qa_THRESH = 2.0
                            _qa_MIN_DUR = 3  # amostras
                            for _qi in range(len(_qa_as)):
                                if not _qa_in_event and _qa_as[_qi] >= _qa_THRESH:
                                    _qa_in_event = True
                                    _qa_start = _qi
                                elif _qa_in_event and (_qa_as[_qi] < _qa_THRESH or _qi == len(_qa_as)-1):
                                    _qa_end = _qi
                                    if _qa_end - _qa_start >= _qa_MIN_DUR:
                                        _seg_a = _qa_as[_qa_start:_qa_end]
                                        _seg_v = _qa_vs[_qa_start:_qa_end]
                                        _seg_t = _qa_ts[_qa_start:_qa_end]
                                        _qa_eventos.append({
                                            'inicio_min': (_seg_t[0] - _qa_ts[0]) / 60,
                                            'pico_a': float(np.max(_seg_a)),
                                            'impulso': float(np.trapezoid(_seg_a, dx=0.1)),
                                            'tdr': float((np.max(_seg_a) - _seg_a[0]) / max(0.1, (_seg_t[np.argmax(_seg_a)] - _seg_t[0]))),
                                            'vel_entrada': float(_seg_v[0]),
                                            'duracao_s': float(len(_seg_a) * 0.1),
                                        })
                                    _qa_in_event = False

                            if _qa_eventos:
                                _df_qa = pd.DataFrame(_qa_eventos)
                                _qa_kc1, _qa_kc2, _qa_kc3, _qa_kc4 = st.columns(4)
                                _qa_kc1.metric("Total de Eventos", len(_qa_eventos))
                                _qa_kc2.metric("Pico de Aceleração", f"{_df_qa['pico_a'].max():.2f} m/s²")
                                _qa_kc3.metric("Impulso Médio", f"{_df_qa['impulso'].mean():.2f} m/s")
                                _qa_kc4.metric("Vel. Entrada Média", f"{_df_qa['vel_entrada'].mean():.1f} km/h")

                                _qa_c1, _qa_c2 = st.columns(2)
                                with _qa_c1:
                                    # Scatter: tempo x pico_a colorido por impulso
                                    _fig_qa_sc = go.Figure()
                                    _fig_qa_sc.add_trace(go.Scatter(
                                        x=_df_qa['inicio_min'], y=_df_qa['pico_a'],
                                        mode='markers',
                                        marker=dict(
                                            size=_df_qa['impulso'].clip(3, 20),
                                            color=_df_qa['impulso'],
                                            colorscale='RdYlGn_r',
                                            showscale=True,
                                            colorbar=dict(title=dict(text='Impulso (m/s)', font=dict(color='white')),
                                                          tickfont=dict(color='white')),
                                        ),
                                        customdata=_df_qa[['impulso','vel_entrada','duracao_s']].values,
                                        hovertemplate=(
                                            'Min: %{x:.1f}<br>'
                                            'Pico Acc: %{y:.2f} m/s²<br>'
                                            'Impulso: %{customdata[0]:.2f} m/s<br>'
                                            'Vel entrada: %{customdata[1]:.1f} km/h<br>'
                                            'Duração: %{customdata[2]:.2f} s<extra></extra>'
                                        ),
                                    ))
                                    # linha de tendência do pico_a ao longo do jogo
                                    if len(_df_qa) >= 4:
                                        _qa_z = np.polyfit(_df_qa['inicio_min'], _df_qa['pico_a'], 1)
                                        _qa_xfit = np.linspace(_df_qa['inicio_min'].min(), _df_qa['inicio_min'].max(), 50)
                                        _fig_qa_sc.add_trace(go.Scatter(
                                            x=_qa_xfit, y=np.polyval(_qa_z, _qa_xfit),
                                            mode='lines', name='Tendência',
                                            line=dict(color='#FFD700', width=2, dash='dash'),
                                        ))
                                    _fig_qa_sc.update_layout(
                                        title=dict(text='Pico de Aceleração ao Longo do Jogo', font=dict(color='white', size=13)),
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='Pico de Aceleração (m/s²)', gridcolor='#333'),
                                        height=340, margin=dict(t=45,b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_qa_sc, use_container_width=True)

                                with _qa_c2:
                                    # Histograma de velocidade de entrada nos sprints
                                    _fig_qa_hist = go.Figure()
                                    _fig_qa_hist.add_trace(go.Histogram(
                                        x=_df_qa['vel_entrada'], nbinsx=12,
                                        marker=dict(color=cor_atleta(_qa_atl), opacity=0.8,
                                                    line=dict(color='white', width=0.5)),
                                        name='Vel. Entrada',
                                        hovertemplate='%{x:.1f}–%{x:.1f} km/h: %{y} eventos<extra></extra>',
                                    ))
                                    _fig_qa_hist.update_layout(
                                        title=dict(text='Velocidade de Entrada nos Eventos de Acc.', font=dict(color='white', size=13)),
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Velocidade (km/h)', gridcolor='#333'),
                                        yaxis=dict(title='Nº de Eventos', gridcolor='#333'),
                                        height=340, margin=dict(t=45,b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_qa_hist, use_container_width=True)
                            else:
                                st.info("Nenhum evento de aceleração >2 m/s² detectado para este atleta/período.")

                        st.markdown("---")

                        # ── Qualidade dos Eventos de Desaceleração ────────────
                        st.markdown("### 🔴 Qualidade de Desaceleração")
                        st.caption(
                            "Cada evento de desaceleração (cruzar −2 m/s² por ≥0.3 s) é "
                            "caracterizado por: pico, impulso (área sob a curva), velocidade "
                            "de entrada e duração. Mostra SE os esforços de frenagem mantêm "
                            "qualidade ao longo do jogo."
                        )
                        _qd_atl = st.selectbox("Atleta:", _av_atletas_disp, key="qd_atleta")
                        if _av_per == _AV_TODOS:
                            _qd_pts: list = []
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _qd_pts += _pv.get(_qd_atl, [])
                        else:
                            _qd_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_qd_atl, [])

                        _qd_vs = np.array([float(p.get('v') or 0) * 3.6 for p in _qd_pts])
                        _qd_as = np.array([float(p.get('a') or 0) for p in _qd_pts])
                        _qd_ts = np.array([float(p.get('ts') or 0) for p in _qd_pts])

                        if len(_qd_as) > 10:
                            # Detectar eventos de desaceleração: a < -2 m/s² por ≥ 3 amostras
                            _qd_eventos = []
                            _qd_in_event = False
                            _qd_start = 0
                            _qd_THRESH = 2.0   # limiar absoluto
                            _qd_MIN_DUR = 3    # amostras mínimas
                            for _qi in range(len(_qd_as)):
                                if not _qd_in_event and _qd_as[_qi] <= -_qd_THRESH:
                                    _qd_in_event = True
                                    _qd_start = _qi
                                elif _qd_in_event and (_qd_as[_qi] > -_qd_THRESH or _qi == len(_qd_as) - 1):
                                    _qd_end = _qi
                                    if _qd_end - _qd_start >= _qd_MIN_DUR:
                                        _seg_ad = _qd_as[_qd_start:_qd_end]
                                        _seg_vd = _qd_vs[_qd_start:_qd_end]
                                        _seg_td = _qd_ts[_qd_start:_qd_end]
                                        _qd_eventos.append({
                                            'inicio_min': (_seg_td[0] - _qd_ts[0]) / 60,
                                            'pico_d': float(abs(np.min(_seg_ad))),   # valor absoluto do pico
                                            'impulso': float(np.trapezoid(np.abs(_seg_ad), dx=0.1)),
                                            'vel_entrada': float(_seg_vd[0]),
                                            'duracao_s': float(len(_seg_ad) * 0.1),
                                        })
                                    _qd_in_event = False

                            if _qd_eventos:
                                _df_qd = pd.DataFrame(_qd_eventos)
                                _qd_kc1, _qd_kc2, _qd_kc3, _qd_kc4 = st.columns(4)
                                _qd_kc1.metric("Total de Eventos",      len(_qd_eventos))
                                _qd_kc2.metric("Pico de Desaceleração", f"{_df_qd['pico_d'].max():.2f} m/s²")
                                _qd_kc3.metric("Impulso Médio",         f"{_df_qd['impulso'].mean():.2f} m/s")
                                _qd_kc4.metric("Vel. Entrada Média",    f"{_df_qd['vel_entrada'].mean():.1f} km/h")

                                _qd_c1, _qd_c2 = st.columns(2)
                                with _qd_c1:
                                    # Scatter: tempo × pico_d colorido por impulso
                                    _fig_qd_sc = go.Figure()
                                    _fig_qd_sc.add_trace(go.Scatter(
                                        x=_df_qd['inicio_min'], y=_df_qd['pico_d'],
                                        mode='markers',
                                        marker=dict(
                                            size=_df_qd['impulso'].clip(3, 20),
                                            color=_df_qd['impulso'],
                                            colorscale='RdYlGn_r',
                                            showscale=True,
                                            colorbar=dict(
                                                title=dict(text='Impulso (m/s)', font=dict(color='white')),
                                                tickfont=dict(color='white'),
                                            ),
                                        ),
                                        customdata=_df_qd[['impulso', 'vel_entrada', 'duracao_s']].values,
                                        hovertemplate=(
                                            'Min: %{x:.1f}<br>'
                                            'Pico Dec: %{y:.2f} m/s²<br>'
                                            'Impulso: %{customdata[0]:.2f} m/s<br>'
                                            'Vel entrada: %{customdata[1]:.1f} km/h<br>'
                                            'Duração: %{customdata[2]:.2f} s<extra></extra>'
                                        ),
                                    ))
                                    # Linha de tendência
                                    if len(_df_qd) >= 4:
                                        _qd_z = np.polyfit(_df_qd['inicio_min'], _df_qd['pico_d'], 1)
                                        _qd_xfit = np.linspace(_df_qd['inicio_min'].min(),
                                                               _df_qd['inicio_min'].max(), 50)
                                        _fig_qd_sc.add_trace(go.Scatter(
                                            x=_qd_xfit, y=np.polyval(_qd_z, _qd_xfit),
                                            mode='lines', name='Tendência',
                                            line=dict(color='#FFD700', width=2, dash='dash'),
                                        ))
                                    _fig_qd_sc.update_layout(
                                        title=dict(text='Pico de Desaceleração ao Longo do Jogo',
                                                   font=dict(color='white', size=13)),
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='Pico de Desaceleração (m/s²)', gridcolor='#333'),
                                        height=340, margin=dict(t=45, b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_qd_sc, use_container_width=True)

                                with _qd_c2:
                                    # Histograma de velocidade de entrada
                                    _fig_qd_hist = go.Figure()
                                    _fig_qd_hist.add_trace(go.Histogram(
                                        x=_df_qd['vel_entrada'], nbinsx=12,
                                        marker=dict(color=cor_atleta(_qd_atl), opacity=0.8,
                                                    line=dict(color='white', width=0.5)),
                                        name='Vel. Entrada',
                                        hovertemplate='%{x:.1f} km/h: %{y} eventos<extra></extra>',
                                    ))
                                    _fig_qd_hist.update_layout(
                                        title=dict(text='Velocidade de Entrada nos Eventos de Dec.',
                                                   font=dict(color='white', size=13)),
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Velocidade (km/h)', gridcolor='#333'),
                                        yaxis=dict(title='Nº de Eventos', gridcolor='#333'),
                                        height=340, margin=dict(t=45, b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_qd_hist, use_container_width=True)
                            else:
                                st.info("Nenhum evento de desaceleração <−2 m/s² detectado para este atleta/período.")

                        st.markdown("---")

                        # ── #11: Player Load Direcional ───────────────────────
                        st.markdown("### 📐 Player Load Direcional")
                        st.caption(
                            "Decompõe o PlayerLoad nos 3 eixos: **Anteroposterior** (frente/trás — corrida "
                            "em linha), **Mediolateral** (esquerda/direita — movimentos defensivos/laterais) "
                            "e **Vertical** (saltos, duelos aéreos, contatos). "
                            "Parâmetros: `pla`, `plml`, `plv` do sensor Catapult."
                        )
                        _pld_atl = st.selectbox("Atleta:", _av_atletas_disp, key="pld_atleta")
                        if _av_per == _AV_TODOS:
                            _pld_pts: list = []
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _pld_pts += _pv.get(_pld_atl, [])
                        else:
                            _pld_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_pld_atl, [])

                        _pld_ap  = sum(float(p.get('pla')  or 0) for p in _pld_pts)
                        _pld_ml  = sum(float(p.get('plml') or 0) for p in _pld_pts)
                        _pld_vt  = sum(float(p.get('plv')  or 0) for p in _pld_pts)
                        _pld_tot = _pld_ap + _pld_ml + _pld_vt

                        # fallback: estimar AP/ML/V a partir de aceleração se sensor não tiver eixos
                        if _pld_tot < 0.01 and _pld_pts:
                            _est_note = True
                            _ap_sum = _ml_sum = _vt_sum = 0.0
                            for _pp in _pld_pts:
                                _av_vel = abs(float(_pp.get('v') or 0))
                                _av_acc = abs(float(_pp.get('a') or 0))
                                # heurística: se velocidade alta → contribuição AP; se baixo → ML
                                _ap_sum += _av_vel * 0.1
                                _ml_sum += max(0, _av_acc - _av_vel * 0.05) * 0.1
                            _pld_ap  = _ap_sum
                            _pld_ml  = _ml_sum
                            _pld_vt  = max(0.0, _pld_ap * 0.15)
                            _pld_tot = _pld_ap + _pld_ml + _pld_vt
                        else:
                            _est_note = False

                        if _pld_tot > 0:
                            if _est_note:
                                st.caption("⚠️ Sensor sem dados de eixo — estimativa heurística.")
                            _pld_c1, _pld_c2 = st.columns(2)
                            with _pld_c1:
                                # KPIs
                                _p1, _p2, _p3 = st.columns(3)
                                _p1.metric("Anteroposterior", f"{_pld_ap:.1f}", f"{_pld_ap/_pld_tot*100:.0f}%")
                                _p2.metric("Mediolateral",   f"{_pld_ml:.1f}", f"{_pld_ml/_pld_tot*100:.0f}%")
                                _p3.metric("Vertical",       f"{_pld_vt:.1f}", f"{_pld_vt/_pld_tot*100:.0f}%")
                                # Pizza
                                _fig_pld_pie = go.Figure(go.Pie(
                                    labels=['Anteroposterior', 'Mediolateral', 'Vertical'],
                                    values=[round(_pld_ap,1), round(_pld_ml,1), round(_pld_vt,1)],
                                    marker=dict(colors=['#2196F3','#4CAF50','#FF9800']),
                                    textinfo='label+percent',
                                    hole=0.42,
                                    hovertemplate='%{label}<br>PL: %{value:.1f}<br>%{percent}<extra></extra>',
                                ))
                                _fig_pld_pie.update_layout(
                                    title=dict(text=f'Distribuição de PL — {_pld_atl}', font=dict(color='white', size=13)),
                                    paper_bgcolor='#0e1117', font=dict(color='white'),
                                    legend=dict(font=dict(color='white', size=10)),
                                    height=320, margin=dict(t=50,b=10,l=10,r=10),
                                )
                                st.plotly_chart(_fig_pld_pie, use_container_width=True)

                            with _pld_c2:
                                # Comparativo entre todos os atletas
                                _pld_rows = []
                                for _pa in _av_atletas_disp:
                                    if _av_per == _AV_TODOS:
                                        _pa_pts: list = []
                                        for _ppv in dados_sensor_por_atleta_por_periodo.values():
                                            _pa_pts += _ppv.get(_pa, [])
                                    else:
                                        _pa_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_pa, [])
                                    _a = sum(float(p.get('pla') or 0)  for p in _pa_pts)
                                    _m = sum(float(p.get('plml') or 0) for p in _pa_pts)
                                    _v = sum(float(p.get('plv') or 0)  for p in _pa_pts)
                                    _t = _a+_m+_v
                                    if _t < 0.01 and _pa_pts:
                                        _a = sum(abs(float(p.get('v') or 0))*0.1 for p in _pa_pts)
                                        _m = sum(max(0,abs(float(p.get('a') or 0))-abs(float(p.get('v') or 0))*0.05)*0.1 for p in _pa_pts)
                                        _v = _a*0.15; _t = _a+_m+_v
                                    if _t > 0:
                                        _pld_rows.append({'Atleta':_pa,'AP':round(_a/_t*100,1),'ML':round(_m/_t*100,1),'VT':round(_v/_t*100,1)})
                                if _pld_rows:
                                    _df_pld = pd.DataFrame(_pld_rows)
                                    _fig_pld_bar = go.Figure()
                                    for _col_lbl, _col_color in [('AP','#2196F3'),('ML','#4CAF50'),('VT','#FF9800')]:
                                        _fig_pld_bar.add_trace(go.Bar(
                                            x=_df_pld['Atleta'], y=_df_pld[_col_lbl],
                                            name=_col_lbl, marker_color=_col_color,
                                            hovertemplate=f'{_col_lbl}: %{{y:.1f}}%<extra></extra>',
                                        ))
                                    _fig_pld_bar.update_layout(
                                        title=dict(text='% PL Direcional por Atleta', font=dict(color='white',size=13)),
                                        barmode='stack', paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(gridcolor='#333', tickangle=-30),
                                        yaxis=dict(title='% PL', gridcolor='#333'),
                                        legend=dict(font=dict(color='white')),
                                        height=320, margin=dict(t=50,b=10),
                                    )
                                    st.plotly_chart(_fig_pld_bar, use_container_width=True)
                        else:
                            st.info("Dados de PL direcional não disponíveis para este atleta/período.")
                else:
                    st.info("Busque os dados de atletas na sessão antes de usar esta análise.")
