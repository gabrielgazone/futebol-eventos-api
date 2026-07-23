# -*- coding: utf-8 -*-
"""Aba extraída de main() (P4 — page-decomposition)."""
from __future__ import annotations

from field import _get_pos_grupo
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import streamlit as st


def render_por_posicao(_POSICAO_COR_LEGENDA, resultados_por_periodo):
                st.subheader("📊 Análise por Posição")
                st.caption("Média das métricas agrupada por posição tática dos atletas.")

                _pos_periodos = list(resultados_por_periodo.keys())
                if _pos_periodos:
                    _pos_periodo_sel = st.selectbox("Período:", _pos_periodos, key="pos_periodo_sel")

                    if resultados_por_periodo.get(_pos_periodo_sel):
                        _df_pos_raw = pd.DataFrame(resultados_por_periodo[_pos_periodo_sel])

                        if 'Posição' not in _df_pos_raw.columns or _df_pos_raw['Posição'].replace('', np.nan).isna().all():
                            st.warning("⚠️ Dados de posição não disponíveis. Verifique se as posições foram cadastradas na API Catapult.")
                        else:
                            _df_pos_raw = _df_pos_raw[_df_pos_raw['Posição'].notna() & (_df_pos_raw['Posição'] != '')]
                            _df_pos_grp = _df_pos_raw.groupby('Posição', as_index=False).mean(numeric_only=True)

                            # Paleta de cores por posição (usa sistema de grupos táticos)
                            _cores_pos = [
                                _POSICAO_COR_LEGENDA.get(_get_pos_grupo(p)[0], '#546E7A')
                                for p in _df_pos_grp['Posição']
                            ]

                            def _bar_pos(y_col, title, suffix=''):
                                if y_col not in _df_pos_grp.columns:
                                    return None
                                _f = go.Figure(go.Bar(
                                    x=_df_pos_grp['Posição'], y=_df_pos_grp[y_col],
                                    marker_color=_cores_pos, text=_df_pos_grp[y_col].round(1),
                                    textposition='outside',
                                    hovertemplate=f'%{{x}}<br>{y_col}: %{{y:.1f}}{suffix}<extra></extra>',
                                ))
                                _f.update_layout(
                                    title=dict(text=title, font=dict(color='white', size=13)),
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(gridcolor='#333', title=y_col),
                                    height=320, margin=dict(t=45, b=10, l=10, r=10),
                                    showlegend=False,
                                )
                                return _f

                            # Linha 1: Distância Total + M/min
                            _c1, _c2 = st.columns(2)
                            with _c1:
                                _f = _bar_pos('Distância (m)', 'Média de Distância Total por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c2:
                                _f = _bar_pos('M/min', 'Média de M/min por Posição', ' m/min')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # Linha 2: Zona 4 + Zona 5 (>24 km/h)
                            _c3, _c4 = st.columns(2)
                            with _c3:
                                _f = _bar_pos('Dist. 19-24 km/h (m)', 'Média de Zona 4 (19-24 km/h) por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c4:
                                _f = _bar_pos('Dist. > 24 km/h (m)', 'Média de Zona 5 (>24 km/h) por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # Linha 3: HSR (>19) + Sprints
                            _c5, _c6 = st.columns(2)
                            with _c5:
                                _f = _bar_pos('Dist. > 19 km/h (m)', 'Média de HSR Distance (>19 km/h) por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c6:
                                _f = _bar_pos('Sprints (>24 km/h)', 'Média de Sprints por Posição')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # Linha 4: Acc 2-3 + Dcc 2-3 (gráfico agrupado)
                            _acc23_col = 'Acc 2-3 (m/s²)'
                            _dcc23_col = 'Dcc 2-3 (m/s²)'
                            if _acc23_col in _df_pos_grp.columns and _dcc23_col in _df_pos_grp.columns:
                                _c7, _c8 = st.columns(2)
                                with _c7:
                                    _fig_acc = go.Figure()
                                    _fig_acc.add_trace(go.Bar(
                                        x=_df_pos_grp['Posição'], y=_df_pos_grp[_acc23_col],
                                        name='Acc 2-3', marker_color='#FFA000',
                                        text=_df_pos_grp[_acc23_col].round(1), textposition='outside',
                                    ))
                                    _fig_acc.add_trace(go.Bar(
                                        x=_df_pos_grp['Posição'], y=_df_pos_grp[_dcc23_col],
                                        name='Dcc 2-3', marker_color='#558B2F',
                                        text=_df_pos_grp[_dcc23_col].round(1), textposition='outside',
                                    ))
                                    _fig_acc.update_layout(
                                        title=dict(text='Média de Acc e Dcc 2-3 por Posição', font=dict(color='white', size=13)),
                                        barmode='group', plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), xaxis=dict(gridcolor='#333'),
                                        yaxis=dict(gridcolor='#333'), height=320,
                                        margin=dict(t=45, b=10, l=10, r=10),
                                        legend=dict(font=dict(color='white')),
                                    )
                                    st.plotly_chart(_fig_acc, use_container_width=True)

                                with _c8:
                                    # Acc >3 + Dcc >3 agrupados
                                    _fig_acc3 = go.Figure()
                                    if 'Acelerações (>3 m/s²)' in _df_pos_grp.columns:
                                        _fig_acc3.add_trace(go.Bar(
                                            x=_df_pos_grp['Posição'], y=_df_pos_grp['Acelerações (>3 m/s²)'],
                                            name='Acc >3', marker_color='#E53935',
                                            text=_df_pos_grp['Acelerações (>3 m/s²)'].round(1), textposition='outside',
                                        ))
                                    if 'Desacelerações (<-3 m/s²)' in _df_pos_grp.columns:
                                        _fig_acc3.add_trace(go.Bar(
                                            x=_df_pos_grp['Posição'], y=_df_pos_grp['Desacelerações (<-3 m/s²)'],
                                            name='Dcc >3', marker_color='#1565C0',
                                            text=_df_pos_grp['Desacelerações (<-3 m/s²)'].round(1), textposition='outside',
                                        ))
                                    _fig_acc3.update_layout(
                                        title=dict(text='Média de Acc e Dcc >3 por Posição', font=dict(color='white', size=13)),
                                        barmode='group', plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), xaxis=dict(gridcolor='#333'),
                                        yaxis=dict(gridcolor='#333'), height=320,
                                        margin=dict(t=45, b=10, l=10, r=10),
                                        legend=dict(font=dict(color='white')),
                                    )
                                    st.plotly_chart(_fig_acc3, use_container_width=True)

                            # Linha 5: PlayerLoad + RHIE Blocos
                            _c9, _c10 = st.columns(2)
                            with _c9:
                                _f = _bar_pos('PlayerLoad', 'Média de PlayerLoad por Posição')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c10:
                                _f = _bar_pos('RHIE Blocos', 'Média de RHIE Blocos por Posição')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # ── Item 16: Radar de Perfil por Posição ─────────────
                            st.markdown("---")
                            st.markdown("### 🕸️ Radar de Perfil Físico por Posição")
                            st.caption(
                                "Normalização min-max do grupo: 100% = maior valor do grupo. "
                                "Permite comparar perfis multidimensionais entre posições."
                            )
                            _radar_metrics = [c for c in [
                                'Distância (m)', 'M/min', 'Dist. > 19 km/h (m)',
                                'Sprints (>24 km/h)', 'Acelerações (>3 m/s²)',
                                'Desacelerações (<-3 m/s²)', 'PlayerLoad', 'RHIE Blocos',
                            ] if c in _df_pos_grp.columns]

                            if len(_radar_metrics) >= 4 and len(_df_pos_grp) >= 2:
                                # Normalização min-max por métrica (0-100%)
                                _radar_norm = _df_pos_grp[_radar_metrics].copy()
                                for _rc in _radar_metrics:
                                    _rmin = _radar_norm[_rc].min()
                                    _rmax = _radar_norm[_rc].max()
                                    _radar_norm[_rc] = ((_radar_norm[_rc] - _rmin) / max(_rmax - _rmin, 1e-9) * 100).round(1)

                                _POS_RADAR_PALETTE = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4','#F44336']
                                _fig_radar = go.Figure()
                                for _ri, (_rpos, _rrow) in enumerate(_radar_norm.iterrows()):
                                    _pos_name = _df_pos_grp['Posição'].iloc[_ri]
                                    _vals = list(_rrow[_radar_metrics]) + [_rrow[_radar_metrics[0]]]
                                    _lbls = _radar_metrics + [_radar_metrics[0]]
                                    _col_r = _POS_RADAR_PALETTE[_ri % len(_POS_RADAR_PALETTE)]
                                    _fig_radar.add_trace(go.Scatterpolar(
                                        r=_vals, theta=_lbls, fill='toself',
                                        name=_pos_name,
                                        line=dict(color=_col_r, width=2),
                                        fillcolor=(lambda c: f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)")(_col_r),
                                        opacity=0.85,
                                        hovertemplate='%{theta}: %{r:.1f}%<extra>' + _pos_name + '</extra>',
                                    ))
                                _fig_radar.update_layout(
                                    polar=dict(
                                        bgcolor='#0e1117',
                                        radialaxis=dict(
                                            visible=True, range=[0, 100],
                                            gridcolor='rgba(255,255,255,0.15)',
                                            tickfont=dict(color='rgba(200,200,200,0.7)', size=9),
                                            ticksuffix='%',
                                        ),
                                        angularaxis=dict(
                                            gridcolor='rgba(255,255,255,0.15)',
                                            tickfont=dict(color='white', size=10),
                                        ),
                                    ),
                                    paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    legend=dict(font=dict(color='white', size=10)),
                                    height=460, margin=dict(t=20, b=20, l=20, r=20),
                                )
                                st.plotly_chart(_fig_radar, use_container_width=True)
                            else:
                                st.info("Necessário ≥ 2 posições e ≥ 4 métricas para gerar o radar.")

                            # ── Tabela resumo por posição ─────────────────────────
                            st.markdown("---")
                            st.markdown("### 📋 Resumo por Posição")
                            _TD_POS_COLS = [c for c in [
                                'Posição', 'Distância (m)', 'M/min', 'Dist. 19-24 km/h (m)',
                                'Dist. > 24 km/h (m)', 'Sprints (>24 km/h)', 'Velocidade Máx (km/h)',
                                'Acc 2-3 (m/s²)', 'Dcc 2-3 (m/s²)', 'Acelerações (>3 m/s²)',
                                'Desacelerações (<-3 m/s²)', 'PlayerLoad', 'RHIE Blocos',
                            ] if c in _df_pos_grp.columns]
                            st.dataframe(
                                _df_pos_grp[_TD_POS_COLS].round(1),
                                use_container_width=True, hide_index=True
                            )

                            # ── Perfil Longitudinal /stats — removido (P8) ──
                    else:
                        st.info("Nenhum dado disponível para este período.")
                else:
                    st.info("Carregue os dados para visualizar a análise por posição.")
