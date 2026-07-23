# -*- coding: utf-8 -*-
"""Aba extraída de main() (P4 — page-decomposition)."""
from __future__ import annotations

from analysis import criar_grafico_aceleracao_tempo
from analysis import criar_grafico_velocidade_tempo
from analysis import get_min_dur_s
import plotly.graph_objects as go
import numpy as np
import streamlit as st


def render_esforcos(_SENSOR_HZ, dados_sensor_por_atleta_por_periodo):
        st.subheader("⏱️ Esforços ao Longo do Tempo")

        if dados_sensor_por_atleta_por_periodo:
            _ESF_TODOS = "🔀 Todos os períodos (combinado)"
            _esf_opcoes = [_ESF_TODOS] + list(dados_sensor_por_atleta_por_periodo.keys())
            periodo_escolhido = st.selectbox("Selecione o período:", _esf_opcoes, key="periodo_esforcos")
            _esf_modo_todos = (periodo_escolhido == _ESF_TODOS)

            if _esf_modo_todos:
                _esf_ats_set = set()
                for _pv in dados_sensor_por_atleta_por_periodo.values():
                    _esf_ats_set.update(_pv.keys())
                _esf_atletas = sorted(_esf_ats_set)
            else:
                _esf_atletas = list(dados_sensor_por_atleta_por_periodo.get(periodo_escolhido, {}).keys())

            if _esf_atletas:
                atleta_escolhido = st.selectbox("Selecione o atleta:", _esf_atletas, key="atleta_esforcos")
                if _esf_modo_todos:
                    sensor_points = []
                    for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                        sensor_points += _pv2.get(atleta_escolhido, [])
                    st.caption(
                        f"📊 Combinando **{len(dados_sensor_por_atleta_por_periodo)} períodos** "
                        f"→ {len(sensor_points):,} amostras para **{atleta_escolhido}**."
                    )
                else:
                    sensor_points = dados_sensor_por_atleta_por_periodo[periodo_escolhido].get(atleta_escolhido, [])

                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    mostrar_tendencia = st.checkbox("Mostrar linha de tendência", value=True)
                    window_size = st.slider("Janela de suavização:", 5, 101, 31, step=2)
                with col_config2:
                    usar_filtro = st.checkbox("Filtrar por intensidade", value=False)
                    if usar_filtro:
                        intensidade_min = st.slider("Intensidade mínima (km/h):", 0.0, 30.0, 5.0, 0.5)
                    else:
                        intensidade_min = None

                _dur_s_aba3 = get_min_dur_s()
                st.caption(
                    f"⚙️ Duração mínima de acc/dec: **{_dur_s_aba3:.1f} s** "
                    f"({max(1, round(_dur_s_aba3 * _SENSOR_HZ))} frames) — "
                    "ajuste na sidebar."
                )
                st.markdown("### 🏃‍♂️ Velocidade ao Longo do Tempo")
                fig_vel = criar_grafico_velocidade_tempo(sensor_points, atleta_escolhido, window_size, mostrar_tendencia, intensidade_min)
                if fig_vel:
                    st.plotly_chart(fig_vel, use_container_width=True)

                st.markdown("### 🔄 Aceleração ao Longo do Tempo")
                fig_acc = criar_grafico_aceleracao_tempo(sensor_points, atleta_escolhido, window_size, mostrar_tendencia, intensidade_min if usar_filtro else None)
                if fig_acc:
                    st.plotly_chart(fig_acc, use_container_width=True)

                # ── Metabolic Power chart (FEATURE 3) ───────────────────
                st.markdown("---")
                _mp_vals_esf = [
                    float(p['mp'])
                    for p in sensor_points
                    if p.get('mp') and float(p.get('mp') or 0) > 0
                ]
                if _mp_vals_esf:
                    st.markdown("### ⚡ Potência Metabólica ao Longo do Tempo")
                    _ts0_mp = float(sensor_points[0].get('ts') or 0)
                    _mp_ts = [
                        (float(p.get('ts') or 0) - _ts0_mp)
                        for p in sensor_points
                        if p.get('mp') and float(p.get('mp') or 0) > 0
                    ]
                    _fig_mp = go.Figure()
                    _fig_mp.add_trace(go.Scatter(
                        x=_mp_ts, y=_mp_vals_esf,
                        mode='lines',
                        name='MP (W/kg)',
                        line=dict(color='#FF9800', width=1.5),
                        hovertemplate='%{x:.0f}s — %{y:.1f} W/kg<extra></extra>',
                    ))
                    _fig_mp.add_hline(y=20, line=dict(color='#F44336', dash='dash'),
                                      annotation_text='20 W/kg', annotation_font=dict(color='#F44336', size=9))
                    _fig_mp.add_hline(y=25, line=dict(color='#9C27B0', dash='dash'),
                                      annotation_text='25 W/kg', annotation_font=dict(color='#9C27B0', size=9))
                    _fig_mp.update_layout(
                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        xaxis=dict(title='Tempo (s)', gridcolor='#333'),
                        yaxis=dict(title='MP (W/kg)', gridcolor='#333'),
                        height=320,
                    )
                    st.plotly_chart(_fig_mp, use_container_width=True)
                    _mp_above20 = sum(1 for v in _mp_vals_esf if v > 20) / max(1, len(_mp_vals_esf)) * 100
                    _mp_above25_s = sum(1 for v in _mp_vals_esf if v > 25) * 0.1
                    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                    _mc1.metric("MP Médio (W/kg)", f"{float(np.mean(_mp_vals_esf)):.1f}")
                    _mc2.metric("MP Máx (W/kg)", f"{float(np.max(_mp_vals_esf)):.1f}")
                    _mc3.metric("MP > 20 W/kg (%)", f"{_mp_above20:.1f}%")
                    _mc4.metric("Tempo > 25 W/kg (s)", f"{_mp_above25_s:.0f}s")

        else:
            st.info("Dados de sensor não disponíveis")


        # ── FEATURE 8: PDF Export — REMOVIDO ──────────────────────────
