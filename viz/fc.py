# -*- coding: utf-8 -*-
"""Aba extraída de main() (P4 — page-decomposition)."""
from __future__ import annotations

from config import _CHAVE_COMBINADO
from field import cor_atleta
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render_fc(dados_sensor_por_atleta_por_periodo, resultados_por_periodo):
                st.subheader("❤️ Frequência Cardíaca & TRIMP")
                st.caption(
                    "**TRIMP** (Training Impulse — Edwards): carga interna calculada pelo tempo em cada zona de "
                    "FC (50–60 / 60–70 / 70–80 / 80–90 / 90–100% HRmax) com multiplicadores 1–2–3–4–5. "
                    "Referência: FC máx = 220 − idade (padrão 180 bpm se não configurada)."
                )

                _fc_periodos = list(dados_sensor_por_atleta_por_periodo.keys())
                if _fc_periodos and resultados_por_periodo:
                    _FC_TODOS = "🔀 Todos os períodos (combinado)"
                    _fc_opcoes_per = [_FC_TODOS] + _fc_periodos

                    _fc_c1, _fc_c2, _fc_c3 = st.columns([2, 1, 1])
                    with _fc_c1:
                        _fc_per = st.selectbox("Período:", _fc_opcoes_per, key="fc_periodo")
                    with _fc_c2:
                        if _fc_per == _FC_TODOS:
                            _fc_ats_set = set()
                            for _fpv in dados_sensor_por_atleta_por_periodo.values():
                                _fc_ats_set.update(_fpv.keys())
                            _fc_atletas_disp = sorted(_fc_ats_set)
                        else:
                            _fc_atletas_disp = list(dados_sensor_por_atleta_por_periodo.get(_fc_per, {}).keys())
                        _fc_atl = st.selectbox("Atleta:", _fc_atletas_disp, key="fc_atleta_sel") if _fc_atletas_disp else None
                    with _fc_c3:
                        _fc_hrmax = st.number_input(
                            "FC Máx (bpm):", min_value=150, max_value=220,
                            value=180, step=1, key="fc_hrmax",
                            help="Padrão: 180 bpm. Ajuste para o valor individual do atleta."
                        )

                    if _fc_atl:
                        # Coletar pontos de sensor
                        if _fc_per == _FC_TODOS:
                            _fc_pts: list = []
                            for _fpv in dados_sensor_por_atleta_por_periodo.values():
                                _fc_pts += _fpv.get(_fc_atl, [])
                        else:
                            _fc_pts = dados_sensor_por_atleta_por_periodo.get(_fc_per, {}).get(_fc_atl, [])

                        _hr_vals = [float(p['hr']) for p in _fc_pts if p.get('hr') and float(p.get('hr') or 0) > 30]

                        if _hr_vals:
                            # ── Zonas Edwards ──────────────────────────────────────
                            _fc_zona_bounds = [
                                (0.50, 0.60, 1, 'Z1  50–60%', '#4CAF50'),
                                (0.60, 0.70, 2, 'Z2  60–70%', '#8BC34A'),
                                (0.70, 0.80, 3, 'Z3  70–80%', '#FFEB3B'),
                                (0.80, 0.90, 4, 'Z4  80–90%', '#FF9800'),
                                (0.90, 1.00, 5, 'Z5  90–100%', '#F44336'),
                            ]
                            _fc_zona_counts = {z[3]: 0 for z in _fc_zona_bounds}
                            _fc_trimp_total = 0.0
                            _fc_dt_s = 0.1  # 10 Hz → 0.1 s por ponto
                            for _hv in _hr_vals:
                                _hpct = _hv / _fc_hrmax
                                for _lo, _hi, _mult, _lbl, _col in _fc_zona_bounds:
                                    if _lo <= _hpct < _hi or (_hi == 1.00 and _hpct >= _lo):
                                        _fc_zona_counts[_lbl] += 1
                                        _fc_trimp_total += _fc_dt_s / 60 * _mult
                                        break

                            # ── KPIs ─────────────────────────────────────────────
                            _fk1, _fk2, _fk3, _fk4, _fk5 = st.columns(5)
                            _fk1.metric("TRIMP Total", f"{_fc_trimp_total:.1f}",
                                        help="Training Impulse (Edwards, 1993)")
                            _fk2.metric("FC Média (bpm)", f"{np.mean(_hr_vals):.0f}")
                            _fk3.metric("FC Máx Atingida", f"{max(_hr_vals):.0f} bpm")
                            _fk4.metric("% FCmax", f"{max(_hr_vals)/_fc_hrmax*100:.1f}%")
                            _fc_tempo_acima_80 = sum(
                                1 for hv in _hr_vals if hv / _fc_hrmax >= 0.80
                            ) * _fc_dt_s / 60
                            _fk5.metric("Tempo > 80% FCmax", f"{_fc_tempo_acima_80:.1f} min")

                            st.markdown("---")
                            _fc_chart_c1, _fc_chart_c2 = st.columns(2)

                            # ── Pizza — distribuição de zonas ─────────────────────
                            with _fc_chart_c1:
                                _fz_lbls = [z[3] for z in _fc_zona_bounds]
                                _fz_vals = [_fc_zona_counts[l] * _fc_dt_s / 60 for l in _fz_lbls]
                                _fz_cols = [z[4] for z in _fc_zona_bounds]
                                _fig_fz = go.Figure(go.Pie(
                                    labels=_fz_lbls, values=_fz_vals,
                                    marker=dict(colors=_fz_cols),
                                    textinfo='label+percent',
                                    hovertemplate='%{label}<br>%{value:.1f} min<extra></extra>',
                                    hole=0.38,
                                ))
                                _fig_fz.update_layout(
                                    title=dict(text=f'Distribuição de Zonas de FC — {_fc_atl}',
                                               font=dict(color='white', size=13)),
                                    paper_bgcolor='#0e1117', font=dict(color='white'),
                                    legend=dict(font=dict(color='white', size=9)),
                                    height=340, margin=dict(t=50, b=10, l=10, r=10),
                                )
                                st.plotly_chart(_fig_fz, use_container_width=True)

                            # ── Barras empilhadas TRIMP por zona ─────────────────
                            with _fc_chart_c2:
                                _fz_trimp = [_fc_zona_counts[l] * _fc_dt_s / 60 * _mult
                                             for l, (_, _, _mult, _, _) in zip(_fz_lbls, _fc_zona_bounds)]
                                _fig_trimp_fc = go.Figure()
                                for _fl, _ft, _fc_col in zip(_fz_lbls, _fz_trimp, _fz_cols):
                                    _fig_trimp_fc.add_trace(go.Bar(
                                        x=[_fc_atl], y=[_ft], name=_fl,
                                        marker_color=_fc_col,
                                        hovertemplate=f'{_fl}: %{{y:.1f}} TRIMP<extra></extra>',
                                    ))
                                _fig_trimp_fc.update_layout(
                                    title=dict(text=f'TRIMP por Zona — {_fc_atl}',
                                               font=dict(color='white', size=13)),
                                    barmode='stack',
                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(title='TRIMP (u.a.)', gridcolor='#333'),
                                    legend=dict(font=dict(color='white', size=9)),
                                    height=340, margin=dict(t=50, b=10, l=10, r=10),
                                )
                                st.plotly_chart(_fig_trimp_fc, use_container_width=True)

                            # ── Curva de FC ao longo do tempo ────────────────────
                            with st.expander("📈 Curva de FC ao longo do tempo", expanded=True):
                                _fc_ts = [float(p.get('ts') or 0) for p in _fc_pts
                                          if p.get('hr') and float(p.get('hr') or 0) > 30]
                                if _fc_ts:
                                    _fc_t0 = _fc_ts[0]
                                    _fc_ts_rel = [(t - _fc_t0) / 60 for t in _fc_ts]
                                    _fig_fc_curve = go.Figure()
                                    _fig_fc_curve.add_trace(go.Scatter(
                                        x=_fc_ts_rel, y=_hr_vals,
                                        mode='lines', line=dict(color='#F44336', width=1.5),
                                        name='FC (bpm)',
                                        hovertemplate='%{x:.1f} min — %{y:.0f} bpm<extra></extra>',
                                    ))
                                    # Linhas de zona
                                    for _lo, _hi, _mult, _lbl, _fcol in _fc_zona_bounds:
                                        _fig_fc_curve.add_hline(
                                            y=_lo * _fc_hrmax,
                                            line=dict(color=_fcol, width=1, dash='dot'),
                                            annotation_text=_lbl,
                                            annotation_font=dict(color=_fcol, size=9),
                                        )
                                    _fig_fc_curve.update_layout(
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='FC (bpm)', gridcolor='#333',
                                                   range=[40, _fc_hrmax * 1.05]),
                                        height=320, margin=dict(t=20, b=10),
                                        showlegend=False,
                                    )
                                    st.plotly_chart(_fig_fc_curve, use_container_width=True)

                            # ── Comparativo TRIMP entre todos os atletas ──────────
                            st.markdown("---")
                            st.markdown("### 📊 Comparativo de TRIMP — Todos os Atletas")
                            _fc_comp_rows = []
                            for _fc_cp_atl in _fc_atletas_disp:
                                if _fc_per == _FC_TODOS:
                                    _fc_cp_pts: list = []
                                    for _fpv in dados_sensor_por_atleta_por_periodo.values():
                                        _fc_cp_pts += _fpv.get(_fc_cp_atl, [])
                                else:
                                    _fc_cp_pts = dados_sensor_por_atleta_por_periodo.get(_fc_per, {}).get(_fc_cp_atl, [])
                                _fc_cp_hr = [float(p['hr']) for p in _fc_cp_pts if p.get('hr') and float(p.get('hr') or 0) > 30]
                                if _fc_cp_hr:
                                    _fc_cp_trimp = 0.0
                                    _fc_cp_z = {z[3]: 0 for z in _fc_zona_bounds}
                                    for _hv2 in _fc_cp_hr:
                                        _hp2 = _hv2 / _fc_hrmax
                                        for _lo2, _hi2, _m2, _l2, _c2 in _fc_zona_bounds:
                                            if _lo2 <= _hp2 < _hi2 or (_hi2 == 1.00 and _hp2 >= _lo2):
                                                _fc_cp_z[_l2] += 1
                                                _fc_cp_trimp += _fc_dt_s / 60 * _m2
                                                break
                                    _fc_comp_rows.append({
                                        'Atleta': _fc_cp_atl,
                                        'TRIMP': round(_fc_cp_trimp, 1),
                                        'FC Média (bpm)': round(float(np.mean(_fc_cp_hr)), 0),
                                        'FC Máx (bpm)': round(float(max(_fc_cp_hr)), 0),
                                        **{l: round(_fc_cp_z[l] * _fc_dt_s / 60, 2) for l in _fc_cp_z}
                                    })
                            if _fc_comp_rows:
                                _df_fc_comp = pd.DataFrame(_fc_comp_rows).sort_values('TRIMP', ascending=False)
                                st.dataframe(_df_fc_comp, use_container_width=True, hide_index=True)
                                # Gráfico de barras TRIMP comparativo
                                _fig_fc_comp_bar = px.bar(
                                    _df_fc_comp, x='Atleta', y='TRIMP',
                                    color='TRIMP', color_continuous_scale='Reds',
                                    title='TRIMP por Atleta',
                                )
                                _fig_fc_comp_bar.update_layout(
                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(title='TRIMP (u.a.)', gridcolor='#333'),
                                    height=350, margin=dict(t=40, b=10),
                                    showlegend=False,
                                )
                                st.plotly_chart(_fig_fc_comp_bar, use_container_width=True)
                        else:
                            st.info("ℹ️ Nenhum dado de FC disponível para este atleta/período.")
                    else:
                        st.info("Selecione um atleta para analisar a FC.")
                else:
                    st.info("Carregue dados da sessão para visualizar a análise de FC.")

                # ── #3: Curva de Fadiga Intra-Jogo ───────────────────────────
                st.markdown("---")
                st.markdown("## 📉 Curva de Fadiga Intra-Jogo")
                st.caption(
                    "Divide cada período em janelas de **5 minutos** e calcula a intensidade "
                    "relativa (m/min e PL/min). Uma linha de tendência negativa indica fadiga "
                    "progressiva. O minuto de início da queda é detectado automaticamente."
                )
                _fat_periodos = [p for p in dados_sensor_por_atleta_por_periodo if p != _CHAVE_COMBINADO]
                if _fat_periodos and resultados_por_periodo:
                    _fat_per = st.selectbox("Período:", _fat_periodos, key="fat_periodo")
                    _fat_atls_disp = list(dados_sensor_por_atleta_por_periodo.get(_fat_per, {}).keys())
                    if _fat_atls_disp:
                        _fat_sel = st.multiselect(
                            "Atletas (até 8):", _fat_atls_disp,
                            default=_fat_atls_disp[:min(5, len(_fat_atls_disp))],
                            key="fat_atletas",
                        )
                        _fat_janela_min = st.select_slider(
                            "Janela:", options=[2, 3, 5, 10], value=3, key="fat_janela",
                            help="Tamanho da janela temporal para cálculo de intensidade"
                        )
                        _fat_janela_s = _fat_janela_min * 60

                        if _fat_sel:
                            _fig_fat = go.Figure()
                            _fat_declive_info = []

                            for _fa in _fat_sel:
                                _fa_pts = dados_sensor_por_atleta_por_periodo.get(_fat_per, {}).get(_fa, [])
                                if not _fa_pts:
                                    continue
                                _fa_ts = np.array([float(p.get('ts') or 0) for p in _fa_pts])
                                _fa_vs = np.array([float(p.get('v')  or 0) * 3.6 for p in _fa_pts])
                                _fa_pl = np.array([float(p.get('pl') or 0) for p in _fa_pts])
                                if len(_fa_ts) < 2:
                                    continue
                                _fa_t0 = _fa_ts[0]
                                _fa_dur = _fa_ts[-1] - _fa_t0
                                if _fa_dur < _fat_janela_s * 2:
                                    continue

                                # janelas deslizantes
                                _win_centers_min = []
                                _win_mmin = []
                                _win_plmin = []
                                _t_start = _fa_t0
                                while _t_start + _fat_janela_s <= _fa_ts[-1]:
                                    _mask = (_fa_ts >= _t_start) & (_fa_ts < _t_start + _fat_janela_s)
                                    if _mask.sum() > 1:
                                        _dist_win = float(np.sum(np.abs(np.diff(_fa_vs[_mask])) * 0.1 / 3.6 * 3.6))
                                        # distância real via integral v×dt
                                        _d_real = float(np.trapezoid(_fa_vs[_mask] / 3.6, _fa_ts[_mask]))
                                        _pl_sum = float(np.sum(_fa_pl[_mask]))
                                        _win_mmin.append(_d_real / _fat_janela_min)
                                        _win_plmin.append(_pl_sum / _fat_janela_min)
                                        _win_centers_min.append((_t_start - _fa_t0 + _fat_janela_s / 2) / 60)
                                    _t_start += _fat_janela_s / 2  # sobreposição 50%

                                if len(_win_centers_min) < 3:
                                    continue

                                _wc = np.array(_win_centers_min)
                                _wm = np.array(_win_mmin)

                                _fa_color = cor_atleta(_fa)
                                _fig_fat.add_trace(go.Scatter(
                                    x=_wc, y=_wm, mode='lines+markers',
                                    name=_fa,
                                    line=dict(color=_fa_color, width=2),
                                    marker=dict(size=5, color=_fa_color),
                                    hovertemplate='%{x:.1f} min — %{y:.1f} m/min<extra>' + _fa + '</extra>',
                                ))
                                # Linha de tendência
                                if len(_wc) >= 4:
                                    _z = np.polyfit(_wc, _wm, 1)
                                    _xfit = np.linspace(_wc[0], _wc[-1], 50)
                                    _fig_fat.add_trace(go.Scatter(
                                        x=_xfit, y=np.polyval(_z, _xfit),
                                        mode='lines', showlegend=False,
                                        line=dict(color=_fa_color, width=1, dash='dot'),
                                        hoverinfo='skip',
                                    ))
                                    # Detectar início de queda: janela com variação negativa sustentada
                                    _rolling_diff = np.diff(_wm)
                                    _neg_idx = np.where(_rolling_diff < -5)[0]
                                    if len(_neg_idx) > 0:
                                        _queda_min = _wc[_neg_idx[0]]
                                        _fat_declive_info.append((_fa, round(float(_z[0]),2), round(_queda_min,1)))

                            _fig_fat.update_layout(
                                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                font=dict(color='white'),
                                xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                yaxis=dict(title='Intensidade (m/min)', gridcolor='#333'),
                                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.4)'),
                                height=380, margin=dict(t=20,b=10),
                            )
                            st.plotly_chart(_fig_fat, use_container_width=True)

                            # Tabela de declive e início de queda
                            if _fat_declive_info:
                                st.markdown("#### ⚠️ Atletas com Queda Detectada")
                                _df_fat_info = pd.DataFrame(
                                    _fat_declive_info,
                                    columns=['Atleta', 'Declive (m/min por min)', 'Início de Queda (min)']
                                ).sort_values('Declive (m/min por min)')
                                # badge de intensidade do declive
                                def _fat_badge(slope):
                                    if slope < -3:   return "🔴 Queda Severa"
                                    elif slope < -1: return "🟡 Queda Moderada"
                                    elif slope < 0:  return "🟢 Queda Leve"
                                    else:            return "✅ Estável"
                                _df_fat_info['Classificação'] = _df_fat_info['Declive (m/min por min)'].apply(_fat_badge)
                                st.dataframe(_df_fat_info, use_container_width=True, hide_index=True)
                    else:
                        st.info("Nenhum atleta com dados de sensor para este período.")
                else:
                    st.info("Carregue dados para visualizar a curva de fadiga.")
