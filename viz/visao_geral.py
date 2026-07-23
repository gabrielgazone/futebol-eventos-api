# -*- coding: utf-8 -*-
"""Aba extraída de main() (P4 — page-decomposition)."""
from __future__ import annotations

from config import _CHAVE_COMBINADO
from ui_theme import _hr
import plotly.graph_objects as go
import pandas as pd
import streamlit as st


def render_visao_geral(resultados_por_periodo):
                st.markdown("## 📊 Resumo da Sessão")

                if resultados_por_periodo:
                    # ── Coleta todas as linhas de todos os períodos (exceto combinado) ──
                    _ov_rows = []
                    for _p, _rs in resultados_por_periodo.items():
                        if _p == _CHAVE_COMBINADO:
                            continue
                        for _r in _rs:
                            _row = dict(_r)
                            _row['Período'] = _p
                            _ov_rows.append(_row)

                    if _ov_rows:
                        _df_ov_raw = pd.DataFrame(_ov_rows)

                        # ── Agrega por atleta — combina todos os períodos ────────────
                        # Métricas que se SOMAM entre períodos
                        _cols_sum = [c for c in [
                            'Distância (m)', 'Dist. > 19 km/h (m)', 'Dist. > 24 km/h (m)',
                            'Dist. 19-24 km/h (m)',
                            'Sprints (>24 km/h)', 'Acelerações (>3 m/s²)',
                            'Desacelerações (>-3 m/s²)', 'Desacelerações (<-3 m/s²)',
                            'RHIE Blocos', 'PlayerLoad', 'TRIMP',
                            'Acc 2-3 (m/s²)', 'Dcc 2-3 (m/s²)',
                            'Duração (min)',
                        ] if c in _df_ov_raw.columns]
                        # Métricas que se tomam o MÁXIMO entre períodos
                        _cols_max = [c for c in [
                            'Velocidade Máx (km/h)', 'Velocidade Bruta Máx (km/h)',
                            'Aceleração Máx (m/s²)', 'Acc Max (m/s²)', 'Dcc Max (m/s²)',
                            'Potência Met. Máx (W/kg)',
                        ] if c in _df_ov_raw.columns]
                        # Métricas que se tomam a MÉDIA entre períodos
                        _cols_mean = [c for c in [
                            'FC Média (bpm)', 'Velocidade Média (km/h)',
                        ] if c in _df_ov_raw.columns]
                        # Campos textuais: mantém o valor do primeiro período
                        _cols_first = [c for c in ['Posição', 'Equipe'] if c in _df_ov_raw.columns]

                        _agg_dict = {}
                        for _c in _cols_sum:  _agg_dict[_c] = 'sum'
                        for _c in _cols_max:  _agg_dict[_c] = 'max'
                        for _c in _cols_mean: _agg_dict[_c] = 'mean'
                        for _c in _cols_first: _agg_dict[_c] = 'first'

                        if _agg_dict and 'Atleta' in _df_ov_raw.columns:
                            _df_ov = (_df_ov_raw.groupby('Atleta', as_index=False)
                                      .agg(_agg_dict))
                            # Arredonda métricas numéricas
                            for _c in _cols_sum + _cols_max + _cols_mean:
                                if _c in _df_ov.columns:
                                    _df_ov[_c] = _df_ov[_c].round(1)
                        else:
                            _df_ov = _df_ov_raw.copy()

                        _n_periodos_ov = _df_ov_raw['Período'].nunique()

                        # ── KPI cards ────────────────────────────────────────────────
                        _ov_c1, _ov_c2, _ov_c3, _ov_c4, _ov_c5 = st.columns(5)
                        _ov_c1.metric("👥 Atletas", len(_df_ov['Atleta'].unique()) if 'Atleta' in _df_ov.columns else 0)
                        _ov_c2.metric("📏 Dist. Média", f"{_df_ov['Distância (m)'].mean():.0f} m" if 'Distância (m)' in _df_ov.columns else "—",
                                      help=f"Soma de todos os {_n_periodos_ov} período(s) por atleta, depois média do grupo")
                        _ov_c3.metric("💨 Vmax do Dia", f"{_df_ov['Velocidade Máx (km/h)'].max():.1f} km/h" if 'Velocidade Máx (km/h)' in _df_ov.columns else "—")
                        _ov_c4.metric("⚡ PL Total Médio", f"{_df_ov['PlayerLoad'].mean():.0f}" if 'PlayerLoad' in _df_ov.columns else "—",
                                      help="Catapult PlayerLoad™ somado em todos os períodos, depois média do grupo")
                        _hsr_col = 'Dist. > 19 km/h (m)'
                        _ov_c5.metric("🏃 HSR Médio", f"{_df_ov[_hsr_col].mean():.0f} m" if _hsr_col in _df_ov.columns else "—",
                                      help=f"HSR somado em {_n_periodos_ov} período(s) por atleta, depois média do grupo")

                        if _n_periodos_ov > 1:
                            st.caption(f"📋 Valores combinados de **{_n_periodos_ov} períodos** — somas, máximos e médias ponderadas por atleta.")

                        _hr("DISTÂNCIA POR ATLETA", "📏")

                        # ── Gráfico de barras — distância combinada (gradiente %) ─
                        if 'Atleta' in _df_ov.columns and 'Distância (m)' in _df_ov.columns:
                            _fig_ov = go.Figure()
                            _dist_vals = _df_ov['Distância (m)'].values
                            _dmin, _dmax = _dist_vals.min(), _dist_vals.max()
                            _drng = max(_dmax - _dmin, 1)
                            # Gradiente azul-escuro → ciano brilhante baseado no percentil
                            def _bar_color(_v):
                                _t = (_v - _dmin) / _drng          # 0=pior, 1=melhor
                                _r = int(21  + _t * (0   - 21))
                                _g = int(101 + _t * (229 - 101))
                                _b = int(192 + _t * (255 - 192))
                                return f'rgb({_r},{_g},{_b})'
                            _bar_colors = [_bar_color(v) for v in _dist_vals]
                            _fig_ov.add_trace(go.Bar(
                                x=_df_ov['Atleta'], y=_dist_vals,
                                marker=dict(
                                    color=_bar_colors,
                                    line=dict(color='rgba(255,255,255,0.08)', width=1),
                                ),
                                text=_dist_vals.round(0).astype(int),
                                textposition='outside',
                                textfont=dict(color='white', size=10),
                                hovertemplate='<b>%{x}</b><br>Distância: %{y:.0f} m<extra></extra>',
                            ))
                            _fig_ov.update_layout(
                                title=dict(
                                    text=f'Distância Total por Atleta ({_n_periodos_ov} período(s) combinado(s))',
                                    font=dict(color='white', size=14, family='Inter, sans-serif')
                                ),
                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                font=dict(color='white', family='Inter, sans-serif'),
                                xaxis=dict(gridcolor='rgba(255,255,255,0.06)',
                                           tickangle=-30, tickfont=dict(size=10)),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.06)',
                                           title='metros'),
                                height=330, margin=dict(t=50, b=80, l=10, r=10),
                                showlegend=False,
                                bargap=0.28,
                            )
                            st.plotly_chart(_fig_ov, use_container_width=True)

                        # ── M/min calculado da agregação (Dist / Duração) ────────────
                        if ('Duração (min)' in _df_ov.columns
                                and 'Distância (m)' in _df_ov.columns):
                            _dur_ov = _df_ov['Duração (min)'].replace(0, float('nan'))
                            _df_ov['M/min'] = (_df_ov['Distância (m)'] / _dur_ov).round(1)

                        # ── %Vmax ──────────────────────────────────────────────────
                        if 'Velocidade Máx (km/h)' in _df_ov.columns:
                            _hvm_ov = st.session_state.get('hist_vmax', {})
                            def _pct_vmax_ov(row):
                                _vel = float(row.get('Velocidade Máx (km/h)', 0) or 0)
                                _hist = float(_hvm_ov.get(row.get('Atleta', ''), 0) or 0) * 3.6
                                if not (5.0 <= _hist <= 60.0):
                                    _hist = 0.0
                                if _hist > 0:
                                    return round(_vel / _hist * 100, 1)
                                _gmax = _df_ov['Velocidade Máx (km/h)'].max()
                                return round(_vel / _gmax * 100, 1) if _gmax > 0 else 0.0
                            _df_ov['%Vmax'] = _df_ov.apply(_pct_vmax_ov, axis=1)

                        # ── Tabela Descritiva com coloração por percentil ─────────
                        _hr("TABELA DESCRITIVA DE DESEMPENHO", "📋")
                        st.subheader("📋 Tabela Descritiva de Desempenho")
                        st.caption("Coloração por percentil do grupo: 🟢 Top 33% · 🟡 Médio · 🔴 Bottom 33%")
                        st.caption("ℹ️ **RHIE** — Repeated High Intensity Efforts · **HSR** — Dist. >19 km/h · **M/min** — Metros por minuto (elite: 110–130)")

                        _OV_TD_COLS = [c for c in [
                            'Atleta', 'Posição',
                            'Duração (min)', 'Distância (m)',
                            'M/min',                              # ← 5ª coluna
                            'Dist. 19-24 km/h (m)', 'Dist. > 24 km/h (m)',
                            'Dist. > 19 km/h (m)', 'Sprints (>24 km/h)',
                            'Velocidade Máx (km/h)', '%Vmax',
                            # 'Acc 2-3 (m/s²)' e 'Dcc 2-3 (m/s²)' removidos a pedido
                            'Acelerações (>3 m/s²)', 'Desacelerações (<-3 m/s²)',
                            'Acc Max (m/s²)', 'Dcc Max (m/s²)',
                            'PlayerLoad', 'RHIE Blocos',
                            'FC Média (bpm)',
                        ] if c in _df_ov.columns]

                        if _OV_TD_COLS:
                            _df_ov_show = (
                                _df_ov[_OV_TD_COLS]
                                # Remove linhas sem atleta ou inteiramente vazias
                                .dropna(subset=['Atleta'])
                                .loc[lambda d: d['Atleta'].astype(str).str.strip() != '']
                                .sort_values(
                                    'Distância (m)' if 'Distância (m)' in _OV_TD_COLS else _OV_TD_COLS[0],
                                    ascending=False
                                )
                                .reset_index(drop=True)
                            )

                            _OV_NUM = [c for c in _OV_TD_COLS if c not in ('Atleta', 'Posição')]

                            def _ov_style_pct(col):
                                if col.name not in _OV_NUM:
                                    return [''] * len(col)
                                p33 = col.quantile(0.33)
                                p66 = col.quantile(0.66)
                                out = []
                                for v in col:
                                    try:
                                        vf = float(v)
                                        if vf >= p66:
                                            out.append('background-color:#1a5c2e;color:white;font-weight:bold')
                                        elif vf >= p33:
                                            out.append('background-color:#7d6a08;color:white')
                                        else:
                                            out.append('background-color:#7b1a1a;color:white')
                                    except Exception:
                                        out.append('')
                                return out

                            _1dec_ov = {
                                'Velocidade Máx (km/h)', 'Velocidade Média (km/h)',
                                'M/min', 'Acc Max (m/s²)', 'Dcc Max (m/s²)',
                                'FC Média (bpm)',
                            }
                            _ov_fmt = {}
                            for _c in _OV_NUM:
                                if _c == '%Vmax':
                                    _ov_fmt[_c] = '{:.1f}%'
                                elif _c in _1dec_ov:
                                    _ov_fmt[_c] = '{:.1f}'
                                else:
                                    _ov_fmt[_c] = '{:.0f}'

                            st.markdown(
                                "<style>"
                                "[data-testid='stDataFrame'] th {"
                                "  text-align:center !important;"
                                "  justify-content:center !important;"
                                "}"
                                "[data-testid='stDataFrame'] td {"
                                "  text-align:center !important;"
                                "}"
                                "</style>",
                                unsafe_allow_html=True,
                            )
                            # Altura dinâmica: mostra todos os atletas sem scroll vertical
                            # ~38 px por linha + 60 px de header/padding
                            _ov_height = max(200, 38 * len(_df_ov_show) + 60)
                            st.dataframe(
                                _df_ov_show.style
                                .apply(_ov_style_pct, axis=0)
                                .format(_ov_fmt, na_rep='—')
                                .set_properties(**{'text-align': 'center'})
                                .set_table_styles([
                                    {'selector': 'th',
                                     'props': [('text-align', 'center'),
                                               ('font-weight', 'bold')]},
                                    {'selector': 'td',
                                     'props': [('text-align', 'center')]},
                                ]),
                                use_container_width=True,
                                hide_index=True,
                                height=_ov_height,
                            )
                            st.download_button(
                                "📥 Exportar Tabela (CSV)",
                                _df_ov_show.to_csv(index=False).encode('utf-8'),
                                "resumo_sessao.csv",
                                mime='text/csv',
                            )
                else:
                    st.info("⚽ Carregue uma sessão na sidebar para visualizar o resumo.")
                    st.markdown("""
**Como começar:**
1. 🔐 Insira seu token Catapult na sidebar
2. 🔄 Clique em "Carregar Dados"
3. 📅 Selecione a atividade
4. 👥 Escolha os atletas
5. ✅ Clique em "Carregar Dados da Sessão"
                    """)
