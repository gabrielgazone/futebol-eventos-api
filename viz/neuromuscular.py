# -*- coding: utf-8 -*-
"""Aba extraída de main() (P4 — page-decomposition)."""
from __future__ import annotations

from field import _neuro_cached
from diagnostics import _selo_fonte
from analysis import get_min_dur_s
import numpy as np
import pandas as pd
from field import plotar_carga_neuromuscular
import streamlit as st


def render_neuromuscular(_SENSOR_HZ, dados_sensor_por_atleta_por_periodo):
                st.divider()
                st.subheader("💪 Análise de Carga Neuromuscular")
                st.markdown("""
                Esforços de **aceleração** e **desaceleração** intensa são indicadores críticos de
                carga neuromuscular/excêntrica — desacelerações superiores a 2 m/s² geram impacto
                muscular frequentemente maior que sprints. Esta aba quantifica esses esforços por
                minuto e acumula a carga ao longo da sessão.
                """)

                if dados_sensor_por_atleta_por_periodo:
                    _NM_TODOS = "🔀 Todos os períodos (combinado)"
                    _nm_opcoes_per = [_NM_TODOS] + list(dados_sensor_por_atleta_por_periodo.keys())
                    _nm_per = st.selectbox("Período:", _nm_opcoes_per, key="nm_periodo")

                    # Monta lista de atletas disponíveis
                    if _nm_per == _NM_TODOS:
                        _nm_ats_set = set()
                        for _pv in dados_sensor_por_atleta_por_periodo.values():
                            _nm_ats_set.update(_pv.keys())
                        _nm_ats = sorted(_nm_ats_set)
                    else:
                        _nm_ats = list(dados_sensor_por_atleta_por_periodo.get(_nm_per, {}).keys())

                    if _nm_ats:
                        _nm_atl = st.selectbox("Atleta:", _nm_ats, key="nm_atleta")
                        _nm_lim = st.slider("Limiar de intensidade (m/s²):", 1.0, 4.0, 2.0, 0.5,
                                            key="nm_limiar",
                                            help="Acelerações/desacelerações acima deste valor são classificadas como intensas.")

                        _nm_dur_s = get_min_dur_s()
                        st.caption(
                            f"⚙️ Duração mínima de acc/dec: **{_nm_dur_s:.1f} s** "
                            f"({max(1, round(_nm_dur_s * _SENSOR_HZ))} frames a 10 Hz) — "
                            "ajuste na sidebar."
                        )

                        # Combina sensor_points de todos os períodos se necessário
                        if _nm_per == _NM_TODOS:
                            _nm_sp = []
                            for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                _nm_sp += _pv2.get(_nm_atl, [])
                            if len(dados_sensor_por_atleta_por_periodo) > 1:
                                st.caption(
                                    f"📊 Combinando **{len(dados_sensor_por_atleta_por_periodo)} períodos** "
                                    f"→ {len(_nm_sp):,} amostras para **{_nm_atl}**."
                                )
                        else:
                            _nm_sp = dados_sensor_por_atleta_por_periodo[_nm_per].get(_nm_atl, [])

                        # (P6) cache: evita recomputar masks/EPM de ~50k amostras
                        # a cada interação de widget (dados grandes fora da chave).
                        _nm_key = (str(st.session_state.get('_token_marker', '')),
                                   str(st.session_state.get('activity_id', '')),
                                   str(_nm_per), str(_nm_atl),
                                   float(_nm_lim), float(_nm_dur_s), len(_nm_sp))
                        _nm_dados = _neuro_cached(_nm_key, _nm_sp, _nm_lim, _nm_dur_s)

                        if _nm_dados:
                            _selo_fonte('sensor')   # (P4) acc/dec do sinal nativo
                            # Métricas resumo
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric(f"🟢 Acels. ≥{_nm_lim} m/s²", _nm_dados['total_hi_acc'])
                            c2.metric(f"🔴 Desacels. ≥{_nm_lim} m/s²", _nm_dados['total_hi_dec'])
                            c3.metric("⚡ Total Acels. (todas)", _nm_dados['total_hi_acc'] + _nm_dados['total_med_acc'])
                            c4.metric("⚡ Total Desacels. (todas)", _nm_dados['total_hi_dec'] + _nm_dados['total_med_dec'])

                            _nm_razao = ((_nm_dados['total_hi_dec'] / _nm_dados['total_hi_acc'])
                                         if _nm_dados['total_hi_acc'] > 0 else 0)
                            if _nm_razao > 1.4:
                                st.warning(f"⚠️ Razão Dec/Acc = **{_nm_razao:.2f}** — alto componente excêntrico. "
                                           "Monitorar recuperação muscular dos membros inferiores.")
                            elif _nm_razao > 0.9:
                                st.info(f"ℹ️ Razão Dec/Acc = **{_nm_razao:.2f}** — carga excêntrica equilibrada.")
                            else:
                                st.success(f"✅ Razão Dec/Acc = **{_nm_razao:.2f}** — perfil predominantemente acelerativo.")

                            _nm_fig = plotar_carga_neuromuscular(_nm_dados, _nm_atl)
                            st.plotly_chart(_nm_fig, use_container_width=True)

                            # Exportar
                            _nm_df = pd.DataFrame({
                                'Tempo (min)': _nm_dados['t_mid'],
                                'Acels. Intensas/min': _nm_dados['hi_acc_min'],
                                'Acels. Médias/min': _nm_dados['med_acc_min'],
                                'Desacels. Intensas/min': _nm_dados['hi_dec_min'],
                                'Desacels. Médias/min': _nm_dados['med_dec_min'],
                            })
                            st.download_button(
                                "📥 Exportar Carga Neuromuscular (CSV)",
                                _nm_df.to_csv(index=False),
                                f"carga_neuro_{_nm_atl.replace(' ','_')}.csv"
                            )
                        else:
                            st.info("Dados de aceleração insuficientes para este atleta/período.")

                        # ── FEATURE 3: Potência Metabólica na aba Carga Neuromuscular ──
                        st.markdown("---")
                        st.markdown("### ⚡ Potência Metabólica (W/kg)")
                        _nm_mp_pts = _nm_sp
                        _nm_mp_vals = [
                            float(p['mp']) for p in _nm_mp_pts
                            if p.get('mp') and float(p.get('mp') or 0) > 0
                        ]
                        if _nm_mp_vals:
                            _nm_mp_mean = float(np.mean(_nm_mp_vals))
                            _nm_mp_max  = float(np.max(_nm_mp_vals))
                            _nm_mp_pct20 = sum(1 for v in _nm_mp_vals if v > 20) / max(1, len(_nm_mp_vals)) * 100
                            _nm_mp_t25   = sum(1 for v in _nm_mp_vals if v > 25) * 0.1
                            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                            _mc1.metric("MP Médio (W/kg)", f"{_nm_mp_mean:.1f}")
                            _mc2.metric("MP Máx (W/kg)", f"{_nm_mp_max:.1f}")
                            _mc3.metric("MP > 20 W/kg (%)", f"{_nm_mp_pct20:.1f}%")
                            _mc4.metric("Tempo > 25 W/kg (s)", f"{_nm_mp_t25:.0f}s")
                        else:
                            st.info(
                                "Dados de potência metabólica (mp) não disponíveis. "
                                "Verifique se o dispositivo suporta este parâmetro."
                            )

                else:
                    st.info("Carregue os dados de um atleta para analisar a carga neuromuscular.")
