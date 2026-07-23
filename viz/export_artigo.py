# -*- coding: utf-8 -*-
"""Aba Exportação para Artigo (P4 — render_* extraída para viz/).

Tabela no formato do export OpenField/Catapult (21 colunas) + validação de
concordância. Depende de modulos ja extraidos (bands, diagnostics, metrics,
validation, config) + streamlit/pandas.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import applog as _applog
import metrics as _mtr
import validation as _valmod
from bands import (_bandas_vel_ativas, _bandas_acc_ativas, _fmt_num_banda,
                   _ACC_KEY_TO_NUM)
from analysis import acc_series_from_vel, get_min_dur_s
from diagnostics import _diag_log
from config import _CHAVE_COMBINADO


# ══════════════════════════════════════════════════════════════════════════
# EXPORTAÇÃO PARA ARTIGO — tabela no formato do export OpenField/Catapult
# ══════════════════════════════════════════════════════════════════════════
_EXPORT_ARTIGO_COLS = [
    "Name", "Date", "Total Distance (m)", "Minutos", "Metros por minuto",
    "Total Player Load",
    "Velocity Band 1 Total Distance (m)", "Velocity Band 2 Total Distance (m)",
    "Velocity Band 3 Total Distance (m)", "Velocity Band 4 Total Distance (m)",
    "Velocity Band 5 Total Distance (m)", "Velocity Band 6 Total Distance (m)",
    "Max Acceleration", "Max Deceleration", "Maximum Velocity (km/h)",
    "Acceleration B1 Efforts (Gen 2)", "Acceleration B2 Efforts (Gen 2)",
    "Acceleration B3 Efforts (Gen 2)",
    "Deceleration B1 Efforts (Gen 2)", "Deceleration B2 Efforts (Gen 2)",
    "Deceleration B3 Efforts (Gen 2)",
]


def _fmt_data_br(valor) -> str:
    """Formata a data da atividade para DD/MM/AAAA (aceita epoch, ISO ou str)."""
    from datetime import datetime as _dt
    if valor is None or valor == '':
        return ''
    try:
        fv = float(valor)
        if fv > 1e8:
            return _dt.fromtimestamp(fv).strftime('%d/%m/%Y')
    except (TypeError, ValueError):
        pass
    s = str(valor)
    try:
        return _dt.fromisoformat(s.replace('Z', '').split('.')[0]).strftime('%d/%m/%Y')
    except Exception:
        _applog.log_debug_exc()
    for _f in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'):
        try:
            return _dt.strptime(s[:10], _f).strftime('%d/%m/%Y')
        except ValueError:
            continue
    return s[:10]


def _dist_por_banda_vel(sensor_points, n_bandas=6):
    """Integra a distância (m) percorrida em cada banda de velocidade, usando as
    bandas ativas (_bandas_vel_ativas, km/h). Retorna lista de n_bandas valores."""
    bandas = _bandas_vel_ativas()
    ordem = sorted(bandas.keys())[:n_bandas]
    faixas = []
    for idx, k in enumerate(ordem):
        lo = float(bandas[k].get('min', 0))
        hi = float(bandas[k].get('max', 9999))
        if idx == len(ordem) - 1:
            hi = 1e9            # última banda: sem teto (captura sprints altos)
        faixas.append((lo, hi))
    # (P1) Delegado ao motor único: metrics.dist_by_velocity_bands.
    vel_kmh = [float(p['v']) * 3.6 for p in sensor_points
               if p.get('v') is not None]
    d = _mtr.dist_by_velocity_bands(vel_kmh, faixas, hz=_mtr.SENSOR_HZ)
    return (d + [0.0] * n_bandas)[:n_bandas]


def _contar_efforts_acc_por_caixa(acc_efforts) -> dict:
    """Conta acceleration_efforts por caixa Gen2 (campo 'band', 1..8).

    (P1) Delegado ao motor único: metrics.count_efforts_by_box."""
    return _mtr.count_efforts_by_box(acc_efforts)


def render_export_artigo(resultados_por_periodo, dados_sensor_por_atleta_por_periodo,
                         dados_efforts_acc_por_periodo):
    """Aba 📤 Exportação para Artigo — tabela no formato do export Catapult."""
    st.markdown("### 📤 Exportação para Artigo")
    st.caption("Monta uma tabela no **mesmo formato do export OpenField/Catapult** "
               "(21 colunas): totais para distâncias e esforços; máximos para "
               "aceleração, desaceleração e velocidade. Selecione datas/períodos e "
               "atletas e exporte em **CSV** pronto para o artigo.")

    _periodos_reais = [p for p in resultados_por_periodo.keys() if p != _CHAVE_COMBINADO]
    if not _periodos_reais:
        st.info("Carregue uma atividade com dados para exportar.")
        return

    _act_nome = st.session_state.get('_atividade_sel_cached', '') or 'Atividade'
    _act_data = ''
    try:
        _dfa = st.session_state.get('df_activities')
        if _dfa is not None and not _dfa.empty and 'data' in _dfa.columns:
            _match = _dfa[_dfa['nome'] == _act_nome]
            if not _match.empty:
                _act_data = _fmt_data_br(_match['data'].values[0])
    except Exception:
        _applog.log_debug_exc()

    _atletas_disp = sorted({r.get('Atleta', '') for p in _periodos_reais
                            for r in resultados_por_periodo.get(p, [])} - {''})
    if not _atletas_disp:
        st.info("Nenhum atleta com métricas neste conjunto.")
        return

    c1, c2 = st.columns(2)
    with c1:
        _per_sel = st.multiselect("Períodos / datas a incluir", _periodos_reais,
                                  default=_periodos_reais, key="exp_art_periodos")
    with c2:
        _atl_sel = st.multiselect("Atletas", _atletas_disp,
                                  default=_atletas_disp, key="exp_art_atletas")
    _agregar = st.checkbox("Agregar períodos em 1 linha por atleta (recomendado)",
                           value=True, key="exp_art_agg",
                           help="Ligado: soma os períodos numa linha por atleta (uma "
                                "sessão). Desligado: uma linha por período×atleta.")

    if not _per_sel or not _atl_sel:
        st.info("Selecione ao menos um período e um atleta.")
        return

    def _linha(atleta, periodos, nome_linha):
        tot_d = tot_m = tot_pl = 0.0
        bands = [0.0] * 6
        max_acc = 0.0
        min_dec = 0.0
        max_vel = 0.0
        cx = {1: 0, 2: 0, 3: 0, 6: 0, 7: 0, 8: 0}
        for per in periodos:
            _row = next((r for r in resultados_por_periodo.get(per, [])
                         if r.get('Atleta') == atleta), None)
            if _row:
                tot_d += float(_row.get('Distância (m)', 0) or 0)
                tot_m += float(_row.get('Duração (min)', 0) or 0)
                tot_pl += float(_row.get('PlayerLoad', 0) or 0)
                max_acc = max(max_acc, float(_row.get('Acc Max (m/s²)', 0) or 0))
                min_dec = min(min_dec, -abs(float(_row.get('Dcc Max (m/s²)', 0) or 0)))
                max_vel = max(max_vel, float(_row.get('Velocidade Máx (km/h)', 0) or 0))
            _sp = dados_sensor_por_atleta_por_periodo.get(per, {}).get(atleta, [])
            if _sp:
                _d6 = _dist_por_banda_vel(_sp)
                for i in range(6):
                    bands[i] += _d6[i] if i < len(_d6) else 0.0
            _effs_api = dados_efforts_acc_por_periodo.get(per, {}).get(atleta, [])
            if _effs_api:
                _cc = _contar_efforts_acc_por_caixa(_effs_api)
            else:
                # (Validação) Conta sem acceleration_efforts na API → detecta
                # AÇÕES no sinal do sensor e classifica pelo PICO na caixa Gen2
                # (antes o export saía zerado nessas contas).
                _cc = {}
                if _sp:
                    _acc_e = [float(_p.get('a') or 0.0) for _p in _sp]
                    if not any(abs(_a) > 0.05 for _a in _acc_e):
                        _vel_e = [float(_p.get('v') or 0.0) * 3.6 for _p in _sp]
                        _ts_e = [float(_p.get('ts') or 0.0) for _p in _sp]
                        _acc_e = acc_series_from_vel(_vel_e, _ts_e, 10.0)
                    _boxes_cfg = {
                        _ACC_KEY_TO_NUM[_k]: (float(_v.get('min')), float(_v.get('max')))
                        for _k, _v in _bandas_acc_ativas().items()
                        if _k in _ACC_KEY_TO_NUM}
                    _cc = _mtr.count_actions_by_box(
                        _acc_e, _boxes_cfg, min_dur_s=get_min_dur_s(), hz=10.0)
                    _diag_log('Export', f"{atleta} ({per}): efforts ausentes na API — "
                                        "ações detectadas no sinal do sensor")
            for b in cx:
                cx[b] += _cc.get(b, 0)
        _mmin = (tot_d / tot_m) if tot_m > 0 else 0.0
        return {
            "Name": nome_linha,
            "Date": _act_data,
            "Total Distance (m)": round(tot_d, 5),
            "Minutos": round(tot_m, 5),
            "Metros por minuto": round(_mmin, 5),
            "Total Player Load": round(tot_pl, 5),
            "Velocity Band 1 Total Distance (m)": round(bands[0], 2),
            "Velocity Band 2 Total Distance (m)": round(bands[1], 2),
            "Velocity Band 3 Total Distance (m)": round(bands[2], 2),
            "Velocity Band 4 Total Distance (m)": round(bands[3], 2),
            "Velocity Band 5 Total Distance (m)": round(bands[4], 2),
            "Velocity Band 6 Total Distance (m)": round(bands[5], 2),
            "Max Acceleration": round(max_acc, 5),
            "Max Deceleration": round(min_dec, 5),
            "Maximum Velocity (km/h)": round(max_vel, 5),
            "Acceleration B1 Efforts (Gen 2)": cx[6],
            "Acceleration B2 Efforts (Gen 2)": cx[7],
            "Acceleration B3 Efforts (Gen 2)": cx[8],
            "Deceleration B1 Efforts (Gen 2)": cx[3],
            "Deceleration B2 Efforts (Gen 2)": cx[2],
            "Deceleration B3 Efforts (Gen 2)": cx[1],
        }

    _rows = []
    if _agregar:
        for _atl in _atl_sel:
            _rows.append(_linha(_atl, _per_sel, f"{_act_nome} - {_atl}"))
    else:
        for _per in _per_sel:
            for _atl in _atl_sel:
                _rows.append(_linha(_atl, [_per], f"{_act_nome} {_per} - {_atl}"))

    _df = pd.DataFrame(_rows, columns=_EXPORT_ARTIGO_COLS)
    st.dataframe(_df, use_container_width=True, hide_index=True,
                 height=min(560, 60 + len(_df) * 35))
    st.caption(f"{len(_df)} linha(s) · 21 colunas · distâncias por banda **integradas do "
               "sinal de velocidade**; esforços **contados dos efforts Gen2** da conta.")

    # (Rastreabilidade/validação) cortes de banda ATIVOS neste export — confira
    # aqui se as bandas calibradas estão em vigor antes de exportar.
    try:
        _bv_atv = _bandas_vel_ativas()
        _cortes_txt = " | ".join(
            _fmt_num_banda(_bv_atv[_k].get('min', 0))
            for _k in sorted(_bv_atv.keys()))
        _src_txt = {'manual': '🎚️ calibradas/manuais', 'api': '🛰️ API da conta',
                    'efforts': '🟢 derivadas dos efforts',
                    'default': '⚪ padrão'}.get(
            st.session_state.get('velocity_zones_source', 'default'), '?')
        st.caption(f"🎚️ **Cortes de banda usados** (início de B1..B6, km/h): "
                   f"**{_cortes_txt}** · fonte: {_src_txt}")
    except Exception:
        _applog.log_debug_exc()

    st.download_button(
        "📥 Exportar CSV (formato Catapult)",
        _df.to_csv(index=False).encode('utf-8'),
        file_name=f"export_artigo_{_act_nome.replace(' ', '_')[:40] or 'sessao'}.csv",
        mime='text/csv')

    # ── P3: Validação de concordância vs. export oficial (OpenField) ────────
    st.divider()
    with st.expander("🔬 Validação vs. export oficial (OpenField)", expanded=False):
        st.caption(
            "Carregue o **CSV oficial exportado do OpenField** (mesma atividade e "
            "mesmos períodos da tabela acima) para medir a concordância do app: "
            "viés, erro %, correlação e **Bland-Altman** por variável — o "
            "instrumento do projeto de validação."
        )
        _up_val = st.file_uploader("CSV oficial (OpenField)", type=['csv'],
                                   key="val_csv_up")
        if _up_val is not None:
            _df_off = None
            try:
                _df_off = pd.read_csv(_up_val)
            except Exception as _e_v:
                st.error(f"Não foi possível ler o CSV: {_e_v}")
            if _df_off is not None:
                _vars_num = [c for c in _EXPORT_ARTIGO_COLS
                             if c not in ('Name', 'Date')]
                _faltam = [c for c in ['Name'] + _vars_num
                           if c not in _df_off.columns]
                if _faltam:
                    st.error("CSV sem as colunas esperadas: "
                             + ", ".join(_faltam[:6])
                             + ("…" if len(_faltam) > 6 else ""))
                else:
                    # CSV oficial multi-sessão → filtra pela atividade para o
                    # pareamento não somar sessões diferentes do mesmo atleta.
                    _acts_of = sorted({str(_n).rsplit(' - ', 1)[0]
                                       for _n in _df_off['Name'].dropna()})
                    _df_off_f = _df_off
                    if len(_acts_of) > 1:
                        _act_pick = st.selectbox(
                            "Atividade do CSV oficial (parear com a tabela acima)",
                            ["(todas — soma por atleta)"] + _acts_of,
                            key="val_act_pick")
                        if _act_pick != "(todas — soma por atleta)":
                            _df_off_f = _df_off[_df_off['Name'].astype(str)
                                                .str.startswith(_act_pick)]
                    _merged, _stats = _valmod.comparar_exportacoes(
                        _df, _df_off_f, _vars_num)
                    if _merged.empty:
                        st.warning("Nenhum atleta em comum entre a tabela do app "
                                   "e o CSV oficial (o pareamento usa o nome após "
                                   "o último ' - ' na coluna Name).")
                    else:
                        st.success(f"✅ {len(_merged)} atleta(s) pareado(s) por nome.")
                        st.markdown("##### 📋 Concordância por variável")
                        st.dataframe(_stats, use_container_width=True,
                                     hide_index=True)
                        st.caption("**Viés** = média (app − oficial); **Viés %** "
                                   "relativo à média oficial; **r** = correlação "
                                   "de Pearson. Interpretação usual: |viés %| < 5% "
                                   "e r > 0,90 indicam boa concordância.")

                        # (Removido) Calibração de cortes de banda pelo CSV: o app
                        # usa limiares FIXOS e documentados (padrão da literatura);
                        # não há ajuste de bandas pelo usuário — instrumento
                        # determinístico para fins de validação científica.

                        _var_ba = st.selectbox("Variável para Bland-Altman",
                                               _vars_num, key="val_var_ba")
                        _ba = _valmod.bland_altman(
                            _merged[f"{_var_ba} (app)"],
                            _merged[f"{_var_ba} (oficial)"])
                        if _ba is None:
                            st.info("Mínimo de 3 pares válidos para o Bland-Altman "
                                    "desta variável.")
                        else:
                            _fig_ba = go.Figure()
                            _fig_ba.add_trace(go.Scatter(
                                x=_ba['mean'], y=_ba['diff'],
                                mode='markers',
                                marker=dict(size=10, color='#42A5F5',
                                            line=dict(color='white', width=1)),
                                text=list(_merged['Atleta']),
                                hovertemplate='%{text}<br>média %{x:.1f} · '
                                              'dif %{y:.2f}<extra></extra>',
                                name='atletas'))
                            for _yv, _lab, _cor, _dsh in [
                                    (_ba['bias'], f"viés {_ba['bias']:.2f}",
                                     '#FFD740', 'solid'),
                                    (_ba['loa_high'], f"+1,96 DP {_ba['loa_high']:.2f}",
                                     '#EF5350', 'dash'),
                                    (_ba['loa_low'], f"−1,96 DP {_ba['loa_low']:.2f}",
                                     '#EF5350', 'dash')]:
                                _fig_ba.add_hline(y=_yv, line_dash=_dsh,
                                                  line_color=_cor,
                                                  annotation_text=_lab,
                                                  annotation_font_color=_cor)
                            _fig_ba.update_layout(
                                title=f"Bland-Altman — {_var_ba} (n={_ba['n']})",
                                xaxis_title='Média dos métodos (app, oficial)',
                                yaxis_title='Diferença (app − oficial)',
                                height=430, paper_bgcolor='#0e1117',
                                plot_bgcolor='#0e1117', font=dict(color='white'),
                                xaxis=dict(gridcolor='#1f2937'),
                                yaxis=dict(gridcolor='#1f2937'))
                            st.plotly_chart(_fig_ba, use_container_width=True)

                        st.download_button(
                            "📥 Exportar comparação completa (CSV)",
                            _merged.to_csv(index=False).encode('utf-8'),
                            "validacao_app_vs_oficial.csv", mime='text/csv',
                            key="dl_val_csv")
