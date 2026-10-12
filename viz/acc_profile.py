# -*- coding: utf-8 -*-
"""Aba: Perfil de Aceleração por Zona de Velocidade de Entrada (ES1–ES4).

Lê a atividade já filtrada na barra lateral e entrega, para TODOS os atletas,
a contagem de esforços de aceleração classificada pela velocidade em que o
esforço começou (% da Vmax individual), com corte derivado do próprio sinal
(percentil da zona) e score ponderado pelo custo mecânico da zona.

Motor em `acc_profile.py` (puro, testado). Aqui só UI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import acc_profile as _ap
from analysis import combinar_periodos_continuo
from config import _CHAVE_COMBINADO
from field import _vmax_individual_kmh

# Cores das zonas — gradiente força→velocidade (vermelho → ciano), como no blueprint
_COR_ZONA = {'ES1': '#B71C1C', 'ES2': '#E8820C', 'ES3': '#F2C300', 'ES4': '#00BCD4'}


def _explicacao():
    with st.expander("📖 Como isto é calculado (explicação simples)", expanded=False):
        st.markdown("""
**A ideia em uma frase:** empurrar um carrinho parado é fácil; empurrar o mesmo
carrinho quando ele já está voando é dificílimo. Esta aba separa as arrancadas
do atleta por **quão rápido ele já estava indo quando começou a arrancada**.

**Por que isso importa.** O jeito tradicional usa um número fixo para todo mundo
e para todo momento — por exemplo, "contar toda aceleração acima de 3 m/s²".
O problema: acelerar a 3 m/s² saindo do lugar é fácil, qualquer jogador faz
muitas vezes. Acelerar a 3 m/s² quando já está a 25 km/h é quase impossível —
e quando acontece, é um esforço brutal. O número fixo conta as duas como se
fossem a mesma coisa.

**Os 5 passos:**

1. **Qual é a velocidade máxima deste atleta?** (a Vmax dele, não a do time)
2. **Divide a velocidade dele em 4 faixas** — ES1 é devagar (até 25% da Vmax),
   ES4 é quase voando (acima de 75%).
3. **Acha todas as arrancadas** do jogo e pergunta de cada uma: *"a que
   velocidade ela começou?"* — e põe a arrancada na faixa certa.
4. **Dentro de cada faixa, o app olha as arrancadas mais fortes** daquela faixa
   e usa isso como régua. O que fica muito abaixo da régua é tranco pequeno ou
   erro do GPS, e é descartado. Sobram as arrancadas que valem.
5. **Soma com peso.** Uma arrancada na faixa mais rápida (ES4) conta **2,5**,
   porque custa muito mais ao músculo. Na faixa devagar (ES1) conta **1**.

**O resultado:** quatro notas (ES1 a ES4) e uma **carga total**. Dois atletas
podem ter o mesmo número de arrancadas e cargas bem diferentes — porque um
arrancou parado e o outro arrancou em velocidade alta.
""")


def _parametros(key_pref='accp'):
    """Painel de parâmetros editáveis. Retorna o dict de configuração."""
    with st.expander("⚙️ Parâmetros do cálculo (editável)", expanded=False):
        st.caption(
            "Os limites das zonas, os pesos e o corte percentual são convenções "
            "do framework de origem — ajuste conforme o seu protocolo.")
        _c1, _c2, _c3 = st.columns(3)
        _perc = _c1.number_input(
            "Percentil de referência da zona", value=float(_ap.DEFAULT_PERCENTIL),
            min_value=50.0, max_value=100.0, step=1.0, key=f'{key_pref}_perc',
            help="Teto de referência de cada zona. 95 = P95 das acelerações de "
                 "pico daquela zona (descarta spike isolado).")
        _pct = _c2.number_input(
            "Corte (% do percentil)", value=float(_ap.DEFAULT_PCT_CORTE * 100),
            min_value=10.0, max_value=100.0, step=5.0, key=f'{key_pref}_pct',
            help="Só eventos com pico acima deste corte contam como válidos.") / 100.0
        _piso = _c3.number_input(
            "Piso de segmentação (m/s²)", value=float(_ap.DEFAULT_PISO_MS2),
            min_value=0.1, max_value=5.0, step=0.1, key=f'{key_pref}_piso',
            help="Serve só para delimitar onde um esforço começa e termina — "
                 "quem filtra de fato é o corte do percentil. É o parâmetro "
                 "MAIS sensível: calibre-o até a contagem total bater com os "
                 "efforts de aceleração que o OpenField reporta neste jogo.")
        _c4, _c5 = st.columns(2)
        _dur = _c4.number_input(
            "Duração mínima do esforço (s)", value=float(_ap.DEFAULT_MIN_DUR_S),
            min_value=0.1, max_value=2.0, step=0.1, key=f'{key_pref}_dur')
        _fonte_vmax = _c5.selectbox(
            "Referência de Vmax", ['Histórico do atleta (recomendado)',
                                   'Pico desta atividade'],
            key=f'{key_pref}_fvmax',
            help="O histórico (OpenField/override) é a capacidade real do atleta. "
                 "O pico da atividade muda a cada jogo e desloca as zonas.")
    return {'percentil': _perc, 'pct_corte': _pct, 'piso_ms2': _piso,
            'min_dur_s': _dur, 'usar_hist': _fonte_vmax.startswith('Histórico')}


def _serie_do_atleta(dados_sensor, atleta, modo_todos, per_sel):
    if modo_todos:
        return combinar_periodos_continuo(dados_sensor, atleta)
    return (dados_sensor.get(per_sel) or {}).get(atleta) or []


def _grafico_barras(resumo_df, key):
    """Barras empilhadas: score de cada zona por atleta."""
    _fig = go.Figure()
    for _rot, _, _, _, _dom in _ap.ES_ZONAS:
        _col = f'{_rot} score'
        if _col not in resumo_df.columns:
            continue
        _fig.add_trace(go.Bar(
            name=f'{_rot} · {_dom}', x=resumo_df['Atleta'], y=resumo_df[_col],
            marker_color=_COR_ZONA[_rot],
            hovertemplate='%{x}<br>' + _rot + ': %{y}<extra></extra>'))
    _fig.update_layout(
        barmode='stack', height=430,
        yaxis_title='Score ponderado', xaxis_title='',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(_fig, use_container_width=True, key=key)


def _grafico_dispersao(perfil, atleta, key):
    """Aceleração de pico × velocidade de entrada, com as faixas ES e os cortes.

    É o gráfico que torna o método visível: cada ponto é um esforço, as faixas
    verticais são as zonas de velocidade de entrada e as linhas horizontais são
    o corte de validade DAQUELA zona.
    """
    _vm = perfil['vmax']
    if _vm <= 0:
        st.info("Sem Vmax de referência para plotar as zonas deste atleta.")
        return
    _fig = go.Figure()
    _ymax = 1.0
    for _rot, _lo, _hi, _peso, _dom in _ap.ES_ZONAS:
        _z = perfil['zonas'][_rot]
        _x0 = _lo * _vm
        _x1 = (_hi * _vm) if np.isfinite(_hi) else max(_vm * 1.05, _x0 + 1)
        # faixa da zona
        _fig.add_vrect(x0=_x0, x1=_x1, fillcolor=_COR_ZONA[_rot], opacity=0.10,
                       layer='below', line_width=0,
                       annotation_text=_rot, annotation_position='top left')
        # pontos: válidos cheios, descartados vazados
        _val = _z['validos']
        _desc = [_e for _e in _z['eventos'] if _e not in _val]
        if _desc:
            _fig.add_trace(go.Scatter(
                x=[_e['vel_entrada'] for _e in _desc],
                y=[_e['pico'] for _e in _desc], mode='markers',
                name=f'{_rot} descartado', showlegend=False,
                marker=dict(color=_COR_ZONA[_rot], size=6, opacity=0.35,
                            symbol='circle-open'),
                hovertemplate=('entrada %{x:.1f} km/h<br>pico %{y:.2f} m/s²'
                               '<br>descartado<extra></extra>')))
        if _val:
            _fig.add_trace(go.Scatter(
                x=[_e['vel_entrada'] for _e in _val],
                y=[_e['pico'] for _e in _val], mode='markers',
                name=f'{_rot} válido ({len(_val)})',
                marker=dict(color=_COR_ZONA[_rot], size=9, opacity=0.9),
                hovertemplate=('entrada %{x:.1f} km/h<br>pico %{y:.2f} m/s²'
                               '<br>válido<extra></extra>')))
        # linha do corte da zona
        if _z['corte'] > 0:
            _fig.add_shape(type='line', x0=_x0, x1=_x1,
                           y0=_z['corte'], y1=_z['corte'],
                           line=dict(color=_COR_ZONA[_rot], width=2, dash='dash'))
        if _z['eventos']:
            _ymax = max(_ymax, max(_e['pico'] for _e in _z['eventos']))
    _fig.add_vline(x=_vm, line=dict(color='#555', width=1, dash='dot'),
                   annotation_text=f'Vmax {_vm:.1f}', annotation_position='top right')
    _fig.update_layout(
        height=480, xaxis_title='Velocidade de ENTRADA do esforço (km/h)',
        yaxis_title='Aceleração de pico do esforço (m/s²)',
        yaxis_range=[0, _ymax * 1.12],
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f'{atleta} — linha tracejada = corte de validade da zona',
                   font=dict(size=13)))
    st.plotly_chart(_fig, use_container_width=True, key=key)


def render_acc_profile(dados_sensor_por_atleta_por_periodo, hz=10.0):
    """Seção completa do perfil de aceleração por zona de entrada."""
    st.subheader("🚀 Perfil de Aceleração por Velocidade de Entrada (ES1–ES4)")
    st.caption(
        "Classifica cada esforço de aceleração pela **velocidade em que ele "
        "começou**, como % da Vmax individual — em vez de um limiar fixo de "
        "m/s² igual para todos. O corte de validade de cada zona sai do próprio "
        "sinal (percentil da zona) e o score é ponderado pelo custo mecânico.")

    _ds = dados_sensor_por_atleta_por_periodo or {}
    if not _ds:
        st.info("Carregue uma atividade na barra lateral para ver o perfil.")
        return

    _explicacao()
    _cfg = _parametros()

    # ── Escopo: período ou partida inteira ────────────────────────────────────
    _periodos = list(_ds.keys())
    _opts = ([_CHAVE_COMBINADO] + _periodos) if len(_periodos) > 1 else _periodos
    _per_sel = st.selectbox("Período:", _opts, key='accp_periodo')
    _modo_todos = (_per_sel == _CHAVE_COMBINADO)

    _atletas = sorted({_a for _p in _ds.values() for _a in _p.keys()})
    if not _atletas:
        st.info("Sem atletas com dados de sensor nesta atividade.")
        return

    # ── Calcula o perfil de TODOS os atletas ──────────────────────────────────
    _perfis, _sem_vmax = {}, []
    for _atl in _atletas:
        _pts = _serie_do_atleta(_ds, _atl, _modo_todos, _per_sel)
        if not _pts:
            continue
        _vk = _ap.vel_kmh_de_pontos(_pts)
        _vm = _vmax_individual_kmh(_atl, _vk) if _cfg['usar_hist'] else 0.0
        if _vm <= 0:
            _vm = _ap.vmax_kmh(_vk, 100.0)
            if _cfg['usar_hist']:
                _sem_vmax.append(_atl)
        if _vm < 10.0:
            continue                      # sem referência confiável de Vmax
        _perfis[_atl] = _ap.perfil_atleta(
            _pts, hz, vmax=_vm, percentil=_cfg['percentil'],
            pct_corte=_cfg['pct_corte'], piso_ms2=_cfg['piso_ms2'],
            min_dur_s=_cfg['min_dur_s'])

    if not _perfis:
        st.warning("Nenhum atleta com Vmax de referência confiável (≥10 km/h).")
        return
    if _sem_vmax:
        st.caption(
            f"ℹ️ Sem Vmax histórica para {len(_sem_vmax)} atleta(s) "
            f"({', '.join(_sem_vmax[:4])}{'…' if len(_sem_vmax) > 4 else ''}) — "
            "usado o pico desta atividade como referência.")

    # ── KPIs do grupo ─────────────────────────────────────────────────────────
    _resumo = pd.DataFrame(_ap.linhas_resumo(_perfis))
    _c1, _c2, _c3, _c4 = st.columns(4)
    _c1.metric("Atletas", len(_perfis))
    _c2.metric("Esforços detectados", int(sum(_p['n_eventos'] for _p in _perfis.values())))
    _c3.metric("Esforços válidos",
               int(sum(_p['zonas'][_z[0]]['n_validos']
                       for _p in _perfis.values() for _z in _ap.ES_ZONAS)))
    _c4.metric("Carga média ponderada", f"{_resumo['Carga total ponderada'].mean():.1f}")

    # Guarda-corpo: contagem implausível denuncia piso baixo para este sinal.
    _med_ev = float(np.mean([_p['n_eventos'] for _p in _perfis.values()]))
    if _med_ev > _ap.TETO_PLAUSIVEL_EVENTOS:
        st.warning(
            f"⚠️ **{_med_ev:.0f} esforços por atleta** em média — implausível "
            f"para uma partida (o esperado fica na casa das dezenas a ~150). "
            f"O piso de segmentação ({_cfg['piso_ms2']:.1f} m/s²) está baixo "
            "para a qualidade deste sinal e micro-oscilações estão virando "
            "eventos. Aumente o piso em *Parâmetros* até a contagem bater com "
            "os efforts de aceleração do OpenField para esta atividade.")

    # ── Tabela-resumo (uma linha por atleta) ──────────────────────────────────
    st.markdown("### 📋 Resumo por atleta")
    st.caption("Ordenado por carga total ponderada. Score da zona = esforços "
               "válidos × peso da zona (ES1 ×1,0 · ES2 ×1,5 · ES3 ×2,0 · ES4 ×2,5).")
    st.dataframe(_resumo, use_container_width=True, hide_index=True,
                 height=min(520, 44 + len(_resumo) * 36), key='accp_tbl_resumo')
    st.download_button(
        "⬇️ Baixar resumo (CSV)",
        _resumo.to_csv(index=False).encode('utf-8-sig'),
        file_name='perfil_aceleracao_ES_resumo.csv', mime='text/csv',
        key='accp_dl_resumo')

    st.markdown("### 📊 Composição da carga por zona")
    _grafico_barras(_resumo, key='accp_barras')

    # ── Detalhe por atleta ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Detalhe por atleta")
    _atl_sel = st.selectbox("Atleta:", list(_resumo['Atleta']), key='accp_atl')
    _p = _perfis[_atl_sel]

    _d1, _d2, _d3 = st.columns(3)
    _d1.metric("Vmax de referência", f"{_p['vmax']:.1f} km/h")
    _d2.metric("Esforços no período", _p['n_eventos'])
    _d3.metric("Carga total ponderada", f"{_p['total']:.1f}")

    _det = pd.DataFrame(_ap.linhas_tabela({_atl_sel: _p}))
    _det = _det.drop(columns=['Atleta', 'Vmax (km/h)'])
    st.dataframe(_det, use_container_width=True, hide_index=True, key='accp_tbl_det')

    _frageis = [_z[0] for _z in _ap.ES_ZONAS
                if 0 < _p['zonas'][_z[0]]['n_total'] < _ap.MIN_EVENTOS_P95]
    if _frageis:
        st.caption(
            f"⚠️ Zonas com menos de {_ap.MIN_EVENTOS_P95} esforços "
            f"({', '.join(_frageis)}): o percentil da zona é instável e o corte "
            "fica pouco confiável. Interprete com cautela.")

    _grafico_dispersao(_p, _atl_sel, key='accp_disp')

    # ── Tabela longa de todos (para export) ───────────────────────────────────
    with st.expander("📤 Tabela completa (atleta × zona) para export"):
        _longa = pd.DataFrame(_ap.linhas_tabela(_perfis))
        st.dataframe(_longa, use_container_width=True, hide_index=True,
                     key='accp_tbl_longa')
        st.download_button(
            "⬇️ Baixar tabela completa (CSV)",
            _longa.to_csv(index=False).encode('utf-8-sig'),
            file_name='perfil_aceleracao_ES_detalhe.csv', mime='text/csv',
            key='accp_dl_longa')

    st.markdown("---")
    st.caption(f"⚠️ **Procedência dos parâmetros:** {_ap.PROVENIENCIA}")
