# -*- coding: utf-8 -*-
"""Campo/plotagem e eventos (P4 — extraído do monólito).

Desenho do campo, trajetórias/heatmaps, pontos por banda, eventos de futebol,
Voronoi, carga neuromuscular e ACWR. Depende de bands + config + streamlit/
numpy/pandas/plotly/folium.
"""
from __future__ import annotations

import colorsys as _colorsys
from datetime import timedelta

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from scipy.ndimage import gaussian_filter as _gf
from scipy.spatial import cKDTree

import applog as _applog
from bands import (_bandas_vel_ativas, _bandas_acc_ativas, _fmt_num_banda,
                   _rotulo_banda_vel, _rotulo_banda_acc, _legenda_vel_js)
from analysis import detectar_eventos_acc, get_min_dur_s
from config import (
    BANDAS_VEL, BANDAS_ACC, _ACC_BAND_MAP, FUTEBOL_EVENTS_CONFIG,
    _NOMES_BANDA_VEL_DEFAULT, _CORES_BANDA_VEL_DEFAULT,
    _ATHLETE_PALETTE, _POSICAO_GRUPOS, _EV_ICONES, _DEFAULT_MIN_DUR_S,
)


def _vmax_individual_kmh(atleta, vel_series=None) -> float:
    """(P9) Vmáx individual (km/h): histórico da conta (hist_vmax em m/s —
    summary OpenField ou override manual) com fallback para o pico da série
    da sessão. Retorna 0.0 quando não há referência confiável (≥10 km/h)."""
    v = 0.0
    try:
        v = float(st.session_state.get('hist_vmax', {}).get(atleta, 0) or 0) * 3.6
    except Exception:
        v = 0.0
    if v < 15.0 and vel_series is not None and len(vel_series) > 0:
        try:
            v = max(v, max(float(x) for x in vel_series))
        except (TypeError, ValueError):
            pass
    return v if v >= 10.0 else 0.0

def cor_atleta(nome: str) -> str:
    """Retorna sempre a mesma cor para o mesmo atleta (determinístico).
    Prioriza a paleta persistida no session_state; usa hash como fallback."""
    cores = st.session_state.get('athlete_colors', {})
    if nome in cores:
        return cores[nome]
    # fallback determinístico via hash do nome
    import hashlib
    idx = int(hashlib.md5(nome.encode()).hexdigest(), 16) % len(_ATHLETE_PALETTE)
    return _ATHLETE_PALETTE[idx]

def _get_pos_grupo(posicao: str) -> tuple:
    """Retorna (nome_do_grupo, hue_base) para uma string de posição."""
    p = (posicao or '').upper().strip()
    for grupo, cfg in _POSICAO_GRUPOS.items():
        if any(tag in p for tag in cfg['tags']):
            return grupo, cfg['h']
    return 'Outro', 0.75

def cor_atleta_pos(nome: str, posicao: str = '', variante_idx: int = 0, total: int = 1) -> str:
    """Cor baseada no grupo de posição com variação tonal individual.

    Atletas do mesmo grupo partilham a mesma tonalidade (hue); saturação e
    luminosidade variam para distinguir jogadores dentro do grupo.
    """
    _, h = _get_pos_grupo(posicao)
    frac = (variante_idx / max(1, total - 1)) if total > 1 else 0.5
    # Saturação: 0.70 → 0.95 (mais vivo conforme índice aumenta)
    s = 0.70 + 0.25 * frac
    # Luminosidade: 0.60 → 0.38 (escurece progressivamente)
    l = 0.60 - 0.22 * frac
    r, g, b = _colorsys.hls_to_rgb(h, l, s)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

def _ev_icone(tipo: str) -> str:
    """Retorna emoji do evento ou '📌' como fallback."""
    for key, icon in _EV_ICONES.items():
        if key.lower() in (tipo or '').lower():
            return icon
    return '📌'

def extrair_dados_sensor(response_data):
    if not response_data:
        return []
    if isinstance(response_data, list):
        for item in response_data:
            if isinstance(item, dict) and 'data' in item:
                return item['data']
    return []

def extrair_efforts_data(response_data):
    if not response_data:
        return [], [], [], [], []

    velocity_efforts = []
    acceleration_efforts = []
    heart_rate_efforts = []
    jump_efforts = []
    step_balance_efforts = []

    if isinstance(response_data, list) and len(response_data) > 0:
        item = response_data[0]
        if isinstance(item, dict) and 'data' in item:
            data_obj = item['data']
            if isinstance(data_obj, dict):
                velocity_efforts = data_obj.get('velocity_efforts', [])
                acceleration_efforts = data_obj.get('acceleration_efforts', [])
                heart_rate_efforts = data_obj.get('heart_rate_efforts', [])
                jump_efforts = data_obj.get('jump_efforts', [])
                step_balance_efforts = data_obj.get('step_balance_efforts', [])

    return velocity_efforts, acceleration_efforts, heart_rate_efforts, jump_efforts, step_balance_efforts

def lat_lon_to_campo_coords(latitudes, longitudes):
    """
    Converte coordenadas de latitude/longitude para coordenadas do campo (0-100m x 0-70m)
    """
    if len(latitudes) == 0 or len(longitudes) == 0:
        return [], []
    
    lats = np.array(latitudes)
    lons = np.array(longitudes)
    
    # Normalizar para 0-1 baseado no range dos dados
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
    if lat_max == lat_min:
        lat_range = 1
    else:
        lat_range = lat_max - lat_min
    
    if lon_max == lon_min:
        lon_range = 1
    else:
        lon_range = lon_max - lon_min
    
    # Normalizar
    lat_norm = (lats - lat_min) / lat_range
    lon_norm = (lons - lon_min) / lon_range
    
    # Escalar para o campo (comprimento 105m, largura 68m — FIFA)
    x = lon_norm * 105
    y = lat_norm * 68
    
    return x.tolist(), y.tolist()

def desenhar_campo_futebol(field_length=105, field_width=68):
    """Desenha o campo de futebol com todas as marcações oficiais FIFA."""
    fig = go.Figure()
    FL, FW = field_length, field_width
    cy = FW / 2  # centro y

    # Fundo
    fig.add_shape(type="rect", x0=-5, y0=-5, x1=FL+5, y1=FW+5,
                  fillcolor="#1a472a", line=dict(color="rgba(0,0,0,0)", width=0))
    # Perímetro
    fig.add_shape(type="rect", x0=0, y0=0, x1=FL, y1=FW,
                  line=dict(color="white", width=3), fillcolor="rgba(0,100,0,0.3)")
    # Linha central
    fig.add_shape(type="line", x0=FL/2, y0=0, x1=FL/2, y1=FW, line=dict(color="white", width=2))
    # Círculo central (r=9.15m)
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=FL/2 + 9.15*np.cos(theta), y=cy + 9.15*np.sin(theta),
                             mode='lines', line=dict(color="white", width=1.5),
                             name="Círculo Central", showlegend=False))
    # Áreas de penalidade (16.5m × 40.32m)
    for x0, x1 in [(0, 16.5), (FL-16.5, FL)]:
        fig.add_shape(type="rect", x0=x0, y0=cy-20.16, x1=x1, y1=cy+20.16,
                      line=dict(color="white", width=1.5))
    # Áreas pequenas (5.5m × 18.32m)
    for x0, x1 in [(0, 5.5), (FL-5.5, FL)]:
        fig.add_shape(type="rect", x0=x0, y0=cy-9.16, x1=x1, y1=cy+9.16,
                      line=dict(color="white", width=1.5))
    # Pontos de penalidade
    for px in [11, FL-11]:
        fig.add_trace(go.Scatter(x=[px], y=[cy], mode='markers',
                                 marker=dict(size=6, color='white'),
                                 showlegend=False))
    # Ponto central
    fig.add_trace(go.Scatter(x=[FL/2], y=[cy], mode='markers',
                             marker=dict(size=6, color='white'), showlegend=False))
    # Gols (7.32m × 2.44m)
    fig.add_shape(type="rect", x0=-2.44, y0=cy-3.66, x1=0, y1=cy+3.66,
                  line=dict(color="#FFD700", width=3))
    fig.add_shape(type="rect", x0=FL, y0=cy-3.66, x1=FL+2.44, y1=cy+3.66,
                  line=dict(color="#FFD700", width=3))

    fig.update_layout(
        title="Campo de Futebol Oficial FIFA — Trajetória e Mapa de Calor",
        xaxis=dict(range=[-5, FL+5], title="Comprimento do Campo (m)",
                   fixedrange=False, gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        yaxis=dict(range=[-5, FW+5], title="Largura do Campo (m)",
                   fixedrange=False, gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        plot_bgcolor='#1a472a', paper_bgcolor='#1a1a2e', height=600, hovermode='closest'
    )
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                       text="⚪ Linhas FIFA | 🟡 Gol | 🔵 Trajetória | 🟢 Início | 🔴 Fim",
                       showarrow=False, font=dict(color="white", size=10),
                       bgcolor='rgba(0,0,0,0.6)', borderpad=4)
    return fig

def plotar_trajetoria_campo(x_coords, y_coords, velocidades, athlete_name):
    """Plota a trajetória do atleta no campo"""
    fig = desenhar_campo_futebol(105, 68)
    
    if len(x_coords) == 0:
        return fig
    
    # Trajetória (linha)
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='lines',
        name=f'{athlete_name} - Trajetória',
        line=dict(color='cyan', width=2),
        hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<extra></extra>'
    ))
    
    # Pontos coloridos por velocidade
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        name='Velocidade',
        marker=dict(size=3, color=velocidades, colorscale='Viridis',
                   showscale=True, colorbar=dict(title=dict(text="Velocidade (km/h)"), x=1.05, len=0.5)),
        hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Vel: %{marker.color:.1f} km/h<extra></extra>'
    ))
    
    # Início
    fig.add_trace(go.Scatter(
        x=[x_coords[0]], y=[y_coords[0]],
        mode='markers',
        name='Início',
        marker=dict(size=14, color='#00FF00', symbol='circle', line=dict(width=2, color='white'))
    ))
    
    # Fim
    fig.add_trace(go.Scatter(
        x=[x_coords[-1]], y=[y_coords[-1]],
        mode='markers',
        name='Fim',
        marker=dict(size=14, color='#FF0000', symbol='x', line=dict(width=2, color='white'))
    ))
    
    return fig

def plotar_heatmap_campo(x_coords, y_coords, velocidades, athlete_name):
    """Mapa de calor de intensidade com suavização gaussiana (KDE-like)."""
    fig = desenhar_campo_futebol(105, 68)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return fig

    # Grade fina para KDE suave
    _nx, _ny = 80, 52
    x_edges = np.linspace(0, 105, _nx + 1)
    y_edges = np.linspace(0,  68, _ny + 1)
    x_ctr   = (x_edges[:-1] + x_edges[1:]) / 2
    y_ctr   = (y_edges[:-1] + y_edges[1:]) / 2

    heatmap, _, _ = np.histogram2d(x_coords, y_coords,
                                   bins=[x_edges, y_edges], weights=velocidades)
    counts,  _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity = np.divide(heatmap, counts,
                              out=np.zeros_like(heatmap), where=counts > 0)

    # Suavização gaussiana — sigma controla o "blur" (4 → efeito broadcast)
    intensity_smooth = _gf(intensity, sigma=4)

    # Máscara de zeros para transparência onde não há dados
    _zdata = intensity_smooth.T.copy()
    _zdata[_zdata < _zdata.max() * 0.02] = np.nan   # corta ruído baixo

    fig.add_trace(go.Heatmap(
        z=_zdata,
        x=x_ctr, y=y_ctr,
        colorscale=[
            [0.0,  'rgba(0,0,128,0)'],
            [0.2,  'rgba(0,80,200,0.35)'],
            [0.45, 'rgba(0,200,100,0.6)'],
            [0.7,  'rgba(255,200,0,0.75)'],
            [0.88, 'rgba(255,80,0,0.88)'],
            [1.0,  'rgba(200,0,0,1.0)'],
        ],
        zsmooth='best',
        opacity=0.72,
        name='Intensidade (Vel.)',
        colorbar=dict(
            title=dict(text='km/h', font=dict(color='white', size=10)),
            tickfont=dict(color='white', size=9), x=1.02, len=0.55,
        ),
        hovertemplate='X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Vel média: %{z:.1f} km/h<extra></extra>',
    ))
    fig.update_layout(title=dict(text=f"🔥 Mapa de Calor — {athlete_name}",
                                  font=dict(color='white', size=13)))
    return fig

def plotar_heatmap_presenca_campo(x_coords, y_coords, athlete_name):
    """Mapa de calor de presença com suavização gaussiana."""
    fig = desenhar_campo_futebol(105, 68)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return fig

    _nx, _ny = 80, 52
    x_edges = np.linspace(0, 105, _nx + 1)
    y_edges = np.linspace(0,  68, _ny + 1)
    x_ctr   = (x_edges[:-1] + x_edges[1:]) / 2
    y_ctr   = (y_edges[:-1] + y_edges[1:]) / 2

    counts, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])
    counts_smooth = _gf(counts, sigma=3.5)
    _zdata = counts_smooth.T.copy()
    _zdata[_zdata < _zdata.max() * 0.015] = np.nan

    fig.add_trace(go.Heatmap(
        z=_zdata,
        x=x_ctr, y=y_ctr,
        colorscale=[
            [0.0,  'rgba(0,0,80,0)'],
            [0.25, 'rgba(0,100,200,0.4)'],
            [0.5,  'rgba(0,220,130,0.65)'],
            [0.75, 'rgba(255,180,0,0.8)'],
            [1.0,  'rgba(200,0,0,1.0)'],
        ],
        zsmooth='best',
        opacity=0.7,
        name='Presença',
        colorbar=dict(
            title=dict(text='Freq.', font=dict(color='white', size=10)),
            tickfont=dict(color='white', size=9), x=1.02, len=0.55,
        ),
        hovertemplate='X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Frequência: %{z:.0f}<extra></extra>',
    ))
    fig.update_layout(title=dict(text=f"🔥 Presença — {athlete_name}",
                                  font=dict(color='white', size=13)))
    return fig

def extrair_eventos_futebol(response_data):
    """Extrai eventos futebol da resposta da API /events."""
    if not response_data or not isinstance(response_data, list):
        return {}
    item = response_data[0] if response_data else {}
    data = item.get('data', {}) if isinstance(item, dict) else {}
    return {k: v for k, v in data.items() if v and k in FUTEBOL_EVENTS_CONFIG}

def enriquecer_eventos_com_posicao(eventos_dict, ts_gps, lats_gps, lons_gps, vels_gps, campo_config=None):
    """Associa cada evento ao ponto GPS (e posição no campo) mais próximo no tempo."""
    if not ts_gps:
        return eventos_dict
    ts_arr = np.array(ts_gps, dtype=float)
    result = {}
    for event_type, events in eventos_dict.items():
        enriched = []
        for ev in events:
            ev2 = dict(ev)
            t = float(ev.get('start_time') or 0)
            idx = int(np.argmin(np.abs(ts_arr - t)))
            ev2['_lat'] = lats_gps[idx]
            ev2['_lon'] = lons_gps[idx]
            ev2['_vel'] = vels_gps[idx]
            if campo_config:
                fx, fy = gps_para_campo_coords(
                    [lats_gps[idx]], [lons_gps[idx]], campo_config)
                ev2['_fx'] = fx[0]
                ev2['_fy'] = fy[0]
            enriched.append(ev2)
        result[event_type] = enriched
    return result

def adicionar_eventos_campo(fig, eventos_dict, tipos_sel):
    """Plota marcadores de eventos futebol sobre o campo esquemático."""
    for event_type in tipos_sel:
        events = eventos_dict.get(event_type, [])
        if not events:
            continue
        cfg_ev = FUTEBOL_EVENTS_CONFIG[event_type]
        xs_ev = [ev.get('_fx') for ev in events if ev.get('_fx') is not None]
        ys_ev = [ev.get('_fy') for ev in events if ev.get('_fy') is not None]
        if not xs_ev:
            continue
        # Texto de hover
        texts = []
        for ev in events:
            if ev.get('_fx') is None:
                continue
            partes = [f"<b>{cfg_ev['label']}</b>"]
            for attr in cfg_ev['attrs']:
                val = ev.get(attr)
                if val is not None:
                    partes.append(f"{attr}: {val}")
            partes.append(f"vel: {ev.get('_vel', 0):.1f} km/h")
            texts.append('<br>'.join(partes))
        fig.add_trace(go.Scatter(
            x=xs_ev, y=ys_ev,
            mode='markers',
            name=cfg_ev['label'],
            marker=dict(
                symbol=cfg_ev['marker'],
                size=cfg_ev['size'],
                color=cfg_ev['color'],
                line=dict(color='white', width=1.5),
            ),
            text=texts,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True,
        ))

def criar_timeline_eventos(sensor_points, eventos_dict, atleta_nome, tipos_sel):
    """
    Gráfico de velocidade ao longo do tempo com pins verticais dos eventos futebol.
    """
    if not sensor_points:
        return None

    ts0 = float(sensor_points[0].get('ts') or 0)
    ts_rel = [(float(p.get('ts') or 0) - ts0) for p in sensor_points]
    vels   = [(p.get('v') or 0) * 3.6 for p in sensor_points]

    fig = go.Figure()

    # Linha de velocidade
    fig.add_trace(go.Scatter(
        x=ts_rel, y=vels,
        mode='lines',
        name='Velocidade (km/h)',
        line=dict(color='#2196F3', width=1.5),
        hovertemplate='%{x:.0f}s — %{y:.1f} km/h<extra></extra>',
    ))

    # Pins de eventos
    for event_type in tipos_sel:
        events = eventos_dict.get(event_type, [])
        if not events:
            continue
        cfg_ev = FUTEBOL_EVENTS_CONFIG[event_type]
        for ev in events:
            t_ev  = float(ev.get('start_time') or 0) - ts0
            v_ev  = ev.get('_vel', 0)
            label = cfg_ev['label']
            partes = [f"<b>{label}</b>", f"t={t_ev:.0f}s", f"vel={v_ev:.1f} km/h"]
            for attr in cfg_ev['attrs']:
                val = ev.get(attr)
                if val is not None:
                    partes.append(f"{attr}: {val}")
            fig.add_vline(
                x=t_ev,
                line=dict(color=cfg_ev['color'], width=1.5, dash='dot'),
            )
            fig.add_trace(go.Scatter(
                x=[t_ev], y=[v_ev],
                mode='markers',
                name=label,
                showlegend=False,
                marker=dict(
                    symbol=cfg_ev['marker'],
                    size=cfg_ev['size'],
                    color=cfg_ev['color'],
                    line=dict(color='white', width=1),
                ),
                text=['<br>'.join(partes)],
                hovertemplate='%{text}<extra></extra>',
            ))

    fig.update_layout(
        title=f"⚽ Timeline Técnico-Físico — {atleta_nome}",
        xaxis_title='Tempo (s)',
        yaxis_title='Velocidade (km/h)',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        height=420,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='left', x=0,
            font=dict(size=11),
        ),
        hovermode='x unified',
    )
    return fig

def analisar_fadiga_eventos(sensor_points, eventos_dict, tipos_sel, janela_s=10):
    """
    Para eventos de contato, computa velocidade média antes e depois do evento.
    Retorna DataFrame com: tipo, tempo, vel_antes, vel_depois, variacao.
    """
    if not sensor_points:
        return pd.DataFrame()

    ts_arr  = np.array([float(p.get('ts') or 0) for p in sensor_points])
    vel_arr = np.array([(p.get('v') or 0) * 3.6 for p in sensor_points])
    ts0     = ts_arr[0] if len(ts_arr) else 0

    rows = []
    for event_type in tipos_sel:
        events = eventos_dict.get(event_type, [])
        cfg_ev = FUTEBOL_EVENTS_CONFIG[event_type]
        for ev in events:
            t = float(ev.get('start_time') or 0)
            mask_before = (ts_arr >= t - janela_s) & (ts_arr < t)
            mask_after  = (ts_arr >  t) & (ts_arr <= t + janela_s)
            vel_antes   = float(vel_arr[mask_before].mean()) if mask_before.any() else None
            vel_depois  = float(vel_arr[mask_after].mean())  if mask_after.any()  else None
            variacao    = round(vel_depois - vel_antes, 2) if (vel_antes is not None and vel_depois is not None) else None
            row = {
                'Tipo': cfg_ev['label'],
                'Tempo (s)': round(t - ts0, 1),
                'Vel. Antes (km/h)': round(vel_antes, 2) if vel_antes is not None else None,
                'Vel. Depois (km/h)': round(vel_depois, 2) if vel_depois is not None else None,
                'Variação (km/h)': variacao,
                'post_event_load': ev.get('post_event_load'),
                'duration': ev.get('duration'),
                'confidence': ev.get('confidence'),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values('Tempo (s)').reset_index(drop=True)
    return df

def desenhar_campo_futebol_bonito(field_length=105, field_width=68, margin=3, title="", attack_direction=None):
    """Campo de futebol com faixas verdes alternadas e marcações oficiais FIFA (top-down).

    attack_direction: None | 'left_to_right' | 'right_to_left'
        Quando definido, exibe uma seta dourada acima do campo indicando o sentido de ataque.
    """
    FL, FW, MG = field_length, field_width, margin
    cy = FW / 2  # centro y
    fig = go.Figure()

    # ── Fundo externo (borda escura ao redor do campo) ───────────────
    fig.add_shape(type="rect", x0=-MG-5, y0=-MG-3, x1=FL+MG+5, y1=FW+MG+3,
                  fillcolor="#111f10", line_width=0, layer="below")

    # ── Faixas verdes alternadas — trama de gramado ──────────────────
    n_st, sw = 12, FL / 12
    cores_faixa = ["#2a7325", "#236b1e"]           # ligeiramente mais escuras / distintas
    for i in range(n_st):
        fig.add_shape(type="rect", x0=i*sw, y0=0, x1=(i+1)*sw, y1=FW,
                      fillcolor=cores_faixa[i % 2], line_width=0, layer="below")

    # ── Glow suave nas linhas brancas (duplicata desfocada) ──────────
    _gkw = dict(fillcolor="rgba(0,0,0,0)", layer="below")
    def _glow(x0, y0, x1, y1):
        fig.add_shape(type="rect", x0=x0-0.25, y0=y0-0.25, x1=x1+0.25, y1=y1+0.25,
                      line=dict(color="rgba(255,255,255,0.12)", width=4), **_gkw)

    _glow(0, 0, FL, FW)
    _glow(FL/2, 0, FL/2, FW)

    # ── Perímetro principal ──────────────────────────────────────────
    fig.add_shape(type="rect", x0=0, y0=0, x1=FL, y1=FW,
                  line=dict(color="white", width=2.5), fillcolor="rgba(0,0,0,0)")

    # ── Linha central ────────────────────────────────────────────────
    fig.add_shape(type="line", x0=FL/2, y0=0, x1=FL/2, y1=FW,
                  line=dict(color="white", width=2))

    # ── Círculo central (r = 9.15 m) ────────────────────────────────
    th = np.linspace(0, 2*np.pi, 120)
    # glow do círculo
    fig.add_trace(go.Scatter(x=FL/2 + 9.15*np.cos(th), y=cy + 9.15*np.sin(th),
                             mode='lines', line=dict(color='rgba(255,255,255,0.12)', width=4),
                             showlegend=False, hoverinfo='skip', name='_circg'))
    fig.add_trace(go.Scatter(x=FL/2 + 9.15*np.cos(th), y=cy + 9.15*np.sin(th),
                             mode='lines', line=dict(color='white', width=1.8),
                             showlegend=False, hoverinfo='skip', name='_circ'))

    # ── Ponto central ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=[FL/2], y=[cy], mode='markers',
                             marker=dict(size=7, color='white',
                                         line=dict(color='rgba(0,0,0,0.5)', width=1)),
                             showlegend=False, hoverinfo='skip', name='_ctr'))

    # ── Área de penalidade (16.5 m × 40.32 m) ───────────────────────
    for x0, x1 in [(0, 16.5), (FL-16.5, FL)]:
        _glow(x0, cy-20.16, x1, cy+20.16)
        fig.add_shape(type="rect", x0=x0, y0=cy-20.16, x1=x1, y1=cy+20.16,
                      line=dict(color="white", width=1.8), fillcolor="rgba(0,0,0,0)")

    # ── Área pequena (5.5 m × 18.32 m) ──────────────────────────────
    for x0, x1 in [(0, 5.5), (FL-5.5, FL)]:
        fig.add_shape(type="rect", x0=x0, y0=cy-9.16, x1=x1, y1=cy+9.16,
                      line=dict(color="white", width=1.5), fillcolor="rgba(0,0,0,0)")

    # ── Arcos de penalidade (r = 9.15 m, fora da área) ───────────────
    th_full = np.linspace(0, 2*np.pi, 240)
    for px_p, lado in [(11, 'esq'), (FL-11, 'dir')]:
        arc_x = px_p + 9.15*np.cos(th_full)
        arc_y = cy  + 9.15*np.sin(th_full)
        mask = (arc_x > 16.5) if lado == 'esq' else (arc_x < FL-16.5)
        if mask.sum() > 1:
            fig.add_trace(go.Scatter(x=arc_x[mask], y=arc_y[mask],
                                     mode='lines',
                                     line=dict(color='rgba(255,255,255,0.1)', width=4),
                                     showlegend=False, hoverinfo='skip', name='_parcg'))
            fig.add_trace(go.Scatter(x=arc_x[mask], y=arc_y[mask],
                                     mode='lines', line=dict(color='white', width=1.5),
                                     showlegend=False, hoverinfo='skip', name='_parc'))

    # ── Pontos de penalidade ─────────────────────────────────────────
    for px_p in [11, FL-11]:
        fig.add_trace(go.Scatter(x=[px_p], y=[cy], mode='markers',
                                 marker=dict(size=7, color='white',
                                             line=dict(color='rgba(0,0,0,0.4)', width=1)),
                                 showlegend=False, hoverinfo='skip', name='_pen'))

    # ── Arcos de canto (r = 1 m) ─────────────────────────────────────
    corners_def = [(0,  0,  0,       np.pi/2),
                   (FL, 0,  np.pi/2, np.pi),
                   (FL, FW, np.pi,   3*np.pi/2),
                   (0,  FW, 3*np.pi/2, 2*np.pi)]
    for cx_c, cy_c, a1, a2 in corners_def:
        th_c = np.linspace(a1, a2, 25)
        fig.add_trace(go.Scatter(x=cx_c + np.cos(th_c), y=cy_c + np.sin(th_c),
                                 mode='lines', line=dict(color='white', width=1.5),
                                 showlegend=False, hoverinfo='skip', name='_corner'))

    # ── Corner flag posts (bandeirinhas) ────────────────────────────
    for fcx, fcy in [(0, 0), (FL, 0), (FL, FW), (0, FW)]:
        fig.add_trace(go.Scatter(x=[fcx], y=[fcy], mode='markers',
                                 marker=dict(size=8, color='#FFD700', symbol='diamond',
                                             line=dict(color='rgba(0,0,0,0.5)', width=1)),
                                 showlegend=False, hoverinfo='skip', name='_flag'))

    # ── Gols — rede (fill cinza) + borda dourada ─────────────────────
    # Rede (fill)
    fig.add_shape(type="rect", x0=-2.44, y0=cy-3.66, x1=0,      y1=cy+3.66,
                  fillcolor="rgba(200,200,200,0.12)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=FL,    y0=cy-3.66, x1=FL+2.44, y1=cy+3.66,
                  fillcolor="rgba(200,200,200,0.12)", line_width=0, layer="below")
    # Grade da rede (linhas horizontais)
    net_y_lines = np.linspace(cy-3.66, cy+3.66, 6)
    for _ny in net_y_lines:
        fig.add_shape(type="line", x0=-2.44, y0=_ny, x1=0, y1=_ny,
                      line=dict(color="rgba(180,180,180,0.25)", width=0.5))
        fig.add_shape(type="line", x0=FL, y0=_ny, x1=FL+2.44, y1=_ny,
                      line=dict(color="rgba(180,180,180,0.25)", width=0.5))
    # Grade da rede (linhas verticais)
    net_x_left  = np.linspace(-2.44, 0, 5)
    net_x_right = np.linspace(FL, FL+2.44, 5)
    for _nx in net_x_left:
        fig.add_shape(type="line", x0=_nx, y0=cy-3.66, x1=_nx, y1=cy+3.66,
                      line=dict(color="rgba(180,180,180,0.25)", width=0.5))
    for _nx in net_x_right:
        fig.add_shape(type="line", x0=_nx, y0=cy-3.66, x1=_nx, y1=cy+3.66,
                      line=dict(color="rgba(180,180,180,0.25)", width=0.5))
    # Borda dourada dos gols
    gd_line = dict(color="#FFD700", width=2.5)
    fig.add_shape(type="rect", x0=-2.44, y0=cy-3.66, x1=0,      y1=cy+3.66, line=gd_line)
    fig.add_shape(type="rect", x0=FL,    y0=cy-3.66, x1=FL+2.44, y1=cy+3.66, line=gd_line)

    # ── Labels das linhas ────────────────────────────────────────────
    lkw = dict(showarrow=False, font=dict(color='rgba(255,255,255,0.45)', size=8,
                                           family='Inter, sans-serif'))
    for xl, txt in [(0, "GL"), (FL/2, "50m"), (FL, "GL")]:
        fig.add_annotation(x=xl, y=-2.2, text=txt, **lkw)

    # ── Seta de direção de ataque ────────────────────────────────────────────
    if attack_direction == 'left_to_right':
        # Seta da esquerda para a direita acima do campo
        fig.add_annotation(
            x=FL * 0.72, y=FW + MG - 0.2,
            ax=FL * 0.28, ay=FW + MG - 0.2,
            axref='x', ayref='y',
            text="", showarrow=True,
            arrowhead=2, arrowsize=1.4, arrowwidth=3,
            arrowcolor='#FFD700',
        )
        fig.add_annotation(
            x=FL * 0.5, y=FW + MG + 0.5,
            text="⚽  ATAQUE",
            showarrow=False,
            font=dict(color='#FFD700', size=11, family='Arial Black'),
            xanchor='center',
        )
    elif attack_direction == 'right_to_left':
        # Seta da direita para a esquerda acima do campo
        fig.add_annotation(
            x=FL * 0.28, y=FW + MG - 0.2,
            ax=FL * 0.72, ay=FW + MG - 0.2,
            axref='x', ayref='y',
            text="", showarrow=True,
            arrowhead=2, arrowsize=1.4, arrowwidth=3,
            arrowcolor='#FFD700',
        )
        fig.add_annotation(
            x=FL * 0.5, y=FW + MG + 0.5,
            text="ATAQUE  ⚽",
            showarrow=False,
            font=dict(color='#FFD700', size=11, family='Arial Black'),
            xanchor='center',
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=13)) if title else {},
        xaxis=dict(range=[-MG-4, FL+MG+4], showgrid=False, zeroline=False,
                   tickfont=dict(color='white', size=9),
                   title=dict(text="metros (comprimento)", font=dict(color='#aaa', size=10))),
        yaxis=dict(range=[-MG-2, FW+MG+2], showgrid=False, zeroline=False,
                   scaleanchor='x', scaleratio=1,
                   tickfont=dict(color='white', size=9),
                   title=dict(text="metros (largura)", font=dict(color='#aaa', size=10))),
        plot_bgcolor='#1a3a18',
        paper_bgcolor='#0e1117',
        height=530,
        margin=dict(l=50, r=160, t=40 if title else 20, b=50),
        hovermode='closest',
        legend=dict(bgcolor='rgba(0,0,0,0.7)', font=dict(color='white'),
                    x=1.01, y=1, bordercolor='rgba(255,255,255,0.2)', borderwidth=1)
    )
    return fig

def adicionar_trajetoria_campo(fig, x_coords, y_coords, velocidades, nome=""):
    """Trajetória colorida por velocidade sobre o campo bonito."""
    if not x_coords:
        return
    xs = list(x_coords)
    ys = list(y_coords)
    vs = list(velocidades) if velocidades else [0]*len(xs)

    _bv_traj = _bandas_vel_ativas()

    def _vc(v):
        for k, b in _bv_traj.items():
            if v < b['max']:
                return b['color']
        return list(_bv_traj.values())[-1]['color']

    # Segmentos coloridos
    seg_x, seg_y = [xs[0]], [ys[0]]
    seg_c = _vc(vs[0])
    for i in range(1, len(xs)):
        c = _vc(vs[i] if i < len(vs) else 0)
        if c == seg_c:
            seg_x.append(xs[i]); seg_y.append(ys[i])
        else:
            if len(seg_x) > 1:
                fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode='lines',
                    line=dict(color=seg_c, width=2.5),
                    showlegend=False, hoverinfo='skip', name='_traj'))
            seg_x, seg_y = [xs[i-1], xs[i]], [ys[i-1], ys[i]]
            seg_c = c
    if len(seg_x) > 1:
        fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode='lines',
            line=dict(color=seg_c, width=2.5),
            showlegend=False, hoverinfo='skip', name='_traj'))

    # Início e fim
    fig.add_trace(go.Scatter(x=[xs[0]], y=[ys[0]], mode='markers', name='Início',
        marker=dict(size=12, color='#00E676', symbol='circle',
                    line=dict(color='white', width=2))))
    fig.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode='markers', name='Fim',
        marker=dict(size=12, color='#F44336', symbol='x',
                    line=dict(color='white', width=2))))

    # Legenda de bandas (usa valores da conta)
    for b in _bv_traj.values():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            name=b['label'], marker=dict(size=9, color=b['color'])))

def _segmentos_continuos(idxs, gap_max=8):
    """Agrupa índices consecutivos em segmentos (gap_max = frames tolerados entre pontos)."""
    if len(idxs) == 0:
        return []
    segs, seg = [], [idxs[0]]
    for i in range(1, len(idxs)):
        if idxs[i] - idxs[i - 1] <= gap_max:
            seg.append(idxs[i])
        else:
            if len(seg) >= 3:
                segs.append(seg)
            seg = [idxs[i]]
    if len(seg) >= 3:
        segs.append(seg)
    return segs

def adicionar_pontos_velocidade_bandas(fig, x_coords, y_coords, velocidades,
                                       bandas_sel, mostrar_setas=False,
                                       atleta_prefix=''):
    """Pontos coloridos por bandas de velocidade selecionadas.

    Se mostrar_setas=True, desenha uma seta direcional ao final de cada
    esforço contínuo dentro de cada banda (direção ofensiva/defensiva).
    atleta_prefix: se fornecido, é adicionado ao início do nome da legenda.
    """
    if not x_coords or not bandas_sel:
        return
    xs = np.array(x_coords)
    ys = np.array(y_coords)
    vs = np.array(velocidades) if velocidades else np.zeros(len(xs))
    _bv_pts = _bandas_vel_ativas()
    for k in bandas_sel:
        b = _bv_pts.get(k) or BANDAS_VEL.get(k, {})
        if not b:
            continue
        mask = (vs >= b['min']) & (vs < b['max'])
        if mask.sum() == 0:
            continue
        _nome = f"{atleta_prefix}: {b['label']}" if atleta_prefix else b['label']
        fig.add_trace(go.Scatter(
            x=xs[mask], y=ys[mask], mode='markers', name=_nome,
            marker=dict(size=3, color=b['color'], opacity=0.8),
            hovertemplate='x=%{x:.1f}m y=%{y:.1f}m<extra>' + _nome + '</extra>'))

        if not mostrar_setas:
            continue

        # ── Uma seta por esforço contínuo dentro da banda ───────────
        for seg in _segmentos_continuos(np.where(mask)[0]):
            sx, sy = xs[seg], ys[seg]
            # Direção: últimos ~25% dos pontos do segmento
            n_t = max(2, min(len(seg) // 4, 10))
            ddx = float(sx[-1] - sx[-n_t])
            ddy = float(sy[-1] - sy[-n_t])
            nm = np.hypot(ddx, ddy)
            if nm < 0.3:                        # fallback início→fim
                ddx = float(sx[-1] - sx[0])
                ddy = float(sy[-1] - sy[0])
                nm  = np.hypot(ddx, ddy)
            if nm < 0.3:
                continue
            ddx /= nm; ddy /= nm
            # Comprimento do cabo: proporcional à extensão do esforço, 1.5–5 m
            span = np.hypot(float(sx[-1] - sx[0]), float(sy[-1] - sy[0]))
            cabo = float(np.clip(span * 0.25, 1.5, 5.0))
            tip_x, tip_y = float(sx[-1]), float(sy[-1])
            fig.add_annotation(
                x=tip_x + ddx * 0.8,   y=tip_y + ddy * 0.8,
                ax=tip_x - ddx * cabo, ay=tip_y - ddy * cabo,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=3,
                arrowsize=1.6, arrowwidth=2.0,
                arrowcolor=b['color'],
            )

def adicionar_pontos_aceleracao_bandas(fig, x_coords, y_coords, aceleracoes,
                                       bandas_sel, mostrar_setas=False,
                                       atleta_prefix=''):
    """Pontos coloridos por bandas de aceleração/desaceleração selecionadas.

    Se mostrar_setas=True, desenha uma seta direcional ao final de cada
    esforço contínuo dentro de cada banda.
    atleta_prefix: se fornecido, é adicionado ao início do nome da legenda.
    """
    if not x_coords or not bandas_sel:
        return
    xs  = np.array(x_coords)
    ys  = np.array(y_coords)
    acc = np.array(aceleracoes) if aceleracoes else np.zeros(len(xs))
    _ba_pts = _bandas_acc_ativas()
    for k in bandas_sel:
        b = _ba_pts.get(k) or BANDAS_ACC.get(k, {})
        if not b:
            continue
        mask = (acc >= b['min']) & (acc < b['max'])
        if mask.sum() == 0:
            continue
        _nome = f"{atleta_prefix}: {b['label']}" if atleta_prefix else b['label']
        fig.add_trace(go.Scatter(
            x=xs[mask], y=ys[mask], mode='markers', name=_nome,
            marker=dict(size=3, color=b['color'], opacity=0.8),
            hovertemplate='x=%{x:.1f}m y=%{y:.1f}m<extra>' + _nome + '</extra>'))

        if not mostrar_setas:
            continue

        # ── Uma seta por esforço contínuo dentro da banda ───────────
        for seg in _segmentos_continuos(np.where(mask)[0]):
            sx, sy = xs[seg], ys[seg]
            n_t = max(2, min(len(seg) // 4, 10))
            ddx = float(sx[-1] - sx[-n_t])
            ddy = float(sy[-1] - sy[-n_t])
            nm  = np.hypot(ddx, ddy)
            if nm < 0.3:
                ddx = float(sx[-1] - sx[0])
                ddy = float(sy[-1] - sy[0])
                nm  = np.hypot(ddx, ddy)
            if nm < 0.3:
                continue
            ddx /= nm; ddy /= nm
            span = np.hypot(float(sx[-1] - sx[0]), float(sy[-1] - sy[0]))
            cabo = float(np.clip(span * 0.25, 1.5, 5.0))
            tip_x, tip_y = float(sx[-1]), float(sy[-1])
            fig.add_annotation(
                x=tip_x + ddx * 0.8,   y=tip_y + ddy * 0.8,
                ax=tip_x - ddx * cabo, ay=tip_y - ddy * cabo,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=3,
                arrowsize=1.6, arrowwidth=2.0,
                arrowcolor=b['color'],
            )

def adicionar_setas_direcao(fig, x_coords, y_coords,
                           xs_effort=None, ys_effort=None):
    """Uma única seta de direção no final do esforço (ou da trajetória).

    Se xs_effort/ys_effort forem fornecidos, a seta é laranja e posicionada
    no final da linha de esforço. Caso contrário, uma seta branca é colocada
    no final da trajetória completa.
    """
    # Escolhe a fonte de coordenadas
    _use_effort = xs_effort is not None and len(xs_effort) >= 2
    xs = np.array(xs_effort if _use_effort else x_coords)
    ys = np.array(ys_effort if _use_effort else y_coords)

    if len(xs) < 2:
        return

    # Calcula direção média dos últimos ~20% dos pontos (robustez contra noise)
    n_tail = max(2, min(len(xs) // 5, 12))
    dx = float(xs[-1] - xs[-n_tail])
    dy = float(ys[-1] - ys[-n_tail])
    norm = np.hypot(dx, dy)
    if norm < 0.3:
        # Fallback: direção global (início → fim)
        dx = float(xs[-1] - xs[0])
        dy = float(ys[-1] - ys[0])
        norm = np.hypot(dx, dy)
        if norm < 0.3:
            return
    dx /= norm; dy /= norm

    tip_x = float(xs[-1])
    tip_y = float(ys[-1])
    tail_len = 5.0   # comprimento do cabo da seta (m)

    cor = '#FF9800' if _use_effort else 'rgba(255,255,255,0.85)'

    fig.add_annotation(
        x=tip_x + dx * 1.5,
        y=tip_y + dy * 1.5,
        ax=tip_x - dx * tail_len,
        ay=tip_y - dy * tail_len,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=3,
        arrowsize=2.2,
        arrowwidth=3.0 if _use_effort else 2.0,
        arrowcolor=cor,
    )

def adicionar_convex_hull(fig, x_coords, y_coords):
    """Polígono de área de atuação (Convex Hull) sobre o campo."""
    if len(x_coords) < 3:
        return
    try:
        from scipy.spatial import ConvexHull
        pts  = np.column_stack([x_coords, y_coords])
        hull = ConvexHull(pts)
        hx   = list(pts[hull.vertices, 0]) + [pts[hull.vertices[0], 0]]
        hy   = list(pts[hull.vertices, 1]) + [pts[hull.vertices[0], 1]]
        area = hull.volume   # em 2D 'volume' = área
        fig.add_trace(go.Scatter(x=hx, y=hy, mode='lines',
            name=f'Área de Atuação ({area:.0f} m²)',
            line=dict(color='#FFD700', width=2, dash='dash'),
            fill='toself', fillcolor='rgba(255,215,0,0.10)'))
    except Exception:
        _applog.log_debug_exc()

def adicionar_tercos_campo(fig, x_coords, y_coords, field_length=105, field_width=68):
    """Overlay dos terços do campo (defensivo / meio / ataque) com % de tempo."""
    if not x_coords:
        return
    xs = np.array(x_coords)
    n  = len(xs)
    tercos = [
        ("Defensivo", 0,            field_length/3,    'rgba(244,67,54,0.13)',  '#F44336'),
        ("Meio",      field_length/3,  2*field_length/3, 'rgba(255,235,59,0.10)', '#FFEB3B'),
        ("Ataque",    2*field_length/3, field_length,    'rgba(76,175,80,0.13)',  '#4CAF50'),
    ]
    for nome, x0, x1, fill, cor in tercos:
        pct = 100 * ((xs >= x0) & (xs < x1)).sum() / n if n > 0 else 0
        fig.add_shape(type="rect", x0=x0, y0=0, x1=x1, y1=field_width,
                      line=dict(color=cor, width=1.5, dash="dot"),
                      fillcolor=fill, layer="above")
        fig.add_annotation(x=(x0+x1)/2, y=field_width-3.5,
                           text=f"<b>{nome}</b><br>{pct:.1f}%",
                           showarrow=False, font=dict(color=cor, size=11),
                           bgcolor='rgba(0,0,0,0.55)', borderpad=3)

def adicionar_grade_quadrantes(fig, x_coords, y_coords, n_cols, n_rows,
                               field_length=105, field_width=68):
    """Grade de quadrantes com % de tempo em cada zona."""
    if not x_coords:
        return
    xs  = np.array(x_coords)
    ys  = np.array(y_coords)
    n   = len(xs)
    cw  = field_length / n_cols
    rh  = field_width  / n_rows

    # Pré-calcular todas as % para escalar a opacidade
    pcts = np.zeros((n_rows, n_cols))
    for r in range(n_rows):
        for c in range(n_cols):
            mask = ((xs >= c*cw) & (xs < (c+1)*cw) &
                    (ys >= r*rh) & (ys < (r+1)*rh))
            pcts[r, c] = 100 * mask.sum() / n if n > 0 else 0
    mx = pcts.max() if pcts.max() > 0 else 1

    for r in range(n_rows):
        for c in range(n_cols):
            x0, x1 = c*cw, (c+1)*cw
            y0, y1 = r*rh, (r+1)*rh
            pct   = pcts[r, c]
            alpha = 0.07 + 0.43 * (pct / mx)
            zid   = f"{chr(65+r)}{c+1}"
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                          line=dict(color="rgba(255,255,255,0.35)", width=1),
                          fillcolor=f"rgba(255,165,0,{alpha:.2f})", layer="above")
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2,
                               text=f"<b>{zid}</b><br>{pct:.1f}%",
                               showarrow=False,
                               font=dict(color='white', size=9),
                               bgcolor='rgba(0,0,0,0.45)', borderpad=2)

def stats_quadrante(x_coords, y_coords, velocidades, aceleracoes, x0, x1, y0, y1):
    """Estatísticas detalhadas de um quadrante."""
    xs  = np.array(x_coords)
    ys  = np.array(y_coords)
    vs  = np.array(velocidades)  if velocidades  else np.zeros(len(xs))
    acc = np.array(aceleracoes)  if aceleracoes  else np.zeros(len(xs))
    mask = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    nz   = int(mask.sum())
    n    = len(xs)
    return {
        'n_pontos':  nz,
        'pct':       round(100 * nz / n,  2) if n > 0 else 0.0,
        'vel_media': round(float(vs[mask].mean()),  2) if nz > 0 else 0.0,
        'vel_max':   round(float(vs[mask].max()),   2) if nz > 0 else 0.0,
        'acc_media': round(float(acc[mask].mean()), 3) if nz > 0 else 0.0,
        'acc_max':   round(float(np.abs(acc[mask]).max()), 3) if nz > 0 else 0.0,
    }

@st.cache_data(show_spinner=False)
def gps_para_campo_coords(lats, lons, campo_config):
    """
    Converte pontos GPS (lat/lon) para coordenadas relativas ao campo (x/y em metros,
    origem no canto inferior esquerdo) usando a configuração aplicada.

    É a operação inversa de campo_para_latlon:
      north_m = (lat - c_lat) * 111320
      east_m  = (lon - c_lon) * 111320 * cos(c_lat)
      [x_m, y_m] = R^{-T} * [north_m, east_m]    (rotação inversa)
      field_x = x_m + fl/2  (deslocar para canto inferior esquerdo)
      field_y = y_m + fw/2
    """
    c_lat = float(campo_config['lat'])
    c_lon = float(campo_config['lon'])
    rot   = np.radians(float(campo_config['rot']))
    fl    = float(campo_config['fl'])
    fw    = float(campo_config['fw'])

    lats_a = np.array(lats, dtype=float)
    lons_a = np.array(lons, dtype=float)

    north_m = (lats_a - c_lat) * 111320.0
    east_m  = (lons_a - c_lon) * 111320.0 * np.cos(np.radians(c_lat))

    # Rotação inversa correta (derivada da função JS campo_para_latlon)
    # JS: nM = yO*cos(r) - xO*sin(r) ; eM = yO*sin(r) + xO*cos(r)
    # Inversa: xO = eM*cos(r) - nM*sin(r) ; yO = nM*cos(r) + eM*sin(r)
    x_m = east_m  * np.cos(rot) - north_m * np.sin(rot)   # comprimento do campo
    y_m = north_m * np.cos(rot) + east_m  * np.sin(rot)   # largura do campo

    # Deslocar: centro do campo → canto inferior esquerdo
    field_x = np.clip(x_m + fl / 2, -5, fl + 5)
    field_y = np.clip(y_m + fw / 2, -5, fw + 5)

    return field_x.tolist(), field_y.tolist()

def campo_para_latlon(centro_lat, centro_lon, x_m, y_m, rotacao_deg):
    """
    Converte um ponto em metros relativos ao centro do campo (x para leste, y para norte)
    em coordenadas geográficas (lat, lon), aplicando rotação pelo bearing do campo.
    rotacao_deg: graus clockwise a partir do norte (ex: 90 = campo alinhado leste-oeste).
    """
    rot = np.radians(rotacao_deg)
    # Rotacionar o vetor (x_m, y_m) pelo bearing
    north_m = x_m * np.cos(rot) - y_m * np.sin(rot)
    east_m  = x_m * np.sin(rot) + y_m * np.cos(rot)
    d_lat = float(north_m / 111320.0)
    d_lon = float(east_m / (111320.0 * np.cos(np.radians(float(centro_lat)))))
    return (float(centro_lat) + d_lat, float(centro_lon) + d_lon)

def criar_mapa_satelite_futebol(
    lats, lons, vels, atleta_nome,
    centro_lat, centro_lon, rotacao_deg,
    field_length=105, field_width=68, in_goal=3,
    mostrar_campo=True
):
    """
    Cria um mapa Folium com:
    - Tiles de satélite Esri World Imagery (gratuito, sem chave de API)
    - Trajetória GPS do atleta colorida por faixa de velocidade
    - Overlay do campo de futebol (linhas FIFA + gols dourados) — opcional via mostrar_campo
    """
    if not lats or not lons:
        return None

    # ---- Mapa base ----
    m = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=17,
        tiles=None
    )

    # Satélite Esri (gratuito, sem chave)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🛰️ Satélite",
        overlay=False,
        control=True
    ).add_to(m)

    # Labels sobre o satélite (opcional)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Labels",
        name="🏷️ Rótulos",
        overlay=True,
        control=True,
        opacity=0.7
    ).add_to(m)

    # ---- Overlay do campo de futebol (só quando pedido) ----
    if mostrar_campo:
        cx = field_length / 2
        cy = field_width / 2

        def pt(x_offset, y_offset):
            return campo_para_latlon(centro_lat, centro_lon, y_offset, x_offset, rotacao_deg)

        def linha(pontos_xy):
            return [pt(x - cx, y - cy) for x, y in pontos_xy]

        campo_group = folium.FeatureGroup(name="⚽ Campo de Futebol", show=True)
        FW_h = field_width / 2

        def add_line(pts_xy, color="white", weight=2, dash=None):
            latlon_pts = linha(pts_xy)
            kwargs = dict(color=color, weight=weight, opacity=0.9)
            if dash:
                kwargs["dash_array"] = dash
            folium.PolyLine(latlon_pts, **kwargs).add_to(campo_group)

        # Perímetro
        add_line([(0,0),(field_length,0),(field_length,field_width),(0,field_width),(0,0)], weight=3)
        # Linha central
        add_line([(field_length/2, 0), (field_length/2, field_width)], weight=2)
        # Área de penalidade esquerda (16.5 × 40.32)
        add_line([(0,FW_h-20.16),(16.5,FW_h-20.16),(16.5,FW_h+20.16),(0,FW_h+20.16)])
        # Área de penalidade direita
        add_line([(field_length,FW_h-20.16),(field_length-16.5,FW_h-20.16),
                  (field_length-16.5,FW_h+20.16),(field_length,FW_h+20.16)])
        # Área pequena esquerda (5.5 × 18.32)
        add_line([(0,FW_h-9.16),(5.5,FW_h-9.16),(5.5,FW_h+9.16),(0,FW_h+9.16)])
        # Área pequena direita
        add_line([(field_length,FW_h-9.16),(field_length-5.5,FW_h-9.16),
                  (field_length-5.5,FW_h+9.16),(field_length,FW_h+9.16)])
        # Gol esquerdo (7.32 × 2.44)
        folium.PolyLine([pt(-2.44-cx,FW_h-3.66-cy), pt(0-cx,FW_h-3.66-cy),
                         pt(0-cx,FW_h+3.66-cy), pt(-2.44-cx,FW_h+3.66-cy),
                         pt(-2.44-cx,FW_h-3.66-cy)],
                        color="#FFD700", weight=3, opacity=1).add_to(campo_group)
        # Gol direito
        folium.PolyLine([pt(field_length+2.44-cx,FW_h-3.66-cy), pt(field_length-cx,FW_h-3.66-cy),
                         pt(field_length-cx,FW_h+3.66-cy), pt(field_length+2.44-cx,FW_h+3.66-cy),
                         pt(field_length+2.44-cx,FW_h-3.66-cy)],
                        color="#FFD700", weight=3, opacity=1).add_to(campo_group)
        campo_group.add_to(m)

    # ---- Trajetória GPS colorida por velocidade ----
    # Usa bandas da conta Catapult se disponíveis, senão defaults
    _bv_map = _bandas_vel_ativas()
    BANDAS_VEL_MAP = [
        (b['min'], b['max'] if b['max'] < 9000 else 9999, b['color'], b['label'])
        for b in _bv_map.values()
    ]

    # Subsample para performance (máx 8000 pontos)
    n = len(lats)
    step = max(1, n // 8000)
    lats_s = lats[::step]
    lons_s = lons[::step]
    vels_s = vels[::step]

    traj_group = folium.FeatureGroup(name="🏃 Trajetória GPS", show=True)

    # Desenha segmentos contíguos agrupados por banda
    def cor_banda(v):
        for vmin, vmax, cor, _ in BANDAS_VEL_MAP:
            if v < vmax:
                return cor
        return BANDAS_VEL_MAP[-1][2] if BANDAS_VEL_MAP else "#F44336"

    seg_lats, seg_lons, seg_cor = [lats_s[0]], [lons_s[0]], cor_banda(vels_s[0])
    for i in range(1, len(lats_s)):
        c = cor_banda(vels_s[i])
        if c == seg_cor:
            seg_lats.append(lats_s[i])
            seg_lons.append(lons_s[i])
        else:
            if len(seg_lats) > 1:
                folium.PolyLine(
                    list(zip(seg_lats, seg_lons)),
                    color=seg_cor, weight=3, opacity=0.85
                ).add_to(traj_group)
            seg_lats = [lats_s[i - 1], lats_s[i]]
            seg_lons = [lons_s[i - 1], lons_s[i]]
            seg_cor = c
    if len(seg_lats) > 1:
        folium.PolyLine(
            list(zip(seg_lats, seg_lons)),
            color=seg_cor, weight=3, opacity=0.85
        ).add_to(traj_group)

    # Marcador de início
    folium.CircleMarker(
        location=[lats_s[0], lons_s[0]],
        radius=7, color="white", fill=True, fill_color="#00E676",
        fill_opacity=1, tooltip="▶ Início"
    ).add_to(traj_group)

    # Marcador de fim
    folium.Marker(
        location=[lats_s[-1], lons_s[-1]],
        icon=folium.Icon(color="red", icon="flag"),
        tooltip="⏹ Fim"
    ).add_to(traj_group)

    traj_group.add_to(m)

    # ---- Marcador do centro do campo (sempre visível) ----
    folium.CircleMarker(
        location=[float(centro_lat), float(centro_lon)],
        radius=7, color="white", weight=2,
        fill=True, fill_color="#FFEB3B", fill_opacity=0.95,
        tooltip=f"🎯 Centro do campo — clique no mapa para mover<br>{float(centro_lat):.6f}, {float(centro_lon):.6f}"
    ).add_to(m)

    # ---- Legenda HTML (bandas da conta Catapult) ----
    _leg_rows = ""
    for _vmin, _vmax, _cor, _lbl in BANDAS_VEL_MAP:
        _leg_rows += (
            f'<span style="color:{_cor}">&#9632;</span> {_lbl}<br>'
        )
    legenda_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:rgba(0,0,0,0.75);
                padding:10px 14px;border-radius:8px;color:white;font-size:12px;line-height:1.7;">
      <b>Velocidade</b><br>
      {_leg_rows}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legenda_html))

    folium.LayerControl(collapsed=False).add_to(m)

    return m

def criar_html_campo_interativo(lats_gps, lons_gps, vels_gps, atleta_nome, height=700):
    """
    Gera HTML auto-contido com Leaflet.js:
    - Satélite Esri (gratuito, sem API key)
    - Trajetória GPS colorida por velocidade
    - Campo de futebol interativo: clique para posicionar centro,
      sliders para rotação/tamanho em tempo real
    - Zero re-renders do Streamlit (toda interação é client-side em JavaScript)
    """
    n = len(lats_gps)
    step = max(1, n // 3000)
    pontos = [
        {"lat": round(lats_gps[i], 7),
         "lon": round(lons_gps[i], 7),
         "v":   round(float(vels_gps[i]) if i < len(vels_gps) else 0.0, 1)}
        for i in range(0, n, step)
    ]
    import json as _json
    lat_c  = round(float(np.median(lats_gps)), 7)
    lon_c  = round(float(np.median(lons_gps)), 7)
    pts_js = _json.dumps(pontos)
    aesc   = atleta_nome.replace('"', '\\"').replace("'", "\\'")
    map_h  = height - 115

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        "  <meta charset='utf-8'>\n"
        "  <link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>\n"
        "  <script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>\n"
        "  <style>\n"
        "    *{box-sizing:border-box;margin:0;padding:0}\n"
        "    body{background:#111;font-family:Arial,sans-serif;color:#eee}\n"
        f"    #map{{width:100%;height:{map_h}px}}\n"
        "    #panel{background:rgba(10,10,25,.95);padding:7px 12px;"
        "display:flex;flex-wrap:wrap;gap:12px;align-items:center;"
        "border-top:1px solid #333;min-height:55px}\n"
        "    .ctrl{display:flex;align-items:center;gap:6px;font-size:12px}\n"
        "    .ctrl label{color:#aaa;white-space:nowrap;min-width:110px}\n"
        "    .ctrl input[type=range]{width:100px;accent-color:#2196F3;cursor:pointer}\n"
        "    .val{color:#FFD700;font-weight:bold;min-width:32px;display:inline-block}\n"
        "    .btn{background:#2196F3;color:#fff;border:none;padding:5px 13px;"
        "border-radius:4px;cursor:pointer;font-size:12px;font-weight:bold;white-space:nowrap}\n"
        "    .btn:hover{background:#1565C0}\n"
        "    .btn.on{background:#4CAF50}\n"
        "    .btn.on:hover{background:#2E7D32}\n"
        "    #status{margin-left:auto;color:#90CAF9;font-size:11px;"
        "max-width:260px;text-align:right}\n"
        "    /* Overlay pane totalmente transparente a cliques — map.on('click') sempre dispara */\n"
        "    .leaflet-overlay-pane{pointer-events:none!important}\n"
        "    .leaflet-overlay-pane *{pointer-events:none!important}\n"
        "  </style>\n</head>\n<body>\n"
        "  <div id='map'></div>\n"
        "  <div id='panel'>\n"
        "    <button class='btn' id='btnC' onclick='toggleCampo()'>⚽ Mostrar Campo</button>\n"
        "    <div class='ctrl'>\n"
        f"      <label>📍 Lat centro:</label>\n"
        f"      <input type='number' id='inLat' value='{lat_c}' step='0.00005'\n"
        "        style='width:105px;background:#1a1a2e;color:#FFD700;border:1px solid #555;"
        "padding:3px 5px;border-radius:3px;font-size:12px' oninput='onCenter()'>\n"
        "    </div>\n"
        "    <div class='ctrl'>\n"
        f"      <label>📍 Lon centro:</label>\n"
        f"      <input type='number' id='inLon' value='{lon_c}' step='0.00005'\n"
        "        style='width:105px;background:#1a1a2e;color:#FFD700;border:1px solid #555;"
        "padding:3px 5px;border-radius:3px;font-size:12px' oninput='onCenter()'>\n"
        "    </div>\n"
        "    <div class='ctrl'>\n"
        "      <label>🧭 Rotação: <span class='val' id='rv'>0</span>°</label>\n"
        "      <input type='range' id='rot' min='0' max='359' value='0' oninput='onRot(this.value)'>\n"
        "    </div>\n"
        "    <div class='ctrl'>\n"
        "      <label>📏 Comprimento: <span class='val' id='flv'>105</span>m</label>\n"
        "      <input type='range' id='fl' min='95' max='115' value='105' oninput='onDim()'>\n"
        "    </div>\n"
        "    <div class='ctrl'>\n"
        "      <label>📏 Largura: <span class='val' id='fwv'>68</span>m</label>\n"
        "      <input type='range' id='fw' min='60' max='80' value='68' oninput='onDim()'>\n"
        "    </div>\n"
        "    <div class='ctrl'>\n"
        "      <label>📐 Margem: <span class='val' id='igv'>3</span>m</label>\n"
        "      <input type='range' id='ig' min='0' max='8' value='1' oninput='onDim()'>\n"
        "    </div>\n"
        "    <span id='status'>📍 Ajuste Lat/Lon acima para mover o centro · ↑↓ no teclado = ~5m</span>\n"
        "  </div>\n"
        "  <script>\n"
        f"  const PTS={pts_js};\n"
        f"  const map=L.map('map').setView([{lat_c},{lon_c}],17);\n"
        "  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',\n"
        "    {attribution:'Esri World Imagery',maxZoom:19}).addTo(map);\n"
        "  L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',\n"
        "    {attribution:'Esri Labels',opacity:0.6,maxZoom:19}).addTo(map);\n"
        "  // ---- Trajetória GPS ----\n"
        "  function vc(v){\n"
        "    if(v<7)return '#2196F3';\n"
        "    if(v<14)return '#4CAF50';\n"
        "    if(v<19)return '#FFEB3B';\n"
        "    if(v<24)return '#FF9800';\n"
        "    return '#F44336';\n"
        "  }\n"
        "  let seg=[],sc=null;\n"
        "  for(let i=0;i<PTS.length;i++){\n"
        "    const c=vc(PTS[i].v);\n"
        "    if(c===sc){seg.push([PTS[i].lat,PTS[i].lon]);}\n"
        "    else{\n"
        "      if(seg.length>1)L.polyline(seg,{color:sc,weight:3,opacity:.85,interactive:false}).addTo(map);\n"
        "      seg=i>0?[[PTS[i-1].lat,PTS[i-1].lon],[PTS[i].lat,PTS[i].lon]]:[[PTS[i].lat,PTS[i].lon]];\n"
        "      sc=c;\n"
        "    }\n"
        "  }\n"
        "  if(seg.length>1)L.polyline(seg,{color:sc,weight:3,opacity:.85,interactive:false}).addTo(map);\n"
        "  if(PTS.length>0){\n"
        "    L.circleMarker([PTS[0].lat,PTS[0].lon],\n"
        "      {radius:7,color:'white',fillColor:'#00E676',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "    L.circleMarker([PTS[PTS.length-1].lat,PTS[PTS.length-1].lon],\n"
        "      {radius:7,color:'white',fillColor:'#F44336',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "  }\n"
        "  // ---- Estado do campo ----\n"
        "  let campoOn=false;\n"
        f"  let cLat={lat_c},cLon={lon_c};\n"
        "  let rotD=0,fL=105,fW=68,fI=1;\n"
        "  const cl=L.layerGroup().addTo(map);\n"
        "  // ---- Marcador arrastável (L.Marker com divIcon — não depende de map.on click) ----\n"
        "  const cmIcon=L.divIcon({\n"
        "    className:'',\n"
        "    html:'<div id=\"cmDiv\" style=\"width:24px;height:24px;background:#FFEB3B;"\
        "border:3px solid white;border-radius:50%;cursor:grab;"\
        "box-shadow:0 2px 8px rgba(0,0,0,.7);margin:-12px 0 0 -12px;\"></div>',\n"
        "    iconSize:[24,24],iconAnchor:[12,12]\n"
        "  });\n"
        "  const cm=L.marker([cLat,cLon],{draggable:true,icon:cmIcon,zIndexOffset:1000}).addTo(map);\n"
        "  // Atualiza centro a partir dos inputs numéricos (sempre funciona)\n"
        "  function onCenter(){\n"
        "    const v1=parseFloat(document.getElementById('inLat').value);\n"
        "    const v2=parseFloat(document.getElementById('inLon').value);\n"
        "    if(isNaN(v1)||isNaN(v2))return;\n"
        "    cLat=v1;cLon=v2;\n"
        "    cm.setLatLng([cLat,cLon]);\n"
        "    map.panTo([cLat,cLon]);\n"
        "    drawField();\n"
        "    document.getElementById('status').textContent='🎯 '+cLat.toFixed(5)+', '+cLon.toFixed(5);\n"
        "  }\n"
        "  // Sincroniza inputs quando marcador é arrastado (bônus se funcionar no browser)\n"
        "  function syncInputs(){\n"
        "    document.getElementById('inLat').value=cLat.toFixed(6);\n"
        "    document.getElementById('inLon').value=cLon.toFixed(6);\n"
        "  }\n"
        "  cm.on('drag',function(e){\n"
        "    const ll=e.target.getLatLng();\n"
        "    cLat=ll.lat;cLon=ll.lng;\n"
        "    syncInputs();\n"
        "    if(campoOn)drawField();\n"
        "    document.getElementById('status').textContent='⟳ '+cLat.toFixed(5)+', '+cLon.toFixed(5);\n"
        "  });\n"
        "  cm.on('dragend',function(e){\n"
        "    const ll=e.target.getLatLng();\n"
        "    cLat=ll.lat;cLon=ll.lng;\n"
        "    syncInputs();\n"
        "    drawField();\n"
        "    document.getElementById('status').textContent='🎯 Centro: '+cLat.toFixed(5)+', '+cLon.toFixed(5);\n"
        "  });\n"
        "  // ---- Geometria do campo ----\n"
        "  function toR(d){return d*Math.PI/180;}\n"
        "  function geo(xO,yO){\n"
        "    const r=toR(rotD);\n"
        "    const nM=yO*Math.cos(r)-xO*Math.sin(r);\n"
        "    const eM=yO*Math.sin(r)+xO*Math.cos(r);\n"
        "    return[cLat+nM/111320,cLon+eM/(111320*Math.cos(toR(cLat)))];\n"
        "  }\n"
        "  function lc(pts){return pts.map(p=>geo(p[0]-fL/2,p[1]-fW/2));}\n"
        "  function pl(pts,opt){L.polyline(lc(pts),Object.assign({interactive:false},opt)).addTo(cl);}\n"
        "  function drawField(){\n"
        "    cl.clearLayers();\n"
        "    if(!campoOn)return;\n"
        "    const FL=fL,FW=fW,mg=fI;\n"
        "    const w={color:'white',weight:2,opacity:.9};\n"
        "    const wb={color:'white',weight:3,opacity:.95};\n"
        "    const wd={color:'white',weight:1,opacity:.7,dashArray:'4 4'};\n"
        "    const gd={color:'#FFD700',weight:3,opacity:1,interactive:false};\n"
        "    // Margem exterior (bounding box)\n"
        "    pl([[-mg,-mg],[FL+mg,-mg],[FL+mg,FW+mg],[-mg,FW+mg],[-mg,-mg]],\n"
        "       {color:'white',weight:1,opacity:.4,dashArray:'4 4'});\n"
        "    // Perímetro do campo\n"
        "    pl([[0,0],[FL,0],[FL,FW],[0,FW],[0,0]],wb);\n"
        "    // Linha central\n"
        "    pl([[FL/2,0],[FL/2,FW]],w);\n"
        "    // Área de penalidade esquerda (40.32 x 16.5)\n"
        "    var pa_w=40.32,pa_d=16.5;\n"
        "    var pa_y1=(FW-pa_w)/2,pa_y2=(FW+pa_w)/2;\n"
        "    pl([[0,pa_y1],[pa_d,pa_y1],[pa_d,pa_y2],[0,pa_y2]],w);\n"
        "    // Área de penalidade direita\n"
        "    pl([[FL,pa_y1],[FL-pa_d,pa_y1],[FL-pa_d,pa_y2],[FL,pa_y2]],w);\n"
        "    // Área pequena esquerda (18.32 x 5.5)\n"
        "    var ga_w=18.32,ga_d=5.5;\n"
        "    var ga_y1=(FW-ga_w)/2,ga_y2=(FW+ga_w)/2;\n"
        "    pl([[0,ga_y1],[ga_d,ga_y1],[ga_d,ga_y2],[0,ga_y2]],w);\n"
        "    // Área pequena direita\n"
        "    pl([[FL,ga_y1],[FL-ga_d,ga_y1],[FL-ga_d,ga_y2],[FL,ga_y2]],w);\n"
        "    // Gols (7.32 x 2.44) — dourado\n"
        "    var gw=7.32,gd2=2.44;\n"
        "    var gy1=(FW-gw)/2,gy2=(FW+gw)/2;\n"
        "    L.polyline([geo(-gd2-FL/2,gy1-FW/2),geo(-gd2-FL/2,gy2-FW/2),\n"
        "                geo(-FL/2,gy2-FW/2),geo(-FL/2,gy1-FW/2)],gd).addTo(cl);\n"
        "    L.polyline([geo(FL+gd2-FL/2,gy1-FW/2),geo(FL+gd2-FL/2,gy2-FW/2),\n"
        "                geo(FL-FL/2,gy2-FW/2),geo(FL-FL/2,gy1-FW/2)],gd).addTo(cl);\n"
        "    // Círculo central (r=9.15)\n"
        "    var N=48,r=9.15;\n"
        "    var cpts=[];\n"
        "    for(var i=0;i<=N;i++){\n"
        "      var a=2*Math.PI*i/N;\n"
        "      cpts.push(geo(FL/2+r*Math.sin(a)-FL/2, FW/2+r*Math.cos(a)-FW/2));\n"
        "    }\n"
        "    L.polyline(cpts,w).addTo(cl);\n"
        "    // Ponto central\n"
        "    L.circleMarker(geo(0,0),{radius:3,color:'white',fillColor:'white',\n"
        "      fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "    // Pontos de penalidade (11m)\n"
        "    L.circleMarker(geo(11-FL/2,-FW/2),{radius:3,color:'white',fillColor:'white',\n"
        "      fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "    L.circleMarker(geo(FL-11-FL/2,-FW/2),{radius:3,color:'white',fillColor:'white',\n"
        "      fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "    // Arcos de penalidade (fora da área)\n"
        "    var rA=9.15,pts1=[],pts2=[];\n"
        "    for(var i=0;i<=60;i++){\n"
        "      var a=-Math.PI/2+Math.PI*i/60;\n"
        "      var px1=11+rA*Math.cos(a);\n"
        "      var py1=FW/2+rA*Math.sin(a);\n"
        "      if(px1>pa_d){pts1.push(geo(px1-FL/2,py1-FW/2));}\n"
        "      var px2=FL-11+rA*Math.cos(Math.PI-a);\n"
        "      var py2=FW/2+rA*Math.sin(a);\n"
        "      if(px2<FL-pa_d){pts2.push(geo(px2-FL/2,py2-FW/2));}\n"
        "    }\n"
        "    if(pts1.length>1)L.polyline(pts1,w).addTo(cl);\n"
        "    if(pts2.length>1)L.polyline(pts2,w).addTo(cl);\n"
        "    // Arcos de canto (r=1)\n"
        "    var rc=1,corners=[[0,0],[FL,0],[0,FW],[FL,FW]];\n"
        "    var aStart=[[0,Math.PI/2],[Math.PI/2,Math.PI],[3*Math.PI/2,2*Math.PI],\n"
        "                [Math.PI,3*Math.PI/2]];\n"
        "    corners.forEach(function(c,i){\n"
        "      var cpts2=[],a0=aStart[i][0],a1=aStart[i][1];\n"
        "      for(var j=0;j<=12;j++){\n"
        "        var a=a0+(a1-a0)*j/12;\n"
        "        cpts2.push(geo(c[0]+rc*Math.cos(a)-FL/2,c[1]+rc*Math.sin(a)-FW/2));\n"
        "      }\n"
        "      L.polyline(cpts2,w).addTo(cl);\n"
        "    });\n"
        "  }\n"
        "  function onRot(v){rotD=+v;document.getElementById('rv').textContent=v;drawField();}\n"
        "  function onDim(){\n"
        "    fL=+document.getElementById('fl').value;\n"
        "    fW=+document.getElementById('fw').value;\n"
        "    fI=+document.getElementById('ig').value;\n"
        "    document.getElementById('flv').textContent=fL;\n"
        "    document.getElementById('fwv').textContent=fW;\n"
        "    document.getElementById('igv').textContent=fI;\n"
        "    drawField();\n"
        "  }\n"
        "  function toggleCampo(){\n"
        "    campoOn=!campoOn;\n"
        "    const b=document.getElementById('btnC');\n"
        "    b.textContent=campoOn?'⚽ Ocultar Campo':'⚽ Mostrar Campo';\n"
        "    b.className=campoOn?'btn on':'btn';\n"
        "    drawField();\n"
        "    document.getElementById('status').textContent=campoOn\n"
        "      ?'✅ Arraste ⊙ para mover · sliders para ajustar em tempo real'\n"
        "      :'⊙ Arraste o marcador amarelo para o centro do campo';\n"
        "  }\n"
        "  const leg=L.control({position:'bottomleft'});\n"
        "  leg.onAdd=function(){\n"
        "    const d=L.DomUtil.create('div');\n"
        "    d.style='background:rgba(0,0,0,.78);padding:8px 11px;border-radius:6px;color:#fff;font-size:11px;line-height:1.9';\n"
        "    d.innerHTML=" + _legenda_vel_js() + ";\n"
        "    L.DomEvent.disableClickPropagation(d);\n"
        "    return d;\n"
        "  };\n"
        "  leg.addTo(map);\n"
        "  </script>\n</body>\n</html>"
    )
    return html

def criar_html_campo_fixo(lats_gps, lons_gps, vels_gps, campo_config,
                          lats_eff=None, lons_eff=None, vels_eff=None,
                          atleta_nome="", esforco_desc="", height=580):
    """
    Mapa satélite com campo de futebol FIXO (já configurado e aplicado).
    - Trajetória GPS completa como fundo (opacidade baixa)
    - Campo de futebol desenhado na posição salva
    - Se lats_eff fornecido: destaca os pontos do esforço selecionado (linha grossa)
    """
    import json as _json

    # Sub-amostra o trace de fundo (máx 3000 pontos)
    n = len(lats_gps)
    step = max(1, n // 3000)
    pontos = [
        {"lat": round(lats_gps[i], 7),
         "lon": round(lons_gps[i], 7),
         "v":   round(float(vels_gps[i]) if i < len(vels_gps) else 0.0, 1)}
        for i in range(0, n, step)
    ]

    # Pontos do esforço destacado
    eff_pontos = []
    if lats_eff and lons_eff:
        eff_pontos = [
            {"lat": round(lats_eff[i], 7),
             "lon": round(lons_eff[i], 7),
             "v":   round(float(vels_eff[i]) if vels_eff and i < len(vels_eff) else 0.0, 1)}
            for i in range(len(lats_eff))
        ]

    cLat = campo_config['lat']
    cLon = campo_config['lon']
    rotD = campo_config['rot']
    fL   = campo_config['fl']
    fW   = campo_config['fw']
    fI   = campo_config['ig']

    pts_js   = _json.dumps(pontos)
    eff_js   = _json.dumps(eff_pontos)
    aesc     = atleta_nome.replace('"', '\\"').replace("'", "\\'")
    desc_esc = esforco_desc.replace('"', '\\"').replace("'", "\\'")
    map_h    = height - 44

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        "  <meta charset='utf-8'>\n"
        "  <link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>\n"
        "  <script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>\n"
        "  <style>\n"
        "    *{box-sizing:border-box;margin:0;padding:0}\n"
        "    body{background:#111;font-family:Arial,sans-serif;color:#eee}\n"
        f"    #map{{width:100%;height:{map_h}px}}\n"
        "    #infobar{background:rgba(10,10,25,.93);padding:6px 14px;display:flex;"
        "align-items:center;gap:14px;border-top:1px solid #2a2a3a;min-height:44px;font-size:12px}\n"
        "    .lk{color:#4CAF50;font-weight:bold;font-size:13px}\n"
        "    .leaflet-overlay-pane{pointer-events:none!important}\n"
        "    .leaflet-overlay-pane *{pointer-events:none!important}\n"
        "  </style>\n</head>\n<body>\n"
        "  <div id='map'></div>\n"
        "  <div id='infobar'>\n"
        "    <span class='lk'>🔒 Campo Aplicado</span>\n"
        f"    <span style='color:#90CAF9'>📍 {cLat:.5f}, {cLon:.5f}</span>\n"
        f"    <span style='color:#aaa'>🧭 {rotD}°</span>\n"
        f"    <span style='color:#aaa'>📏 {fL}×{fW}m + margem {fI}m (FIFA)</span>\n"
        "    <span id='effinfo' style='color:#FFD700;margin-left:auto;font-weight:bold'></span>\n"
        "  </div>\n"
        "  <script>\n"
        f"  const PTS={pts_js};\n"
        f"  const EFF={eff_js};\n"
        f"  const cLat={cLat},cLon={cLon},rotD={rotD},fL={fL},fW={fW},fI={fI};\n"
        f"  const map=L.map('map').setView([cLat,cLon],17);\n"
        "  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',\n"
        "    {attribution:'Esri World Imagery',maxZoom:19}).addTo(map);\n"
        "  L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',\n"
        "    {attribution:'Esri Labels',opacity:0.6,maxZoom:19}).addTo(map);\n"
        "  function vc(v){\n"
        "    if(v<7)return '#2196F3';if(v<14)return '#4CAF50';\n"
        "    if(v<19)return '#FFEB3B';if(v<24)return '#FF9800';return '#F44336';\n"
        "  }\n"
        "  // ---- Trace de fundo (opacidade baixa) ----\n"
        "  let seg=[],sc=null;\n"
        "  for(let i=0;i<PTS.length;i++){\n"
        "    const c=vc(PTS[i].v);\n"
        "    if(c===sc){seg.push([PTS[i].lat,PTS[i].lon]);}\n"
        "    else{\n"
        "      if(seg.length>1)L.polyline(seg,{color:sc,weight:2,opacity:.28,interactive:false}).addTo(map);\n"
        "      seg=i>0?[[PTS[i-1].lat,PTS[i-1].lon],[PTS[i].lat,PTS[i].lon]]:[[PTS[i].lat,PTS[i].lon]];\n"
        "      sc=c;\n"
        "    }\n"
        "  }\n"
        "  if(seg.length>1)L.polyline(seg,{color:sc,weight:2,opacity:.28,interactive:false}).addTo(map);\n"
        "  // ---- Esforço destacado (linha grossa) ----\n"
        f"  const effDesc='{desc_esc}';\n"
        "  if(EFF.length>0){\n"
        "    let es=[],ec=null;\n"
        "    for(let i=0;i<EFF.length;i++){\n"
        "      const c=vc(EFF[i].v);\n"
        "      if(c===ec){es.push([EFF[i].lat,EFF[i].lon]);}\n"
        "      else{\n"
        "        if(es.length>1)L.polyline(es,{color:ec,weight:7,opacity:1,interactive:false}).addTo(map);\n"
        "        es=i>0?[[EFF[i-1].lat,EFF[i-1].lon],[EFF[i].lat,EFF[i].lon]]:[[EFF[i].lat,EFF[i].lon]];\n"
        "        ec=c;\n"
        "      }\n"
        "    }\n"
        "    if(es.length>1)L.polyline(es,{color:ec,weight:7,opacity:1,interactive:false}).addTo(map);\n"
        "    L.circleMarker([EFF[0].lat,EFF[0].lon],\n"
        "      {radius:9,color:'white',fillColor:'#00E676',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "    L.circleMarker([EFF[EFF.length-1].lat,EFF[EFF.length-1].lon],\n"
        "      {radius:9,color:'white',fillColor:'#F44336',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "    const midIdx=Math.floor(EFF.length/2);\n"
        "    map.setView([EFF[midIdx].lat,EFF[midIdx].lon],18);\n"
        "    document.getElementById('effinfo').textContent='⚡ '+effDesc;\n"
        "  } else {\n"
        "    if(PTS.length>0){\n"
        "      L.circleMarker([PTS[0].lat,PTS[0].lon],\n"
        "        {radius:6,color:'white',fillColor:'#00E676',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "      L.circleMarker([PTS[PTS.length-1].lat,PTS[PTS.length-1].lon],\n"
        "        {radius:6,color:'white',fillColor:'#F44336',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "    }\n"
        "  }\n"
        "  // ---- Campo fixo ----\n"
        "  const cl=L.layerGroup().addTo(map);\n"
        "  function toR(d){return d*Math.PI/180;}\n"
        "  function geo(xO,yO){\n"
        "    const r=toR(rotD);\n"
        "    const nM=yO*Math.cos(r)-xO*Math.sin(r);\n"
        "    const eM=yO*Math.sin(r)+xO*Math.cos(r);\n"
        "    return[cLat+nM/111320,cLon+eM/(111320*Math.cos(toR(cLat)))];\n"
        "  }\n"
        "  function lc(pts){return pts.map(p=>geo(p[0]-fL/2,p[1]-fW/2));}\n"
        "  function pl(pts,opt){L.polyline(lc(pts),Object.assign({interactive:false},opt)).addTo(cl);}\n"
        "  (function(){\n"
        "    var w={color:'white',weight:2,opacity:.9};\n"
        "    var wb={color:'white',weight:3,opacity:.9};\n"
        "    var wd={color:'white',weight:1.5,opacity:.9};\n"
        "    var gd={color:'#FFD700',weight:3,opacity:1,interactive:false};\n"
        "    var i,a,pts,px,px2;\n"
        "    pl([[0,0],[fL,0],[fL,fW],[0,fW],[0,0]],wb);\n"
        "    pl([[fL/2,0],[fL/2,fW]],w);\n"
        "    pl([[0,fW/2-20.16],[16.5,fW/2-20.16],[16.5,fW/2+20.16],[0,fW/2+20.16]],w);\n"
        "    pl([[fL,fW/2-20.16],[fL-16.5,fW/2-20.16],[fL-16.5,fW/2+20.16],[fL,fW/2+20.16]],w);\n"
        "    pl([[0,fW/2-9.16],[5.5,fW/2-9.16],[5.5,fW/2+9.16],[0,fW/2+9.16]],w);\n"
        "    pl([[fL,fW/2-9.16],[fL-5.5,fW/2-9.16],[fL-5.5,fW/2+9.16],[fL,fW/2+9.16]],w);\n"
        "    pl([[-2.44,fW/2-3.66],[0,fW/2-3.66],[0,fW/2+3.66],[-2.44,fW/2+3.66],[-2.44,fW/2-3.66]],gd);\n"
        "    pl([[fL+2.44,fW/2-3.66],[fL,fW/2-3.66],[fL,fW/2+3.66],[fL+2.44,fW/2+3.66],[fL+2.44,fW/2-3.66]],gd);\n"
        "    pts=[];\n"
        "    for(i=0;i<=60;i++){a=i/60*2*Math.PI;pts.push([fL/2+9.15*Math.cos(a),fW/2+9.15*Math.sin(a)]);}\n"
        "    pl(pts,wd);\n"
        "    pts=[];\n"
        "    for(i=0;i<=60;i++){a=-Math.PI+i/60*2*Math.PI;px=11+9.15*Math.cos(a);if(px>16.5)pts.push([px,fW/2+9.15*Math.sin(a)]);}\n"
        "    if(pts.length>1)pl(pts,wd);\n"
        "    pts=[];\n"
        "    for(i=0;i<=60;i++){a=i/60*2*Math.PI;px2=(fL-11)+9.15*Math.cos(a);if(px2<fL-16.5)pts.push([px2,fW/2+9.15*Math.sin(a)]);}\n"
        "    if(pts.length>1)pl(pts,wd);\n"
        "    var corners=[[0,0,0,Math.PI/2],[fL,0,Math.PI/2,Math.PI],[fL,fW,Math.PI,3*Math.PI/2],[0,fW,3*Math.PI/2,2*Math.PI]];\n"
        "    for(var ci=0;ci<corners.length;ci++){var cx2=corners[ci][0],cy2=corners[ci][1],a1=corners[ci][2],a2=corners[ci][3];pts=[];for(i=0;i<=10;i++){a=a1+(a2-a1)*i/10;pts.push([cx2+Math.cos(a),cy2+Math.sin(a)]);}pl(pts,Object.assign({},w,{weight:1}));}\n"
        "    L.circleMarker(geo(0,0),{radius:3,color:'white',fillColor:'white',fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "    L.circleMarker(geo(11-fL/2,0),{radius:3,color:'white',fillColor:'white',fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "    L.circleMarker(geo(fL/2-11,0),{radius:3,color:'white',fillColor:'white',fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "  })();\n"
        "  // ---- Legenda ----\n"
        "  const leg=L.control({position:'bottomleft'});\n"
        "  leg.onAdd=function(){\n"
        "    const d=L.DomUtil.create('div');\n"
        "    d.style='background:rgba(0,0,0,.78);padding:8px 11px;border-radius:6px;color:#fff;font-size:11px;line-height:1.9';\n"
        "    d.innerHTML=" + _legenda_vel_js() + ";\n"
        "    L.DomEvent.disableClickPropagation(d);\n"
        "    return d;\n"
        "  };\n"
        "  leg.addTo(map);\n"
        "  </script>\n</body>\n</html>"
    )
    return html

def lat_lon_to_xy(latitudes, longitudes):
    if len(latitudes) == 0:
        return [], []

    lat_ref = latitudes[0]
    lon_ref = longitudes[0]
    R = 6371000

    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    ref_lat_rad = np.radians(lat_ref)
    ref_lon_rad = np.radians(lon_ref)
    
    x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
    y = R * (lat_rad - ref_lat_rad)
    
    return x, y

def gerar_heatmap_segmentado(xs, ys, ts_list, bloco_min, bloco_idx, field_length=105, field_width=68):
    """Heatmap de presença para um bloco temporal específico da partida."""
    if not ts_list or not xs or not ys:
        return None, 0, ""

    ts_arr = np.array(ts_list, dtype=float)
    xs_arr = np.array(xs,      dtype=float)
    ys_arr = np.array(ys,      dtype=float)

    ts_rel  = ts_arr - ts_arr.min()
    bloco_s = bloco_min * 60.0
    t0, t1  = bloco_idx * bloco_s, (bloco_idx + 1) * bloco_s
    label   = f"{int(t0 // 60)}–{int(t1 // 60)} min"
    mascara = (ts_rel >= t0) & (ts_rel < t1)

    xb = xs_arr[mascara]
    yb = ys_arr[mascara]
    # Filtra ambos com a mesma máscara para garantir mesmo comprimento
    _valido = (xb >= 0) & (xb <= field_length) & (yb >= 0) & (yb <= field_width)
    xb = xb[_valido]
    yb = yb[_valido]

    if len(xb) < 5:
        return None, int(mascara.sum()), label

    H, xe, ye = np.histogram2d(xb, yb, bins=[52, 34],
                                range=[[0, field_length], [0, field_width]])
    H = _gf(H, sigma=2.0)
    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2

    fig = desenhar_campo_futebol_bonito(field_length, field_width,
                                        title=f"🕐 Heatmap por Fase — {label}")
    fig.add_trace(go.Heatmap(
        x=xc, y=yc, z=H.T,
        colorscale=[[0, 'rgba(0,0,0,0)'],  [0.0001, '#1a237e'],
                    [0.3, '#1565C0'],        [0.6,    '#FFEB3B'],
                    [0.85,'#FF9800'],        [1,      '#F44336']],
        opacity=0.72, showscale=True,
        colorbar=dict(
            title=dict(text='Freq.', font=dict(color='white')),
            tickfont=dict(color='white'),
            x=1.01, thickness=12
        ),
        name=f'Bloco {bloco_idx + 1}',
        hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Freq: %{z:.0f}<extra></extra>'
    ))
    return fig, int(mascara.sum()), label

def enriquecer_esforcos_taticos(df: pd.DataFrame,
                               coords_atletas: dict,
                               field_length: float = 105) -> pd.DataFrame:
    """
    Adiciona colunas 'Zona' e 'Direção' ao DataFrame de esforços com
    base nas coordenadas x/y do atleta:
      - Zona: Defensivo / Meio / Ataque (terços do campo)
      - Direção: ⬆ Ofensivo / ⬇ Defensivo / ↔ Lateral
    """
    if df.empty:
        return df

    zonas, direcoes = [], []
    fl3 = field_length / 3.0

    for _, row in df.iterrows():
        atl  = str(row.get('Atleta', ''))
        c    = coords_atletas.get(atl, {})
        xn   = c.get('xn', [])
        tsn  = c.get('ts', [])
        x_ini = x_fim = None

        # Tier 0: índices de segmento (cálculo local)
        si = int(row.get('_seg_start_idx', -1))
        ei = int(row.get('_seg_end_idx',   -1))
        if 0 <= si < ei <= len(xn):
            x_ini = float(xn[si])
            x_fim = float(xn[ei - 1])

        # Tier 1: timestamps (esforços da API)
        if x_ini is None and tsn and row.get('_start_ts', 0) > 0:
            _ts_a = np.array(tsn)
            _msk  = (_ts_a >= row['_start_ts']) & (_ts_a <= row['_end_ts'])
            if _msk.any():
                _idxs = np.where(_msk)[0]
                x_ini = float(xn[_idxs[0]])
                x_fim = float(xn[_idxs[-1]])

        # Zona
        if x_ini is not None:
            if x_ini < fl3:
                zona = '🔴 Defensivo'
            elif x_ini < 2 * fl3:
                zona = '🟡 Meio'
            else:
                zona = '🟢 Ataque'
        else:
            zona = '—'

        # Direção
        if x_ini is not None and x_fim is not None:
            dx = x_fim - x_ini
            if dx > 5:
                direcao = '⬆ Ofensivo'
            elif dx < -5:
                direcao = '⬇ Defensivo'
            else:
                direcao = '↔ Lateral'
        else:
            direcao = '—'

        zonas.append(zona)
        direcoes.append(direcao)

    df = df.copy()
    df.insert(df.columns.get_loc('Banda') + 1 if 'Banda' in df.columns else len(df.columns),
              'Zona', zonas)
    df.insert(df.columns.get_loc('Zona') + 1, 'Direção', direcoes)
    return df

def calcular_voronoi_campo(posicoes_atletas, field_length=105, field_width=68, resolucao=1.5):
    """Diagrama de Voronoi (nearest-neighbor grid) — raio de ação por atleta."""
    centroids = {}
    for nome, dados in posicoes_atletas.items():
        xs = [x for x in dados.get('xs', []) if 0 <= x <= field_length]
        ys = [y for y in dados.get('ys', []) if 0 <= y <= field_width]
        if len(xs) > 10:
            centroids[nome] = (float(np.median(xs)), float(np.median(ys)))

    if len(centroids) < 2:
        return None

    names = list(centroids.keys())
    cx    = np.array([centroids[n][0] for n in names])
    cy    = np.array([centroids[n][1] for n in names])
    n     = len(names)

    gx = np.arange(0, field_length + resolucao, resolucao)
    gy = np.arange(0, field_width  + resolucao, resolucao)
    GX, GY = np.meshgrid(gx, gy)
    pts_flat = np.column_stack([GX.ravel(), GY.ravel()])
    tree     = cKDTree(np.column_stack([cx, cy]))
    _, zone_flat = tree.query(pts_flat)
    Z = zone_flat.reshape(GX.shape).astype(float)

    base_colors = ['#2196F3','#F44336','#4CAF50','#FF9800',
                   '#9C27B0','#00BCD4','#FFEB3B','#E91E63',
                   '#FF5722','#607D8B']
    cs = []
    for i in range(n):
        lo = i / n
        hi = min((i + 1) / n, 1.0)
        c  = base_colors[i % len(base_colors)]
        cs.append([lo, c])
        cs.append([hi, c])
    cs[0][0]  = 0.0
    cs[-1][0] = 1.0

    fig = desenhar_campo_futebol_bonito(field_length, field_width,
                                        title="🔷 Voronoi — Raio de Ação por Atleta")
    fig.add_trace(go.Heatmap(
        x=gx, y=gy, z=Z,
        colorscale=cs, opacity=0.38, showscale=False,
        zmin=0, zmax=max(n - 0.001, 1),
        hoverinfo='skip', name='Zonas'
    ))
    for i, nome in enumerate(names):
        cx_i, cy_i = centroids[nome]
        fig.add_trace(go.Scatter(
            x=[cx_i], y=[cy_i],
            mode='markers+text',
            marker=dict(size=14, color=base_colors[i % len(base_colors)],
                        symbol='diamond', line=dict(width=2, color='white')),
            text=[nome.split()[0]], textposition='top center',
            textfont=dict(color='white', size=10, family='Arial Black'),
            name=nome, showlegend=True
        ))
    fig.update_layout(
        legend=dict(bgcolor='rgba(0,0,30,.75)', font=dict(color='white'),
                    bordercolor='#555', borderwidth=1)
    )
    return fig

def calcular_carga_neuromuscular(sensor_points, limiar=2.0, min_dur_s=None):
    """Analisa esforços de acc/dec intensos como indicador de carga neuromuscular.
    Conta EVENTOS (entradas na zona sustentadas por min_dur_s), não amostras.
    """
    if not sensor_points:
        return None

    if min_dur_s is None:
        min_dur_s = get_min_dur_s()

    ts_l, acc_l, vel_l = [], [], []
    for p in sensor_points:
        if p.get('a') is not None and p.get('ts') is not None:
            ts_l.append(float(p['ts']))
            acc_l.append(float(p['a']))
            vel_l.append(float(p.get('v') or 0) * 3.6)

    if len(ts_l) < 10:
        return None

    ts_arr  = np.array(ts_l,  dtype=float)
    acc_arr = np.array(acc_l, dtype=float)
    vel_arr = np.array(vel_l, dtype=float)
    ts_rel  = ts_arr - ts_arr.min()
    duracao = float(ts_rel.max())

    lm = limiar * 0.65  # limiar médio

    # Máscaras de EVENTOS (um True por evento, no frame que completa a duração mínima)
    mask_hi_acc  = detectar_eventos_acc(acc_arr,  limiar, min_dur_s=min_dur_s, acima=True)
    mask_hi_dec  = detectar_eventos_acc(acc_arr,  limiar, min_dur_s=min_dur_s, acima=False)
    mask_med_acc = detectar_eventos_acc(acc_arr,  lm,    min_dur_s=min_dur_s, acima=True)  & ~mask_hi_acc
    mask_med_dec = detectar_eventos_acc(acc_arr,  lm,    min_dur_s=min_dur_s, acima=False) & ~mask_hi_dec

    t_bins = np.arange(0, duracao + 60, 60)
    n_bins = max(1, len(t_bins) - 1)

    def _epm(mask):
        """Eventos por minuto em cada bin."""
        return np.array([
            mask[(ts_rel >= t_bins[i]) & (ts_rel < t_bins[i + 1])].sum()
            for i in range(n_bins)
        ])

    t_mid = [(t_bins[i] + t_bins[i + 1]) / 2 / 60 for i in range(n_bins)]
    return {
        'ts_rel': ts_rel, 'acc': acc_arr, 'vel': vel_arr, 't_mid': t_mid,
        'hi_acc_min':  _epm(mask_hi_acc),  'hi_dec_min':  _epm(mask_hi_dec),
        'med_acc_min': _epm(mask_med_acc), 'med_dec_min': _epm(mask_med_dec),
        'total_hi_acc':  int(mask_hi_acc.sum()),  'total_hi_dec':  int(mask_hi_dec.sum()),
        'total_med_acc': int(mask_med_acc.sum()), 'total_med_dec': int(mask_med_dec.sum()),
        'limiar': limiar,
        'min_dur_s': min_dur_s,
    }

@st.cache_data(show_spinner=False, max_entries=16)
def _neuro_cached(cache_key, _sensor_points, limiar, min_dur_s):
    """(P6) Cache da análise neuromuscular: os ~50k pontos ficam FORA da chave
    (prefixo _); a chave pequena identifica token/atividade/período/atleta/
    parâmetros. Evita redetectar eventos a cada interação de widget."""
    return calcular_carga_neuromuscular(_sensor_points, limiar=limiar,
                                        min_dur_s=min_dur_s)

def plotar_carga_neuromuscular(dados, atleta_nome):
    """Painel Plotly 2×2 com análise de carga neuromuscular."""
    lim      = dados['limiar']
    t        = dados['t_mid']
    _dur_lbl = f"{dados.get('min_dur_s', _DEFAULT_MIN_DUR_S):.1f}s"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"🟢 Acelerações ≥{lim} m/s² / min",
            f"🔴 Desacelerações ≥{lim} m/s² / min",
            "📊 Aceleração × Velocidade",
            "⚡ Carga Neuromuscular Acumulada"
        ),
        vertical_spacing=0.18, horizontal_spacing=0.1
    )
    # Acelerações
    fig.add_trace(go.Bar(x=t, y=dados['hi_acc_min'],  name=f'Alta Acc ≥{lim}',
                         marker_color='#4CAF50', opacity=0.9), row=1, col=1)
    fig.add_trace(go.Bar(x=t, y=dados['med_acc_min'], name='Média Acc',
                         marker_color='#81C784', opacity=0.6), row=1, col=1)
    # Desacelerações
    fig.add_trace(go.Bar(x=t, y=dados['hi_dec_min'],  name=f'Alta Dec ≥{lim}',
                         marker_color='#F44336', opacity=0.9), row=1, col=2)
    fig.add_trace(go.Bar(x=t, y=dados['med_dec_min'], name='Média Dec',
                         marker_color='#E57373', opacity=0.6), row=1, col=2)
    # Scatter acc × vel
    step = max(1, len(dados['ts_rel']) // 2000)
    ac_s = dados['acc'][::step];  vl_s = dados['vel'][::step]
    c_s  = np.where(ac_s >= lim, '#4CAF50', np.where(ac_s <= -lim, '#F44336', '#90CAF9'))
    fig.add_trace(go.Scatter(
        x=vl_s, y=ac_s, mode='markers',
        marker=dict(size=3, color=c_s, opacity=0.5),
        name='Acc × Vel', showlegend=False
    ), row=2, col=1)
    fig.add_hline(y= lim, line_dash='dash', line_color='#4CAF50', opacity=0.6, row=2, col=1)
    fig.add_hline(y=-lim, line_dash='dash', line_color='#F44336', opacity=0.6, row=2, col=1)
    fig.add_hline(y=0,    line_color='white', opacity=0.3, row=2, col=1)
    # Carga acumulada
    dt       = np.diff(dados['ts_rel'], prepend=dados['ts_rel'][0])
    dt       = np.clip(dt, 0, 1)
    carga    = np.cumsum(np.abs(dados['acc']) * dt)
    step2    = max(1, len(carga) // 1500)
    fig.add_trace(go.Scatter(
        x=dados['ts_rel'][::step2] / 60, y=carga[::step2],
        mode='lines', line=dict(color='#FFEB3B', width=2),
        fill='tozeroy', fillcolor='rgba(255,235,59,0.12)',
        name='Carga Acum.'
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text=f'💪 Carga Neuromuscular — {atleta_nome}  (dur. mín. {_dur_lbl})',
                   font=dict(size=16, color='white')),
        height=620, paper_bgcolor='#0a0a1e', plot_bgcolor='#1a1a2e',
        font=dict(color='white'), barmode='stack',
        legend=dict(bgcolor='rgba(0,0,0,.5)', font=dict(color='white'))
    )
    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(gridcolor='#333', row=r, col=c)
            fig.update_yaxes(gridcolor='#333', row=r, col=c)
    fig.update_xaxes(title_text='Tempo (min)',        row=1, col=1)
    fig.update_xaxes(title_text='Tempo (min)',        row=1, col=2)
    fig.update_xaxes(title_text='Velocidade (km/h)',  row=2, col=1)
    fig.update_xaxes(title_text='Tempo (min)',        row=2, col=2)
    fig.update_yaxes(title_text='Contagem / min',     row=1, col=1)
    fig.update_yaxes(title_text='Contagem / min',     row=1, col=2)
    fig.update_yaxes(title_text='Aceleração (m/s²)',  row=2, col=1)
    fig.update_yaxes(title_text='Carga Acumulada',    row=2, col=2)
    return fig

def calcular_acwr_df(df_cargas):
    """
    ACWR (Acute:Chronic Workload Ratio) por atleta e data.
    df_cargas: DataFrame com colunas 'atleta', 'data' (datetime), 'player_load', 'atividade'.
    """
    resultados = []
    for atleta in df_cargas['atleta'].unique():
        df_a = df_cargas[df_cargas['atleta'] == atleta].sort_values('data')
        for _, row in df_a.iterrows():
            d_ref  = row['data']
            aguda  = df_a[(df_a['data'] <= d_ref) &
                           (df_a['data'] >  d_ref - timedelta(days=7))]['player_load'].sum()
            cronica = df_a[(df_a['data'] <= d_ref) &
                            (df_a['data'] >  d_ref - timedelta(days=28))]['player_load'].sum()
            cr_sem = cronica / 4 if cronica > 0 else 0
            acwr   = round(aguda / cr_sem, 3) if cr_sem > 0 else None
            resultados.append({
                'Atleta': atleta,
                'Data': d_ref,
                'Atividade': row.get('atividade', ''),
                'PlayerLoad': round(row['player_load'], 1),
                'Carga Aguda 7d': round(aguda, 1),
                'Carga Crônica 28d': round(cronica, 1),
                'ACWR': acwr,
            })
    return pd.DataFrame(resultados)

def plotar_acwr(df_acwr):
    """Gráfico de ACWR com zonas de risco coloridas."""
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('📊 PlayerLoad por Sessão', '🎯 ACWR — Índice de Carga Aguda/Crônica'),
                        vertical_spacing=0.15, shared_xaxes=True)
    cores = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0',
             '#00BCD4','#FFEB3B','#E91E63','#FF5722','#607D8B']
    atletas = df_acwr['Atleta'].unique()

    for i, atl in enumerate(atletas):
        df_a = df_acwr[df_acwr['Atleta'] == atl].copy()
        c    = cores[i % len(cores)]
        fig.add_trace(go.Bar(
            x=df_a['Data'], y=df_a['PlayerLoad'],
            name=atl, marker_color=c, opacity=0.8,
            hovertemplate='%{x|%d/%m/%y}<br>PL: %{y:.0f}<extra>' + atl + '</extra>'
        ), row=1, col=1)
        df_acwr_v = df_a.dropna(subset=['ACWR'])
        if not df_acwr_v.empty:
            fig.add_trace(go.Scatter(
                x=df_acwr_v['Data'], y=df_acwr_v['ACWR'],
                mode='lines+markers', name=atl + ' ACWR',
                line=dict(color=c, width=2), marker=dict(size=6),
                showlegend=False,
                hovertemplate='%{x|%d/%m/%y}<br>ACWR: %{y:.2f}<extra>' + atl + '</extra>'
            ), row=2, col=1)

    # Zonas de risco no gráfico ACWR
    acwr_vals = df_acwr.dropna(subset=['ACWR'])['ACWR']
    y_max = max(float(acwr_vals.max()) * 1.2, 2.0) if len(acwr_vals) else 2.0
    for y0, y1, cor, label in [
        (0.0, 0.8,  'rgba(33,150,243,.10)',  'Subcarregado'),
        (0.8, 1.3,  'rgba(76,175,80,.15)',   '✅ Zona Ótima'),
        (1.3, 1.5,  'rgba(255,152,0,.15)',   '⚠️ Atenção'),
        (1.5, y_max,'rgba(244,67,54,.15)',   '🔴 Risco'),
    ]:
        fig.add_hrect(y0=y0, y1=min(y1, y_max), fillcolor=cor,
                      line_width=0, row=2, col=1, annotation_text=label,
                      annotation_position='right',
                      annotation_font=dict(size=9, color='white'))
    fig.add_hline(y=1.5, line_dash='dash', line_color='#F44336', opacity=0.8, row=2, col=1)
    fig.add_hline(y=1.3, line_dash='dash', line_color='#FF9800', opacity=0.7, row=2, col=1)
    fig.add_hline(y=0.8, line_dash='dash', line_color='#2196F3', opacity=0.6, row=2, col=1)

    fig.update_layout(
        height=650, paper_bgcolor='#0a0a1e', plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        legend=dict(bgcolor='rgba(0,0,0,.5)', font=dict(color='white')),
        barmode='group',
        xaxis2=dict(title='Data', gridcolor='#333'),
        yaxis=dict(title='PlayerLoad', gridcolor='#333'),
        yaxis2=dict(title='ACWR', gridcolor='#333'),
    )
    return fig
