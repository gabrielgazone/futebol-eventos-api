# -*- coding: utf-8 -*-
"""Aba Tática Coletiva (P4 — render_* extraída para viz/).

O time como sistema: Pitch Control, respiração do bloco, Voronoi e replay 3D.
Depende de módulos já extraídos (diagnostics, persistence) + streamlit/numpy/
pandas/plotly (scipy é importado localmente nas views).
"""
from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import applog as _applog
from diagnostics import _diag_log
from persistence import _carregar_venues
from field import gps_para_campo_coords
from config import _TATICA_PALETA


def _tatica_cor_atleta(i: int) -> str:
    return _TATICA_PALETA[i % len(_TATICA_PALETA)]


def _tatica_iniciais(nome: str) -> str:
    """Iniciais curtas para rotular o marcador do atleta no campo."""
    partes = [p for p in str(nome).strip().split() if p]
    if not partes:
        return '?'
    if len(partes) == 1:
        return partes[0][:3].upper()
    return (partes[0][0] + partes[-1][0]).upper()


def _convex_hull(points):
    """Casco convexo (Andrew's monotone chain), em Python puro — sem scipy.
    points: lista de (x, y). Retorna vértices do hull em sentido anti-horário."""
    pts = sorted(set((round(float(x), 3), round(float(y), 3)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _poly_area(hull):
    """Área de um polígono (fórmula do cadarço / shoelace)."""
    n = len(hull)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _tatica_intervalo(dados_periodo: dict, atletas_sel):
    """Retorna (t0, t1) — janela temporal aproveitável (sobreposição dos atletas,
    com fallback para a união) — para alimentar o seletor de janela na UI."""
    import numpy as _np
    starts, ends = [], []
    for a in atletas_sel:
        d = dados_periodo.get(a, {})
        ts = d.get('ts_pos', [])
        if len(ts) < 5:
            continue
        ts = _np.asarray(ts, dtype=float)
        ts = ts[_np.isfinite(ts)]
        if ts.size < 5:
            continue
        starts.append(float(ts.min()))
        ends.append(float(ts.max()))
    if len(starts) < 2:
        return None
    t0, t1 = max(starts), min(ends)        # sobreposição
    if not (t1 - t0 > 1.0):
        t0, t1 = min(starts), max(ends)    # fallback: união
    if not (t1 - t0 > 1.0):
        return None
    return t0, t1


def _tatica_frames_sincronizados(dados_periodo: dict, atletas_sel, max_frames: int = 160,
                                 t_ini=None, t_fim=None, passo_min: float = 0.5):
    """Constrói frames sincronizados (posição de todos no mesmo instante).

    Para cada atleta com x/y de campo, interpola xs/ys/vel numa grade temporal
    comum. O passo entre frames é adaptativo: ~`passo_min`s (tempo real) numa
    janela curta, crescendo apenas o necessário para não passar de `max_frames`.
    Onde o atleta não tem cobertura, o valor fica NaN.

    Se `t_ini`/`t_fim` forem dados, recorta a animação a essa janela — é isso que
    permite ver o deslocamento contínuo (em vez de saltos de ~26 s no jogo todo).

    Retorna (tempos, nomes, equipes, posicoes, PX, PY, PV) com PX/PY/PV de
    shape [n_frames x n_atletas], ou None se não houver ≥2 atletas alinháveis.
    """
    import numpy as _np
    series = []
    for a in atletas_sel:
        d = dados_periodo.get(a, {})
        xs = d.get('xs', [])
        ys = d.get('ys', [])
        ts = d.get('ts_pos', [])
        vel = d.get('vel', [])
        n = min(len(xs), len(ys), len(ts))
        if n < 5:
            continue
        ts_a = _np.asarray(ts[:n], dtype=float)
        order = _np.argsort(ts_a)
        ts_a = ts_a[order]
        xa = _np.asarray(xs[:n], dtype=float)[order]
        ya = _np.asarray(ys[:n], dtype=float)[order]
        if len(vel) >= n:
            va = _np.asarray(vel[:n], dtype=float)[order]
        else:
            va = _np.zeros(n, dtype=float)
        uniq = _np.concatenate(([True], _np.diff(ts_a) > 0))
        if uniq.sum() < 5:
            continue
        series.append({
            'nome': a, 'equipe': d.get('equipe', ''), 'posicao': d.get('posicao', ''),
            'ts': ts_a[uniq], 'x': xa[uniq], 'y': ya[uniq], 'v': va[uniq],
        })
    if len(series) < 2:
        return None

    t0 = max(s['ts'][0] for s in series)
    t1 = min(s['ts'][-1] for s in series)
    if not (t1 - t0 > 1.0):
        # Sem sobreposição suficiente → usa a união (atletas terão trechos NaN)
        t0 = min(s['ts'][0] for s in series)
        t1 = max(s['ts'][-1] for s in series)
    if not (t1 - t0 > 1.0):
        return None

    # Recorte opcional à janela escolhida na UI.
    if t_ini is not None:
        t0 = max(t0, float(t_ini))
    if t_fim is not None:
        t1 = min(t1, float(t_fim))
    if not (t1 - t0 > 0.5):
        return None

    janela = t1 - t0
    passo = max(passo_min, janela / max_frames)          # ~tempo real em janelas curtas
    nf = int(max(2, min(max_frames, int(round(janela / passo)) + 1)))
    tempos = _np.linspace(t0, t1, nf)
    nomes, equipes, posicoes = [], [], []
    PX, PY, PV = [], [], []
    for s in series:
        xi = _np.interp(tempos, s['ts'], s['x'], left=_np.nan, right=_np.nan)
        yi = _np.interp(tempos, s['ts'], s['y'], left=_np.nan, right=_np.nan)
        vi = _np.interp(tempos, s['ts'], s['v'], left=_np.nan, right=_np.nan)
        fora = (tempos < s['ts'][0]) | (tempos > s['ts'][-1])
        xi[fora] = _np.nan
        yi[fora] = _np.nan
        vi[fora] = _np.nan
        nomes.append(s['nome'])
        equipes.append(s['equipe'])
        posicoes.append(s['posicao'])
        PX.append(xi)
        PY.append(yi)
        PV.append(vi)
    PX = _np.array(PX).T
    PY = _np.array(PY).T
    PV = _np.array(PV).T
    return tempos, nomes, equipes, posicoes, PX, PY, PV


def _tatica_pos_ok(d: dict) -> bool:
    """True se o atleta tem posição utilizável: x/y de campo nativo OU GPS
    (lat/lon + timestamps) para reconstruir as coordenadas."""
    if len(d.get('xs', [])) >= 5 and len(d.get('ys', [])) >= 5:
        return True
    if len(d.get('lats', [])) >= 5 and len(d.get('ts_gps', [])) >= 5:
        return True
    return False


def _tatica_resolver_campo_config(dados_periodo: dict, atletas_sel):
    """Resolve um campo_config (lat/lon/rot/fl/fw) para projetar GPS→campo.

    Prioridade: (1) campo já aplicado na aba Campo & GPS; (2) venue salvo no
    banco compartilhado; (3) AUTO — centro = mediana do GPS de todos os atletas
    e rotação estimada por PCA (eixo principal da nuvem ≈ comprimento do campo).
    Retorna (cfg_dict, fonte_str) ou (None, motivo_str)."""
    import numpy as _np

    # 1) Campo já aplicado pelo usuário na aba Campo & GPS (vale para todos).
    for a in atletas_sel:
        c = st.session_state.get(f"campo_cfg__{a}")
        if c and c.get('lat') and c.get('lon'):
            return dict(c), 'campo aplicado na aba Campo & GPS'

    # 2) Venue salvo no banco compartilhado.
    try:
        vname = st.session_state.get('venue', {}).get('name', '')
        vdb = _carregar_venues()
        if vname and vname in vdb:
            v = vdb[vname]
            if v.get('lat') and v.get('lon'):
                return ({'lat': float(v['lat']), 'lon': float(v['lon']),
                         'rot': float(v.get('rot', 0)), 'fl': float(v.get('fl', 105)),
                         'fw': float(v.get('fw', 68)), 'ig': int(v.get('ig', 1))},
                        f'venue salvo "{vname}"')
    except Exception:
        _applog.log_debug_exc()

    # 3) AUTO: centro = mediana GPS; rotação = eixo principal (PCA).
    lats, lons = [], []
    for a in atletas_sel:
        d = dados_periodo.get(a, {})
        lats += list(d.get('lats', []))
        lons += list(d.get('lons', []))
    if len(lats) < 20:
        return None, 'sem GPS suficiente para reconstruir o campo'
    lat0 = float(_np.median(lats))
    lon0 = float(_np.median(lons))
    la = _np.asarray(lats, dtype=float)
    lo = _np.asarray(lons, dtype=float)
    north = (la - lat0) * 111320.0
    east = (lo - lon0) * 111320.0 * _np.cos(_np.radians(lat0))
    rot = 0.0
    try:
        cov = _np.cov(_np.vstack([east, north]))
        vals, vecs = _np.linalg.eigh(cov)
        pe, pn = vecs[:, int(_np.argmax(vals))]
        rot = float(-_np.degrees(_np.arctan2(pn, pe)))
    except Exception:
        rot = 0.0
    venue = st.session_state.get('venue', {})
    fl = float(venue.get('length') or 105.0)
    fw = float(venue.get('width') or 68.0)
    if fl < fw:
        fl, fw = fw, fl
    return ({'lat': lat0, 'lon': lon0, 'rot': rot, 'fl': fl, 'fw': fw, 'ig': 1},
            'auto (centro = mediana GPS · rotação por PCA)')


def _tatica_preparar_dados(dados_periodo: dict, atletas_sel):
    """Garante coordenadas de campo (xs/ys/ts_pos/vel) para cada atleta. Usa o
    x/y nativo quando existe; senão projeta o GPS (lat/lon → campo) com um
    campo_config resolvido. Retorna (dados_prep, fonte, FL, FW, n_projetados)."""
    nativos = [a for a in atletas_sel
               if len(dados_periodo.get(a, {}).get('xs', [])) >= 5
               and len(dados_periodo.get(a, {}).get('ys', [])) >= 5]
    gps_only = [a for a in atletas_sel
                if a not in nativos
                and len(dados_periodo.get(a, {}).get('lats', [])) >= 5
                and len(dados_periodo.get(a, {}).get('ts_gps', [])) >= 5]

    venue = st.session_state.get('venue', {})
    FL = float(venue.get('length') or 105.0)
    FW = float(venue.get('width') or 68.0)
    dados_prep = {a: dict(dados_periodo.get(a, {})) for a in atletas_sel}
    fonte = 'x/y de campo nativo (API)'
    n_proj = 0

    if gps_only:
        cfg, fnt = _tatica_resolver_campo_config(dados_periodo, atletas_sel)
        if cfg:
            FL = float(cfg.get('fl', FL))
            FW = float(cfg.get('fw', FW))
            for a in gps_only:
                d = dados_periodo.get(a, {})
                lats = d.get('lats', [])
                lons = d.get('lons', [])
                ts = d.get('ts_gps', [])
                vel = d.get('vels_gps', [])
                n = min(len(lats), len(lons), len(ts))
                if n < 5:
                    continue
                try:
                    fx, fy = gps_para_campo_coords(list(lats[:n]), list(lons[:n]), cfg)
                except Exception:
                    _diag_log('Tática', f"{a}: falha ao projetar GPS→campo — "
                                        "atleta ignorado nas visões coletivas")
                    continue
                dd = dict(d)
                dd['xs'] = fx
                dd['ys'] = fy
                dd['ts_pos'] = list(ts[:n])
                dd['vel'] = list(vel[:n]) if len(vel) >= n else [0.0] * n
                dados_prep[a] = dd
                n_proj += 1
            if n_proj:
                fonte = (f'misto: nativo + GPS→campo ({fnt})' if nativos
                         else f'GPS→campo · {fnt}')
    return dados_prep, fonte, FL, FW, n_proj


def _tatica_add_campo_shapes(fig, FL, FW, line_color='rgba(255,255,255,0.85)'):
    """Marcações brancas do campo como shapes (layer='above') — ficam por cima
    de heatmaps (Pitch Control / Voronoi)."""
    cy = FW / 2.0
    L = dict(color=line_color, width=1.6)
    fig.add_shape(type="rect", x0=0, y0=0, x1=FL, y1=FW, line=L, layer='above')
    fig.add_shape(type="line", x0=FL / 2, y0=0, x1=FL / 2, y1=FW, line=L, layer='above')
    fig.add_shape(type="circle", x0=FL / 2 - 9.15, y0=cy - 9.15,
                  x1=FL / 2 + 9.15, y1=cy + 9.15, line=L, layer='above')
    for x0, x1 in [(0, 16.5), (FL - 16.5, FL)]:
        fig.add_shape(type="rect", x0=x0, y0=cy - 20.16, x1=x1, y1=cy + 20.16, line=L, layer='above')
    for x0, x1 in [(0, 5.5), (FL - 5.5, FL)]:
        fig.add_shape(type="rect", x0=x0, y0=cy - 9.16, x1=x1, y1=cy + 9.16, line=L, layer='above')


def _tatica_anim_layout(fig, tempos, height=560, right_margin=80, redraw=True, tween=True):
    """Play/Pause + slider de tempo (mm:ss). A velocidade é controlada pelo
    slider de Velocidade da UI (st.session_state['tatica_vel_mult']): o Play roda
    no tempo real do jogo dividido pelo multiplicador, usando o espaçamento real
    entre frames (assim '1×' = tempo real do mundo em qualquer janela).

    `tween=True` ativa a **interpolação** entre frames (o atleta desliza
    suavemente entre as posições, em vez de teleportar). `redraw=False` é usado
    nas views só-scatter (deslocamento mais fluido); heatmaps precisam de
    `redraw=True`."""
    import numpy as _np
    t0 = tempos[0]
    difs = _np.diff(_np.asarray(tempos, dtype=float))
    dt = float(_np.median(difs)) if difs.size else 0.5      # segundos reais entre frames
    vel = float(st.session_state.get('tatica_vel_mult', 1.0))
    dur = int(max(20, round(dt * 1000.0 / max(0.1, vel))))  # ms por frame na reprodução
    # Tween: interpola o movimento ao longo do tempo do frame (cap 1200 ms p/ não
    # arrastar demais em janelas longas). Sem tween em 3D (não suporta bem).
    tdur = int(min(dur, 1200)) if tween else 0
    labels = [f"{int((t - t0) // 60):02d}:{int((t - t0) % 60):02d}" for t in tempos]
    steps = [dict(method='animate',
                  args=[[f"f{i}"],
                        dict(mode='immediate', frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0))],
                  label=labels[i]) for i in range(len(tempos))]
    fig.update_layout(
        updatemenus=[dict(type='buttons', direction='right', showactive=False,
                          x=0.0, y=0, xanchor='left', yanchor='top',
                          pad=dict(t=0, r=8),
                          bgcolor='#1f2937', font=dict(color='white', size=11),
                          buttons=[
                              dict(label='▶ Play', method='animate',
                                   args=[None, dict(frame=dict(duration=dur, redraw=redraw),
                                                    fromcurrent=True,
                                                    transition=dict(duration=tdur, easing='linear'))]),
                              dict(label='⏸ Pause', method='animate',
                                   args=[[None], dict(frame=dict(duration=0, redraw=True),
                                                      mode='immediate', transition=dict(duration=0))]),
                          ])],
        sliders=[dict(active=0, x=0.15, len=0.83, y=0, xanchor='left', yanchor='top',
                      currentvalue=dict(prefix='⏱️ ', font=dict(color='white')),
                      font=dict(color='#9ca3af', size=9),
                      steps=steps)],
        height=height, paper_bgcolor='#0e1117', plot_bgcolor='#1a3a18',
        margin=dict(l=30, r=right_margin, t=45, b=45),
        font=dict(color='white'), showlegend=False,
    )


def _tatica_view_pitch_control(tempos, nomes, equipes, PX, PY, PV, FL, FW):
    """🎯 Pitch Control (modelo de William Spearman): cada ponto do campo é
    colorido pelo tempo de chegada do jogador mais próximo (posição projetada
    pela velocidade). 1 time → domínio de espaço; 2 times → controle contestado."""
    import numpy as _np
    import plotly.graph_objects as _go
    nf, natl = PX.shape

    # Velocidades (m/s) e direção por diferenças finitas (NaN-safe).
    dt = _np.gradient(tempos)
    _pos_dt = dt[dt > 0]
    dt[dt <= 0] = (_np.median(_pos_dt) if _pos_dt.size else 0.1)
    VX = _np.zeros_like(PX)
    VY = _np.zeros_like(PY)
    VX[1:] = _np.nan_to_num(PX[1:] - PX[:-1])
    VY[1:] = _np.nan_to_num(PY[1:] - PY[:-1])
    VX = VX / dt[:, None]
    VY = VY / dt[:, None]
    sp = _np.hypot(VX, VY)
    cap = 8.0
    scl = _np.where(sp > cap, cap / _np.maximum(sp, 1e-6), 1.0)
    VX *= scl
    VY *= scl

    eq_validas = [e for e in dict.fromkeys(equipes) if e]
    dois_times = len(eq_validas) >= 2
    eq_arr = _np.array(equipes, dtype=object)

    step = 2.5
    gx = _np.arange(step / 2, FL, step)
    gy = _np.arange(step / 2, FW, step)
    GX, GY = _np.meshgrid(gx, gy)
    flatx = GX.ravel()
    flaty = GY.ravel()
    Vmax, tctrl, tau, sigma = 7.0, 0.7, 2.0, 0.6

    def _z_frame(k):
        ex = PX[k] + VX[k] * tctrl
        ey = PY[k] + VY[k] * tctrl
        valid = ~_np.isnan(ex) & ~_np.isnan(ey)
        if valid.sum() == 0:
            return _np.zeros_like(GX)
        evx = ex[valid]
        evy = ey[valid]
        dx = flatx[:, None] - evx[None, :]
        dy = flaty[:, None] - evy[None, :]
        tt = _np.hypot(dx, dy) / Vmax
        if dois_times:
            eqv = eq_arr[valid]
            hm = eqv == eq_validas[0]
            tt_home = _np.min(tt[:, hm], axis=1) if hm.any() else _np.full(tt.shape[0], 99.0)
            tt_away = _np.min(tt[:, ~hm], axis=1) if (~hm).any() else _np.full(tt.shape[0], 99.0)
            z = 1.0 / (1.0 + _np.exp((tt_home - tt_away) / sigma))
        else:
            z = _np.exp(-_np.min(tt, axis=1) / tau)
        return z.reshape(GX.shape)

    def _players(k):
        cols = []
        for i in range(natl):
            if dois_times:
                cols.append('#2196F3' if equipes[i] == eq_validas[0] else '#E53935')
            else:
                cols.append(_tatica_cor_atleta(i))
        return PX[k], PY[k], cols

    txt = [_tatica_iniciais(n) for n in nomes]

    if dois_times:
        cs = [[0.0, 'rgba(229,57,53,0.85)'], [0.5, 'rgba(0,0,0,0.0)'],
              [1.0, 'rgba(33,150,243,0.85)']]
        cbtitle = f"Controle<br>🔵 {eq_validas[0][:10]}"
        op = 0.55
    else:
        cs = [[0.0, 'rgba(0,0,0,0.0)'], [0.35, 'rgba(255,235,59,0.40)'],
              [0.7, 'rgba(255,152,0,0.75)'], [1.0, 'rgba(213,0,0,0.92)']]
        cbtitle = "Domínio<br>de espaço"
        op = 0.6

    z0 = _z_frame(0)
    px0, py0, c0 = _players(0)
    heat = _go.Heatmap(x=gx, y=gy, z=z0, zmin=0.0, zmax=1.0, colorscale=cs,
                       opacity=op, showscale=True, zsmooth='best',
                       colorbar=dict(title=dict(text=cbtitle, font=dict(size=10)),
                                     len=0.55, x=1.0, thickness=12,
                                     tickfont=dict(size=8)),
                       hoverinfo='skip', name='pc')
    players = _go.Scatter(x=px0, y=py0, mode='markers+text', text=txt,
                          textposition='middle center', textfont=dict(color='white', size=8),
                          marker=dict(size=16, color=c0, line=dict(color='white', width=1.5)),
                          hovertext=nomes, hoverinfo='text', name='atletas')
    fig = _go.Figure(data=[heat, players])
    _tatica_add_campo_shapes(fig, FL, FW)
    frames = []
    for k in range(nf):
        pxk, pyk, ck = _players(k)
        frames.append(_go.Frame(name=f"f{k}",
                                data=[_go.Heatmap(z=_z_frame(k)),
                                      _go.Scatter(x=pxk, y=pyk, text=txt,
                                                  marker=dict(size=16, color=ck,
                                                              line=dict(color='white', width=1.5)))],
                                traces=[0, 1]))
    fig.frames = frames
    fig.update_xaxes(range=[-3, FL + 3], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-3, FW + 3], showgrid=False, zeroline=False,
                     scaleanchor='x', scaleratio=1, visible=False)
    _tatica_anim_layout(fig, tempos, right_margin=95, tween=False)  # heatmap: mantém sincronia
    st.plotly_chart(fig, use_container_width=True)


def _tatica_view_respiracao(tempos, nomes, equipes, PX, PY, PV, FL, FW):
    """🫁 Respiração da equipe: centroide + casco convexo animados, e a evolução
    de largura/comprimento/área do bloco ao longo do tempo."""
    import numpy as _np
    import plotly.graph_objects as _go
    nf, natl = PX.shape

    widths, lengths, areas, spreads = [], [], [], []
    cxs, cys, hulls = [], [], []
    for k in range(nf):
        xs = PX[k]
        ys = PY[k]
        m = ~_np.isnan(xs) & ~_np.isnan(ys)
        px = xs[m]
        py = ys[m]
        if len(px) < 3:
            widths.append(_np.nan); lengths.append(_np.nan)
            areas.append(_np.nan); spreads.append(_np.nan)
            cxs.append(_np.nan); cys.append(_np.nan)
            hulls.append(([], []))
            continue
        cx = float(px.mean()); cy = float(py.mean())
        pts = list(zip(px.tolist(), py.tolist()))
        h = _convex_hull(pts)
        hx = [p[0] for p in h] + ([h[0][0]] if h else [])
        hy = [p[1] for p in h] + ([h[0][1]] if h else [])
        widths.append(float(py.max() - py.min()))
        lengths.append(float(px.max() - px.min()))
        areas.append(_poly_area(h))
        spreads.append(float(_np.mean(_np.hypot(px - cx, py - cy))))
        cxs.append(cx); cys.append(cy)
        hulls.append((hx, hy))

    def _hud(k):
        w = widths[k]; l = lengths[k]; a = areas[k]
        if _np.isnan(w):
            return "🫁 Respiração da equipe"
        return (f"🫁  Largura {w:.0f} m   ·   Comprimento {l:.0f} m   ·   "
                f"Área {a:.0f} m²   ·   Dispersão {spreads[k]:.0f} m")

    hx0, hy0 = hulls[0]
    hull_tr = _go.Scatter(x=hx0, y=hy0, mode='lines', fill='toself',
                          fillcolor='rgba(68,138,255,0.18)',
                          line=dict(color='#448AFF', width=2),
                          hoverinfo='skip', name='bloco')
    cols = [_tatica_cor_atleta(i) for i in range(natl)]
    txt = [_tatica_iniciais(n) for n in nomes]
    players = _go.Scatter(x=PX[0], y=PY[0], mode='markers+text', text=txt,
                          textposition='middle center', textfont=dict(color='white', size=8),
                          marker=dict(size=15, color=cols, line=dict(color='white', width=1.2)),
                          hovertext=nomes, hoverinfo='text', name='atletas')
    centro = _go.Scatter(x=[cxs[0]], y=[cys[0]], mode='markers',
                         marker=dict(size=16, color='#FFD740', symbol='x',
                                     line=dict(color='black', width=1)),
                         hoverinfo='skip', name='centroide')
    fig = _go.Figure(data=[hull_tr, players, centro])
    _tatica_add_campo_shapes(fig, FL, FW)
    frames = []
    for k in range(nf):
        hxk, hyk = hulls[k]
        frames.append(_go.Frame(name=f"f{k}",
                                data=[_go.Scatter(x=hxk, y=hyk),
                                      _go.Scatter(x=PX[k], y=PY[k], text=txt),
                                      _go.Scatter(x=[cxs[k]], y=[cys[k]])],
                                traces=[0, 1, 2],
                                layout=dict(title=dict(text=_hud(k),
                                                       font=dict(color='white', size=12)))))
    fig.frames = frames
    fig.update_xaxes(range=[-3, FL + 3], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-3, FW + 3], showgrid=False, zeroline=False,
                     scaleanchor='x', scaleratio=1, visible=False)
    _tatica_anim_layout(fig, tempos, redraw=False)  # só-scatter: jogadores deslizam
    fig.update_layout(title=dict(text=_hud(0), font=dict(color='white', size=12)))
    st.plotly_chart(fig, use_container_width=True)

    # ── Evolução temporal (largura/comprimento + área) ──────────────────
    tmin = [(t - tempos[0]) / 60.0 for t in tempos]
    ev = _go.Figure()
    ev.add_trace(_go.Scatter(x=tmin, y=widths, mode='lines', name='Largura (m)',
                             line=dict(color='#40C4FF', width=2)))
    ev.add_trace(_go.Scatter(x=tmin, y=lengths, mode='lines', name='Comprimento (m)',
                             line=dict(color='#69F0AE', width=2)))
    ev.add_trace(_go.Scatter(x=tmin, y=areas, mode='lines', name='Área (m²)',
                             line=dict(color='#FFAB40', width=2, dash='dot'), yaxis='y2'))
    ev.update_layout(
        height=240, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        margin=dict(l=40, r=50, t=30, b=35), font=dict(color='white', size=10),
        title=dict(text='📈 Evolução do bloco (compactação × expansão)',
                   font=dict(color='white', size=12)),
        xaxis=dict(title='minutos', gridcolor='#1f2937'),
        yaxis=dict(title='metros', gridcolor='#1f2937'),
        yaxis2=dict(title='m²', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation='h', y=1.18, font=dict(size=9)),
    )
    st.plotly_chart(ev, use_container_width=True)


def _tatica_view_voronoi(tempos, nomes, equipes, PX, PY, PV, FL, FW):
    """🔷 Voronoi: cada ponto do campo pertence ao jogador mais próximo —
    o 'vitral' tático que mostra cobertura de espaço e buracos."""
    import numpy as _np
    import plotly.graph_objects as _go
    nf, natl = PX.shape

    step = 2.0
    gx = _np.arange(step / 2, FL, step)
    gy = _np.arange(step / 2, FW, step)
    GX, GY = _np.meshgrid(gx, gy)
    flatx = GX.ravel()
    flaty = GY.ravel()

    cores = [_tatica_cor_atleta(i) for i in range(natl)]
    # Colorscale discreta: faixa i → cor do atleta i (z = idx + 0.5).
    cs = []
    for i, c in enumerate(cores):
        cs.append([i / natl, c])
        cs.append([(i + 1) / natl, c])

    def _z_frame(k):
        ex = PX[k]
        ey = PY[k]
        valid = ~_np.isnan(ex) & ~_np.isnan(ey)
        idxs = _np.where(valid)[0]
        if idxs.size == 0:
            return _np.full(GX.shape, _np.nan)
        dx = flatx[:, None] - ex[idxs][None, :]
        dy = flaty[:, None] - ey[idxs][None, :]
        d2 = dx * dx + dy * dy
        nearest = idxs[_np.argmin(d2, axis=1)]
        return (nearest + 0.5).reshape(GX.shape)

    txt = [_tatica_iniciais(n) for n in nomes]
    z0 = _z_frame(0)
    heat = _go.Heatmap(x=gx, y=gy, z=z0, zmin=0.0, zmax=float(natl),
                       colorscale=cs, opacity=0.5, showscale=False,
                       hoverinfo='skip', name='voronoi')
    players = _go.Scatter(x=PX[0], y=PY[0], mode='markers+text', text=txt,
                          textposition='middle center', textfont=dict(color='black', size=8),
                          marker=dict(size=15, color=cores, line=dict(color='white', width=2)),
                          hovertext=nomes, hoverinfo='text', name='atletas')
    fig = _go.Figure(data=[heat, players])
    _tatica_add_campo_shapes(fig, FL, FW, line_color='rgba(255,255,255,0.95)')
    frames = []
    for k in range(nf):
        frames.append(_go.Frame(name=f"f{k}",
                                data=[_go.Heatmap(z=_z_frame(k)),
                                      _go.Scatter(x=PX[k], y=PY[k], text=txt)],
                                traces=[0, 1]))
    fig.frames = frames
    fig.update_xaxes(range=[-3, FL + 3], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-3, FW + 3], showgrid=False, zeroline=False,
                     scaleanchor='x', scaleratio=1, visible=False)
    _tatica_anim_layout(fig, tempos, tween=False)  # heatmap: mantém sincronia
    st.plotly_chart(fig, use_container_width=True)


def _tatica_view_replay3d(tempos, nomes, equipes, PX, PY, PV, FL, FW):
    """🎥 Replay 3D: 'broadcast sintético' — gramado com faixas de corte,
    marcações oficiais, traves 3D com rede e jogadores com sombra/haste,
    tudo navegável (câmera livre) a partir só das coordenadas."""
    import numpy as _np
    import plotly.graph_objects as _go
    nf, natl = PX.shape
    cx, cy = FL / 2.0, FW / 2.0
    LZ = 0.06  # altura das linhas, rente ao gramado

    data = []

    # --- Entorno (fora das quatro linhas), bem escuro p/ destacar o campo ---
    mX, mY = _np.meshgrid(_np.linspace(-10, FL + 10, 2), _np.linspace(-8, FW + 8, 2))
    data.append(_go.Surface(x=mX, y=mY, z=_np.full_like(mX, -0.12), showscale=False,
                            colorscale=[[0, '#10301a'], [1, '#10301a']],
                            surfacecolor=_np.zeros_like(mX), hoverinfo='skip',
                            lighting=dict(ambient=0.9, diffuse=0.1)))

    # --- Gramado com faixas de corte (mowing stripes) ---
    n_stripes = 14
    sw = FL / n_stripes
    greens = ['#2f7d28', '#2a7022']
    for s in range(n_stripes):
        x0, x1 = s * sw, (s + 1) * sw
        Xf, Yf = _np.meshgrid(_np.linspace(x0, x1, 2), _np.linspace(0, FW, 2))
        Zf = _np.zeros_like(Xf)
        c = greens[s % 2]
        data.append(_go.Surface(x=Xf, y=Yf, z=Zf, showscale=False,
                                colorscale=[[0, c], [1, c]], surfacecolor=Zf,
                                lighting=dict(ambient=0.88, diffuse=0.32, specular=0.04),
                                hoverinfo='skip', name='grama'))

    # --- Linhas oficiais do campo ---
    LCOL = 'rgba(255,255,255,0.92)'

    def _line(xs, ys, w=4, color=LCOL):
        return _go.Scatter3d(x=list(xs), y=list(ys), z=[LZ] * len(xs), mode='lines',
                             line=dict(color=color, width=w), hoverinfo='skip',
                             showlegend=False)

    def _dot(x, y, sz=3):
        return _go.Scatter3d(x=[x], y=[y], z=[LZ], mode='markers',
                             marker=dict(size=sz, color='white'),
                             hoverinfo='skip', showlegend=False)

    th = _np.linspace(0, 2 * _np.pi, 64)
    data.append(_line([0, FL, FL, 0, 0], [0, 0, FW, FW, 0]))           # perímetro
    data.append(_line([cx, cx], [0, FW]))                              # meio-campo
    data.append(_line(cx + 9.15 * _np.cos(th), cy + 9.15 * _np.sin(th), w=3))  # círculo central
    data.append(_dot(cx, cy))                                          # marca central

    pa_d, pa_h = 16.5, 20.16   # área de penálti (profundidade, meia-largura)
    ga_d, ga_h = 5.5, 9.16     # pequena área
    pspot = 11.0
    for side in (0, 1):
        sgn = 1 if side == 0 else -1
        xg = 0 if side == 0 else FL
        data.append(_line([xg, xg + sgn * pa_d, xg + sgn * pa_d, xg],
                          [cy - pa_h, cy - pa_h, cy + pa_h, cy + pa_h]))   # grande área
        data.append(_line([xg, xg + sgn * ga_d, xg + sgn * ga_d, xg],
                          [cy - ga_h, cy - ga_h, cy + ga_h, cy + ga_h]))   # pequena área
        xp = xg + sgn * pspot
        data.append(_dot(xp, cy))                                          # marca do penálti
        a = _np.linspace(0, 2 * _np.pi, 90)
        ax = xp + sgn * 9.15 * _np.cos(a)
        ay = cy + 9.15 * _np.sin(a)
        mask = (ax - xg) * sgn > pa_d                                      # só o arco fora da área
        axm = _np.where(mask, ax, _np.nan)
        aym = _np.where(mask, ay, _np.nan)
        data.append(_line(axm, aym, w=3))                                  # arco do penálti

    for (xc, yc, a0) in [(0, 0, 0), (FL, 0, 90), (FL, FW, 180), (0, FW, 270)]:
        a = _np.linspace(_np.radians(a0), _np.radians(a0 + 90), 14)
        data.append(_line(xc + 1.0 * _np.cos(a), yc + 1.0 * _np.sin(a), w=2))  # arcos de escanteio

    # --- Traves 3D (postes + travessão + rede) ---
    gw, gh, depth = 7.32, 2.44, 1.9

    def _goal(xg, sgn):
        yl, yr = cy - gw / 2, cy + gw / 2
        xb = xg + sgn * depth
        frame = [((xg, yl, 0), (xg, yl, gh)), ((xg, yr, 0), (xg, yr, gh)),
                 ((xg, yl, gh), (xg, yr, gh)),                       # travessão
                 ((xg, yl, gh), (xb, yl, 0)), ((xg, yr, gh), (xb, yr, 0)),
                 ((xb, yl, 0), (xb, yr, 0))]
        xs, ys, zs = [], [], []
        for p0, p1 in frame:
            xs += [p0[0], p1[0], None]; ys += [p0[1], p1[1], None]; zs += [p0[2], p1[2], None]
        estrut = _go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                               line=dict(color='white', width=6), hoverinfo='skip', showlegend=False)
        # rede: malha suave entre travessão e fundo
        nx, ny = [], []
        nz = []
        for t in _np.linspace(0, 1, 5):       # verticais
            yy = yl + (yr - yl) * t
            nx += [xg, xb, None]; ny += [yy, yy, None]; nz += [gh, 0, None]
        for t in _np.linspace(0, 1, 3):       # horizontais
            xx = xg + (xb - xg) * t
            zz = gh * (1 - t)
            nx += [xx, xx, None]; ny += [yl, yr, None]; nz += [zz, zz, None]
        rede = _go.Scatter3d(x=nx, y=ny, z=nz, mode='lines',
                             line=dict(color='rgba(255,255,255,0.30)', width=1),
                             hoverinfo='skip', showlegend=False)
        return [estrut, rede]

    data += _goal(0, -1)
    data += _goal(FL, 1)

    # --- Jogadores (sombra no chão + haste + marcador) ---
    base = len(data)
    eq_validas = [e for e in dict.fromkeys(equipes) if e]
    dois_times = len(eq_validas) >= 2
    if dois_times:
        cols = ['#2196F3' if equipes[i] == eq_validas[0] else '#E53935' for i in range(natl)]
    else:
        cols = [_tatica_cor_atleta(i) for i in range(natl)]
    txt = [_tatica_iniciais(n) for n in nomes]
    z_dot = 2.2

    def _shadow(k):
        return _go.Scatter3d(x=PX[k], y=PY[k], z=[0.08] * natl, mode='markers',
                             marker=dict(size=9, color='rgba(0,0,0,0.28)'),
                             hoverinfo='skip', showlegend=False)

    def _stems(k):
        xs, ys, zs = [], [], []
        for i in range(natl):
            xs += [PX[k][i], PX[k][i], None]
            ys += [PY[k][i], PY[k][i], None]
            zs += [0.08, z_dot, None]
        return _go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                             line=dict(color='rgba(255,255,255,0.35)', width=2),
                             hoverinfo='skip', showlegend=False)

    def _players(k):
        return _go.Scatter3d(x=PX[k], y=PY[k], z=[z_dot] * natl, mode='markers+text',
                             text=txt, textposition='top center',
                             textfont=dict(color='white', size=9),
                             marker=dict(size=7, color=cols, line=dict(color='white', width=1)),
                             hovertext=nomes, hoverinfo='text', name='atletas')

    i_sh, i_st, i_pl = base, base + 1, base + 2
    data += [_shadow(0), _stems(0), _players(0)]

    fig = _go.Figure(data=data)
    fig.frames = [_go.Frame(name=f"f{k}",
                            data=[_shadow(k), _stems(k), _players(k)],
                            traces=[i_sh, i_st, i_pl]) for k in range(nf)]
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-8, FL + 8], visible=False),
            yaxis=dict(range=[-6, FW + 6], visible=False),
            zaxis=dict(range=[0, 12], visible=False),
            aspectmode='manual', aspectratio=dict(x=2.0, y=1.3, z=0.42),
            camera=dict(eye=dict(x=0.2, y=-1.6, z=0.85)),
            bgcolor='#0e1117',
        ),
    )
    _tatica_anim_layout(fig, tempos, height=620, tween=False)
    st.plotly_chart(fig, use_container_width=True)


def _tatica_view_distancias(tempos, nomes, equipes, PX, PY, PV, FL, FW):
    """📏 Distância entre atletas: matriz de distância média, evolução temporal
    da distância média entre pares e navegação por instante (maior/menor
    distância) — tudo derivado das posições sincronizadas no campo."""
    import numpy as _np
    import plotly.graph_objects as _go
    nf, natl = PX.shape
    if natl < 2:
        st.info("Selecione pelo menos 2 atletas para analisar distâncias.")
        return

    ini = [_tatica_iniciais(n) for n in nomes]
    # rótulos únicos p/ eixos (evita iniciais repetidas se houver)
    _seen = {}
    rot = []
    for s in ini:
        if s in _seen:
            _seen[s] += 1
            rot.append(f"{s}{_seen[s]}")
        else:
            _seen[s] = 1
            rot.append(s)

    # ── Distâncias por frame: D[k,i,j] = dist no campo (m), NaN se faltar ──
    dx = PX[:, :, None] - PX[:, None, :]
    dy = PY[:, :, None] - PY[:, None, :]
    D = _np.sqrt(dx * dx + dy * dy)                 # [nf, natl, natl]

    iu = _np.triu_indices(natl, k=1)
    pares_serie = D[:, iu[0], iu[1]]                # [nf, npares]
    with _np.errstate(all='ignore'):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            mean_dist = _np.nanmean(pares_serie, axis=1)        # média entre pares por frame
            M_janela = _np.nanmean(D, axis=0)                   # matriz média na janela

    if _np.all(_np.isnan(mean_dist)):
        st.info("Sem pares de atletas com cobertura simultânea nesta janela.")
        return

    t0 = tempos[0]
    tmin = (_np.asarray(tempos, dtype=float) - t0) / 60.0       # minutos relativos

    def _mmss(seg):
        seg = int(round(seg))
        return f"{seg // 60:02d}:{seg % 60:02d}"

    k_max = int(_np.nanargmax(mean_dist))
    k_min = int(_np.nanargmin(mean_dist))

    # ── Campo animado com linhas de distância em tempo real ─────────────────
    st.markdown("##### 🎥 Campo com distâncias em tempo real")
    eq_validas = [e for e in dict.fromkeys(equipes) if e]
    dois_times = len(eq_validas) >= 2
    if dois_times:
        cores_pl = ['#2196F3' if equipes[i] == eq_validas[0] else '#E53935' for i in range(natl)]
    else:
        cores_pl = [_tatica_cor_atleta(i) for i in range(natl)]

    cmo1, cmo2 = st.columns([1.3, 1])
    with cmo1:
        modo_conn = st.radio("Conexões a desenhar",
                             ["A partir de um atleta", "Vizinho mais próximo", "Todos os pares"],
                             horizontal=True, key="tatica_dist_conn")
    ref_idx = 0
    if modo_conn == "A partir de um atleta":
        with cmo2:
            ref_nome = st.selectbox("Atleta de referência", nomes, key="tatica_dist_ref")
            ref_idx = nomes.index(ref_nome)

    def _pares_frame(k):
        xs = PX[k]; ys = PY[k]
        val = ~_np.isnan(xs) & ~_np.isnan(ys)
        pares = []
        if modo_conn == "Todos os pares":
            for i in range(natl):
                for j in range(i + 1, natl):
                    if val[i] and val[j]:
                        pares.append((i, j))
        elif modo_conn == "Vizinho mais próximo":
            s = set()
            for i in range(natl):
                if not val[i]:
                    continue
                best, bd = -1, 1e18
                for j in range(natl):
                    if j == i or not val[j]:
                        continue
                    dd = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2
                    if dd < bd:
                        bd, best = dd, j
                if best >= 0:
                    s.add(tuple(sorted((i, best))))
            pares = list(s)
        else:
            if val[ref_idx]:
                for j in range(natl):
                    if j != ref_idx and val[j]:
                        pares.append((ref_idx, j))
        return pares

    _rotular = (modo_conn != "Todos os pares")

    def _line_data(k):
        lx, ly, tx, ty, tt = [], [], [], [], []
        for (i, j) in _pares_frame(k):
            lx += [PX[k][i], PX[k][j], None]
            ly += [PY[k][i], PY[k][j], None]
            if _rotular:
                tx.append((PX[k][i] + PX[k][j]) / 2.0)
                ty.append((PY[k][i] + PY[k][j]) / 2.0)
                tt.append(f"{D[k, i, j]:.0f}")
        return lx, ly, tx, ty, tt

    lx0, ly0, tx0, ty0, tt0 = _line_data(0)
    line_tr = _go.Scatter(x=lx0, y=ly0, mode='lines',
                          line=dict(color='rgba(255,255,255,0.5)', width=1.4),
                          hoverinfo='skip', name='dist')
    lab_tr = _go.Scatter(x=tx0, y=ty0, mode='text', text=tt0,
                         textfont=dict(color='#FFD740', size=10), hoverinfo='skip', name='m')
    pl_tr = _go.Scatter(x=PX[0], y=PY[0], mode='markers+text', text=ini,
                        textposition='middle center', textfont=dict(color='white', size=8),
                        marker=dict(size=15, color=cores_pl, line=dict(color='white', width=1.2)),
                        hovertext=nomes, hoverinfo='text', name='atletas')
    figc = _go.Figure(data=[line_tr, lab_tr, pl_tr])
    _tatica_add_campo_shapes(figc, FL, FW)
    figc.frames = []
    _frs = []
    for k in range(nf):
        lx, ly, tx, ty, tt = _line_data(k)
        _frs.append(_go.Frame(name=f"f{k}",
                              data=[_go.Scatter(x=lx, y=ly),
                                    _go.Scatter(x=tx, y=ty, text=tt),
                                    _go.Scatter(x=PX[k], y=PY[k], text=ini)],
                              traces=[0, 1, 2]))
    figc.frames = _frs
    figc.update_xaxes(range=[-3, FL + 3], showgrid=False, zeroline=False, visible=False)
    figc.update_yaxes(range=[-3, FW + 3], showgrid=False, zeroline=False,
                      scaleanchor='x', scaleratio=1, visible=False)
    _tatica_anim_layout(figc, tempos, height=520, redraw=False)  # só-scatter: jogadores deslizam
    st.plotly_chart(figc, use_container_width=True)
    st.caption("As linhas conectam os atletas e os números mostram a distância (m) **a cada "
               "instante**. Use ▶ Play (e o slider de velocidade acima) para ver em tempo real.")

    # ── Estado do instante selecionado ──────────────────────────────────────
    if ('tatica_dist_k' not in st.session_state
            or not isinstance(st.session_state.get('tatica_dist_k'), int)
            or st.session_state['tatica_dist_k'] >= nf):
        st.session_state['tatica_dist_k'] = k_max

    # ── Cartões: momentos de maior e menor distância ────────────────────────
    cM, cm = st.columns(2)
    with cM:
        st.metric("⤢ Maior distância média (equipe mais aberta)",
                  f"{mean_dist[k_max]:.1f} m", f"aos {_mmss(tempos[k_max]-t0)}")
        if st.button("Ver este instante ⤢", key="btn_dist_max", use_container_width=True):
            st.session_state['tatica_dist_k'] = k_max
            st.rerun()
    with cm:
        st.metric("⤡ Menor distância média (equipe mais compacta)",
                  f"{mean_dist[k_min]:.1f} m", f"aos {_mmss(tempos[k_min]-t0)}", delta_color="inverse")
        if st.button("Ver este instante ⤡", key="btn_dist_min", use_container_width=True):
            st.session_state['tatica_dist_k'] = k_min
            st.rerun()

    # ── Slider de linha do tempo (instante) ─────────────────────────────────
    if nf > 1:
        k_sel = st.slider("Instante na linha do tempo", 0, nf - 1,
                          key="tatica_dist_k",
                          format="frame %d",
                          help="Arraste para navegar; ou use os botões acima para pular "
                               "aos momentos de maior/menor distância.")
    else:
        k_sel = 0
    st.caption(f"⏱️ Instante selecionado: **{_mmss(tempos[k_sel]-t0)}** "
               f"· distância média neste instante: **{mean_dist[k_sel]:.1f} m**")

    # ── Evolução temporal da distância média entre pares ────────────────────
    ev = _go.Figure()
    ev.add_trace(_go.Scatter(x=tmin, y=mean_dist, mode='lines',
                             line=dict(color='#42A5F5', width=2),
                             name='Dist. média', hovertemplate='%{x:.1f} min — %{y:.1f} m<extra></extra>'))
    ev.add_trace(_go.Scatter(x=[tmin[k_max]], y=[mean_dist[k_max]], mode='markers+text',
                             marker=dict(color='#EF5350', size=11, symbol='triangle-up'),
                             text=['máx'], textposition='top center',
                             textfont=dict(color='#EF5350', size=10), hoverinfo='skip', showlegend=False))
    ev.add_trace(_go.Scatter(x=[tmin[k_min]], y=[mean_dist[k_min]], mode='markers+text',
                             marker=dict(color='#66BB6A', size=11, symbol='triangle-down'),
                             text=['mín'], textposition='bottom center',
                             textfont=dict(color='#66BB6A', size=10), hoverinfo='skip', showlegend=False))
    ev.add_vline(x=tmin[k_sel], line_dash='dot', line_color='#FFD740', line_width=2)
    ev.update_layout(
        height=260, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        margin=dict(l=45, r=20, t=34, b=35), font=dict(color='white', size=10),
        title=dict(text='📈 Distância média entre atletas ao longo do tempo',
                   font=dict(color='white', size=12)),
        xaxis=dict(title='minutos', gridcolor='#1f2937'),
        yaxis=dict(title='metros', gridcolor='#1f2937'),
        showlegend=False,
    )
    st.plotly_chart(ev, use_container_width=True)

    # ── Matriz de distância (média da janela OU instante) ───────────────────
    cmsel1, cmsel2 = st.columns([1, 1])
    with cmsel1:
        modo_mat = st.radio("Matriz de distância",
                            ["Média da janela", "Instante selecionado"],
                            horizontal=True, key="tatica_dist_modo")
    if modo_mat == "Instante selecionado":
        M = D[k_sel].copy()
        sub = f"instante {_mmss(tempos[k_sel]-t0)}"
    else:
        M = M_janela.copy()
        sub = f"média de {_mmss(dur := tempos[-1]-tempos[0])}"
    _np.fill_diagonal(M, _np.nan)

    txt = [[("" if _np.isnan(M[i, j]) else f"{M[i, j]:.0f}") for j in range(natl)]
           for i in range(natl)]
    # rótulos = nomes dos atletas (garante unicidade p/ não fundir células)
    _vis = {}
    eixo = []
    for n in nomes:
        if n in _vis:
            _vis[n] += 1
            eixo.append(f"{n} ({_vis[n]})")
        else:
            _vis[n] = 1
            eixo.append(n)
    _maxlen = max((len(n) for n in eixo), default=8)
    heat = _go.Figure(data=_go.Heatmap(
        z=M, x=eixo, y=eixo, text=txt, texttemplate="%{text}",
        textfont=dict(size=9, color='white'),
        colorscale='YlOrRd_r', reversescale=False,
        colorbar=dict(title=dict(text='m', font=dict(color='white')),
                      tickfont=dict(color='white'), thickness=12),
        hovertemplate='%{y} ↔ %{x}: %{z:.1f} m<extra></extra>'))
    heat.update_layout(
        height=max(360, 30 * natl + 130), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        margin=dict(l=10, r=20, t=40, b=10), font=dict(color='white', size=10),
        title=dict(text=f'🔲 Matriz de distância média entre atletas ({sub})',
                   font=dict(color='white', size=12)),
        xaxis=dict(side='top', tickangle=-40, tickfont=dict(size=9), automargin=True),
        yaxis=dict(autorange='reversed', tickfont=dict(size=9), automargin=True),
    )
    st.plotly_chart(heat, use_container_width=True)

    # ── Tabelas: pares + por atleta ─────────────────────────────────────────
    pares = [(rot[i], rot[j], nomes[i], nomes[j], M_janela[i, j])
             for i in range(natl) for j in range(i + 1, natl)
             if not _np.isnan(M_janela[i, j])]
    pares.sort(key=lambda p: p[4])

    cP1, cP2 = st.columns(2)
    with cP1:
        st.markdown("##### 🤝 Pares mais próximos (média)")
        _df_perto = pd.DataFrame(
            [{'Atleta A': p[2], 'Atleta B': p[3], 'Dist. média (m)': round(p[4], 1)}
             for p in pares[:6]])
        st.dataframe(_df_perto, use_container_width=True, hide_index=True)
    with cP2:
        st.markdown("##### ↔️ Pares mais distantes (média)")
        _df_longe = pd.DataFrame(
            [{'Atleta A': p[2], 'Atleta B': p[3], 'Dist. média (m)': round(p[4], 1)}
             for p in pares[::-1][:6]])
        st.dataframe(_df_longe, use_container_width=True, hide_index=True)

    with _np.errstate(all='ignore'):
        import warnings as _w2
        with _w2.catch_warnings():
            _w2.simplefilter('ignore')
            _Mna = M_janela.copy()
            _np.fill_diagonal(_Mna, _np.nan)
            media_por_atl = _np.nanmean(_Mna, axis=1)
    _df_atl = pd.DataFrame({
        'Atleta': nomes,
        'Equipe': [e or '—' for e in equipes],
        'Dist. média aos demais (m)': [round(float(v), 1) if not _np.isnan(v) else None
                                       for v in media_por_atl],
    }).sort_values('Dist. média aos demais (m)', na_position='last')
    st.markdown("##### 🧍 Distância média de cada atleta aos demais")
    st.dataframe(_df_atl, use_container_width=True, hide_index=True)

    # ── Export da matriz (média da janela) ──────────────────────────────────
    _df_mat = pd.DataFrame(M_janela, index=nomes, columns=nomes).round(1)
    st.download_button(
        "📥 Exportar matriz de distância média (CSV)",
        _df_mat.to_csv().encode('utf-8'),
        "matriz_distancia_atletas.csv", mime='text/csv')


@st.cache_data(show_spinner=False, max_entries=24)
def _tatica_frames_cached(cache_key, _dados_prep, atletas_sel, t_ini, t_fim,
                          max_frames):
    """(P6) Cache dos frames sincronizados da Tática Coletiva.

    Os mesmos frames alimentam as 5 visões — sem cache, cada interação (troca
    de visão, velocidade, slider) reinterpolava todos os atletas. Os dados
    grandes (_dados_prep) ficam fora da chave; a chave pequena identifica
    token/atividade/período/atletas/campo/janela."""
    return _tatica_frames_sincronizados(_dados_prep, list(atletas_sel),
                                        t_ini=t_ini, t_fim=t_fim,
                                        max_frames=max_frames)


def render_tatica_coletiva(dados_posicao_por_periodo, periodos_selecionados, atletas_sel):
    """Aba 🧠 Tática Coletiva — orquestra as 5 visões coletivas."""
    import numpy as _np

    st.markdown("### 🧠 Tática Coletiva")
    st.caption("O time como **sistema**: visões que cruzam a posição de todos os "
               "atletas no mesmo instante. Pitch Control, respiração do bloco, "
               "domínio de espaço (Voronoi) e replay 3D navegável.")

    if not dados_posicao_por_periodo:
        st.info("Carregue dados de posição (x/y de campo) para usar a Tática Coletiva.")
        return

    pers_validos = []
    for p, dd in dados_posicao_por_periodo.items():
        n_ok = sum(1 for a in atletas_sel if _tatica_pos_ok(dd.get(a, {})))
        if n_ok >= 2:
            pers_validos.append((p, n_ok))
    if not pers_validos:
        st.warning("Esta aba precisa de **pelo menos 2 atletas com posição no mesmo período** — "
                   "x/y de campo nativo **ou** trajetória GPS (lat/lon) para reconstruir as "
                   "coordenadas. Confirme se os atletas têm GPS carregado nesta atividade.")
        return

    n_ok_map = dict(pers_validos)
    c1, c2 = st.columns([1, 2])
    with c1:
        per_sel = st.selectbox("Período", [p for p, _ in pers_validos],
                               format_func=lambda p: f"{p} ({n_ok_map[p]} atletas)",
                               key="tatica_periodo")
    with c2:
        vis = st.radio("Visualização",
                       ["🎯 Pitch Control", "🫁 Respiração da equipe", "🔷 Voronoi",
                        "🎥 Replay 3D", "📏 Distância entre atletas"],
                       horizontal=True, key="tatica_vis")

    dados_periodo = dados_posicao_por_periodo.get(per_sel, {})
    dados_prep, _fonte_pos, FL, FW, _n_proj = _tatica_preparar_dados(dados_periodo, atletas_sel)

    _intervalo = _tatica_intervalo(dados_prep, atletas_sel)
    if _intervalo is None:
        st.warning("Não foi possível sincronizar os atletas neste período "
                   "(sem sobreposição temporal suficiente).")
        return
    _t0_abs, _t1_abs = _intervalo
    _total_s = _t1_abs - _t0_abs

    def _mmss(s):
        s = int(round(s))
        return f"{s // 60:02d}:{s % 60:02d}"

    _jan_map = {"30 s": 30.0, "1 min": 60.0, "2 min": 120.0, "5 min": 300.0, "10 min": 600.0}
    _opcoes = ["Período inteiro"] + [k for k, v in _jan_map.items() if v < _total_s]
    _opcoes += ["Personalizada…"]
    _idx_pad = 0  # padrão: período inteiro → o slider do gráfico cobre a partida toda

    cj1, cj2 = st.columns([1, 2])
    with cj1:
        jan_sel = st.selectbox(
            "Janela de análise", _opcoes, index=_idx_pad, key="tatica_janela",
            help="**Período inteiro** (padrão): o slider abaixo do gráfico percorre a "
                 "**partida toda** — arraste para qualquer momento. Para ver o "
                 "**deslocamento contínuo em tempo real**, escolha uma janela menor "
                 "(1–2 min) e use 'Início da janela' para posicioná-la no trecho desejado.")

    if jan_sel == "Período inteiro":
        _win = None
        _t_ini, _t_fim = _t0_abs, _t1_abs
    elif jan_sel == "Personalizada…":
        with cj2:
            _win_min = st.number_input(
                "Duração da janela (min)", min_value=0.25,
                max_value=round(_total_s / 60.0, 2),
                value=float(min(2.0, round(_total_s / 60.0, 2))), step=0.5,
                key="tatica_win_custom",
                help="Defina qualquer duração — de 15 s ao período inteiro.")
        _win = float(_win_min) * 60.0
    else:
        _win = _jan_map[jan_sel]

    if _win is not None:
        _win = float(min(_win, _total_s))
        _ini_max = max(0.0, _total_s - _win)
        if _ini_max > 0.5:
            _ini_max_min = round(_ini_max / 60.0, 2)
            _prev_min = min(float(st.session_state.get("tatica_inicio_min", 0.0)), _ini_max_min)
            _ini_min = st.slider(
                "Início da janela (min) — arraste para escolher o trecho do jogo",
                0.0, _ini_max_min, _prev_min, step=0.25,
                key="tatica_inicio_min", format="%.2f min",
                help="Posiciona a janela em qualquer ponto da partida. "
                     "Ex.: leve até o fim para analisar os minutos finais.")
            _ini_rel = _ini_min * 60.0
        else:
            _ini_rel = 0.0
        _t_ini = _t0_abs + _ini_rel
        _t_fim = _t_ini + _win
        st.caption(f"🎬 Janela: **{_mmss(_ini_rel)} → {_mmss(_ini_rel + _win)}** "
                   f"(de {_mmss(_total_s)} totais) · o slider abaixo do gráfico percorre "
                   f"os frames **dentro** desta janela.")
    else:
        st.caption(f"🎬 Janela: **período inteiro** ({_mmss(_total_s)}) · o slider abaixo do "
                   f"gráfico percorre a **partida toda** — arraste-o para qualquer momento.")

    _vel_opts = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    st.select_slider(
        "Velocidade da animação", options=_vel_opts, value=1.0, key="tatica_vel_mult",
        format_func=lambda v: ("1× (tempo real)" if v == 1.0 else f"{v:g}×"),
        help="1× reproduz no tempo real do jogo. Abaixo de 1× = câmera lenta; "
             "acima = acelerado. Aplica-se ao botão ▶ Play.")

    # (P6) frames em cache — chave pequena; dados grandes fora da chave
    _fr_key = (str(st.session_state.get('_token_marker', '')),
               str(st.session_state.get('activity_id', '')),
               str(per_sel), tuple(atletas_sel or []), str(_fonte_pos),
               round(float(FL), 1), round(float(FW), 1))
    frames = _tatica_frames_cached(_fr_key, dados_prep, tuple(atletas_sel or []),
                                   float(_t_ini), float(_t_fim), 400)
    if frames is None:
        st.warning("Janela sem sobreposição temporal suficiente — ajuste o início ou a duração.")
        return
    tempos, nomes, equipes, posicoes, PX, PY, PV = frames
    nf, natl = PX.shape

    dur_s = float(tempos[-1] - tempos[0])
    _dt = dur_s / max(1, nf - 1)
    st.caption(f"⏱️ {nf} frames · ~{_dt:.1f}s entre frames · {natl} atletas sincronizados · "
               f"campo {FL:.0f}×{FW:.0f} m · 📍 {_fonte_pos}")
    if _dt > 3.0:
        st.caption("ℹ️ Trecho longo: os frames ficam espaçados (~{:.0f}s) e o movimento "
                   "aparece em **saltos**. Para ver o deslocamento contínuo, reduza a janela "
                   "(ex.: 1–2 min) e arraste o **início** pelo trecho que quer analisar.".format(_dt))
    if _n_proj > 0:
        st.info("📍 Coordenadas de campo **reconstruídas a partir do GPS** (lat/lon → campo). "
                "As posições **relativas** entre jogadores são fiéis; o alinhamento absoluto do "
                "campo é aproximado. Para registro exato, configure o campo na aba "
                "**Campo & GPS** (será usado automaticamente aqui).")

    if vis.startswith("🎯"):
        _tatica_view_pitch_control(tempos, nomes, equipes, PX, PY, PV, FL, FW)
        st.caption("🎯 **Pitch Control** (modelo de William Spearman, Liverpool FC). A cor mostra "
                   "quão **dominado** está cada ponto do campo — calculado pelo tempo de chegada "
                   "do jogador mais próximo, com a posição projetada pela velocidade atual. "
                   "Quente = espaço sob controle; transparente = espaço livre. "
                   "Com 2 equipes nos dados, vira controle **contestado** (azul × vermelho). "
                   "⚠️ O mapa de calor avança **em passos** (é uma sequência de fotos do campo, "
                   "não desliza). Para ver os atletas **deslizando**, use 🫁 Respiração ou "
                   "📏 Distância; e reduza a janela (1–2 min) para passos menores.")
    elif vis.startswith("🫁"):
        _tatica_view_respiracao(tempos, nomes, equipes, PX, PY, PV, FL, FW)
        st.caption("🫁 **Respiração da equipe**: o polígono (casco convexo) e o centroide (✕) "
                   "mostram o bloco **comprimindo** na marcação e **expandindo** na posse. "
                   "▶ No Play os atletas **deslizam** de forma contínua. "
                   "O gráfico abaixo acompanha largura, comprimento e área ao longo do tempo.")
    elif vis.startswith("🔷"):
        _tatica_view_voronoi(tempos, nomes, equipes, PX, PY, PV, FL, FW)
        st.caption("🔷 **Voronoi**: cada célula do campo é colorida pelo jogador **mais próximo**. "
                   "Células grandes = jogador cobrindo muito espaço; zonas sem dono = buracos "
                   "de cobertura. ⚠️ Como o Pitch Control, o 'vitral' avança **em passos**; "
                   "para deslizamento contínuo use 🫁 Respiração ou 📏 Distância.")
    elif vis.startswith("🎥"):
        _tatica_view_replay3d(tempos, nomes, equipes, PX, PY, PV, FL, FW)
        st.caption("🎥 **Replay 3D**: 'broadcast sintético' reconstruído só das coordenadas. "
                   "Arraste para girar a câmera, role para dar zoom e use Play para animar.")
    else:
        _tatica_view_distancias(tempos, nomes, equipes, PX, PY, PV, FL, FW)
        st.caption("📏 **Distância entre atletas**: a matriz mostra a distância média (em metros) "
                   "entre cada par; o gráfico e o slider permitem achar os instantes de **maior** "
                   "(equipe aberta) e **menor** distância (equipe compacta). Útil para ler "
                   "compactação, linhas e relações entre setores. "
                   "▶ No campo animado os atletas **deslizam** de forma contínua.")


