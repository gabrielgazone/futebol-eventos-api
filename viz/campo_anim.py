# -*- coding: utf-8 -*-
"""Animacao de esforcos no campo — extraido de viz/janelas para reuso pelas abas
Esforcos, WCS e Janelas (e pelo modulo RHIE). Contem os mapas de tempo dos
periodos e o plot animado (play/pause) de um conjunto de atletas numa janela."""
from __future__ import annotations

import applog as _applog
from field import desenhar_campo_futebol_bonito
from field import gps_para_campo_coords
import numpy as np  # noqa: F401  (usado por helpers importados no futuro)
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _build_period_maps(dados_sensor, modo_todos, periodo_janela, dados_posicao):
    """Mapas de tempo (match-time) dos períodos, para localizar o segmento de
    GPS de uma janela de esforço. Trata sub-períodos sobrepostos (substituto que
    entra no meio de um período). Mesma lógica do modo Time Completo, extraída
    para ser reutilizada pelos modos Individual e Por Posição.

    Função PURA (sem Streamlit) — coberta por teste. Retorna dict com:
      order        — lista de períodos considerados
      abs          — {período: (first_ts, last_ts)} em segundos (sensor IMU)
      sorted_by_ts — períodos ordenados pelo ts inicial
      start_min    — {período: minuto de início no tempo de jogo}
      atl_offset   — fn(atleta) -> minuto de match-time do 1º período do atleta
    """
    _order = list(dados_sensor.keys()) if modo_todos else [periodo_janela]

    def _abs_ts(_pnm):
        _mn, _mx = None, None
        for _spl in dados_sensor.get(_pnm, {}).values():
            for _pp in _spl:
                _tt = float(_pp.get('ts') or 0) + float(_pp.get('cs') or 0) / 100.0
                if _tt <= 0:
                    continue
                if _mn is None or _tt < _mn:
                    _mn = _tt
                if _mx is None or _tt > _mx:
                    _mx = _tt
        return (_mn or 0.0, _mx or 0.0)

    _abs = {_pn: _abs_ts(_pn) for _pn in _order}
    _sorted_by_ts = sorted(_order, key=lambda _p: _abs[_p][0])
    _start_min: dict = {}
    _cum = 0.0
    _active: list = []
    for _pn_s in _sorted_by_ts:
        _ft_s, _lt_s = _abs[_pn_s]
        _dur = (_lt_s - _ft_s) / 60.0 if _lt_s > _ft_s else 0.0
        _active = [_m for _m in _active if _m[2] > _ft_s]
        _par = next((_m for _m in _active if _m[1] < _ft_s < _m[2]), None)
        if _par is None:
            _start_min[_pn_s] = _cum
            _active.append((_pn_s, _ft_s, _lt_s, _cum))
            _cum += _dur
        else:
            _, _par_ft, _, _par_ms = _par
            _start_min[_pn_s] = _par_ms + (_ft_s - _par_ft) / 60.0

    def _atl_offset(_atl_nm):
        for _pn in _sorted_by_ts:
            if (dados_posicao.get(_pn, {}).get(_atl_nm, {}).get('vel')
                    or dados_sensor.get(_pn, {}).get(_atl_nm)):
                return _start_min.get(_pn, 0.0)
        return 0.0

    return {'order': _order, 'abs': _abs, 'sorted_by_ts': _sorted_by_ts,
            'start_min': _start_min, 'atl_offset': _atl_offset}


def _periodo_label_de_t(maps, t_min):
    """Rótulo do período que contém o minuto de match-time t_min."""
    for _pn in maps['order']:
        _s = maps['start_min'].get(_pn, 0.0)
        _ab = maps['abs'].get(_pn, (0.0, 0.0))
        _e = _s + (_ab[1] - _ab[0]) / 60.0
        if _s <= t_min <= _e + 0.1:
            return _pn
    return maps['order'][0] if maps['order'] else '?'


def _animar_esforco_campo(atletas, t_ini_min, window_minutes, dados_posicao,
                          maps, unidade, lbl_ini, lbl_fim, valor, per_lbl,
                          campo_cfg=None, min_atl=1):
    """Anima no campo os `atletas` durante a janela [t_ini_min, +window_minutes]
    (em match-time). `maps` vem de _build_period_maps(). Extrai, para cada
    atleta, o segmento de GPS que cobre a janela (com fallback lats/lons →
    coords de campo) e monta um plotly animado (play/pause/slider). Mostra
    st.info se não houver GPS suficiente. Mesma mecânica do Time Completo."""
    _start_min = maps['start_min']
    _abs = maps['abs']
    _order = maps['order']
    _sorted_by_ts = maps['sorted_by_ts']

    _cfg = campo_cfg
    if _cfg is None:
        for _hk in list(st.session_state.keys()):
            if (_hk.startswith("campo_cfg__")
                    and isinstance(st.session_state[_hk], dict)):
                _cfg = st.session_state[_hk]
                break
    _fl = float(_cfg.get('fl', 105) if _cfg else 105)
    _fw = float(_cfg.get('fw', 68) if _cfg else 68)

    _pend = {
        _pn: (_start_min.get(_pn, 0.0)
              + (_abs.get(_pn, (0.0, 0.0))[1] - _abs.get(_pn, (0.0, 0.0))[0]) / 60.0)
        for _pn in _order}

    _pal = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
            '#98D8C8', '#F7DC6F', '#BB8FCE', '#76D7C4', '#F1948A', '#85C1E9',
            '#82E0AA', '#F8C471', '#AED6F1']

    _amap: dict = {}
    _hz_ref = 10.0
    for _ci, _atl in enumerate(atletas):
        for _pn in _sorted_by_ts:
            _pos = dados_posicao.get(_pn, {}).get(_atl, {})
            _xs = list(_pos.get('xs', []))
            _ys = list(_pos.get('ys', []))
            if not _xs and _cfg:
                _lts = _pos.get('lats', [])
                _lns = _pos.get('lons', [])
                if _lts and _lns:
                    try:
                        _xs, _ys = gps_para_campo_coords(_lts, _lns, _cfg)
                    except Exception:
                        _applog.log_debug_exc()
            if not _xs:
                continue
            _ps = _start_min.get(_pn, 0.0)
            _pe = _pend.get(_pn, _ps)
            if not (_ps <= t_ini_min <= _pe):
                continue
            _dur_s = (_abs[_pn][1] - _abs[_pn][0]) if _abs.get(_pn) else 0.0
            _hz = len(_xs) / _dur_s if _dur_s > 0 else 10.0
            _hz_ref = _hz
            _off_s = (t_ini_min - _ps) * 60.0
            _n_smp = max(2, int(window_minutes * 60 * _hz))
            _is = int(_off_s * _hz)
            _ie = min(_is + _n_smp, len(_xs))
            if 0 <= _is < len(_xs):
                _vs = list(_pos.get('vel', []))
                _amap[_atl] = {
                    'xs': _xs[_is:_ie],
                    'ys': _ys[_is:_ie] if _ys else [0] * (_ie - _is),
                    'vel': _vs[_is:_ie] if _vs else [0] * (_ie - _is),
                    'color': _pal[_ci % len(_pal)],
                    'label': (_atl.split()[-1][:10] if _atl.split() else _atl[:10]),
                }
            break

    if len(_amap) < max(1, min_atl):
        st.info(
            "GPS insuficiente para este esforço "
            f"({len(_amap)} atleta(s) com dados de posição). Verifique se o campo "
            "foi configurado e se os dados de GPS foram importados.")
        return

    _fig = desenhar_campo_futebol_bonito(
        field_length=_fl, field_width=_fw,
        title=f"🎬 {lbl_ini}→{lbl_fim} | {per_lbl} | {valor:.1f} {unidade}")

    _atls = list(_amap.keys())
    _tidxs = []
    for _pa in _atls:
        _wd = _amap[_pa]
        _fig.add_trace(go.Scatter(
            x=[_wd['xs'][0]] if _wd['xs'] else [0],
            y=[_wd['ys'][0]] if _wd['ys'] else [0],
            mode='markers+text',
            marker=dict(size=20, color=_wd['color'], symbol='circle',
                        line=dict(color='white', width=2)),
            text=[_wd['label']], textposition='top center',
            textfont=dict(color='white', size=8),
            name=_pa, showlegend=True))
        _tidxs.append(len(_fig.data) - 1)

    _wl = max(len(_amap[a]['xs']) for a in _atls)
    _step = max(1, _wl // 80)
    _fr = list(range(0, _wl, _step))
    if _fr and _fr[-1] != _wl - 1:
        _fr.append(_wl - 1)

    _frames = []
    for _fi in _fr:
        _ts = _fi / _hz_ref
        _mm = int(_ts // 60)
        _ss = int(_ts % 60)
        _fd = []
        for _pa in _atls:
            _wd = _amap[_pa]
            _xi = (_wd['xs'][_fi] if _fi < len(_wd['xs'])
                   else (_wd['xs'][-1] if _wd['xs'] else 0))
            _yi = (_wd['ys'][_fi] if _fi < len(_wd['ys'])
                   else (_wd['ys'][-1] if _wd['ys'] else 0))
            _fd.append(go.Scatter(
                x=[_xi], y=[_yi], mode='markers+text',
                marker=dict(size=20, color=_wd['color'], symbol='circle',
                            line=dict(color='white', width=2)),
                text=[_wd['label']], textposition='top center',
                textfont=dict(color='white', size=8)))
        _frames.append(go.Frame(
            data=_fd, traces=_tidxs, name=str(_fi),
            layout=go.Layout(title=dict(
                text=f"🎬 {lbl_ini}→{lbl_fim} | ⏱️ +{_mm}:{_ss:02d} min",
                font=dict(color='white', size=12)))))

    _fig.frames = _frames
    _fig.update_layout(
        height=580,
        updatemenus=[dict(
            type='buttons', showactive=False, y=0, x=0.5, xanchor='center',
            buttons=[
                dict(label='▶ Play', method='animate', args=[None, dict(
                    frame=dict(duration=100, redraw=True), fromcurrent=True,
                    transition=dict(duration=100, easing='linear'),
                    mode='immediate')]),
                dict(label='⏸ Pause', method='animate', args=[[None], dict(
                    frame=dict(duration=0, redraw=False), mode='immediate')])])],
        sliders=[dict(
            steps=[dict(args=[[f.name], dict(
                frame=dict(duration=0, redraw=True), mode='immediate')],
                method='animate', label='') for f in _frames],
            x=0.0, y=-0.05, len=1.0, currentvalue=dict(visible=False))],
        legend=dict(orientation='h', yanchor='bottom', y=-0.30,
                    xanchor='center', x=0.5, font=dict(color='white', size=8)))
    st.plotly_chart(_fig, use_container_width=True)


def _tabela_e_anima_esforcos(eventos, atletas_anim, maps, dados_posicao,
                             window_minutes, unidade, tipo_metrica, key, titulo,
                             offset_min=0.0, min_atl=1):
    """Mostra uma tabela selecionável de esforços e anima no campo o esforço
    escolhido. Reutilizado pelos modos Individual e Por Posição.

    offset_min desloca o t_ini_min do evento para match-time (Individual usa o
    offset do atleta; Por Posição usa 0, pois a média já está no grid da posição).
    """
    if not eventos:
        st.info("Nenhum esforço de média-alta ou alta intensidade encontrado.")
        return
    st.markdown("---")
    st.markdown(titulo)
    _rows = []
    for _rk, _ev in enumerate(eventos, 1):
        _tm = _ev['t_ini_min'] + offset_min
        _rows.append({
            '#': _rk, 'Início': _ev['inicio'], 'Fim': _ev['fim'],
            'Período': _periodo_label_de_t(maps, _tm),
            f'{tipo_metrica} ({unidade})': _ev['valor'],
            '% do Máx': _ev['pct_max'], 'Intensidade': _ev['intensidade'],
        })
    _df = pd.DataFrame(_rows)
    _fmt = {f'{tipo_metrica} ({unidade})': '{:.1f}', '% do Máx': '{:.1f}%'}
    _sel = st.dataframe(
        _df.style.format(_fmt), use_container_width=True,
        height=min(430, 44 + len(_rows) * 36),
        on_select="rerun", selection_mode="single-row", key=key)
    _idx = (_sel.selection.rows[0]
            if (hasattr(_sel, 'selection') and _sel.selection.rows) else 0)
    _ev = eventos[_idx]
    _tm_sel = _ev['t_ini_min'] + offset_min
    _per = _periodo_label_de_t(maps, _tm_sel)
    st.caption(
        f"**Clique numa linha** da tabela para escolher o esforço. Exibindo: "
        f"**{_ev['inicio']}→{_ev['fim']}** ({_per})")
    _animar_esforco_campo(
        atletas_anim, _tm_sel, window_minutes, dados_posicao, maps, unidade,
        _ev['inicio'], _ev['fim'], _ev['valor'], _per, min_atl=min_atl)
