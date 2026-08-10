# -*- coding: utf-8 -*-
"""Testes do motor de export WCS multi-atividade (wcs_export)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wcs_export as wx  # noqa: E402


def _pts(vels_kmh, pl=None, acc=None):
    """Amostras de sensor a partir de velocidade (km/h) → 'v' em m/s."""
    out = []
    for i, v in enumerate(vels_kmh):
        p = {'v': v / 3.6}
        if pl is not None:
            p['pl'] = pl[i]
        if acc is not None:
            p['a'] = acc[i]
        out.append(p)
    return out


# ── serie_por_amostra ────────────────────────────────────────────────────────
def test_serie_distancia():
    # 36 km/h = 10 m/s → a 10 Hz cada amostra contribui 1 m
    sv = wx.serie_por_amostra(_pts([36.0] * 10), wx.VAR_DIST, hz=10)
    assert len(sv) == 10
    assert all(abs(s - 1.0) < 1e-9 for s in sv)


def test_serie_hsr_filtra_abaixo_do_corte():
    sv = wx.serie_por_amostra(_pts([10.0] * 5 + [36.0] * 5), wx.VAR_HSR,
                              hz=10, hsr_kmh=19.8)
    assert sum(sv[:5]) == 0.0          # abaixo do corte não conta
    assert abs(sum(sv[5:]) - 5.0) < 1e-9


def test_serie_sprint_corte_mais_alto():
    # 22 km/h entra no HSR mas NÃO no sprint (25.2)
    sv = wx.serie_por_amostra(_pts([22.0] * 10), wx.VAR_SPRINT, hz=10,
                              sprint_kmh=25.2)
    assert sum(sv) == 0.0


def test_serie_vmax_e_playerload():
    # tolerância: km/h → m/s → km/h introduz erro de ponto flutuante
    assert abs(wx.serie_por_amostra(_pts([18.0, 30.0]), wx.VAR_VMAX)[1] - 30.0) < 1e-9
    sv = wx.serie_por_amostra(_pts([10.0] * 3, pl=[1.0, 2.0, 3.0]), wx.VAR_PL)
    assert sv == [1.0, 2.0, 3.0]


def test_serie_acc_e_dec_contam_acoes():
    acc = [0.0] * 5 + [3.5] * 8 + [0.0] * 5 + [-3.5] * 8 + [0.0] * 4
    pts = _pts([10.0] * len(acc), acc=acc)
    assert sum(wx.serie_por_amostra(pts, wx.VAR_ACC, hz=10, acc_ms2=3.0)) == 1.0
    assert sum(wx.serie_por_amostra(pts, wx.VAR_DEC, hz=10, dec_ms2=3.0)) == 1.0


def test_serie_variavel_desconhecida():
    try:
        wx.serie_por_amostra(_pts([10.0]), 'inexistente')
        assert False, "deveria levantar ValueError"
    except ValueError:
        pass


# ── pico_janela ──────────────────────────────────────────────────────────────
def test_pico_soma_acha_melhor_janela():
    sv = [1.0] * 5 + [10.0] * 3 + [1.0] * 5      # pico nos 3 do meio
    val, si, ei = wx.pico_janela(sv, 3, is_max=False)
    assert abs(val - 30.0) < 1e-9
    assert (si, ei) == (5, 8)


def test_pico_max_usa_maximo_da_janela():
    val, _, _ = wx.pico_janela([5.0, 30.0, 7.0, 8.0], 2, is_max=True)
    assert val == 30.0


def test_pico_serie_menor_que_janela():
    assert wx.pico_janela([1.0, 2.0], 10) == (0.0, 0, 0)
    assert wx.pico_janela([], 5) == (0.0, 0, 0)


def test_pico_bate_com_rolling_sum_canonico():
    """Garante o MESMO método da aba WCS (metrics.rolling_sum + argmax)."""
    import numpy as np
    import metrics as _mtr
    sv = [float(v) for v in np.random.default_rng(7).random(300) * 3]
    n = 40
    esperado = max(_mtr.rolling_sum(sv, n))
    val, _, _ = wx.pico_janela(sv, n)
    assert abs(val - esperado) < 1e-9


# ── calcular_wcs ─────────────────────────────────────────────────────────────
def test_calcular_wcs_multi_janela_e_relativa():
    # 60 s a 36 km/h (10 m/s) = 600 m em 1 min
    pts = _pts([36.0] * 600)
    r = wx.calcular_wcs(pts, [wx.VAR_DIST, wx.VAR_DIST_REL], [1], hz=10)
    assert abs(r[(wx.VAR_DIST, 1)] - 600.0) < 0.5
    assert abs(r[(wx.VAR_DIST_REL, 1)] - 600.0) < 0.5   # 600 m / 1 min


def test_calcular_wcs_omite_janela_maior_que_serie():
    pts = _pts([36.0] * 600)                  # só 1 min de dados
    r = wx.calcular_wcs(pts, [wx.VAR_DIST], [1, 3, 5], hz=10)
    assert (wx.VAR_DIST, 1) in r
    assert (wx.VAR_DIST, 3) not in r          # não vira 0 — é omitida
    assert (wx.VAR_DIST, 5) not in r


def test_calcular_wcs_relativa_3min():
    # 3 min a 36 km/h → 1800 m na janela; relativa = 600 m/min
    r = wx.calcular_wcs(_pts([36.0] * 1800), [wx.VAR_DIST_REL], [3], hz=10)
    assert abs(r[(wx.VAR_DIST_REL, 3)] - 600.0) < 1.0


def test_calcular_wcs_vazio():
    assert wx.calcular_wcs([], [wx.VAR_DIST], [1]) == {}
