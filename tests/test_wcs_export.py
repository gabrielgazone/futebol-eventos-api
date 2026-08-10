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


def test_serie_playerload():
    sv = wx.serie_por_amostra(_pts([10.0] * 3, pl=[1.0, 2.0, 3.0]), wx.VAR_PL)
    assert sv == [1.0, 2.0, 3.0]


def test_serie_acc_por_banda_b2_vs_b3():
    # bandas explicitas (independe dos defaults): B2=3-4, B3=4-10
    # pico 3.5 -> B2; pico 5.0 -> B3
    acc = [0.0] * 5 + [3.5] * 8 + [0.0] * 5 + [5.0] * 8 + [0.0] * 4
    pts = _pts([10.0] * len(acc), acc=acc)
    _b = dict(acc_b2=(3.0, 4.0), acc_b3=(4.0, 10.0))
    assert sum(wx.serie_por_amostra(pts, wx.VAR_ACC_B2, hz=10, **_b)) == 1.0
    assert sum(wx.serie_por_amostra(pts, wx.VAR_ACC_B3, hz=10, **_b)) == 1.0
    # B2+ = B2 + B3
    assert sum(wx.serie_por_amostra(pts, wx.VAR_ACC_B2P, hz=10, **_b)) == 2.0


def test_serie_dec_por_banda_b2_vs_b3():
    acc = [0.0] * 5 + [-3.5] * 8 + [0.0] * 5 + [-5.0] * 8 + [0.0] * 4
    pts = _pts([10.0] * len(acc), acc=acc)
    _b = dict(dec_b2=(-4.0, -3.0), dec_b3=(-10.0, -4.0))
    assert sum(wx.serie_por_amostra(pts, wx.VAR_DEC_B2, hz=10, **_b)) == 1.0
    assert sum(wx.serie_por_amostra(pts, wx.VAR_DEC_B3, hz=10, **_b)) == 1.0
    assert sum(wx.serie_por_amostra(pts, wx.VAR_DEC_B2P, hz=10, **_b)) == 2.0


def test_b2mais_e_soma_de_b2_e_b3():
    """Invariante: B2+ == B2 + B3 (acel e desacel), para qualquer sinal."""
    import numpy as _np
    rng = _np.random.default_rng(11)
    acc = (rng.random(3000) * 12 - 6).tolist()
    pts = _pts([10.0] * len(acc), acc=acc)
    for b2, b3, b2p in ((wx.VAR_ACC_B2, wx.VAR_ACC_B3, wx.VAR_ACC_B2P),
                        (wx.VAR_DEC_B2, wx.VAR_DEC_B3, wx.VAR_DEC_B2P)):
        n2 = sum(wx.serie_por_amostra(pts, b2, hz=10))
        n3 = sum(wx.serie_por_amostra(pts, b3, hz=10))
        n2p = sum(wx.serie_por_amostra(pts, b2p, hz=10))
        assert n2p == n2 + n3, f"{b2p}: {n2p} != {n2} + {n3}"


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
def test_calcular_wcs_distancia_na_janela():
    # 60 s a 36 km/h (10 m/s) = 600 m em 1 min
    pts = _pts([36.0] * 600)
    r = wx.calcular_wcs(pts, [wx.VAR_DIST], [1], hz=10)
    assert abs(r[(wx.VAR_DIST, 1)] - 600.0) < 0.5


def test_calcular_wcs_omite_janela_maior_que_serie():
    pts = _pts([36.0] * 600)                  # só 1 min de dados
    r = wx.calcular_wcs(pts, [wx.VAR_DIST], [1, 3, 5], hz=10)
    assert (wx.VAR_DIST, 1) in r
    assert (wx.VAR_DIST, 3) not in r          # não vira 0 — é omitida
    assert (wx.VAR_DIST, 5) not in r


def test_calcular_wcs_vazio():
    assert wx.calcular_wcs([], [wx.VAR_DIST], [1]) == {}


def test_variaveis_sem_relativa_e_sem_vmax():
    """Distância relativa e velocidade máxima foram removidas do export."""
    assert not hasattr(wx, 'VAR_DIST_REL')
    assert not hasattr(wx, 'VAR_VMAX')
    assert len(wx.VARIAVEIS) == 10
    for _v in wx.VARIAVEIS:
        assert 'relativa' not in _v.lower() and 'máx' not in _v.lower()


def test_b2mais_equivale_ao_motor_do_app():
    """B2+ (união das bandas) deve casar com metrics.detect_actions — garante
    que o export não divirja do que as outras abas do app mostram."""
    import numpy as _np
    import metrics as _mtr
    rng = _np.random.default_rng(23)
    acc = (rng.random(4000) * 14 - 7).tolist()
    pts = _pts([10.0] * len(acc), acc=acc)

    for b2p, bandas, in ((wx.VAR_ACC_B2P, [wx.DEFAULT_ACC_B2, wx.DEFAULT_ACC_B3]),
                         (wx.VAR_DEC_B2P, [wx.DEFAULT_DEC_B2, wx.DEFAULT_DEC_B3])):
        meu = sum(wx.serie_por_amostra(pts, b2p, hz=10, min_dur_acc_s=0.6))
        app = len(_mtr.detect_actions(
            _np.asarray(acc), [{'min': lo, 'max': hi} for lo, hi in bandas],
            min_dur_s=0.6, hz=10))
        assert meu == app, f"{b2p}: export={meu} vs app={app}"


def test_defaults_das_bandas_do_estudo():
    """Padrões definidos pelo usuário do estudo: B2 = 2,5–3,5 · B3 = 3,5–10."""
    assert wx.DEFAULT_ACC_B2 == (2.5, 3.5)
    assert wx.DEFAULT_ACC_B3 == (3.5, 10.0)
    assert wx.DEFAULT_DEC_B2 == (-3.5, -2.5)     # magnitude 2,5–3,5
    assert wx.DEFAULT_DEC_B3 == (-10.0, -3.5)
    # contíguas: o topo de B2 é o piso de B3 (sem lacuna nem sobreposição)
    assert wx.DEFAULT_ACC_B2[1] == wx.DEFAULT_ACC_B3[0]
    assert wx.DEFAULT_DEC_B2[0] == wx.DEFAULT_DEC_B3[1]


def test_defaults_classificam_pico_3_0_como_b2():
    """Com os novos padrões, um pico de 3,0 m/s² cai em B2 (2,5–3,5)."""
    acc = [0.0] * 5 + [3.0] * 8 + [0.0] * 5
    pts = _pts([10.0] * len(acc), acc=acc)
    assert sum(wx.serie_por_amostra(pts, wx.VAR_ACC_B2, hz=10)) == 1.0
    assert sum(wx.serie_por_amostra(pts, wx.VAR_ACC_B3, hz=10)) == 0.0


def test_acc_dec_presentes_mesmo_sem_aceleracao_nativa():
    """Sem campo 'a', as bandas nao devem sumir da tabela: a serie e derivada
    por dv/dt da velocidade (espelha o fallback da aba WCS).

    Usa RAMPAS sustentadas: ~3 m/s² exige dv = 0,3 m/s por amostra a 10 Hz
    (= 1,08 km/h), mantido por >= 6 frames (duracao minima de 0,6 s).
    """
    vel = [5.0]
    for _ in range(15):                       # aceleracao sustentada ~3 m/s²
        vel.append(vel[-1] + 1.1)
    vel += [vel[-1]] * 20                     # plato
    for _ in range(15):                       # desaceleracao sustentada
        vel.append(max(0.0, vel[-1] - 1.1))
    vel += [vel[-1]] * 20
    pts = _pts(vel)                           # SEM 'a'
    for _v in (wx.VAR_ACC_B2P, wx.VAR_DEC_B2P):
        sv = wx.serie_por_amostra(pts, _v, hz=10)
        assert sv, f"{_v}: serie vazia"
        assert sum(sv) > 0, f"{_v}: derivacao dv/dt nao gerou acoes"


def test_bandas_presentes_na_saida_mesmo_com_zero():
    """Mesmo sem nenhuma acao, a variavel aparece com 0 (coluna nao desaparece)."""
    pts = _pts([10.0] * 600, acc=[0.0] * 600)   # 'a' presente e plano
    r = wx.calcular_wcs(pts, [wx.VAR_ACC_B2, wx.VAR_DEC_B3], [1], hz=10)
    assert (wx.VAR_ACC_B2, 1) in r and (wx.VAR_DEC_B3, 1) in r
