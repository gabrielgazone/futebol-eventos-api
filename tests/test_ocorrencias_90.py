# -*- coding: utf-8 -*-
"""Ocorrências >= X% do máximo DO PRÓPRIO ATLETA (esforços não-sobrepostos)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wcs_export as wx  # noqa: E402


def test_dois_esforcos_distintos_nao_contam_como_sete():
    """O caso do exemplo: 2 passagens intensas separadas. Contando todas as
    janelas deslizantes daria ~7 (artefato do passo); esforços distintos = 2."""
    # janela de 10 amostras; dois blocos altos separados por um vale longo
    sv = ([1.0] * 20 + [10.0] * 15 + [1.0] * 60 + [10.0] * 15 + [1.0] * 20)
    n = 10
    assert wx.ocorrencias_acima_pct(sv, n, pct=0.90) == 2


def test_um_esforco_do_tamanho_da_janela_conta_uma_vez():
    sv = [1.0] * 20 + [10.0] * 12 + [1.0] * 20      # ~1 janela
    assert wx.ocorrencias_acima_pct(sv, 10, pct=0.90) == 1


def test_esforco_sustentado_por_3_janelas_conta_3():
    """Semântica de não-sobreposição (a mesma das abas WCS/Janelas): manter o
    pico por 3x o tamanho da janela SÃO 3 janelas distintas no máximo. Ex.: quem
    sustenta a intensidade do seu pico de 1 min por 3 min fez 3 janelas de 1 min
    em nível de pico — não 1."""
    sv = [1.0] * 20 + [10.0] * 30 + [1.0] * 20      # 3x a janela
    assert wx.ocorrencias_acima_pct(sv, 10, pct=0.90) == 3


def test_limiar_e_relativo_ao_proprio_maximo():
    """Dois atletas com magnitudes diferentes e o MESMO padrão devem dar a mesma
    contagem — o limiar é 90% do máximo de cada um."""
    padrao = [1.0] * 20 + [10.0] * 15 + [1.0] * 60 + [10.0] * 15 + [1.0] * 20
    atleta_a = padrao
    atleta_b = [_v * 3.7 for _v in padrao]        # muito mais intenso
    assert (wx.ocorrencias_acima_pct(atleta_a, 10, 0.90)
            == wx.ocorrencias_acima_pct(atleta_b, 10, 0.90) == 2)


def test_pct_menor_encontra_mais_esforcos():
    # um pico alto e outro médio: 90% pega só o alto; 60% pega os dois
    sv = ([1.0] * 20 + [10.0] * 12 + [1.0] * 60 + [7.0] * 12 + [1.0] * 20)
    assert wx.ocorrencias_acima_pct(sv, 10, pct=0.90) == 1
    assert wx.ocorrencias_acima_pct(sv, 10, pct=0.60) == 2


def test_serie_menor_que_janela_ou_vazia():
    assert wx.ocorrencias_acima_pct([1.0] * 5, 10) == 0
    assert wx.ocorrencias_acima_pct([], 10) == 0
    assert wx.ocorrencias_acima_pct([0.0] * 50, 10) == 0     # máximo 0


def test_o_pico_sempre_conta_pelo_menos_uma_vez():
    """Se há qualquer atividade, o próprio máximo é >= pct*máximo → >= 1."""
    sv = [0.0] * 30 + [5.0] * 12 + [0.0] * 30
    assert wx.ocorrencias_acima_pct(sv, 10, pct=0.90) >= 1


# ── calcular_ocorrencias (por variável × janela) ─────────────────────────────
def _pts_dois_picos():
    """~4 min: dois trechos rápidos separados por trote."""
    vel = ([8.0] * 600 + [30.0] * 300 + [8.0] * 600 + [30.0] * 300
           + [8.0] * 600)
    return [{'v': v / 3.6, 'ts': 1000 + i / 10.0, 'cs': 0} for i, v in
            enumerate(vel)]


def test_calcular_ocorrencias_por_variavel_e_janela():
    pts = _pts_dois_picos()
    r = wx.calcular_ocorrencias(pts, [wx.VAR_DIST, wx.VAR_HSR], [1, 3],
                                hz=10.0, pct=0.90)
    assert (wx.VAR_DIST, 1) in r and (wx.VAR_DIST, 3) in r
    assert (wx.VAR_HSR, 1) in r
    assert all(_v >= 1 for _v in r.values())      # o pico sempre conta


def test_calcular_ocorrencias_omite_janela_maior_que_serie():
    pts = _pts_dois_picos()                        # ~4 min
    r = wx.calcular_ocorrencias(pts, [wx.VAR_DIST], [1, 10], hz=10.0)
    assert (wx.VAR_DIST, 1) in r
    assert (wx.VAR_DIST, 10) not in r              # não cabe → omitida


def test_calcular_ocorrencias_vazio():
    assert wx.calcular_ocorrencias([], [wx.VAR_DIST], [1]) == {}
