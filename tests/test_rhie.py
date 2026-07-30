# -*- coding: utf-8 -*-
"""Testes do motor RHIE (analysis): detecção de ações de alta intensidade e
agrupamento em blocos (Repeated High-Intensity Efforts)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import (  # noqa: E402
    detectar_acoes_rhie, agrupar_blocos_rhie, calcular_blocos_rhie)


# ── agrupar_blocos_rhie ──────────────────────────────────────────────────────
def test_agrupar_menos_que_minimo_nao_e_bloco():
    acoes = [{'frame': 0, 'tipo': 'vel'}, {'frame': 10, 'tipo': 'acc'}]
    assert agrupar_blocos_rhie(acoes, hz=10, min_acoes=3, recuperacao_s=21) == []


def test_agrupar_um_bloco_composicao_e_duracao():
    acoes = [{'frame': 0, 'tipo': 'vel'},
             {'frame': 50, 'tipo': 'acc'},
             {'frame': 100, 'tipo': 'dec'}]
    b = agrupar_blocos_rhie(acoes, hz=10, min_acoes=3, recuperacao_s=21)
    assert len(b) == 1
    assert b[0]['n_vel'] == 1 and b[0]['n_acc'] == 1 and b[0]['n_dec'] == 1
    assert b[0]['n_total'] == 3
    assert b[0]['frame_ini'] == 0 and b[0]['frame_fim'] == 100
    assert abs(b[0]['dur_s'] - 10.0) < 1e-9        # 100 frames / 10 Hz


def test_agrupar_split_por_recuperacao():
    # dois grupos de 3, separados por > 21 s (210 frames a 10 Hz)
    acoes = [{'frame': f, 'tipo': 'vel'} for f in (0, 50, 100, 500, 550, 600)]
    b = agrupar_blocos_rhie(acoes, hz=10, min_acoes=3, recuperacao_s=21)
    assert len(b) == 2


def test_agrupar_grupo_pequeno_descartado():
    # 1º grupo com 3 (bloco); 2º com 2 (descartado)
    acoes = [{'frame': f, 'tipo': 'vel'} for f in (0, 50, 100, 500, 550)]
    b = agrupar_blocos_rhie(acoes, hz=10, min_acoes=3, recuperacao_s=21)
    assert len(b) == 1
    assert b[0]['frame_ini'] == 0


# ── detectar_acoes_rhie ──────────────────────────────────────────────────────
def test_detectar_velocidade_conta_entradas():
    vel = [10] * 5 + [22] * 5 + [10] * 5 + [22] * 5 + [10] * 5   # 2 entradas
    acc = [0.0] * len(vel)
    acoes = detectar_acoes_rhie(vel, acc, hz=10, vel_thr_kmh=19.8,
                                acc_thr_ms2=3.0, dec_thr_ms2=3.0)
    vel_ac = [a for a in acoes if a['tipo'] == 'vel']
    assert len(vel_ac) == 2
    assert vel_ac[0]['frame'] == 5                 # 1º frame acima do corte


def test_detectar_acc_e_dec_sustentados():
    vel = [10] * 30
    # min_dur padrão 0.6 s → 6 frames sustentados
    acc = [0.0] * 5 + [3.5] * 8 + [0.0] * 5 + [-3.5] * 8 + [0.0] * 4
    acoes = detectar_acoes_rhie(vel, acc, hz=10, vel_thr_kmh=19.8,
                                acc_thr_ms2=3.0, dec_thr_ms2=3.0)
    assert sum(1 for a in acoes if a['tipo'] == 'acc') == 1
    assert sum(1 for a in acoes if a['tipo'] == 'dec') == 1


# ── calcular_blocos_rhie (ponta a ponta) ─────────────────────────────────────
def test_calcular_blocos_rhie_e2e():
    # 3 entradas de velocidade próximas → 1 bloco de 3 ações de vel
    vel = ([10] * 3 + [22] * 3) * 3 + [10] * 10
    acc = [0.0] * len(vel)
    b = calcular_blocos_rhie(vel, acc, hz=10, vel_thr_kmh=19.8, acc_thr_ms2=3.0,
                             dec_thr_ms2=3.0, min_acoes=3, recuperacao_s=21)
    assert len(b) == 1
    assert b[0]['n_vel'] == 3 and b[0]['n_total'] == 3


def test_calcular_blocos_rhie_vazio():
    assert calcular_blocos_rhie([10] * 20, [0.0] * 20, hz=10, vel_thr_kmh=19.8,
                                acc_thr_ms2=3.0, dec_thr_ms2=3.0) == []
