# -*- coding: utf-8 -*-
"""Testes unitários da camada de compute/campo (P8) — funções puras dos
módulos analysis e field, agora testáveis isoladamente."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("streamlit")
import analysis as A  # noqa: E402
import field as F  # noqa: E402


# ── analysis.detectar_eventos_acc ───────────────────────────────────────────
class TestDetectarEventosAcc:
    def test_um_evento_sustentado(self):
        acc = np.array([0, 0, 0] + [5.0] * 8 + [0, 0, 0])
        ev = A.detectar_eventos_acc(acc, limiar=3, min_dur_s=0.6, freq_hz=10)
        assert ev.sum() == 1                     # min_frames=6; run de 8 -> 1 evento

    def test_curto_nao_conta(self):
        acc = np.array([5.0] * 4)                # < 6 frames
        assert A.detectar_eventos_acc(acc, limiar=3, min_dur_s=0.6, freq_hz=10).sum() == 0

    def test_desaceleracao_acima_false(self):
        acc = np.array([-5.0] * 8)
        assert A.detectar_eventos_acc(acc, limiar=3, acima=False, min_dur_s=0.6).sum() == 1


# ── analysis._segmentos_de_mask ─────────────────────────────────────────────
def test_segmentos_de_mask():
    assert A._segmentos_de_mask([False, True, True, False, True]) == [(1, 3), (4, 5)]
    assert A._segmentos_de_mask([False, False]) == []


# ── analysis.classificar_intensidade ────────────────────────────────────────
def test_classificar_intensidade():
    cores, cls = A.classificar_intensidade([100, 50, 10], limiar_alta=80, limiar_media=30)
    assert cores == ['#ef4444', '#f59e0b', '#22c55e']
    assert cls[0].startswith('Alta') and cls[1].startswith('Média') and cls[2].startswith('Baixa')


# ── analysis.acc_series_from_vel (delega a metrics) ─────────────────────────
def test_acc_series_from_vel_vel_constante():
    acc = A.acc_series_from_vel([10, 10, 10, 10], [0.0, 0.1, 0.2, 0.3], 10.0)
    assert len(acc) == 4
    assert max(abs(a) for a in acc) < 0.5        # vel constante -> ~0 m/s²


# ── field._get_pos_grupo ────────────────────────────────────────────────────
def test_get_pos_grupo():
    assert F._get_pos_grupo('Volante')[0] == 'Meio-campo'
    assert F._get_pos_grupo('GK')[0] == 'Goleiro'
    assert F._get_pos_grupo('Atacante')[0] == 'Atacante'
    assert F._get_pos_grupo('xyz')[0] == 'Outro'
    assert F._get_pos_grupo('')[0] == 'Outro'


# ── field._segmentos_continuos ──────────────────────────────────────────────
def test_segmentos_continuos():
    assert F._segmentos_continuos([1, 2, 3, 7, 8, 9], gap_max=1) == [[1, 2, 3], [7, 8, 9]]
    assert F._segmentos_continuos([1, 2], gap_max=1) == []     # < 3 pontos descartado
    assert F._segmentos_continuos([]) == []


# ── field.lat_lon_to_campo_coords ───────────────────────────────────────────
def test_lat_lon_to_campo_coords():
    x, y = F.lat_lon_to_campo_coords([0, 1], [0, 2])
    assert x == [0.0, 105.0] and y == [0.0, 68.0]       # normaliza -> escala FIFA
    assert F.lat_lon_to_campo_coords([], []) == ([], [])
