# -*- coding: utf-8 -*-
"""Testes do motor único de métricas (metrics.py) — cobre as fórmulas críticas
que já causaram bugs em produção (Hz, integração, ações, bandas, rolling)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as m  # noqa: E402

BANDAS_ACC = [
    {'min': 2, 'max': 3}, {'min': 3, 'max': 4}, {'min': 4, 'max': 10},
    {'min': -3, 'max': -2}, {'min': -4, 'max': -3}, {'min': -10, 'max': -4},
]
FAIXAS_VEL = [(0, 7), (7, 14.4), (14.4, 19.8), (19.8, 25.2), (25.2, 29.9), (29.9, 1e9)]


# ── estimate_hz ─────────────────────────────────────────────────────────────
class TestEstimateHz:
    def test_10hz_limpo(self):
        ts = [1000.0 + i * 0.1 for i in range(600)]
        assert abs(m.estimate_hz([ts]) - 10.0) < 0.2

    def test_10hz_com_ts_arredondado_para_segundos(self):
        # BUG histórico: mediana das diferenças detectava 1 Hz e inflava ~10×.
        ts = [1000 + int(i / 10) for i in range(600)]
        assert m.estimate_hz([ts]) > 9.0

    def test_1hz_real(self):
        ts = [1000.0 + i for i in range(60)]
        assert abs(m.estimate_hz([ts]) - 1.0) < 0.1

    def test_vazio_usa_default(self):
        assert m.estimate_hz([]) == 10.0
        assert m.estimate_hz(None, default=5.0) == 5.0

    def test_serie_curta_ignorada(self):
        assert m.estimate_hz([[1, 2, 3]], default=7.0) == 7.0

    def test_mediana_entre_series(self):
        ts10 = [1000.0 + i * 0.1 for i in range(600)]
        ts_ruim = [0.0] * 30            # span 0 → descartada
        assert abs(m.estimate_hz([ts10, ts_ruim]) - 10.0) < 0.2


# ── distância ───────────────────────────────────────────────────────────────
class TestDistancia:
    def test_velocidade_constante(self):
        # 18 km/h por 60 s a 10 Hz = 300 m
        d = m.per_sample_distance([18.0] * 600, 10.0)
        assert abs(sum(d) - 300.0) < 1.0

    def test_none_conta_zero(self):
        d = m.per_sample_distance([18.0, None, 18.0], 10.0)
        assert d[1] == 0.0 and d[0] > 0

    def test_bandas_dentro_e_fora(self):
        # 22 km/h → só banda 4 (19.8–25.2)
        d = m.dist_by_velocity_bands([22.0] * 100, FAIXAS_VEL, 10.0)
        assert d[3] > 0
        assert sum(d) == pytest.approx(d[3])

    def test_banda_ultima_sem_teto(self):
        d = m.dist_by_velocity_bands([40.0] * 100, FAIXAS_VEL, 10.0)
        assert d[5] > 0

    def test_limite_inferior_inclusivo(self):
        d = m.dist_by_velocity_bands([19.8] * 10, FAIXAS_VEL, 10.0)
        assert d[3] > 0 and d[2] == 0

    def test_per_sample_in_bands_fora_zera(self):
        sv = m.per_sample_distance_in_bands([10.0] * 50, [(19.8, 25.2)], 10.0)
        assert sum(sv) == 0.0

    def test_per_sample_in_bands_valor(self):
        # 22 km/h em banda → 22/36 m por amostra
        sv = m.per_sample_distance_in_bands([22.0], [(19.8, 25.2)], 10.0)
        assert sv[0] == pytest.approx(22.0 / 36.0)


# ── aceleração derivada ─────────────────────────────────────────────────────
class TestDeriveAcc:
    def test_velocidade_constante_acc_zero(self):
        ts = [i * 0.1 for i in range(100)]
        acc = m.derive_acc_from_vel([18.0] * 100, ts, 10.0)
        assert max(abs(a) for a in acc) < 1e-9

    def test_rampa_linear(self):
        # +1 m/s por segundo = 1 m/s² (checa o miolo, longe das bordas)
        ts = [i * 0.1 for i in range(100)]
        vel = [(i * 0.1) * 3.6 for i in range(100)]   # m/s → km/h
        acc = m.derive_acc_from_vel(vel, ts, 10.0)
        assert np.mean(acc[10:90]) == pytest.approx(1.0, abs=0.05)

    def test_saturacao(self):
        ts = [0, 0.1, 0.2]
        vel = [0.0, 360.0, 0.0]   # salto absurdo
        acc = m.derive_acc_from_vel(vel, ts, 10.0)
        assert max(acc) <= 10.0 and min(acc) >= -10.0

    def test_curto(self):
        assert m.derive_acc_from_vel([5.0], [0.0], 10.0) == [0.0]


# ── detecção de ações ───────────────────────────────────────────────────────
class TestDetectActions:
    def _sinal(self, eventos_pos, eventos_neg, n=30000, dur=10):
        acc = [0.1] * n
        for s in eventos_pos:
            for k in range(dur):
                acc[s + k] = 3.5
        for s in eventos_neg:
            for k in range(dur):
                acc[s + k] = -3.5
        return acc

    def test_conta_acoes_sustentadas_uma_vez(self):
        acc = self._sinal([1000, 5000, 9000], [3000, 12000])
        idx = m.detect_actions(acc, BANDAS_ACC, min_dur_s=0.6, hz=10)
        assert len(idx) == 5

    def test_ignora_mais_curto_que_min_dur(self):
        acc = [0.1] * 1000
        for k in range(3):                 # 0.3 s < 0.6 s
            acc[100 + k] = 3.5
        assert m.detect_actions(acc, BANDAS_ACC, min_dur_s=0.6, hz=10) == []

    def test_pico_b3_satura_no_topo(self):
        # BUG histórico: B3 (4–10) com pico acima do teto deve contar.
        acc = [0.1] * 1000
        for k in range(10):
            acc[100 + k] = 12.0            # acima de 10 (satura)
        b3 = [{'min': 4, 'max': 10}]
        assert len(m.detect_actions(acc, b3, min_dur_s=0.6, hz=10)) == 1

    def test_bandas_negativas(self):
        acc = [0.0] * 1000
        for k in range(10):
            acc[500 + k] = -5.0
        d3 = [{'min': -10, 'max': -4}]
        assert len(m.detect_actions(acc, d3, min_dur_s=0.6, hz=10)) == 1

    def test_sem_bandas_retorna_vazio(self):
        assert m.detect_actions([3.0] * 100, [], hz=10) == []

    def test_sinal_vazio(self):
        assert m.detect_actions([], BANDAS_ACC, hz=10) == []


# ── efforts por caixa Gen2 ──────────────────────────────────────────────────
class TestCountEffortsByBox:
    def test_reproduz_linha_do_export_modelo(self):
        effs = ([{'band': 6}] * 49 + [{'band': 7}] * 14 + [{'band': 8}] * 7
                + [{'band': 3}] * 53 + [{'band': 2}] * 16 + [{'band': 1}] * 8)
        c = m.count_efforts_by_box(effs)
        assert (c[6], c[7], c[8]) == (49, 14, 7)
        assert (c[3], c[2], c[1]) == (53, 16, 8)

    def test_band_invalida_ignorada(self):
        c = m.count_efforts_by_box([{'band': None}, {'band': 'x'}, {}, {'band': 5}])
        assert sum(c.values()) == 0

    def test_band_float_arredonda(self):
        assert m.count_efforts_by_box([{'band': 6.0}])[6] == 1


# ── rolling sum ─────────────────────────────────────────────────────────────
class TestRollingSum:
    def test_pico_correto(self):
        sv = [0.0] * 100
        sv[10] = 1.0
        sv[12] = 1.0
        roll = m.rolling_sum(sv, 5)
        assert max(roll) == 2.0
        assert len(roll) == 96

    def test_serie_curta(self):
        assert m.rolling_sum([1, 2], 5) == []

    def test_equivale_a_soma_ingenua(self):
        rng = np.random.default_rng(7)
        sv = rng.random(500).tolist()
        n = 60
        roll = m.rolling_sum(sv, n)
        ingenuo = [sum(sv[i:i + n]) for i in range(len(sv) - n + 1)]
        assert np.allclose(roll, ingenuo)

    def test_nan_tratado_como_zero(self):
        roll = m.rolling_sum([1.0, float('nan'), 1.0], 3)
        assert roll == [2.0]
