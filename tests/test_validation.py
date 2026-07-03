# -*- coding: utf-8 -*-
"""Testes do módulo de validação de concordância (validation.py)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import validation as val  # noqa: E402

VARS = ['Total Distance (m)', 'Minutos', 'Metros por minuto',
        'Max Acceleration', 'Max Deceleration', 'Maximum Velocity (km/h)']


def _tabela(nomes, dist, minutos, macc=None, mdec=None, mvel=None, prefixo='JOGO X'):
    n = len(nomes)
    return pd.DataFrame({
        'Name': [f"{prefixo} - {a}" for a in nomes],
        'Date': ['20/06/2026'] * n,
        'Total Distance (m)': dist,
        'Minutos': minutos,
        'Metros por minuto': [d / mn if mn else 0 for d, mn in zip(dist, minutos)],
        'Max Acceleration': macc or [4.0] * n,
        'Max Deceleration': mdec or [-4.5] * n,
        'Maximum Velocity (km/h)': mvel or [29.0] * n,
    })


class TestAthleteKey:
    def test_extrai_ultima_parte(self):
        assert val.athlete_key('RESENDE X SERRANO A2 2026 - Vitor santos') == 'vitor santos'

    def test_nome_com_hifen_na_atividade(self):
        assert val.athlete_key('MD-1 22 DE JUNHO - Hugo Barreto') == 'hugo barreto'

    def test_case_insensitive(self):
        assert val.athlete_key('X - LUAN.F') == val.athlete_key('Y - luan.f')


class TestAggregate:
    def test_soma_totais_e_max_min(self):
        df = _tabela(['Ana', 'Ana'], [1000.0, 500.0], [45.0, 45.0],
                     macc=[3.0, 5.0], mdec=[-3.0, -6.0], mvel=[28.0, 31.0])
        g = val.aggregate_by_athlete(df, VARS)
        assert len(g) == 1
        row = g.iloc[0]
        assert row['Total Distance (m)'] == 1500.0
        assert row['Minutos'] == 90.0
        assert row['Max Acceleration'] == 5.0
        assert row['Max Deceleration'] == -6.0
        assert row['Maximum Velocity (km/h)'] == 31.0

    def test_metros_por_minuto_recalculado(self):
        df = _tabela(['Ana', 'Ana'], [900.0, 900.0], [45.0, 45.0])
        g = val.aggregate_by_athlete(df, VARS)
        assert g.iloc[0]['Metros por minuto'] == pytest.approx(1800.0 / 90.0)


class TestComparar:
    def test_tabelas_identicas_vies_zero(self):
        app = _tabela(['Ana', 'Bia', 'Carla', 'Dani'],
                      [5000.0, 6000.0, 7000.0, 8000.0], [90.0] * 4)
        merged, stats = val.comparar_exportacoes(app, app.copy(), VARS)
        assert len(merged) == 4
        row = stats[stats['Variável'] == 'Total Distance (m)'].iloc[0]
        assert row['Viés (app − oficial)'] == 0
        assert row['r'] == pytest.approx(1.0)

    def test_vies_percentual_detectado(self):
        app = _tabela(['Ana', 'Bia', 'Carla', 'Dani'],
                      [5000.0, 6000.0, 7000.0, 8000.0], [90.0] * 4)
        off = app.copy()
        off['Total Distance (m)'] = off['Total Distance (m)'] / 1.10  # app 10% acima
        merged, stats = val.comparar_exportacoes(app, off, VARS)
        row = stats[stats['Variável'] == 'Total Distance (m)'].iloc[0]
        assert row['Viés %'] == pytest.approx(10.0, abs=0.2)

    def test_atletas_sem_par_excluidos(self):
        app = _tabela(['Ana', 'Bia', 'Carla'], [5000.0, 6000.0, 7000.0], [90.0] * 3)
        off = _tabela(['Bia', 'Carla', 'Zeca'], [6000.0, 7000.0, 4000.0], [90.0] * 3,
                      prefixo='OUTRO NOME DE JOGO')
        merged, stats = val.comparar_exportacoes(app, off, VARS)
        assert sorted(merged['Atleta'].str.lower()) == ['bia', 'carla']
        assert stats.iloc[0]['n'] == 2

    def test_sem_atletas_em_comum(self):
        app = _tabela(['Ana'], [5000.0], [90.0])
        off = _tabela(['Zeca'], [4000.0], [90.0])
        merged, stats = val.comparar_exportacoes(app, off, VARS)
        assert merged.empty


class TestBlandAltman:
    def test_identicos_bias_zero(self):
        ba = val.bland_altman([100, 200, 300, 400], [100, 200, 300, 400])
        assert ba['bias'] == 0.0
        assert ba['loa_low'] == 0.0 and ba['loa_high'] == 0.0

    def test_offset_constante(self):
        ba = val.bland_altman([110, 210, 310], [100, 200, 300])
        assert ba['bias'] == pytest.approx(10.0)
        assert ba['sd'] == pytest.approx(0.0, abs=1e-9)

    def test_poucos_pares_retorna_none(self):
        assert val.bland_altman([1, 2], [1, 2]) is None

    def test_nan_filtrado(self):
        ba = val.bland_altman([100, np.nan, 300, 400], [100, 200, 300, 400])
        assert ba['n'] == 3
