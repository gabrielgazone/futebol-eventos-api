# -*- coding: utf-8 -*-
"""Testes do _build_period_maps (viz/janelas): mapa de tempo dos períodos usado
para localizar o segmento de GPS de um esforço nos modos Individual/Por Posição."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.janelas import _build_period_maps, _periodo_label_de_t  # noqa: E402


def _periodo(ts_ini, ts_fim, atleta='atl1'):
    """Um período com 1 atleta e 2 amostras de sensor (início/fim)."""
    return {atleta: [{'ts': ts_ini, 'cs': 0}, {'ts': ts_fim, 'cs': 0}]}


def test_periodo_unico():
    ds = {'1T': _periodo(1000, 1600)}          # 600 s = 10 min
    m = _build_period_maps(ds, False, '1T', {})
    assert m['order'] == ['1T']
    assert m['abs']['1T'] == (1000.0, 1600.0)
    assert m['start_min']['1T'] == 0.0


def test_dois_periodos_sequenciais():
    ds = {'1T': _periodo(1000, 1600),          # 0–10 min
          '2T': _periodo(2000, 2600)}          # começa após o 1T terminar
    m = _build_period_maps(ds, True, None, {})
    assert m['start_min']['1T'] == 0.0
    assert abs(m['start_min']['2T'] - 10.0) < 1e-6   # encadeado por duração


def test_subperiodo_sobreposto_substituto():
    # 2T principal (2000–2600); 2T1 (substituto) entra no meio (2300–2600)
    ds = {'2T': _periodo(2000, 2600),
          '2T1': _periodo(2300, 2600)}
    m = _build_period_maps(ds, True, None, {})
    assert m['start_min']['2T'] == 0.0
    # sub-período = início do pai + (2300-2000)/60 = 0 + 5 min
    assert abs(m['start_min']['2T1'] - 5.0) < 1e-6


def test_atl_offset_por_periodo_do_atleta():
    ds = {'1T': _periodo(1000, 1600, 'atlA'),
          '2T': _periodo(2000, 2600, 'atlB')}
    m = _build_period_maps(ds, True, None, {})
    assert m['atl_offset']('atlA') == 0.0            # joga no 1T
    assert abs(m['atl_offset']('atlB') - 10.0) < 1e-6  # entra só no 2T
    assert m['atl_offset']('ninguem') == 0.0         # sem dados → 0


def test_periodo_label_de_t():
    ds = {'1T': _periodo(1000, 1600),          # 0–10 min
          '2T': _periodo(2000, 2600)}          # 10–20 min
    m = _build_period_maps(ds, True, None, {})
    assert _periodo_label_de_t(m, 3.0) == '1T'
    assert _periodo_label_de_t(m, 15.0) == '2T'
