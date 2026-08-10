# -*- coding: utf-8 -*-
"""Réplica da minutagem do OpenField e filtro de participação por período.

Cenário real (RESENDE x AMERICANO, 25/07/2026): Enzo Zaidan NÃO consta no export
do 1º tempo do OpenField (não jogou), tem 55,7 min no 2º e 55,7 no jogo todo.
O app mostrava 107,1 min e linhas de 1º tempo (dispositivo ligado no banco).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wcs_export as wx  # noqa: E402
from viz.export_wcs_multi import (  # noqa: E402
    duracoes_periodos, participou_do_periodo, _linhas_wcs_atividade,
    _participantes_por_periodo)

_HZ = 10.0


def _serie(vel_kmh, n, ts0=1000.0):
    """n amostras a `vel_kmh`, com ts crescente a 10 Hz."""
    return [{'v': vel_kmh / 3.6, 'ts': ts0 + i / _HZ, 'cs': 0, 'a': 0.0,
             'pl': i * 0.01} for i in range(n)]


# ── Durações dos períodos (= "Minutos" do OpenField) ─────────────────────────
def test_duracoes_periodos_por_janela_de_tempo():
    ds = {'1 Tempo': {'A': _serie(10, 600, ts0=1000)},        # 60 s
          '2 Tempo': {'A': _serie(10, 1200, ts0=5000)}}       # 120 s
    d = duracoes_periodos(ds, _HZ)
    assert abs(d['1 Tempo'] - 1.0) < 0.01                     # 1 min
    assert abs(d['2 Tempo'] - 2.0) < 0.01                     # 2 min


def test_duracoes_periodo_sem_dados():
    assert duracoes_periodos({'1 Tempo': {}}, _HZ)['1 Tempo'] == 0.0


# ── Participação por período ────────────────────────────────────────────────
def test_participacao_lista_oficial_manda():
    ds = {'1 Tempo': {'Enzo': _serie(0.5, 600), 'Titular': _serie(20, 600)}}
    part = {'1 Tempo': {'Titular'}}                 # Enzo NÃO participou
    assert participou_do_periodo('Titular', '1 Tempo', ds, part, _HZ) is True
    assert participou_do_periodo('Enzo', '1 Tempo', ds, part, _HZ) is False


def test_participacao_piso_quando_api_nao_informa():
    # Sem lista oficial, vale o piso de m/min: reserva com dispositivo ligado
    # (~0,5 km/h = 8 m/min) fica abaixo de 25; quem joga (20 km/h) passa longe.
    ds = {'1 Tempo': {'Enzo': _serie(0.5, 600),
                      'T1': _serie(20, 600), 'T2': _serie(22, 600)}}
    part = {'1 Tempo': None}
    dur = duracoes_periodos(ds, _HZ)
    assert participou_do_periodo('T1', '1 Tempo', ds, part, _HZ, dur) is True
    assert participou_do_periodo('Enzo', '1 Tempo', ds, part, _HZ, dur) is False


def test_participacao_sem_dados_no_periodo():
    ds = {'1 Tempo': {'T1': _serie(20, 600)}}
    assert participou_do_periodo('Ausente', '1 Tempo', ds, {'1 Tempo': None},
                                 _HZ) is False


# ── Cenário Enzo: minutagem e ausência de linhas fantasma ───────────────────
def _cenario_enzo():
    """1º tempo: Enzo no banco (não participante). 2º tempo: Enzo joga."""
    ds = {
        '1 Tempo': {'Enzo': _serie(0.5, 600, ts0=1000),
                    'Titular': _serie(20, 600, ts0=1000)},
        '2 Tempo': {'Enzo': _serie(20, 1200, ts0=5000),
                    'Titular': _serie(20, 1200, ts0=5000)},
    }
    part = {'1 Tempo': {'Titular'}, '2 Tempo': {'Enzo', 'Titular'}}
    return ds, part, duracoes_periodos(ds, _HZ)


def test_enzo_sem_linhas_no_primeiro_tempo():
    ds, part, dur = _cenario_enzo()
    rows, excl = _linhas_wcs_atividade(
        'Jogo', '25/07/2026', ds, {'Enzo': ('Volante', 'Resende')},
        [wx.VAR_DIST], [1], ['Partida inteira', 'Por período'], _HZ, {},
        participantes=part, duracoes=dur)
    escopos_enzo = {r['Escopo'] for r in rows if r['Atleta'] == 'Enzo'}
    assert '1 Tempo' not in escopos_enzo      # não sugere presença no 1º tempo
    assert '2 Tempo' in escopos_enzo
    assert excl >= 1                          # par excluído é reportado


def test_enzo_minutos_replicam_openfield():
    ds, part, dur = _cenario_enzo()
    rows, _ = _linhas_wcs_atividade(
        'Jogo', '25/07/2026', ds, {}, [wx.VAR_DIST], [1],
        ['Partida inteira', 'Por período'], _HZ, {},
        participantes=part, duracoes=dur)
    _enzo = [r for r in rows if r['Atleta'] == 'Enzo']
    # Minutos do jogo inteiro = só o 2º tempo (como no OpenField)
    _pi = [r for r in _enzo if r['Escopo'] == 'Partida inteira'][0]
    assert abs(_pi['Minutos'] - dur['2 Tempo']) < 0.01
    assert _pi['Periodos_jogados'] == 1
    # Titular soma os dois períodos
    _tit = [r for r in rows if r['Atleta'] == 'Titular'
            and r['Escopo'] == 'Partida inteira'][0]
    assert abs(_tit['Minutos'] - (dur['1 Tempo'] + dur['2 Tempo'])) < 0.01
    assert _tit['Periodos_jogados'] == 2


def test_minutos_do_escopo_de_periodo_e_a_duracao_do_periodo():
    ds, part, dur = _cenario_enzo()
    rows, _ = _linhas_wcs_atividade(
        'Jogo', '25/07/2026', ds, {}, [wx.VAR_DIST], [1], ['Por período'],
        _HZ, {}, participantes=part, duracoes=dur)
    for r in rows:
        assert abs(r['Minutos'] - dur[r['Escopo']]) < 0.01


# ── PlayerLoad: 'pl' é acumulado, não pode ser somado bruto ─────────────────
def test_playerload_usa_incrementos_da_serie_acumulada():
    # série acumulada 0..5.99 (incremento 0.01/amostra) em 600 amostras
    pts = _serie(10, 600)
    r = wx.calcular_wcs(pts, [wx.VAR_PL], [1], hz=_HZ)
    # 1 min = 600 amostras x 0.01 = ~6 (e NÃO a soma dos acumulados ~1797)
    assert abs(r[(wx.VAR_PL, 1)] - 6.0) < 0.5


def test_playerload_serie_incremental_tambem_funciona():
    pts = [{'v': 2.8, 'ts': 1000 + i / _HZ, 'cs': 0, 'pl': 0.01}
           for i in range(600)]
    r = wx.calcular_wcs(pts, [wx.VAR_PL], [1], hz=_HZ)
    assert abs(r[(wx.VAR_PL, 1)] - 6.0) < 0.5


# ── Lista de participantes vinda da API ─────────────────────────────────────
def test_participantes_por_periodo_da_api():
    class Api:
        def get_athletes_in_period(self, pid):
            return ([{'id': 'a1'}, {'id': 'a2'}] if pid == 'p1'
                    else [{'id': 'a2'}])

    part = _participantes_por_periodo(
        Api(), {'1 Tempo': 'p1', '2 Tempo': 'p2'},
        {'a1': 'Titular', 'a2': 'Enzo'})
    assert part['1 Tempo'] == {'Titular', 'Enzo'}
    assert part['2 Tempo'] == {'Enzo'}


def test_participantes_periodo_api_vazia_vira_none():
    class Vazia:
        def get_athletes_in_period(self, pid):
            return []

    assert _participantes_por_periodo(Vazia(), {'1T': 'p1'}, {})['1T'] is None


# ── Atividade profunda: /activities/{id}?include=all (fonte autoritativa) ────
def _deep_resp():
    """Resposta como a doc descreve: periods com start/end_time (epoch) e os
    atletas de cada periodo. 1o tempo = 51,40233 min; 2o = 55,7 min (valores
    reais do OpenField em RESENDE x AMERICANO)."""
    _ini1 = 1785000000
    _fim1 = _ini1 + int(51.40233 * 60)
    _ini2 = _fim1 + 900                       # intervalo
    _fim2 = _ini2 + int(55.7 * 60)
    return [{
        'id': 'act1',
        'teams': [{'id': 't1', 'name': 'Resende FC'}],
        'athletes': [
            {'id': 'a1', 'first_name': 'Enzo', 'last_name': 'Zaidan',
             'position_name': 'Volante'},
            {'id': 'a2', 'first_name': 'Matheus', 'last_name': 'Goiano',
             'position_name': 'Atacante'},
        ],
        'periods': [
            {'id': 'p1', 'name': '1 Tempo', 'start_time': _ini1,
             'start_centiseconds': 0, 'end_time': _fim1, 'end_centiseconds': 14,
             'athletes': [{'id': 'a2'}]},          # SÓ Matheus no 1o tempo
            {'id': 'p2', 'name': '2 Tempo', 'start_time': _ini2,
             'start_centiseconds': 0, 'end_time': _fim2, 'end_centiseconds': 0,
             'athletes': [{'id': 'a1'}, {'id': 'a2'}]},
        ],
    }]


def test_deep_duracoes_batem_com_openfield():
    from viz.export_wcs_multi import ler_atividade_profunda
    dur, part, info = ler_atividade_profunda(_deep_resp())
    assert abs(dur['1 Tempo'] - 51.40233) < 0.01     # como no export do site
    assert abs(dur['2 Tempo'] - 55.7) < 0.01


def test_deep_participacao_exclui_enzo_do_primeiro_tempo():
    from viz.export_wcs_multi import ler_atividade_profunda
    _, part, _ = ler_atividade_profunda(_deep_resp())
    assert part['1 Tempo'] == {'Matheus Goiano'}      # Enzo ausente
    assert part['2 Tempo'] == {'Enzo Zaidan', 'Matheus Goiano'}


def test_deep_traz_posicao_e_equipe():
    from viz.export_wcs_multi import ler_atividade_profunda
    _, _, info = ler_atividade_profunda(_deep_resp())
    assert info['Enzo Zaidan'] == ('Volante', 'Resende FC')
    assert info['Matheus Goiano'][0] == 'Atacante'


def test_deep_minutos_totais_por_atleta():
    """Enzo = só 2o tempo (55,7); Matheus = os dois (107,10233)."""
    from viz.export_wcs_multi import ler_atividade_profunda
    dur, part, _ = ler_atividade_profunda(_deep_resp())
    _min_enzo = sum(d for p, d in dur.items() if 'Enzo Zaidan' in (part[p] or ()))
    _min_mat = sum(d for p, d in dur.items()
                   if 'Matheus Goiano' in (part[p] or ()))
    assert abs(_min_enzo - 55.7) < 0.01
    assert abs(_min_mat - 107.10233) < 0.02


def test_deep_resposta_vazia_ou_invalida():
    from viz.export_wcs_multi import ler_atividade_profunda
    assert ler_atividade_profunda([]) == ({}, {}, {})
    assert ler_atividade_profunda(None) == ({}, {}, {})
    assert ler_atividade_profunda({'periods': []}) == ({}, {}, {})


def test_deep_periodo_com_period_athletes_e_athlete_id():
    """Formato alternativo da doc: period_athletes com athlete_id."""
    from viz.export_wcs_multi import ler_atividade_profunda
    resp = {
        'athletes': [{'id': 'a1', 'first_name': 'Enzo', 'last_name': 'Zaidan'}],
        'periods': [{'name': '2 Tempo', 'start_time': 1000, 'end_time': 4342,
                     'period_athletes': [{'athlete_id': 'a1'}]}],
    }
    dur, part, _ = ler_atividade_profunda(resp)
    assert abs(dur['2 Tempo'] - 55.7) < 0.01
    assert part['2 Tempo'] == {'Enzo Zaidan'}
