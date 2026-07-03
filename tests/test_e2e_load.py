# -*- coding: utf-8 -*-
"""Teste E2E do fluxo 'carregar atividade' com a API Catapult simulada.

Executa o app inteiro no harness oficial do Streamlit (AppTest), com um
mock de `requests` que simula a API Connect v6: conexão → seleção de
períodos → busca de atletas → loop de carregamento → renderização de TODAS
as abas, com um atleta de x/y nativo e um só-GPS. Qualquer exceção em
qualquer aba falha o teste — é a rede de segurança do caminho mais crítico
do app.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import requests  # noqa: E402

BASE_TS = 1_750_000_000
_APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "futebol-eventos.py")


def _points(n=1200, gps_only=False):
    pts = []
    for i in range(n):
        t = i / 10.0
        v = 4.0 + 3.5 * math.sin(t / 9.0)          # m/s (~2–27 km/h)
        p = {
            'ts': BASE_TS + int(t), 'cs': int((t % 1) * 100),
            'v': round(v, 3), 'a': round(1.2 * math.cos(t / 5.0), 3),
            'hr': 150 + int(10 * math.sin(t / 30.0)),
            'pl': 0.05, 'mp': 8.0,
            'lat': -22.9 + 0.0001 * math.sin(t / 7.0),
            'long': -43.2 + 0.0001 * math.cos(t / 7.0),
            'pq': 90, 'hdop': 0.8, 'ref': 12, 'o': 1000 + i * 0.5,
        }
        if not gps_only:
            p['x'] = 50 + 30 * math.sin(t / 7.0)
            p['y'] = 34 + 20 * math.cos(t / 7.0)
        else:
            p['x'] = None
            p['y'] = None
        pts.append(p)
    return pts


_EFFORTS = [{'data': {
    'acceleration_efforts': [
        {'band': 6, 'acceleration': 2.5, 'start_time': BASE_TS + 10,
         'end_time': BASE_TS + 11, 'distance': 5.0},
        {'band': 3, 'acceleration': -2.6, 'start_time': BASE_TS + 30,
         'end_time': BASE_TS + 31, 'distance': 4.0},
    ],
    'velocity_efforts': [
        {'band': 5, 'start_time': BASE_TS + 20, 'end_time': BASE_TS + 23,
         'max_velocity': 7.2, 'start_velocity': 3.0, 'distance': 18.0},
    ],
    'heart_rate_efforts': [], 'jump_efforts': [], 'step_balance_efforts': [],
}}]

_ATHS = [
    {'id': 'ath1', 'first_name': 'Ana', 'last_name': 'Silva',
     'jersey': '9', 'position_id': 'pos1'},
    {'id': 'ath2', 'first_name': 'Bia', 'last_name': 'Souza',
     'jersey': '10', 'position_id': 'pos1'},
]


def _route(path):
    if path.endswith('/teams'):
        return [{'id': 'team1', 'name': 'Time A', 'slug': 'ta'}]
    if '/teams/team1/athletes' in path:
        return [{'id': 'ath1'}, {'id': 'ath2'}]
    if path.endswith('/positions'):
        return [{'id': 'pos1', 'name': 'Atacante', 'slug': 'atk'}]
    if path.endswith('/athletes') and '/activities/' not in path \
            and '/periods/' not in path and '/teams/' not in path:
        return _ATHS
    if path.endswith('/activities'):
        return [{'id': 'act1', 'name': 'JOGO TESTE', 'start_time': BASE_TS,
                 'venue': {'name': 'CT', 'lat': -22.9, 'lng': -43.2,
                           'rotation': 0, 'length': 105, 'width': 68}}]
    if '/activities/act1/periods' in path:
        return [{'id': 'per1', 'name': '1 Tempo', 'start_time': BASE_TS,
                 'end_time': BASE_TS + 130}]
    if ('/periods/per1/athletes' in path or '/activities/act1/athletes' in path) \
            and '/sensor' not in path and '/efforts' not in path \
            and '/events' not in path:
        return _ATHS
    if '/athletes/ath1/sensor' in path:
        return [{'data': _points(gps_only=False)}]
    if '/athletes/ath2/sensor' in path:
        return [{'data': _points(gps_only=True)}]
    if '/efforts' in path:
        return _EFFORTS
    if '/events' in path:
        return []
    return None


class _Resp:
    def __init__(self, data):
        self._d = data
        self.status_code = 200 if data is not None else 404

    def json(self):
        return self._d


@pytest.fixture()
def api_mock(monkeypatch):
    monkeypatch.setattr(requests, 'get',
                        lambda url, *a, **k: _Resp(_route(url.split('?')[0])))
    monkeypatch.setattr(requests, 'post', lambda url, *a, **k: _Resp([]))


def _sem_excecoes(at, etapa):
    msgs = [f"{e.value}\n{''.join(str(s) for s in (e.stack_trace or []))[-800:]}"
            for e in at.exception]
    assert not msgs, f"exceções em '{etapa}':\n" + "\n---\n".join(msgs)


@pytest.mark.e2e
def test_fluxo_completo_carregar_atividade(api_mock):
    at = AppTest.from_file(_APP, default_timeout=600)
    at.run()
    _sem_excecoes(at, 'inicial')

    tok = [w for w in at.sidebar.text_input if 'Token' in (w.label or '')]
    assert tok, "campo de token não encontrado"
    tok[0].input('FAKE_TOKEN_123')
    at.run()
    _sem_excecoes(at, 'token')

    btn = [b for b in at.sidebar.button if 'Carregar Dados' in (b.label or '')]
    assert btn, "botão Carregar Dados não encontrado"
    btn[0].click()
    at.run()
    _sem_excecoes(at, 'conexão')

    ms_per = [m for m in at.sidebar.multiselect if 'per' in (m.label or '').lower()]
    assert ms_per, "multiselect de períodos não encontrado"
    ms_per[0].set_value(['1 Tempo'])
    at.run()
    _sem_excecoes(at, 'períodos')

    btn2 = [b for b in at.sidebar.button if 'Buscar Atletas' in (b.label or '')]
    assert btn2, "botão Buscar Atletas não encontrado"
    btn2[0].click()
    at.run()
    _sem_excecoes(at, 'carga completa (todas as abas)')

    assert at.session_state['atletas_sel'] == ['Ana Silva', 'Bia Souza']
    assert len(at.tabs) >= 10, "abas não renderizaram após a carga"
    assert len(at.dataframe) >= 3, "tabelas não renderizaram após a carga"
