# -*- coding: utf-8 -*-
"""Testes da pré-busca paralela do carregador (P9)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loader as dl  # noqa: E402


class FakeApi:
    def __init__(self):
        self.calls = []

    def get_sensor_data(self, act, aid):
        self.calls.append(('sensor', act, aid))

    def get_activity_efforts(self, act, aid, et):
        self.calls.append(('eff', act, aid))

    def get_period_sensor_data(self, pid, aid):
        self.calls.append(('psensor', pid, aid))

    def get_period_efforts(self, pid, aid, et):
        self.calls.append(('peff', pid, aid))


def test_prefetch_atividade_completa():
    api = FakeApi()
    n = dl.prefetch_sensores(api, 'A1', {}, ['Atividade Completa'], ['id1', 'id2'])
    assert n == 2
    sensores = {c[2] for c in api.calls if c[0] == 'sensor'}
    assert sensores == {'id1', 'id2'}
    assert sum(1 for c in api.calls if c[0] == 'eff') == 2


def test_prefetch_por_periodo():
    api = FakeApi()
    n = dl.prefetch_sensores(api, 'A1', {'1T': 'p1'}, ['1T'], ['id1'])
    assert n == 1
    assert ('psensor', 'p1', 'id1') in api.calls
    assert ('peff', 'p1', 'id1') in api.calls


def test_prefetch_multiplos_periodos_x_atletas():
    api = FakeApi()
    n = dl.prefetch_sensores(api, 'A', {'1T': 'p1', '2T': 'p2'}, ['1T', '2T'],
                             ['a', 'b', 'c'])
    assert n == 6                       # 2 períodos × 3 atletas


def test_prefetch_vazio():
    api = FakeApi()
    assert dl.prefetch_sensores(api, 'A1', {}, [], []) == 0
    assert api.calls == []


def test_prefetch_ignora_erros():
    class BoomApi:
        def get_sensor_data(self, *a):
            raise RuntimeError('down')

        def get_activity_efforts(self, *a):
            raise RuntimeError('down')

    n = dl.prefetch_sensores(BoomApi(), 'A', {}, ['Comp'], ['i1'])
    assert n == 1                       # não levanta; erro é logado e ignorado
