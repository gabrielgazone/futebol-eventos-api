# -*- coding: utf-8 -*-
"""Carregador enxuto/paralelo do export WCS: só o sinal, 1 chamada por
atleta×período, respeitando a lista oficial de participantes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.export_wcs_multi import carregar_sensor_export  # noqa: E402


class SpyApi:
    """Conta as chamadas e registra QUAIS endpoints foram usados."""

    def __init__(self, vazio_para=()):
        self.sensor = []
        self.outros = 0
        self._vazio = set(vazio_para)

    def get_period_sensor_data(self, pid, aid):
        self.sensor.append((pid, aid))
        if (pid, aid) in self._vazio:
            return None
        return [{'data': [{'v': 5.0, 'ts': 1000, 'cs': 0}]}]

    def get_sensor_data(self, act, aid):
        self.sensor.append(('ACT', aid))
        return [{'data': [{'v': 5.0, 'ts': 1000, 'cs': 0}]}]

    # endpoints que o export NÃO deve chamar
    def get_period_efforts(self, *a, **k):
        self.outros += 1
        return []

    def get_athlete_thresholds(self, *a, **k):
        self.outros += 1
        return {}

    def get_athletes_in_period(self, *a, **k):
        self.outros += 1
        return []


_ATLETAS = [('Titular', 'a1'), ('Substituto', 'a2')]


def test_uma_chamada_por_atleta_periodo():
    api = SpyApi()
    r = carregar_sensor_export(api, 'act', {'1T': 'p1', '2T': 'p2'}, _ATLETAS)
    assert len(api.sensor) == 4                 # 2 períodos x 2 atletas
    assert api.outros == 0                      # nada de efforts/limiares
    assert set(r) == {'1T', '2T'}
    assert set(r['1T']) == {'Titular', 'Substituto'}


def test_respeita_lista_oficial_e_economiza_chamadas():
    api = SpyApi()
    partic = {'1T': {'Titular'}, '2T': {'Titular', 'Substituto'}}
    r = carregar_sensor_export(api, 'act', {'1T': 'p1', '2T': 'p2'}, _ATLETAS,
                               participantes=partic)
    assert len(api.sensor) == 3                 # o substituto não é baixado no 1T
    assert set(r['1T']) == {'Titular'}
    assert set(r['2T']) == {'Titular', 'Substituto'}


def test_periodo_sem_lista_baixa_todos():
    api = SpyApi()
    r = carregar_sensor_export(api, 'act', {'1T': 'p1'}, _ATLETAS,
                               participantes={'1T': None})
    assert len(api.sensor) == 2
    assert set(r['1T']) == {'Titular', 'Substituto'}


def test_atividade_completa_usa_endpoint_da_atividade():
    api = SpyApi()
    carregar_sensor_export(api, 'act', {'Atividade Completa': None}, _ATLETAS)
    assert all(_p == 'ACT' for _p, _ in api.sensor)


def test_atleta_sem_dados_nao_entra():
    api = SpyApi(vazio_para=[('p1', 'a2')])
    r = carregar_sensor_export(api, 'act', {'1T': 'p1'}, _ATLETAS)
    assert set(r['1T']) == {'Titular'}          # 'Substituto' sem sinal


def test_falha_de_um_atleta_nao_derruba_os_demais():
    class Instavel(SpyApi):
        def get_period_sensor_data(self, pid, aid):
            if aid == 'a2':
                raise RuntimeError('500')
            return super().get_period_sensor_data(pid, aid)

    r = carregar_sensor_export(Instavel(), 'act', {'1T': 'p1'}, _ATLETAS)
    assert set(r['1T']) == {'Titular'}


def test_sem_tarefas_retorna_vazio():
    api = SpyApi()
    assert carregar_sensor_export(api, 'act', {}, _ATLETAS) == {}
    assert carregar_sensor_export(api, 'act', {'1T': 'p1'}, []) == {}
    assert api.sensor == []


def test_prefere_endpoint_sem_cache_para_nao_estourar_memoria():
    """Com ~10 jogos, o cache de 15 min do _api_fetch reteria todo o sinal
    10 Hz na memória. O export deve usar a variante SEM cache quando existe."""
    class ApiComNC(SpyApi):
        def __init__(self):
            super().__init__()
            self.nc = 0

        def get_period_sensor_data_nc(self, pid, aid):
            self.nc += 1
            return [{'data': [{'v': 5.0, 'ts': 1000, 'cs': 0}]}]

    api = ApiComNC()
    carregar_sensor_export(api, 'act', {'1T': 'p1'}, _ATLETAS)
    assert api.nc == 2                 # usou a versão sem cache
    assert api.sensor == []            # NÃO usou a cacheada


def test_cai_para_endpoint_cacheado_se_nao_houver_nc():
    api = SpyApi()                      # sem os métodos _nc
    carregar_sensor_export(api, 'act', {'1T': 'p1'}, _ATLETAS)
    assert len(api.sensor) == 2         # compatível com APIs antigas
