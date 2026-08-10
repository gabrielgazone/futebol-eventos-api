# -*- coding: utf-8 -*-
"""Testes da resolução de elenco/posição/equipe via API no export WCS multi."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.export_wcs_multi import (  # noqa: E402
    _mapa_posicoes, _mapa_equipes, _atletas_da_atividade,
    _linhas_wcs_atividade)
import wcs_export as wx  # noqa: E402


class FakeApi:
    def get_positions(self):
        return [{'id': 'p1', 'name': 'Zagueiro'},
                {'id': 'p2', 'name': 'Atacante'},
                {'id': None, 'name': 'Ignorado'}]

    def get_teams(self):
        return [{'id': 't1', 'name': 'Vasco'}, {'id': 't2', 'name': 'Palmeiras'}]

    def get_team_athletes(self, tid):
        return ([{'id': 'a1'}, {'id': 'a2'}] if tid == 't1'
                else [{'id': 'a3'}])

    def get_activity_athletes(self, aid):
        return [
            {'id': 'a1', 'first_name': 'João', 'last_name': 'Silva',
             'position_id': 'p1'},
            {'id': 'a2', 'first_name': 'Ana', 'last_name': 'Souza',
             'position_id': 'p2'},
            {'id': 'a3', 'name': 'Carlos Dias', 'position_id': 'inexistente'},
            {'id': 'a9'},                      # sem nome → descartado
        ]


def test_mapa_posicoes_da_api():
    m = _mapa_posicoes(FakeApi())
    assert m == {'p1': 'Zagueiro', 'p2': 'Atacante'}   # id None ignorado


def test_mapa_equipes_da_api():
    m = _mapa_equipes(FakeApi())
    assert m == {'a1': 'Vasco', 'a2': 'Vasco', 'a3': 'Palmeiras'}


def test_atletas_da_atividade_resolve_posicao_e_equipe():
    api = FakeApi()
    df = _atletas_da_atividade(api, 'act1', _mapa_posicoes(api),
                               _mapa_equipes(api))
    assert list(df.columns) == ['id', 'nome', 'posicao', 'equipe']
    assert len(df) == 3                        # 'a9' sem nome foi descartado
    r = df.set_index('nome')
    assert r.loc['João Silva', 'posicao'] == 'Zagueiro'
    assert r.loc['João Silva', 'equipe'] == 'Vasco'
    assert r.loc['Ana Souza', 'posicao'] == 'Atacante'
    # position_id inexistente → string vazia (não quebra)
    assert r.loc['Carlos Dias', 'posicao'] == ''
    assert r.loc['Carlos Dias', 'equipe'] == 'Palmeiras'


def test_atletas_da_atividade_api_falha():
    class Boom:
        def get_activity_athletes(self, aid):
            raise RuntimeError('500')
    assert _atletas_da_atividade(Boom(), 'x', {}, {}).empty


def test_linhas_wcs_usa_posicao_da_api():
    # 1 min a 36 km/h (10 m/s) → 600 m
    pts = [{'v': 10.0} for _ in range(600)]
    dados_sensor = {'1T': {'João Silva': pts}}
    info = {'João Silva': ('Zagueiro', 'Vasco')}
    rows = _linhas_wcs_atividade(
        'Jogo A', '2026-03-01', dados_sensor, info,
        [wx.VAR_DIST], [1], ['Partida inteira'], 10.0, {})
    assert len(rows) == 1
    assert rows[0]['Posicao'] == 'Zagueiro'
    assert rows[0]['Equipe'] == 'Vasco'
    assert rows[0]['Atividade'] == 'Jogo A'
    assert rows[0]['Janela_min'] == 1
    assert abs(rows[0]['Valor'] - 600.0) < 1.0


def test_linhas_wcs_atleta_sem_info_nao_quebra():
    pts = [{'v': 10.0} for _ in range(600)]
    rows = _linhas_wcs_atividade(
        'Jogo B', '2026-03-02', {'1T': {'Desconhecido': pts}}, {},
        [wx.VAR_DIST], [1], ['Partida inteira'], 10.0, {})
    assert len(rows) == 1
    assert rows[0]['Posicao'] == '' and rows[0]['Equipe'] == ''
