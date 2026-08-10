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


# ── Data (bug do epoch cru) ──────────────────────────────────────────────────
def test_fmt_data_br_epoch_unix():
    from viz.export_wcs_multi import _fmt_data_br
    # 1785005112 = 25/07/2026 (era exibido cru na coluna Data)
    assert _fmt_data_br(1785005112) == '25/07/2026'
    assert _fmt_data_br('1785005112') == '25/07/2026'


def test_fmt_data_br_iso_e_vazio():
    from viz.export_wcs_multi import _fmt_data_br
    assert _fmt_data_br('2026-07-25T10:00:00Z') == '25/07/2026'
    assert _fmt_data_br('2026-07-25') == '25/07/2026'
    assert _fmt_data_br(None) == '' and _fmt_data_br('') == ''


# ── Minutos oficiais (/stats) ────────────────────────────────────────────────
def test_para_minutos_segundos_e_minutos():
    from viz.export_wcs_multi import _para_minutos
    assert _para_minutos(5400) == 90.0        # segundos → minutos
    assert _para_minutos(90) == 90.0          # já em minutos
    assert _para_minutos(0) is None and _para_minutos(None) is None


def test_minutos_openfield_encontra_parametro():
    from viz.export_wcs_multi import _minutos_openfield

    class StatsApi:
        def __init__(self):
            self.pedidos = []

        def get_stats(self, payload):
            self.pedidos.append(payload['parameters'][0])
            if payload['parameters'][0] != 'total_duration':
                return None                    # só este parâmetro existe
            return [{'athlete': 'João Silva', 'parameters': {'total_duration': 5400}},
                    {'athlete': 'Ana Souza', 'parameters': {'total_duration': 2700}}]

    api = StatsApi()
    mins, par = _minutos_openfield(api, 1785005112, 1785091512)
    assert par == 'total_duration'
    assert mins == {'João Silva': 90.0, 'Ana Souza': 45.0}


def test_minutos_openfield_sem_resposta():
    from viz.export_wcs_multi import _minutos_openfield

    class Vazio:
        def get_stats(self, payload):
            return None

    assert _minutos_openfield(Vazio(), 1785005112, None) == ({}, None)
    # sem epoch → nem tenta
    assert _minutos_openfield(Vazio(), None, None) == ({}, None)


def test_linhas_incluem_minutos_of_e_sensor():
    pts = [{'v': 10.0} for _ in range(600)]     # 600 amostras a 10 Hz = 1 min
    rows = _linhas_wcs_atividade(
        'Jogo A', '25/07/2026', {'1T': {'João Silva': pts}},
        {'João Silva': ('Zagueiro', 'Vasco')},
        [wx.VAR_DIST], [1], ['Partida inteira'], 10.0, {},
        min_map={'João Silva': 90.0})
    assert rows[0]['Minutos_OpenField'] == 90.0
    assert rows[0]['Minutos_sensor'] == 1.0     # derivado das amostras
    assert rows[0]['Data'] == '25/07/2026'


# ── Pivô: variáveis em colunas ───────────────────────────────────────────────
def test_pivotar_variaveis_colunas_e_linhas():
    import pandas as pd
    from viz.export_wcs_multi import pivotar_variaveis
    base = {'Atividade': 'J1', 'Data': '25/07/2026', 'Equipe': 'Vasco',
            'Posicao': 'Meia', 'Minutos_OpenField': 90.0,
            'Minutos_sensor': 89.5, 'Escopo': 'Partida inteira', 'Janela_min': 1}
    df = pd.DataFrame([
        dict(base, Atleta='A', Variavel=wx.VAR_DIST, Valor=180.0),
        dict(base, Atleta='A', Variavel=wx.VAR_HSR, Valor=40.0),
        dict(base, Atleta='B', Variavel=wx.VAR_DIST, Valor=170.0),
        dict(base, Atleta='B', Variavel=wx.VAR_HSR, Valor=35.0),
    ])
    p = pivotar_variaveis(df)
    assert len(p) == 2                          # 1 linha por ATLETA
    assert wx.VAR_DIST in p.columns and wx.VAR_HSR in p.columns
    assert 'Variavel' not in p.columns and 'Valor' not in p.columns
    linha_a = p[p['Atleta'] == 'A'].iloc[0]
    assert linha_a[wx.VAR_DIST] == 180.0 and linha_a[wx.VAR_HSR] == 40.0
    assert linha_a['Minutos_OpenField'] == 90.0
    # identificadores vêm antes das variáveis
    assert list(p.columns).index('Atleta') < list(p.columns).index(wx.VAR_DIST)


def test_pivotar_variaveis_df_vazio():
    import pandas as pd
    from viz.export_wcs_multi import pivotar_variaveis
    vazio = pd.DataFrame()
    assert getattr(pivotar_variaveis(vazio), 'empty', False)
