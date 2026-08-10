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
    rows, _ = _linhas_wcs_atividade(
        'Jogo A', '2026-03-01', dados_sensor, info,
        [wx.VAR_DIST], [1], ['Partida inteira'], 10.0, {},
        participantes={'1T': {'João Silva'}}, duracoes={'1T': 1.0})
    assert len(rows) == 1
    assert rows[0]['Posicao'] == 'Zagueiro'
    assert rows[0]['Equipe'] == 'Vasco'
    assert rows[0]['Atividade'] == 'Jogo A'
    assert rows[0]['Janela_min'] == 1
    assert abs(rows[0]['Valor'] - 600.0) < 1.0


def test_linhas_wcs_atleta_sem_info_nao_quebra():
    pts = [{'v': 10.0} for _ in range(600)]
    rows, _ = _linhas_wcs_atividade(
        'Jogo B', '2026-03-02', {'1T': {'Desconhecido': pts}}, {},
        [wx.VAR_DIST], [1], ['Partida inteira'], 10.0, {},
        participantes={'1T': {'Desconhecido'}}, duracoes={'1T': 1.0})
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




def test_linhas_incluem_minutos_do_openfield():
    pts = [{'v': 10.0} for _ in range(600)]     # 600 amostras a 10 Hz = 1 min
    rows, _ = _linhas_wcs_atividade(
        'Jogo A', '25/07/2026', {'1T': {'João Silva': pts}},
        {'João Silva': ('Zagueiro', 'Vasco')},
        [wx.VAR_DIST], [1], ['Partida inteira'], 10.0, {},
        participantes={'1T': {'João Silva'}}, duracoes={'1T': 51.4})
    assert rows[0]['Minutos'] == 51.4           # duração do período (OpenField)
    assert 'Minutos_dispositivo' not in rows[0]  # removida (minutos vêm da API)
    assert rows[0]['Periodos_jogados'] == 1
    assert rows[0]['Data'] == '25/07/2026'


# ── Pivô: variáveis em colunas ───────────────────────────────────────────────
def test_pivotar_variaveis_colunas_e_linhas():
    import pandas as pd
    from viz.export_wcs_multi import pivotar_variaveis
    base = {'Atividade': 'J1', 'Data': '25/07/2026', 'Equipe': 'Vasco',
            'Posicao': 'Meia', 'Minutos': 90.0,
            'Periodos_jogados': 2, 'Escopo': 'Partida inteira', 'Janela_min': 1}
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
    assert linha_a['Minutos'] == 90.0
    # identificadores vêm antes das variáveis
    assert list(p.columns).index('Atleta') < list(p.columns).index(wx.VAR_DIST)


def test_pivotar_variaveis_df_vazio():
    import pandas as pd
    from viz.export_wcs_multi import pivotar_variaveis
    vazio = pd.DataFrame()
    assert getattr(pivotar_variaveis(vazio), 'empty', False)


# ── Descoberta do parâmetro de duração + disjuntor (bug do app travando) ─────





# ── Formato: variáveis × janelas em colunas (1 linha por atleta) ─────────────
def _df_tres_janelas():
    import pandas as pd
    base = {'Atividade': 'J1', 'Data': '25/07/2026', 'Equipe': 'Resende',
            'Posicao': 'Volante', 'Minutos': 55.7,
            'Periodos_jogados': 1, 'Escopo': 'Partida inteira'}
    _rows = []
    for _jan, _dist, _hsr in ((1, 215.1, 76.9), (3, 547.8, 112.8),
                              (5, 846.1, 136.4)):
        _rows.append(dict(base, Atleta='Enzo', Variavel=wx.VAR_DIST,
                          Janela_min=_jan, Valor=_dist))
        _rows.append(dict(base, Atleta='Enzo', Variavel=wx.VAR_HSR,
                          Janela_min=_jan, Valor=_hsr))
    return pd.DataFrame(_rows)


def test_pivot_var_x_janelas_uma_linha_por_atleta():
    from viz.export_wcs_multi import pivotar_variaveis_x_janelas
    p = pivotar_variaveis_x_janelas(_df_tres_janelas())
    assert len(p) == 1                       # 1 linha (antes: 3, uma por janela)
    assert 'Janela_min' not in p.columns
    assert 'Variavel' not in p.columns and 'Valor' not in p.columns


def test_pivot_var_x_janelas_tres_colunas_por_variavel():
    from viz.export_wcs_multi import pivotar_variaveis_x_janelas
    p = pivotar_variaveis_x_janelas(_df_tres_janelas())
    for _v in (wx.VAR_DIST, wx.VAR_HSR):
        for _j in (1, 3, 5):
            assert f"{_v} {_j}min" in p.columns
    _l = p.iloc[0]
    assert _l[f"{wx.VAR_DIST} 1min"] == 215.1
    assert _l[f"{wx.VAR_DIST} 3min"] == 547.8
    assert _l[f"{wx.VAR_DIST} 5min"] == 846.1
    assert _l[f"{wx.VAR_HSR} 5min"] == 136.4


def test_pivot_var_x_janelas_ordem_das_colunas():
    """Identificadores primeiro; depois cada variável com suas 3 janelas juntas."""
    from viz.export_wcs_multi import pivotar_variaveis_x_janelas
    _cols = list(pivotar_variaveis_x_janelas(_df_tres_janelas()).columns)
    assert _cols.index('Atleta') < _cols.index(f"{wx.VAR_DIST} 1min")
    assert _cols.index('Minutos') < _cols.index(f"{wx.VAR_DIST} 1min")
    # janelas em ordem crescente dentro da variável
    assert (_cols.index(f"{wx.VAR_DIST} 1min")
            < _cols.index(f"{wx.VAR_DIST} 3min")
            < _cols.index(f"{wx.VAR_DIST} 5min"))
    # variáveis agrupadas (todas as janelas de Distância antes de HSR)
    assert _cols.index(f"{wx.VAR_DIST} 5min") < _cols.index(f"{wx.VAR_HSR} 1min")


def test_pivot_var_x_janelas_preserva_minutos_e_escopo():
    from viz.export_wcs_multi import pivotar_variaveis_x_janelas
    _l = pivotar_variaveis_x_janelas(_df_tres_janelas()).iloc[0]
    assert _l['Minutos'] == 55.7
    assert _l['Escopo'] == 'Partida inteira'
    assert _l['Periodos_jogados'] == 1


def test_pivot_var_x_janelas_df_vazio():
    import pandas as pd
    from viz.export_wcs_multi import pivotar_variaveis_x_janelas
    assert getattr(pivotar_variaveis_x_janelas(pd.DataFrame()), 'empty', False)


# ── Contagem no feedback: nome de atividade repetido em datas diferentes ─────
def _df_dois_palmeiras():
    """Cenário real: duas partidas com o MESMO nome, datas diferentes."""
    import pandas as pd
    base = {'Equipe': 'Vasco', 'Posicao': 'Meia', 'Minutos': 90.0,
            'Periodos_jogados': 2, 'Escopo': 'Partida inteira',
            'Variavel': wx.VAR_DIST, 'Janela_min': 1, 'Valor': 200.0}
    return pd.DataFrame([
        dict(base, Atividade='Jogo x Cuiabá', Data='01/08/2026', Atleta='A'),
        dict(base, Atividade='Jogo x Palmeiras', Data='02/08/2026', Atleta='A'),
        dict(base, Atividade='Jogo x Palmeiras', Data='05/08/2026', Atleta='A'),
        dict(base, Atividade='Jogo x Botafogo', Data='08/08/2026', Atleta='A'),
    ])


def test_contar_atividades_desambigua_por_data():
    from viz.export_wcs_multi import contar_unicos
    df = _df_dois_palmeiras()
    assert contar_unicos(df, 'Atividade') == 4      # antes dizia 3
    assert df['Atividade'].nunique() == 3           # o defeito antigo


def test_contar_atletas_desambigua_por_equipe():
    import pandas as pd
    from viz.export_wcs_multi import contar_unicos
    df = pd.DataFrame([
        {'Atleta': 'João Silva', 'Equipe': 'Vasco'},
        {'Atleta': 'João Silva', 'Equipe': 'Palmeiras'},   # homônimo
        {'Atleta': 'Ana Souza', 'Equipe': 'Vasco'},
    ])
    assert contar_unicos(df, 'Atleta') == 3


def test_contar_unicos_sem_coluna_ou_vazio():
    import pandas as pd
    from viz.export_wcs_multi import contar_unicos
    assert contar_unicos(pd.DataFrame(), 'Atividade') == 0
    assert contar_unicos(None, 'Atividade') == 0
    # sem a coluna acompanhante, cai no nunique simples
    assert contar_unicos(pd.DataFrame({'Atividade': ['A', 'A', 'B']}),
                         'Atividade') == 2


def test_chave_de_linha_inclui_data():
    """Sem 'Data' na chave, o modo Acumular apagaria um dos jogos homonimos."""
    from viz.export_wcs_multi import _CHAVE_LINHA
    assert 'Data' in _CHAVE_LINHA
    df = _df_dois_palmeiras()
    assert len(df.drop_duplicates(subset=_CHAVE_LINHA)) == 4   # nada se perde
    _sem_data = [_c for _c in _CHAVE_LINHA if _c != 'Data']
    assert len(df.drop_duplicates(subset=_sem_data)) == 3      # perderia 1 jogo
