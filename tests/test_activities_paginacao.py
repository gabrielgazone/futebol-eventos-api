# -*- coding: utf-8 -*-
"""Paginação de /activities: sem ela, contas com muitas atividades perdiam as
MAIS ANTIGAS (o app só pedia uma página de 500)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import catapult_api as ca  # noqa: E402


def _api():
    return ca.CatapultAPI('http://b', 'tok')


def _fake_paginado(paginas):
    """Fake de _api_fetch que devolve `paginas[page-1]`."""
    def _f(base, tok, path, params=()):
        _p = dict(params)
        _pg = int(_p.get('page', 1))
        return paginas[_pg - 1] if 1 <= _pg <= len(paginas) else []
    return _f


def test_junta_todas_as_paginas(monkeypatch):
    p1 = [{'id': f'a{i}', 'name': f'Jogo {i}'} for i in range(500)]
    p2 = [{'id': f'b{i}', 'name': f'Antigo {i}'} for i in range(120)]
    monkeypatch.setattr(ca, '_api_fetch', _fake_paginado([p1, p2]))
    r = _api().get_activities(page_size=500)
    assert len(r) == 620                      # antes parava em 500
    assert any(_a['name'].startswith('Antigo') for _a in r)   # as antigas entram


def test_para_na_pagina_incompleta(monkeypatch):
    _chamadas = {'n': 0}

    def _f(base, tok, path, params=()):
        _chamadas['n'] += 1
        return [{'id': f'a{i}'} for i in range(10)]     # < page_size

    monkeypatch.setattr(ca, '_api_fetch', _f)
    r = _api().get_activities(page_size=500)
    assert len(r) == 10
    assert _chamadas['n'] == 1                # não pede a página 2 à toa


def test_api_que_ignora_page_nao_gera_loop(monkeypatch):
    """Se a API devolver sempre a mesma página, para na 2ª (ids repetidos)."""
    _chamadas = {'n': 0}
    _fixa = [{'id': f'a{i}'} for i in range(500)]

    def _f(base, tok, path, params=()):
        _chamadas['n'] += 1
        return list(_fixa)

    monkeypatch.setattr(ca, '_api_fetch', _f)
    r = _api().get_activities(page_size=500)
    assert len(r) == 500                      # sem duplicatas
    assert _chamadas['n'] == 2                # detectou e parou


def test_respeita_max_paginas(monkeypatch):
    _cont = {'n': 0}

    def _f(base, tok, path, params=()):
        _cont['n'] += 1
        _base = _cont['n'] * 1000
        return [{'id': f'a{_base + i}'} for i in range(500)]   # sempre cheia

    monkeypatch.setattr(ca, '_api_fetch', _f)
    r = _api().get_activities(page_size=500, max_paginas=3)
    assert _cont['n'] == 3                    # teto respeitado
    assert len(r) == 1500


def test_resposta_envelopada_em_data(monkeypatch):
    def _f(base, tok, path, params=()):
        _pg = int(dict(params).get('page', 1))
        return {'data': [{'id': f'a{_pg}'}]} if _pg == 1 else {'data': []}

    monkeypatch.setattr(ca, '_api_fetch', _f)
    assert len(_api().get_activities(page_size=500)) == 1


def test_resposta_vazia_ou_nula(monkeypatch):
    monkeypatch.setattr(ca, '_api_fetch', lambda *a, **k: None)
    assert _api().get_activities() == []
