# -*- coding: utf-8 -*-
"""Testes da camada de dados: retry/backoff do cliente HTTP (P7)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("streamlit")
import requests  # noqa: E402
import catapult_api as capi  # noqa: E402

_NOSLEEP = lambda *_a, **_k: None  # noqa: E731


class FakeResp:
    def __init__(self, status, body=None, headers=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._body


def test_retry_429_respeita_retry_after(monkeypatch):
    seq = [FakeResp(429, headers={'Retry-After': '0'}), FakeResp(200, {'ok': True})]
    calls = []
    monkeypatch.setattr(requests, 'get', lambda url, **kw: (calls.append(url), seq.pop(0))[1])
    r = capi._http('get', 'http://x/y', sleep=_NOSLEEP)
    assert r.status_code == 200 and r.json() == {'ok': True}
    assert len(calls) == 2                      # 1 tentativa + 1 retry


def test_retry_500_esgota_retorna_ultimo(monkeypatch):
    monkeypatch.setattr(requests, 'get', lambda url, **kw: FakeResp(500))
    r = capi._http('get', 'http://x', max_retries=2, sleep=_NOSLEEP)
    assert r.status_code == 500                 # devolve o último Response


def test_erro_de_rede_depois_ok(monkeypatch):
    seq = [requests.ConnectionError('boom'), FakeResp(200, {'ok': 1})]
    def fake(url, **kw):
        v = seq.pop(0)
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr(requests, 'get', fake)
    r = capi._http('get', 'http://x', sleep=_NOSLEEP)
    assert r.status_code == 200


def test_erro_de_rede_esgota_relanca(monkeypatch):
    monkeypatch.setattr(requests, 'get', lambda url, **kw: (_ for _ in ()).throw(
        requests.ConnectionError('down')))
    with pytest.raises(requests.ConnectionError):
        capi._http('get', 'http://x', max_retries=1, sleep=_NOSLEEP)


def test_200_nao_faz_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, 'get', lambda url, **kw: (calls.append(1), FakeResp(200, {'a': 1}))[1])
    capi._http('get', 'http://x', sleep=_NOSLEEP)
    assert len(calls) == 1


def test_post_com_json(monkeypatch):
    got = {}
    def fake_post(url, **kw):
        got.update(kw)
        return FakeResp(200, {'done': 1})
    monkeypatch.setattr(requests, 'post', fake_post)
    r = capi._http('post', 'http://x/stats', json={'p': [1]}, sleep=_NOSLEEP)
    assert r.status_code == 200 and got.get('json') == {'p': [1]}


# ── Timeout não deve ser retentado 3x (travava a barra lateral) ──────────────
def test_timeout_retenta_no_maximo_uma_vez(monkeypatch):
    """Um GET que estoura timeout custava 60s x4 = ~4min por chamada. Agora:
    1 nova tentativa no máximo."""
    import requests as _rq
    import catapult_api as _ca

    _chamadas = {'n': 0}

    def _boom(url, **kw):
        _chamadas['n'] += 1
        raise _rq.exceptions.ReadTimeout('read timed out')

    monkeypatch.setattr(_ca.requests, 'get', _boom)
    _esperas = []
    try:
        _ca._http('get', 'http://x', max_retries=3,
                  sleep=lambda s: _esperas.append(s))
        assert False, "deveria relancar o timeout"
    except _rq.exceptions.ReadTimeout:
        pass
    assert _chamadas['n'] == 2          # 1 tentativa + 1 retry (antes: 4)
    assert len(_esperas) == 1


def test_erro_de_rede_nao_timeout_mantem_retry_completo(monkeypatch):
    """ConnectionError (conexão recusada/DNS) é barato de repetir → 3 retries."""
    import requests as _rq
    import catapult_api as _ca

    _chamadas = {'n': 0}

    def _boom(url, **kw):
        _chamadas['n'] += 1
        raise _rq.exceptions.ConnectionError('recusada')

    monkeypatch.setattr(_ca.requests, 'get', _boom)
    try:
        _ca._http('get', 'http://x', max_retries=3, sleep=lambda s: None)
        assert False, "deveria relancar"
    except _rq.exceptions.ConnectionError:
        pass
    assert _chamadas['n'] == 4          # 1 + 3 retries


def test_get_cacheado_usa_timeout_curto(monkeypatch):
    """O GET da API não deve mais usar 60s (era o multiplicador do travamento)."""
    import catapult_api as _ca
    _visto = {}

    def _fake_http(method, url, **kw):
        _visto['timeout'] = kw.get('timeout')

        class _R:
            status_code = 200

            def json(self):
                return {'ok': True}
        return _R()

    monkeypatch.setattr(_ca, '_http', _fake_http)
    _ca._api_fetch.clear()
    _ca._api_fetch('http://b', 'tok', 'teams')
    assert _visto['timeout'] <= 30
