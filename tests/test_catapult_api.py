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
