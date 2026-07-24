# -*- coding: utf-8 -*-
"""Testes do logging estruturado (P3)."""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import applog  # noqa: E402


def test_logger_configurado():
    assert applog.logger.name == "pitchpulse"
    assert applog.logger.handlers, "logger deve ter handler"


def test_helpers_nao_lancam():
    applog.log_info("info de teste")
    applog.log_warn("warn de teste")
    applog.log_error("error de teste")


def test_redact_remove_jwt():
    tok = ("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9."
           "eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV")
    out = applog.redact(f"erro de rede: {tok} (fim)")
    assert tok not in out            # propriedade de segurança: token removido
    assert "***" in out and "JWT-REDACTED" in out
    # e sem a palavra-gatilho 'token' por perto, o marcador do JWT sobrevive:
    assert applog.redact(tok) == "***JWT-REDACTED***"


def test_redact_bearer_e_apikey():
    assert "abcdef123456789" not in applog.redact("Authorization: Bearer abcdef123456789")
    assert "supersecretkey99" not in applog.redact('apikey="supersecretkey99"')


def test_mask_token():
    assert applog.mask_token("eyJ0eXAiOiXXXXXXXXXXlongtoken1234ABCD") == "eyJ0eX…ABCD"
    assert applog.mask_token("curto") == "***"
    assert applog.mask_token(None) == "***"


def test_log_funcs_redigem(caplog):
    import logging as _lg
    tok = "eyJhbGciOiJIUzI1NiJ9.eyJhIjoxfQ.ZZZZZZ123456"
    with caplog.at_level(_lg.WARNING, logger="pitchpulse"):
        applog.log_warn(f"vazou {tok}")
    assert all(tok not in r.message for r in caplog.records)


def test_log_exc_registra_traceback(caplog):
    with caplog.at_level(logging.ERROR, logger="pitchpulse"):
        try:
            raise ValueError("boom")
        except Exception:
            applog.log_exc("contexto de teste")
    assert any("contexto de teste" in r.message for r in caplog.records)
    assert any(r.exc_info for r in caplog.records), "deve incluir traceback"
