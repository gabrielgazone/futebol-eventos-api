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


def test_log_exc_registra_traceback(caplog):
    with caplog.at_level(logging.ERROR, logger="pitchpulse"):
        try:
            raise ValueError("boom")
        except Exception:
            applog.log_exc("contexto de teste")
    assert any("contexto de teste" in r.message for r in caplog.records)
    assert any(r.exc_info for r in caplog.records), "deve incluir traceback"
