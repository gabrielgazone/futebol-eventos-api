# -*- coding: utf-8 -*-
"""Logging estruturado (P3) — parar de engolir erros em silêncio.

Um app usado em produção não pode ter falhas invisíveis. Este módulo configura
um logger que escreve em stderr (capturado pelos logs do Streamlit Cloud) e
oferece helpers para registrar exceções COM traceback. As funções são puras
(sem Streamlit), para uso em qualquer módulo e em testes.

Uso típico:
    import applog as _log
    try:
        ...
    except Exception:
        _log.log_exc("Carga: sensor do atleta X")   # loga com traceback
"""
from __future__ import annotations

import logging
import os
import sys

_LEVEL = os.environ.get("PITCHPULSE_LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("pitchpulse")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] pitchpulse: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(getattr(logging, _LEVEL, logging.INFO))
    logger.propagate = False


def log_exc(context: str = "") -> None:
    """Registra a exceção ATUAL (dentro de um except) com traceback completo.
    Nível ERROR — use para falhas que importam."""
    logger.exception(context or "exceção")


def log_debug_exc(context: str = "") -> None:
    """Registra a exceção ATUAL em nível DEBUG (com traceback), para fallbacks
    'esperados' que antes eram `except: pass`. Não aparece em produção (nível
    INFO), mas fica disponível com PITCHPULSE_LOG_LEVEL=DEBUG — nada mais é
    engolido em silêncio."""
    logger.debug(context or "fallback", exc_info=True)


def log_error(msg: str) -> None:
    logger.error(msg)


def log_warn(msg: str) -> None:
    logger.warning(msg)


def log_info(msg: str) -> None:
    logger.info(msg)
