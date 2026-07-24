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
import re
import sys

_LEVEL = os.environ.get("PITCHPULSE_LOG_LEVEL", "INFO").upper()

# ── (P5) Redação de segredos ────────────────────────────────────────────────
# Defesa em profundidade: qualquer mensagem que passe pelo logger tem tokens
# JWT / cabeçalhos Bearer / chaves removidos, para que NENHUM log (nem no Cloud)
# vaze credenciais — mesmo que algum código passe um token por engano.
_JWT_RE = re.compile(r'eyJ[A-Za-z0-9_=-]{8,}\.[A-Za-z0-9_=-]{6,}\.[A-Za-z0-9_=-]{6,}')
_BEARER_RE = re.compile(r'(?i)\b(bearer|apikey|authorization|token|key)\b(["\'\s:=]+)\S{8,}')


def redact(text: object) -> str:
    """Remove JWTs e segredos (Bearer/apikey/token/key=...) de um texto."""
    s = str(text)
    s = _JWT_RE.sub('***JWT-REDACTED***', s)
    s = _BEARER_RE.sub(r'\1\2***', s)
    return s


def mask_token(tok: object) -> str:
    """Máscara para EXIBIR um token (6 primeiros + 4 últimos)."""
    t = str(tok or '')
    return f"{t[:6]}…{t[-4:]}" if len(t) > 14 else '***'

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
    logger.exception(redact(context or "exceção"))


def log_debug_exc(context: str = "") -> None:
    """Registra a exceção ATUAL em nível DEBUG (com traceback), para fallbacks
    'esperados' que antes eram `except: pass`. Não aparece em produção (nível
    INFO), mas fica disponível com PITCHPULSE_LOG_LEVEL=DEBUG — nada mais é
    engolido em silêncio."""
    logger.debug(context or "fallback", exc_info=True)


def log_error(msg: str) -> None:
    logger.error(redact(msg))


def log_warn(msg: str) -> None:
    logger.warning(redact(msg))


def log_info(msg: str) -> None:
    logger.info(redact(msg))
