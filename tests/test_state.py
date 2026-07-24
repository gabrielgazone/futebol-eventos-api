# -*- coding: utf-8 -*-
"""Testes do esquema de estado da sessão (P6)."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import state  # noqa: E402


def _fake_st(ss=None):
    return types.SimpleNamespace(session_state=ss if ss is not None else {})


def test_init_aplica_defaults():
    st = _fake_st()
    resetou = state.init(st)
    assert resetou is True                      # 1ª vez = schema ausente
    assert st.session_state[state.KEYS.MODO_APRES] is False
    assert st.session_state["_state_schema"] == state.SCHEMA_VERSION


def test_idempotente_sem_reset():
    st = _fake_st()
    state.init(st)
    st.session_state["df_teams"] = "dados"      # simula dados carregados
    resetou = state.init(st)                     # 2ª chamada, mesma versão
    assert resetou is False
    assert st.session_state["df_teams"] == "dados"   # NÃO limpou


def test_bump_de_versao_limpa_volatil():
    st = _fake_st({"_state_schema": "versao-antiga",
                   "df_teams": "x", "activity_id": "a1",
                   "atletas_sel": ["Ana"], "modo_apresentacao": True})
    resetou = state.init(st)
    assert resetou is True
    # voláteis limpos:
    for k in ("df_teams", "activity_id", "atletas_sel"):
        assert k not in st.session_state
    assert st.session_state["_state_schema"] == state.SCHEMA_VERSION


def test_nao_perde_defaults_setados_pelo_usuario():
    st = _fake_st()
    state.init(st)
    st.session_state[state.KEYS.MODO_APRES] = True   # usuário ligou
    state.init(st)                                    # rerun
    assert st.session_state[state.KEYS.MODO_APRES] is True  # setdefault preserva


def test_keys_e_volatile_consistentes():
    assert state.KEYS.DF_TEAMS in state.VOLATILE
    assert state.KEYS.API not in state.VOLATILE      # api não é limpa aqui
