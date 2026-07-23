# -*- coding: utf-8 -*-
"""Persistência de configuração por organização (P4 — extraído do monólito).

Helpers app-facing sobre storage.py: escolha do backend (_get_store), chave por
organização (_org_key), banco de venues e bandas do usuário. Streamlit-aware
(secrets/sessão) + storage + applog.
"""
from __future__ import annotations

import os as _os
from datetime import datetime

import streamlit as st

import storage as _storage
import applog as _applog


# ==================== BANCO COMPARTILHADO DE VENUES ====================
# (P7) O arquivo de venues é escopado POR ORGANIZAÇÃO (marcador derivado do
# menor team_id da conta, estável entre tokens de usuários do mesmo clube).
# Clubes diferentes no mesmo servidor NÃO leem nem sobrescrevem os campos
# uns dos outros; o compartilhamento continua funcionando dentro do clube.
# Estrutura: { "Nome do Venue": { lat, lon, rot, fl, fw, ig, saved_at } }

def _get_store():
    """(P2) Retorna o armazenamento durável (Supabase) se configurado em
    st.secrets['supabase'] = {url, key, [table]}; senão, o local (efêmero) e
    marca um aviso na sessão. Cacheado por sessão."""
    _cached = st.session_state.get('_kv_store')
    if _cached is not None:
        return _cached
    store = None
    try:
        _sb = None
        try:
            _sb = st.secrets.get('supabase')
        except Exception:
            _sb = None
        if _sb and _sb.get('url') and _sb.get('key'):
            store = _storage.SupabaseStore(_sb['url'], _sb['key'],
                                           _sb.get('table', 'app_kv'))
    except Exception:
        store = None
    if store is None:
        store = _storage.LocalJSONStore(_os.path.dirname(_os.path.abspath(__file__)))
        st.session_state['_persist_efemera'] = True
    st.session_state['_kv_store'] = store
    return store


def _org_key(prefix: str) -> str:
    """Chave de armazenamento com escopo por organização (menor team_id)."""
    _mk = str(st.session_state.get('_org_marker', '') or '').strip() or 'default'
    return f"{prefix}_{_mk}"


def _carregar_venues() -> dict:
    """Retorna o dicionário de venues salvos (nome → config)."""
    return _get_store().get(_org_key('venues')) or {}

def _salvar_venue(nome: str, cfg: dict) -> None:
    """Persiste (ou sobrescreve) a configuração de um venue no escopo da conta."""
    dados = _carregar_venues()
    dados[nome.strip()] = {
        'lat':      cfg.get('lat', 0.0),
        'lon':      cfg.get('lon', 0.0),
        'rot':      cfg.get('rot', 0),
        'fl':       cfg.get('fl',  105),
        'fw':       cfg.get('fw',  68),
        'ig':       cfg.get('ig',  1),
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    _get_store().set(_org_key('venues'), dados)

def _excluir_venue(nome: str) -> None:
    """Remove um venue do banco da conta."""
    dados = _carregar_venues()
    dados.pop(nome, None)
    _get_store().set(_org_key('venues'), dados)


# ── Bandas DEFINIDAS PELO USUÁRIO (configuração, não calibração) ─────────────
# A Connect API v6 não expõe os cortes das "Bandas Globais". Como o usuário
# precisa reproduzir a configuração da sua conta OpenField, ele DIGITA os
# limiares (velocidade km/h e aceleração m/s²) nos editores. Isso é
# CONFIGURAÇÃO — como o próprio OpenField deixa configurar — e não fitagem ao
# resultado. Persistimos por organização para o usuário definir uma única vez.

def _salvar_bandas_usuario(vel=None, acc=None) -> None:
    """Grava as bandas digitadas pelo usuário (mescla com o que já existir)."""
    _cur = _carregar_bandas_usuario() or {}
    if vel is not None:
        _cur['velocity_zones'] = vel
    if acc is not None:
        _cur['acceleration_zones'] = acc
    _cur['saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    _get_store().set(_org_key('bandas_usuario'), _cur)


def _carregar_bandas_usuario():
    """Retorna {'velocity_zones': [...], 'acceleration_zones': [...]} ou None."""
    return _get_store().get(_org_key('bandas_usuario'))


def _excluir_bandas_usuario(qual='all') -> None:
    """Remove bandas persistidas (qual='velocity'|'acceleration'|'all')."""
    if qual == 'all':
        _get_store().delete(_org_key('bandas_usuario'))
        return
    _cur = _carregar_bandas_usuario() or {}
    _cur.pop('velocity_zones' if qual == 'velocity' else 'acceleration_zones', None)
    _get_store().set(_org_key('bandas_usuario'), _cur)


import json  # (P4) prefs


# (P4) preferências locais do usuário -> movidas do monólito
_PREFS_FILE = _os.path.join(_os.path.expanduser("~"), ".futebol_prefs.json")


def _carregar_prefs() -> dict:
    """Carrega preferências salvas do usuário (arquivo local JSON)."""
    try:
        with open(_PREFS_FILE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _salvar_prefs(prefs: dict) -> None:
    """Persiste preferências do usuário em arquivo local JSON."""
    try:
        with open(_PREFS_FILE, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)
    except Exception:
        _applog.log_debug_exc()
