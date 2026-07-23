# -*- coding: utf-8 -*-
"""Helpers de bandas de velocidade/aceleração (P4 — extraído do monólito).

Cortes ativos (sessão do usuário ou padrão), rótulos e formatação. Streamlit-
aware (lê as zonas da sessão); constantes vêm do config.
"""
from __future__ import annotations

import streamlit as st

from config import (
    BANDAS_VEL, BANDAS_ACC, _ACC_BAND_MAP,
    _NOMES_BANDA_VEL_DEFAULT, _CORES_BANDA_VEL_DEFAULT,
)


# ── Helpers para nomes/cores padrão por índice de banda ──────────────────────


def _bandas_vel_ativas() -> dict:
    """Retorna BANDAS_VEL usando zonas da conta Catapult (session_state) ou
    o dict hardcoded como fallback.

    A API retorna velocidades em m/s — converte para km/h multiplicando por 3.6.
    """
    try:
        zones = st.session_state.get('velocity_zones_account')
    except Exception:
        return BANDAS_VEL
    if not zones:
        return BANDAS_VEL
    result = {}
    for i, z in enumerate(zones, start=1):
        min_kmh = round(float(z.get('min_ms', 0)) * 3.6, 1)
        max_raw = float(z.get('max_ms', 9999))
        max_kmh = round(max_raw * 3.6, 1) if max_raw < 9000 else 9999
        nome    = z.get('name') or _NOMES_BANDA_VEL_DEFAULT.get(i, f'B{i}')
        if max_kmh >= 9999:
            label = f"B{i} — > {min_kmh} km/h ({nome})"
        else:
            label = f"B{i} — {min_kmh}-{max_kmh} km/h ({nome})"
        color = (z.get('color') or _CORES_BANDA_VEL_DEFAULT.get(i, '#888888'))
        result[i] = {'label': label, 'min': min_kmh, 'max': max_kmh, 'color': color}
    return result if result else BANDAS_VEL


def _legenda_vel_js() -> str:
    """Gera a expressão JS (innerHTML) da legenda de velocidade a partir das
    bandas ativas (_bandas_vel_ativas), para os campos interativo/fixo.

    Retorna uma string JS do tipo: "'<b>Velocidade</b><br>'+'<span ...'+..."
    de modo que a legenda no mapa SEMPRE reflita os valores reais das bandas.
    """
    import re as _re

    def _fmt(v):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return str(v)
        return str(int(fv)) if fv == int(fv) else f"{fv:g}"

    bandas = _bandas_vel_ativas()
    itens = list(bandas.items())
    n = len(itens)
    partes = ["'<b>Velocidade</b><br>'"]
    for idx, (_k, b) in enumerate(itens):
        cor = b.get('color', '#888888')
        mn, mx = b.get('min', 0), b.get('max', 9999)
        _m = _re.search(r'\(([^)]*)\)', b.get('label', '') or '')
        nome = _m.group(1) if _m else ''
        if idx == 0:
            txt = f"&lt;{_fmt(mx)} km/h {nome}"
        elif idx == n - 1 or float(mx) >= 9000:
            txt = f"&gt;{_fmt(mn)} km/h {nome}"
        else:
            txt = f"{_fmt(mn)}-{_fmt(mx)} km/h {nome}"
        br = '' if idx == n - 1 else '<br>'
        partes.append(
            f"'<span style=\"color:{cor}\">■</span> {txt}{br}'"
        )
    return "+".join(partes)


def _legenda_vel_items() -> list:
    """Retorna a legenda de velocidade como lista de dicts {'color','text'},
    a partir das bandas ativas (_bandas_vel_ativas). Usada pelo componente
    bidirecional do mapa (_campo_component) para renderizar a legenda real.
    """
    import re as _re

    def _fmt(v):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return str(v)
        return str(int(fv)) if fv == int(fv) else f"{fv:g}"

    bandas = _bandas_vel_ativas()
    itens = list(bandas.items())
    n = len(itens)
    out = []
    for idx, (_k, b) in enumerate(itens):
        cor = b.get('color', '#888888')
        mn, mx = b.get('min', 0), b.get('max', 9999)
        _m = _re.search(r'\(([^)]*)\)', b.get('label', '') or '')
        nome = _m.group(1) if _m else ''
        if idx == 0:
            txt = f"<{_fmt(mx)} km/h {nome}".strip()
        elif idx == n - 1 or float(mx) >= 9000:
            txt = f">{_fmt(mn)} km/h {nome}".strip()
        else:
            txt = f"{_fmt(mn)}-{_fmt(mx)} km/h {nome}".strip()
        out.append({'color': cor, 'text': txt})
    return out


def _fmt_num_banda(v) -> str:
    """Formata número de banda removendo .0 (7.0→7, 14.4→14.4)."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(fv)) if fv == int(fv) else f"{fv:g}"


def _rotulo_banda_vel(band_raw) -> str:
    """Mapeia o NÚMERO da banda de velocidade vindo da API Catapult (campo
    'band', 1–8) para um rótulo legível com a faixa configurada pelo usuário,
    ex.: '2 — 7-14.4 km/h (Trote)'. Mantém o número da API (fonte oficial).
    """
    import re as _re
    s = str(band_raw).strip()
    if not s:
        return s
    try:
        n = int(float(s))
    except (TypeError, ValueError):
        return s
    bc = _bandas_vel_ativas().get(n)
    if not bc:
        return f"Banda {n}"
    mn, mx = bc.get('min', 0), bc.get('max', 9999)
    _m = _re.search(r'\(([^)]*)\)', bc.get('label', '') or '')
    nome = _m.group(1) if _m else ''
    faixa = (f">{_fmt_num_banda(mn)} km/h" if float(mx) >= 9000
             else f"{_fmt_num_banda(mn)}-{_fmt_num_banda(mx)} km/h")
    return f"{n} — {faixa}" + (f" ({nome})" if nome else "")


# API de aceleração: número da caixa Gen2Acceleration (1..8) → chave interna.
#   Aceleração   → caixas 6,7,8 = A1,A2,A3
#   Desaceleração → caixas 3,2,1 = D1,D2,D3
# As caixas 4 e 5 (zona leve/neutra) não têm chave (não são exibidas).
# Mapa inverso (chave interna → número da caixa), usado no fallback local.
_ACC_KEY_TO_NUM = {v: k for k, v in _ACC_BAND_MAP.items()}


def _rotulo_banda_acc(band_raw) -> str:
    """Mapeia o NÚMERO da banda de aceleração da API Catapult (campo 'band',
    -3 a 3) para um rótulo legível, ex.: '2 — Acc +2 — 1-2 m/s²'."""
    s = str(band_raw).strip()
    if not s:
        return s
    try:
        n = int(float(s))
    except (TypeError, ValueError):
        return s
    bc = _bandas_acc_ativas().get(_ACC_BAND_MAP.get(n))
    if not bc:
        return f"Banda {n}"
    return f"{n} — {bc.get('label', '')}"


def _bandas_acc_ativas() -> dict:
    """Retorna BANDAS_ACC usando zonas da conta Catapult (session_state) ou
    o dict hardcoded como fallback.
    """
    try:
        zones = st.session_state.get('acceleration_zones_account')
    except Exception:
        return BANDAS_ACC
    if not zones:
        return BANDAS_ACC
    def _fa(v):
        """Formata limite de aceleração (m/s²), tratando ±infinito."""
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return str(v)
        if fv <= -9000:
            return '-∞'
        if fv >= 9000:
            return '∞'
        return str(int(fv)) if fv == int(fv) else f"{fv:g}"

    result = {}
    # ACELERAÇÃO (positivas): B1 = mais leve (menor) … B3 = máxima (maior).
    #   ordena por min crescente → A1, A2, A3.
    # DESACELERAÇÃO (negativas): B1 = mais leve (perto de zero) … B3 = máxima.
    #   ordena por max decrescente (mais perto de zero primeiro) → D1, D2, D3.
    pos_z = sorted([z for z in zones if z.get('min_ms2', 0) >= 0],
                   key=lambda z: z['min_ms2'])
    neg_z = sorted([z for z in zones if z.get('max_ms2', 0) <= 0],
                   key=lambda z: z['max_ms2'], reverse=True)
    for i, z in enumerate(pos_z, start=1):
        _mn, _mx = z['min_ms2'], z['max_ms2']
        _nome = (z.get('name') or '').strip() or f'Aceleração B{i}'
        result[f'A{i}'] = {
            'label': f"{_nome} — {_fa(_mn)} a {_fa(_mx)} m/s²",
            'min':   _mn, 'max': _mx,
            'color': z.get('color', '#69F0AE'),
        }
    for i, z in enumerate(neg_z, start=1):
        _mn, _mx = z['min_ms2'], z['max_ms2']
        _nome = (z.get('name') or '').strip() or f'Desaceleração B{i}'
        result[f'D{i}'] = {
            'label': f"{_nome} — {_fa(_mn)} a {_fa(_mx)} m/s²",
            'min':   _mn, 'max': _mx,
            'color': z.get('color', '#FF6D00'),
        }
    return result if result else BANDAS_ACC


import applog as _applog  # (P4)
from config import _DEFAULT_VELOCITY_ZONES, _DEFAULT_ACCELERATION_ZONES  # (P4)


# (P4) parsers de zonas da API -> movidos do monólito
def _parse_api_velocity_zones(api_response):
    """Converte resposta da API /velocity_zones para lista de dicts padronizados."""
    if not api_response:
        return _DEFAULT_VELOCITY_ZONES[:]
    try:
        zones_raw = (api_response if isinstance(api_response, list)
                     else api_response.get('data', api_response.get('velocity_zones', [])))
        if not zones_raw:
            return _DEFAULT_VELOCITY_ZONES[:]
        result = []
        for z in zones_raw:
            # Aceita vários formatos de chave da API Catapult
            min_val = float(z.get('min_velocity',
                            z.get('lower_threshold',
                            z.get('min', 0))) or 0)
            max_val = float(z.get('max_velocity',
                            z.get('upper_threshold',
                            z.get('max', 9999))) or 9999)
            result.append({
                'name':   z.get('name', z.get('label', '')),
                'min_ms': min_val,
                'max_ms': max_val,
                'color':  z.get('color', '#888888'),
            })
        if not result:
            return _DEFAULT_VELOCITY_ZONES[:]

        # ── Auto-detecção de unidade: m/s ou km/h ─────────────────────────
        # Se qualquer valor finito de max for > 20 → API retornou km/h.
        # Sprint típico ≤ 35 km/h = 9.7 m/s; limiar 20 distingue com segurança.
        _finite_maxes = [z['max_ms'] for z in result if z['max_ms'] < 9000]
        if _finite_maxes and max(_finite_maxes) > 20:
            # Converte km/h → m/s para padronizar com _DEFAULT_VELOCITY_ZONES
            for z in result:
                z['min_ms'] = z['min_ms'] / 3.6
                if z['max_ms'] < 9000:
                    z['max_ms'] = z['max_ms'] / 3.6

        return result
    except Exception:
        return _DEFAULT_VELOCITY_ZONES[:]

def _parse_api_acceleration_zones(api_response):
    """Converte resposta da API /acceleration_zones para lista de dicts padronizados."""
    if not api_response:
        return _DEFAULT_ACCELERATION_ZONES[:]
    try:
        zones_raw = (api_response if isinstance(api_response, list)
                     else api_response.get('data', []))
        if not zones_raw:
            return _DEFAULT_ACCELERATION_ZONES[:]
        result = []
        for z in zones_raw:
            min_val = float(z.get('min_acceleration', z.get('min', 0)))
            max_val = float(z.get('max_acceleration', z.get('max', 9999)))
            result.append({
                'name':    z.get('name', ''),
                'min_ms2': min_val,
                'max_ms2': max_val,
                'color':   z.get('color', '#888888'),
            })
        return result if result else _DEFAULT_ACCELERATION_ZONES[:]
    except Exception:
        return _DEFAULT_ACCELERATION_ZONES[:]

def _resp_tem_zonas(resp) -> bool:
    """True se a resposta da API realmente traz uma lista de zonas (não vazia).
    Evita aceitar como 'da conta' uma resposta vazia que o parser converteria
    nos valores padrão."""
    if isinstance(resp, list):
        return len(resp) > 0
    if isinstance(resp, dict):
        for _k in ('data', 'velocity_zones', 'acceleration_zones', 'zones', 'items'):
            _v = resp.get(_k)
            if isinstance(_v, list) and _v:
                return True
    return False

def _zonas_conta_via_api(api, team_ids):
    """Busca as bandas configuradas NA CONTA via API (fonte primária).

    Tenta o nível de conta e, como alternativa, o nível de equipe (onde a
    Catapult costuma expor as 'Bandas Globais'). Retorna (vel_zones, acc_zones);
    cada um é None quando a API não expõe os cortes (cai-se então na derivação
    por efforts). Nunca lança exceção."""
    vel = acc = None
    try:
        _rv = api.get_velocity_zones()
        if _resp_tem_zonas(_rv):
            _z = _parse_api_velocity_zones(_rv)
            if _z and len(_z) >= 2:
                vel = _z
    except Exception:
        _applog.log_debug_exc()
    try:
        _ra = api.get_acceleration_zones()
        if _resp_tem_zonas(_ra):
            _z = _parse_api_acceleration_zones(_ra)
            if _z and len(_z) >= 2:
                acc = _z
    except Exception:
        _applog.log_debug_exc()
    for _tid in (team_ids or []):
        if vel and acc:
            break
        if vel is None:
            try:
                _rv = api.get_team_velocity_zones(_tid)
                if _resp_tem_zonas(_rv):
                    _z = _parse_api_velocity_zones(_rv)
                    if _z and len(_z) >= 2:
                        vel = _z
            except Exception:
                _applog.log_debug_exc()
        if acc is None:
            try:
                _ra = api.get_team_acceleration_zones(_tid)
                if _resp_tem_zonas(_ra):
                    _z = _parse_api_acceleration_zones(_ra)
                    if _z and len(_z) >= 2:
                        acc = _z
            except Exception:
                _applog.log_debug_exc()
    return vel, acc
