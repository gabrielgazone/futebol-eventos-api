# -*- coding: utf-8 -*-
"""
metrics.py — Motor único de métricas (fonte canônica de cálculo).

Todas as abas do app (Janelas Temporais, WCS, Neuromuscular, Exportação) e os
testes automatizados consomem estas funções PURAS (sem Streamlit). O objetivo é
que a mesma métrica seja calculada num único lugar — o mesmo número em qualquer
tela — e que cada fórmula seja coberta por testes (tests/test_metrics.py).

Unidades padronizadas:
- velocidade  : km/h (exceto onde indicado)
- aceleração  : m/s²
- distância   : m
- timestamps  : segundos (Unix ou relativos)
"""
from __future__ import annotations

import numpy as np

# Frequência nominal do sensor Catapult (amostras/s).
SENSOR_HZ = 10.0

# Caixas Gen2Acceleration (campo 'band' dos acceleration_efforts, 1..8):
#   Aceleração   → caixas 6, 7, 8  = A1, A2, A3 (leve → máxima)
#   Desaceleração → caixas 3, 2, 1 = D1, D2, D3 (leve → máxima)
GEN2_ACC_BOXES = {'A1': 6, 'A2': 7, 'A3': 8, 'D1': 3, 'D2': 2, 'D3': 1}


# ══════════════════════════════════════════════════════════════════════════
# Frequência de amostragem
# ══════════════════════════════════════════════════════════════════════════

def estimate_hz(series_ts, default: float = 10.0) -> float:
    """Estima a frequência de amostragem (Hz) como nº de amostras ÷ duração.

    `series_ts` é uma lista de séries de timestamps (uma por atleta/período);
    o resultado é a mediana entre as séries válidas (>20 amostras, >1 s).

    Contagem/duração é robusto a timestamps arredondados para segundos
    inteiros — caso em que a mediana das diferenças detectaria 1 Hz
    erroneamente e a integração de distância superestimaria ~N×.
    """
    ests = []
    for ts in (series_ts or []):
        if ts is None or len(ts) <= 20:
            continue
        try:
            span = float(ts[-1]) - float(ts[0])
        except (TypeError, ValueError):
            continue
        if span > 1.0:
            ests.append((len(ts) - 1) / span)
    if not ests:
        return float(default)
    hz = float(np.median(ests))
    return round(hz, 1) if hz > 0 else float(default)


# ══════════════════════════════════════════════════════════════════════════
# Distância (integração da velocidade)
# ══════════════════════════════════════════════════════════════════════════

def per_sample_distance(vel_kmh, hz: float):
    """Distância (m) percorrida em CADA amostra: v/(3,6·Hz).

    Integração retangular — a mesma usada pelo WCS e pelas Janelas Temporais.
    Valores None/inválidos contam 0.
    """
    hz = float(hz) if hz else SENSOR_HZ
    out = []
    for v in (vel_kmh or []):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        out.append(fv / (3.6 * hz) if np.isfinite(fv) else 0.0)
    return out


def per_sample_distance_in_bands(vel_kmh, faixas, hz: float):
    """Distância (m) por amostra APENAS quando a velocidade cai numa das
    `faixas` [(lo, hi) em km/h); fora delas a amostra vale 0.

    É o valor-por-amostra canônico da métrica '🏃 Velocidade (bandas)'
    (Janelas Temporais e WCS).
    """
    hz = float(hz) if hz else SENSOR_HZ
    fx = [(float(lo), float(hi)) for lo, hi in (faixas or [])]
    out = []
    for v in (vel_kmh or []):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        if not np.isfinite(fv):
            fv = 0.0
        dentro = any(lo <= fv < hi for lo, hi in fx)
        out.append(fv / (3.6 * hz) if dentro else 0.0)
    return out


def dist_by_velocity_bands(vel_kmh, faixas, hz: float):
    """Distância total (m) acumulada em cada faixa de velocidade.

    Retorna uma lista com len(faixas) valores (primeira faixa que contém a
    velocidade recebe a distância da amostra). Usado pela Exportação para
    Artigo (Velocity Band 1..6 Total Distance).
    """
    hz = float(hz) if hz else SENSOR_HZ
    fx = [(float(lo), float(hi)) for lo, hi in (faixas or [])]
    dists = [0.0] * len(fx)
    for v in (vel_kmh or []):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(fv):
            continue
        seg = fv / (3.6 * hz)
        for i, (lo, hi) in enumerate(fx):
            if lo <= fv < hi:
                dists[i] += seg
                break
    return dists


# ══════════════════════════════════════════════════════════════════════════
# Aceleração
# ══════════════════════════════════════════════════════════════════════════

def derive_acc_from_vel(vel_kmh, ts_list, hz: float = 10.0):
    """Deriva a aceleração (m/s²) de uma série de velocidade (km/h) + ts.

    dv/dt com passo mediano de segurança, suavizado por média móvel de 3 e
    saturado em ±10 m/s². Usado apenas como FALLBACK quando o dispositivo não
    tem aceleração nativa ('a' do sensor).
    """
    n = len(vel_kmh or [])
    if n < 2:
        return [0.0] * n
    vms = [float(v) / 3.6 for v in vel_kmh]           # km/h → m/s
    dts = []
    for i in range(1, min(len(ts_list or []), n)):
        d = float(ts_list[i]) - float(ts_list[i - 1])
        dts.append(d if (d and 0 < d < 2) else None)
    valid = [d for d in dts if d]
    dt_med = (float(np.median(valid)) if valid
              else (1.0 / hz if hz else 0.1))
    acc = [0.0] * n
    for i in range(1, n):
        dt = (dts[i - 1] if (i - 1 < len(dts) and dts[i - 1]) else dt_med)
        if dt and dt > 0:
            acc[i] = (vms[i] - vms[i - 1]) / dt
    sm = []
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n, i + 2)
        mv = sum(acc[lo:hi]) / (hi - lo)
        sm.append(max(-10.0, min(10.0, mv)))
    return sm


def detect_actions(acc_arr, bands, min_dur_s: float = 0.6, hz: float = 10.0):
    """Detecta AÇÕES discretas de acel/desacel no sinal de aceleração (m/s²).

    Uma ação = entrada sustentada por ≥ `min_dur_s` numa zona de threshold,
    classificada pelo PICO numa das `bands` selecionadas (dicts com min/max).
    Cada ação conta UMA vez (equivalente ao conceito de 'effort' da Catapult).

    Retorna a lista ordenada de índices (frame de início de cada ação).
    """
    if acc_arr is None or len(acc_arr) == 0 or not bands:
        return []
    min_frames = max(1, int(round(float(min_dur_s) * float(hz))))

    faixas_pos = [(float(b.get('min', 0)), float(b.get('max', 0)))
                  for b in bands if float(b.get('min', 0)) >= 0]
    faixas_neg = [(float(b.get('min', 0)), float(b.get('max', 0)))
                  for b in bands if float(b.get('max', 0)) <= 0]

    a = np.asarray(acc_arr, dtype=float)
    n = len(a)
    starts = []

    def _scan(thr, positivo, faixas):
        if not faixas:
            return
        top_hi = max(hi for _, hi in faixas)   # banda extrema (satura no topo)
        run = 0
        start_i = -1
        peak = 0.0
        counted = False
        for i in range(n):
            v = a[i]
            cond = (v >= thr) if positivo else (v <= -thr)
            if cond:
                if run == 0:
                    start_i = i
                    peak = v
                run += 1
                if (v > peak) if positivo else (v < peak):
                    peak = v
                if run >= min_frames and not counted:
                    ok = any(lo <= peak < hi for lo, hi in faixas)
                    if not ok and positivo and peak >= top_hi:
                        ok = True
                    if ok:
                        starts.append(start_i)
                    counted = True
            else:
                run = 0
                counted = False
                peak = 0.0

    if faixas_pos:
        _scan(min(lo for lo, _ in faixas_pos), True, faixas_pos)
    if faixas_neg:
        _scan(min(abs(hi) for _, hi in faixas_neg), False, faixas_neg)
    return sorted(starts)


def count_efforts_by_box(efforts) -> dict:
    """Conta acceleration_efforts da API por caixa Gen2 (campo 'band', 1..8).

    A caixa é a classificação OFICIAL do esforço — contá-la é robusto (não
    depende do valor médio 'acceleration' cair no intervalo derivado).
    """
    cont = {b: 0 for b in (1, 2, 3, 6, 7, 8)}
    for ef in (efforts or []):
        try:
            b = int(round(float(ef.get('band'))))
        except (TypeError, ValueError):
            continue
        if b in cont:
            cont[b] += 1
    return cont


# ══════════════════════════════════════════════════════════════════════════
# Janela rolante
# ══════════════════════════════════════════════════════════════════════════

def rolling_sum(values, n: int):
    """Soma rolante de janela `n` (passo 1), via soma cumulativa (O(N)).

    Retorna lista de len(values)-n+1 somas, ou [] se a série for curta.
    É a janela rolante canônica do WCS e das Janelas Temporais.
    """
    v = np.asarray(list(values or []), dtype=float)
    n = int(n)
    if n <= 0 or v.size < n:
        return []
    cs = np.cumsum(np.insert(np.nan_to_num(v), 0, 0.0))
    out = cs[n:] - cs[:-n]
    return [float(x) for x in out]
