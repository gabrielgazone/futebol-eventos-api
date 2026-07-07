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

# Versão do esquema deste módulo. O app confere este número no import e força
# importlib.reload quando o Streamlit Cloud mantém em cache uma versão antiga
# (sys.modules sobrevive ao hot-reload do script principal). Incrementar a
# cada mudança na superfície pública; o teste de sincronia no CI garante que
# o valor esperado no app acompanhe.
SCHEMA_VERSION = 3

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


# ══════════════════════════════════════════════════════════════════════════
# Monitoramento longitudinal (P10) — ACWR, monotonia e strain
# ══════════════════════════════════════════════════════════════════════════

def ewma(values, span: int):
    """Média móvel exponencial com alpha = 2/(span+1) (Williams et al., 2017)."""
    alpha = 2.0 / (float(span) + 1.0)
    out = []
    prev = None
    for v in (values or []):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        prev = fv if prev is None else alpha * fv + (1.0 - alpha) * prev
        out.append(prev)
    return out


def acwr_ewma(daily_loads, acute_span: int = 7, chronic_span: int = 28):
    """ACWR diário (método EWMA): agudo (7 d) ÷ crônico (28 d).

    Retorna uma lista alinhada às cargas diárias; None onde o crônico ≈ 0.
    """
    ac = ewma(daily_loads, acute_span)
    ch = ewma(daily_loads, chronic_span)
    return [(a / c if c > 1e-9 else None) for a, c in zip(ac, ch)]


def acwr_semanal(weekly_loads, n_chronic: int = 4):
    """ACWR semanal acoplado: carga da semana ÷ média das n semanas anteriores.

    Primeira(s) semana(s) sem histórico → None.
    """
    out = []
    loads = [float(w or 0) for w in (weekly_loads or [])]
    for i, w in enumerate(loads):
        prev = loads[max(0, i - n_chronic):i]
        out.append((w / (sum(prev) / len(prev)))
                   if prev and sum(prev) > 0 else None)
    return out


def monotonia_strain(daily_loads):
    """Monotonia e strain de Foster para UMA semana de cargas diárias.

    monotonia = média diária ÷ DP diário; strain = carga total × monotonia.
    Retorna (None, None) com <2 dias ou DP ≈ 0 (monotonia indefinida).
    """
    arr = np.asarray([float(v or 0) for v in (daily_loads or [])], dtype=float)
    if arr.size < 2:
        return None, None
    sd = float(np.std(arr, ddof=1))
    if sd < 1e-9:
        return None, None
    mono = float(np.mean(arr)) / sd
    return mono, float(arr.sum()) * mono


# ══════════════════════════════════════════════════════════════════════════
# Validação vs. OpenField — PlayerLoad oficial, ações por caixa e calibração
# ══════════════════════════════════════════════════════════════════════════

def playerload_total(pl_series):
    """PlayerLoad total a partir do parâmetro 'pl' do sensor (mesma fonte do
    OpenField). Detecta série ACUMULADA (≥98% não-decrescente) → último −
    primeiro; caso contrário soma os incrementos positivos. 0.0 sem dados."""
    vals = []
    for v in (pl_series if pl_series is not None else []):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            vals.append(fv)
    arr = np.asarray(vals, dtype=float)
    if arr.size < 10:
        return 0.0
    difs = np.diff(arr)
    if arr[-1] > arr[0] and float((difs >= -1e-6).mean()) > 0.98:
        return float(arr[-1] - arr[0])          # acumulada
    return float(np.clip(arr, 0.0, None).sum())  # incremental


def count_actions_by_box(acc_arr, boxes, min_dur_s: float = 0.6, hz: float = 10.0):
    """Conta AÇÕES sustentadas no sinal de aceleração e classifica cada uma
    pela caixa Gen2 cujo intervalo [lo, hi) contém o PICO da ação (caixa
    extrema aberta no topo/fundo). `boxes` = {nº_caixa: (lo, hi)} misturando
    aceleração (lo ≥ 0) e desaceleração (hi ≤ 0).

    Fallback do export quando a conta não retorna acceleration_efforts.
    Retorna {nº_caixa: contagem}."""
    out = {b: 0 for b in (boxes or {})}
    if acc_arr is None or len(acc_arr) == 0 or not boxes:
        return out
    min_frames = max(1, int(round(float(min_dur_s) * float(hz))))
    a = np.asarray(acc_arr, dtype=float)
    n = len(a)

    def _scan(pos):
        sel = {b: (float(lo), float(hi)) for b, (lo, hi) in boxes.items()
               if (lo >= 0 if pos else hi <= 0)}
        if not sel:
            return
        if pos:
            thr = min(lo for lo, _ in sel.values())
            ext = max(sel, key=lambda b: sel[b][1])
        else:
            thr = min(abs(hi) for _, hi in sel.values())
            ext = min(sel, key=lambda b: sel[b][0])

        def _fechar(run, peak):
            if run < min_frames:
                return
            box = None
            for b, (lo, hi) in sel.items():
                if lo <= peak < hi:
                    box = b
                    break
            if box is None:
                lo_e, hi_e = sel[ext]
                if (pos and peak >= hi_e) or ((not pos) and peak < lo_e):
                    box = ext
            if box is not None:
                out[box] += 1

        run = 0
        peak = 0.0
        for i in range(n):
            v = a[i]
            if (v >= thr) if pos else (v <= -thr):
                if run == 0:
                    peak = v
                run += 1
                if (v > peak) if pos else (v < peak):
                    peak = v
            else:
                _fechar(run, peak)
                run = 0
                peak = 0.0
        _fechar(run, peak)

    _scan(True)
    _scan(False)
    return out


def calibrate_velocity_cutoffs(sessions, hz: float = 10.0, n_bands: int = 6,
                               vmax_kmh: float = 45.0):
    """Calibra os cortes das bandas de velocidade para REPRODUZIR o export
    oficial do OpenField.

    `sessions` = lista de (serie_vel_kmh, dist_bandas_oficiais[n_bands]) por
    atleta. Para cada corte k, resolve por bisseção o limiar c tal que a
    distância agregada acima de c (integrada do sinal) iguale a distância
    oficial acumulada nas bandas > k. Retorna n_bands-1 cortes (km/h),
    monotônicos."""
    svs, csums, targets_of = [], [], []
    for vel, bands in (sessions or []):
        v = np.asarray([float(x) for x in vel if x is not None], dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 10:
            continue
        v.sort()
        svs.append(v)
        csums.append(np.concatenate([[0.0], np.cumsum(v)]))
        targets_of.append([float(x or 0) for x in bands])
    if not svs:
        return []

    def _above(c):
        tot = 0.0
        for s, cs in zip(svs, csums):
            i = int(np.searchsorted(s, c, side='left'))
            tot += float(cs[-1] - cs[i]) / (3.6 * hz)
        return tot

    cuts = []
    for k in range(1, n_bands):
        target = sum(sum(b[k:]) for b in targets_of)
        lo, hi = 0.0, float(vmax_kmh)
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if _above(mid) > target:
                lo = mid
            else:
                hi = mid
        cuts.append(round((lo + hi) / 2.0, 2))
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = round(cuts[i - 1] + 0.1, 2)
    return cuts


def classificar_acwr(acwr):
    """Zona de risco do ACWR (Gabbett, 2016): <0,8 subcarga · 0,8–1,3 ideal ·
    1,3–1,5 atenção · >1,5 alto risco. '—' quando indisponível."""
    if acwr is None:
        return '—'
    try:
        a = float(acwr)
    except (TypeError, ValueError):
        return '—'
    if not np.isfinite(a):
        return '—'
    if a < 0.8:
        return 'subcarga'
    if a <= 1.3:
        return 'ideal'
    if a <= 1.5:
        return 'atenção'
    return 'alto risco'
