# -*- coding: utf-8 -*-
"""Compute de análise (P4 — extraído do monólito).

Métricas por atleta (calcular_metricas), janelas temporais, esforços por sinal,
gráficos de velocidade/aceleração/intensidade. Depende de módulos já extraídos
(metrics, bands, diagnostics, config) + streamlit/pandas/numpy/plotly.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import metrics as _mtr
from bands import _bandas_vel_ativas, _bandas_acc_ativas, _fmt_num_banda
from diagnostics import _diag_log, _selo_fonte
from config import (
    BANDAS_VEL, BANDAS_ACC, _ACC_BAND_MAP, _CHAVE_COMBINADO,
    _DEFAULT_MIN_DUR_S, _DEFAULT_MIN_DUR_VEL_S,
    _DEFAULT_VELOCITY_ZONES, _DEFAULT_ACCELERATION_ZONES,
    _NOMES_BANDA_VEL_DEFAULT, _CORES_BANDA_VEL_DEFAULT,
)


def detectar_eventos_acc(acc_arr, limiar, min_dur_s=0.6, acima=True, freq_hz=10):
    """
    Retorna máscara booleana onde True = primeiro frame que completou
    a duração mínima (min_dur_s) dentro de uma zona de threshold contínua.
    Isso conta cada evento UMA vez (entrada na zona sustentada).

    acima=True → acc >= limiar (aceleração)
    acima=False → acc <= -limiar (desaceleração)
    """
    min_frames = max(1, round(min_dur_s * freq_hz))
    n = len(acc_arr)
    eventos = np.zeros(n, dtype=bool)
    run = 0
    in_event = False
    for i in range(n):
        v = acc_arr[i]
        cond = (v >= limiar) if acima else (v <= -limiar)
        if cond:
            run += 1
            if run == min_frames and not in_event:
                eventos[i] = True
                in_event = True
        else:
            run = 0
            in_event = False
    return eventos

def acc_series_from_vel(vel_kmh, ts_list, freq_hz=10.0):
    """Deriva aceleração (m/s²) da velocidade (km/h) + timestamps.

    (P1) Delegado ao motor único: metrics.derive_acc_from_vel."""
    return _mtr.derive_acc_from_vel(vel_kmh, ts_list, freq_hz)

def detectar_acoes_acc_idx(acc_arr, sel_acc_bands, min_dur_s=None, freq_hz=10):
    """
    Conta AÇÕES discretas de acel/desacel a partir da série de aceleração
    (m/s², por amostra) — usado como FALLBACK quando a API não retorna
    `acceleration_efforts`. Cada ação é uma entrada sustentada por pelo menos
    `min_dur_s` numa zona de threshold, classificada pelo pico numa das bandas
    selecionadas. Conta cada ação UMA vez (igual ao conceito de "effort").

    Retorna a lista de índices (frame de início de cada ação) — equivalente ao
    start_time dos efforts da Catapult, para ser somado na janela rolante.
    """
    # (P1) Delegado ao motor único: metrics.detect_actions.
    if min_dur_s is None:
        min_dur_s = get_min_dur_s()
    return _mtr.detect_actions(acc_arr, sel_acc_bands,
                               min_dur_s=min_dur_s, hz=freq_hz)

def get_min_dur_s():
    """Lê o slider de duração mínima de acc/dec do session_state."""
    return float(st.session_state.get('min_dur_esforco', _DEFAULT_MIN_DUR_S))

def get_min_dur_vel_s():
    """Lê o slider de duração mínima de velocidade do session_state."""
    return float(st.session_state.get('min_dur_vel', _DEFAULT_MIN_DUR_VEL_S))

def get_zones_for_athlete(athlete_name):
    """Retorna zonas de velocidade para o atleta (override > conta > defaults)."""
    overrides = st.session_state.get('velocity_zones_athlete', {})
    if athlete_name in overrides and overrides[athlete_name]:
        return overrides[athlete_name]
    account_zones = st.session_state.get('velocity_zones_account')
    if account_zones:
        return account_zones
    return _DEFAULT_VELOCITY_ZONES[:]

def calcular_metricas(sensor_points, athlete_name, min_dur_s=None, zones=None):
    """Calcula métricas de desempenho a partir dos sensor_points.

    zones: lista de dicts {'name', 'min_ms', 'max_ms', 'color'} com as zonas de velocidade
           individuais do atleta. Se None, usa limiares padrão (19 e 24 km/h).
    """
    if not sensor_points:
        return None

    if min_dur_s is None:
        min_dur_s = get_min_dur_s()

    # Determinar limiares de zona alta (HSR) e sprint a partir das zonas fornecidas
    if zones and len(zones) >= 2:
        # Zona n-2 (penúltima) = HSR, zona n-1 (última) = Sprint
        _z_hsr    = zones[-2]['min_ms'] * 3.6   # em km/h
        _z_sprint = zones[-1]['min_ms'] * 3.6   # em km/h
    else:
        _z_hsr    = 19.0   # km/h padrão
        _z_sprint = 24.0   # km/h padrão

    distancia_total = 0
    dist_hi = 0
    dist_sprint = 0
    dist_z4 = 0          # Zone intermediária entre HSR e sprint
    player_load = 0
    pl_vals = []          # parâmetro 'pl' do sensor (fonte oficial do PlayerLoad)
    velocidades = []
    fcs = []
    acc_list = []
    mp_list = []

    prev_v = None
    in_sprint = False
    in_hi = False
    sprints = 0
    n_esforcos_hi = 0
    rhie_effort_frames = []   # timestamps de cada entrada >19 km/h
    _frame_idx = 0
    _in_hi_rhie = False

    for ponto in sensor_points:
        if ponto.get('v') is not None:
            v_ms = float(ponto['v'])
            v_kmh = v_ms * 3.6
            velocidades.append(v_kmh)

            if prev_v is not None:
                dt = 0.1  # 10 Hz
                dist_seg = ((prev_v + v_ms) / 2) * dt
                distancia_total += dist_seg
                if v_kmh > _z_hsr:
                    dist_hi += dist_seg
                if v_kmh > _z_sprint:
                    dist_sprint += dist_seg
                if _z_hsr < v_kmh <= _z_sprint:
                    dist_z4 += dist_seg

            if v_kmh > _z_sprint and not in_sprint:
                sprints += 1
                in_sprint = True
            elif v_kmh <= _z_sprint:
                in_sprint = False

            if v_kmh > _z_hsr and not in_hi:
                n_esforcos_hi += 1
                in_hi = True
            elif v_kmh <= _z_hsr:
                in_hi = False

            # Registra frame de entrada em alta intensidade para RHIE
            if v_kmh > _z_hsr and not _in_hi_rhie:
                rhie_effort_frames.append(_frame_idx)
                _in_hi_rhie = True
            elif v_kmh <= _z_hsr:
                _in_hi_rhie = False

            prev_v = v_ms
        _frame_idx += 1

        if ponto.get('a') is not None:
            acc = float(ponto['a'])
            acc_list.append(acc)
        else:
            acc_list.append(0.0)

        if ponto.get('pl') is not None:
            try:
                pl_vals.append(float(ponto['pl']))
            except (TypeError, ValueError):
                pass

        if ponto.get('hr') is not None:
            hr = float(ponto['hr'])
            if hr > 0:
                fcs.append(hr)

        if ponto.get('mp') is not None:
            mp_val = float(ponto['mp'])
            if mp_val > 0:
                mp_list.append(mp_val)

    # Conta eventos de acc/dec com duração mínima sustentada
    acc_arr = np.array(acc_list)
    mask_acel   = detectar_eventos_acc(acc_arr, 3.0, min_dur_s=min_dur_s, acima=True)
    mask_decel  = detectar_eventos_acc(acc_arr, 3.0, min_dur_s=min_dur_s, acima=False)
    mask_acel23 = detectar_eventos_acc(acc_arr, 2.0, min_dur_s=min_dur_s, acima=True)  & ~mask_acel
    mask_dec23  = detectar_eventos_acc(acc_arr, 2.0, min_dur_s=min_dur_s, acima=False) & ~mask_decel
    acels_intensas    = int(mask_acel.sum())
    desacels_intensas = int(mask_decel.sum())
    acels_23          = int(mask_acel23.sum())
    desacels_23       = int(mask_dec23.sum())
    acc_max = float(np.max(acc_arr))  if len(acc_arr) > 0 else 0.0
    dcc_max = float(np.min(acc_arr))  if len(acc_arr) > 0 else 0.0  # valor mais negativo

    # RHIE: blocos de esforços repetidos em alta intensidade (≥2 entradas >19 km/h separadas por <21s)
    rhie_blocos = 0
    if len(rhie_effort_frames) >= 2:
        _cluster_size = 1
        _in_cluster   = False
        for _j in range(1, len(rhie_effort_frames)):
            _gap = rhie_effort_frames[_j] - rhie_effort_frames[_j - 1]
            if _gap <= 210:          # < 21 segundos a 10 Hz
                _cluster_size += 1
                if _cluster_size == 2 and not _in_cluster:
                    rhie_blocos += 1
                    _in_cluster = True
            else:
                _cluster_size = 1
                _in_cluster   = False

    # (Validação vs OpenField) PlayerLoad OFICIAL: usa o parâmetro 'pl' do
    # sensor — mesma fonte do OpenField. O Σa² antigo superestimava 15–54×
    # com viés NÃO constante (CV 25% entre atletas); fica só como último
    # fallback quando a conta não envia 'pl'.
    _pl_oficial = _mtr.playerload_total(pl_vals)
    if _pl_oficial > 0:
        player_load = _pl_oficial
    else:
        player_load = float(sum(_a * _a for _a in acc_list))
        _diag_log('Métricas', f"{athlete_name}: sensor sem parâmetro 'pl' — "
                              "PlayerLoad aproximado por Σa² (escala difere do OpenField)")

    duracao_min = len(sensor_points) * 0.1 / 60
    m_min = round(distancia_total / duracao_min, 1) if duracao_min > 0 else 0.0

    return {
        'Atleta': athlete_name,
        'Duração (min)': round(duracao_min, 1),
        'Distância (m)': round(distancia_total, 0),
        'Dist. 19-24 km/h (m)': round(dist_z4, 0),
        'Dist. > 19 km/h (m)': round(dist_hi, 0),
        'Dist. > 24 km/h (m)': round(dist_sprint, 0),
        'PlayerLoad': round(player_load, 1),
        'Velocidade Máx (km/h)': round(max(velocidades), 1) if velocidades else 0,
        'Velocidade Média (km/h)': round(np.mean(velocidades), 1) if velocidades else 0,
        'M/min': m_min,
        'FC Máx (bpm)': round(max(fcs), 0) if fcs else 0,
        'FC Média (bpm)': round(np.mean(fcs), 0) if fcs else 0,
        'Sprints (>24 km/h)': sprints,
        'Esforços Alta Int.': n_esforcos_hi,
        'Acc 2-3 (m/s²)': acels_23,
        'Dcc 2-3 (m/s²)': desacels_23,
        'Acelerações (>3 m/s²)': acels_intensas,
        'Desacelerações (<-3 m/s²)': desacels_intensas,
        'Acc Max (m/s²)': round(acc_max, 2),
        'Dcc Max (m/s²)': round(abs(dcc_max), 2),
        'RHIE Blocos': rhie_blocos,
        'Total Pontos': len(sensor_points),
        'MP Médio (W/kg)': round(float(np.mean(mp_list)), 2) if mp_list else 0,
        'MP Máx (W/kg)': round(float(np.max(mp_list)), 2) if mp_list else 0,
    }

def calcular_janelas_discretas_10s(sensor_points, window_minutes, metric_name, band_filter=None):
    """
    Rolling window (janela deslizante) para métricas de velocidade, aceleração
    e PlayerLoad. A janela desliza a cada 10 s sobre a série temporal completa,
    produzindo um ponto por passo — mais fiel ao pico real do que janelas fixas.
    """
    if not sensor_points or len(sensor_points) < 20:
        return [], []

    # ── Extrair série temporal da métrica ─────────────────────────────────────
    tempos  = []
    valores = []
    t_ini   = None

    for ponto in sensor_points:
        if metric_name not in ponto or ponto[metric_name] is None:
            continue
        ts = ponto.get('ts', 0)
        cs = ponto.get('cs', 0)
        t  = ts + (cs / 100) if cs else ts
        if t_ini is None:
            t_ini = t
        t_rel = t - t_ini

        if metric_name == 'v':
            val = float(ponto[metric_name]) * 3.6
        else:
            val = float(ponto[metric_name])

        # Filtro de bandas
        if band_filter is not None:
            if 'velocity_bands' in band_filter and metric_name == 'v':
                vel_kmh = float(ponto['v']) * 3.6
                if   vel_kmh < 10: band = 1
                elif vel_kmh < 15: band = 2
                elif vel_kmh < 20: band = 3
                elif vel_kmh < 25: band = 4
                elif vel_kmh < 30: band = 5
                elif vel_kmh < 35: band = 6
                else:              band = 7
                if band not in band_filter['velocity_bands']:
                    continue
            elif 'acceleration_bands' in band_filter and metric_name == 'a':
                acc = float(ponto['a'])
                if   acc >  2: band =  3
                elif acc >  1: band =  2
                elif acc >  0: band =  1
                elif acc == 0: band =  0
                elif acc > -1: band = -1
                elif acc > -2: band = -2
                else:          band = -3
                if band not in band_filter['acceleration_bands']:
                    continue

        tempos.append(t_rel)
        valores.append(val)

    if len(tempos) < 20:
        return [], []

    t_arr = np.array(tempos,  dtype=float)
    v_arr = np.array(valores, dtype=float)

    # ── Rolling window por contagem de amostras (coerente com WCS) ────────────
    _HZ      = 10.0
    n_window = int(round(window_minutes * 60.0 * _HZ))   # amostras fixas na janela
    step     = int(_HZ * 10)                              # passo de 10 s
    n        = len(v_arr)

    if n < n_window:
        return [], []

    t_out, d_out = [], []
    for i in range(0, n - n_window + 1, step):
        janela_vals = v_arr[i:i + n_window]
        t_out.append(t_arr[i] / 60.0)          # início da janela (não o centro)
        d_out.append(float(np.mean(janela_vals)))

    return t_out, d_out

def calcular_distancia_janelas_discretas_10s(sensor_points, window_minutes):
    """
    Rolling window para distância (fallback via sensor_points).
    Slide de 1 amostra de cada vez → verdadeiro máximo global.
    Grava para exibição a cada 1 s; injeta o pico real se necessário.
    Retorna (tempos_em_min, valores_em_m_por_min).
    """
    if not sensor_points or len(sensor_points) < 20:
        return [], []

    _HZ = 10.0   # frequência nominal GPS Catapult

    sv     = []
    t_abs  = []
    t_ini  = None

    for ponto in sensor_points:
        v = ponto.get('v')
        if v is None:
            continue
        ts = ponto.get('ts', 0)
        cs = ponto.get('cs', 0)
        t  = ts + (cs / 100) if cs else ts
        if t_ini is None:
            t_ini = t
        sv.append(float(v) / _HZ)
        t_abs.append(t - t_ini)

    n_total      = len(sv)
    n_window     = int(round(window_minutes * 60.0 * _HZ))
    step_display = max(1, int(_HZ))   # 1 s por ponto de exibição

    if n_total < n_window + 1:
        return [], []

    sv_arr = np.array(sv,    dtype=float)
    t_arr  = np.array(t_abs, dtype=float)

    w_sum    = float(sv_arr[:n_window].sum())
    best_sum = w_sum
    best_i   = 0

    t_out = [t_arr[0] / 60.0]
    d_out = [w_sum / window_minutes]

    for i in range(1, n_total - n_window + 1):
        w_sum += sv_arr[i + n_window - 1] - sv_arr[i - 1]
        if w_sum > best_sum:
            best_sum = w_sum
            best_i   = i
        if i % step_display == 0:
            t_out.append(t_arr[i] / 60.0)
            d_out.append(w_sum / window_minutes)

    # Injeta o pico real se não caiu num ponto de exibição
    if best_i % step_display != 0:
        best_t = t_arr[best_i] / 60.0
        best_d = best_sum / window_minutes
        ins = next((k for k, t in enumerate(t_out) if t > best_t), len(t_out))
        t_out.insert(ins, best_t)
        d_out.insert(ins, best_d)

    return t_out, d_out

def calcular_distancia_janelas_por_vel_posicao(vel_kmh_list, ts_list, window_minutes, hz=10.0):
    """
    Rolling window de Distância usando EXATAMENTE os mesmos dados de posição (vel GPS, km/h)
    e o mesmo algoritmo do WCS — garante coerência total entre as duas abas.

    Parâmetros:
        vel_kmh_list  : velocidades em km/h (dados_posicao_por_periodo[p][a]['vel'])
        ts_list       : timestamps Unix em segundos (dados_posicao_por_periodo[p][a]['ts_pos'])
        window_minutes: tamanho da janela em minutos
        hz            : frequência de amostragem detectada (padrão 10 Hz)

    Retorna (tempos_em_min, valores_em_m_por_min).

    Algoritmo idêntico ao WCS:
      - Slide de 1 amostra de cada vez → encontra o VERDADEIRO máximo global
      - Registra para exibição a cada 1 s (hz amostras) para manter o gráfico fluido
      - Se o pico real não cair num ponto de exibição, ele é inserido explicitamente
        para que o evento reportado coincida exatamente com o WCS.
    """
    if not vel_kmh_list or len(vel_kmh_list) < 20:
        return [], []

    # Mesma conversão do WCS: km/h ÷ 3.6 ÷ Hz = m/amostra (integração retangular)
    sv = [float(v) / (3.6 * hz) for v in vel_kmh_list]

    # Timestamps relativos em segundos
    n = len(sv)
    if ts_list and len(ts_list) >= n:
        t0    = float(ts_list[0])
        t_abs = [float(ts_list[i]) - t0 for i in range(n)]
    else:
        t_abs = [i / hz for i in range(n)]

    n_window     = int(round(window_minutes * 60.0 * hz))
    step_display = max(1, int(hz))   # 1 s por ponto de exibição (hz amostras)

    if n < n_window + 1:
        return [], []

    sv_arr = np.array(sv,    dtype=float)
    t_arr  = np.array(t_abs, dtype=float)

    # ── Sliding window sum em resolução total (igual ao WCS) ────────────────
    w_sum    = float(sv_arr[:n_window].sum())
    best_sum = w_sum
    best_i   = 0

    t_out = [t_arr[0] / 60.0]
    d_out = [w_sum / window_minutes]

    for i in range(1, n - n_window + 1):
        w_sum += sv_arr[i + n_window - 1] - sv_arr[i - 1]
        # Rastreia o pico real (toda amostra, como o WCS)
        if w_sum > best_sum:
            best_sum = w_sum
            best_i   = i
        # Grava para exibição a cada 1 s
        if i % step_display == 0:
            t_out.append(t_arr[i] / 60.0)
            d_out.append(w_sum / window_minutes)

    # ── Injeta o pico real se não caiu num ponto de exibição ────────────────
    if best_i % step_display != 0:
        best_t = t_arr[best_i] / 60.0
        best_d = best_sum / window_minutes
        ins = next((k for k, t in enumerate(t_out) if t > best_t), len(t_out))
        t_out.insert(ins, best_t)
        d_out.insert(ins, best_d)

    return t_out, d_out

def combinar_periodos_continuo_posicao(dados_posicao_por_periodo: dict, atleta: str):
    """
    Combina vel (km/h) + ts_pos de múltiplos períodos de dados_posicao_por_periodo
    em uma linha do tempo contínua — espelha combinar_periodos_continuo() mas para
    dados de posição GPS, garantindo que o cálculo de Distância use exatamente os
    mesmos dados e filtros que o WCS.

    Retorna (vel_kmh_list, ts_list) prontos para
    calcular_distancia_janelas_por_vel_posicao().
    """
    vel_combined: list = []
    ts_combined:  list = []
    t_offset = 0.0

    for _dados_per in dados_posicao_por_periodo.values():
        da  = _dados_per.get(atleta, {})
        vel = da.get('vel', [])
        ts  = da.get('ts_pos', [])
        if not vel:
            continue

        n = min(len(vel), len(ts)) if ts else len(vel)
        if n == 0:
            continue

        if ts and len(ts) >= n:
            t_ini = float(ts[0])
            t_fim = float(ts[n - 1])
            dur   = max(0.0, t_fim - t_ini)
            for i in range(n):
                ts_combined.append(t_offset + (float(ts[i]) - t_ini))
                vel_combined.append(float(vel[i]))
            t_offset += dur + 0.1
        else:
            hz_est = 10.0
            for i in range(n):
                ts_combined.append(t_offset + i / hz_est)
                vel_combined.append(float(vel[i]))
            t_offset += n / hz_est + 0.1

    return vel_combined, ts_combined

def obter_limites_periodos_posicao(dados_posicao_por_periodo: dict, atleta: str) -> list:
    """
    Retorna a lista de fronteiras de tempo de cada período no timeline contínuo
    gerado por combinar_periodos_continuo_posicao().

    Retorna [(t_start_min, t_end_min, nome_periodo), ...] ordenados pelo timeline.
    Útil para identificar em qual período cai cada evento de Janelas Temporais.
    """
    boundaries = []
    t_offset   = 0.0

    for nome, _dados_per in dados_posicao_por_periodo.items():
        da  = _dados_per.get(atleta, {})
        vel = da.get('vel', [])
        ts  = da.get('ts_pos', [])
        if not vel:
            continue
        n = min(len(vel), len(ts)) if ts else len(vel)
        if n == 0:
            continue
        if ts and len(ts) >= n:
            t_ini = float(ts[0])
            t_fim = float(ts[n - 1])
            dur   = max(0.0, t_fim - t_ini)
            boundaries.append((t_offset / 60.0, (t_offset + dur) / 60.0, nome))
            t_offset += dur + 0.1
        else:
            hz_est = 10.0
            boundaries.append((t_offset / 60.0, (t_offset + n / hz_est) / 60.0, nome))
            t_offset += n / hz_est + 0.1

    return boundaries

def combinar_periodos_continuo(dados_sensor_por_atleta_por_periodo: dict, atleta: str) -> list:
    """
    Combina sensor_points de múltiplos períodos em uma linha do tempo contínua.
    Elimina o gap de intervalo/pausa: cada período começa imediatamente após
    o fim do anterior, gerando um eixo X corrido sem buracos.

    Retorna lista de sensor_points com 'ts' reescrito e 'cs'=0.
    """
    result   = []
    t_offset = 0.0           # tempo acumulado em segundos

    for _dados_per in dados_sensor_por_atleta_por_periodo.values():
        pts = _dados_per.get(atleta, [])
        if not pts:
            continue

        # Encontra t_inicio e t_fim do período
        t_ini_per = None
        t_fim_per = None
        for p in pts:
            _ts = p.get('ts', 0)
            _cs = p.get('cs', 0)
            _t  = _ts + (_cs / 100) if _cs else _ts
            if t_ini_per is None or _t < t_ini_per:
                t_ini_per = _t
            if t_fim_per is None or _t > t_fim_per:
                t_fim_per = _t

        if t_ini_per is None:
            continue

        duracao_per = max(0.0, t_fim_per - t_ini_per)

        for p in pts:
            _ts = p.get('ts', 0)
            _cs = p.get('cs', 0)
            _t  = _ts + (_cs / 100) if _cs else _ts
            p_cpy = dict(p)
            p_cpy['ts'] = t_offset + (_t - t_ini_per)
            p_cpy['cs'] = 0
            result.append(p_cpy)

        t_offset += duracao_per + 0.1   # 0.1 s de margem entre períodos

    return result

def encontrar_eventos_nao_sobrepostos(t_start_min, d_out, window_minutes, limiar_alta, limiar_media, max_val):
    """
    Encontra eventos distintos e não-sobrepostos acima dos limiares de intensidade.

    Algoritmo (idêntico ao WCS):
      1. Ordena todas as janelas por valor decrescente.
      2. Seleciona a melhor janela.
      3. Marca como 'usadas' todas as janelas que se sobrepõem a ela
         (separação mínima = window_minutes).
      4. Repete até não haver janelas acima do limiar.

    Retorna (alta_events, media_alta_events) — listas de dicts ordenadas por valor.
    """
    if not d_out or max_val <= 0:
        return [], []

    n = len(d_out)
    # Quantos passos (de 10 s cada) equivalem a 1 janela completa
    step_min = (t_start_min[1] - t_start_min[0]) if n > 1 else (10.0 / 60.0)
    # Blindagem: timestamps duplicados → step_min = 0 → divisão (numpy) vira inf →
    # int(inf) levantaria OverflowError. Usa o passo padrão de 10 s nesse caso.
    if not np.isfinite(step_min) or step_min <= 0:
        step_min = 10.0 / 60.0
    excl_steps = max(1, int(round(window_minutes / step_min)))

    usado = [False] * n
    alta_events, media_events = [], []

    for idx in sorted(range(n), key=lambda k: d_out[k], reverse=True):
        if usado[idx]:
            continue
        val = d_out[idx]
        if val < limiar_media:
            break                          # lista ordenada → nada abaixo será útil

        # Marca a janela e todas sobrepostas (±1 window) como usadas
        for j in range(max(0, idx - excl_steps + 1), min(n, idx + excl_steps)):
            usado[j] = True

        t_ini   = t_start_min[idx]
        t_fim   = t_ini + window_minutes
        if not (np.isfinite(t_ini) and np.isfinite(t_fim)):
            continue   # ignora janelas com tempo não-finito (evita int(inf)/int(nan))
        mins_i  = int(t_ini);  segs_i = int((t_ini - mins_i) * 60)
        mins_f  = int(t_fim);  segs_f = int((t_fim - mins_f) * 60)

        event = dict(
            inicio=f"{mins_i:02d}:{segs_i:02d}",
            fim=f"{mins_f:02d}:{segs_f:02d}",
            t_ini_min=t_ini,           # float → usado para lookup de período
            valor=round(val, 1),
            pct_max=round(val / max_val * 100, 1),
            intensidade='Alta Intensidade 🔴' if val >= limiar_alta
                        else 'Média-Alta Intensidade 🟡',
        )
        if val >= limiar_alta:
            alta_events.append(event)
        else:
            media_events.append(event)

    alta_events.sort(key=lambda e: e['valor'], reverse=True)
    media_events.sort(key=lambda e: e['valor'], reverse=True)
    return alta_events, media_events

def processar_efforts_velocidade(efforts_data, historical_vmax_ms=None):
    """Processa esforços de velocidade.

    historical_vmax_ms: Vmax histórico em m/s para calcular '% do Máximo'.
                        Se None, usa o máximo da sessão como denominador.
    """
    if not efforts_data:
        return pd.DataFrame()

    records = []
    max_vel_encontrada = 0

    for effort in efforts_data:
        max_vel = effort.get('max_velocity', 0)
        if max_vel:
            max_vel_encontrada = max(max_vel_encontrada, max_vel)

    # Se histórico fornecido e é maior que o da sessão, usa histórico
    if historical_vmax_ms and historical_vmax_ms > max_vel_encontrada:
        velocidade_max = historical_vmax_ms
    else:
        velocidade_max = max_vel_encontrada

    for i, effort in enumerate(efforts_data, 1):
        start_time = effort.get('start_time', 0)
        max_vel = effort.get('max_velocity', 0)
        start_vel = effort.get('start_velocity', 0)
        end_time = effort.get('end_time', 0)
        duration = (end_time - start_time) if end_time else 0
        distance = effort.get('distance', 0)
        band = effort.get('band', '')

        percent_of_max = (max_vel / velocidade_max * 100) if velocidade_max > 0 else 0

        hora_str = ''
        if start_time:
            try:
                hora_str = datetime.fromtimestamp(start_time).strftime('%H:%M:%S')
            except Exception:
                hora_str = str(start_time)

        records.append({
            'Esforço': i,
            'Duração (s)': round(duration, 1),
            'Início': hora_str,
            'Vel. Inicial (km/h)': round(start_vel * 3.6, 1) if start_vel else 0,
            'Vel. Máx (km/h)': round(max_vel * 3.6, 1) if max_vel else 0,
            'Distância (m)': round(distance, 1),
            '% do Máximo': round(percent_of_max, 1),
            'Banda': _rotulo_banda_vel(band),
            '_band_num': band,
            '_start_ts': start_time,
            '_end_ts': end_time,
        })

    return pd.DataFrame(records)

def processar_efforts_aceleracao(efforts_data, historical_max_acc=None):
    """Processa esforços de aceleração.

    historical_max_acc: aceleração máxima histórica (m/s²) para '% do Máximo'.
                        Se None, usa o máximo da sessão.
    """
    if not efforts_data:
        return pd.DataFrame()

    records = []

    max_acc_positiva = 0
    max_acc_negativa = 0

    for effort in efforts_data:
        acc = effort.get('acceleration', 0)
        if acc > 0:
            max_acc_positiva = max(max_acc_positiva, acc)
        elif acc < 0:
            max_acc_negativa = min(max_acc_negativa, acc)

    if historical_max_acc and historical_max_acc > max_acc_positiva:
        max_acc_positiva = historical_max_acc
    
    for i, effort in enumerate(efforts_data, 1):
        start_time = effort.get('start_time', 0)
        acceleration = effort.get('acceleration', 0)
        end_time = effort.get('end_time', 0)
        duration = (end_time - start_time) if end_time else 0
        distance = effort.get('distance', 0)
        band = effort.get('band', '')
        
        if acceleration > 0:
            percent_of_max = (acceleration / max_acc_positiva * 100) if max_acc_positiva > 0 else 0
            tipo = 'Aceleração'
        elif acceleration < 0:
            percent_of_max = (abs(acceleration) / abs(max_acc_negativa) * 100) if max_acc_negativa < 0 else 0
            tipo = 'Desaceleração'
        else:
            percent_of_max = 0
            tipo = 'Constante'
        
        hora_str = ''
        if start_time:
            try:
                hora_str = datetime.fromtimestamp(start_time).strftime('%H:%M:%S')
            except:
                hora_str = str(start_time)
        
        records.append({
            'Esforço': i,
            'Duração (s)': round(duration, 1),
            'Início': hora_str,
            'Aceleração (m/s²)': round(acceleration, 2),
            'Distância (m)': round(distance, 1),
            '% do Máximo': round(percent_of_max, 1),
            'Tipo': tipo,
            'Banda': _rotulo_banda_acc(band),
            '_band_num': band,
            '_start_ts': start_time,
            '_end_ts': end_time
        })
    
    return pd.DataFrame(records)

def combinar_periodos(resultados_por_periodo: dict) -> list:
    """
    Combina os resultados de múltiplos períodos em uma lista única de atletas.
    - Métricas quantitativas acumuláveis → soma
    - Métricas de pico (máximos) → máximo
    - 'Velocidade Média', 'FC Média', 'M/min' → recalculados a partir dos totais
    - 'Posição', 'Equipe', 'Atleta' → mantidos do primeiro período com dado
    Retorna lista de dicts no mesmo formato que resultados_por_periodo[periodo].
    """
    from collections import defaultdict
    atleta_rows: dict[str, list] = defaultdict(list)
    for resultados in resultados_por_periodo.values():
        for row in resultados:
            nome = row.get('Atleta', '')
            if nome:
                atleta_rows[nome].append(row)

    combinados = []
    for nome, rows in atleta_rows.items():
        comb = {'Atleta': nome}
        # Copia campos não-numéricos do primeiro registro disponível
        for campo in ('Posição', 'Equipe'):
            comb[campo] = next((r.get(campo, '') for r in rows if r.get(campo)), '')

        # Coleta todas as chaves numéricas
        todas_keys = set()
        for r in rows:
            todas_keys |= set(r.keys())
        todas_keys -= {'Atleta', 'Posição', 'Equipe'}

        for key in todas_keys:
            vals = [r[key] for r in rows if key in r and r[key] is not None]
            if not vals:
                comb[key] = 0
            elif key in _METRICAS_MAX:
                comb[key] = max(vals)
            elif key in _METRICAS_SUM:
                comb[key] = round(sum(vals), 2)
            else:
                # Por padrão: soma (cobre novos campos quantitativos futuros)
                try:
                    comb[key] = round(sum(float(v) for v in vals), 2)
                except (TypeError, ValueError):
                    comb[key] = vals[0]

        # Recalcula métricas derivadas a partir dos totais combinados
        dur_min = comb.get('Duração (min)', 0)
        dist_m  = comb.get('Distância (m)', 0)
        if dur_min and dur_min > 0:
            comb['M/min'] = round(dist_m / dur_min, 1)
        # FC Média e Velocidade Média: média simples dos períodos (aproximação)
        for campo_avg in ('FC Média (bpm)', 'Velocidade Média (km/h)'):
            vals_avg = [r.get(campo_avg, 0) for r in rows if r.get(campo_avg)]
            if vals_avg:
                comb[campo_avg] = round(sum(vals_avg) / len(vals_avg), 1)

        combinados.append(comb)
    return combinados

def _segmentos_de_mask(mask):
    """Retorna lista de (start, end) para segmentos contínuos True em mask."""
    segs, n, i = [], len(mask), 0
    while i < n:
        if mask[i]:
            s = i
            while i < n and mask[i]:
                i += 1
            segs.append((s, i))
        else:
            i += 1
    return segs

def calcular_efforts_velocidade_sensor(
        xn, yn, vel_arr, ts_pos=None, min_dur_s=1.0, freq_hz=10):
    """
    Detecta esforços de velocidade diretamente dos dados do sensor Catapult.
    Usa os mesmos dados de posição/velocidade que calcular_metricas(),
    garantindo totais idênticos com a Tabela Descritiva.

    Para cada banda de BANDAS_VEL, detecta segmentos contínuos onde a
    velocidade permanece dentro da banda por pelo menos min_dur_s.
    Retorna DataFrame com mesma estrutura de processar_efforts_velocidade().
    """
    if not vel_arr or not xn or not yn:
        return pd.DataFrame()

    n          = len(vel_arr)
    min_frames = max(1, round(min_dur_s * freq_hz))

    # Distâncias ponto-a-ponto
    dists = [0.0] * n
    for i in range(1, min(n, len(xn), len(yn))):
        dx = xn[i] - xn[i - 1]
        dy = yn[i] - yn[i - 1]
        dists[i] = (dx * dx + dy * dy) ** 0.5

    max_vel_global = max(vel_arr) if vel_arr else 1.0

    records = []
    esf_num = 1

    _bv_det = _bandas_vel_ativas()
    for banda_id, bcfg in _bv_det.items():
        bmin, bmax = bcfg['min'], bcfg['max']

        # Máscara booleana: ponto dentro da banda
        mask = np.array([(bmin <= v < bmax) for v in vel_arr], dtype=bool)
        for seg_s, seg_e in _segmentos_de_mask(mask):
            dur_frames = seg_e - seg_s
            if dur_frames < min_frames:
                continue

            seg_vel  = vel_arr[seg_s:seg_e]
            seg_dist = sum(dists[seg_s:seg_e])
            dur_s    = dur_frames / freq_hz
            vel_max  = max(seg_vel)
            vel_ini  = seg_vel[0]
            pct_max  = round(vel_max / max_vel_global * 100, 1) if max_vel_global > 0 else 0

            # Timestamps
            _ts_s = _ts_e_val = 0.0
            hora_str = f"{seg_s / freq_hz:.1f}s"
            if ts_pos and len(ts_pos) > seg_s and ts_pos[seg_s] and ts_pos[seg_s] > 0:
                _ts_s   = float(ts_pos[seg_s])
                _ts_e_val = float(ts_pos[seg_e - 1]) if seg_e - 1 < len(ts_pos) else _ts_s + dur_s
                try:
                    hora_str = datetime.fromtimestamp(_ts_s).strftime('%H:%M:%S')
                except Exception:
                    _applog.log_debug_exc()

            records.append({
                'Esforço':            esf_num,
                'Início':             hora_str,
                'Duração (s)':        round(dur_s, 1),
                'Vel. Máx (km/h)':    round(vel_max, 1),
                'Vel. Inicial (km/h)': round(vel_ini, 1),
                'Distância (m)':      round(seg_dist, 1),
                '% do Máximo':        pct_max,
                'Banda':              _rotulo_banda_vel(banda_id),
                '_band_num':          banda_id,
                '_start_ts':          _ts_s,
                '_end_ts':            _ts_e_val,
                '_seg_start_idx':     seg_s,
                '_seg_end_idx':       seg_e,
            })
            esf_num += 1

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if df['_start_ts'].sum() > 0:
        df = df.sort_values('_start_ts').reset_index(drop=True)
    df['Esforço'] = range(1, len(df) + 1)
    return df

def calcular_efforts_aceleracao_sensor(
        xn, yn, acc_arr, vel_arr=None, ts_pos=None, min_dur_s=0.6, freq_hz=10):
    """
    Detecta esforços de aceleração/desaceleração diretamente dos dados do sensor.
    Usa os mesmos dados de aceleração que calcular_metricas(), garantindo
    totais idênticos com a Tabela Descritiva.
    Retorna DataFrame com mesma estrutura de processar_efforts_aceleracao().
    """
    if not acc_arr or not xn or not yn:
        return pd.DataFrame()

    n          = len(acc_arr)
    min_frames = max(1, round(min_dur_s * freq_hz))

    dists = [0.0] * n
    for i in range(1, min(n, len(xn), len(yn))):
        dx = xn[i] - xn[i - 1]
        dy = yn[i] - yn[i - 1]
        dists[i] = (dx * dx + dy * dy) ** 0.5

    max_acc_global = max((abs(a) for a in acc_arr), default=1.0)
    vel_arr = vel_arr or [0.0] * n

    records = []
    esf_num = 1

    _ba_det = _bandas_acc_ativas()
    for banda_id, bcfg in _ba_det.items():
        bmin, bmax = bcfg['min'], bcfg['max']
        # Ordenar para uso uniforme
        lo, hi = min(bmin, bmax), max(bmin, bmax)

        mask = np.array([(lo <= a <= hi) for a in acc_arr], dtype=bool)
        # Aplica mínimo de frames (contínuo)
        for seg_s, seg_e in _segmentos_de_mask(mask):
            dur_frames = seg_e - seg_s
            if dur_frames < min_frames:
                continue

            seg_acc  = acc_arr[seg_s:seg_e]
            dur_s    = dur_frames / freq_hz
            acc_max  = max(abs(a) for a in seg_acc)
            acc_avg  = sum(seg_acc) / len(seg_acc)
            pct_max  = round(acc_max / max_acc_global * 100, 1) if max_acc_global > 0 else 0
            tipo     = 'Aceleração' if acc_avg >= 0 else 'Desaceleração'

            _ts_s = _ts_e_val = 0.0
            hora_str = f"{seg_s / freq_hz:.1f}s"
            if ts_pos and len(ts_pos) > seg_s and ts_pos[seg_s] and ts_pos[seg_s] > 0:
                _ts_s   = float(ts_pos[seg_s])
                _ts_e_val = float(ts_pos[seg_e - 1]) if seg_e - 1 < len(ts_pos) else _ts_s + dur_s
                try:
                    hora_str = datetime.fromtimestamp(_ts_s).strftime('%H:%M:%S')
                except Exception:
                    _applog.log_debug_exc()

            records.append({
                'Esforço':        esf_num,
                'Início':         hora_str,
                'Duração (s)':    round(dur_s, 1),
                'Acc. Máx (m/s²)': round(acc_max, 2),
                'Acc. Médio (m/s²)': round(acc_avg, 2),
                'Vel. Máx (km/h)': round(max(vel_arr[seg_s:seg_e]), 1)
                                   if vel_arr else 0,
                '% do Máximo':    pct_max,
                'Tipo':           tipo,
                'Banda':          _rotulo_banda_acc(_ACC_KEY_TO_NUM.get(banda_id, banda_id)),
                '_band_num':      banda_id,
                '_start_ts':      _ts_s,
                '_end_ts':        _ts_e_val,
                '_seg_start_idx': seg_s,
                '_seg_end_idx':   seg_e,
            })
            esf_num += 1

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if df['_start_ts'].sum() > 0:
        df = df.sort_values('_start_ts').reset_index(drop=True)
    df['Esforço'] = range(1, len(df) + 1)
    return df

def criar_grafico_velocidade_tempo(sensor_points, athlete_name, window_size=31, show_trend=True, intensity_filter=None):
    if not sensor_points or len(sensor_points) == 0:
        return None
    
    tempos = []
    velocidades = []
    tempo_inicial = None
    
    for ponto in sensor_points:
        if 'v' in ponto and ponto['v']:
            ts = ponto.get('ts', 0)
            cs = ponto.get('cs', 0)
            tempo = ts + (cs / 100) if cs else ts
            
            if tempo_inicial is None:
                tempo_inicial = tempo
            
            tempo_relativo = (tempo - tempo_inicial) / 60
            v_kmh = float(ponto['v']) * 3.6
            
            if intensity_filter is None or v_kmh >= intensity_filter:
                tempos.append(tempo_relativo)
                velocidades.append(v_kmh)
    
    if len(tempos) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tempos,
        y=velocidades,
        mode='lines',
        name='Velocidade Original',
        line=dict(color='lightblue', width=1, dash='dot'),
        opacity=0.5
    ))
    
    if show_trend and len(velocidades) > window_size:
        try:
            window = min(window_size, len(velocidades) - (len(velocidades) % 2 - 1))
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                velocidades_suavizadas = savgol_filter(velocidades, window, 3)
                fig.add_trace(go.Scatter(
                    x=tempos,
                    y=velocidades_suavizadas,
                    mode='lines',
                    name='Tendência (Suavizada)',
                    line=dict(color='blue', width=2)
                ))
        except:
            moving_avg = np.convolve(velocidades, np.ones(window_size)/window_size, mode='valid')
            tempos_ma = tempos[window_size//2:-(window_size//2)] if window_size//2 > 0 else tempos[:len(moving_avg)]
            fig.add_trace(go.Scatter(
                x=tempos_ma,
                y=moving_avg,
                mode='lines',
                name='Tendência (Média Móvel)',
                line=dict(color='blue', width=2)
            ))
    
    fig.update_layout(
        title=f"Velocidade ao Longo do Tempo - {athlete_name}",
        xaxis_title="Tempo (minutos)",
        yaxis_title="Velocidade (km/h)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def criar_grafico_aceleracao_tempo(sensor_points, athlete_name, window_size=31, show_trend=True, intensity_filter=None):
    if not sensor_points or len(sensor_points) == 0:
        return None
    
    tempos = []
    aceleracoes = []
    tempo_inicial = None
    
    for ponto in sensor_points:
        if 'a' in ponto and ponto['a']:
            ts = ponto.get('ts', 0)
            cs = ponto.get('cs', 0)
            tempo = ts + (cs / 100) if cs else ts
            
            if tempo_inicial is None:
                tempo_inicial = tempo
            
            tempo_relativo = (tempo - tempo_inicial) / 60
            acc = float(ponto['a'])
            
            if intensity_filter is None or abs(acc) >= intensity_filter:
                tempos.append(tempo_relativo)
                aceleracoes.append(acc)
    
    if len(tempos) == 0:
        return None
    
    colors = ['green' if a >= 0 else 'red' for a in aceleracoes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tempos,
        y=aceleracoes,
        mode='markers',
        name='Aceleração',
        marker=dict(size=2, color=colors, opacity=0.3)
    ))
    
    if show_trend and len(aceleracoes) > window_size:
        try:
            window = min(window_size, len(aceleracoes) - (len(aceleracoes) % 2 - 1))
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                aceleracoes_suavizadas = savgol_filter(aceleracoes, window, 3)
                fig.add_trace(go.Scatter(
                    x=tempos,
                    y=aceleracoes_suavizadas,
                    mode='lines',
                    name='Tendência (Suavizada)',
                    line=dict(color='purple', width=2)
                ))
        except:
            _applog.log_debug_exc()
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title=f"Aceleração ao Longo do Tempo - {athlete_name}",
        xaxis_title="Tempo (minutos)",
        yaxis_title="Aceleração (m/s²)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def classificar_intensidade(valores, limiar_alta, limiar_media):
    """Classifica janelas por limiares absolutos da literatura (não percentis)."""
    cores = []
    classificacoes = []
    for valor in valores:
        if valor >= limiar_alta:
            cores.append('#ef4444')          # vermelho
            classificacoes.append('Alta Intensidade 🔴')
        elif valor >= limiar_media:
            cores.append('#f59e0b')          # âmbar
            classificacoes.append('Média-Alta Intensidade 🟡')
        else:
            cores.append('#22c55e')          # verde
            classificacoes.append('Baixa Intensidade 🟢')
    return cores, classificacoes

def criar_grafico_intensidade(tempos, valores, cores, metric_name, athlete_name,
                              window_minutes, unidade, limiar_alta=None, limiar_media=None):
    if not tempos or not valores:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tempos,
        y=valores,
        mode='lines+markers',
        name=f'{metric_name} (rolling {window_minutes} min)',
        line=dict(color='rgba(180,180,180,0.35)', width=1.2),
        marker=dict(size=7, color=cores, line=dict(width=0.8, color='rgba(0,0,0,0.4)')),
        hovertemplate=f'%{{x:.1f}} min — %{{y:.1f}} {unidade}<extra></extra>',
    ))

    # ── Linhas de referência dos limiares ─────────────────────────────────────
    if limiar_alta is not None:
        fig.add_hline(
            y=limiar_alta, line_dash='dash', line_color='rgba(239,68,68,0.55)',
            line_width=1.5,
            annotation_text=f'Alta ≥ {limiar_alta} {unidade}',
            annotation_font_color='#ef4444', annotation_font_size=11,
            annotation_position='top right',
        )
    if limiar_media is not None:
        fig.add_hline(
            y=limiar_media, line_dash='dot', line_color='rgba(245,158,11,0.55)',
            line_width=1.5,
            annotation_text=f'Média-Alta ≥ {limiar_media} {unidade}',
            annotation_font_color='#f59e0b', annotation_font_size=11,
            annotation_position='top right',
        )

    fig.update_layout(
        title=f"Intensidade de {metric_name} — {athlete_name}  "
              f"(Rolling Window: {window_minutes} min | Passo: 10 s)",
        xaxis_title="Tempo (minutos)",
        yaxis_title=f"{metric_name} ({unidade})",
        height=500,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.07)', zerolinecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.07)', zerolinecolor='rgba(255,255,255,0.1)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
    )

    fig.add_annotation(
        x=0.01, y=0.985, xref='paper', yref='paper', showarrow=False,
        text='🟢 Baixa  |  🟡 Média-Alta  |  🔴 Alta',
        font=dict(size=10, color='rgba(255,255,255,0.55)'),
        align='left',
    )

    return fig

def criar_tabela_intensidade(tempos, valores, classificacoes, metric_name, unidade):
    if not tempos or not valores:
        return pd.DataFrame()
    
    dados_tabela = []
    for i, (tempo, valor, classificacao) in enumerate(zip(tempos, valores, classificacoes), 1):
        if 'Alta' in classificacao or 'Média-Alta' in classificacao:
            minutos = int(tempo)
            segundos = int((tempo - minutos) * 60)
            inicio_str = f"{minutos:02d}:{segundos:02d}:00"
            
            percentual = (valor / max(valores) * 100) if max(valores) > 0 else 0
            
            dados_tabela.append({
                'Esforço': i,
                'Duração (s)': 60,
                'Início': inicio_str,
                f'{metric_name}': round(valor, 1),
                '% do Máximo': round(percentual, 1),
                'Intensidade': classificacao
            })
    
    return pd.DataFrame(dados_tabela)

def exibir_resultados_janela(tempos_janela, valores_janela, nome_metrica, atleta_janela, window_minutes, unidade,
                             period_boundaries=None):
    if not tempos_janela or not valores_janela:
        st.warning("Dados insuficientes para calcular as janelas")
        return

    # Saneamento: remove pares com tempo/valor não-finito (inf/nan). Sem isso,
    # conversões int() adiante (formatação de tempo) podem levantar OverflowError.
    _t_arr = np.asarray(tempos_janela, dtype=float)
    _v_arr = np.asarray(valores_janela, dtype=float)
    _fin = np.isfinite(_t_arr) & np.isfinite(_v_arr)
    if not _fin.all():
        tempos_janela = _t_arr[_fin].tolist()
        valores_janela = _v_arr[_fin].tolist()
        if not tempos_janela or not valores_janela:
            st.warning("Dados insuficientes para calcular as janelas")
            return

    valores_array = np.array(valores_janela)

    # ── Limiares como % do valor máximo da sessão ─────────────────────────────
    # Alta Intensidade  : > 90 % do máximo
    # Média-Alta        : ≥ 75 % e ≤ 90 % do máximo
    # Baixa             : < 75 % do máximo
    _max_val      = float(valores_array.max()) if len(valores_array) > 0 else 1.0
    _limiar_alta  = round(_max_val * 0.90, 1)
    _limiar_media = round(_max_val * 0.75, 1)

    cores, classificacoes = classificar_intensidade(valores_janela, _limiar_alta, _limiar_media)

    fig = criar_grafico_intensidade(
        tempos_janela, valores_janela, cores, nome_metrica, atleta_janela,
        window_minutes, unidade, _limiar_alta, _limiar_media,
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # ── Eventos distintos e não-sobrepostos (greedy, igual ao WCS) ───────────
    _alta_ev, _media_ev = encontrar_eventos_nao_sobrepostos(
        tempos_janela, valores_janela,
        window_minutes, _limiar_alta, _limiar_media, _max_val,
    )
    alta_count       = len(_alta_ev)
    media_alta_count = len(_media_ev)

    _card_alta = f"""
    <div style="
        background: linear-gradient(135deg, rgba(220,38,38,0.18) 0%, rgba(153,27,27,0.08) 100%);
        border: 1px solid rgba(239,68,68,0.55);
        border-radius: 18px;
        padding: 32px 24px 26px;
        text-align: center;
        box-shadow: 0 0 32px rgba(220,38,38,0.22), 0 2px 8px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(255,255,255,0.07);
        position: relative; overflow: hidden;
    ">
      <div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;
                  border-radius:50%;background:rgba(220,38,38,0.10);pointer-events:none;"></div>
      <div style="font-size:11px;font-weight:600;letter-spacing:2px;color:rgba(255,255,255,0.5);
                  text-transform:uppercase;margin-bottom:10px;">Alta Intensidade</div>
      <div style="font-size:72px;font-weight:800;color:#f87171;line-height:1;
                  text-shadow:0 0 24px rgba(248,113,113,0.5);">{alta_count}</div>
      <div style="font-size:12px;color:rgba(255,255,255,0.38);margin-top:14px;line-height:1.6;">
        janelas distintas com <strong style="color:rgba(248,113,113,0.8);">{nome_metrica} ≥ {_limiar_alta} {unidade}</strong><br>
        &gt; 90% do máximo ({_max_val:.1f} {unidade})
      </div>
    </div>"""

    _card_media = f"""
    <div style="
        background: linear-gradient(135deg, rgba(202,138,4,0.18) 0%, rgba(133,77,14,0.08) 100%);
        border: 1px solid rgba(234,179,8,0.50);
        border-radius: 18px;
        padding: 32px 24px 26px;
        text-align: center;
        box-shadow: 0 0 32px rgba(202,138,4,0.22), 0 2px 8px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(255,255,255,0.07);
        position: relative; overflow: hidden;
    ">
      <div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;
                  border-radius:50%;background:rgba(202,138,4,0.10);pointer-events:none;"></div>
      <div style="font-size:11px;font-weight:600;letter-spacing:2px;color:rgba(255,255,255,0.5);
                  text-transform:uppercase;margin-bottom:10px;">Média-Alta Intensidade</div>
      <div style="font-size:72px;font-weight:800;color:#fbbf24;line-height:1;
                  text-shadow:0 0 24px rgba(251,191,36,0.5);">{media_alta_count}</div>
      <div style="font-size:12px;color:rgba(255,255,255,0.38);margin-top:14px;line-height:1.6;">
        janelas distintas com <strong style="color:rgba(251,191,36,0.8);">{_limiar_media} ≤ {nome_metrica} &lt; {_limiar_alta} {unidade}</strong><br>
        75–90% do máximo ({_max_val:.1f} {unidade})
      </div>
    </div>"""

    _c1, _c2 = st.columns(2)
    with _c1:
        st.markdown(_card_alta,  unsafe_allow_html=True)
    with _c2:
        st.markdown(_card_media, unsafe_allow_html=True)

    # ── Feedback interpretativo automático ───────────────────────────────────
    _t_total_min = (max(tempos_janela) + window_minutes) if tempos_janela else 0
    _t_total_h   = int(_t_total_min // 60)
    _t_total_m   = int(_t_total_min % 60)
    _t_total_s   = int(round((_t_total_min - int(_t_total_min)) * 60))
    if _t_total_h > 0:
        _t_total_str = f"{_t_total_h}h {_t_total_m:02d}min"
    elif _t_total_s > 0:
        _t_total_str = f"{_t_total_m}min {_t_total_s:02d}s"
    else:
        _t_total_str = f"{_t_total_m} min"

    _n_total_ev  = alta_count + media_alta_count
    _s_ev        = "s" if _n_total_ev != 1 else ""
    _s_alta      = "s" if alta_count   != 1 else ""
    _s_media     = "s" if media_alta_count != 1 else ""

    _feedback_html = f"""
<div style="
    background: linear-gradient(135deg, rgba(25,35,55,0.65) 0%, rgba(15,25,45,0.45) 100%);
    border: 1px solid rgba(255,255,255,0.09);
    border-left: 3px solid rgba(93,173,226,0.55);
    border-radius: 10px;
    padding: 14px 20px;
    margin: 20px 0 10px 0;
    font-size: 0.875rem;
    line-height: 1.75;
    color: rgba(255,255,255,0.72);
">
  💬 &nbsp;<strong style="color:white">{atleta_janela}</strong> teve
  <strong style="color:#f87171">{alta_count}</strong> período{_s_alta} de <span style="color:#f87171">alta intensidade</span> e
  <strong style="color:#fbbf24">{media_alta_count}</strong> de <span style="color:#fbbf24">média-alta</span> —
  totalizando <strong style="color:white">{_n_total_ev} período{_s_ev} distinto{_s_ev} de {window_minutes} min</strong>
  com <em>{nome_metrica}</em> ≥ <strong>{_limiar_media:.1f} {unidade}</strong>,
  ao longo de <strong style="color:#5dade2">{_t_total_str}</strong> de atividade analisada.
  Pico máximo registrado: <strong style="color:white">{_max_val:.1f} {unidade}</strong>
  <span style="color:rgba(255,255,255,0.38)">(100% do máximo individual)</span>.
</div>"""
    st.markdown(_feedback_html, unsafe_allow_html=True)

    # ── Tabela de eventos distintos (Alta + Média-Alta, ordenados por valor) ──
    _todos_ev = (
        [dict(e, _cat='alta')  for e in _alta_ev] +
        [dict(e, _cat='media') for e in _media_ev]
    )
    _todos_ev.sort(key=lambda e: e['valor'], reverse=True)

    st.markdown("#### 📋 Eventos de Média-Alta e Alta Intensidade")
    st.caption(
        f"Cada linha é uma janela de **{window_minutes} min** distinta e não-sobreposta, "
        f"selecionada pelo pico máximo. Separação mínima entre eventos: {window_minutes} min."
    )

    # Helper: encontra o nome do período para um instante t_min (minutos)
    def _periodo_para_t(t_min):
        if not period_boundaries:
            return None
        for (t_s, t_e, nome) in period_boundaries:
            if t_s <= t_min <= t_e + 0.1:   # +0.1 min de tolerância
                return nome
        # fallback: período mais próximo
        return min(period_boundaries, key=lambda b: abs(b[0] - t_min))[2]

    _mostrar_periodo = bool(period_boundaries)

    if _todos_ev:
        _rows = []
        for _rank, _ev in enumerate(_todos_ev, 1):
            row = {
                '#': _rank,
                'Início': _ev['inicio'],
                'Fim':    _ev['fim'],
                f'{nome_metrica} ({unidade})': _ev['valor'],
                '↓ % do Máximo': _ev['pct_max'],
                'Intensidade': _ev['intensidade'],
            }
            if _mostrar_periodo:
                row['Período'] = _periodo_para_t(_ev.get('t_ini_min', 0.0))
            _rows.append(row)
        _df_ev = pd.DataFrame(_rows)

        # Reordena colunas: Período logo após Fim (se presente)
        if _mostrar_periodo and 'Período' in _df_ev.columns:
            _cols = ['#', 'Início', 'Fim', 'Período',
                     f'{nome_metrica} ({unidade})', '↓ % do Máximo', 'Intensidade']
            _df_ev = _df_ev[[c for c in _cols if c in _df_ev.columns]]

        def _style_row(row):
            if 'Alta Intensidade' in str(row.get('Intensidade', '')) and 'Média' not in str(row.get('Intensidade', '')):
                return ['background-color:rgba(239,68,68,0.12)'] * len(row)
            elif 'Média-Alta' in str(row.get('Intensidade', '')):
                return ['background-color:rgba(245,158,11,0.10)'] * len(row)
            return [''] * len(row)

        _fmt = {f'{nome_metrica} ({unidade})': '{:.1f}', '↓ % do Máximo': '{:.1f}%'}
        _styled = _df_ev.style.apply(_style_row, axis=1).format(_fmt)
        st.dataframe(_styled, use_container_width=True, height=min(600, 40 + len(_rows) * 36))
        st.download_button(
            f"📥 Exportar Eventos - {nome_metrica} (CSV)",
            _df_ev.to_csv(index=False).encode('utf-8'),
            f"eventos_{nome_metrica}_{atleta_janela}_{window_minutes}min.csv",
            mime='text/csv',
        )
    else:
        st.info("Nenhum evento de média-alta ou alta intensidade encontrado")
