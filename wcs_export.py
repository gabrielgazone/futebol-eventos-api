# -*- coding: utf-8 -*-
"""Motor de export WCS multi-atividade (para artigo científico).

Calcula o pico de pior cenário (worst-case scenario) por atleta, variável e
janela temporal (1/3/5 min), no MESMO método canônico da aba WCS:
séries por amostra → `metrics.rolling_sum` + argmax (ou rolling-max para
velocidade máxima). Funções puras (sem Streamlit) — cobertas por testes.

Variáveis suportadas (ver VARIAVEIS): distância total e relativa (m/min),
distância em alta velocidade (HSR) e sprint, velocidade máxima, PlayerLoad e
contagem de acelerações/desacelerações.
"""
from __future__ import annotations

from collections import deque

import numpy as np

import metrics as _mtr
from analysis import detectar_eventos_acc

# Rótulos das variáveis exportáveis (ordem de exibição/coluna).
VAR_DIST      = 'Distância (m)'
VAR_DIST_REL  = 'Distância relativa (m/min)'
VAR_HSR       = 'Distância HSR (m)'
VAR_SPRINT    = 'Distância Sprint (m)'
VAR_VMAX      = 'Velocidade Máx (km/h)'
VAR_PL        = 'PlayerLoad'
VAR_ACC       = 'Acelerações (n)'
VAR_DEC       = 'Desacelerações (n)'

VARIAVEIS = [VAR_DIST, VAR_DIST_REL, VAR_HSR, VAR_SPRINT, VAR_VMAX, VAR_PL,
             VAR_ACC, VAR_DEC]

# Cortes padrão (editáveis pela UI); HSR/Sprint em km/h, acc/dec em m/s².
DEFAULT_HSR_KMH    = 19.8
DEFAULT_SPRINT_KMH = 25.2
DEFAULT_ACC_MS2    = 3.0
DEFAULT_DEC_MS2    = 3.0


def _vel_kmh(sensor_points):
    """Velocidade (km/h) por amostra; 'v' do sensor é m/s."""
    return [float(_p.get('v') or 0.0) * 3.6 for _p in sensor_points]


def serie_por_amostra(sensor_points, variavel, hz=10.0, *,
                      hsr_kmh=DEFAULT_HSR_KMH, sprint_kmh=DEFAULT_SPRINT_KMH,
                      acc_ms2=DEFAULT_ACC_MS2, dec_ms2=DEFAULT_DEC_MS2,
                      min_dur_acc_s=0.6):
    """Valor por amostra da `variavel`, pronto para a janela rolante.

    Para variáveis acumulativas (distância, PlayerLoad, contagens) o valor é a
    contribuição da amostra — a janela SOMA. Para velocidade máxima é a própria
    velocidade — a janela pega o MÁXIMO.
    """
    if not sensor_points:
        return []

    if variavel in (VAR_DIST, VAR_DIST_REL):
        return [_v / (3.6 * hz) for _v in _vel_kmh(sensor_points)]

    if variavel == VAR_HSR:
        return [(_v / (3.6 * hz)) if _v >= hsr_kmh else 0.0
                for _v in _vel_kmh(sensor_points)]

    if variavel == VAR_SPRINT:
        return [(_v / (3.6 * hz)) if _v >= sprint_kmh else 0.0
                for _v in _vel_kmh(sensor_points)]

    if variavel == VAR_VMAX:
        return _vel_kmh(sensor_points)

    if variavel == VAR_PL:
        return [float(_p.get('pl') or 0.0) for _p in sensor_points]

    if variavel in (VAR_ACC, VAR_DEC):
        _acc = np.asarray([float(_p.get('a') or 0.0) for _p in sensor_points],
                          dtype=float)
        if not _acc.size:
            return []
        _mask = detectar_eventos_acc(
            _acc, acc_ms2 if variavel == VAR_ACC else dec_ms2,
            min_dur_s=min_dur_acc_s, acima=(variavel == VAR_ACC), freq_hz=hz)
        return [1.0 if _b else 0.0 for _b in _mask]

    raise ValueError(f"variável desconhecida: {variavel}")


def pico_janela(sv, n_amostras, is_max=False):
    """Pico da janela rolante de `n_amostras` sobre a série `sv`.

    is_max=False → soma da janela (metrics.rolling_sum + argmax) — método
    canônico da aba WCS. is_max=True → máximo dentro da janela (velocidade máx),
    via deque monotônica. Retorna (valor, idx_ini, idx_fim) ou (0.0, 0, 0).
    """
    _n = int(n_amostras)
    if not sv or _n < 1 or len(sv) < _n:
        return 0.0, 0, 0

    if is_max:
        _dq: deque = deque()
        _best, _bsi, _bei = -1.0, 0, _n
        for _i in range(len(sv)):
            while _dq and sv[_dq[-1]] <= sv[_i]:
                _dq.pop()
            _dq.append(_i)
            if _dq[0] <= _i - _n:
                _dq.popleft()
            if _i >= _n - 1:
                _c = sv[_dq[0]]
                if _c > _best:
                    _best, _bei = _c, _i + 1
                    _bsi = _bei - _n
        return (max(_best, 0.0), _bsi, _bei)

    _rw = _mtr.rolling_sum(sv, _n)
    if not _rw:
        return 0.0, 0, 0
    _bi = int(np.argmax(_rw))
    return float(_rw[_bi]), _bi, _bi + _n


def calcular_wcs(sensor_points, variaveis, janelas_min, hz=10.0, **cortes):
    """Pico WCS de cada (variável × janela) para UM atleta num escopo.

    Retorna {(variavel, janela_min): valor}. Janelas maiores que a série
    disponível são omitidas (não viram 0 — evita contaminar a estatística).
    `janelas_min` em minutos (ex.: [1, 3, 5]).
    """
    _out = {}
    for _var in variaveis:
        # m/min deriva da distância na janela (mesma série, normalizada no fim)
        _base_var = VAR_DIST if _var == VAR_DIST_REL else _var
        _sv = serie_por_amostra(sensor_points, _base_var, hz, **cortes)
        if not _sv:
            continue
        _is_max = (_var == VAR_VMAX)
        for _wmin in janelas_min:
            _n = int(round(_wmin * 60 * hz))
            if _n < 1 or len(_sv) < _n:
                continue
            _val, _, _ = pico_janela(_sv, _n, _is_max)
            if _var == VAR_DIST_REL:
                _val = _val / float(_wmin) if _wmin > 0 else 0.0
            _out[(_var, _wmin)] = round(float(_val), 2)
    return _out
