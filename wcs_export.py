# -*- coding: utf-8 -*-
"""Motor de export WCS multi-atividade (para artigo científico).

Calcula o pico de pior cenário (worst-case scenario) por atleta, variável e
janela temporal (1/3/5 min), no MESMO método canônico da aba WCS:
séries por amostra → `metrics.rolling_sum` + argmax. Funções puras (sem
Streamlit) — cobertas por testes.

Variáveis suportadas (ver VARIAVEIS): distância total, distância em alta
velocidade (HSR) e sprint, PlayerLoad e contagem de acelerações/desacelerações
POR BANDA (B2, B3 e B2+ = B2+B3), classificadas pelo pico da ação.
"""
from __future__ import annotations

from collections import deque

import numpy as np

import metrics as _mtr


# Rótulos das variáveis exportáveis (ordem de exibição/coluna).
# Nota: não há "distância relativa" — a distância JÁ é relativa dentro de uma
# janela de duração fixa (m por 1/3/5 min), então seria a mesma variável.
VAR_DIST      = 'Distância (m)'
VAR_HSR       = 'Distância HSR (m)'
VAR_SPRINT    = 'Distância Sprint (m)'
VAR_PL        = 'PlayerLoad'
# Ações de acel/desacel contadas POR BANDA (classificadas pelo pico da ação).
# "B2+" = B2 + B3 (as duas bandas somadas).
VAR_ACC_B2    = 'Acelerações B2 (n)'
VAR_ACC_B3    = 'Acelerações B3 (n)'
VAR_ACC_B2P   = 'Acelerações B2+ (n)'
VAR_DEC_B2    = 'Desacelerações B2 (n)'
VAR_DEC_B3    = 'Desacelerações B3 (n)'
VAR_DEC_B2P   = 'Desacelerações B2+ (n)'

VARIAVEIS = [VAR_DIST, VAR_HSR, VAR_SPRINT, VAR_PL,
             VAR_ACC_B2, VAR_ACC_B3, VAR_ACC_B2P,
             VAR_DEC_B2, VAR_DEC_B3, VAR_DEC_B2P]

# Cortes padrão (editáveis pela UI); HSR/Sprint em km/h.
DEFAULT_HSR_KMH    = 19.8
DEFAULT_SPRINT_KMH = 25.2

# Bandas de acel/desacel em m/s², definidas pelo usuário do estudo:
# B2 = 2,5–3,5 · B3 = 3,5–10 (desaceleração espelhada, em magnitude).
# Editáveis na UI (expander "Cortes das variáveis"); B2+ = B2 + B3.
DEFAULT_ACC_B2 = (2.5, 3.5)
DEFAULT_ACC_B3 = (3.5, 10.0)
DEFAULT_DEC_B2 = (-3.5, -2.5)
DEFAULT_DEC_B3 = (-10.0, -3.5)

# Variável → bandas que a compõem (chaves resolvidas em serie_por_amostra).
_VAR_BANDAS = {
    VAR_ACC_B2:  ('acc_b2',),
    VAR_ACC_B3:  ('acc_b3',),
    VAR_ACC_B2P: ('acc_b2', 'acc_b3'),
    VAR_DEC_B2:  ('dec_b2',),
    VAR_DEC_B3:  ('dec_b3',),
    VAR_DEC_B2P: ('dec_b2', 'dec_b3'),
}


def _vel_kmh(sensor_points):
    """Velocidade (km/h) por amostra; 'v' do sensor é m/s."""
    return [float(_p.get('v') or 0.0) * 3.6 for _p in sensor_points]


def _acoes_nas_bandas(acc_arr, faixas_alvo, faixas_todas, min_frames, positivo):
    """Frames de início das ações cujo PICO cai em `faixas_alvo`.

    Detalhe metodológico importante: o scan usa SEMPRE o limiar da UNIÃO de
    `faixas_todas` (a banda mais baixa, ex.: B2), e só depois classifica a ação
    pelo pico. Se cada banda fosse escaneada com o próprio limiar, as fronteiras
    da ação mudariam por banda e ações se perderiam (uma ação sustentada acima de
    3 m/s² com pico -5 não sustenta o limiar de B3 nem cai na faixa de B2).
    Assim as fronteiras são idênticas para B2, B3 e B2+, e vale o invariante
    B2+ = B2 + B3. Mesma semântica de `metrics.detect_actions` (classificação
    pelo pico no momento em que a ação se confirma) + saturação na banda extrema,
    para não descartar picos além do limite superior configurado.
    """
    if acc_arr is None or len(acc_arr) == 0 or not faixas_alvo:
        return []
    _thr = min(abs(_lo) if positivo else abs(_hi)
               for _lo, _hi in faixas_todas)
    # Borda extrema do conjunto-alvo (satura: pico além dela ainda conta).
    _ext_alvo = (max(_hi for _, _hi in faixas_alvo) if positivo
                 else min(_lo for _lo, _ in faixas_alvo))
    _ext_todas = (max(_hi for _, _hi in faixas_todas) if positivo
                  else min(_lo for _lo, _ in faixas_todas))
    _alvo_tem_extremo = (_ext_alvo == _ext_todas)

    _starts = []
    _run, _start_i, _peak, _counted = 0, -1, 0.0, False
    for _i in range(len(acc_arr)):
        _v = float(acc_arr[_i])
        _cond = (_v >= _thr) if positivo else (_v <= -_thr)
        if _cond:
            if _run == 0:
                _start_i, _peak = _i, _v
            _run += 1
            if (_v > _peak) if positivo else (_v < _peak):
                _peak = _v
            if _run >= min_frames and not _counted:
                _ok = any(_lo <= _peak < _hi for _lo, _hi in faixas_alvo)
                if not _ok and _alvo_tem_extremo:
                    _ok = (_peak >= _ext_alvo) if positivo else (_peak <= _ext_alvo)
                if _ok:
                    _starts.append(_start_i)
                _counted = True
        else:
            _run, _counted, _peak = 0, False, 0.0
    return _starts


def serie_por_amostra(sensor_points, variavel, hz=10.0, *,
                      hsr_kmh=DEFAULT_HSR_KMH, sprint_kmh=DEFAULT_SPRINT_KMH,
                      acc_b2=DEFAULT_ACC_B2, acc_b3=DEFAULT_ACC_B3,
                      dec_b2=DEFAULT_DEC_B2, dec_b3=DEFAULT_DEC_B3,
                      min_dur_acc_s=0.6):
    """Valor por amostra da `variavel`, pronto para a janela rolante (SOMA).

    Distância/HSR/Sprint: metros contribuídos pela amostra. PlayerLoad: o 'pl'
    do sensor. Ações de acel/desacel por banda: 1.0 no frame de início de cada
    ação, classificada pelo PICO na banda (via metrics.detect_actions — o mesmo
    motor da aba WCS/Neuromuscular). Bandas em m/s², como (min, max).
    """
    if not sensor_points:
        return []

    if variavel == VAR_DIST:
        return [_v / (3.6 * hz) for _v in _vel_kmh(sensor_points)]

    if variavel == VAR_HSR:
        return [(_v / (3.6 * hz)) if _v >= hsr_kmh else 0.0
                for _v in _vel_kmh(sensor_points)]

    if variavel == VAR_SPRINT:
        return [(_v / (3.6 * hz)) if _v >= sprint_kmh else 0.0
                for _v in _vel_kmh(sensor_points)]

    if variavel == VAR_PL:
        # O 'pl' do sensor é ACUMULADO (série não-decrescente). Somar o valor
        # bruto na janela dá números absurdos (ex.: 435 mil num minuto, contra
        # ~550 de PlayerLoad TOTAL do jogo). Usa os INCREMENTOS — mesma detecção
        # de metrics.playerload_total (≥98% não-decrescente → acumulada).
        _pl = []
        for _p in sensor_points:
            try:
                _pl.append(float(_p.get('pl') or 0.0))
            except (TypeError, ValueError):
                _pl.append(0.0)
        _arr = np.asarray(_pl, dtype=float)
        if _arr.size >= 10:
            _d = np.diff(_arr)
            if _arr[-1] > _arr[0] and float((_d >= -1e-6).mean()) > 0.98:
                return [0.0] + np.clip(_d, 0.0, None).tolist()
        return [max(0.0, _v) for _v in _pl]      # já incremental

    if variavel in _VAR_BANDAS:
        _acc = np.asarray([float(_p.get('a') or 0.0) for _p in sensor_points],
                          dtype=float)
        if not _acc.size:
            return []
        # Sem aceleração nativa (campo 'a' ausente ou achatado) as contagens
        # sairiam todas ZERO. Espelha o fallback da aba WCS: deriva a série por
        # dv/dt da velocidade, para não subestimar acel/desacel.
        if not np.any(np.abs(_acc) > 0.05):
            from analysis import acc_series_from_vel as _acc_de_vel
            _vk = _vel_kmh(sensor_points)
            _ts = [float(_p.get('ts') or 0.0) for _p in sensor_points]
            if any(_v > 0.1 for _v in _vk):
                _acc = np.asarray(_acc_de_vel(_vk, _ts, hz), dtype=float)
                if not _acc.size:
                    return []
        _disp = {'acc_b2': acc_b2, 'acc_b3': acc_b3,
                 'dec_b2': dec_b2, 'dec_b3': dec_b3}
        _pos = variavel in (VAR_ACC_B2, VAR_ACC_B3, VAR_ACC_B2P)
        _todas = [_disp['acc_b2'], _disp['acc_b3']] if _pos else \
                 [_disp['dec_b2'], _disp['dec_b3']]
        _alvo = [_disp[_k] for _k in _VAR_BANDAS[variavel]]
        _min_frames = max(1, int(round(float(min_dur_acc_s) * float(hz))))
        _idxs = _acoes_nas_bandas(_acc, _alvo, _todas, _min_frames, _pos)
        _sv = [0.0] * len(_acc)
        for _ix in _idxs:
            if 0 <= _ix < len(_sv):
                _sv[_ix] += 1.0
        return _sv

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
        _sv = serie_por_amostra(sensor_points, _var, hz, **cortes)
        if not _sv:
            continue
        for _wmin in janelas_min:
            _n = int(round(_wmin * 60 * hz))
            if _n < 1 or len(_sv) < _n:
                continue
            _val, _, _ = pico_janela(_sv, _n)
            _out[(_var, _wmin)] = round(float(_val), 2)
    return _out
