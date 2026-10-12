# -*- coding: utf-8 -*-
"""Perfil de aceleração por zona de velocidade de entrada (Entry-Speed / ES).

Implementa o framework "Acceleration Profiling Blueprint": em vez de um limiar
fixo de aceleração (ex.: 3 m/s²) para todo o jogo, cada esforço de aceleração é
classificado pela VELOCIDADE DE ENTRADA do atleta — a velocidade no instante em
que o esforço começa — expressa como % da Vmax individual. Dentro de cada zona,
o corte de validade é derivado do próprio sinal (percentil 95 das acelerações de
pico daquela zona), e a contagem de eventos válidos é ponderada pelo custo
mecânico da zona.

Motivação biomecânica: a capacidade de gerar aceleração decai à medida que a
velocidade de entrada sobe (relação força-velocidade). Acelerar a 2,5 m/s²
partindo de 2 km/h é trivial; a mesma aceleração partindo de 25 km/h está perto
do limite mecânico. Um limiar único trata as duas como o mesmo evento.

Funções puras (sem Streamlit) — cobertas por tests/test_acc_profile.py.
"""
from __future__ import annotations

import numpy as np

# ── Zonas de velocidade de entrada ────────────────────────────────────────────
# (rótulo, piso %Vmax, teto %Vmax, peso, dominância mecânica)
# Limites e pesos conforme o blueprint. São CONVENÇÕES do framework, não
# constantes derivadas de um estudo — ver nota em `PROVENIENCIA`.
ES_ZONAS = (
    ('ES1', 0.00, 0.25,            1.0, 'Força dominante'),
    ('ES2', 0.25, 0.50,            1.5, 'Força-Potência'),
    ('ES3', 0.50, 0.75,            2.0, 'Potência-Velocidade'),
    ('ES4', 0.75, float('inf'),    2.5, 'Velocidade dominante'),
)

PROVENIENCIA = (
    "Os limites das zonas (25/50/75% da Vmax), os pesos (1,0/1,5/2,0/2,5) e o "
    "corte de 50% do P95 são convenções do blueprint de origem, que não traz "
    "referências bibliográficas. O conceito subjacente — decaimento da "
    "capacidade de aceleração conforme sobe a velocidade de entrada — é "
    "consolidado na literatura, mas estes NÚMEROS não são atribuíveis a um "
    "estudo verificado. Tratar como parametrização editável, não como padrão."
)

# Padrões (todos editáveis na UI)
DEFAULT_PERCENTIL   = 95.0   # percentil que define o teto de referência da zona
DEFAULT_PCT_CORTE   = 0.50   # corte = 50% do P95 da zona
DEFAULT_PISO_MS2    = 1.50   # piso só para SEGMENTAR eventos (ver `detectar_eventos`)
DEFAULT_MIN_DUR_S   = 0.50   # duração mínima de um esforço
MIN_EVENTOS_P95     = 5      # abaixo disso o P95 da zona é instável

# Acima disto, a contagem de esforços num jogo é implausível (um jogo costuma
# ficar na casa das dezenas a ~150 acelerações relevantes) — sinal de que o
# piso de segmentação está baixo demais para a qualidade do sinal em questão.
TETO_PLAUSIVEL_EVENTOS = 400


def _as_float_array(x):
    return np.asarray(list(x or []), dtype=float)


def vel_kmh_de_pontos(sensor_points):
    """Velocidade em km/h por amostra ('v' do sensor vem em m/s)."""
    return [float(_p.get('v') or 0.0) * 3.6 for _p in (sensor_points or [])]


def acc_de_pontos(sensor_points, hz=10.0):
    """Aceleração (m/s²) por amostra, com fallback por dv/dt.

    Espelha o fallback de `wcs_export.serie_por_amostra`: se o campo 'a' do
    sensor estiver ausente ou achatado, deriva a aceleração da velocidade —
    sem isso as contagens sairiam todas zero em arquivos sem aceleração nativa.
    """
    _pts = sensor_points or []
    _acc = _as_float_array([_p.get('a') or 0.0 for _p in _pts])
    if _acc.size and np.any(np.abs(_acc) > 0.05):
        return _acc.tolist()
    _vk = vel_kmh_de_pontos(_pts)
    if not any(_v > 0.1 for _v in _vk):
        return _acc.tolist()
    from analysis import acc_series_from_vel as _acc_de_vel
    _ts = [float(_p.get('ts') or 0.0) for _p in _pts]
    return list(_acc_de_vel(_vk, _ts, hz) or [])


def vmax_kmh(vel_kmh, percentil=100.0):
    """Vmax individual (km/h). percentil=100 → máximo observado.

    Um percentil < 100 (ex.: 99,9) protege contra spike isolado de GPS, que
    inflaria a Vmax e deslocaria TODAS as zonas para baixo.
    """
    _v = _as_float_array(vel_kmh)
    if not _v.size:
        return 0.0
    _v = _v[np.isfinite(_v)]
    if not _v.size:
        return 0.0
    _p = float(percentil)
    if _p >= 100.0:
        return float(_v.max())
    return float(np.percentile(_v, _p))


def detectar_eventos(vel_kmh, acc_ms2, hz=10.0, piso_ms2=DEFAULT_PISO_MS2,
                     min_dur_s=DEFAULT_MIN_DUR_S):
    """Eventos de aceleração discretos, com velocidade de entrada e pico.

    Um evento = trecho em que a aceleração se mantém >= `piso_ms2` por ao menos
    `min_dur_s`. Eventos são naturalmente NÃO sobrepostos (são trechos
    contíguos distintos do sinal).

    Sobre o `piso_ms2`: o blueprint diz "coletar todos os eventos, sem cortes
    iniciais". Na prática é preciso ALGUM piso para delimitar onde um esforço
    começa e termina — com piso zero o jogo inteiro seria um único evento. Ele
    só SEGMENTA; quem filtra de fato é o corte derivado do percentil da zona.

    O padrão (1,5 m/s²) foi calibrado por simulação: com piso 0,5 um jogo de
    90 min gerava ~1.700 "eventos" (contra ~50-150 acelerações relevantes
    reais), porque cada micro-oscilação do sinal virava evento. O piso é o
    parâmetro mais sensível desta análise — calibre-o comparando a contagem
    total com os efforts de aceleração que o próprio OpenField reporta para a
    mesma atividade.

    Retorna lista de dicts: frame_ini, frame_fim, dur_s, pico, vel_entrada.
    """
    _a = _as_float_array(acc_ms2)
    _v = _as_float_array(vel_kmh)
    if not _a.size:
        return []
    _n = min(_a.size, _v.size) if _v.size else _a.size
    if _n < 1:
        return []
    _thr = float(piso_ms2)
    _minf = max(1, int(round(float(min_dur_s) * float(hz))))

    _ev = []
    _ini, _run, _pico = -1, 0, 0.0
    for _i in range(_n):
        _x = float(_a[_i])
        if _x >= _thr:
            if _run == 0:
                _ini, _pico = _i, _x
            _run += 1
            if _x > _pico:
                _pico = _x
        else:
            if _run >= _minf and _ini >= 0:
                _ev.append({
                    'frame_ini': _ini, 'frame_fim': _i,
                    'dur_s': round(_run / float(hz), 2),
                    'pico': round(_pico, 3),
                    'vel_entrada': round(float(_v[_ini]), 2) if _v.size else 0.0,
                })
            _ini, _run, _pico = -1, 0, 0.0
    if _run >= _minf and _ini >= 0:          # evento aberto no fim da série
        _ev.append({
            'frame_ini': _ini, 'frame_fim': _n,
            'dur_s': round(_run / float(hz), 2),
            'pico': round(_pico, 3),
            'vel_entrada': round(float(_v[_ini]), 2) if _v.size else 0.0,
        })
    return _ev


def zona_de_entrada(vel_entrada_kmh, vmax):
    """Rótulo da zona ES a que pertence uma velocidade de entrada."""
    if not vmax or float(vmax) <= 0:
        return None
    _frac = float(vel_entrada_kmh) / float(vmax)
    for _rot, _lo, _hi, _peso, _dom in ES_ZONAS:
        if _lo <= _frac < _hi:
            return _rot
    return ES_ZONAS[-1][0] if _frac >= ES_ZONAS[-1][1] else None


def perfil_atleta(sensor_points, hz=10.0, *, vmax=None,
                  percentil=DEFAULT_PERCENTIL, pct_corte=DEFAULT_PCT_CORTE,
                  piso_ms2=DEFAULT_PISO_MS2, min_dur_s=DEFAULT_MIN_DUR_S,
                  vmax_percentil=100.0):
    """Perfil ES completo de UM atleta num escopo (período ou partida inteira).

    Pipeline (nós 1 a 6 do blueprint):
      1. Vmax individual (do próprio sinal, ou fornecida em `vmax`)
      2. zonas ES por % da Vmax
      3. coleta os eventos de aceleração e os distribui nas zonas
      4. P95 das acelerações de pico DENTRO de cada zona → corte = pct × P95
      5. conta eventos válidos (pico >= corte) → score = válidos × peso
      6. ES1..ES4 + carga total ponderada

    Retorna dict com 'vmax', 'zonas' ({rótulo: {...}}), 'total', 'n_eventos'.
    """
    _vk = vel_kmh_de_pontos(sensor_points)
    _ac = acc_de_pontos(sensor_points, hz)
    _vm = float(vmax) if vmax else vmax_kmh(_vk, vmax_percentil)
    _eventos = detectar_eventos(_vk, _ac, hz, piso_ms2, min_dur_s)

    _por_zona = {_z[0]: [] for _z in ES_ZONAS}
    for _e in _eventos:
        _z = zona_de_entrada(_e['vel_entrada'], _vm)
        if _z:
            _por_zona[_z].append(_e)

    _zonas, _total = {}, 0.0
    for _rot, _lo, _hi, _peso, _dom in ES_ZONAS:
        _evs = _por_zona[_rot]
        _picos = _as_float_array([_e['pico'] for _e in _evs])
        if _picos.size:
            _p95 = float(np.percentile(_picos, float(percentil)))
            _corte = float(pct_corte) * _p95
            _validos = [_e for _e in _evs if _e['pico'] >= _corte]
        else:
            _p95, _corte, _validos = 0.0, 0.0, []
        _score = len(_validos) * float(_peso)
        _total += _score
        _zonas[_rot] = {
            'zona': _rot,
            'faixa_pct': (_lo, _hi),
            'faixa_kmh': (round(_lo * _vm, 1),
                          (round(_hi * _vm, 1) if np.isfinite(_hi) else None)),
            'dominancia': _dom,
            'peso': float(_peso),
            'n_total': len(_evs),
            'p95': round(_p95, 3),
            'corte': round(_corte, 3),
            'n_validos': len(_validos),
            'score': round(_score, 2),
            'pico_max': round(float(_picos.max()), 3) if _picos.size else 0.0,
            'confiavel': len(_evs) >= MIN_EVENTOS_P95,
            'eventos': _evs,
            'validos': _validos,
        }
    return {
        'vmax': round(_vm, 2),
        'zonas': _zonas,
        'total': round(_total, 2),
        'n_eventos': len(_eventos),
    }


def perfil_equipe(dados_sensor, atletas=None, hz=10.0, *, vmax_por_atleta=None,
                  **kw):
    """Perfil ES de TODOS os atletas de um escopo já carregado.

    dados_sensor: {atleta: [pontos]} (um período) — para a partida inteira,
    encadeie os períodos antes (ver analysis.combinar_periodos_continuo).
    vmax_por_atleta: {atleta: vmax_kmh} para fixar a Vmax de referência (ex.:
    a maior do histórico do atleta, em vez da do jogo).

    Retorna {atleta: perfil_atleta(...)}.
    """
    _ds = dados_sensor or {}
    _alvos = list(atletas) if atletas else sorted(_ds.keys())
    _vmx = vmax_por_atleta or {}
    _out = {}
    for _a in _alvos:
        _pts = _ds.get(_a) or []
        if not _pts:
            continue
        _out[_a] = perfil_atleta(_pts, hz, vmax=_vmx.get(_a), **kw)
    return _out


def linhas_tabela(perfis):
    """Achata {atleta: perfil} em linhas tidy (uma por atleta × zona)."""
    _rows = []
    for _atl, _p in (perfis or {}).items():
        for _rot, _lo, _hi, _peso, _dom in ES_ZONAS:
            _z = _p['zonas'][_rot]
            _rows.append({
                'Atleta': _atl,
                'Vmax (km/h)': _p['vmax'],
                'Zona': _rot,
                'Entrada (%Vmax)': (f"{int(_lo*100)}–{int(_hi*100)}%"
                                    if np.isfinite(_hi) else f">{int(_lo*100)}%"),
                'Entrada (km/h)': (f"{_z['faixa_kmh'][0]}–{_z['faixa_kmh'][1]}"
                                   if _z['faixa_kmh'][1] is not None
                                   else f">{_z['faixa_kmh'][0]}"),
                'Dominância': _dom,
                'Eventos': _z['n_total'],
                f"P{int(DEFAULT_PERCENTIL)} (m/s²)": _z['p95'],
                'Corte (m/s²)': _z['corte'],
                'Válidos': _z['n_validos'],
                'Peso': _z['peso'],
                'Score': _z['score'],
            })
    return _rows


def linhas_resumo(perfis):
    """Uma linha por atleta: scores por zona + carga total ponderada."""
    _rows = []
    for _atl, _p in (perfis or {}).items():
        _r = {'Atleta': _atl, 'Vmax (km/h)': _p['vmax']}
        for _rot, _, _, _, _ in ES_ZONAS:
            _r[f'{_rot} score'] = _p['zonas'][_rot]['score']
        for _rot, _, _, _, _ in ES_ZONAS:
            _r[f'{_rot} válidos'] = _p['zonas'][_rot]['n_validos']
        _r['Carga total ponderada'] = _p['total']
        _r['Eventos totais'] = _p['n_eventos']
        _rows.append(_r)
    return sorted(_rows, key=lambda _x: _x['Carga total ponderada'], reverse=True)
