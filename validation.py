# -*- coding: utf-8 -*-
"""
validation.py — Validação de concordância app × export oficial (OpenField).

Funções PURAS (sem Streamlit) para o módulo de validação embutido: pareia a
tabela gerada pelo app com o CSV oficial exportado do OpenField (mesma
atividade/períodos) e calcula viés, erro %, correlação e Bland-Altman por
variável. Coberto por tests/test_validation.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Versão do esquema (ver nota em metrics.SCHEMA_VERSION).
SCHEMA_VERSION = 2

# Regras de agregação quando um mesmo atleta tem várias linhas (multi-período):
_MAX_COLS = {'Max Acceleration', 'Maximum Velocity (km/h)'}
_MIN_COLS = {'Max Deceleration'}
_RECOMPUTE_MMIN = 'Metros por minuto'   # recalculado = distância / minutos


def athlete_key(name) -> str:
    """Chave de pareamento: última parte do 'Name' ('ATIVIDADE - Atleta')."""
    return str(name).split(' - ')[-1].strip().lower()


def aggregate_by_athlete(df: pd.DataFrame, variables, name_col: str = 'Name') -> pd.DataFrame:
    """Agrega múltiplas linhas do mesmo atleta em uma só.

    Totais são somados; máximos usam max (e Max Deceleration usa min, por ser
    negativo); 'Metros por minuto' é recalculado após a agregação.
    Retorna colunas: _k (chave), Atleta, e as `variables`.
    """
    d = df.copy()
    d['_k'] = d[name_col].map(athlete_key)
    d['Atleta'] = d[name_col].map(lambda s: str(s).split(' - ')[-1].strip())
    variables = [v for v in variables if v in d.columns]
    for v in variables:
        d[v] = pd.to_numeric(d[v], errors='coerce')
    agg = {'Atleta': 'first'}
    for v in variables:
        if v in _MAX_COLS:
            agg[v] = 'max'
        elif v in _MIN_COLS:
            agg[v] = 'min'
        else:
            agg[v] = 'sum'
    g = d.groupby('_k', as_index=False).agg(agg)
    if (_RECOMPUTE_MMIN in variables and 'Total Distance (m)' in variables
            and 'Minutos' in variables):
        _min = g['Minutos'].astype(float)
        g[_RECOMPUTE_MMIN] = np.where(_min > 0,
                                      g['Total Distance (m)'].astype(float) / _min,
                                      0.0)
    return g


def bland_altman(app_vals, off_vals):
    """Estatísticas de Bland-Altman entre duas séries pareadas.

    Retorna dict com mean (média dos pares), diff (app − oficial), bias,
    sd, loa_low e loa_high (±1,96 DP) — ou None com <3 pares válidos.
    """
    a = pd.to_numeric(pd.Series(list(app_vals)), errors='coerce').to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(list(off_vals)), errors='coerce').to_numpy(dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return None
    diff = a - b
    mean = (a + b) / 2.0
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if a.size > 1 else 0.0
    return {
        'mean': [float(x) for x in mean],
        'diff': [float(x) for x in diff],
        'bias': bias,
        'sd': sd,
        'loa_low': bias - 1.96 * sd,
        'loa_high': bias + 1.96 * sd,
        'n': int(a.size),
    }


def comparar_exportacoes(df_app: pd.DataFrame, df_off: pd.DataFrame, variables,
                         name_col: str = 'Name'):
    """Compara a tabela do app com o CSV oficial, atleta a atleta.

    Retorna (merged, stats):
    - merged: 1 linha/atleta com '<var> (app)', '<var> (oficial)', '<var> Δ',
      '<var> Δ%' para cada variável.
    - stats: 1 linha/variável com n, médias, viés (app − oficial), viés %,
      erro médio absoluto % e correlação r.
    """
    variables = [v for v in variables
                 if v in df_app.columns and v in df_off.columns]
    ga = aggregate_by_athlete(df_app, variables, name_col)
    go_ = aggregate_by_athlete(df_off, variables, name_col)
    merged_raw = ga.merge(go_, on='_k', suffixes=(' (app)', ' (oficial)'))
    if merged_raw.empty or not variables:
        return pd.DataFrame(), pd.DataFrame()

    out = pd.DataFrame()
    out['Atleta'] = merged_raw['Atleta (app)']
    stats_rows = []
    for v in variables:
        ca, co = f"{v} (app)", f"{v} (oficial)"
        a = merged_raw[ca].to_numpy(dtype=float)
        b = merged_raw[co].to_numpy(dtype=float)
        out[ca] = np.round(a, 2)
        out[co] = np.round(b, 2)
        diff = a - b
        out[f"{v} Δ"] = np.round(diff, 2)
        with np.errstate(all='ignore'):
            pct = np.where(np.abs(b) > 1e-9, diff / b * 100.0, np.nan)
        out[f"{v} Δ%"] = np.round(pct, 1)

        m = np.isfinite(a) & np.isfinite(b)
        av, bv = a[m], b[m]
        n = int(av.size)
        bias = float(np.mean(av - bv)) if n else np.nan
        mean_off = float(np.mean(bv)) if n else np.nan
        bias_pct = (bias / mean_off * 100.0
                    if n and np.isfinite(mean_off) and abs(mean_off) > 1e-9
                    else np.nan)
        with np.errstate(all='ignore'):
            abs_pct = np.abs(np.where(np.abs(bv) > 1e-9,
                                      (av - bv) / bv * 100.0, np.nan))
        mae_pct = float(np.nanmean(abs_pct)) if n else np.nan
        r = np.nan
        if n >= 3 and np.std(av) > 1e-12 and np.std(bv) > 1e-12:
            r = float(np.corrcoef(av, bv)[0, 1])
        stats_rows.append({
            'Variável': v,
            'n': n,
            'Média app': round(float(np.mean(av)), 2) if n else None,
            'Média oficial': round(mean_off, 2) if n else None,
            'Viés (app − oficial)': round(bias, 3) if n else None,
            'Viés %': round(bias_pct, 2) if np.isfinite(bias_pct) else None,
            'Erro abs. médio %': round(mae_pct, 2) if np.isfinite(mae_pct) else None,
            'r': round(r, 4) if np.isfinite(r) else None,
        })
    return out, pd.DataFrame(stats_rows)
