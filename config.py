# -*- coding: utf-8 -*-
"""Constantes de configuração (P4 — extraído do monólito).

Dados puros: servidores da API, i18n (LANGUAGES), bandas de velocidade/
aceleração padrão, mapeamentos Gen2 e config de eventos de futebol. Sem lógica,
sem Streamlit — importado de volta pelo app pelos mesmos nomes.
"""

# Chave do período sintético "combinado" (soma de todos os períodos).
_CHAVE_COMBINADO = '📊 Períodos Combinados'

# Durações mínimas padrão de esforço (s).
_DEFAULT_MIN_DUR_S     = 0.6   # acc / dec
_DEFAULT_MIN_DUR_VEL_S = 1.0   # esforços de velocidade

SERVERS = {
    "Américas (US)": "https://connect-us.catapultsports.com/api/v6",
    "Europa/Oriente Médio/África (EU)": "https://connect-eu.catapultsports.com/api/v6",
    "Ásia-Pacífico (AU)": "https://connect-au.catapultsports.com/api/v6",
    "China (CN)": "https://connect-cn.catapultsports-cn.com/api/v6",
}

LANGUAGES = {
    "🇧🇷 Português (Brasil)": "pt",
    "🇺🇸 English (US)":       "en",
    "🇲🇽 Español (Latino)":   "es",
    "🇫🇷 Français":           "fr",
}

BANDAS_VEL = {
    1: {'label': 'B1 — 0-7 km/h (Caminhada)',           'min': 0,     'max': 7,     'color': '#2196F3'},
    2: {'label': 'B2 — 7-14.4 km/h (Trote)',            'min': 7,     'max': 14.4,  'color': '#4CAF50'},
    3: {'label': 'B3 — 14.4-19.8 km/h (Corrida)',       'min': 14.4,  'max': 19.8,  'color': '#CDDC39'},
    4: {'label': 'B4 — 19.8-25.2 km/h (Corrida Intensa)', 'min': 19.8, 'max': 25.2, 'color': '#FF9800'},
    5: {'label': 'B5 — 25.2-29.9 km/h (Alta Velocidade)', 'min': 25.2, 'max': 29.9, 'color': '#FF5722'},
    6: {'label': 'B6 — 29.9-45 km/h (Sprint)',          'min': 29.9,  'max': 45,    'color': '#F44336'},
}

BANDAS_ACC = {
    'A1': {'label': 'Aceleração B1 — 2 a 3 m/s²',     'min': 2,    'max': 3,   'color': '#69F0AE'},
    'A2': {'label': 'Aceleração B2 — 3 a 4 m/s²',     'min': 3,    'max': 4,   'color': '#43A047'},
    'A3': {'label': 'Aceleração B3 — 4 a 10 m/s²',    'min': 4,    'max': 10,  'color': '#00C853'},
    'D1': {'label': 'Desaceleração B1 — -3 a -2 m/s²', 'min': -3,  'max': -2,  'color': '#FFD180'},
    'D2': {'label': 'Desaceleração B2 — -4 a -3 m/s²', 'min': -4,  'max': -3,  'color': '#FF6D00'},
    'D3': {'label': 'Desaceleração B3 — -10 a -4 m/s²','min': -10, 'max': -4,  'color': '#B71C1C'},
}

_DEFAULT_VELOCITY_ZONES = [
    {'name': 'B1 — Caminhada',        'min_ms': 0/3.6,     'max_ms': 7/3.6,    'color': '#2196F3'},
    {'name': 'B2 — Trote',            'min_ms': 7/3.6,     'max_ms': 14.4/3.6, 'color': '#4CAF50'},
    {'name': 'B3 — Corrida',          'min_ms': 14.4/3.6,  'max_ms': 19.8/3.6, 'color': '#CDDC39'},
    {'name': 'B4 — Corrida Intensa',  'min_ms': 19.8/3.6,  'max_ms': 25.2/3.6, 'color': '#FF9800'},
    {'name': 'B5 — Alta Velocidade',  'min_ms': 25.2/3.6,  'max_ms': 29.9/3.6, 'color': '#FF5722'},
    {'name': 'B6 — Sprint',           'min_ms': 29.9/3.6,  'max_ms': 45/3.6,   'color': '#F44336'},
]

_DEFAULT_ACCELERATION_ZONES = [
    {'name': 'Aceleração B1',    'min_ms2': 2.0,   'max_ms2': 3.0,   'color': '#69F0AE'},
    {'name': 'Aceleração B2',    'min_ms2': 3.0,   'max_ms2': 4.0,   'color': '#43A047'},
    {'name': 'Aceleração B3',    'min_ms2': 4.0,   'max_ms2': 10.0,  'color': '#00C853'},
    {'name': 'Desaceleração B1', 'min_ms2': -3.0,  'max_ms2': -2.0,  'color': '#FFD180'},
    {'name': 'Desaceleração B2', 'min_ms2': -4.0,  'max_ms2': -3.0,  'color': '#FF6D00'},
    {'name': 'Desaceleração B3', 'min_ms2': -10.0, 'max_ms2': -4.0,  'color': '#B71C1C'},
]

_ZONES_SCHEMA_VERSION = "2026-06-04-acc6"

_NOMES_BANDA_VEL_DEFAULT = {
    1: 'Caminhada', 2: 'Trote', 3: 'Corrida',
    4: 'Corrida Intensa', 5: 'Alta Velocidade', 6: 'Sprint',
}

_CORES_BANDA_VEL_DEFAULT = {
    1: '#2196F3', 2: '#4CAF50', 3: '#CDDC39',
    4: '#FF9800', 5: '#FF5722', 6: '#F44336',
}

_ACC_BAND_MAP = {6: 'A1', 7: 'A2', 8: 'A3',
                 3: 'D1', 2: 'D2', 1: 'D3'}

FUTEBOL_EVENTS_CONFIG = {
    'football_kick': {
        'label': '🟡 Chute',
        'color': '#FFEB3B', 'marker': 'star', 'size': 14,
        'attrs': ['confidence', 'class'],
    },
    'football_header': {
        'label': '🔵 Cabeceio',
        'color': '#2196F3', 'marker': 'diamond', 'size': 13,
        'attrs': ['confidence'],
    },
    'football_tackle': {
        'label': '🔴 Disputa/Tackle',
        'color': '#F44336', 'marker': 'x', 'size': 12,
        'attrs': ['confidence', 'duration'],
    },
    'football_cross': {
        'label': '🟠 Cruzamento',
        'color': '#FF9800', 'marker': 'triangle-up', 'size': 13,
        'attrs': ['confidence', 'class'],
    },
    'ima_impact': {
        'label': '⚪ Impacto (IMA)',
        'color': '#90CAF9', 'marker': 'diamond', 'size': 11,
        'attrs': ['impact', 'direction'],
    },
    'ima_jump': {
        'label': '🟢 Salto (IMA)',
        'color': '#4CAF50', 'marker': 'triangle-up', 'size': 11,
        'attrs': ['height'],
    },
    'running_symmetry': {
        'label': '🔄 Simetria de Corrida',
        'color': '#CE93D8', 'marker': 'circle', 'size': 10,
        'attrs': ['left_right_ratio', 'asymmetry_index'],
    },
    'ima_acceleration': {
        'label': '⚡ Acc. Explosiva (IMA)',
        'color': '#FFD54F', 'marker': 'arrow-up', 'size': 12,
        'attrs': ['direction', 'magnitude'],
    },
}
