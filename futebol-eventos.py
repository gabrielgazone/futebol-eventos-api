# futebol_eventos_completo_final.py
# PARTE 1 - IMPORTS, CONSTANTES E CLASSE API

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import base64
import io
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter as _gf
from scipy.spatial import cKDTree
import folium
from streamlit_folium import st_folium
import os as _os

st.set_page_config(page_title="Futebol Eventos - Catapult", layout="wide")

# Componente bidirecional para o mapa interativo de posicionamento de campo
_CAMPO_COMP_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_campo_component")
_campo_component = st.components.v1.declare_component("campo_interativo_v1", path=_CAMPO_COMP_DIR)

SERVERS = {
    "Américas (US)": "https://connect-us.catapultsports.com/api/v6",
    "Europa/África (EU)": "https://connect-eu.catapultsports.com/api/v6",
    "Ásia-Pacífico (AU)": "https://connect-au.catapultsports.com/api/v6",
}

# ==================== SISTEMA DE IDIOMAS ====================
LANGUAGES = {
    "🇧🇷 Português (Brasil)": "pt",
    "🇺🇸 English (US)":       "en",
    "🇲🇽 Español (Latino)":   "es",
    "🇫🇷 Français":           "fr",
}

TRANSLATIONS = {
    "pt": {
        "app_title":           "⚽ Futebol Eventos - Catapult Sports",
        "app_subtitle":        "Análise de Performance com Filtros por Equipe, Posição e Período",
        "server_header":       "🌍 Servidor",
        "server_select":       "Selecione:",
        "language_header":     "🌐 Idioma",
        "language_select":     "Selecione o idioma:",
        "token_header":        "🔐 Token",
        "token_label":         "Token JWT:",
        "load_btn":            "🔄 Carregar Dados",
        "loading":             "Carregando...",
        "loading_teams":       "📋 Carregando Equipes...",
        "loading_athletes":    "📋 Carregando Atletas...",
        "loading_activities":  "📋 Carregando Atividades...",
        "loading_positions":   "📋 Carregando Posições...",
        "mapping_athletes":    "📋 Mapeando atletas por equipe...",
        "filters_header":      "🎯 Filtros",
        "team_filter":         "🏢 Filtrar por Equipe",
        "select_teams":        "Selecione as equipes:",
        "position_filter":     "🎯 Filtrar por Posição",
        "select_positions":    "Selecione as posições:",
        "activity_header":     "📅 Atividade",
        "select_activity":     "Selecione a atividade:",
        "periods_header":      "📊 Selecionar Período(s)",
        "select_periods":      "Selecione um ou mais períodos para análise:",
        "search_btn":          "🔍 Buscar Atletas da Atividade",
        "athletes_header":     "🏃 Selecionar Atletas",
        "select_athletes":     "Selecione os atletas para análise:",
        "events_header":       "⚽ Eventos Futebol",
        "select_all":          "Selecionar todos",
        "event_types":         "Tipos de evento:",
        "tab_charts":          "📈 Gráficos Comparativos",
        "tab_field":           "🗺️ Campo de Futebol",
        "tab_efforts":         "⏱️ Esforços ao Longo do Tempo",
        "tab_windows":         "📊 Janelas Temporais Móveis",
        "metric_athletes":     "🏃 Atletas",
        "metric_distance":     "📏 Distância Total",
        "metric_pl":           "⚡ PlayerLoad Total",
        "metric_maxspeed":     "💨 Velocidade Máx",
        "compare_athletes":    "Comparar Atletas no Mesmo Período",
        "compare_periods":     "Comparar Mesmo Atleta em Diferentes Períodos",
        "field_title":         "🗺️ Campo de Futebol — Análise de Movimentação",
        "phase1_title":        "### 1️⃣ Posicionamento no Campo Físico",
        "phase1_desc":         "Ajuste o campo de futebol sobre a imagem de satélite e clique **✅ Aplicar Campo** no painel inferior do mapa.",
        "phase1_info":         "📌 **Edite Lat/Lon** (ou use ↑↓) para mover o ⊙ amarelo · ⚽ **Mostrar Campo** para ativar o overlay · Sliders ajustam rotação e dimensões **em tempo real** · **✅ Aplicar Campo** quando estiver satisfeito",
        "no_gps_warning":      "⚠️ Nenhum ponto GPS real (lat/lon) encontrado para este atleta.\n\nIsso pode ocorrer se o sensor não obteve lock GPS durante a sessão.",
        "phase2_title":        "### 2️⃣ Análise de Esforços no Campo",
        "phase3_title":        "### 3️⃣ Análise de Movimentação no Campo",
        "readjust_btn":        "🔄 Reajustar",
        "overlay_events":      "⚽ Eventos Futebol",
        "timeline_title":      "### ⚽ Timeline Técnico-Físico",
        "fatigue_title":       "### 💪 Fadiga & Recuperação por Evento",
        "velocity_graph":      "### 🏃‍♂️ Velocidade ao Longo do Tempo",
        "accel_graph":         "### 🔄 Aceleração ao Longo do Tempo",
        "efforts_table":       "## 📋 Tabela de Esforços",
        "no_data":             "Dados de sensor não disponíveis",
        "select_athlete_msg":  "Selecione um atleta para análise",
        "no_position_data":    "Dados de posição não disponíveis. Verifique se o sensor GPS estava ativo durante a sessão.",
        "select_period_ath":   "Selecione um período e atleta",
        "no_events_loaded":    "Nenhum evento futebol carregado para este atleta. Ative os eventos na sidebar e recarregue os dados.",
        "no_events_sidebar":   "Nenhum evento futebol carregado. Recarregue os dados com eventos ativados na sidebar.",
        "no_events_types":     "Nenhum evento futebol carregado para este atleta/período.",
    },
    "en": {
        "app_title":           "⚽ Football Events - Catapult Sports",
        "app_subtitle":        "Performance Analysis with Team, Position and Period Filters",
        "server_header":       "🌍 Server",
        "server_select":       "Select:",
        "language_header":     "🌐 Language",
        "language_select":     "Select language:",
        "token_header":        "🔐 Token",
        "token_label":         "JWT Token:",
        "load_btn":            "🔄 Load Data",
        "loading":             "Loading...",
        "loading_teams":       "📋 Loading Teams...",
        "loading_athletes":    "📋 Loading Athletes...",
        "loading_activities":  "📋 Loading Activities...",
        "loading_positions":   "📋 Loading Positions...",
        "mapping_athletes":    "📋 Mapping athletes by team...",
        "filters_header":      "🎯 Filters",
        "team_filter":         "🏢 Filter by Team",
        "select_teams":        "Select teams:",
        "position_filter":     "🎯 Filter by Position",
        "select_positions":    "Select positions:",
        "activity_header":     "📅 Activity",
        "select_activity":     "Select activity:",
        "periods_header":      "📊 Select Period(s)",
        "select_periods":      "Select one or more periods for analysis:",
        "search_btn":          "🔍 Search Activity Athletes",
        "athletes_header":     "🏃 Select Athletes",
        "select_athletes":     "Select athletes for analysis:",
        "events_header":       "⚽ Football Events",
        "select_all":          "Select all",
        "event_types":         "Event types:",
        "tab_charts":          "📈 Comparative Charts",
        "tab_field":           "🗺️ Football Pitch",
        "tab_efforts":         "⏱️ Efforts Over Time",
        "tab_windows":         "📊 Moving Time Windows",
        "metric_athletes":     "🏃 Athletes",
        "metric_distance":     "📏 Total Distance",
        "metric_pl":           "⚡ Total PlayerLoad",
        "metric_maxspeed":     "💨 Max Speed",
        "compare_athletes":    "Compare Athletes in Same Period",
        "compare_periods":     "Compare Same Athlete Across Periods",
        "field_title":         "🗺️ Football Pitch — Movement Analysis",
        "phase1_title":        "### 1️⃣ Physical Pitch Positioning",
        "phase1_desc":         "Adjust the football pitch over the satellite image and click **✅ Apply Pitch** in the map panel.",
        "phase1_info":         "📌 **Edit Lat/Lon** (or use ↑↓) to move the ⊙ marker · ⚽ **Show Pitch** to activate overlay · Sliders adjust rotation and dimensions **in real time** · **✅ Apply Pitch** when satisfied",
        "no_gps_warning":      "⚠️ No real GPS points (lat/lon) found for this athlete.\n\nThis may occur if the sensor did not acquire GPS lock during the session.",
        "phase2_title":        "### 2️⃣ Effort Analysis on Pitch",
        "phase3_title":        "### 3️⃣ Movement Analysis on Pitch",
        "readjust_btn":        "🔄 Readjust",
        "overlay_events":      "⚽ Football Events",
        "timeline_title":      "### ⚽ Technical-Physical Timeline",
        "fatigue_title":       "### 💪 Fatigue & Recovery by Event",
        "velocity_graph":      "### 🏃‍♂️ Velocity Over Time",
        "accel_graph":         "### 🔄 Acceleration Over Time",
        "efforts_table":       "## 📋 Efforts Table",
        "no_data":             "Sensor data not available",
        "select_athlete_msg":  "Select an athlete for analysis",
        "no_position_data":    "Position data not available. Check that the GPS sensor was active during the session.",
        "select_period_ath":   "Select a period and athlete",
        "no_events_loaded":    "No football events loaded for this athlete. Enable events in the sidebar and reload data.",
        "no_events_sidebar":   "No football events loaded. Reload data with events enabled in the sidebar.",
        "no_events_types":     "No football events loaded for this athlete/period.",
    },
    "es": {
        "app_title":           "⚽ Fútbol Eventos - Catapult Sports",
        "app_subtitle":        "Análisis de Rendimiento con Filtros por Equipo, Posición y Período",
        "server_header":       "🌍 Servidor",
        "server_select":       "Seleccione:",
        "language_header":     "🌐 Idioma",
        "language_select":     "Seleccione el idioma:",
        "token_header":        "🔐 Token",
        "token_label":         "Token JWT:",
        "load_btn":            "🔄 Cargar Datos",
        "loading":             "Cargando...",
        "loading_teams":       "📋 Cargando Equipos...",
        "loading_athletes":    "📋 Cargando Atletas...",
        "loading_activities":  "📋 Cargando Actividades...",
        "loading_positions":   "📋 Cargando Posiciones...",
        "mapping_athletes":    "📋 Mapeando atletas por equipo...",
        "filters_header":      "🎯 Filtros",
        "team_filter":         "🏢 Filtrar por Equipo",
        "select_teams":        "Seleccione los equipos:",
        "position_filter":     "🎯 Filtrar por Posición",
        "select_positions":    "Seleccione las posiciones:",
        "activity_header":     "📅 Actividad",
        "select_activity":     "Seleccione la actividad:",
        "periods_header":      "📊 Seleccionar Período(s)",
        "select_periods":      "Seleccione uno o más períodos para análisis:",
        "search_btn":          "🔍 Buscar Atletas de la Actividad",
        "athletes_header":     "🏃 Seleccionar Atletas",
        "select_athletes":     "Seleccione los atletas para análisis:",
        "events_header":       "⚽ Eventos de Fútbol",
        "select_all":          "Seleccionar todos",
        "event_types":         "Tipos de evento:",
        "tab_charts":          "📈 Gráficos Comparativos",
        "tab_field":           "🗺️ Cancha de Fútbol",
        "tab_efforts":         "⏱️ Esfuerzos en el Tiempo",
        "tab_windows":         "📊 Ventanas Temporales",
        "metric_athletes":     "🏃 Atletas",
        "metric_distance":     "📏 Distancia Total",
        "metric_pl":           "⚡ PlayerLoad Total",
        "metric_maxspeed":     "💨 Vel. Máxima",
        "compare_athletes":    "Comparar Atletas en el Mismo Período",
        "compare_periods":     "Comparar el Mismo Atleta en Distintos Períodos",
        "field_title":         "🗺️ Cancha de Fútbol — Análisis de Movimiento",
        "phase1_title":        "### 1️⃣ Posicionamiento en la Cancha",
        "phase1_desc":         "Ajuste la cancha de fútbol sobre la imagen satelital y haga clic en **✅ Aplicar Cancha** en el panel del mapa.",
        "phase1_info":         "📌 **Edite Lat/Lon** (o use ↑↓) para mover el ⊙ amarillo · ⚽ **Mostrar Cancha** para activar el overlay · Sliders ajustan rotación y dimensiones **en tiempo real** · **✅ Aplicar Cancha** cuando esté listo",
        "no_gps_warning":      "⚠️ No se encontraron puntos GPS reales (lat/lon) para este atleta.\n\nEsto puede ocurrir si el sensor no obtuvo señal GPS durante la sesión.",
        "phase2_title":        "### 2️⃣ Análisis de Esfuerzos en la Cancha",
        "phase3_title":        "### 3️⃣ Análisis de Movimiento en la Cancha",
        "readjust_btn":        "🔄 Reajustar",
        "overlay_events":      "⚽ Eventos de Fútbol",
        "timeline_title":      "### ⚽ Timeline Técnico-Físico",
        "fatigue_title":       "### 💪 Fatiga y Recuperación por Evento",
        "velocity_graph":      "### 🏃‍♂️ Velocidad en el Tiempo",
        "accel_graph":         "### 🔄 Aceleración en el Tiempo",
        "efforts_table":       "## 📋 Tabla de Esfuerzos",
        "no_data":             "Datos del sensor no disponibles",
        "select_athlete_msg":  "Seleccione un atleta para análisis",
        "no_position_data":    "Datos de posición no disponibles. Verifique que el sensor GPS estaba activo durante la sesión.",
        "select_period_ath":   "Seleccione un período y atleta",
        "no_events_loaded":    "No hay eventos de fútbol cargados para este atleta. Active los eventos en la barra lateral y recargue los datos.",
        "no_events_sidebar":   "No hay eventos de fútbol cargados. Recargue los datos con eventos activados en la barra lateral.",
        "no_events_types":     "No hay eventos de fútbol cargados para este atleta/período.",
    },
    "fr": {
        "app_title":           "⚽ Football Événements - Catapult Sports",
        "app_subtitle":        "Analyse de Performance avec Filtres par Équipe, Position et Période",
        "server_header":       "🌍 Serveur",
        "server_select":       "Sélectionnez:",
        "language_header":     "🌐 Langue",
        "language_select":     "Sélectionnez la langue:",
        "token_header":        "🔐 Token",
        "token_label":         "Token JWT:",
        "load_btn":            "🔄 Charger les Données",
        "loading":             "Chargement...",
        "loading_teams":       "📋 Chargement des Équipes...",
        "loading_athletes":    "📋 Chargement des Athlètes...",
        "loading_activities":  "📋 Chargement des Activités...",
        "loading_positions":   "📋 Chargement des Positions...",
        "mapping_athletes":    "📋 Mapping des athlètes par équipe...",
        "filters_header":      "🎯 Filtres",
        "team_filter":         "🏢 Filtrer par Équipe",
        "select_teams":        "Sélectionnez les équipes:",
        "position_filter":     "🎯 Filtrer par Position",
        "select_positions":    "Sélectionnez les positions:",
        "activity_header":     "📅 Activité",
        "select_activity":     "Sélectionnez l'activité:",
        "periods_header":      "📊 Sélectionner Période(s)",
        "select_periods":      "Sélectionnez une ou plusieurs périodes pour l'analyse:",
        "search_btn":          "🔍 Chercher Athlètes de l'Activité",
        "athletes_header":     "🏃 Sélectionner Athlètes",
        "select_athletes":     "Sélectionnez les athlètes pour l'analyse:",
        "events_header":       "⚽ Événements Football",
        "select_all":          "Sélectionner tout",
        "event_types":         "Types d'événement:",
        "tab_charts":          "📈 Graphiques Comparatifs",
        "tab_field":           "🗺️ Terrain de Football",
        "tab_efforts":         "⏱️ Efforts dans le Temps",
        "tab_windows":         "📊 Fenêtres Temporelles",
        "metric_athletes":     "🏃 Athlètes",
        "metric_distance":     "📏 Distance Totale",
        "metric_pl":           "⚡ PlayerLoad Total",
        "metric_maxspeed":     "💨 Vitesse Max",
        "compare_athletes":    "Comparer Athlètes dans la Même Période",
        "compare_periods":     "Comparer le Même Athlète sur Différentes Périodes",
        "field_title":         "🗺️ Terrain de Football — Analyse de Déplacement",
        "phase1_title":        "### 1️⃣ Positionnement sur le Terrain",
        "phase1_desc":         "Ajustez le terrain de football sur l'image satellite et cliquez **✅ Appliquer Terrain** dans le panneau de la carte.",
        "phase1_info":         "📌 **Éditez Lat/Lon** (ou utilisez ↑↓) pour déplacer le ⊙ jaune · ⚽ **Afficher Terrain** pour activer l'overlay · Sliders pour rotation/dimensions **en temps réel** · **✅ Appliquer Terrain** quand c'est bon",
        "no_gps_warning":      "⚠️ Aucun point GPS réel (lat/lon) trouvé pour cet athlète.\n\nCela peut se produire si le capteur n'a pas obtenu de verrouillage GPS pendant la session.",
        "phase2_title":        "### 2️⃣ Analyse des Efforts sur le Terrain",
        "phase3_title":        "### 3️⃣ Analyse de Déplacement sur le Terrain",
        "readjust_btn":        "🔄 Réajuster",
        "overlay_events":      "⚽ Événements Football",
        "timeline_title":      "### ⚽ Timeline Technico-Physique",
        "fatigue_title":       "### 💪 Fatigue & Récupération par Événement",
        "velocity_graph":      "### 🏃‍♂️ Vitesse dans le Temps",
        "accel_graph":         "### 🔄 Accélération dans le Temps",
        "efforts_table":       "## 📋 Tableau des Efforts",
        "no_data":             "Données du capteur non disponibles",
        "select_athlete_msg":  "Sélectionnez un athlète pour l'analyse",
        "no_position_data":    "Données de position non disponibles. Vérifiez que le capteur GPS était actif pendant la session.",
        "select_period_ath":   "Sélectionnez une période et un athlète",
        "no_events_loaded":    "Aucun événement football chargé pour cet athlète. Activez les événements dans la barre latérale et rechargez les données.",
        "no_events_sidebar":   "Aucun événement football chargé. Rechargez les données avec les événements activés dans la barre latérale.",
        "no_events_types":     "Aucun événement football chargé pour cet athlète/période.",
    },
}

def t(key):
    """Retorna string traduzida para o idioma selecionado."""
    lang_display = st.session_state.get("lang_selector", "🇧🇷 Português (Brasil)")
    lang = LANGUAGES.get(lang_display, "pt")
    return TRANSLATIONS.get(lang, TRANSLATIONS["pt"]).get(key, TRANSLATIONS["pt"].get(key, key))

# ==================== REFERÊNCIAS BIBLIOGRÁFICAS ====================
REFERENCIAS = {
    "janelas": """
    **Referência:** Aughey, R.J. (2011). "Applications of GPS technologies to field sports". 
    *International Journal of Sports Physiology and Performance*, 6(3), 295-310.
    """,
    "campo": """
    **Referência:** FIFA/IFAB (2023). "Regra 1: O Campo de Jogo". Regras do Jogo de Futebol.
    Campo oficial FIFA: 105m de comprimento × 68m de largura (partidas internacionais).
    Área de penalidade: 40,32m × 16,5m | Área pequena: 18,32m × 5,5m | Círculo central: r = 9,15m.
    """
}

# ==================== SISTEMA GLOBAL DE CORES POR ATLETA ====================
_ATHLETE_PALETTE = [
    '#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0',
    '#00BCD4','#F44336','#FFEB3B','#26A69A','#78909C',
    '#AB47BC','#EC407A','#66BB6A','#FFA726','#42A5F5',
    '#EF5350','#26C6DA','#D4E157','#8D6E63','#5C6BC0',
]

def cor_atleta(nome: str) -> str:
    """Retorna sempre a mesma cor para o mesmo atleta (determinístico).
    Prioriza a paleta persistida no session_state; usa hash como fallback."""
    cores = st.session_state.get('athlete_colors', {})
    if nome in cores:
        return cores[nome]
    # fallback determinístico via hash do nome
    import hashlib
    idx = int(hashlib.md5(nome.encode()).hexdigest(), 16) % len(_ATHLETE_PALETTE)
    return _ATHLETE_PALETTE[idx]

# ==================== CORES POR GRUPO DE POSIÇÃO ====================
import colorsys as _colorsys

# Hue base para cada grupo tático (HSL)
_POSICAO_GRUPOS = {
    'Goleiro':      {'tags': ['GK','GR','GOL','GOLEIRO','GOALKEEPER','KEEPER','POR','PORTERO','ARQUEIRO'],    'h': 0.14},
    'Defensor':     {'tags': ['CB','ZAG','LD','LE','RB','LB','DEF','DEFENSOR','ZAGUEIRO','LATERAL','BACK'],  'h': 0.60},
    'Meio-campo':   {'tags': ['CM','CDM','VOL','MC','MCD','MEI','MED','VOLANTE','MEDIO','MEIA','MIDFIELD'],  'h': 0.35},
    'Ala/Extremo':  {'tags': ['LW','RW','PE','PD','ALA','EXT','ALE','ALD','PONTA','WINGER','EXTREMO'],      'h': 0.08},
    'Atacante':     {'tags': ['ST','CF','CAM','SS','CA','SA','AT','C9','ATACANTE','CENTROAVANTE','STRIKER','FORWARD'], 'h': 0.02},
}

# Cor representativa de cada grupo (para legenda)
_POSICAO_COR_LEGENDA = {
    'Goleiro':     '#F4D03F',
    'Defensor':    '#2196F3',
    'Meio-campo':  '#4CAF50',
    'Ala/Extremo': '#FF9800',
    'Atacante':    '#F44336',
    'Outro':       '#9C27B0',
}

def _get_pos_grupo(posicao: str) -> tuple:
    """Retorna (nome_do_grupo, hue_base) para uma string de posição."""
    p = (posicao or '').upper().strip()
    for grupo, cfg in _POSICAO_GRUPOS.items():
        if any(tag in p for tag in cfg['tags']):
            return grupo, cfg['h']
    return 'Outro', 0.75

def cor_atleta_pos(nome: str, posicao: str = '', variante_idx: int = 0, total: int = 1) -> str:
    """Cor baseada no grupo de posição com variação tonal individual.

    Atletas do mesmo grupo partilham a mesma tonalidade (hue); saturação e
    luminosidade variam para distinguir jogadores dentro do grupo.
    """
    _, h = _get_pos_grupo(posicao)
    frac = (variante_idx / max(1, total - 1)) if total > 1 else 0.5
    # Saturação: 0.70 → 0.95 (mais vivo conforme índice aumenta)
    s = 0.70 + 0.25 * frac
    # Luminosidade: 0.60 → 0.38 (escurece progressivamente)
    l = 0.60 - 0.22 * frac
    r, g, b = _colorsys.hls_to_rgb(h, l, s)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

# Ícone emoji por tipo de evento de futebol (para timeline e overlay)
_EV_ICONES = {
    'Goal':          '⚽', 'Gol': '⚽', 'goal': '⚽',
    'Card':          '🟨', 'Cartão': '🟨', 'Yellow Card': '🟨', 'Red Card': '🟥',
    'Substitution':  '🔄', 'Substituição': '🔄',
    'Shot':          '🎯', 'Chute': '🎯', 'Finalização': '🎯',
    'Foul Committed':'🤚', 'Falta': '🤚',
    'Offside':       '🚩', 'Impedimento': '🚩',
    'Save':          '🧤',
    'Corner':        '🏳️', 'Escanteio': '🏳️',
    'Free Kick':     '⚡',
    'Penalty':       '🎯', 'Pênalti': '🎯',
}

def _ev_icone(tipo: str) -> str:
    """Retorna emoji do evento ou '📌' como fallback."""
    for key, icon in _EV_ICONES.items():
        if key.lower() in (tipo or '').lower():
            return icon
    return '📌'

# ==================== CACHE DE API (TTL 15 min) ====================
# Standalone function cacheada pelo Streamlit. Parâmetros como
# primitivos/tuples para serem hashable. TTL de 900s (15 min).
@st.cache_data(ttl=900, show_spinner=False)
def _api_fetch(base_url: str, token: str, path: str,
               params: tuple = ()) -> object:
    """Chamada HTTP GET à API Catapult com cache automático de 15 min."""
    headers = {'Authorization': f'Bearer {token}'}
    try:
        r = requests.get(f"{base_url}/{path}",
                         headers=headers,
                         params=dict(params),
                         timeout=60)
        if r.status_code == 200:
            return r.json()
        # Salva o status de erro para exibir ao usuário
        try:
            import streamlit as _st_fetch
            _st_fetch.session_state['_api_last_err'] = r.status_code
        except Exception:
            pass
        return None
    except Exception as _exc:
        try:
            import streamlit as _st_fetch
            _st_fetch.session_state['_api_last_err'] = str(_exc)
        except Exception:
            pass
        return None


# ==================== PREFERÊNCIAS DO USUÁRIO ====================
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
        pass


# ==================== BANCO COMPARTILHADO DE VENUES ====================
# venues.json fica na mesma pasta do script → qualquer usuário que acesse
# o app no mesmo servidor lê e escreve neste arquivo.
# Estrutura: { "Nome do Venue": { lat, lon, rot, fl, fw, ig, saved_at } }
_VENUES_FILE = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "venues.json"
)

def _carregar_venues() -> dict:
    """Retorna o dicionário de venues salvos (nome → config)."""
    try:
        with open(_VENUES_FILE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _salvar_venue(nome: str, cfg: dict) -> None:
    """Persiste (ou sobrescreve) a configuração de um venue no arquivo compartilhado."""
    dados = _carregar_venues()
    dados[nome.strip()] = {
        'lat':      cfg.get('lat', 0.0),
        'lon':      cfg.get('lon', 0.0),
        'rot':      cfg.get('rot', 0),
        'fl':       cfg.get('fl',  105),
        'fw':       cfg.get('fw',  68),
        'ig':       cfg.get('ig',  3),
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    try:
        with open(_VENUES_FILE, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _excluir_venue(nome: str) -> None:
    """Remove um venue do banco compartilhado."""
    dados = _carregar_venues()
    dados.pop(nome, None)
    try:
        with open(_VENUES_FILE, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class CatapultAPI:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self._token   = token                                  # usado no cache
        self.headers  = {'Authorization': f'Bearer {token}'}
    
    def get_athletes(self):
        return _api_fetch(self.base_url, self._token, "athletes")

    def get_athlete(self, athlete_id):
        """Perfil completo de um atleta (GET /athletes/{id})."""
        return _api_fetch(self.base_url, self._token, f"athletes/{athlete_id}")

    def get_teams(self):
        return _api_fetch(self.base_url, self._token, "teams")

    def get_team_athletes(self, team_id):
        return _api_fetch(self.base_url, self._token, f"teams/{team_id}/athletes")

    def get_activities(self):
        return _api_fetch(self.base_url, self._token, "activities",
                          params=(("page_size", "500"),))

    def get_activity_athletes(self, activity_id):
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/athletes")

    def get_activity_periods(self, activity_id):
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/periods")

    def get_all_periods(self):
        return _api_fetch(self.base_url, self._token, "periods")

    def get_athletes_in_period(self, period_id):
        return _api_fetch(self.base_url, self._token,
                          f"periods/{period_id}/athletes")

    def get_positions(self):
        return _api_fetch(self.base_url, self._token, "positions")

    def get_parameters(self):
        return _api_fetch(self.base_url, self._token, "parameters")
    
    _SENSOR_PARAMS = (
        ("parameters", "ts,lat,long,v,rv,a,hr,pl,xy,pq,hdop,ref,o,mp"),
        ("nulls",      "1"),
    )

    def get_sensor_data(self, activity_id, athlete_id):
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/athletes/{athlete_id}/sensor",
                          params=self._SENSOR_PARAMS)

    def get_period_sensor_data(self, period_id, athlete_id):
        return _api_fetch(self.base_url, self._token,
                          f"periods/{period_id}/athletes/{athlete_id}/sensor",
                          params=self._SENSOR_PARAMS)

    def get_activity_efforts(self, activity_id, athlete_id,
                             effort_types="velocity,acceleration"):
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/athletes/{athlete_id}/efforts",
                          params=(("effort_types", effort_types),))

    def get_period_efforts(self, period_id, athlete_id,
                           effort_types="velocity,acceleration"):
        return _api_fetch(self.base_url, self._token,
                          f"periods/{period_id}/athletes/{athlete_id}/efforts",
                          params=(("effort_types", effort_types),))

    def get_activity_events(self, activity_id, athlete_id, event_types):
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/athletes/{athlete_id}/events",
                          params=(("event_types", event_types),))

    def get_period_events(self, period_id, athlete_id, event_types):
        return _api_fetch(self.base_url, self._token,
                          f"periods/{period_id}/athletes/{athlete_id}/events",
                          params=(("event_types", event_types),))

    # ── Live endpoints (sem cache — dados em tempo real) ──────────────
    def get_live_info(self):
        """Metadados da sessão ao vivo ativa (GET /live/info)."""
        import requests as _req
        try:
            r = _req.get(
                f"{self.base_url}/live/info",
                headers=self.headers, timeout=8,
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def get_live_athletes(self):
        """Métricas ao vivo de todos os atletas na sessão ativa (GET /live)."""
        import requests as _req
        try:
            r = _req.get(
                f"{self.base_url}/live",
                headers=self.headers, timeout=8,
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    # ── Tags, thresholds e stats (sem cache onde necessário) ─────────────
    def get_tags(self):
        """Todas as tags disponíveis no sistema (GET /tags)."""
        return _api_fetch(self.base_url, self._token, "tags")

    def get_activity_tags(self, activity_id):
        """Tags associadas a uma atividade específica (GET /activities/{id}/tags)."""
        return _api_fetch(self.base_url, self._token, f"activities/{activity_id}/tags")

    def get_athlete_thresholds(self, athlete_id):
        """Limiares individuais de velocidade/acc do atleta (GET /athletes/{id}/thresholds)."""
        return _api_fetch(self.base_url, self._token, f"athletes/{athlete_id}/thresholds")

    def get_stats(self, payload):
        """Estatísticas agregadas por grupo (POST /stats). Não usa cache — dados dinâmicos."""
        import requests as _req
        try:
            r = _req.post(
                f"{self.base_url}/stats",
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload, timeout=20,
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    # ── Velocity zones ───────────────────────────────────────────────────────
    def get_velocity_zones(self):
        """Bandas de velocidade configuradas na conta (GET /velocity_zones)."""
        return _api_fetch(self.base_url, self._token, "velocity_zones")

    def get_athlete_velocity_zones(self, athlete_id):
        """Bandas de velocidade personalizadas por atleta (GET /athletes/{id}/velocity_zones)."""
        return _api_fetch(self.base_url, self._token, f"athletes/{athlete_id}/velocity_zones")

    # ── Activity/period summaries (pre-computed by OpenField) ───────────────
    def get_athlete_activity_summary(self, activity_id, athlete_id):
        """Resumo pré-computado pelo OpenField (GET /activities/{id}/athletes/{aid}/summary)."""
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/athletes/{athlete_id}/summary")

    def get_athlete_period_summary(self, period_id, athlete_id):
        """Resumo pré-computado por período (GET /periods/{id}/athletes/{aid}/summary)."""
        return _api_fetch(self.base_url, self._token,
                          f"periods/{period_id}/athletes/{athlete_id}/summary")

    # ── Session parameters (dynamic device discovery) ────────────────────────
    def get_session_parameters(self, activity_id):
        """Parâmetros disponíveis nesta sessão (GET /activities/{id}/parameters)."""
        return _api_fetch(self.base_url, self._token, f"activities/{activity_id}/parameters")

    # ── Venues from Catapult account ─────────────────────────────────────────
    def get_venues(self):
        """Venues cadastrados na conta (GET /venues)."""
        return _api_fetch(self.base_url, self._token, "venues")

    # ── Annotations ──────────────────────────────────────────────────────────
    def get_activity_annotations(self, activity_id):
        """Lista anotações de uma atividade (GET /activities/{id}/annotations)."""
        return _api_fetch(self.base_url, self._token, f"activities/{activity_id}/annotations")

    def create_activity_annotation(self, activity_id, name, start_time, end_time,
                                   annotation_type="phase"):
        """Cria nova anotação (POST /activities/{id}/annotations)."""
        import requests as _req
        try:
            payload = {
                "name": name,
                "start_time": start_time,
                "end_time": end_time,
                "annotation_type": annotation_type,
            }
            r = _req.post(
                f"{self.base_url}/activities/{activity_id}/annotations",
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload, timeout=20,
            )
            return r.json() if r.status_code in (200, 201) else None
        except Exception:
            return None

    def delete_annotation(self, annotation_id):
        """Remove uma anotação (DELETE /annotations/{id})."""
        import requests as _req
        try:
            r = _req.delete(
                f"{self.base_url}/annotations/{annotation_id}",
                headers=self.headers, timeout=15,
            )
            return r.status_code in (200, 204)
        except Exception:
            return False

    # ── Async export ─────────────────────────────────────────────────────────
    def submit_export(self, payload):
        """Submete job de exportação assíncrona (POST /export)."""
        import requests as _req
        try:
            r = _req.post(
                f"{self.base_url}/export",
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload, timeout=20,
            )
            return r.json() if r.status_code in (200, 201, 202) else None
        except Exception:
            return None

    def get_export_status(self, job_id):
        """Verifica status de um job de exportação (GET /export/{job_id})."""
        import requests as _req
        try:
            r = _req.get(
                f"{self.base_url}/export/{job_id}",
                headers=self.headers, timeout=15,
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def download_export(self, job_id):
        """Download do arquivo exportado (GET /export/{job_id}/download)."""
        import requests as _req
        try:
            r = _req.get(
                f"{self.base_url}/export/{job_id}/download",
                headers=self.headers, timeout=60,
            )
            return r.content if r.status_code == 200 else None
        except Exception:
            return None

    # PARTE 2 - FUNÇÕES DE EXTRAÇÃO, CONVERSÃO E CÁLCULO

# ── Parâmetros globais de duração mínima de esforço (segundos) ────────────────
# Acc/Dec: 0.6 s (Catapult OpenField)  |  Velocidade: 1.0 s
_DEFAULT_MIN_DUR_S     = 0.6   # acc / dec
_DEFAULT_MIN_DUR_VEL_S = 1.0   # esforços de velocidade
_SENSOR_HZ = 10  # frequência de amostragem Catapult (10 Hz)


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


def get_min_dur_s():
    """Lê o slider de duração mínima de acc/dec do session_state."""
    return float(st.session_state.get('min_dur_esforco', _DEFAULT_MIN_DUR_S))

def get_min_dur_vel_s():
    """Lê o slider de duração mínima de velocidade do session_state."""
    return float(st.session_state.get('min_dur_vel', _DEFAULT_MIN_DUR_VEL_S))


def extrair_dados_sensor(response_data):
    if not response_data:
        return []
    if isinstance(response_data, list):
        for item in response_data:
            if isinstance(item, dict) and 'data' in item:
                return item['data']
    return []

def extrair_efforts_data(response_data):
    if not response_data:
        return [], [], [], [], []

    velocity_efforts = []
    acceleration_efforts = []
    heart_rate_efforts = []
    jump_efforts = []
    step_balance_efforts = []

    if isinstance(response_data, list) and len(response_data) > 0:
        item = response_data[0]
        if isinstance(item, dict) and 'data' in item:
            data_obj = item['data']
            if isinstance(data_obj, dict):
                velocity_efforts = data_obj.get('velocity_efforts', [])
                acceleration_efforts = data_obj.get('acceleration_efforts', [])
                heart_rate_efforts = data_obj.get('heart_rate_efforts', [])
                jump_efforts = data_obj.get('jump_efforts', [])
                step_balance_efforts = data_obj.get('step_balance_efforts', [])

    return velocity_efforts, acceleration_efforts, heart_rate_efforts, jump_efforts, step_balance_efforts

# ==================== CONVERSÃO DE COORDENADAS GPS → CAMPO ====================

def lat_lon_to_campo_coords(latitudes, longitudes):
    """
    Converte coordenadas de latitude/longitude para coordenadas do campo (0-100m x 0-70m)
    """
    if len(latitudes) == 0 or len(longitudes) == 0:
        return [], []
    
    lats = np.array(latitudes)
    lons = np.array(longitudes)
    
    # Normalizar para 0-1 baseado no range dos dados
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
    if lat_max == lat_min:
        lat_range = 1
    else:
        lat_range = lat_max - lat_min
    
    if lon_max == lon_min:
        lon_range = 1
    else:
        lon_range = lon_max - lon_min
    
    # Normalizar
    lat_norm = (lats - lat_min) / lat_range
    lon_norm = (lons - lon_min) / lon_range
    
    # Escalar para o campo (comprimento 105m, largura 68m — FIFA)
    x = lon_norm * 105
    y = lat_norm * 68
    
    return x.tolist(), y.tolist()

def desenhar_campo_futebol(field_length=105, field_width=68):
    """Desenha o campo de futebol com todas as marcações oficiais FIFA."""
    fig = go.Figure()
    FL, FW = field_length, field_width
    cy = FW / 2  # centro y

    # Fundo
    fig.add_shape(type="rect", x0=-5, y0=-5, x1=FL+5, y1=FW+5,
                  fillcolor="#1a472a", line=dict(color="rgba(0,0,0,0)", width=0))
    # Perímetro
    fig.add_shape(type="rect", x0=0, y0=0, x1=FL, y1=FW,
                  line=dict(color="white", width=3), fillcolor="rgba(0,100,0,0.3)")
    # Linha central
    fig.add_shape(type="line", x0=FL/2, y0=0, x1=FL/2, y1=FW, line=dict(color="white", width=2))
    # Círculo central (r=9.15m)
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=FL/2 + 9.15*np.cos(theta), y=cy + 9.15*np.sin(theta),
                             mode='lines', line=dict(color="white", width=1.5),
                             name="Círculo Central", showlegend=False))
    # Áreas de penalidade (16.5m × 40.32m)
    for x0, x1 in [(0, 16.5), (FL-16.5, FL)]:
        fig.add_shape(type="rect", x0=x0, y0=cy-20.16, x1=x1, y1=cy+20.16,
                      line=dict(color="white", width=1.5))
    # Áreas pequenas (5.5m × 18.32m)
    for x0, x1 in [(0, 5.5), (FL-5.5, FL)]:
        fig.add_shape(type="rect", x0=x0, y0=cy-9.16, x1=x1, y1=cy+9.16,
                      line=dict(color="white", width=1.5))
    # Pontos de penalidade
    for px in [11, FL-11]:
        fig.add_trace(go.Scatter(x=[px], y=[cy], mode='markers',
                                 marker=dict(size=6, color='white'),
                                 showlegend=False))
    # Ponto central
    fig.add_trace(go.Scatter(x=[FL/2], y=[cy], mode='markers',
                             marker=dict(size=6, color='white'), showlegend=False))
    # Gols (7.32m × 2.44m)
    fig.add_shape(type="rect", x0=-2.44, y0=cy-3.66, x1=0, y1=cy+3.66,
                  line=dict(color="#FFD700", width=3))
    fig.add_shape(type="rect", x0=FL, y0=cy-3.66, x1=FL+2.44, y1=cy+3.66,
                  line=dict(color="#FFD700", width=3))

    fig.update_layout(
        title="Campo de Futebol Oficial FIFA — Trajetória e Mapa de Calor",
        xaxis=dict(range=[-5, FL+5], title="Comprimento do Campo (m)",
                   fixedrange=False, gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        yaxis=dict(range=[-5, FW+5], title="Largura do Campo (m)",
                   fixedrange=False, gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        plot_bgcolor='#1a472a', paper_bgcolor='#1a1a2e', height=600, hovermode='closest'
    )
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                       text="⚪ Linhas FIFA | 🟡 Gol | 🔵 Trajetória | 🟢 Início | 🔴 Fim",
                       showarrow=False, font=dict(color="white", size=10),
                       bgcolor='rgba(0,0,0,0.6)', borderpad=4)
    return fig

def plotar_trajetoria_campo(x_coords, y_coords, velocidades, athlete_name):
    """Plota a trajetória do atleta no campo"""
    fig = desenhar_campo_futebol(105, 68)
    
    if len(x_coords) == 0:
        return fig
    
    # Trajetória (linha)
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='lines',
        name=f'{athlete_name} - Trajetória',
        line=dict(color='cyan', width=2),
        hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<extra></extra>'
    ))
    
    # Pontos coloridos por velocidade
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='markers',
        name='Velocidade',
        marker=dict(size=3, color=velocidades, colorscale='Viridis',
                   showscale=True, colorbar=dict(title=dict(text="Velocidade (km/h)"), x=1.05, len=0.5)),
        hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Vel: %{marker.color:.1f} km/h<extra></extra>'
    ))
    
    # Início
    fig.add_trace(go.Scatter(
        x=[x_coords[0]], y=[y_coords[0]],
        mode='markers',
        name='Início',
        marker=dict(size=14, color='#00FF00', symbol='circle', line=dict(width=2, color='white'))
    ))
    
    # Fim
    fig.add_trace(go.Scatter(
        x=[x_coords[-1]], y=[y_coords[-1]],
        mode='markers',
        name='Fim',
        marker=dict(size=14, color='#FF0000', symbol='x', line=dict(width=2, color='white'))
    ))
    
    return fig

def plotar_heatmap_campo(x_coords, y_coords, velocidades, athlete_name):
    """Plota mapa de calor de intensidade no campo"""
    fig = desenhar_campo_futebol(105, 68)

    if len(x_coords) == 0 or len(y_coords) == 0:
        return fig

    x_edges = np.linspace(0, 105, 40)
    y_edges = np.linspace(0, 68, 40)
    
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges], weights=velocidades)
    counts, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts > 0)
    
    fig.add_trace(go.Heatmap(
        z=intensity.T,
        x=xedges[:-1],
        y=yedges[:-1],
        colorscale='Hot',
        opacity=0.6,
        name='Intensidade (Velocidade)',
        colorbar=dict(title=dict(text="Velocidade (km/h)"), x=1.05, len=0.5),
        hovertemplate='X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Vel: %{z:.1f} km/h<extra></extra>'
    ))
    
    return fig

def plotar_heatmap_presenca_campo(x_coords, y_coords, athlete_name):
    """Plota mapa de calor de presença (frequência de posições) no campo"""
    fig = desenhar_campo_futebol(105, 68)

    if len(x_coords) == 0 or len(y_coords) == 0:
        return fig

    x_edges = np.linspace(0, 105, 40)
    y_edges = np.linspace(0, 68, 40)

    counts, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])

    fig.add_trace(go.Heatmap(
        z=counts.T,
        x=xedges[:-1],
        y=yedges[:-1],
        colorscale='YlOrRd',
        opacity=0.65,
        name='Presença',
        colorbar=dict(title=dict(text="Frequência"), x=1.05, len=0.5),
        hovertemplate='X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Frequência: %{z:.0f}<extra></extra>'
    ))

    fig.update_layout(title=f"🔥 Mapa de Calor de Presença — {athlete_name}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# BANDAS DE VELOCIDADE E ACELERAÇÃO (referências Catapult Football)
# ══════════════════════════════════════════════════════════════════════════════
BANDAS_VEL = {
    1: {'label': 'B1 — < 8 km/h (Caminhada)',          'min': 0,    'max': 8,    'color': '#2196F3'},
    2: {'label': 'B2 — 8-14 km/h (Trote)',             'min': 8,    'max': 14,   'color': '#4CAF50'},
    3: {'label': 'B3 — 14-19 km/h (Corrida)',          'min': 14,   'max': 19,   'color': '#CDDC39'},
    4: {'label': 'B4 — 19-23 km/h (Corrida Intensa)',  'min': 19,   'max': 23,   'color': '#FF9800'},
    5: {'label': 'B5 — 23-25 km/h (Alta Velocidade)',  'min': 23,   'max': 25,   'color': '#FF5722'},
    6: {'label': 'B6 — > 25 km/h (Sprint)',            'min': 25,   'max': 9999, 'color': '#F44336'},
}
BANDAS_ACC = {
    'A3': {'label': 'Acc +3 — > 2 m/s² (Alta Aceleração)',   'min': 2,     'max': 9999, 'color': '#00C853'},
    'A2': {'label': 'Acc +2 — 1-2 m/s²',                    'min': 1,     'max': 2,    'color': '#69F0AE'},
    'A1': {'label': 'Acc +1 — 0.1-1 m/s²',                  'min': 0.1,   'max': 1,    'color': '#B9F6CA'},
    'D1': {'label': 'Dec -1 — 0 a -1 m/s²',                 'min': -1,    'max': -0.1, 'color': '#FFD180'},
    'D2': {'label': 'Dec -2 — -1 a -2 m/s²',                'min': -2,    'max': -1,   'color': '#FF6D00'},
    'D3': {'label': 'Dec -3 — < -2 m/s² (Alta Desacel.)',   'min': -9999, 'max': -2,   'color': '#DD2C00'},
}

# ══════════════════════════════════════════════════════════════════════════════
# BANDAS DE VELOCIDADE — helpers de zonas individuais / da conta
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_VELOCITY_ZONES = [
    {'name': 'B1 — Caminhada',        'min_ms': 0/3.6,   'max_ms': 7/3.6,   'color': '#2196F3'},
    {'name': 'B2 — Trote',            'min_ms': 7/3.6,   'max_ms': 14/3.6,  'color': '#4CAF50'},
    {'name': 'B3 — Corrida',          'min_ms': 14/3.6,  'max_ms': 19/3.6,  'color': '#CDDC39'},
    {'name': 'B4 — Corrida Intensa',  'min_ms': 19/3.6,  'max_ms': 24/3.6,  'color': '#FF9800'},
    {'name': 'B5 — Alta Velocidade',  'min_ms': 24/3.6,  'max_ms': 30/3.6,  'color': '#FF5722'},
    {'name': 'B6 — Sprint',           'min_ms': 30/3.6,  'max_ms': 9999,    'color': '#F44336'},
]


def _parse_api_velocity_zones(api_response):
    """Converte resposta da API /velocity_zones para lista de dicts padronizados."""
    if not api_response:
        return _DEFAULT_VELOCITY_ZONES[:]
    try:
        zones_raw = api_response if isinstance(api_response, list) else api_response.get('data', [])
        if not zones_raw:
            return _DEFAULT_VELOCITY_ZONES[:]
        result = []
        for z in zones_raw:
            min_val = z.get('min_velocity', z.get('min', 0))
            max_val = z.get('max_velocity', z.get('max', 9999))
            result.append({
                'name': z.get('name', ''),
                'min_ms': float(min_val),
                'max_ms': float(max_val),
                'color': z.get('color', '#888888'),
            })
        return result if result else _DEFAULT_VELOCITY_ZONES[:]
    except Exception:
        return _DEFAULT_VELOCITY_ZONES[:]


def get_zones_for_athlete(athlete_name):
    """Retorna zonas de velocidade para o atleta (override > conta > defaults)."""
    overrides = st.session_state.get('velocity_zones_athlete', {})
    if athlete_name in overrides and overrides[athlete_name]:
        return overrides[athlete_name]
    account_zones = st.session_state.get('velocity_zones_account')
    if account_zones:
        return account_zones
    return _DEFAULT_VELOCITY_ZONES[:]


# ── Configuração dos tipos de eventos futebol ──────────────────────────────
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

def extrair_eventos_futebol(response_data):
    """Extrai eventos futebol da resposta da API /events."""
    if not response_data or not isinstance(response_data, list):
        return {}
    item = response_data[0] if response_data else {}
    data = item.get('data', {}) if isinstance(item, dict) else {}
    return {k: v for k, v in data.items() if v and k in FUTEBOL_EVENTS_CONFIG}

def enriquecer_eventos_com_posicao(eventos_dict, ts_gps, lats_gps, lons_gps, vels_gps, campo_config=None):
    """Associa cada evento ao ponto GPS (e posição no campo) mais próximo no tempo."""
    if not ts_gps:
        return eventos_dict
    ts_arr = np.array(ts_gps, dtype=float)
    result = {}
    for event_type, events in eventos_dict.items():
        enriched = []
        for ev in events:
            ev2 = dict(ev)
            t = float(ev.get('start_time') or 0)
            idx = int(np.argmin(np.abs(ts_arr - t)))
            ev2['_lat'] = lats_gps[idx]
            ev2['_lon'] = lons_gps[idx]
            ev2['_vel'] = vels_gps[idx]
            if campo_config:
                fx, fy = gps_para_campo_coords(
                    [lats_gps[idx]], [lons_gps[idx]], campo_config)
                ev2['_fx'] = fx[0]
                ev2['_fy'] = fy[0]
            enriched.append(ev2)
        result[event_type] = enriched
    return result

def adicionar_eventos_campo(fig, eventos_dict, tipos_sel):
    """Plota marcadores de eventos futebol sobre o campo esquemático."""
    for event_type in tipos_sel:
        events = eventos_dict.get(event_type, [])
        if not events:
            continue
        cfg_ev = FUTEBOL_EVENTS_CONFIG[event_type]
        xs_ev = [ev.get('_fx') for ev in events if ev.get('_fx') is not None]
        ys_ev = [ev.get('_fy') for ev in events if ev.get('_fy') is not None]
        if not xs_ev:
            continue
        # Texto de hover
        texts = []
        for ev in events:
            if ev.get('_fx') is None:
                continue
            partes = [f"<b>{cfg_ev['label']}</b>"]
            for attr in cfg_ev['attrs']:
                val = ev.get(attr)
                if val is not None:
                    partes.append(f"{attr}: {val}")
            partes.append(f"vel: {ev.get('_vel', 0):.1f} km/h")
            texts.append('<br>'.join(partes))
        fig.add_trace(go.Scatter(
            x=xs_ev, y=ys_ev,
            mode='markers',
            name=cfg_ev['label'],
            marker=dict(
                symbol=cfg_ev['marker'],
                size=cfg_ev['size'],
                color=cfg_ev['color'],
                line=dict(color='white', width=1.5),
            ),
            text=texts,
            hovertemplate='%{text}<extra></extra>',
            showlegend=True,
        ))

def criar_timeline_eventos(sensor_points, eventos_dict, atleta_nome, tipos_sel):
    """
    Gráfico de velocidade ao longo do tempo com pins verticais dos eventos futebol.
    """
    if not sensor_points:
        return None

    ts0 = float(sensor_points[0].get('ts') or 0)
    ts_rel = [(float(p.get('ts') or 0) - ts0) for p in sensor_points]
    vels   = [(p.get('v') or 0) * 3.6 for p in sensor_points]

    fig = go.Figure()

    # Linha de velocidade
    fig.add_trace(go.Scatter(
        x=ts_rel, y=vels,
        mode='lines',
        name='Velocidade (km/h)',
        line=dict(color='#2196F3', width=1.5),
        hovertemplate='%{x:.0f}s — %{y:.1f} km/h<extra></extra>',
    ))

    # Pins de eventos
    for event_type in tipos_sel:
        events = eventos_dict.get(event_type, [])
        if not events:
            continue
        cfg_ev = FUTEBOL_EVENTS_CONFIG[event_type]
        for ev in events:
            t_ev  = float(ev.get('start_time') or 0) - ts0
            v_ev  = ev.get('_vel', 0)
            label = cfg_ev['label']
            partes = [f"<b>{label}</b>", f"t={t_ev:.0f}s", f"vel={v_ev:.1f} km/h"]
            for attr in cfg_ev['attrs']:
                val = ev.get(attr)
                if val is not None:
                    partes.append(f"{attr}: {val}")
            fig.add_vline(
                x=t_ev,
                line=dict(color=cfg_ev['color'], width=1.5, dash='dot'),
            )
            fig.add_trace(go.Scatter(
                x=[t_ev], y=[v_ev],
                mode='markers',
                name=label,
                showlegend=False,
                marker=dict(
                    symbol=cfg_ev['marker'],
                    size=cfg_ev['size'],
                    color=cfg_ev['color'],
                    line=dict(color='white', width=1),
                ),
                text=['<br>'.join(partes)],
                hovertemplate='%{text}<extra></extra>',
            ))

    fig.update_layout(
        title=f"⚽ Timeline Técnico-Físico — {atleta_nome}",
        xaxis_title='Tempo (s)',
        yaxis_title='Velocidade (km/h)',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        height=420,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='left', x=0,
            font=dict(size=11),
        ),
        hovermode='x unified',
    )
    return fig

def analisar_fadiga_eventos(sensor_points, eventos_dict, tipos_sel, janela_s=10):
    """
    Para eventos de contato, computa velocidade média antes e depois do evento.
    Retorna DataFrame com: tipo, tempo, vel_antes, vel_depois, variacao.
    """
    if not sensor_points:
        return pd.DataFrame()

    ts_arr  = np.array([float(p.get('ts') or 0) for p in sensor_points])
    vel_arr = np.array([(p.get('v') or 0) * 3.6 for p in sensor_points])
    ts0     = ts_arr[0] if len(ts_arr) else 0

    rows = []
    for event_type in tipos_sel:
        events = eventos_dict.get(event_type, [])
        cfg_ev = FUTEBOL_EVENTS_CONFIG[event_type]
        for ev in events:
            t = float(ev.get('start_time') or 0)
            mask_before = (ts_arr >= t - janela_s) & (ts_arr < t)
            mask_after  = (ts_arr >  t) & (ts_arr <= t + janela_s)
            vel_antes   = float(vel_arr[mask_before].mean()) if mask_before.any() else None
            vel_depois  = float(vel_arr[mask_after].mean())  if mask_after.any()  else None
            variacao    = round(vel_depois - vel_antes, 2) if (vel_antes is not None and vel_depois is not None) else None
            row = {
                'Tipo': cfg_ev['label'],
                'Tempo (s)': round(t - ts0, 1),
                'Vel. Antes (km/h)': round(vel_antes, 2) if vel_antes is not None else None,
                'Vel. Depois (km/h)': round(vel_depois, 2) if vel_depois is not None else None,
                'Variação (km/h)': variacao,
                'post_event_load': ev.get('post_event_load'),
                'duration': ev.get('duration'),
                'confidence': ev.get('confidence'),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values('Tempo (s)').reset_index(drop=True)
    return df


def desenhar_campo_futebol_bonito(field_length=105, field_width=68, margin=3, title="", attack_direction=None):
    """Campo de futebol com faixas verdes alternadas e marcações oficiais FIFA (top-down).

    attack_direction: None | 'left_to_right' | 'right_to_left'
        Quando definido, exibe uma seta dourada acima do campo indicando o sentido de ataque.
    """
    FL, FW, MG = field_length, field_width, margin
    cy = FW / 2  # centro y
    fig = go.Figure()

    # Fundo externo
    fig.add_shape(type="rect", x0=-MG-4, y0=-MG-2, x1=FL+MG+4, y1=FW+MG+2,
                  fillcolor="#1a3a18", line_width=0, layer="below")

    # Faixas verdes alternadas no campo de jogo
    n_st, sw = 10, FL / 10
    cores_faixa = ["#2d7828", "#257022"]
    for i in range(n_st):
        fig.add_shape(type="rect", x0=i*sw, y0=0, x1=(i+1)*sw, y1=FW,
                      fillcolor=cores_faixa[i % 2], line_width=0, layer="below")

    # Perímetro principal
    fig.add_shape(type="rect", x0=0, y0=0, x1=FL, y1=FW,
                  line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")

    # Linha central
    fig.add_shape(type="line", x0=FL/2, y0=0, x1=FL/2, y1=FW,
                  line=dict(color="white", width=2))

    # Círculo central (r = 9.15m)
    th = np.linspace(0, 2*np.pi, 80)
    fig.add_trace(go.Scatter(x=FL/2 + 9.15*np.cos(th), y=cy + 9.15*np.sin(th),
                             mode='lines', line=dict(color='white', width=1.5),
                             showlegend=False, hoverinfo='skip', name='_circ'))
    # Ponto central
    fig.add_trace(go.Scatter(x=[FL/2], y=[cy], mode='markers',
                             marker=dict(size=5, color='white'),
                             showlegend=False, hoverinfo='skip', name='_ctr'))

    # Área de penalidade (16.5m × 40.32m)
    for x0, x1 in [(0, 16.5), (FL-16.5, FL)]:
        fig.add_shape(type="rect", x0=x0, y0=cy-20.16, x1=x1, y1=cy+20.16,
                      line=dict(color="white", width=1.5), fillcolor="rgba(0,0,0,0)")

    # Área pequena (5.5m × 18.32m)
    for x0, x1 in [(0, 5.5), (FL-5.5, FL)]:
        fig.add_shape(type="rect", x0=x0, y0=cy-9.16, x1=x1, y1=cy+9.16,
                      line=dict(color="white", width=1.5), fillcolor="rgba(0,0,0,0)")

    # Arcos de penalidade (r = 9.15m, apenas fora da área de penalidade)
    th_full = np.linspace(0, 2*np.pi, 200)
    for px_p, lado in [(11, 'esq'), (FL-11, 'dir')]:
        arc_x = px_p + 9.15*np.cos(th_full)
        arc_y = cy + 9.15*np.sin(th_full)
        mask = (arc_x > 16.5) if lado == 'esq' else (arc_x < FL-16.5)
        if mask.sum() > 1:
            fig.add_trace(go.Scatter(x=arc_x[mask], y=arc_y[mask],
                                     mode='lines', line=dict(color='white', width=1.5),
                                     showlegend=False, hoverinfo='skip', name='_parc'))

    # Pontos de penalidade
    for px_p in [11, FL-11]:
        fig.add_trace(go.Scatter(x=[px_p], y=[cy], mode='markers',
                                 marker=dict(size=6, color='white'),
                                 showlegend=False, hoverinfo='skip', name='_pen'))

    # Arcos de canto (r = 1m)
    corners_def = [(0, 0, 0, np.pi/2), (FL, 0, np.pi/2, np.pi),
                   (FL, FW, np.pi, 3*np.pi/2), (0, FW, 3*np.pi/2, 2*np.pi)]
    for cx_c, cy_c, a1, a2 in corners_def:
        th_c = np.linspace(a1, a2, 20)
        fig.add_trace(go.Scatter(x=cx_c + np.cos(th_c), y=cy_c + np.sin(th_c),
                                 mode='lines', line=dict(color='white', width=1.5),
                                 showlegend=False, hoverinfo='skip', name='_corner'))

    # Gols (7.32m × 2.44m) — dourado
    gd_line = dict(color="#FFD700", width=3)
    fig.add_shape(type="rect", x0=-2.44, y0=cy-3.66, x1=0,   y1=cy+3.66, line=dict(**gd_line))
    fig.add_shape(type="rect", x0=FL,   y0=cy-3.66, x1=FL+2.44, y1=cy+3.66, line=dict(**gd_line))

    # Labels das linhas
    lkw = dict(showarrow=False, font=dict(color='rgba(255,255,255,0.55)', size=9))
    for xl, txt in [(0, "GL"), (FL/2, "50m"), (FL, "GL")]:
        fig.add_annotation(x=xl, y=-2.5, text=txt, **lkw)

    # ── Seta de direção de ataque ────────────────────────────────────────────
    if attack_direction == 'left_to_right':
        # Seta da esquerda para a direita acima do campo
        fig.add_annotation(
            x=FL * 0.72, y=FW + MG - 0.2,
            ax=FL * 0.28, ay=FW + MG - 0.2,
            axref='x', ayref='y',
            text="", showarrow=True,
            arrowhead=2, arrowsize=1.4, arrowwidth=3,
            arrowcolor='#FFD700',
        )
        fig.add_annotation(
            x=FL * 0.5, y=FW + MG + 0.5,
            text="⚽  ATAQUE",
            showarrow=False,
            font=dict(color='#FFD700', size=11, family='Arial Black'),
            xanchor='center',
        )
    elif attack_direction == 'right_to_left':
        # Seta da direita para a esquerda acima do campo
        fig.add_annotation(
            x=FL * 0.28, y=FW + MG - 0.2,
            ax=FL * 0.72, ay=FW + MG - 0.2,
            axref='x', ayref='y',
            text="", showarrow=True,
            arrowhead=2, arrowsize=1.4, arrowwidth=3,
            arrowcolor='#FFD700',
        )
        fig.add_annotation(
            x=FL * 0.5, y=FW + MG + 0.5,
            text="ATAQUE  ⚽",
            showarrow=False,
            font=dict(color='#FFD700', size=11, family='Arial Black'),
            xanchor='center',
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=13)) if title else {},
        xaxis=dict(range=[-MG-4, FL+MG+4], showgrid=False, zeroline=False,
                   tickfont=dict(color='white', size=9),
                   title=dict(text="metros (comprimento)", font=dict(color='#aaa', size=10))),
        yaxis=dict(range=[-MG-2, FW+MG+2], showgrid=False, zeroline=False,
                   scaleanchor='x', scaleratio=1,
                   tickfont=dict(color='white', size=9),
                   title=dict(text="metros (largura)", font=dict(color='#aaa', size=10))),
        plot_bgcolor='#1a3a18',
        paper_bgcolor='#0e1117',
        height=530,
        margin=dict(l=50, r=160, t=40 if title else 20, b=50),
        hovermode='closest',
        legend=dict(bgcolor='rgba(0,0,0,0.7)', font=dict(color='white'),
                    x=1.01, y=1, bordercolor='rgba(255,255,255,0.2)', borderwidth=1)
    )
    return fig


def adicionar_trajetoria_campo(fig, x_coords, y_coords, velocidades, nome=""):
    """Trajetória colorida por velocidade sobre o campo bonito."""
    if not x_coords:
        return
    xs = list(x_coords)
    ys = list(y_coords)
    vs = list(velocidades) if velocidades else [0]*len(xs)

    def _vc(v):
        for k, b in BANDAS_VEL.items():
            if v < b['max']:
                return b['color']
        return BANDAS_VEL[6]['color']

    # Segmentos coloridos
    seg_x, seg_y = [xs[0]], [ys[0]]
    seg_c = _vc(vs[0])
    for i in range(1, len(xs)):
        c = _vc(vs[i] if i < len(vs) else 0)
        if c == seg_c:
            seg_x.append(xs[i]); seg_y.append(ys[i])
        else:
            if len(seg_x) > 1:
                fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode='lines',
                    line=dict(color=seg_c, width=2.5),
                    showlegend=False, hoverinfo='skip', name='_traj'))
            seg_x, seg_y = [xs[i-1], xs[i]], [ys[i-1], ys[i]]
            seg_c = c
    if len(seg_x) > 1:
        fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode='lines',
            line=dict(color=seg_c, width=2.5),
            showlegend=False, hoverinfo='skip', name='_traj'))

    # Início e fim
    fig.add_trace(go.Scatter(x=[xs[0]], y=[ys[0]], mode='markers', name='Início',
        marker=dict(size=12, color='#00E676', symbol='circle',
                    line=dict(color='white', width=2))))
    fig.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode='markers', name='Fim',
        marker=dict(size=12, color='#F44336', symbol='x',
                    line=dict(color='white', width=2))))

    # Legenda de bandas
    for b in BANDAS_VEL.values():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            name=b['label'], marker=dict(size=9, color=b['color'])))


def _segmentos_continuos(idxs, gap_max=8):
    """Agrupa índices consecutivos em segmentos (gap_max = frames tolerados entre pontos)."""
    if len(idxs) == 0:
        return []
    segs, seg = [], [idxs[0]]
    for i in range(1, len(idxs)):
        if idxs[i] - idxs[i - 1] <= gap_max:
            seg.append(idxs[i])
        else:
            if len(seg) >= 3:
                segs.append(seg)
            seg = [idxs[i]]
    if len(seg) >= 3:
        segs.append(seg)
    return segs


def adicionar_pontos_velocidade_bandas(fig, x_coords, y_coords, velocidades,
                                       bandas_sel, mostrar_setas=False,
                                       atleta_prefix=''):
    """Pontos coloridos por bandas de velocidade selecionadas.

    Se mostrar_setas=True, desenha uma seta direcional ao final de cada
    esforço contínuo dentro de cada banda (direção ofensiva/defensiva).
    atleta_prefix: se fornecido, é adicionado ao início do nome da legenda.
    """
    if not x_coords or not bandas_sel:
        return
    xs = np.array(x_coords)
    ys = np.array(y_coords)
    vs = np.array(velocidades) if velocidades else np.zeros(len(xs))
    for k in bandas_sel:
        b = BANDAS_VEL[k]
        mask = (vs >= b['min']) & (vs < b['max'])
        if mask.sum() == 0:
            continue
        _nome = f"{atleta_prefix}: {b['label']}" if atleta_prefix else b['label']
        fig.add_trace(go.Scatter(
            x=xs[mask], y=ys[mask], mode='markers', name=_nome,
            marker=dict(size=3, color=b['color'], opacity=0.8),
            hovertemplate='x=%{x:.1f}m y=%{y:.1f}m<extra>' + _nome + '</extra>'))

        if not mostrar_setas:
            continue

        # ── Uma seta por esforço contínuo dentro da banda ───────────
        for seg in _segmentos_continuos(np.where(mask)[0]):
            sx, sy = xs[seg], ys[seg]
            # Direção: últimos ~25% dos pontos do segmento
            n_t = max(2, min(len(seg) // 4, 10))
            ddx = float(sx[-1] - sx[-n_t])
            ddy = float(sy[-1] - sy[-n_t])
            nm = np.hypot(ddx, ddy)
            if nm < 0.3:                        # fallback início→fim
                ddx = float(sx[-1] - sx[0])
                ddy = float(sy[-1] - sy[0])
                nm  = np.hypot(ddx, ddy)
            if nm < 0.3:
                continue
            ddx /= nm; ddy /= nm
            # Comprimento do cabo: proporcional à extensão do esforço, 1.5–5 m
            span = np.hypot(float(sx[-1] - sx[0]), float(sy[-1] - sy[0]))
            cabo = float(np.clip(span * 0.25, 1.5, 5.0))
            tip_x, tip_y = float(sx[-1]), float(sy[-1])
            fig.add_annotation(
                x=tip_x + ddx * 0.8,   y=tip_y + ddy * 0.8,
                ax=tip_x - ddx * cabo, ay=tip_y - ddy * cabo,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=3,
                arrowsize=1.6, arrowwidth=2.0,
                arrowcolor=b['color'],
            )


def adicionar_pontos_aceleracao_bandas(fig, x_coords, y_coords, aceleracoes,
                                       bandas_sel, mostrar_setas=False,
                                       atleta_prefix=''):
    """Pontos coloridos por bandas de aceleração/desaceleração selecionadas.

    Se mostrar_setas=True, desenha uma seta direcional ao final de cada
    esforço contínuo dentro de cada banda.
    atleta_prefix: se fornecido, é adicionado ao início do nome da legenda.
    """
    if not x_coords or not bandas_sel:
        return
    xs  = np.array(x_coords)
    ys  = np.array(y_coords)
    acc = np.array(aceleracoes) if aceleracoes else np.zeros(len(xs))
    for k in bandas_sel:
        b = BANDAS_ACC[k]
        mask = (acc >= b['min']) & (acc < b['max'])
        if mask.sum() == 0:
            continue
        _nome = f"{atleta_prefix}: {b['label']}" if atleta_prefix else b['label']
        fig.add_trace(go.Scatter(
            x=xs[mask], y=ys[mask], mode='markers', name=_nome,
            marker=dict(size=3, color=b['color'], opacity=0.8),
            hovertemplate='x=%{x:.1f}m y=%{y:.1f}m<extra>' + _nome + '</extra>'))

        if not mostrar_setas:
            continue

        # ── Uma seta por esforço contínuo dentro da banda ───────────
        for seg in _segmentos_continuos(np.where(mask)[0]):
            sx, sy = xs[seg], ys[seg]
            n_t = max(2, min(len(seg) // 4, 10))
            ddx = float(sx[-1] - sx[-n_t])
            ddy = float(sy[-1] - sy[-n_t])
            nm  = np.hypot(ddx, ddy)
            if nm < 0.3:
                ddx = float(sx[-1] - sx[0])
                ddy = float(sy[-1] - sy[0])
                nm  = np.hypot(ddx, ddy)
            if nm < 0.3:
                continue
            ddx /= nm; ddy /= nm
            span = np.hypot(float(sx[-1] - sx[0]), float(sy[-1] - sy[0]))
            cabo = float(np.clip(span * 0.25, 1.5, 5.0))
            tip_x, tip_y = float(sx[-1]), float(sy[-1])
            fig.add_annotation(
                x=tip_x + ddx * 0.8,   y=tip_y + ddy * 0.8,
                ax=tip_x - ddx * cabo, ay=tip_y - ddy * cabo,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=3,
                arrowsize=1.6, arrowwidth=2.0,
                arrowcolor=b['color'],
            )


def adicionar_setas_direcao(fig, x_coords, y_coords,
                           xs_effort=None, ys_effort=None):
    """Uma única seta de direção no final do esforço (ou da trajetória).

    Se xs_effort/ys_effort forem fornecidos, a seta é laranja e posicionada
    no final da linha de esforço. Caso contrário, uma seta branca é colocada
    no final da trajetória completa.
    """
    # Escolhe a fonte de coordenadas
    _use_effort = xs_effort is not None and len(xs_effort) >= 2
    xs = np.array(xs_effort if _use_effort else x_coords)
    ys = np.array(ys_effort if _use_effort else y_coords)

    if len(xs) < 2:
        return

    # Calcula direção média dos últimos ~20% dos pontos (robustez contra noise)
    n_tail = max(2, min(len(xs) // 5, 12))
    dx = float(xs[-1] - xs[-n_tail])
    dy = float(ys[-1] - ys[-n_tail])
    norm = np.hypot(dx, dy)
    if norm < 0.3:
        # Fallback: direção global (início → fim)
        dx = float(xs[-1] - xs[0])
        dy = float(ys[-1] - ys[0])
        norm = np.hypot(dx, dy)
        if norm < 0.3:
            return
    dx /= norm; dy /= norm

    tip_x = float(xs[-1])
    tip_y = float(ys[-1])
    tail_len = 5.0   # comprimento do cabo da seta (m)

    cor = '#FF9800' if _use_effort else 'rgba(255,255,255,0.85)'

    fig.add_annotation(
        x=tip_x + dx * 1.5,
        y=tip_y + dy * 1.5,
        ax=tip_x - dx * tail_len,
        ay=tip_y - dy * tail_len,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True,
        arrowhead=3,
        arrowsize=2.2,
        arrowwidth=3.0 if _use_effort else 2.0,
        arrowcolor=cor,
    )


def adicionar_convex_hull(fig, x_coords, y_coords):
    """Polígono de área de atuação (Convex Hull) sobre o campo."""
    if len(x_coords) < 3:
        return
    try:
        from scipy.spatial import ConvexHull
        pts  = np.column_stack([x_coords, y_coords])
        hull = ConvexHull(pts)
        hx   = list(pts[hull.vertices, 0]) + [pts[hull.vertices[0], 0]]
        hy   = list(pts[hull.vertices, 1]) + [pts[hull.vertices[0], 1]]
        area = hull.volume   # em 2D 'volume' = área
        fig.add_trace(go.Scatter(x=hx, y=hy, mode='lines',
            name=f'Área de Atuação ({area:.0f} m²)',
            line=dict(color='#FFD700', width=2, dash='dash'),
            fill='toself', fillcolor='rgba(255,215,0,0.10)'))
    except Exception:
        pass


def adicionar_tercos_campo(fig, x_coords, y_coords, field_length=105, field_width=68):
    """Overlay dos terços do campo (defensivo / meio / ataque) com % de tempo."""
    if not x_coords:
        return
    xs = np.array(x_coords)
    n  = len(xs)
    tercos = [
        ("Defensivo", 0,            field_length/3,    'rgba(244,67,54,0.13)',  '#F44336'),
        ("Meio",      field_length/3,  2*field_length/3, 'rgba(255,235,59,0.10)', '#FFEB3B'),
        ("Ataque",    2*field_length/3, field_length,    'rgba(76,175,80,0.13)',  '#4CAF50'),
    ]
    for nome, x0, x1, fill, cor in tercos:
        pct = 100 * ((xs >= x0) & (xs < x1)).sum() / n if n > 0 else 0
        fig.add_shape(type="rect", x0=x0, y0=0, x1=x1, y1=field_width,
                      line=dict(color=cor, width=1.5, dash="dot"),
                      fillcolor=fill, layer="above")
        fig.add_annotation(x=(x0+x1)/2, y=field_width-3.5,
                           text=f"<b>{nome}</b><br>{pct:.1f}%",
                           showarrow=False, font=dict(color=cor, size=11),
                           bgcolor='rgba(0,0,0,0.55)', borderpad=3)


def adicionar_grade_quadrantes(fig, x_coords, y_coords, n_cols, n_rows,
                               field_length=105, field_width=68):
    """Grade de quadrantes com % de tempo em cada zona."""
    if not x_coords:
        return
    xs  = np.array(x_coords)
    ys  = np.array(y_coords)
    n   = len(xs)
    cw  = field_length / n_cols
    rh  = field_width  / n_rows

    # Pré-calcular todas as % para escalar a opacidade
    pcts = np.zeros((n_rows, n_cols))
    for r in range(n_rows):
        for c in range(n_cols):
            mask = ((xs >= c*cw) & (xs < (c+1)*cw) &
                    (ys >= r*rh) & (ys < (r+1)*rh))
            pcts[r, c] = 100 * mask.sum() / n if n > 0 else 0
    mx = pcts.max() if pcts.max() > 0 else 1

    for r in range(n_rows):
        for c in range(n_cols):
            x0, x1 = c*cw, (c+1)*cw
            y0, y1 = r*rh, (r+1)*rh
            pct   = pcts[r, c]
            alpha = 0.07 + 0.43 * (pct / mx)
            zid   = f"{chr(65+r)}{c+1}"
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                          line=dict(color="rgba(255,255,255,0.35)", width=1),
                          fillcolor=f"rgba(255,165,0,{alpha:.2f})", layer="above")
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2,
                               text=f"<b>{zid}</b><br>{pct:.1f}%",
                               showarrow=False,
                               font=dict(color='white', size=9),
                               bgcolor='rgba(0,0,0,0.45)', borderpad=2)


def stats_quadrante(x_coords, y_coords, velocidades, aceleracoes, x0, x1, y0, y1):
    """Estatísticas detalhadas de um quadrante."""
    xs  = np.array(x_coords)
    ys  = np.array(y_coords)
    vs  = np.array(velocidades)  if velocidades  else np.zeros(len(xs))
    acc = np.array(aceleracoes)  if aceleracoes  else np.zeros(len(xs))
    mask = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    nz   = int(mask.sum())
    n    = len(xs)
    return {
        'n_pontos':  nz,
        'pct':       round(100 * nz / n,  2) if n > 0 else 0.0,
        'vel_media': round(float(vs[mask].mean()),  2) if nz > 0 else 0.0,
        'vel_max':   round(float(vs[mask].max()),   2) if nz > 0 else 0.0,
        'acc_media': round(float(acc[mask].mean()), 3) if nz > 0 else 0.0,
        'acc_max':   round(float(np.abs(acc[mask]).max()), 3) if nz > 0 else 0.0,
    }


@st.cache_data(show_spinner=False)
def gps_para_campo_coords(lats, lons, campo_config):
    """
    Converte pontos GPS (lat/lon) para coordenadas relativas ao campo (x/y em metros,
    origem no canto inferior esquerdo) usando a configuração aplicada.

    É a operação inversa de campo_para_latlon:
      north_m = (lat - c_lat) * 111320
      east_m  = (lon - c_lon) * 111320 * cos(c_lat)
      [x_m, y_m] = R^{-T} * [north_m, east_m]    (rotação inversa)
      field_x = x_m + fl/2  (deslocar para canto inferior esquerdo)
      field_y = y_m + fw/2
    """
    c_lat = float(campo_config['lat'])
    c_lon = float(campo_config['lon'])
    rot   = np.radians(float(campo_config['rot']))
    fl    = float(campo_config['fl'])
    fw    = float(campo_config['fw'])

    lats_a = np.array(lats, dtype=float)
    lons_a = np.array(lons, dtype=float)

    north_m = (lats_a - c_lat) * 111320.0
    east_m  = (lons_a - c_lon) * 111320.0 * np.cos(np.radians(c_lat))

    # Rotação inversa correta (derivada da função JS campo_para_latlon)
    # JS: nM = yO*cos(r) - xO*sin(r) ; eM = yO*sin(r) + xO*cos(r)
    # Inversa: xO = eM*cos(r) - nM*sin(r) ; yO = nM*cos(r) + eM*sin(r)
    x_m = east_m  * np.cos(rot) - north_m * np.sin(rot)   # comprimento do campo
    y_m = north_m * np.cos(rot) + east_m  * np.sin(rot)   # largura do campo

    # Deslocar: centro do campo → canto inferior esquerdo
    field_x = np.clip(x_m + fl / 2, -5, fl + 5)
    field_y = np.clip(y_m + fw / 2, -5, fw + 5)

    return field_x.tolist(), field_y.tolist()


def campo_para_latlon(centro_lat, centro_lon, x_m, y_m, rotacao_deg):
    """
    Converte um ponto em metros relativos ao centro do campo (x para leste, y para norte)
    em coordenadas geográficas (lat, lon), aplicando rotação pelo bearing do campo.
    rotacao_deg: graus clockwise a partir do norte (ex: 90 = campo alinhado leste-oeste).
    """
    rot = np.radians(rotacao_deg)
    # Rotacionar o vetor (x_m, y_m) pelo bearing
    north_m = x_m * np.cos(rot) - y_m * np.sin(rot)
    east_m  = x_m * np.sin(rot) + y_m * np.cos(rot)
    d_lat = float(north_m / 111320.0)
    d_lon = float(east_m / (111320.0 * np.cos(np.radians(float(centro_lat)))))
    return (float(centro_lat) + d_lat, float(centro_lon) + d_lon)


def criar_mapa_satelite_futebol(
    lats, lons, vels, atleta_nome,
    centro_lat, centro_lon, rotacao_deg,
    field_length=105, field_width=68, in_goal=3,
    mostrar_campo=True
):
    """
    Cria um mapa Folium com:
    - Tiles de satélite Esri World Imagery (gratuito, sem chave de API)
    - Trajetória GPS do atleta colorida por faixa de velocidade
    - Overlay do campo de futebol (linhas FIFA + gols dourados) — opcional via mostrar_campo
    """
    if not lats or not lons:
        return None

    # ---- Mapa base ----
    m = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=17,
        tiles=None
    )

    # Satélite Esri (gratuito, sem chave)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🛰️ Satélite",
        overlay=False,
        control=True
    ).add_to(m)

    # Labels sobre o satélite (opcional)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Labels",
        name="🏷️ Rótulos",
        overlay=True,
        control=True,
        opacity=0.7
    ).add_to(m)

    # ---- Overlay do campo de futebol (só quando pedido) ----
    if mostrar_campo:
        cx = field_length / 2
        cy = field_width / 2

        def pt(x_offset, y_offset):
            return campo_para_latlon(centro_lat, centro_lon, y_offset, x_offset, rotacao_deg)

        def linha(pontos_xy):
            return [pt(x - cx, y - cy) for x, y in pontos_xy]

        campo_group = folium.FeatureGroup(name="⚽ Campo de Futebol", show=True)
        FW_h = field_width / 2

        def add_line(pts_xy, color="white", weight=2, dash=None):
            latlon_pts = linha(pts_xy)
            kwargs = dict(color=color, weight=weight, opacity=0.9)
            if dash:
                kwargs["dash_array"] = dash
            folium.PolyLine(latlon_pts, **kwargs).add_to(campo_group)

        # Perímetro
        add_line([(0,0),(field_length,0),(field_length,field_width),(0,field_width),(0,0)], weight=3)
        # Linha central
        add_line([(field_length/2, 0), (field_length/2, field_width)], weight=2)
        # Área de penalidade esquerda (16.5 × 40.32)
        add_line([(0,FW_h-20.16),(16.5,FW_h-20.16),(16.5,FW_h+20.16),(0,FW_h+20.16)])
        # Área de penalidade direita
        add_line([(field_length,FW_h-20.16),(field_length-16.5,FW_h-20.16),
                  (field_length-16.5,FW_h+20.16),(field_length,FW_h+20.16)])
        # Área pequena esquerda (5.5 × 18.32)
        add_line([(0,FW_h-9.16),(5.5,FW_h-9.16),(5.5,FW_h+9.16),(0,FW_h+9.16)])
        # Área pequena direita
        add_line([(field_length,FW_h-9.16),(field_length-5.5,FW_h-9.16),
                  (field_length-5.5,FW_h+9.16),(field_length,FW_h+9.16)])
        # Gol esquerdo (7.32 × 2.44)
        folium.PolyLine([pt(-2.44-cx,FW_h-3.66-cy), pt(0-cx,FW_h-3.66-cy),
                         pt(0-cx,FW_h+3.66-cy), pt(-2.44-cx,FW_h+3.66-cy),
                         pt(-2.44-cx,FW_h-3.66-cy)],
                        color="#FFD700", weight=3, opacity=1).add_to(campo_group)
        # Gol direito
        folium.PolyLine([pt(field_length+2.44-cx,FW_h-3.66-cy), pt(field_length-cx,FW_h-3.66-cy),
                         pt(field_length-cx,FW_h+3.66-cy), pt(field_length+2.44-cx,FW_h+3.66-cy),
                         pt(field_length+2.44-cx,FW_h-3.66-cy)],
                        color="#FFD700", weight=3, opacity=1).add_to(campo_group)
        campo_group.add_to(m)

    # ---- Trajetória GPS colorida por velocidade ----
    BANDAS_VEL = [
        (0,   7,   "#2196F3", "Caminhada / Trote (<7 km/h)"),
        (7,   14,  "#4CAF50", "Corrida Leve (7-14 km/h)"),
        (14,  19,  "#FFEB3B", "Corrida Moderada (14-19 km/h)"),
        (19,  24,  "#FF9800", "Corrida Intensa (19-24 km/h)"),
        (24,  999, "#F44336", "Sprint (>24 km/h)"),
    ]

    # Subsample para performance (máx 8000 pontos)
    n = len(lats)
    step = max(1, n // 8000)
    lats_s = lats[::step]
    lons_s = lons[::step]
    vels_s = vels[::step]

    traj_group = folium.FeatureGroup(name="🏃 Trajetória GPS", show=True)

    # Desenha segmentos contíguos agrupados por banda
    def cor_banda(v):
        for vmin, vmax, cor, _ in BANDAS_VEL:
            if v < vmax:
                return cor
        return "#F44336"

    seg_lats, seg_lons, seg_cor = [lats_s[0]], [lons_s[0]], cor_banda(vels_s[0])
    for i in range(1, len(lats_s)):
        c = cor_banda(vels_s[i])
        if c == seg_cor:
            seg_lats.append(lats_s[i])
            seg_lons.append(lons_s[i])
        else:
            if len(seg_lats) > 1:
                folium.PolyLine(
                    list(zip(seg_lats, seg_lons)),
                    color=seg_cor, weight=3, opacity=0.85
                ).add_to(traj_group)
            seg_lats = [lats_s[i - 1], lats_s[i]]
            seg_lons = [lons_s[i - 1], lons_s[i]]
            seg_cor = c
    if len(seg_lats) > 1:
        folium.PolyLine(
            list(zip(seg_lats, seg_lons)),
            color=seg_cor, weight=3, opacity=0.85
        ).add_to(traj_group)

    # Marcador de início
    folium.CircleMarker(
        location=[lats_s[0], lons_s[0]],
        radius=7, color="white", fill=True, fill_color="#00E676",
        fill_opacity=1, tooltip="▶ Início"
    ).add_to(traj_group)

    # Marcador de fim
    folium.Marker(
        location=[lats_s[-1], lons_s[-1]],
        icon=folium.Icon(color="red", icon="flag"),
        tooltip="⏹ Fim"
    ).add_to(traj_group)

    traj_group.add_to(m)

    # ---- Marcador do centro do campo (sempre visível) ----
    folium.CircleMarker(
        location=[float(centro_lat), float(centro_lon)],
        radius=7, color="white", weight=2,
        fill=True, fill_color="#FFEB3B", fill_opacity=0.95,
        tooltip=f"🎯 Centro do campo — clique no mapa para mover<br>{float(centro_lat):.6f}, {float(centro_lon):.6f}"
    ).add_to(m)

    # ---- Legenda HTML ----
    legenda_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:rgba(0,0,0,0.75);
                padding:10px 14px;border-radius:8px;color:white;font-size:12px;line-height:1.7;">
      <b>Velocidade</b><br>
      <span style="color:#2196F3">&#9632;</span> &lt;7 km/h — Caminhada<br>
      <span style="color:#4CAF50">&#9632;</span> 7-14 km/h — Trote<br>
      <span style="color:#FFEB3B">&#9632;</span> 14-19 km/h — Corrida<br>
      <span style="color:#FF9800">&#9632;</span> 19-24 km/h — Corrida intensa<br>
      <span style="color:#F44336">&#9632;</span> &gt;24 km/h — Sprint
    </div>
    """
    m.get_root().html.add_child(folium.Element(legenda_html))

    folium.LayerControl(collapsed=False).add_to(m)

    return m


def criar_html_campo_interativo(lats_gps, lons_gps, vels_gps, atleta_nome, height=700):
    """
    Gera HTML auto-contido com Leaflet.js:
    - Satélite Esri (gratuito, sem API key)
    - Trajetória GPS colorida por velocidade
    - Campo de futebol interativo: clique para posicionar centro,
      sliders para rotação/tamanho em tempo real
    - Zero re-renders do Streamlit (toda interação é client-side em JavaScript)
    """
    n = len(lats_gps)
    step = max(1, n // 3000)
    pontos = [
        {"lat": round(lats_gps[i], 7),
         "lon": round(lons_gps[i], 7),
         "v":   round(float(vels_gps[i]) if i < len(vels_gps) else 0.0, 1)}
        for i in range(0, n, step)
    ]
    import json as _json
    lat_c  = round(float(np.median(lats_gps)), 7)
    lon_c  = round(float(np.median(lons_gps)), 7)
    pts_js = _json.dumps(pontos)
    aesc   = atleta_nome.replace('"', '\\"').replace("'", "\\'")
    map_h  = height - 115

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        "  <meta charset='utf-8'>\n"
        "  <link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>\n"
        "  <script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>\n"
        "  <style>\n"
        "    *{box-sizing:border-box;margin:0;padding:0}\n"
        "    body{background:#111;font-family:Arial,sans-serif;color:#eee}\n"
        f"    #map{{width:100%;height:{map_h}px}}\n"
        "    #panel{background:rgba(10,10,25,.95);padding:7px 12px;"
        "display:flex;flex-wrap:wrap;gap:12px;align-items:center;"
        "border-top:1px solid #333;min-height:55px}\n"
        "    .ctrl{display:flex;align-items:center;gap:6px;font-size:12px}\n"
        "    .ctrl label{color:#aaa;white-space:nowrap;min-width:110px}\n"
        "    .ctrl input[type=range]{width:100px;accent-color:#2196F3;cursor:pointer}\n"
        "    .val{color:#FFD700;font-weight:bold;min-width:32px;display:inline-block}\n"
        "    .btn{background:#2196F3;color:#fff;border:none;padding:5px 13px;"
        "border-radius:4px;cursor:pointer;font-size:12px;font-weight:bold;white-space:nowrap}\n"
        "    .btn:hover{background:#1565C0}\n"
        "    .btn.on{background:#4CAF50}\n"
        "    .btn.on:hover{background:#2E7D32}\n"
        "    #status{margin-left:auto;color:#90CAF9;font-size:11px;"
        "max-width:260px;text-align:right}\n"
        "    /* Overlay pane totalmente transparente a cliques — map.on('click') sempre dispara */\n"
        "    .leaflet-overlay-pane{pointer-events:none!important}\n"
        "    .leaflet-overlay-pane *{pointer-events:none!important}\n"
        "  </style>\n</head>\n<body>\n"
        "  <div id='map'></div>\n"
        "  <div id='panel'>\n"
        "    <button class='btn' id='btnC' onclick='toggleCampo()'>⚽ Mostrar Campo</button>\n"
        "    <div class='ctrl'>\n"
        f"      <label>📍 Lat centro:</label>\n"
        f"      <input type='number' id='inLat' value='{lat_c}' step='0.00005'\n"
        "        style='width:105px;background:#1a1a2e;color:#FFD700;border:1px solid #555;"
        "padding:3px 5px;border-radius:3px;font-size:12px' oninput='onCenter()'>\n"
        "    </div>\n"
        "    <div class='ctrl'>\n"
        f"      <label>📍 Lon centro:</label>\n"
        f"      <input type='number' id='inLon' value='{lon_c}' step='0.00005'\n"
        "        style='width:105px;background:#1a1a2e;color:#FFD700;border:1px solid #555;"
        "padding:3px 5px;border-radius:3px;font-size:12px' oninput='onCenter()'>\n"
        "    </div>\n"
        "    <div class='ctrl'>\n"
        "      <label>🧭 Rotação: <span class='val' id='rv'>0</span>°</label>\n"
        "      <input type='range' id='rot' min='0' max='359' value='0' oninput='onRot(this.value)'>\n"
        "    </div>\n"
        "    <div class='ctrl'>\n"
        "      <label>📏 Comprimento: <span class='val' id='flv'>105</span>m</label>\n"
        "      <input type='range' id='fl' min='95' max='115' value='105' oninput='onDim()'>\n"
        "    </div>\n"
        "    <div class='ctrl'>\n"
        "      <label>📏 Largura: <span class='val' id='fwv'>68</span>m</label>\n"
        "      <input type='range' id='fw' min='60' max='80' value='68' oninput='onDim()'>\n"
        "    </div>\n"
        "    <div class='ctrl'>\n"
        "      <label>📐 Margem: <span class='val' id='igv'>3</span>m</label>\n"
        "      <input type='range' id='ig' min='0' max='8' value='3' oninput='onDim()'>\n"
        "    </div>\n"
        "    <span id='status'>📍 Ajuste Lat/Lon acima para mover o centro · ↑↓ no teclado = ~5m</span>\n"
        "  </div>\n"
        "  <script>\n"
        f"  const PTS={pts_js};\n"
        f"  const map=L.map('map').setView([{lat_c},{lon_c}],17);\n"
        "  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',\n"
        "    {attribution:'Esri World Imagery',maxZoom:19}).addTo(map);\n"
        "  L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',\n"
        "    {attribution:'Esri Labels',opacity:0.6,maxZoom:19}).addTo(map);\n"
        "  // ---- Trajetória GPS ----\n"
        "  function vc(v){\n"
        "    if(v<7)return '#2196F3';\n"
        "    if(v<14)return '#4CAF50';\n"
        "    if(v<19)return '#FFEB3B';\n"
        "    if(v<24)return '#FF9800';\n"
        "    return '#F44336';\n"
        "  }\n"
        "  let seg=[],sc=null;\n"
        "  for(let i=0;i<PTS.length;i++){\n"
        "    const c=vc(PTS[i].v);\n"
        "    if(c===sc){seg.push([PTS[i].lat,PTS[i].lon]);}\n"
        "    else{\n"
        "      if(seg.length>1)L.polyline(seg,{color:sc,weight:3,opacity:.85,interactive:false}).addTo(map);\n"
        "      seg=i>0?[[PTS[i-1].lat,PTS[i-1].lon],[PTS[i].lat,PTS[i].lon]]:[[PTS[i].lat,PTS[i].lon]];\n"
        "      sc=c;\n"
        "    }\n"
        "  }\n"
        "  if(seg.length>1)L.polyline(seg,{color:sc,weight:3,opacity:.85,interactive:false}).addTo(map);\n"
        "  if(PTS.length>0){\n"
        "    L.circleMarker([PTS[0].lat,PTS[0].lon],\n"
        "      {radius:7,color:'white',fillColor:'#00E676',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "    L.circleMarker([PTS[PTS.length-1].lat,PTS[PTS.length-1].lon],\n"
        "      {radius:7,color:'white',fillColor:'#F44336',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "  }\n"
        "  // ---- Estado do campo ----\n"
        "  let campoOn=false;\n"
        f"  let cLat={lat_c},cLon={lon_c};\n"
        "  let rotD=0,fL=105,fW=68,fI=3;\n"
        "  const cl=L.layerGroup().addTo(map);\n"
        "  // ---- Marcador arrastável (L.Marker com divIcon — não depende de map.on click) ----\n"
        "  const cmIcon=L.divIcon({\n"
        "    className:'',\n"
        "    html:'<div id=\"cmDiv\" style=\"width:24px;height:24px;background:#FFEB3B;"\
        "border:3px solid white;border-radius:50%;cursor:grab;"\
        "box-shadow:0 2px 8px rgba(0,0,0,.7);margin:-12px 0 0 -12px;\"></div>',\n"
        "    iconSize:[24,24],iconAnchor:[12,12]\n"
        "  });\n"
        "  const cm=L.marker([cLat,cLon],{draggable:true,icon:cmIcon,zIndexOffset:1000}).addTo(map);\n"
        "  // Atualiza centro a partir dos inputs numéricos (sempre funciona)\n"
        "  function onCenter(){\n"
        "    const v1=parseFloat(document.getElementById('inLat').value);\n"
        "    const v2=parseFloat(document.getElementById('inLon').value);\n"
        "    if(isNaN(v1)||isNaN(v2))return;\n"
        "    cLat=v1;cLon=v2;\n"
        "    cm.setLatLng([cLat,cLon]);\n"
        "    map.panTo([cLat,cLon]);\n"
        "    drawField();\n"
        "    document.getElementById('status').textContent='🎯 '+cLat.toFixed(5)+', '+cLon.toFixed(5);\n"
        "  }\n"
        "  // Sincroniza inputs quando marcador é arrastado (bônus se funcionar no browser)\n"
        "  function syncInputs(){\n"
        "    document.getElementById('inLat').value=cLat.toFixed(6);\n"
        "    document.getElementById('inLon').value=cLon.toFixed(6);\n"
        "  }\n"
        "  cm.on('drag',function(e){\n"
        "    const ll=e.target.getLatLng();\n"
        "    cLat=ll.lat;cLon=ll.lng;\n"
        "    syncInputs();\n"
        "    if(campoOn)drawField();\n"
        "    document.getElementById('status').textContent='⟳ '+cLat.toFixed(5)+', '+cLon.toFixed(5);\n"
        "  });\n"
        "  cm.on('dragend',function(e){\n"
        "    const ll=e.target.getLatLng();\n"
        "    cLat=ll.lat;cLon=ll.lng;\n"
        "    syncInputs();\n"
        "    drawField();\n"
        "    document.getElementById('status').textContent='🎯 Centro: '+cLat.toFixed(5)+', '+cLon.toFixed(5);\n"
        "  });\n"
        "  // ---- Geometria do campo ----\n"
        "  function toR(d){return d*Math.PI/180;}\n"
        "  function geo(xO,yO){\n"
        "    const r=toR(rotD);\n"
        "    const nM=yO*Math.cos(r)-xO*Math.sin(r);\n"
        "    const eM=yO*Math.sin(r)+xO*Math.cos(r);\n"
        "    return[cLat+nM/111320,cLon+eM/(111320*Math.cos(toR(cLat)))];\n"
        "  }\n"
        "  function lc(pts){return pts.map(p=>geo(p[0]-fL/2,p[1]-fW/2));}\n"
        "  function pl(pts,opt){L.polyline(lc(pts),Object.assign({interactive:false},opt)).addTo(cl);}\n"
        "  function drawField(){\n"
        "    cl.clearLayers();\n"
        "    if(!campoOn)return;\n"
        "    const FL=fL,FW=fW,mg=fI;\n"
        "    const w={color:'white',weight:2,opacity:.9};\n"
        "    const wb={color:'white',weight:3,opacity:.95};\n"
        "    const wd={color:'white',weight:1,opacity:.7,dashArray:'4 4'};\n"
        "    const gd={color:'#FFD700',weight:3,opacity:1,interactive:false};\n"
        "    // Margem exterior (bounding box)\n"
        "    pl([[-mg,-mg],[FL+mg,-mg],[FL+mg,FW+mg],[-mg,FW+mg],[-mg,-mg]],\n"
        "       {color:'white',weight:1,opacity:.4,dashArray:'4 4'});\n"
        "    // Perímetro do campo\n"
        "    pl([[0,0],[FL,0],[FL,FW],[0,FW],[0,0]],wb);\n"
        "    // Linha central\n"
        "    pl([[FL/2,0],[FL/2,FW]],w);\n"
        "    // Área de penalidade esquerda (40.32 x 16.5)\n"
        "    var pa_w=40.32,pa_d=16.5;\n"
        "    var pa_y1=(FW-pa_w)/2,pa_y2=(FW+pa_w)/2;\n"
        "    pl([[0,pa_y1],[pa_d,pa_y1],[pa_d,pa_y2],[0,pa_y2]],w);\n"
        "    // Área de penalidade direita\n"
        "    pl([[FL,pa_y1],[FL-pa_d,pa_y1],[FL-pa_d,pa_y2],[FL,pa_y2]],w);\n"
        "    // Área pequena esquerda (18.32 x 5.5)\n"
        "    var ga_w=18.32,ga_d=5.5;\n"
        "    var ga_y1=(FW-ga_w)/2,ga_y2=(FW+ga_w)/2;\n"
        "    pl([[0,ga_y1],[ga_d,ga_y1],[ga_d,ga_y2],[0,ga_y2]],w);\n"
        "    // Área pequena direita\n"
        "    pl([[FL,ga_y1],[FL-ga_d,ga_y1],[FL-ga_d,ga_y2],[FL,ga_y2]],w);\n"
        "    // Gols (7.32 x 2.44) — dourado\n"
        "    var gw=7.32,gd2=2.44;\n"
        "    var gy1=(FW-gw)/2,gy2=(FW+gw)/2;\n"
        "    L.polyline([geo(-gd2-FL/2,gy1-FW/2),geo(-gd2-FL/2,gy2-FW/2),\n"
        "                geo(-FL/2,gy2-FW/2),geo(-FL/2,gy1-FW/2)],gd).addTo(cl);\n"
        "    L.polyline([geo(FL+gd2-FL/2,gy1-FW/2),geo(FL+gd2-FL/2,gy2-FW/2),\n"
        "                geo(FL-FL/2,gy2-FW/2),geo(FL-FL/2,gy1-FW/2)],gd).addTo(cl);\n"
        "    // Círculo central (r=9.15)\n"
        "    var N=48,r=9.15;\n"
        "    var cpts=[];\n"
        "    for(var i=0;i<=N;i++){\n"
        "      var a=2*Math.PI*i/N;\n"
        "      cpts.push(geo(FL/2+r*Math.sin(a)-FL/2, FW/2+r*Math.cos(a)-FW/2));\n"
        "    }\n"
        "    L.polyline(cpts,w).addTo(cl);\n"
        "    // Ponto central\n"
        "    L.circleMarker(geo(0,0),{radius:3,color:'white',fillColor:'white',\n"
        "      fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "    // Pontos de penalidade (11m)\n"
        "    L.circleMarker(geo(11-FL/2,-FW/2),{radius:3,color:'white',fillColor:'white',\n"
        "      fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "    L.circleMarker(geo(FL-11-FL/2,-FW/2),{radius:3,color:'white',fillColor:'white',\n"
        "      fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "    // Arcos de penalidade (fora da área)\n"
        "    var rA=9.15,pts1=[],pts2=[];\n"
        "    for(var i=0;i<=60;i++){\n"
        "      var a=-Math.PI/2+Math.PI*i/60;\n"
        "      var px1=11+rA*Math.cos(a);\n"
        "      var py1=FW/2+rA*Math.sin(a);\n"
        "      if(px1>pa_d){pts1.push(geo(px1-FL/2,py1-FW/2));}\n"
        "      var px2=FL-11+rA*Math.cos(Math.PI-a);\n"
        "      var py2=FW/2+rA*Math.sin(a);\n"
        "      if(px2<FL-pa_d){pts2.push(geo(px2-FL/2,py2-FW/2));}\n"
        "    }\n"
        "    if(pts1.length>1)L.polyline(pts1,w).addTo(cl);\n"
        "    if(pts2.length>1)L.polyline(pts2,w).addTo(cl);\n"
        "    // Arcos de canto (r=1)\n"
        "    var rc=1,corners=[[0,0],[FL,0],[0,FW],[FL,FW]];\n"
        "    var aStart=[[0,Math.PI/2],[Math.PI/2,Math.PI],[3*Math.PI/2,2*Math.PI],\n"
        "                [Math.PI,3*Math.PI/2]];\n"
        "    corners.forEach(function(c,i){\n"
        "      var cpts2=[],a0=aStart[i][0],a1=aStart[i][1];\n"
        "      for(var j=0;j<=12;j++){\n"
        "        var a=a0+(a1-a0)*j/12;\n"
        "        cpts2.push(geo(c[0]+rc*Math.cos(a)-FL/2,c[1]+rc*Math.sin(a)-FW/2));\n"
        "      }\n"
        "      L.polyline(cpts2,w).addTo(cl);\n"
        "    });\n"
        "  }\n"
        "  function onRot(v){rotD=+v;document.getElementById('rv').textContent=v;drawField();}\n"
        "  function onDim(){\n"
        "    fL=+document.getElementById('fl').value;\n"
        "    fW=+document.getElementById('fw').value;\n"
        "    fI=+document.getElementById('ig').value;\n"
        "    document.getElementById('flv').textContent=fL;\n"
        "    document.getElementById('fwv').textContent=fW;\n"
        "    document.getElementById('igv').textContent=fI;\n"
        "    drawField();\n"
        "  }\n"
        "  function toggleCampo(){\n"
        "    campoOn=!campoOn;\n"
        "    const b=document.getElementById('btnC');\n"
        "    b.textContent=campoOn?'⚽ Ocultar Campo':'⚽ Mostrar Campo';\n"
        "    b.className=campoOn?'btn on':'btn';\n"
        "    drawField();\n"
        "    document.getElementById('status').textContent=campoOn\n"
        "      ?'✅ Arraste ⊙ para mover · sliders para ajustar em tempo real'\n"
        "      :'⊙ Arraste o marcador amarelo para o centro do campo';\n"
        "  }\n"
        "  const leg=L.control({position:'bottomleft'});\n"
        "  leg.onAdd=function(){\n"
        "    const d=L.DomUtil.create('div');\n"
        "    d.style='background:rgba(0,0,0,.78);padding:8px 11px;border-radius:6px;color:#fff;font-size:11px;line-height:1.9';\n"
        "    d.innerHTML='<b>Velocidade</b><br>'\n"
        "      +'<span style=\"color:#2196F3\">■</span> &lt;7 km/h Caminhada<br>'\n"
        "      +'<span style=\"color:#4CAF50\">■</span> 7-14 km/h Trote<br>'\n"
        "      +'<span style=\"color:#FFEB3B\">■</span> 14-19 km/h Corrida<br>'\n"
        "      +'<span style=\"color:#FF9800\">■</span> 19-24 km/h Intensa<br>'\n"
        "      +'<span style=\"color:#F44336\">■</span> &gt;24 km/h Sprint';\n"
        "    L.DomEvent.disableClickPropagation(d);\n"
        "    return d;\n"
        "  };\n"
        "  leg.addTo(map);\n"
        "  </script>\n</body>\n</html>"
    )
    return html


def criar_html_campo_fixo(lats_gps, lons_gps, vels_gps, campo_config,
                          lats_eff=None, lons_eff=None, vels_eff=None,
                          atleta_nome="", esforco_desc="", height=580):
    """
    Mapa satélite com campo de futebol FIXO (já configurado e aplicado).
    - Trajetória GPS completa como fundo (opacidade baixa)
    - Campo de futebol desenhado na posição salva
    - Se lats_eff fornecido: destaca os pontos do esforço selecionado (linha grossa)
    """
    import json as _json

    # Sub-amostra o trace de fundo (máx 3000 pontos)
    n = len(lats_gps)
    step = max(1, n // 3000)
    pontos = [
        {"lat": round(lats_gps[i], 7),
         "lon": round(lons_gps[i], 7),
         "v":   round(float(vels_gps[i]) if i < len(vels_gps) else 0.0, 1)}
        for i in range(0, n, step)
    ]

    # Pontos do esforço destacado
    eff_pontos = []
    if lats_eff and lons_eff:
        eff_pontos = [
            {"lat": round(lats_eff[i], 7),
             "lon": round(lons_eff[i], 7),
             "v":   round(float(vels_eff[i]) if vels_eff and i < len(vels_eff) else 0.0, 1)}
            for i in range(len(lats_eff))
        ]

    cLat = campo_config['lat']
    cLon = campo_config['lon']
    rotD = campo_config['rot']
    fL   = campo_config['fl']
    fW   = campo_config['fw']
    fI   = campo_config['ig']

    pts_js   = _json.dumps(pontos)
    eff_js   = _json.dumps(eff_pontos)
    aesc     = atleta_nome.replace('"', '\\"').replace("'", "\\'")
    desc_esc = esforco_desc.replace('"', '\\"').replace("'", "\\'")
    map_h    = height - 44

    html = (
        "<!DOCTYPE html>\n<html>\n<head>\n"
        "  <meta charset='utf-8'>\n"
        "  <link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>\n"
        "  <script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>\n"
        "  <style>\n"
        "    *{box-sizing:border-box;margin:0;padding:0}\n"
        "    body{background:#111;font-family:Arial,sans-serif;color:#eee}\n"
        f"    #map{{width:100%;height:{map_h}px}}\n"
        "    #infobar{background:rgba(10,10,25,.93);padding:6px 14px;display:flex;"
        "align-items:center;gap:14px;border-top:1px solid #2a2a3a;min-height:44px;font-size:12px}\n"
        "    .lk{color:#4CAF50;font-weight:bold;font-size:13px}\n"
        "    .leaflet-overlay-pane{pointer-events:none!important}\n"
        "    .leaflet-overlay-pane *{pointer-events:none!important}\n"
        "  </style>\n</head>\n<body>\n"
        "  <div id='map'></div>\n"
        "  <div id='infobar'>\n"
        "    <span class='lk'>🔒 Campo Aplicado</span>\n"
        f"    <span style='color:#90CAF9'>📍 {cLat:.5f}, {cLon:.5f}</span>\n"
        f"    <span style='color:#aaa'>🧭 {rotD}°</span>\n"
        f"    <span style='color:#aaa'>📏 {fL}×{fW}m + margem {fI}m (FIFA)</span>\n"
        "    <span id='effinfo' style='color:#FFD700;margin-left:auto;font-weight:bold'></span>\n"
        "  </div>\n"
        "  <script>\n"
        f"  const PTS={pts_js};\n"
        f"  const EFF={eff_js};\n"
        f"  const cLat={cLat},cLon={cLon},rotD={rotD},fL={fL},fW={fW},fI={fI};\n"
        f"  const map=L.map('map').setView([cLat,cLon],17);\n"
        "  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',\n"
        "    {attribution:'Esri World Imagery',maxZoom:19}).addTo(map);\n"
        "  L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',\n"
        "    {attribution:'Esri Labels',opacity:0.6,maxZoom:19}).addTo(map);\n"
        "  function vc(v){\n"
        "    if(v<7)return '#2196F3';if(v<14)return '#4CAF50';\n"
        "    if(v<19)return '#FFEB3B';if(v<24)return '#FF9800';return '#F44336';\n"
        "  }\n"
        "  // ---- Trace de fundo (opacidade baixa) ----\n"
        "  let seg=[],sc=null;\n"
        "  for(let i=0;i<PTS.length;i++){\n"
        "    const c=vc(PTS[i].v);\n"
        "    if(c===sc){seg.push([PTS[i].lat,PTS[i].lon]);}\n"
        "    else{\n"
        "      if(seg.length>1)L.polyline(seg,{color:sc,weight:2,opacity:.28,interactive:false}).addTo(map);\n"
        "      seg=i>0?[[PTS[i-1].lat,PTS[i-1].lon],[PTS[i].lat,PTS[i].lon]]:[[PTS[i].lat,PTS[i].lon]];\n"
        "      sc=c;\n"
        "    }\n"
        "  }\n"
        "  if(seg.length>1)L.polyline(seg,{color:sc,weight:2,opacity:.28,interactive:false}).addTo(map);\n"
        "  // ---- Esforço destacado (linha grossa) ----\n"
        f"  const effDesc='{desc_esc}';\n"
        "  if(EFF.length>0){\n"
        "    let es=[],ec=null;\n"
        "    for(let i=0;i<EFF.length;i++){\n"
        "      const c=vc(EFF[i].v);\n"
        "      if(c===ec){es.push([EFF[i].lat,EFF[i].lon]);}\n"
        "      else{\n"
        "        if(es.length>1)L.polyline(es,{color:ec,weight:7,opacity:1,interactive:false}).addTo(map);\n"
        "        es=i>0?[[EFF[i-1].lat,EFF[i-1].lon],[EFF[i].lat,EFF[i].lon]]:[[EFF[i].lat,EFF[i].lon]];\n"
        "        ec=c;\n"
        "      }\n"
        "    }\n"
        "    if(es.length>1)L.polyline(es,{color:ec,weight:7,opacity:1,interactive:false}).addTo(map);\n"
        "    L.circleMarker([EFF[0].lat,EFF[0].lon],\n"
        "      {radius:9,color:'white',fillColor:'#00E676',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "    L.circleMarker([EFF[EFF.length-1].lat,EFF[EFF.length-1].lon],\n"
        "      {radius:9,color:'white',fillColor:'#F44336',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "    const midIdx=Math.floor(EFF.length/2);\n"
        "    map.setView([EFF[midIdx].lat,EFF[midIdx].lon],18);\n"
        "    document.getElementById('effinfo').textContent='⚡ '+effDesc;\n"
        "  } else {\n"
        "    if(PTS.length>0){\n"
        "      L.circleMarker([PTS[0].lat,PTS[0].lon],\n"
        "        {radius:6,color:'white',fillColor:'#00E676',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "      L.circleMarker([PTS[PTS.length-1].lat,PTS[PTS.length-1].lon],\n"
        "        {radius:6,color:'white',fillColor:'#F44336',fillOpacity:1,weight:2,interactive:false}).addTo(map);\n"
        "    }\n"
        "  }\n"
        "  // ---- Campo fixo ----\n"
        "  const cl=L.layerGroup().addTo(map);\n"
        "  function toR(d){return d*Math.PI/180;}\n"
        "  function geo(xO,yO){\n"
        "    const r=toR(rotD);\n"
        "    const nM=yO*Math.cos(r)-xO*Math.sin(r);\n"
        "    const eM=yO*Math.sin(r)+xO*Math.cos(r);\n"
        "    return[cLat+nM/111320,cLon+eM/(111320*Math.cos(toR(cLat)))];\n"
        "  }\n"
        "  function lc(pts){return pts.map(p=>geo(p[0]-fL/2,p[1]-fW/2));}\n"
        "  function pl(pts,opt){L.polyline(lc(pts),Object.assign({interactive:false},opt)).addTo(cl);}\n"
        "  (function(){\n"
        "    var w={color:'white',weight:2,opacity:.9};\n"
        "    var wb={color:'white',weight:3,opacity:.9};\n"
        "    var wd={color:'white',weight:1.5,opacity:.9};\n"
        "    var gd={color:'#FFD700',weight:3,opacity:1,interactive:false};\n"
        "    var i,a,pts,px,px2;\n"
        "    pl([[0,0],[fL,0],[fL,fW],[0,fW],[0,0]],wb);\n"
        "    pl([[fL/2,0],[fL/2,fW]],w);\n"
        "    pl([[0,fW/2-20.16],[16.5,fW/2-20.16],[16.5,fW/2+20.16],[0,fW/2+20.16]],w);\n"
        "    pl([[fL,fW/2-20.16],[fL-16.5,fW/2-20.16],[fL-16.5,fW/2+20.16],[fL,fW/2+20.16]],w);\n"
        "    pl([[0,fW/2-9.16],[5.5,fW/2-9.16],[5.5,fW/2+9.16],[0,fW/2+9.16]],w);\n"
        "    pl([[fL,fW/2-9.16],[fL-5.5,fW/2-9.16],[fL-5.5,fW/2+9.16],[fL,fW/2+9.16]],w);\n"
        "    pl([[-2.44,fW/2-3.66],[0,fW/2-3.66],[0,fW/2+3.66],[-2.44,fW/2+3.66],[-2.44,fW/2-3.66]],gd);\n"
        "    pl([[fL+2.44,fW/2-3.66],[fL,fW/2-3.66],[fL,fW/2+3.66],[fL+2.44,fW/2+3.66],[fL+2.44,fW/2-3.66]],gd);\n"
        "    pts=[];\n"
        "    for(i=0;i<=60;i++){a=i/60*2*Math.PI;pts.push([fL/2+9.15*Math.cos(a),fW/2+9.15*Math.sin(a)]);}\n"
        "    pl(pts,wd);\n"
        "    pts=[];\n"
        "    for(i=0;i<=60;i++){a=-Math.PI+i/60*2*Math.PI;px=11+9.15*Math.cos(a);if(px>16.5)pts.push([px,fW/2+9.15*Math.sin(a)]);}\n"
        "    if(pts.length>1)pl(pts,wd);\n"
        "    pts=[];\n"
        "    for(i=0;i<=60;i++){a=i/60*2*Math.PI;px2=(fL-11)+9.15*Math.cos(a);if(px2<fL-16.5)pts.push([px2,fW/2+9.15*Math.sin(a)]);}\n"
        "    if(pts.length>1)pl(pts,wd);\n"
        "    var corners=[[0,0,0,Math.PI/2],[fL,0,Math.PI/2,Math.PI],[fL,fW,Math.PI,3*Math.PI/2],[0,fW,3*Math.PI/2,2*Math.PI]];\n"
        "    for(var ci=0;ci<corners.length;ci++){var cx2=corners[ci][0],cy2=corners[ci][1],a1=corners[ci][2],a2=corners[ci][3];pts=[];for(i=0;i<=10;i++){a=a1+(a2-a1)*i/10;pts.push([cx2+Math.cos(a),cy2+Math.sin(a)]);}pl(pts,Object.assign({},w,{weight:1}));}\n"
        "    L.circleMarker(geo(0,0),{radius:3,color:'white',fillColor:'white',fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "    L.circleMarker(geo(11-fL/2,0),{radius:3,color:'white',fillColor:'white',fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "    L.circleMarker(geo(fL/2-11,0),{radius:3,color:'white',fillColor:'white',fillOpacity:1,weight:1,interactive:false}).addTo(cl);\n"
        "  })();\n"
        "  // ---- Legenda ----\n"
        "  const leg=L.control({position:'bottomleft'});\n"
        "  leg.onAdd=function(){\n"
        "    const d=L.DomUtil.create('div');\n"
        "    d.style='background:rgba(0,0,0,.78);padding:8px 11px;border-radius:6px;color:#fff;font-size:11px;line-height:1.9';\n"
        "    d.innerHTML='<b>Velocidade</b><br>'\n"
        "      +'<span style=\"color:#2196F3\">■</span> &lt;7 km/h Caminhada<br>'\n"
        "      +'<span style=\"color:#4CAF50\">■</span> 7-14 km/h Trote<br>'\n"
        "      +'<span style=\"color:#FFEB3B\">■</span> 14-19 km/h Corrida<br>'\n"
        "      +'<span style=\"color:#FF9800\">■</span> 19-24 km/h Intensa<br>'\n"
        "      +'<span style=\"color:#F44336\">■</span> &gt;24 km/h Sprint';\n"
        "    L.DomEvent.disableClickPropagation(d);\n"
        "    return d;\n"
        "  };\n"
        "  leg.addTo(map);\n"
        "  </script>\n</body>\n</html>"
    )
    return html


def lat_lon_to_xy(latitudes, longitudes):
    if len(latitudes) == 0:
        return [], []

    lat_ref = latitudes[0]
    lon_ref = longitudes[0]
    R = 6371000

    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    ref_lat_rad = np.radians(lat_ref)
    ref_lon_rad = np.radians(lon_ref)
    
    x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
    y = R * (lat_rad - ref_lat_rad)
    
    return x, y

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
            player_load += acc ** 2
            acc_list.append(acc)
        else:
            acc_list.append(0.0)

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

    duracao_min = len(sensor_points) * 0.1 / 60
    m_min = round(distancia_total / duracao_min, 1) if duracao_min > 0 else 0.0

    return {
        'Atleta': athlete_name,
        'Duração (min)': round(duracao_min, 1),
        'Distância (m)': round(distancia_total, 0),
        'Dist. 19-24 km/h (m)': round(dist_z4, 0),
        'Dist. > 19 km/h (m)': round(dist_hi, 0),
        'Dist. > 24 km/h (m)': round(dist_sprint, 0),
        'PlayerLoad': round(player_load, 0),
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

# PARTE 3 - FUNÇÕES DE HRV, JANELAS E ESFORÇOS



def calcular_janelas_discretas_10s(sensor_points, window_minutes, metric_name, band_filter=None):
    if not sensor_points or len(sensor_points) < 10:
        return [], []
    
    tempos = []
    valores = []
    tempo_inicial = None
    
    for ponto in sensor_points:
        if metric_name in ponto and ponto[metric_name] is not None:
            ts = ponto.get('ts', 0)
            cs = ponto.get('cs', 0)
            tempo = ts + (cs / 100) if cs else ts
            
            if tempo_inicial is None:
                tempo_inicial = tempo
            
            tempo_relativo = (tempo - tempo_inicial)
            
            if metric_name == 'v':
                valor = ponto[metric_name] * 3.6
            elif metric_name == 'a':
                valor = ponto[metric_name]
            elif metric_name == 'pl':
                valor = ponto[metric_name]
            else:
                valor = ponto[metric_name]
            
            if band_filter is not None:
                if 'velocity_bands' in band_filter and metric_name == 'v':
                    vel_kmh = ponto['v'] * 3.6
                    if vel_kmh < 10:
                        banda_atual = 1
                    elif vel_kmh < 15:
                        banda_atual = 2
                    elif vel_kmh < 20:
                        banda_atual = 3
                    elif vel_kmh < 25:
                        banda_atual = 4
                    elif vel_kmh < 30:
                        banda_atual = 5
                    elif vel_kmh < 35:
                        banda_atual = 6
                    else:
                        banda_atual = 7
                    
                    if banda_atual not in band_filter['velocity_bands']:
                        continue
                
                elif 'acceleration_bands' in band_filter and metric_name == 'a':
                    acc = ponto['a']
                    if acc > 2:
                        banda_atual = 3
                    elif acc > 1:
                        banda_atual = 2
                    elif acc > 0:
                        banda_atual = 1
                    elif acc == 0:
                        banda_atual = 0
                    elif acc > -1:
                        banda_atual = -1
                    elif acc > -2:
                        banda_atual = -2
                    else:
                        banda_atual = -3
                    
                    if banda_atual not in band_filter['acceleration_bands']:
                        continue
            
            tempos.append(tempo_relativo)
            valores.append(valor)
    
    if len(tempos) == 0:
        return [], []
    
    window_seconds = window_minutes * 60
    pontos_por_bloco = 100
    blocos_por_janela = int(window_seconds / 10)
    
    tempos_janela = []
    valores_media = []
    
    for bloco_idx in range(0, len(valores) // pontos_por_bloco):
        inicio_bloco = bloco_idx * pontos_por_bloco
        fim_bloco = min(inicio_bloco + pontos_por_bloco, len(valores))
        
        if fim_bloco - inicio_bloco >= 10:
            valores_bloco = valores[inicio_bloco:fim_bloco]
            media_bloco = np.mean(valores_bloco)
            
            if inicio_bloco < len(tempos):
                tempo_central = tempos[inicio_bloco + (fim_bloco - inicio_bloco)//2] / 60
                janela_id = bloco_idx // blocos_por_janela
                
                if len(tempos_janela) <= janela_id:
                    tempos_janela.append(0)
                    valores_media.append([])
                
                valores_media[janela_id].append(media_bloco)
                tempos_janela[janela_id] = tempo_central
    
    tempos_final = []
    valores_final = []
    for i in range(len(valores_media)):
        if len(valores_media[i]) == blocos_por_janela:
            tempos_final.append(tempos_janela[i])
            valores_final.append(np.mean(valores_media[i]))
    
    return tempos_final, valores_final

def calcular_distancia_janelas_discretas_10s(sensor_points, window_minutes):
    if not sensor_points or len(sensor_points) < 10:
        return [], []
    
    tempos = []
    distancias_acumuladas = []
    tempo_inicial = None
    distancia_acumulada = 0
    vel_anterior = None

    for ponto in sensor_points:
        if ponto.get('v') is not None:
            ts = ponto.get('ts', 0)
            cs = ponto.get('cs', 0)
            tempo = ts + (cs / 100) if cs else ts

            if tempo_inicial is None:
                tempo_inicial = tempo

            tempo_relativo = (tempo - tempo_inicial)
            v_ms = float(ponto['v'])

            if vel_anterior is not None:
                distancia_intervalo = ((vel_anterior + v_ms) / 2) * 0.1
                distancia_acumulada += distancia_intervalo
            vel_anterior = v_ms
            
            tempos.append(tempo_relativo)
            distancias_acumuladas.append(distancia_acumulada)
    
    if len(tempos) == 0:
        return [], []
    
    window_seconds = window_minutes * 60
    pontos_por_bloco = 100
    blocos_por_janela = int(window_seconds / 10)
    
    tempos_janela = []
    distancias_janela = []
    
    for bloco_idx in range(0, len(distancias_acumuladas) // pontos_por_bloco):
        inicio_bloco = bloco_idx * pontos_por_bloco
        fim_bloco = min(inicio_bloco + pontos_por_bloco, len(distancias_acumuladas))
        
        if fim_bloco - inicio_bloco >= 10:
            dist_inicio_bloco = distancias_acumuladas[inicio_bloco]
            dist_fim_bloco = distancias_acumuladas[fim_bloco - 1]
            dist_bloco = dist_fim_bloco - dist_inicio_bloco
            
            tempo_central = tempos[inicio_bloco + (fim_bloco - inicio_bloco)//2] / 60
            janela_id = bloco_idx // blocos_por_janela
            
            if len(tempos_janela) <= janela_id:
                tempos_janela.append(0)
                distancias_janela.append([])
            
            tempos_janela[janela_id] = tempo_central
            distancias_janela[janela_id].append(dist_bloco)
    
    tempos_final = []
    valores_final = []
    for i in range(len(distancias_janela)):
        if len(distancias_janela[i]) == blocos_por_janela:
            distancia_total_janela = sum(distancias_janela[i])
            tempo_janela_min = window_seconds / 60
            tempos_final.append(tempos_janela[i])
            valores_final.append(distancia_total_janela / tempo_janela_min)
    
    return tempos_final, valores_final

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
            'Banda': band,
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
            'Banda': band,
            '_start_ts': start_time,
            '_end_ts': end_time
        })
    
    return pd.DataFrame(records)

# ── Métricas que devem ser SOMADAS ao combinar períodos ──────────────────────
_METRICAS_SUM = {
    'Duração (min)', 'Distância (m)', 'Dist. 19-24 km/h (m)',
    'Dist. > 19 km/h (m)', 'Dist. > 24 km/h (m)', 'PlayerLoad',
    'Sprints (>24 km/h)', 'Esforços Alta Int.', 'Acc 2-3 (m/s²)',
    'Dcc 2-3 (m/s²)', 'Acelerações (>3 m/s²)', 'Desacelerações (<-3 m/s²)',
    'RHIE Blocos', 'Total Pontos',
}
# ── Métricas que devem manter o MÁXIMO registrado ────────────────────────────
_METRICAS_MAX = {
    'Velocidade Máx (km/h)', 'FC Máx (bpm)', 'Acc Max (m/s²)', 'Dcc Max (m/s²)',
}


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

    for banda_id, bcfg in BANDAS_VEL.items():
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
                    pass

            records.append({
                'Esforço':            esf_num,
                'Início':             hora_str,
                'Duração (s)':        round(dur_s, 1),
                'Vel. Máx (km/h)':    round(vel_max, 1),
                'Vel. Inicial (km/h)': round(vel_ini, 1),
                'Distância (m)':      round(seg_dist, 1),
                '% do Máximo':        pct_max,
                'Banda':              banda_id,
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

    for banda_id, bcfg in BANDAS_ACC.items():
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
                    pass

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
                'Banda':          banda_id,
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


# PARTE 4 - FUNÇÕES DE GRÁFICOS, INTENSIDADE E CLASSIFICAÇÃO

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
            pass
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title=f"Aceleração ao Longo do Tempo - {athlete_name}",
        xaxis_title="Tempo (minutos)",
        yaxis_title="Aceleração (m/s²)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def classificar_intensidade(valores, percentil_50, percentil_75):
    cores = []
    classificacoes = []
    
    for valor in valores:
        if valor >= percentil_75:
            cores.append('orange')
            classificacoes.append('Alta Intensidade 🟠')
        elif valor >= percentil_50:
            cores.append('gold')
            classificacoes.append('Média-Alta Intensidade 🟡')
        else:
            cores.append('green')
            classificacoes.append('Baixa Intensidade 🟢')
    
    return cores, classificacoes

def criar_grafico_intensidade(tempos, valores, cores, metric_name, athlete_name, window_minutes, unidade):
    if not tempos or not valores:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tempos,
        y=valores,
        mode='lines+markers',
        name=f'{metric_name} (janela {window_minutes} min)',
        line=dict(color='lightgray', width=1),
        marker=dict(size=8, color=cores, line=dict(width=1, color='black'))
    ))
    
    fig.update_layout(
        title=f"Intensidade de {metric_name} - {athlete_name} (Janela: {window_minutes} min | Blocos de 10s)",
        xaxis_title="Tempo (minutos)",
        yaxis_title=f"{metric_name} ({unidade})",
        height=500,
        hovermode='closest'
    )
    
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                       text="🟢 Baixa (<50%) | 🟡 Média-Alta (50-75%) | 🟠 Alta (>75%)",
                       showarrow=False, font=dict(size=10))
    
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

# PARTE 5 - FUNÇÃO MAIN COMPLETA

def exibir_resultados_janela(tempos_janela, valores_janela, nome_metrica, atleta_janela, window_minutes, unidade):
    if not tempos_janela or not valores_janela:
        st.warning("Dados insuficientes para calcular as janelas")
        return

    valores_array = np.array(valores_janela)
    percentil_50 = np.percentile(valores_array, 50)
    percentil_75 = np.percentile(valores_array, 75)

    cores, classificacoes = classificar_intensidade(valores_janela, percentil_50, percentil_75)

    fig = criar_grafico_intensidade(tempos_janela, valores_janela, cores, nome_metrica, atleta_janela, window_minutes, unidade)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        alta_count = sum(1 for c in classificacoes if 'Alta' in c and 'Média' not in c)
        st.metric("🔴 Eventos de Alta Intensidade", alta_count)
    with col2:
        media_alta_count = sum(1 for c in classificacoes if 'Média-Alta' in c)
        st.metric("🟡 Eventos de Média-Alta Intensidade", media_alta_count)

    st.markdown("#### 📋 Eventos de Média-Alta e Alta Intensidade")
    df_eventos = criar_tabela_intensidade(tempos_janela, valores_janela, classificacoes, nome_metrica, unidade)
    if not df_eventos.empty:
        st.dataframe(df_eventos, use_container_width=True, height=400)
        csv_eventos = df_eventos.to_csv(index=False)
        st.download_button(
            f"📥 Exportar Eventos - {nome_metrica} (CSV)",
            csv_eventos,
            f"eventos_{nome_metrica}_{atleta_janela}_{window_minutes}min.csv"
        )
    else:
        st.info("Nenhum evento de média-alta ou alta intensidade encontrado")


# ==================== FEATURE 1: HEATMAP TEMPORAL SEGMENTADO ====================

def gerar_heatmap_segmentado(xs, ys, ts_list, bloco_min, bloco_idx, field_length=105, field_width=68):
    """Heatmap de presença para um bloco temporal específico da partida."""
    if not ts_list or not xs or not ys:
        return None, 0, ""

    ts_arr = np.array(ts_list, dtype=float)
    xs_arr = np.array(xs,      dtype=float)
    ys_arr = np.array(ys,      dtype=float)

    ts_rel  = ts_arr - ts_arr.min()
    bloco_s = bloco_min * 60.0
    t0, t1  = bloco_idx * bloco_s, (bloco_idx + 1) * bloco_s
    label   = f"{int(t0 // 60)}–{int(t1 // 60)} min"
    mascara = (ts_rel >= t0) & (ts_rel < t1)

    xb = xs_arr[mascara]
    yb = ys_arr[mascara]
    # Filtra ambos com a mesma máscara para garantir mesmo comprimento
    _valido = (xb >= 0) & (xb <= field_length) & (yb >= 0) & (yb <= field_width)
    xb = xb[_valido]
    yb = yb[_valido]

    if len(xb) < 5:
        return None, int(mascara.sum()), label

    H, xe, ye = np.histogram2d(xb, yb, bins=[52, 34],
                                range=[[0, field_length], [0, field_width]])
    H = _gf(H, sigma=2.0)
    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2

    fig = desenhar_campo_futebol_bonito(field_length, field_width,
                                        title=f"🕐 Heatmap por Fase — {label}")
    fig.add_trace(go.Heatmap(
        x=xc, y=yc, z=H.T,
        colorscale=[[0, 'rgba(0,0,0,0)'],  [0.0001, '#1a237e'],
                    [0.3, '#1565C0'],        [0.6,    '#FFEB3B'],
                    [0.85,'#FF9800'],        [1,      '#F44336']],
        opacity=0.72, showscale=True,
        colorbar=dict(
            title=dict(text='Freq.', font=dict(color='white')),
            tickfont=dict(color='white'),
            x=1.01, thickness=12
        ),
        name=f'Bloco {bloco_idx + 1}',
        hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Freq: %{z:.0f}<extra></extra>'
    ))
    return fig, int(mascara.sum()), label


# ==================== FEATURE 2: DIAGRAMA DE VORONOI ====================

def enriquecer_esforcos_taticos(df: pd.DataFrame,
                               coords_atletas: dict,
                               field_length: float = 105) -> pd.DataFrame:
    """
    Adiciona colunas 'Zona' e 'Direção' ao DataFrame de esforços com
    base nas coordenadas x/y do atleta:
      - Zona: Defensivo / Meio / Ataque (terços do campo)
      - Direção: ⬆ Ofensivo / ⬇ Defensivo / ↔ Lateral
    """
    if df.empty:
        return df

    zonas, direcoes = [], []
    fl3 = field_length / 3.0

    for _, row in df.iterrows():
        atl  = str(row.get('Atleta', ''))
        c    = coords_atletas.get(atl, {})
        xn   = c.get('xn', [])
        tsn  = c.get('ts', [])
        x_ini = x_fim = None

        # Tier 0: índices de segmento (cálculo local)
        si = int(row.get('_seg_start_idx', -1))
        ei = int(row.get('_seg_end_idx',   -1))
        if 0 <= si < ei <= len(xn):
            x_ini = float(xn[si])
            x_fim = float(xn[ei - 1])

        # Tier 1: timestamps (esforços da API)
        if x_ini is None and tsn and row.get('_start_ts', 0) > 0:
            _ts_a = np.array(tsn)
            _msk  = (_ts_a >= row['_start_ts']) & (_ts_a <= row['_end_ts'])
            if _msk.any():
                _idxs = np.where(_msk)[0]
                x_ini = float(xn[_idxs[0]])
                x_fim = float(xn[_idxs[-1]])

        # Zona
        if x_ini is not None:
            if x_ini < fl3:
                zona = '🔴 Defensivo'
            elif x_ini < 2 * fl3:
                zona = '🟡 Meio'
            else:
                zona = '🟢 Ataque'
        else:
            zona = '—'

        # Direção
        if x_ini is not None and x_fim is not None:
            dx = x_fim - x_ini
            if dx > 5:
                direcao = '⬆ Ofensivo'
            elif dx < -5:
                direcao = '⬇ Defensivo'
            else:
                direcao = '↔ Lateral'
        else:
            direcao = '—'

        zonas.append(zona)
        direcoes.append(direcao)

    df = df.copy()
    df.insert(df.columns.get_loc('Banda') + 1 if 'Banda' in df.columns else len(df.columns),
              'Zona', zonas)
    df.insert(df.columns.get_loc('Zona') + 1, 'Direção', direcoes)
    return df


def calcular_voronoi_campo(posicoes_atletas, field_length=105, field_width=68, resolucao=1.5):
    """Diagrama de Voronoi (nearest-neighbor grid) — raio de ação por atleta."""
    centroids = {}
    for nome, dados in posicoes_atletas.items():
        xs = [x for x in dados.get('xs', []) if 0 <= x <= field_length]
        ys = [y for y in dados.get('ys', []) if 0 <= y <= field_width]
        if len(xs) > 10:
            centroids[nome] = (float(np.median(xs)), float(np.median(ys)))

    if len(centroids) < 2:
        return None

    names = list(centroids.keys())
    cx    = np.array([centroids[n][0] for n in names])
    cy    = np.array([centroids[n][1] for n in names])
    n     = len(names)

    gx = np.arange(0, field_length + resolucao, resolucao)
    gy = np.arange(0, field_width  + resolucao, resolucao)
    GX, GY = np.meshgrid(gx, gy)
    pts_flat = np.column_stack([GX.ravel(), GY.ravel()])
    tree     = cKDTree(np.column_stack([cx, cy]))
    _, zone_flat = tree.query(pts_flat)
    Z = zone_flat.reshape(GX.shape).astype(float)

    base_colors = ['#2196F3','#F44336','#4CAF50','#FF9800',
                   '#9C27B0','#00BCD4','#FFEB3B','#E91E63',
                   '#FF5722','#607D8B']
    cs = []
    for i in range(n):
        lo = i / n
        hi = min((i + 1) / n, 1.0)
        c  = base_colors[i % len(base_colors)]
        cs.append([lo, c])
        cs.append([hi, c])
    cs[0][0]  = 0.0
    cs[-1][0] = 1.0

    fig = desenhar_campo_futebol_bonito(field_length, field_width,
                                        title="🔷 Voronoi — Raio de Ação por Atleta")
    fig.add_trace(go.Heatmap(
        x=gx, y=gy, z=Z,
        colorscale=cs, opacity=0.38, showscale=False,
        zmin=0, zmax=max(n - 0.001, 1),
        hoverinfo='skip', name='Zonas'
    ))
    for i, nome in enumerate(names):
        cx_i, cy_i = centroids[nome]
        fig.add_trace(go.Scatter(
            x=[cx_i], y=[cy_i],
            mode='markers+text',
            marker=dict(size=14, color=base_colors[i % len(base_colors)],
                        symbol='diamond', line=dict(width=2, color='white')),
            text=[nome.split()[0]], textposition='top center',
            textfont=dict(color='white', size=10, family='Arial Black'),
            name=nome, showlegend=True
        ))
    fig.update_layout(
        legend=dict(bgcolor='rgba(0,0,30,.75)', font=dict(color='white'),
                    bordercolor='#555', borderwidth=1)
    )
    return fig


# ==================== FEATURE 6: CARGA NEUROMUSCULAR ====================

def calcular_carga_neuromuscular(sensor_points, limiar=2.0, min_dur_s=None):
    """Analisa esforços de acc/dec intensos como indicador de carga neuromuscular.
    Conta EVENTOS (entradas na zona sustentadas por min_dur_s), não amostras.
    """
    if not sensor_points:
        return None

    if min_dur_s is None:
        min_dur_s = get_min_dur_s()

    ts_l, acc_l, vel_l = [], [], []
    for p in sensor_points:
        if p.get('a') is not None and p.get('ts') is not None:
            ts_l.append(float(p['ts']))
            acc_l.append(float(p['a']))
            vel_l.append(float(p.get('v') or 0) * 3.6)

    if len(ts_l) < 10:
        return None

    ts_arr  = np.array(ts_l,  dtype=float)
    acc_arr = np.array(acc_l, dtype=float)
    vel_arr = np.array(vel_l, dtype=float)
    ts_rel  = ts_arr - ts_arr.min()
    duracao = float(ts_rel.max())

    lm = limiar * 0.65  # limiar médio

    # Máscaras de EVENTOS (um True por evento, no frame que completa a duração mínima)
    mask_hi_acc  = detectar_eventos_acc(acc_arr,  limiar, min_dur_s=min_dur_s, acima=True)
    mask_hi_dec  = detectar_eventos_acc(acc_arr,  limiar, min_dur_s=min_dur_s, acima=False)
    mask_med_acc = detectar_eventos_acc(acc_arr,  lm,    min_dur_s=min_dur_s, acima=True)  & ~mask_hi_acc
    mask_med_dec = detectar_eventos_acc(acc_arr,  lm,    min_dur_s=min_dur_s, acima=False) & ~mask_hi_dec

    t_bins = np.arange(0, duracao + 60, 60)
    n_bins = max(1, len(t_bins) - 1)

    def _epm(mask):
        """Eventos por minuto em cada bin."""
        return np.array([
            mask[(ts_rel >= t_bins[i]) & (ts_rel < t_bins[i + 1])].sum()
            for i in range(n_bins)
        ])

    t_mid = [(t_bins[i] + t_bins[i + 1]) / 2 / 60 for i in range(n_bins)]
    return {
        'ts_rel': ts_rel, 'acc': acc_arr, 'vel': vel_arr, 't_mid': t_mid,
        'hi_acc_min':  _epm(mask_hi_acc),  'hi_dec_min':  _epm(mask_hi_dec),
        'med_acc_min': _epm(mask_med_acc), 'med_dec_min': _epm(mask_med_dec),
        'total_hi_acc':  int(mask_hi_acc.sum()),  'total_hi_dec':  int(mask_hi_dec.sum()),
        'total_med_acc': int(mask_med_acc.sum()), 'total_med_dec': int(mask_med_dec.sum()),
        'limiar': limiar,
        'min_dur_s': min_dur_s,
    }


def plotar_carga_neuromuscular(dados, atleta_nome):
    """Painel Plotly 2×2 com análise de carga neuromuscular."""
    lim      = dados['limiar']
    t        = dados['t_mid']
    _dur_lbl = f"{dados.get('min_dur_s', _DEFAULT_MIN_DUR_S):.1f}s"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"🟢 Acelerações ≥{lim} m/s² / min",
            f"🔴 Desacelerações ≥{lim} m/s² / min",
            "📊 Aceleração × Velocidade",
            "⚡ Carga Neuromuscular Acumulada"
        ),
        vertical_spacing=0.18, horizontal_spacing=0.1
    )
    # Acelerações
    fig.add_trace(go.Bar(x=t, y=dados['hi_acc_min'],  name=f'Alta Acc ≥{lim}',
                         marker_color='#4CAF50', opacity=0.9), row=1, col=1)
    fig.add_trace(go.Bar(x=t, y=dados['med_acc_min'], name='Média Acc',
                         marker_color='#81C784', opacity=0.6), row=1, col=1)
    # Desacelerações
    fig.add_trace(go.Bar(x=t, y=dados['hi_dec_min'],  name=f'Alta Dec ≥{lim}',
                         marker_color='#F44336', opacity=0.9), row=1, col=2)
    fig.add_trace(go.Bar(x=t, y=dados['med_dec_min'], name='Média Dec',
                         marker_color='#E57373', opacity=0.6), row=1, col=2)
    # Scatter acc × vel
    step = max(1, len(dados['ts_rel']) // 2000)
    ac_s = dados['acc'][::step];  vl_s = dados['vel'][::step]
    c_s  = np.where(ac_s >= lim, '#4CAF50', np.where(ac_s <= -lim, '#F44336', '#90CAF9'))
    fig.add_trace(go.Scatter(
        x=vl_s, y=ac_s, mode='markers',
        marker=dict(size=3, color=c_s, opacity=0.5),
        name='Acc × Vel', showlegend=False
    ), row=2, col=1)
    fig.add_hline(y= lim, line_dash='dash', line_color='#4CAF50', opacity=0.6, row=2, col=1)
    fig.add_hline(y=-lim, line_dash='dash', line_color='#F44336', opacity=0.6, row=2, col=1)
    fig.add_hline(y=0,    line_color='white', opacity=0.3, row=2, col=1)
    # Carga acumulada
    dt       = np.diff(dados['ts_rel'], prepend=dados['ts_rel'][0])
    dt       = np.clip(dt, 0, 1)
    carga    = np.cumsum(np.abs(dados['acc']) * dt)
    step2    = max(1, len(carga) // 1500)
    fig.add_trace(go.Scatter(
        x=dados['ts_rel'][::step2] / 60, y=carga[::step2],
        mode='lines', line=dict(color='#FFEB3B', width=2),
        fill='tozeroy', fillcolor='rgba(255,235,59,0.12)',
        name='Carga Acum.'
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text=f'💪 Carga Neuromuscular — {atleta_nome}  (dur. mín. {_dur_lbl})',
                   font=dict(size=16, color='white')),
        height=620, paper_bgcolor='#0a0a1e', plot_bgcolor='#1a1a2e',
        font=dict(color='white'), barmode='stack',
        legend=dict(bgcolor='rgba(0,0,0,.5)', font=dict(color='white'))
    )
    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(gridcolor='#333', row=r, col=c)
            fig.update_yaxes(gridcolor='#333', row=r, col=c)
    fig.update_xaxes(title_text='Tempo (min)',        row=1, col=1)
    fig.update_xaxes(title_text='Tempo (min)',        row=1, col=2)
    fig.update_xaxes(title_text='Velocidade (km/h)',  row=2, col=1)
    fig.update_xaxes(title_text='Tempo (min)',        row=2, col=2)
    fig.update_yaxes(title_text='Contagem / min',     row=1, col=1)
    fig.update_yaxes(title_text='Contagem / min',     row=1, col=2)
    fig.update_yaxes(title_text='Aceleração (m/s²)',  row=2, col=1)
    fig.update_yaxes(title_text='Carga Acumulada',    row=2, col=2)
    return fig


# ==================== FEATURE 8: RELATÓRIO PDF ====================

def gerar_pdf_relatorio(atleta_nome, periodo_nome, metricas, sensor_points, dados_pos,
                        field_length=105, field_width=68):
    """Gera relatório PDF A3 landscape em memória via matplotlib. Retorna bytes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle as MplRect, Circle as MplCircle
    from matplotlib.gridspec import GridSpec

    BG   = '#0a0a1e'; PAN = '#1a1a2e'; GOLD = '#FFD700'
    BLU  = '#2196F3'; GRN = '#4CAF50'; ORG  = '#FF9800'
    RED  = '#F44336'; WHT = '#FFFFFF'; GRY  = '#90CAF9'

    plt.rcParams.update({
        'text.color': WHT, 'axes.labelcolor': WHT,
        'xtick.color': WHT, 'ytick.color': WHT,
        'axes.edgecolor': '#555', 'axes.facecolor': PAN,
        'figure.facecolor': BG,   'axes.grid': True,
        'grid.color': '#333',     'grid.alpha': 0.5,
        'font.size': 8,
    })

    fig = plt.figure(figsize=(16.54, 11.69), facecolor=BG)
    gs  = GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38,
                   left=0.04, right=0.97, top=0.92, bottom=0.06)

    # ── Cabeçalho ──────────────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_facecolor('#0d1117'); ax_h.axis('off')
    ax_h.text(0.5, 0.80, '⚽  RELATÓRIO DE PERFORMANCE — FUTEBOL',
              transform=ax_h.transAxes, ha='center', va='center',
              fontsize=20, fontweight='bold', color=GOLD)
    ax_h.text(0.5, 0.44, atleta_nome,
              transform=ax_h.transAxes, ha='center', va='center',
              fontsize=14, color=WHT)
    ax_h.text(0.5, 0.12,
              f'Período: {periodo_nome}   |   {datetime.now().strftime("%d/%m/%Y  %H:%M")}   |   Catapult Sports API v6',
              transform=ax_h.transAxes, ha='center', va='center',
              fontsize=8, color=GRY, style='italic')
    ax_h.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax_h.transAxes,
                                       fill=False, edgecolor=GOLD, linewidth=1.5))

    # ── Métricas biométricas ───────────────────────────────────────────────
    ax_m = fig.add_subplot(gs[1, 0]); ax_m.axis('off')
    ax_m.text(0.5, 0.97, '📊 MÉTRICAS BIOMÉTRICAS', transform=ax_m.transAxes,
              ha='center', va='top', fontsize=9, fontweight='bold', color=GRN)
    kpis = [
        ('Distância Total',    f"{metricas.get('Distância (m)', 0):,.0f} m",            BLU),
        ('PlayerLoad',         f"{metricas.get('PlayerLoad', 0):,.0f}",                 '#FFEB3B'),
        ('Vel. Máxima',        f"{metricas.get('Velocidade Máx (km/h)', 0):.1f} km/h",  ORG),
        ('Vel. Média',         f"{metricas.get('Velocidade Média (km/h)', 0):.1f} km/h", GRN),
        ('FC Média',           f"{metricas.get('FC Média (bpm)', 0):.0f} bpm",          RED),
        ('Dist. >19 km/h',     f"{metricas.get('Dist. > 19 km/h (m)', 0):.0f} m",      ORG),
        ('Dist. >24 km/h',     f"{metricas.get('Dist. > 24 km/h (m)', 0):.0f} m",      RED),
        ('Sprints (>24)',       f"{metricas.get('Sprints (>24 km/h)', 0)}",             '#E91E63'),
        ('Acels. (>3 m/s²)',   f"{metricas.get('Acelerações (>3 m/s²)', 0)}",          GRN),
    ]
    for i, (lbl, val, clr) in enumerate(kpis):
        y = 0.84 - i * 0.092
        ax_m.text(0.05, y, lbl + ':', transform=ax_m.transAxes,
                  ha='left', va='top', fontsize=7.5, color='#aaa')
        ax_m.text(0.95, y, val, transform=ax_m.transAxes,
                  ha='right', va='top', fontsize=8.5, fontweight='bold', color=clr)

    # ── Gráfico de velocidade ──────────────────────────────────────────────
    ax_v = fig.add_subplot(gs[1, 1:])
    if sensor_points:
        ts_v, vl_v = [], []
        for p in sensor_points:
            if p.get('ts') is not None and p.get('v') is not None:
                ts_v.append(float(p['ts'])); vl_v.append(float(p['v']) * 3.6)
        if ts_v:
            ta = np.array(ts_v); ta -= ta.min(); ta /= 60
            va = np.array(vl_v)
            wl = min(61, max(11, (len(va) // 100) * 2 + 1))
            if wl % 2 == 0: wl -= 1
            try:    vs = savgol_filter(va, wl, 3)
            except: vs = va
            c_pts = np.select([va < 7, va < 14, va < 19, va < 24],
                               [BLU, GRN, '#FFEB3B', ORG], default=RED)
            ax_v.scatter(ta[::3], va[::3], c=c_pts[::3], s=1.5, alpha=0.4, linewidths=0)
            ax_v.plot(ta, vs, color=WHT, lw=1.2, alpha=0.85)
            ax_v.axhline(24, color=RED, lw=0.9, ls='--', alpha=0.7, label='Sprint 24 km/h')
            ax_v.axhline(19, color=ORG, lw=0.9, ls='--', alpha=0.7, label='Alta Int. 19 km/h')
            ax_v.legend(fontsize=7, loc='upper right',
                        facecolor=PAN, labelcolor=WHT, edgecolor='#555')
    ax_v.set_xlabel('Tempo (min)'); ax_v.set_ylabel('Velocidade (km/h)')
    ax_v.set_title('📈 Perfil de Velocidade', fontsize=10, pad=6, color=WHT)
    ax_v.spines[['top', 'right']].set_visible(False)

    # ── Mapa de calor no campo ─────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[2, :2])
    ax_f.set_facecolor('#1a4a2e')
    ax_f.set_xlim(-2, field_length + 2); ax_f.set_ylim(-2, field_width + 2)
    ax_f.set_aspect('equal')
    ax_f.set_title('🗺️ Mapa de Calor — Posicionamento', fontsize=9, pad=5, color=WHT)
    for xy in [(0, 0, field_length, field_width),
               (0, (field_width - 40.32) / 2, 16.5,            40.32),
               (field_length - 16.5, (field_width - 40.32) / 2, 16.5, 40.32)]:
        ax_f.add_patch(MplRect((xy[0], xy[1]), xy[2], xy[3],
                               fill=False, edgecolor='white', lw=0.9, alpha=0.9))
    ax_f.axvline(field_length / 2, color='white', lw=0.7, alpha=0.7)
    ax_f.add_patch(MplCircle((field_length / 2, field_width / 2), 9.15,
                              fill=False, edgecolor='white', lw=0.7, alpha=0.7))
    if dados_pos and 'xs' in dados_pos and 'ys' in dados_pos:
        xf = np.array([x for x in dados_pos['xs'] if 0 <= x <= field_length])
        yf = np.array([y for y in dados_pos['ys'] if 0 <= y <= field_width])
        if len(xf) > 10:
            H, xe, ye = np.histogram2d(xf, yf, bins=[52, 34],
                                       range=[[0, field_length], [0, field_width]])
            H = _gf(H, sigma=2.0)
            Hm = np.ma.masked_where(H.T < H.max() * 0.02, H.T)
            ax_f.pcolormesh(xe, ye, Hm, cmap='hot', alpha=0.72, vmin=0)
    ax_f.set_xlabel('Comprimento (m)'); ax_f.set_ylabel('Largura (m)')

    # ── Sprints resumo ─────────────────────────────────────────────────────
    ax_s = fig.add_subplot(gs[2, 2]); ax_s.axis('off')
    ax_s.text(0.5, 0.97, '⚡ SPRINTS & ACELERAÇÕES', transform=ax_s.transAxes,
              ha='center', va='top', fontsize=9, fontweight='bold', color=ORG)
    sprint_kpis = [
        ('N° Sprints >24 km/h', f"{metricas.get('Sprints (>24 km/h)', 0)}",              RED),
        ('Dist. em Sprint',     f"{metricas.get('Dist. > 24 km/h (m)', 0):.0f} m",       RED),
        ('Esforços >19 km/h',   f"{metricas.get('Esforços Alta Int.', 0)}",              ORG),
        ('Dist. Alta Int.',     f"{metricas.get('Dist. > 19 km/h (m)', 0):.0f} m",       ORG),
        ('Acels. Intensas',     f"{metricas.get('Acelerações (>3 m/s²)', 0)}",           GRN),
        ('Desacels. Intensas',  f"{metricas.get('Desacelerações (<-3 m/s²)', 0)}",       '#9C27B0'),
    ]
    for i, (lbl, val, clr) in enumerate(sprint_kpis):
        y = 0.83 - i * 0.13
        ax_s.text(0.05, y, lbl + ':', transform=ax_s.transAxes,
                  ha='left', va='top', fontsize=7.5, color='#aaa')
        ax_s.text(0.95, y, val, transform=ax_s.transAxes,
                  ha='right', va='top', fontsize=9, fontweight='bold', color=clr)

    # ── Aceleração ao longo do tempo ───────────────────────────────────────
    ax_a = fig.add_subplot(gs[2, 3])
    if sensor_points:
        ts_a, ac_a = [], []
        for p in sensor_points:
            if p.get('ts') is not None and p.get('a') is not None:
                ts_a.append(float(p['ts'])); ac_a.append(float(p['a']))
        if ts_a:
            ta2 = np.array(ts_a); ta2 -= ta2.min(); ta2 /= 60
            aa  = np.array(ac_a)
            wl2 = min(31, max(5, (len(aa) // 200) * 2 + 1))
            if wl2 % 2 == 0: wl2 -= 1
            try:    as_ = savgol_filter(aa, wl2, 2)
            except: as_ = aa
            ax_a.fill_between(ta2[::3], as_[::3], 0,
                              where=as_[::3] >= 0, color=GRN, alpha=0.5, label='Acc')
            ax_a.fill_between(ta2[::3], as_[::3], 0,
                              where=as_[::3] < 0,  color=RED, alpha=0.5, label='Dec')
            ax_a.axhline(0,  color=WHT, lw=0.6, alpha=0.5)
            ax_a.axhline(3,  color=GRN, lw=0.8, ls='--', alpha=0.6)
            ax_a.axhline(-3, color=RED, lw=0.8, ls='--', alpha=0.6)
            ax_a.legend(fontsize=7, facecolor=PAN, labelcolor=WHT, edgecolor='#555')
    ax_a.set_xlabel('Tempo (min)'); ax_a.set_ylabel('Aceleração (m/s²)')
    ax_a.set_title('🔄 Perfil de Aceleração', fontsize=9, pad=5, color=WHT)
    ax_a.spines[['top', 'right']].set_visible(False)

    fig.text(0.5, 0.005,
             '⚽  Futebol Eventos — Powered by Catapult Sports API v6 | Claude AI',
             ha='center', fontsize=7, color='#444', style='italic')

    buf = io.BytesIO()
    plt.savefig(buf, format='pdf', bbox_inches='tight', facecolor=BG, dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ==================== FEATURE 10: ACWR ====================

def calcular_acwr_df(df_cargas):
    """
    ACWR (Acute:Chronic Workload Ratio) por atleta e data.
    df_cargas: DataFrame com colunas 'atleta', 'data' (datetime), 'player_load', 'atividade'.
    """
    resultados = []
    for atleta in df_cargas['atleta'].unique():
        df_a = df_cargas[df_cargas['atleta'] == atleta].sort_values('data')
        for _, row in df_a.iterrows():
            d_ref  = row['data']
            aguda  = df_a[(df_a['data'] <= d_ref) &
                           (df_a['data'] >  d_ref - timedelta(days=7))]['player_load'].sum()
            cronica = df_a[(df_a['data'] <= d_ref) &
                            (df_a['data'] >  d_ref - timedelta(days=28))]['player_load'].sum()
            cr_sem = cronica / 4 if cronica > 0 else 0
            acwr   = round(aguda / cr_sem, 3) if cr_sem > 0 else None
            resultados.append({
                'Atleta': atleta,
                'Data': d_ref,
                'Atividade': row.get('atividade', ''),
                'PlayerLoad': round(row['player_load'], 1),
                'Carga Aguda 7d': round(aguda, 1),
                'Carga Crônica 28d': round(cronica, 1),
                'ACWR': acwr,
            })
    return pd.DataFrame(resultados)


def plotar_acwr(df_acwr):
    """Gráfico de ACWR com zonas de risco coloridas."""
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('📊 PlayerLoad por Sessão', '🎯 ACWR — Índice de Carga Aguda/Crônica'),
                        vertical_spacing=0.15, shared_xaxes=True)
    cores = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0',
             '#00BCD4','#FFEB3B','#E91E63','#FF5722','#607D8B']
    atletas = df_acwr['Atleta'].unique()

    for i, atl in enumerate(atletas):
        df_a = df_acwr[df_acwr['Atleta'] == atl].copy()
        c    = cores[i % len(cores)]
        fig.add_trace(go.Bar(
            x=df_a['Data'], y=df_a['PlayerLoad'],
            name=atl, marker_color=c, opacity=0.8,
            hovertemplate='%{x|%d/%m/%y}<br>PL: %{y:.0f}<extra>' + atl + '</extra>'
        ), row=1, col=1)
        df_acwr_v = df_a.dropna(subset=['ACWR'])
        if not df_acwr_v.empty:
            fig.add_trace(go.Scatter(
                x=df_acwr_v['Data'], y=df_acwr_v['ACWR'],
                mode='lines+markers', name=atl + ' ACWR',
                line=dict(color=c, width=2), marker=dict(size=6),
                showlegend=False,
                hovertemplate='%{x|%d/%m/%y}<br>ACWR: %{y:.2f}<extra>' + atl + '</extra>'
            ), row=2, col=1)

    # Zonas de risco no gráfico ACWR
    acwr_vals = df_acwr.dropna(subset=['ACWR'])['ACWR']
    y_max = max(float(acwr_vals.max()) * 1.2, 2.0) if len(acwr_vals) else 2.0
    for y0, y1, cor, label in [
        (0.0, 0.8,  'rgba(33,150,243,.10)',  'Subcarregado'),
        (0.8, 1.3,  'rgba(76,175,80,.15)',   '✅ Zona Ótima'),
        (1.3, 1.5,  'rgba(255,152,0,.15)',   '⚠️ Atenção'),
        (1.5, y_max,'rgba(244,67,54,.15)',   '🔴 Risco'),
    ]:
        fig.add_hrect(y0=y0, y1=min(y1, y_max), fillcolor=cor,
                      line_width=0, row=2, col=1, annotation_text=label,
                      annotation_position='right',
                      annotation_font=dict(size=9, color='white'))
    fig.add_hline(y=1.5, line_dash='dash', line_color='#F44336', opacity=0.8, row=2, col=1)
    fig.add_hline(y=1.3, line_dash='dash', line_color='#FF9800', opacity=0.7, row=2, col=1)
    fig.add_hline(y=0.8, line_dash='dash', line_color='#2196F3', opacity=0.6, row=2, col=1)

    fig.update_layout(
        height=650, paper_bgcolor='#0a0a1e', plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        legend=dict(bgcolor='rgba(0,0,0,.5)', font=dict(color='white')),
        barmode='group',
        xaxis2=dict(title='Data', gridcolor='#333'),
        yaxis=dict(title='PlayerLoad', gridcolor='#333'),
        yaxis2=dict(title='ACWR', gridcolor='#333'),
    )
    return fig


def main():
    st.title(t("app_title"))
    st.markdown(f"### {t('app_subtitle')}")

    # Inicializar session state
    if 'df_athletes' not in st.session_state:
        st.session_state.df_athletes = pd.DataFrame()
    if 'df_activities' not in st.session_state:
        st.session_state.df_activities = pd.DataFrame()
    if 'df_positions' not in st.session_state:
        st.session_state.df_positions = pd.DataFrame()
    if 'df_teams' not in st.session_state:
        st.session_state.df_teams = pd.DataFrame()
    if 'df_parameters' not in st.session_state:
        st.session_state.df_parameters = pd.DataFrame()
    if 'athlete_team_map' not in st.session_state:
        st.session_state.athlete_team_map = {}
    
    # ── Carregar preferências salvas ─────────────────────────────────
    _prefs = _carregar_prefs()

    # Sidebar
    with st.sidebar:

        # ── FEATURE 5: Tour guiado para novos usuários ───────────────
        if not _prefs.get('tour_done'):
            with st.expander("📖 Como usar — Guia Rápido", expanded=True):
                st.markdown("""
**Bem-vindo! Siga estes passos:**

**1️⃣ Servidor & Token**
Selecione o servidor da sua região e cole o token JWT do OpenField.

**2️⃣ Carregar Dados**
Clique em **🔄 Carregar Dados** — equipes, atletas e atividades serão buscados automaticamente.

**3️⃣ Filtrar**
Filtre por equipe e posição, selecione a atividade e os períodos desejados.

**4️⃣ Selecionar Atletas**
Escolha um ou mais atletas para análise simultânea.

**5️⃣ Explorar as ABAs**
- 📈 Gráficos Comparativos
- 🗺️ Campo de Futebol (posição, esforços, animação)
- ⏱️ Esforços ao Longo do Tempo
- 📊 Janelas Temporais Móveis
- 💪 Carga Neuromuscular
- 🏎️ Perfil Aceleração × Velocidade
""")
                if st.button("✅ Entendi, não mostrar novamente", key="_tour_ok"):
                    _prefs['tour_done'] = True
                    _salvar_prefs(_prefs)
                    st.rerun()

        st.header("🌍 Servidor")
        # ── FEATURE 2: pré-seleciona servidor salvo nas prefs ────────
        _server_opts  = list(SERVERS.keys())
        _server_saved = _prefs.get('server', _server_opts[0])
        _server_idx   = _server_opts.index(_server_saved) if _server_saved in _server_opts else 0
        server = st.selectbox("Selecione:", _server_opts, index=_server_idx)
        base_url = SERVERS[server]
        # Salva imediatamente se mudou
        if server != _prefs.get('server'):
            _prefs['server'] = server
            _salvar_prefs(_prefs)

        st.header(t("language_header"))
        st.selectbox(t("language_select"), list(LANGUAGES.keys()), key="lang_selector")

        st.header("🔐 Token")
        token = st.text_area("Token JWT:", height=80)
        
        if token and st.button("🔄 Carregar Dados", type="primary"):
            with st.spinner("Carregando..."):
                api = CatapultAPI(base_url, token)
                
                st.subheader("📋 Carregando Equipes...")
                teams_raw = api.get_teams()
                if teams_raw:
                    teams_data = []
                    for team in teams_raw:
                        teams_data.append({'id': team.get('id'), 'nome': team.get('name'), 'slug': team.get('slug')})
                    st.session_state.df_teams = pd.DataFrame(teams_data)
                    st.success(f"✅ {len(teams_data)} equipes carregadas")
                    
                    st.subheader("📋 Mapeando atletas por equipe...")
                    athlete_team_map = {}
                    for _, team in st.session_state.df_teams.iterrows():
                        team_athletes = api.get_team_athletes(team['id'])
                        if team_athletes:
                            if isinstance(team_athletes, dict):
                                team_athletes = team_athletes.get('data', team_athletes.get('items', []))
                            for ath in team_athletes:
                                athlete_team_map[ath.get('id')] = team['nome']
                    st.session_state.athlete_team_map = athlete_team_map
                    st.success(f"✅ {len(athlete_team_map)} atletas mapeados")
                
                st.subheader("📋 Carregando Posições...")
                positions_raw = api.get_positions()
                if positions_raw:
                    positions_data = []
                    for p in positions_raw:
                        positions_data.append({'id': p.get('id'), 'nome': p.get('name'), 'slug': p.get('slug')})
                    st.session_state.df_positions = pd.DataFrame(positions_data)
                    st.success(f"✅ {len(positions_data)} posições carregadas")
                
                st.subheader("📋 Carregando Atletas...")
                athletes_raw = api.get_athletes()
                if athletes_raw:
                    atletas = []
                    for a in athletes_raw:
                        nome = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
                        if not nome:
                            nome = a.get('name', 'Sem nome')
                        position_name = ''
                        pos_id = a.get('position_id')
                        if pos_id and not st.session_state.df_positions.empty:
                            pos_row = st.session_state.df_positions[st.session_state.df_positions['id'] == pos_id]
                            if not pos_row.empty:
                                position_name = pos_row.iloc[0]['nome']
                        team_name = st.session_state.athlete_team_map.get(a.get('id'), '')
                        atletas.append({
                            'id': a.get('id'), 'nome': nome, 'camisa': a.get('jersey', ''),
                            'posicao': position_name, 'equipe': team_name,
                        })
                    st.session_state.df_athletes = pd.DataFrame(atletas)

                    # ── Auto-busca Vmax histórico via /stats (cached_stats) ──
                    # É o mesmo endpoint do botão fallback manual, mas chamado
                    # automaticamente logo após carregar a lista de atletas.
                    _hvm_auto = st.session_state.get('hist_vmax', {})
                    _src_auto = st.session_state.get('hist_vmax_source', {})
                    _n_vmax_auto = 0
                    try:
                        _stats_auto = api.get_stats({
                            "group_by": ["athlete"],
                            "parameters": ["max_velocity"],
                            "source": "cached_stats",
                        })
                        if _stats_auto:
                            _sa_list = (_stats_auto if isinstance(_stats_auto, list)
                                        else _stats_auto.get('data', []))
                            for _sa in (_sa_list or []):
                                _sa_name = str(
                                    _sa.get('athlete') or _sa.get('athlete_name') or
                                    _sa.get('name') or ''
                                )
                                # max_velocity pode estar em parâmetros aninhados ou direto
                                _sa_params = _sa.get('parameters') or _sa
                                _sa_val = float(
                                    _sa_params.get('max_velocity') or
                                    _sa_params.get('max_speed') or
                                    _sa.get('max_velocity') or
                                    _sa.get('max_speed') or 0
                                )
                                if _sa_name and _sa_val > 0:
                                    # Converte km/h → m/s se necessário (valor > 15 = provavelmente km/h)
                                    _sa_ms = _sa_val / 3.6 if _sa_val > 15 else _sa_val
                                    if _src_auto.get(_sa_name, '') != 'manual':
                                        _hvm_auto[_sa_name] = _sa_ms
                                        _src_auto[_sa_name] = 'catapult_stats'
                                        _n_vmax_auto += 1
                    except Exception:
                        pass
                    st.session_state['hist_vmax']        = _hvm_auto
                    st.session_state['hist_vmax_source'] = _src_auto
                    _vmax_msg = f" · {_n_vmax_auto} Vmax hist. detectadas" if _n_vmax_auto else ""
                    st.success(f"✅ {len(atletas)} atletas carregados{_vmax_msg}")
                
                st.subheader("📋 Carregando Atividades...")
                activities_raw = api.get_activities()
                if activities_raw:
                    if isinstance(activities_raw, dict):
                        atvs = activities_raw.get('data', activities_raw.get('items', []))
                    else:
                        atvs = activities_raw
                    atividades = []
                    for a in atvs:
                        atividades.append({
                            'id':    a.get('id'),
                            'nome':  a.get('name'),
                            'data':  a.get('start_time'),
                            'venue': a.get('venue') or {},   # campo: lat, lng, rotation, length, width
                        })
                    st.session_state.df_activities = pd.DataFrame(atividades)
                    st.success(f"✅ {len(atividades)} atividades")
                
                st.session_state.api = api
        
        if not st.session_state.df_activities.empty and token:
            st.markdown("---")
            st.header("🎯 Filtros")
            
            if not st.session_state.df_teams.empty:
                st.subheader("🏢 Filtrar por Equipe")
                equipes = ['Todas'] + sorted(st.session_state.df_teams['nome'].unique().tolist())
                equipes_selecionadas = st.multiselect("Selecione as equipes:", equipes, default=['Todas'])
                st.session_state.equipes_selecionadas = equipes_selecionadas
            
            if not st.session_state.df_positions.empty:
                st.subheader("🎯 Filtrar por Posição")
                posicoes = ['Todas'] + sorted(st.session_state.df_positions['nome'].unique().tolist())
                posicoes_selecionadas = st.multiselect("Selecione as posições:", posicoes, default=['Todas'])
                st.session_state.posicoes_selecionadas = posicoes_selecionadas
            
            st.subheader("📅 Atividade")
            atividade_sel = st.selectbox("Selecione a atividade:", st.session_state.df_activities['nome'].tolist())
            st.session_state['_atividade_sel_cached'] = atividade_sel

            if atividade_sel:
                _act_row = st.session_state.df_activities[
                    st.session_state.df_activities['nome'] == atividade_sel]
                activity_id = _act_row['id'].values[0]
                st.session_state.activity_id = activity_id

                # ── Venue (campo) da atividade ────────────────────────────────
                # Salva lat, lng, rotation, length, width para pré-popular o
                # componente de posicionamento de campo automaticamente.
                if 'venue' in _act_row.columns:
                    _venue_val = _act_row['venue'].values[0]
                    st.session_state.venue = _venue_val if isinstance(_venue_val, dict) else {}
                else:
                    st.session_state.venue = {}

                # ── FEATURE 6: Descoberta dinâmica de parâmetros ─────────────
                _api_params_disc = st.session_state.get('api')
                if _api_params_disc:
                    try:
                        _sess_params_raw = _api_params_disc.get_session_parameters(activity_id)
                        if _sess_params_raw:
                            _p_list = (_sess_params_raw if isinstance(_sess_params_raw, list)
                                       else _sess_params_raw.get('data', []))
                            _param_names = []
                            for _pp in _p_list:
                                _pn = _pp.get('name') or _pp.get('slug') or str(_pp)
                                if _pn:
                                    _param_names.append(_pn)
                            st.session_state['available_params'] = _param_names
                    except Exception:
                        pass

                # ── FEATURE 7: Venues da conta Catapult ──────────────────────
                _api_venues = st.session_state.get('api')
                if _api_venues:
                    try:
                        if 'venues_catapult' not in st.session_state:
                            _venues_raw = _api_venues.get_venues()
                            if _venues_raw:
                                _venues_list = (_venues_raw if isinstance(_venues_raw, list)
                                                else _venues_raw.get('data', []))
                                st.session_state['venues_catapult'] = _venues_list
                    except Exception:
                        pass

                # Auto-match venue desta atividade com venues da conta
                _venue_act = st.session_state.get('venue', {})
                _venue_name_act = str(_venue_act.get('name', _venue_act.get('venue_name', '')))
                _venues_cat = st.session_state.get('venues_catapult', [])
                for _vc in _venues_cat:
                    _vc_name = str(_vc.get('name', ''))
                    if _vc_name and _vc_name.lower() == _venue_name_act.lower():
                        st.info(f"✅ Campo carregado automaticamente da conta Catapult: {_vc_name}")
                        # Auto-populate lat/lng/rotation/dimensions if available
                        for _fld in ('lat', 'lng', 'lon', 'rotation', 'length', 'width'):
                            if _vc.get(_fld) is not None:
                                _venue_act[_fld] = _vc[_fld]
                        st.session_state['venue'] = _venue_act
                        break

                # ── Item 14: Tags da Atividade ────────────────────────────────
                _api_tags = st.session_state.get('api')
                if _api_tags:
                    try:
                        _act_tags_raw = _api_tags.get_activity_tags(activity_id)
                        _tag_list = []
                        if _act_tags_raw:
                            if isinstance(_act_tags_raw, list):
                                for _tg in _act_tags_raw:
                                    _tn = _tg.get('name') or _tg.get('label') or str(_tg)
                                    if _tn:
                                        _tag_list.append(_tn)
                            elif isinstance(_act_tags_raw, dict):
                                for _tg in _act_tags_raw.get('data', []):
                                    _tn = _tg.get('name') or _tg.get('label') or ''
                                    if _tn:
                                        _tag_list.append(_tn)
                        if _tag_list:
                            _tags_html = "".join(
                                f"<span style='background:#1e3a5f;color:#90CAF9;border-radius:12px;"
                                f"padding:2px 10px;margin:2px 4px 2px 0;font-size:11px;"
                                f"display:inline-block'>{_t}</span>"
                                for _t in _tag_list
                            )
                            st.markdown(
                                f"<div style='margin-bottom:6px'><strong>🏷️ Tags:</strong><br>{_tags_html}</div>",
                                unsafe_allow_html=True
                            )
                    except Exception:
                        pass

                with st.spinner("Buscando períodos da atividade..."):
                    api = st.session_state.api
                    periods_raw = api.get_activity_periods(activity_id)
                    
                    # Se a API retornou períodos individuais, usamos apenas eles.
                    # "Atividade Completa" (period_id=None) é mantida APENAS como
                    # fallback quando nenhum período é encontrado.
                    if periods_raw and isinstance(periods_raw, list):
                        period_options = []
                        period_ids = {}
                        for p in periods_raw:
                            period_options.append(p.get('name', 'Período'))
                            period_ids[p.get('name', 'Período')] = p.get('id')
                        st.session_state.period_options = period_options
                        st.session_state.period_ids = period_ids
                        st.success(f"✅ {len(period_options)} períodos encontrados")
                    else:
                        # Sem períodos — fallback para atividade completa
                        period_options = ['Atividade Completa']
                        period_ids = {'Atividade Completa': None}
                        st.session_state.period_options = period_options
                        st.session_state.period_ids = period_ids

                st.subheader("📊 Selecionar Período(s)")
                _default_periodos = st.session_state.period_options  # todos por padrão
                periodos_selecionados = st.multiselect(
                    "Selecione um ou mais períodos para análise:",
                    options=st.session_state.period_options,
                    default=_default_periodos
                )
                st.session_state.periodos_selecionados = periodos_selecionados

                if st.button("🔍 Buscar Atletas da Atividade"):
                    with st.spinner("Buscando atletas em todos os períodos..."):
                        api = st.session_state.api
                        # ── Busca atletas em TODOS os períodos selecionados e faz união ──
                        _periodos_para_busca = periodos_selecionados if periodos_selecionados else (
                            st.session_state.period_options if st.session_state.period_options else []
                        )
                        athletes_by_id = {}  # {athlete_id: athlete_dict} — chave para deduplicar

                        def _extrair_lista_atletas(resp):
                            """Normaliza a resposta da API para lista de dicts de atleta."""
                            if not resp:
                                return []
                            if isinstance(resp, list):
                                return resp
                            if isinstance(resp, dict):
                                for _k in ['data', 'items', 'athletes']:
                                    if _k in resp and isinstance(resp[_k], list):
                                        return resp[_k]
                                if 'id' in resp:
                                    return [resp]
                            return []

                        if _periodos_para_busca:
                            for _per_nome in _periodos_para_busca:
                                _pid = st.session_state.period_ids.get(_per_nome)
                                if _pid:
                                    _resp = api.get_athletes_in_period(_pid)
                                else:
                                    _resp = api.get_activity_athletes(activity_id)
                                for _a in _extrair_lista_atletas(_resp):
                                    _aid = _a.get('id')
                                    if _aid and _aid not in athletes_by_id:
                                        athletes_by_id[_aid] = _a
                        else:
                            # Nenhum período selecionado — busca atletas da atividade inteira
                            _resp = api.get_activity_athletes(activity_id)
                            for _a in _extrair_lista_atletas(_resp):
                                _aid = _a.get('id')
                                if _aid and _aid not in athletes_by_id:
                                    athletes_by_id[_aid] = _a

                        athletes_in = list(athletes_by_id.values())

                        if athletes_in:
                            atletas_temp = []
                            for a in athletes_in:
                                nome = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
                                if not nome:
                                    nome = a.get('name', 'Sem nome')
                                position_name = a.get('position_name', a.get('position', ''))
                                team_name = st.session_state.athlete_team_map.get(a.get('id'), '')
                                atletas_temp.append({
                                    'id': a.get('id'), 'nome': nome, 'camisa': a.get('jersey', ''),
                                    'posicao': position_name, 'equipe': team_name
                                })

                            df_atletas_temp = pd.DataFrame(atletas_temp)

                            if 'equipes_selecionadas' in st.session_state:
                                if 'Todas' not in st.session_state.equipes_selecionadas:
                                    df_atletas_temp = df_atletas_temp[df_atletas_temp['equipe'].isin(st.session_state.equipes_selecionadas)]

                            if 'posicoes_selecionadas' in st.session_state:
                                if 'Todas' not in st.session_state.posicoes_selecionadas:
                                    df_atletas_temp = df_atletas_temp[df_atletas_temp['posicao'].isin(st.session_state.posicoes_selecionadas)]

                            st.session_state.atletas_filtrados = df_atletas_temp
                            # Auto-seleciona TODOS os atletas encontrados — o usuário
                            # já escolheu os períodos, não precisa selecionar novamente.
                            _todos_nomes = df_atletas_temp['nome'].tolist()
                            st.session_state['atletas_sel'] = _todos_nomes
                            st.session_state['atletas_selecionados'] = _todos_nomes
                            _n_per_buscados = len(_periodos_para_busca) if _periodos_para_busca else 1
                            st.success(f"✅ {len(df_atletas_temp)} atletas encontrados e selecionados em {_n_per_buscados} período(s)")
                
                if 'atletas_filtrados' in st.session_state and not st.session_state.atletas_filtrados.empty:
                    st.subheader("🏃 Selecionar Atletas")

                    # Search box
                    _busca_atl = st.text_input("🔍 Buscar atleta:", placeholder="Nome ou camisa...", key="busca_atleta")

                    # Position preset buttons
                    if not st.session_state.atletas_filtrados.empty and 'posicao' in st.session_state.atletas_filtrados.columns:
                        _posicoes_disponiveis = sorted(
                            st.session_state.atletas_filtrados['posicao'].dropna().unique().tolist()
                        )
                        if _posicoes_disponiveis:
                            _pos_cols = st.columns(min(len(_posicoes_disponiveis) + 1, 4))
                            with _pos_cols[0]:
                                if st.button("👥 Todos", key="preset_todos", use_container_width=True):
                                    st.session_state['_preset_posicao'] = None
                            for _pi, _pn in enumerate(_posicoes_disponiveis[:3]):
                                with _pos_cols[_pi + 1]:
                                    if st.button(_pn[:8], key=f"preset_{_pn}", use_container_width=True):
                                        st.session_state['_preset_posicao'] = _pn

                    # Apply filters
                    _preset_pos = st.session_state.get('_preset_posicao')
                    _df_atl_disp = st.session_state.atletas_filtrados.copy()
                    if _busca_atl:
                        _df_atl_disp = _df_atl_disp[
                            _df_atl_disp['nome'].str.contains(_busca_atl, case=False, na=False) |
                            _df_atl_disp['camisa'].astype(str).str.contains(_busca_atl, case=False, na=False)
                        ]
                    if _preset_pos:
                        _df_atl_disp = _df_atl_disp[_df_atl_disp['posicao'] == _preset_pos]

                    atletas_disponiveis_sidebar = _df_atl_disp['nome'].tolist()

                    # Default: usa atletas já pré-selecionados (pelo botão Buscar)
                    # filtrando apenas os que ainda aparecem na lista atual.
                    _sel_prev = st.session_state.get('atletas_sel', [])
                    _sel_valido = [a for a in _sel_prev if a in atletas_disponiveis_sidebar]
                    if not _sel_valido:
                        _sel_valido = atletas_disponiveis_sidebar  # tudo selecionado por padrão
                    atletas_sel = st.multiselect(
                        "Atletas selecionados:",
                        options=atletas_disponiveis_sidebar,
                        default=_sel_valido,
                        key="atletas_selecionados"
                    )
                    st.session_state.atletas_sel = atletas_sel

        # ── Parâmetros de análise de esforço ─────────────────────────
        if not st.session_state.get('df_activities', pd.DataFrame()).empty and token:
            st.markdown("---")
            st.header("⚙️ Parâmetros de Esforço")
            st.slider(
                "⏱️ Duração mínima de acc/dec (s):",
                min_value=0.1, max_value=1.5,
                value=float(st.session_state.get('min_dur_esforco', _DEFAULT_MIN_DUR_S)),
                step=0.1,
                key="min_dur_esforco",
                help=(
                    "Tempo mínimo contínuo na zona de threshold para contar um evento.\n\n"
                    "🔵 Catapult OpenField: 0.6 s\n"
                    "🟡 Sistema alternativo: 0.4 s\n"
                    "Ajuste conforme o sistema de referência da sua análise."
                )
            )
            _dur_atual = st.session_state.get('min_dur_esforco', _DEFAULT_MIN_DUR_S)
            st.caption(
                f"Mínimo: **{_dur_atual:.1f} s** = "
                f"**{max(1, round(_dur_atual * _SENSOR_HZ))} frames** a 10 Hz"
            )

            st.slider(
                "⏱️ Duração mínima de velocidade (s):",
                min_value=0.1, max_value=3.0,
                value=float(st.session_state.get('min_dur_vel', _DEFAULT_MIN_DUR_VEL_S)),
                step=0.1,
                key="min_dur_vel",
                help=(
                    "Tempo mínimo contínuo dentro de uma banda de velocidade para "
                    "que o segmento seja contabilizado como um esforço.\n\n"
                    "💡 Valores maiores filtram movimentos breves e capturam "
                    "apenas esforços sustentados. Padrão: 1.0 s."
                )
            )
            _dur_vel_atual = st.session_state.get('min_dur_vel', _DEFAULT_MIN_DUR_VEL_S)
            st.caption(
                f"Mínimo: **{_dur_vel_atual:.1f} s** = "
                f"**{max(1, round(_dur_vel_atual * _SENSOR_HZ))} frames** a 10 Hz"
            )

        # ── Seletor de Eventos Futebol ────────────────────────────────
        if not st.session_state.get('df_activities', pd.DataFrame()).empty and token:
            st.markdown("---")
            st.header("⚽ Eventos Futebol")
            _todos_ev = list(FUTEBOL_EVENTS_CONFIG.keys())
            _sel_all  = st.checkbox("Selecionar todos", value=True, key="eventos_sel_all")
            if _sel_all:
                eventos_futebol_sel = _todos_ev
            else:
                eventos_futebol_sel = st.multiselect(
                    "Tipos de evento:",
                    options=_todos_ev,
                    default=_todos_ev[:4],
                    format_func=lambda k: FUTEBOL_EVENTS_CONFIG[k]['label'],
                    key="eventos_futebol_ms"
                )
            st.session_state.eventos_futebol_sel = eventos_futebol_sel
            if eventos_futebol_sel:
                st.caption(f"{len(eventos_futebol_sel)} tipo(s) selecionado(s). "
                           "Os eventos serão carregados junto com os dados.")

        # ── Máximo Histórico (CORRECTION 1) ──────────────────────────────
        if not st.session_state.get('df_activities', pd.DataFrame()).empty and token:
            st.markdown("---")
            with st.expander("📊 Máximo Histórico", expanded=False):
                st.caption(
                    "Vmax histórico buscado automaticamente do cadastro Catapult "
                    "(limiares → perfil → zonas). Usado em '% do Máximo' nas abas "
                    "de esforços. Você pode sobrescrever manualmente por atleta."
                )
                _hist_api = st.session_state.get('api')

                # ── Botão para re-buscar o perfil (limpa cache de perfil/thresholds) ──
                if _hist_api and st.button("🔄 Re-buscar via /stats Catapult", key="btn_refetch_hist_vmax"):
                    # Limpa apenas as fontes automáticas (preserva 'manual')
                    _src_g = st.session_state.get('hist_vmax_source', {})
                    _hvm_g = st.session_state.get('hist_vmax', {})
                    for _an_r in list(_hvm_g.keys()):
                        if _src_g.get(_an_r) != 'manual':
                            _hvm_g.pop(_an_r, None)
                            _src_g.pop(_an_r, None)
                    for _k in list(st.session_state.keys()):
                        if _k.startswith('profile_') or _k.startswith('thresholds_'):
                            del st.session_state[_k]
                    # Re-busca imediatamente via /stats
                    try:
                        _rb_stats = _hist_api.get_stats({
                            "group_by": ["athlete"],
                            "parameters": ["max_velocity"],
                            "source": "cached_stats",
                        })
                        if _rb_stats:
                            _rb_list = _rb_stats if isinstance(_rb_stats, list) else _rb_stats.get('data', [])
                            _n_rb = 0
                            for _rbs in (_rb_list or []):
                                _rbs_name = str(_rbs.get('athlete') or _rbs.get('athlete_name') or _rbs.get('name') or '')
                                _rbs_params = _rbs.get('parameters') or _rbs
                                _rbs_val = float(_rbs_params.get('max_velocity') or _rbs_params.get('max_speed') or
                                                 _rbs.get('max_velocity') or _rbs.get('max_speed') or 0)
                                if _rbs_name and _rbs_val > 0 and _src_g.get(_rbs_name) != 'manual':
                                    _rbs_ms = _rbs_val / 3.6 if _rbs_val > 15 else _rbs_val
                                    _hvm_g[_rbs_name] = _rbs_ms
                                    _src_g[_rbs_name] = 'catapult_stats'
                                    _n_rb += 1
                            st.toast(f"✅ {_n_rb} Vmax atualizadas via /stats")
                    except Exception:
                        pass
                    st.session_state['hist_vmax'] = _hvm_g
                    st.session_state['hist_vmax_source'] = _src_g
                    st.rerun()

                # ── Botão fallback: /stats ─────────────────────────────────────
                with st.expander("🔍 Buscar via /stats (fallback)", expanded=False):
                    st.caption("Use se os campos abaixo aparecerem como 'não detectado'.")
                    if _hist_api and st.button("Executar busca /stats", key="btn_fetch_hist_vmax"):
                        try:
                            _stats_resp = _hist_api.get_stats({
                                "group_by": ["athlete"],
                                "parameters": ["max_velocity"],
                                "source": "cached_stats",
                            })
                            if _stats_resp:
                                _sv_list = _stats_resp if isinstance(_stats_resp, list) else _stats_resp.get('data', [])
                                _hvm = st.session_state.get('hist_vmax', {})
                                _src_g2 = st.session_state.get('hist_vmax_source', {})
                                _n_upd = 0
                                for _sv in (_sv_list or []):
                                    _sv_name = str(_sv.get('athlete') or _sv.get('athlete_name', ''))
                                    _sv_val  = (_sv.get('parameters') or _sv).get('max_velocity', 0) or 0
                                    if _sv_name and float(_sv_val) > 0 and _src_g2.get(_sv_name) != 'manual':
                                        _hvm[_sv_name] = float(_sv_val)
                                        _src_g2[_sv_name] = 'stats'
                                        _n_upd += 1
                                st.session_state['hist_vmax'] = _hvm
                                st.session_state['hist_vmax_source'] = _src_g2
                                st.success(f"✅ Atualizado para {_n_upd} atleta(s) via /stats.")
                        except Exception as _hve:
                            st.error(f"Erro: {_hve}")

                # ── Tabela de valores detectados ───────────────────────────────
                _atletas_sidebar = (
                    st.session_state.atletas_sel
                    if st.session_state.get('atletas_sel') else []
                )
                _hvm_dict  = st.session_state.get('hist_vmax', {})
                _src_dict  = st.session_state.get('hist_vmax_source', {})

                _src_labels = {
                    'catapult_stats':    '📡 Catapult /stats',
                    'catapult_athletes': '📡 Cadastro',
                    'thresholds':        '📋 Limiares',
                    'profile':           '👤 Perfil',
                    'zones':             '🏷️ Zonas',
                    'stats':             '📊 /stats',
                    'manual':            '✏️ Manual',
                    '':                  '❓ Não detectado',
                }

                for _an in _atletas_sidebar:
                    _cur_ms  = _hvm_dict.get(_an, 0.0)
                    _cur_kmh = round(_cur_ms * 3.6, 2) if _cur_ms else 0.0
                    _src_tag = _src_labels.get(_src_dict.get(_an, ''), '❓ Não detectado')
                    _col_v, _col_s = st.columns([3, 2])
                    with _col_v:
                        if _cur_kmh > 0:
                            st.markdown(f"**{_an}**  \n`{_cur_kmh} km/h`")
                        else:
                            st.markdown(f"**{_an}**  \n`— não detectado`")
                    with _col_s:
                        st.caption(_src_tag)

                    # Override manual com toggle
                    if st.toggle("✏️ Editar", key=f"_tgl_hvm_{_an}", value=False):
                        _inp = st.number_input(
                            "Vmax (km/h):",
                            min_value=0.0, max_value=50.0,
                            value=_cur_kmh,
                            step=0.1,
                            key=f"hist_vmax_{_an}",
                        )
                        if _inp > 0:
                            _hvm_dict[_an] = _inp / 3.6
                            _src_dict[_an] = 'manual'
                        elif _inp == 0 and _src_dict.get(_an) == 'manual':
                            # Apagar override manual
                            _hvm_dict.pop(_an, None)
                            _src_dict.pop(_an, None)

                st.session_state['hist_vmax']        = _hvm_dict
                st.session_state['hist_vmax_source'] = _src_dict

        # ── Bandas de Velocidade (CORRECTION 2) ──────────────────────────
        if not st.session_state.get('df_activities', pd.DataFrame()).empty and token:
            with st.expander("🏷️ Bandas de Velocidade", expanded=False):
                _vz_api = st.session_state.get('api')
                if _vz_api and st.button("🔄 Carregar zonas da conta", key="btn_load_vel_zones"):
                    try:
                        _vz_raw = _vz_api.get_velocity_zones()
                        _parsed = _parse_api_velocity_zones(_vz_raw)
                        st.session_state['velocity_zones_account'] = _parsed
                        st.success(f"{len(_parsed)} zonas carregadas da conta.")
                    except Exception as _vze:
                        st.error(f"Erro: {_vze}")

                _cur_zones = (
                    st.session_state.get('velocity_zones_account')
                    or _DEFAULT_VELOCITY_ZONES
                )
                _df_zones = pd.DataFrame([
                    {
                        'Zona': z['name'],
                        'Mín (km/h)': round(z['min_ms'] * 3.6, 1),
                        'Máx (km/h)': '∞' if z['max_ms'] >= 9000 else f"{round(z['max_ms'] * 3.6, 1):.1f}",
                    }
                    for z in _cur_zones
                ])
                st.dataframe(_df_zones, use_container_width=True, hide_index=True)

        # ── Parâmetros disponíveis (FEATURE 6) ───────────────────────────
        _avail_params = st.session_state.get('available_params')
        if _avail_params:
            with st.expander("🔧 Parâmetros disponíveis", expanded=False):
                _desired = {"ts", "lat", "long", "v", "rv", "a", "hr", "pl",
                            "xy", "pq", "hdop", "ref", "o", "mp"}
                _ap_set = set(_avail_params) if isinstance(_avail_params, list) else set()
                _present = _desired & _ap_set
                _missing = _desired - _ap_set
                st.caption(f"Dispositivo: {len(_present)} parâmetros disponíveis.")
                if _missing:
                    st.warning(
                        f"Parâmetros não disponíveis neste dispositivo: "
                        f"`{'`, `'.join(sorted(_missing))}`"
                    )
                    if 'rv' in _missing:
                        st.info("rv (velocidade bruta) não disponível — perfil F-V usará v.")
                    if 'mp' in _missing:
                        st.info("mp (potência metabólica) não disponível.")
                st.write("Disponíveis:", sorted(_present))

    # Área principal
    if ('api' in st.session_state and 'atletas_sel' in st.session_state and
        st.session_state.atletas_sel and 'activity_id' in st.session_state):

        api = st.session_state.api
        activity_id = st.session_state.activity_id
        periodos_selecionados = st.session_state.get('periodos_selecionados', ['Atividade Completa'])
        period_ids = st.session_state.get('period_ids', {})
        _atividade_nome = locals().get('atividade_sel', st.session_state.get('_atividade_sel_cached', ''))
        if _atividade_nome:
            st.session_state['_atividade_sel_cached'] = _atividade_nome
        else:
            _atividade_nome = st.session_state.get('_atividade_sel_cached', '')

        # Dicionários para armazenar dados por período
        resultados_por_periodo = {}
        dados_sensor_por_atleta_por_periodo = {}
        dados_efforts_vel_por_periodo = {}
        dados_efforts_acc_por_periodo = {}
        dados_hr_efforts_por_periodo = {}
        dados_jump_efforts_por_periodo = {}
        dados_step_efforts_por_periodo = {}
        dados_posicao_por_periodo = {}
        dados_eventos_por_periodo = {}   # ← eventos futebol

        # Tipos de eventos futebol selecionados na sidebar
        eventos_futebol_sel = st.session_state.get('eventos_futebol_sel', list(FUTEBOL_EVENTS_CONFIG.keys()))
        eventos_futebol_str = ','.join(eventos_futebol_sel) if eventos_futebol_sel else ''

        # ── Container único de carregamento (substituído a cada atleta, apagado no fim) ──
        _n_per_ld   = len(periodos_selecionados)
        _n_atl_ld   = len(st.session_state.atletas_sel)
        _total_ld   = max(1, _n_per_ld * _n_atl_ld)
        _done_ld    = 0
        _ok_ld      = 0
        _warn_ld    = []
        _ld_box     = st.empty()

        for periodo_nome in periodos_selecionados:
            period_id = period_ids.get(periodo_nome)

            resultados = []
            dados_sensor_por_atleta = {}
            dados_efforts_vel = {}
            dados_efforts_acc = {}
            dados_hr_efforts = {}
            dados_jump_efforts = {}
            dados_step_efforts = {}
            dados_posicao = {}
            dados_eventos = {}   # ← eventos futebol deste período

            _idx_per_ld = periodos_selecionados.index(periodo_nome) + 1

            for i, atleta_nome in enumerate(st.session_state.atletas_sel):
                _done_ld += 1
                _pct_ld   = _done_ld / _total_ld
                with _ld_box.container():
                    st.markdown(
                        f"<div style='padding:14px 20px;background:#111827;border-radius:10px;"
                        f"border:1px solid #1f2937'>"
                        f"<div style='color:#6b7280;font-size:12px;margin-bottom:6px'>"
                        f"⏳ &nbsp;Período &nbsp;<b style='color:#93c5fd'>{periodo_nome}</b>"
                        f"&nbsp; <span style='opacity:.6'>({_idx_per_ld}/{_n_per_ld})</span>"
                        f"</div>"
                        f"<div style='color:#f9fafb;font-size:15px;font-weight:600'>{atleta_nome}</div>"
                        f"<div style='color:#6b7280;font-size:12px;margin-top:4px'>"
                        f"{_done_ld} / {_total_ld} &nbsp;atletas</div></div>",
                        unsafe_allow_html=True
                    )
                    st.progress(_pct_ld)
                
                athlete_row = st.session_state.atletas_filtrados[st.session_state.atletas_filtrados['nome'] == atleta_nome]
                if athlete_row.empty:
                    continue
                    
                athlete_id = athlete_row['id'].values[0]
                athlete_posicao = athlete_row['posicao'].values[0] if 'posicao' in athlete_row.columns else ''
                athlete_equipe = athlete_row['equipe'].values[0] if 'equipe' in athlete_row.columns else ''

                # ── Item 15: limiares individuais do atleta ───────────────────
                # Tenta buscar limiares específicos deste atleta via API.
                # Se disponíveis, usa-os para a análise (armazenado em session_state).
                _thr_key = f"thresholds_{athlete_id}"
                if _thr_key not in st.session_state:
                    try:
                        _thr_raw = api.get_athlete_thresholds(athlete_id)
                        _thr_data = {}
                        if _thr_raw:
                            _thr_items = _thr_raw if isinstance(_thr_raw, list) else _thr_raw.get('data', [])
                            for _thr in (_thr_items if isinstance(_thr_items, list) else []):
                                _tname = _thr.get('name') or _thr.get('type', '')
                                _tval  = _thr.get('value') or _thr.get('threshold')
                                if _tname and _tval is not None:
                                    _thr_data[_tname] = float(_tval)
                        st.session_state[_thr_key] = _thr_data
                    except Exception:
                        st.session_state[_thr_key] = {}

                # ── Auto-popular Vmax histórico a partir do cadastro ──────────
                # Prioridade: (1) limiares do atleta, (2) perfil do atleta,
                # (3) zona de velocidade mais alta. Só sobrescreve se ainda
                # não tiver um valor (ou se o valor for 0 / manual não definido).
                _hvm_global = st.session_state.get('hist_vmax', {})
                _src_global = st.session_state.get('hist_vmax_source', {})
                _src_atual  = _src_global.get(atleta_nome, '')
                # Não sobrescreve sobreposições manuais do usuário
                if _src_atual != 'manual':
                    _vmax_encontrado = 0.0
                    _vmax_fonte      = ''
                    # — Fonte 1: limiares cadastrados (thresholds endpoint) ——
                    _thr_data_now = st.session_state.get(_thr_key, {})
                    _vmax_keys_thr = [
                        'max_velocity', 'velocity_max', 'peak_speed',
                        'max_speed', 'v_max', 'maximum_velocity',
                        'MaxVelocity', 'MaxSpeed', 'velocidade_maxima',
                        'vmax',
                    ]
                    for _vk in _vmax_keys_thr:
                        _vv = _thr_data_now.get(_vk, 0)
                        if _vv and float(_vv) > _vmax_encontrado:
                            _vmax_encontrado = float(_vv)
                            _vmax_fonte = 'thresholds'
                    # — Fonte 2: perfil do atleta (GET /athletes/{id}) ————
                    if _vmax_encontrado == 0:
                        try:
                            _prof_key = f"profile_{athlete_id}"
                            if _prof_key not in st.session_state:
                                _prof_raw = api.get_athlete(athlete_id)
                                st.session_state[_prof_key] = _prof_raw or {}
                            _prof_outer = st.session_state.get(_prof_key, {})
                            # Suporta resposta direta OU aninhada em {"data": {...}}
                            _prof = (
                                _prof_outer.get('data', _prof_outer)
                                if isinstance(_prof_outer, dict)
                                else (_prof_outer[0] if isinstance(_prof_outer, list) and _prof_outer else {})
                            )
                            _vmax_prof_keys = [
                                'max_speed', 'max_velocity', 'maximum_velocity',
                                'maximum_speed', 'peak_speed', 'v_max',
                            ]
                            for _pk in _vmax_prof_keys:
                                _pv = _prof.get(_pk, 0)
                                if _pv:
                                    _pvf = float(_pv)
                                    # Converte km/h → m/s se necessário
                                    if _pvf > 15:
                                        _pvf /= 3.6
                                    if _pvf > _vmax_encontrado:
                                        _vmax_encontrado = _pvf
                                        _vmax_fonte = 'profile'
                        except Exception:
                            pass
                    # — Fonte 3: topo da zona de velocidade mais alta ——————
                    # Exclui valores >= 9000 m/s (sentinel "infinito" da última zona)
                    if _vmax_encontrado == 0:
                        try:
                            _zones_atl = get_zones_for_athlete(atleta_nome)
                            if _zones_atl:
                                _top_ms = max(
                                    (float(z.get('max_ms', 0)) for z in _zones_atl
                                     if float(z.get('max_ms', 0) or 0) < 9000),
                                    default=0
                                )
                                if _top_ms > 0:
                                    _vmax_encontrado = _top_ms
                                    _vmax_fonte = 'zones'
                        except Exception:
                            pass
                    # — Persiste se encontrou algo ────────────────────────
                    if _vmax_encontrado > 0:
                        _hvm_global[atleta_nome] = _vmax_encontrado
                        _src_global[atleta_nome] = _vmax_fonte
                st.session_state['hist_vmax']        = _hvm_global
                st.session_state['hist_vmax_source'] = _src_global

                if period_id:
                    response         = api.get_period_sensor_data(period_id, athlete_id)
                    efforts_response = api.get_period_efforts(period_id, athlete_id,
                                                             "velocity,acceleration,heart_rate,jump,step_balance")
                    events_response  = api.get_period_events(period_id, athlete_id, eventos_futebol_str) if eventos_futebol_str else None
                else:
                    response         = api.get_sensor_data(activity_id, athlete_id)
                    efforts_response = api.get_activity_efforts(activity_id, athlete_id,
                                                               "velocity,acceleration,heart_rate,jump,step_balance")
                    events_response  = api.get_activity_events(activity_id, athlete_id, eventos_futebol_str) if eventos_futebol_str else None
                
                sensor_points = extrair_dados_sensor(response)
                
                if sensor_points:
                    dados_sensor_por_atleta[atleta_nome] = sensor_points

                    _atleta_zones = get_zones_for_athlete(atleta_nome)
                    metricas = calcular_metricas(sensor_points, atleta_nome,
                                                 zones=_atleta_zones)
                    if metricas:
                        metricas['Posição'] = athlete_posicao
                        metricas['Equipe'] = athlete_equipe
                        resultados.append(metricas)

                    if efforts_response:
                        _vel_eff, _acc_eff, _hr_eff, _jmp_eff, _step_eff = extrair_efforts_data(efforts_response)
                        if _vel_eff:
                            dados_efforts_vel[atleta_nome] = _vel_eff
                        if _acc_eff:
                            dados_efforts_acc[atleta_nome] = _acc_eff
                        if _hr_eff:
                            dados_hr_efforts[atleta_nome] = _hr_eff
                        if _jmp_eff:
                            dados_jump_efforts[atleta_nome] = _jmp_eff
                        if _step_eff:
                            dados_step_efforts[atleta_nome] = _step_eff

                    # ── OpenField pre-computed summary ────────────────────────
                    try:
                        if period_id:
                            _of_sum = api.get_athlete_period_summary(period_id, athlete_id)
                        else:
                            _of_sum = api.get_athlete_activity_summary(activity_id, athlete_id)
                        if _of_sum:
                            if atleta_nome not in dados_posicao:
                                dados_posicao[atleta_nome] = {
                                    'vel': [], 'xs': [], 'ys': [], 'acc': [], 'ts_pos': [],
                                    'posicao': athlete_posicao, 'equipe': athlete_equipe,
                                    'n_pontos': 0,
                                }
                            dados_posicao[atleta_nome]['openfield_summary'] = _of_sum
                    except Exception:
                        pass
                    
                    # A API devolve x,y com origem no canto inferior esquerdo (0,0).
                    # Filtra nulos e artefactos de projeção GPS (valores absurdamente altos).
                    _venue   = st.session_state.get('venue', {})
                    _fl_v    = float(_venue.get('length') or 105)
                    _fw_v    = float(_venue.get('width')  or 68)
                    pontos_pos = [
                        (float(p['x']), float(p['y']),
                         (p.get('v') or 0) * 3.6,
                         float(p.get('a') or 0),
                         float(p.get('ts') or 0))
                        for p in sensor_points
                        if p.get('x') is not None and p.get('y') is not None
                        and float(p['x']) > -15 and float(p['x']) < _fl_v + 15
                        and float(p['y']) > -15 and float(p['y']) < _fw_v + 15
                    ]
                    if pontos_pos:
                        xs          = [pt[0] for pt in pontos_pos]
                        ys          = [pt[1] for pt in pontos_pos]
                        velocidades = [pt[2] for pt in pontos_pos]
                        aceleracoes = [pt[3] for pt in pontos_pos]
                        ts_pos      = [pt[4] for pt in pontos_pos]
                        dados_posicao[atleta_nome] = {
                            'vel': velocidades, 'xs': xs, 'ys': ys,
                            'acc': aceleracoes, 'ts_pos': ts_pos,
                            'posicao': athlete_posicao, 'equipe': athlete_equipe,
                            'n_pontos': len(pontos_pos)
                        }

                    # Coleta lat/lon reais (GPS) para o mapa satélite.
                    # Filtra zeros (sem lock de GPS) e valores geograficamente inválidos.
                    # Armazena ts (Unix timestamp) para filtrar pontos por esforço.
                    pontos_gps = [
                        (float(p['lat']), float(p['long']),
                         (p.get('v') or 0) * 3.6,
                         float(p.get('ts') or 0))
                        for p in sensor_points
                        if p.get('lat') is not None and p.get('long') is not None
                        and abs(float(p['lat'])) > 1e-6 and abs(float(p['long'])) > 1e-6
                        and -90 < float(p['lat']) < 90
                        and -180 < float(p['long']) < 180
                    ]
                    if pontos_gps:
                        step_gps = max(1, len(pontos_gps) // 30000)
                        gps_sub = pontos_gps[::step_gps]
                        if atleta_nome not in dados_posicao:
                            dados_posicao[atleta_nome] = {
                                'vel': [], 'xs': [], 'ys': [], 'acc': [], 'ts_pos': [],
                                'posicao': athlete_posicao, 'equipe': athlete_equipe,
                                'n_pontos': 0
                            }
                        dados_posicao[atleta_nome]['lats'] = [pt[0] for pt in gps_sub]
                        dados_posicao[atleta_nome]['lons'] = [pt[1] for pt in gps_sub]
                        dados_posicao[atleta_nome]['vels_gps'] = [pt[2] for pt in gps_sub]
                        dados_posicao[atleta_nome]['ts_gps']  = [pt[3] for pt in gps_sub]

                    # ── GPS Quality (pq, hdop, ref) e Odômetro (o) ────────────
                    # Item 8: qualidade do sinal GPS; Item 12: distância pelo odômetro nativo
                    _pq_vals   = [float(p['pq'])   for p in sensor_points if p.get('pq')   is not None and float(p.get('pq') or 0) > 0]
                    _hdop_vals = [float(p['hdop'])  for p in sensor_points if p.get('hdop') is not None]
                    _ref_vals  = [float(p['ref'])   for p in sensor_points if p.get('ref')  is not None and float(p.get('ref') or 0) > 0]
                    _o_vals    = [float(p['o'])     for p in sensor_points if p.get('o')    is not None]
                    if atleta_nome in dados_posicao:
                        dados_posicao[atleta_nome]['pq_mean']   = round(float(np.mean(_pq_vals)),   1) if _pq_vals   else None
                        dados_posicao[atleta_nome]['hdop_mean'] = round(float(np.mean(_hdop_vals)), 2) if _hdop_vals else None
                        dados_posicao[atleta_nome]['ref_mean']  = round(float(np.mean(_ref_vals)),  1) if _ref_vals  else None
                        # Odometer: distância acumulada nativa do dispositivo (mais preciso que integrar v)
                        if len(_o_vals) >= 2:
                            _o_start = min(_o_vals[0], _o_vals[-1])
                            _o_end   = max(_o_vals[0], _o_vals[-1])
                            dados_posicao[atleta_nome]['odometro_m'] = round(_o_end - _o_start, 1)
                        else:
                            dados_posicao[atleta_nome]['odometro_m'] = None
                        # Série temporal do odômetro para gráfico de evolução
                        if _o_vals:
                            _o_base = _o_vals[0]
                            dados_posicao[atleta_nome]['o_series'] = [v - _o_base for v in _o_vals]
                        else:
                            dados_posicao[atleta_nome]['o_series'] = []

                    # ── Processar eventos futebol ─────────────────────────────
                    if events_response:
                        ev_raw = extrair_eventos_futebol(events_response)
                        if ev_raw:
                            ts_g   = dados_posicao.get(atleta_nome, {}).get('ts_gps', [])
                            lats_g = dados_posicao.get(atleta_nome, {}).get('lats', [])
                            lons_g = dados_posicao.get(atleta_nome, {}).get('lons', [])
                            vels_g = dados_posicao.get(atleta_nome, {}).get('vels_gps', [])
                            dados_eventos[atleta_nome] = enriquecer_eventos_com_posicao(
                                ev_raw, ts_g, lats_g, lons_g, vels_g
                                # campo_config será enriquecido depois, no momento da visualização
                            )
                            n_ev = sum(len(v) for v in ev_raw.values())
                            _ok_ld += 1
                        else:
                            _ok_ld += 1
                    else:
                        _ok_ld += 1
                
            resultados_por_periodo[periodo_nome] = resultados
            dados_sensor_por_atleta_por_periodo[periodo_nome] = dados_sensor_por_atleta
            dados_efforts_vel_por_periodo[periodo_nome] = dados_efforts_vel
            dados_efforts_acc_por_periodo[periodo_nome] = dados_efforts_acc
            dados_hr_efforts_por_periodo[periodo_nome] = dados_hr_efforts
            dados_jump_efforts_por_periodo[periodo_nome] = dados_jump_efforts
            dados_step_efforts_por_periodo[periodo_nome] = dados_step_efforts
            dados_posicao_por_periodo[periodo_nome] = dados_posicao
            dados_eventos_por_periodo[periodo_nome] = dados_eventos

        # Apagar container de loading e mostrar resumo compacto
        _ld_box.empty()
        _warn_ld_n = _n_atl_ld - _ok_ld
        _per_label = ', '.join(periodos_selecionados)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;padding:10px 16px;"
            f"background:#052e16;border-radius:8px;border:1px solid #166534;margin-bottom:8px'>"
            f"<span style='font-size:18px'>✅</span>"
            f"<div><span style='color:#86efac;font-weight:600'>{_ok_ld} atletas carregados</span>"
            f"<span style='color:#4ade80;font-size:12px'>&nbsp;·&nbsp;{_per_label}</span>"
            + (f"<span style='color:#fca5a5;font-size:12px'>&nbsp;·&nbsp;{_warn_ld_n} sem dados</span>" if _warn_ld_n else "")
            + f"</div></div>",
            unsafe_allow_html=True
        )

        # ── Alerta de falha de carregamento ────────────────────────────────
        if _ok_ld == 0 and _n_atl_ld > 0:
            _api_err = st.session_state.pop('_api_last_err', None)
            if _api_err == 401:
                st.error(
                    "⚠️ **Token expirado ou inválido (HTTP 401).** "
                    "Gere um novo token em **Settings → API** no OpenField e cole na sidebar."
                )
            elif _api_err == 403:
                st.error(
                    "⚠️ **Acesso negado (HTTP 403).** "
                    "Seu token não tem permissão para acessar esses dados."
                )
            elif _api_err:
                st.error(
                    f"⚠️ **Erro na API Catapult** ({_api_err}). "
                    "Verifique o token e a conexão com a internet."
                )
            else:
                st.warning(
                    "⚠️ **Nenhum dado retornado.** Possíveis causas: "
                    "token expirado, atividade sem dados de sensor, ou atletas sem GPS ativo."
                )

        # ── Gerar "Períodos Combinados" quando há mais de 1 período ──────────
        _CHAVE_COMBINADO = '📊 Períodos Combinados'
        _periodos_reais = [k for k in resultados_por_periodo if k != _CHAVE_COMBINADO]
        if len(_periodos_reais) > 1:
            _res_combinado = combinar_periodos(
                {k: resultados_por_periodo[k] for k in _periodos_reais}
            )
            if _res_combinado:
                resultados_por_periodo[_CHAVE_COMBINADO] = _res_combinado

        # ── Paleta global persistente por atleta (usa _ATHLETE_PALETTE global) ─
        if 'athlete_colors' not in st.session_state:
            st.session_state['athlete_colors'] = {}
        _all_loaded_athletes = sorted({
            r.get('Atleta', '')
            for _p_res in resultados_por_periodo.values()
            for r in _p_res
            if r.get('Atleta')
        })
        for _i, _aname in enumerate(_all_loaded_athletes):
            if _aname not in st.session_state['athlete_colors']:
                st.session_state['athlete_colors'][_aname] = _ATHLETE_PALETTE[_i % len(_ATHLETE_PALETTE)]

        # ══════════════════════════════════════════════════════════════════
        # COACH AI — painel de insights pró-ativos na sidebar
        # ══════════════════════════════════════════════════════════════════
        with st.sidebar:
            st.markdown("---")
            with st.expander("🤖 Coach AI — Insights Pró-ativos", expanded=True):
                _cai = []

                # ── 1. Evolução da intensidade entre períodos ─────────────
                _per_vel_avg = {}
                for _pn, _pd in dados_posicao_por_periodo.items():
                    if _pn == _CHAVE_COMBINADO:
                        continue
                    _pvs = []
                    for _av in _pd.values():
                        _pvs.extend(_av.get('vels_gps', _av.get('vel', [])))
                    if _pvs:
                        _per_vel_avg[_pn] = float(np.mean(_pvs))

                _pva_items = list(_per_vel_avg.items())
                if len(_pva_items) >= 2:
                    _drop = _pva_items[0][1] - _pva_items[-1][1]
                    if _drop > 0.5:
                        _cai.append(
                            f"📉 **Queda de {_drop:.1f} km/h** na velocidade média do "
                            f"_{_pva_items[0][0]}_ → _{_pva_items[-1][0]}_. "
                            f"Possível fadiga acumulada."
                        )
                    elif _drop < -0.5:
                        _cai.append(
                            f"📈 **Alta de {abs(_drop):.1f} km/h** no _{_pva_items[-1][0]}_. "
                            f"Equipa respondeu bem após intervalo."
                        )

                # ── 2. Velocidade máxima (destaque de atleta) ─────────────
                _vmax_champ, _vmax_val, _vmax_per = None, 0.0, ''
                for _pn, _pd in dados_posicao_por_periodo.items():
                    for _an, _av in _pd.items():
                        _avs = np.array(_av.get('vels_gps', _av.get('vel', [])), dtype=float)
                        if len(_avs) > 0 and float(_avs.max()) > _vmax_val:
                            _vmax_val = float(_avs.max())
                            _vmax_champ = _an
                            _vmax_per = _pn
                if _vmax_champ and _vmax_val > 0:
                    _vmax_flag = "⚠️ Verifique o sensor!" if _vmax_val > 32 else "✅"
                    _cai.append(
                        f"🚀 **Vel. máx**: {_vmax_champ.split(' ')[0]} — "
                        f"**{_vmax_val:.1f} km/h** ({_vmax_per}) {_vmax_flag}"
                    )

                # ── 3. Zona de alta intensidade geral ────────────────────
                _all_vels_cai = []
                for _pd in dados_posicao_por_periodo.values():
                    for _av in _pd.values():
                        _all_vels_cai.extend(_av.get('vels_gps', _av.get('vel', [])))
                if _all_vels_cai:
                    _av_arr = np.array(_all_vels_cai, dtype=float)
                    _hsr_p = float((_av_arr >= 14).mean() * 100)
                    _spr_p = float((_av_arr >= 21).mean() * 100)
                    if _hsr_p < 3:
                        _cai.append(
                            f"⚠️ **Intensidade baixa**: {_hsr_p:.1f}% >14 km/h — "
                            f"sessão de recuperação ou aquecimento."
                        )
                    elif _hsr_p > 18:
                        _cai.append(
                            f"🔥 **Alta intensidade elevada**: {_hsr_p:.1f}% >14 km/h | "
                            f"{_spr_p:.1f}% sprint. Monitorar recuperação."
                        )
                    else:
                        _cai.append(
                            f"✅ **Intensidade equilibrada**: {_hsr_p:.1f}% >14 km/h | "
                            f"{_spr_p:.1f}% sprint."
                        )

                # ── 4. Atleta com maior distância ─────────────────────────
                _dmax_champ, _dmax_val = None, 0.0
                for _pn, _rs in resultados_por_periodo.items():
                    if _pn == _CHAVE_COMBINADO:
                        continue
                    for _r in _rs:
                        _rd = _r.get('Distância (m)', 0)
                        if _rd > _dmax_val:
                            _dmax_val = _rd
                            _dmax_champ = _r.get('Atleta', '?')
                if _dmax_champ:
                    _vol_flag = " ⚠️ Volume muito alto!" if _dmax_val > 13000 else ""
                    _cai.append(
                        f"🏃 **Maior distância**: {_dmax_champ.split(' ')[0]} — "
                        f"**{_dmax_val:,.0f} m**{_vol_flag}"
                    )

                # ── 5. PlayerLoad mais elevado ────────────────────────────
                _plmax_champ, _plmax_val = None, 0.0
                for _pn, _rs in resultados_por_periodo.items():
                    if _pn == _CHAVE_COMBINADO:
                        continue
                    for _r in _rs:
                        _pl = _r.get('PlayerLoad', 0) or 0
                        if _pl > _plmax_val:
                            _plmax_val = _pl
                            _plmax_champ = _r.get('Atleta', '?')
                if _plmax_champ and _plmax_val > 0:
                    _cai.append(
                        f"⚡ **Maior PlayerLoad**: {_plmax_champ.split(' ')[0]} — "
                        f"**{_plmax_val:.0f}** UA"
                    )

                # ── Exibir ────────────────────────────────────────────────
                if _cai:
                    for _ins in _cai:
                        st.markdown(f"> {_ins}")
                        st.markdown("")
                else:
                    st.caption("Carregue dados GPS para ver insights automáticos.")
                st.caption("_Análise automática baseada nos dados carregados._")

        # ── Onboarding guiado — primeiro uso ─────────────────────────────────────
        if 'onboarding_done' not in st.session_state:
            st.session_state['onboarding_done'] = False
        if 'onboarding_step' not in st.session_state:
            st.session_state['onboarding_step'] = 1

        if not st.session_state['onboarding_done'] and 'api' not in st.session_state:
            with st.container():
                _ob_step = st.session_state['onboarding_step']

                _ob_steps = {
                    1: ("🔐 Token de Acesso", "Insira seu token Catapult Connect na sidebar à esquerda. O token é gerado em **Settings → API** na sua conta."),
                    2: ("🔄 Carregar Dados", "Clique em **🔄 Carregar Dados** na sidebar. Aguarde o carregamento de equipes, atletas e atividades."),
                    3: ("📅 Selecionar Sessão", "Escolha a **atividade** (sessão de treino ou jogo) e os **períodos** que deseja analisar."),
                    4: ("👥 Selecionar Atletas", "Busque atletas pelo nome ou use os **presets de posição**. Clique em ✅ Carregar Dados da Sessão."),
                }

                _ob_title, _ob_text = _ob_steps.get(_ob_step, ("✅ Pronto!", "Seu dashboard está configurado."))

                st.markdown(
                    f"<div style='background:linear-gradient(135deg,#1a3a5c,#0d2137);border:1px solid #2d5a8e;"
                    f"border-radius:12px;padding:20px 24px;margin-bottom:16px'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<div>"
                    f"<div style='color:#90CAF9;font-size:11px;font-weight:600;letter-spacing:1px'>PASSO {_ob_step}/4</div>"
                    f"<div style='color:white;font-size:18px;font-weight:700;margin:4px 0'>{_ob_title}</div>"
                    f"<div style='color:#bcd;font-size:13px'>{_ob_text}</div>"
                    f"</div>"
                    f"<div style='display:flex;gap:6px;align-items:center'>"
                    + "".join(f"<div style='width:8px;height:8px;border-radius:50%;background:{'#2196F3' if i+1==_ob_step else '#2d4a6a'}'></div>" for i in range(4))
                    + f"</div></div></div>",
                    unsafe_allow_html=True
                )

                _ob_col1, _ob_col2 = st.columns([1, 5])
                with _ob_col1:
                    if st.button("⏭️ Pular Tour", key="skip_onboarding", use_container_width=True):
                        st.session_state['onboarding_done'] = True
                        st.rerun()
                # Auto-advance when API is connected
                if _ob_step < 4 and 'df_activities' in st.session_state and not st.session_state.df_activities.empty:
                    st.session_state['onboarding_step'] = min(_ob_step + 1, 4)

        if resultados_por_periodo:
            st.subheader("📊 Métricas Biométricas")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏃 Atletas", len(st.session_state.atletas_sel))
            with col2:
                total_dist = 0
                for resultados in resultados_por_periodo.values():
                    for r in resultados:
                        total_dist += r.get('Distância (m)', 0)
                st.metric("📏 Distância Total", f"{total_dist:,.0f} m")
            with col3:
                total_pl = 0
                for resultados in resultados_por_periodo.values():
                    for r in resultados:
                        total_pl += r.get('PlayerLoad', 0)
                st.metric("⚡ PlayerLoad Total", f"{total_pl:,.0f}", help="Catapult PlayerLoad™: raiz quadrada da soma das acelerações ao quadrado nos 3 eixos (inercial)")
            with col4:
                max_vel = 0
                for resultados in resultados_por_periodo.values():
                    for r in resultados:
                        max_vel = max(max_vel, r.get('Velocidade Máx (km/h)', 0))
                st.metric("💨 Velocidade Máx", f"{max_vel:.1f} km/h")
            
            st.markdown("---")
            
            _main_tabs = st.tabs([
                "🏠 Resumo",
                "🗺️ Campo & GPS",
                "📈 Carga Física",
                "📊 Estatísticas",
                "📡 Ao Vivo",
            ])

            # ── Criar sub-tabs dentro de cada aba principal ────────────────────
            with _main_tabs[1]:
                _sub_campo = st.tabs(["🗺️ Campo de Futebol", "🎬 História do Jogo", "⚡ WCS"])
            with _main_tabs[2]:
                _sub_carga = st.tabs(["⏱️ Esforços", "📊 Janelas Temporais", "💪 Neuromuscular", "🏎️ Acc-Vel", "❤️ FC"])
            with _main_tabs[3]:
                _sub_stats = st.tabs(["📈 Gráficos Comparativos", "📋 Tabela Descritiva", "📊 Por Posição"])

            # Mapeamento: abas[N] aponta para o container correto na nova estrutura
            abas = [
                _sub_stats[0],    # 0: Gráficos Comparativos  → Estatísticas
                _sub_campo[0],    # 1: Campo de Futebol        → Campo & GPS
                _sub_carga[0],    # 2: Esforços                → Carga Física
                _sub_carga[1],    # 3: Janelas Temporais       → Carga Física
                _sub_carga[2],    # 4: Neuromuscular           → Carga Física
                _sub_carga[3],    # 5: Acc-Vel                 → Carga Física
                _sub_carga[4],    # 6: FC (TRIMP + Zonas)      → Carga Física
                _sub_stats[1],    # 7: Tabela Descritiva       → Estatísticas
                _sub_stats[2],    # 8: Por Posição             → Estatísticas
                _sub_campo[1],    # 9: História do Jogo        → Campo & GPS
                _main_tabs[4],    # 10: Ao Vivo               → Ao Vivo (tab principal)
            ]

            # ==================== ABA RESUMO: OVERVIEW DASHBOARD ====================
            with _main_tabs[0]:
                st.markdown("## 📊 Resumo da Sessão")

                if resultados_por_periodo:
                    # ── Coleta todas as linhas de todos os períodos (exceto combinado) ──
                    _ov_rows = []
                    for _p, _rs in resultados_por_periodo.items():
                        if _p == _CHAVE_COMBINADO:
                            continue
                        for _r in _rs:
                            _row = dict(_r)
                            _row['Período'] = _p
                            _ov_rows.append(_row)

                    if _ov_rows:
                        _df_ov_raw = pd.DataFrame(_ov_rows)

                        # ── Agrega por atleta — combina todos os períodos ────────────
                        # Métricas que se SOMAM entre períodos
                        _cols_sum = [c for c in [
                            'Distância (m)', 'Dist. > 19 km/h (m)', 'Dist. > 24 km/h (m)',
                            'Sprints (>24 km/h)', 'Acelerações (>3 m/s²)',
                            'Desacelerações (>-3 m/s²)', 'RHIE Blocos', 'PlayerLoad',
                            'TRIMP',
                        ] if c in _df_ov_raw.columns]
                        # Métricas que se tomam o MÁXIMO entre períodos
                        _cols_max = [c for c in [
                            'Velocidade Máx (km/h)', 'Velocidade Bruta Máx (km/h)',
                            'Aceleração Máx (m/s²)', 'Potência Met. Máx (W/kg)',
                        ] if c in _df_ov_raw.columns]
                        # Métricas que se tomam a MÉDIA entre períodos
                        _cols_mean = [c for c in [
                            'FC Média (bpm)', 'Velocidade Média (km/h)',
                        ] if c in _df_ov_raw.columns]
                        # Campos textuais: mantém o valor do primeiro período
                        _cols_first = [c for c in ['Posição', 'Equipe'] if c in _df_ov_raw.columns]

                        _agg_dict = {}
                        for _c in _cols_sum:  _agg_dict[_c] = 'sum'
                        for _c in _cols_max:  _agg_dict[_c] = 'max'
                        for _c in _cols_mean: _agg_dict[_c] = 'mean'
                        for _c in _cols_first: _agg_dict[_c] = 'first'

                        if _agg_dict and 'Atleta' in _df_ov_raw.columns:
                            _df_ov = (_df_ov_raw.groupby('Atleta', as_index=False)
                                      .agg(_agg_dict))
                            # Arredonda métricas numéricas
                            for _c in _cols_sum + _cols_max + _cols_mean:
                                if _c in _df_ov.columns:
                                    _df_ov[_c] = _df_ov[_c].round(1)
                        else:
                            _df_ov = _df_ov_raw.copy()

                        _n_periodos_ov = _df_ov_raw['Período'].nunique()

                        # ── KPI cards ────────────────────────────────────────────────
                        _ov_c1, _ov_c2, _ov_c3, _ov_c4, _ov_c5 = st.columns(5)
                        _ov_c1.metric("👥 Atletas", len(_df_ov['Atleta'].unique()) if 'Atleta' in _df_ov.columns else 0)
                        _ov_c2.metric("📏 Dist. Média", f"{_df_ov['Distância (m)'].mean():.0f} m" if 'Distância (m)' in _df_ov.columns else "—",
                                      help=f"Soma de todos os {_n_periodos_ov} período(s) por atleta, depois média do grupo")
                        _ov_c3.metric("💨 Vmax do Dia", f"{_df_ov['Velocidade Máx (km/h)'].max():.1f} km/h" if 'Velocidade Máx (km/h)' in _df_ov.columns else "—")
                        _ov_c4.metric("⚡ PL Total Médio", f"{_df_ov['PlayerLoad'].mean():.0f}" if 'PlayerLoad' in _df_ov.columns else "—",
                                      help="Catapult PlayerLoad™ somado em todos os períodos, depois média do grupo")
                        _hsr_col = 'Dist. > 19 km/h (m)'
                        _ov_c5.metric("🏃 HSR Médio", f"{_df_ov[_hsr_col].mean():.0f} m" if _hsr_col in _df_ov.columns else "—",
                                      help=f"HSR somado em {_n_periodos_ov} período(s) por atleta, depois média do grupo")

                        if _n_periodos_ov > 1:
                            st.caption(f"📋 Valores combinados de **{_n_periodos_ov} períodos** — somas, máximos e médias ponderadas por atleta.")

                        st.markdown("---")

                        # ── Gráfico de barras — distância combinada ────────────────
                        if 'Atleta' in _df_ov.columns and 'Distância (m)' in _df_ov.columns:
                            _fig_ov = go.Figure()
                            _fig_ov.add_trace(go.Bar(
                                x=_df_ov['Atleta'], y=_df_ov['Distância (m)'],
                                marker_color=[st.session_state.get('athlete_colors', {}).get(a, '#2196F3') for a in _df_ov['Atleta']],
                                text=_df_ov['Distância (m)'].round(0), textposition='outside',
                                hovertemplate='%{x}<br>Distância total: %{y:.0f} m<extra></extra>',
                            ))
                            _fig_ov.update_layout(
                                title=dict(
                                    text=f'Distância Total por Atleta ({_n_periodos_ov} período(s) combinado(s))',
                                    font=dict(color='white', size=14)
                                ),
                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                font=dict(color='white'), xaxis=dict(gridcolor='#333', tickangle=-30),
                                yaxis=dict(gridcolor='#333'), height=320, margin=dict(t=45,b=80,l=10,r=10),
                                showlegend=False,
                            )
                            st.plotly_chart(_fig_ov, use_container_width=True)

                        # ── Tabela combinada ────────────────────────────────────────
                        _ov_display_cols = [c for c in [
                            'Atleta', 'Posição', 'Distância (m)', 'Velocidade Máx (km/h)',
                            'Sprints (>24 km/h)', 'PlayerLoad', 'RHIE Blocos',
                            'Dist. > 19 km/h (m)', 'FC Média (bpm)',
                        ] if c in _df_ov.columns]
                        if _ov_display_cols:
                            _ov_sort_col = 'Distância (m)' if 'Distância (m)' in _ov_display_cols else _ov_display_cols[0]
                            st.dataframe(
                                _df_ov[_ov_display_cols].sort_values(_ov_sort_col, ascending=False).reset_index(drop=True),
                                use_container_width=True, hide_index=True
                            )
                else:
                    st.info("⚽ Carregue uma sessão na sidebar para visualizar o resumo.")
                    st.markdown("""
**Como começar:**
1. 🔐 Insira seu token Catapult na sidebar
2. 🔄 Clique em "Carregar Dados"
3. 📅 Selecione a atividade
4. 👥 Escolha os atletas
5. ✅ Clique em "Carregar Dados da Sessão"
                    """)

            # ==================== ABA 1: GRÁFICOS COMPARATIVOS ====================
            with abas[0]:
                st.subheader("📈 Gráficos Comparativos")

                # ── Filtro por Posição (conectado aos dados da atividade) ────────
                _gc_pos_dict: dict = {}   # posição → [atleta1, atleta2, ...]
                for _gc_per_data in resultados_por_periodo.values():
                    for _gc_r in _gc_per_data:
                        _gc_atl_n = _gc_r.get('Atleta', '')
                        _gc_pos_n = (_gc_r.get('Posição', '') or '').strip()
                        if _gc_atl_n:
                            _gc_pos_dict.setdefault(_gc_pos_n, [])
                            if _gc_atl_n not in _gc_pos_dict[_gc_pos_n]:
                                _gc_pos_dict[_gc_pos_n].append(_gc_atl_n)

                _gc_posicoes_disp = sorted([p for p in _gc_pos_dict if p])

                _gc_filter_col1, _gc_filter_col2 = st.columns([2, 3])
                with _gc_filter_col1:
                    if _gc_posicoes_disp:
                        _gc_pos_sel = st.selectbox(
                            "🎯 Filtrar por posição:",
                            ["— Todas as posições —"] + _gc_posicoes_disp,
                            key="gc_pos_filter",
                            help="Selecione uma posição para ver apenas os atletas daquela função tática.",
                        )
                    else:
                        _gc_pos_sel = "— Todas as posições —"
                        st.selectbox("🎯 Filtrar por posição:", ["— Sem dados de posição —"],
                                     key="gc_pos_filter_empty", disabled=True)

                if _gc_pos_sel != "— Todas as posições —" and _gc_pos_sel in _gc_pos_dict:
                    _gc_atletas_pos_filtro: set | None = set(_gc_pos_dict[_gc_pos_sel])
                    with _gc_filter_col2:
                        st.caption(
                            f"🎯 **{_gc_pos_sel}** — "
                            f"{len(_gc_atletas_pos_filtro)} atleta(s): "
                            f"{', '.join(sorted(_gc_atletas_pos_filtro))}"
                        )
                else:
                    _gc_atletas_pos_filtro = None   # sem filtro → todos os atletas

                st.markdown("---")
                st.markdown("### Modo de Comparação")

                modo_comparacao = st.radio(
                    "Escolha o tipo de comparação:",
                    ["Comparar Atletas no Mesmo Período", "Comparar Mesmo Atleta em Diferentes Períodos"],
                    horizontal=True
                )

                if modo_comparacao == "Comparar Atletas no Mesmo Período":
                    periodo_comp = st.selectbox("Selecione o período:", list(resultados_por_periodo.keys()), key="periodo_comp")

                    if periodo_comp in resultados_por_periodo and resultados_por_periodo[periodo_comp]:
                        df_comp = pd.DataFrame(resultados_por_periodo[periodo_comp])
                        # Aplica filtro por posição
                        if _gc_atletas_pos_filtro is not None:
                            df_comp = df_comp[df_comp['Atleta'].isin(_gc_atletas_pos_filtro)]
                        atletas_disponiveis = df_comp['Atleta'].tolist()
                        atletas_selecionados_comp = st.multiselect(
                            "Selecione os atletas para comparar:",
                            atletas_disponiveis,
                            default=atletas_disponiveis[:min(3, len(atletas_disponiveis))]
                        )
                        
                        if atletas_selecionados_comp:
                            df_filtrado = df_comp[df_comp['Atleta'].isin(atletas_selecionados_comp)]
                            _todas_metricas = ['Distância (m)', 'Dist. > 19 km/h (m)', 'Dist. > 24 km/h (m)',
                                               'PlayerLoad', 'Velocidade Máx (km/h)', 'Velocidade Média (km/h)',
                                               'FC Média (bpm)', 'Sprints (>24 km/h)', 'Acelerações (>3 m/s²)']
                            metricas_comp = [m for m in _todas_metricas if m in df_filtrado.columns]

                            # ── Mapeamento atleta → posição ────────────────────────────
                            _atleta_pos_map_gc: dict = {}
                            for _r_gc in resultados_por_periodo.get(periodo_comp, []):
                                _an_gc = _r_gc.get('Atleta', '')
                                _ap_gc = (_r_gc.get('Posição', '') or '').strip()
                                if _an_gc:
                                    _atleta_pos_map_gc[_an_gc] = _ap_gc

                            # Agrupa atletas selecionados por grupo tático para variação tonal
                            _gc_grupo_to_atls: dict = {}
                            for _atl_gc in atletas_selecionados_comp:
                                _grp_gc, _ = _get_pos_grupo(_atleta_pos_map_gc.get(_atl_gc, ''))
                                _gc_grupo_to_atls.setdefault(_grp_gc, []).append(_atl_gc)

                            # Cor final por atleta: tonalidade do grupo + variação de saturação/luminosidade
                            _color_map_comp: dict = {}
                            for _grp_gc, _atls_gc in _gc_grupo_to_atls.items():
                                for _vi_gc, _atl_gc in enumerate(_atls_gc):
                                    _color_map_comp[_atl_gc] = cor_atleta_pos(
                                        _atl_gc, _atleta_pos_map_gc.get(_atl_gc, ''),
                                        _vi_gc, len(_atls_gc)
                                    )

                            # ── Legenda de grupos de posição ───────────────────────────
                            _grupos_presentes_gc = sorted(set(
                                _get_pos_grupo(_atleta_pos_map_gc.get(a, ''))[0]
                                for a in atletas_selecionados_comp
                            ))
                            if _grupos_presentes_gc:
                                _leg_html = " &nbsp; ".join(
                                    f"<span style='background:{_POSICAO_COR_LEGENDA.get(g,'#9C27B0')};"
                                    f"color:white;padding:3px 12px;border-radius:14px;"
                                    f"font-size:12px;font-weight:600'>{g}</span>"
                                    for g in _grupos_presentes_gc
                                )
                                st.markdown(f"**Grupos táticos:** {_leg_html}", unsafe_allow_html=True)

                            col1, col2 = st.columns(2)
                            with col1:
                                df_melted = df_filtrado.melt(id_vars=['Atleta'], value_vars=metricas_comp, var_name='Métrica', value_name='Valor')
                                fig = px.bar(df_melted, x='Atleta', y='Valor', color='Atleta', barmode='group',
                                            title=f"Comparação de Métricas — {periodo_comp}",
                                            facet_col='Métrica', facet_col_wrap=3,
                                            color_discrete_map=_color_map_comp)
                                fig.update_layout(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                                  font=dict(color='white'), height=420,
                                                  showlegend=True, margin=dict(t=60,b=10))
                                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=10)))
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                if len(atletas_selecionados_comp) <= 8:
                                    fig_radar = go.Figure()
                                    _radar_df = df_filtrado[df_filtrado['Atleta'].isin(atletas_selecionados_comp[:8])][['Atleta'] + metricas_comp].copy()
                                    _radar_max = _radar_df[metricas_comp].max().replace(0, 1)
                                    for atleta in atletas_selecionados_comp[:8]:
                                        _row = _radar_df[_radar_df['Atleta'] == atleta][metricas_comp].iloc[0]
                                        valores_norm = (_row / _radar_max).tolist()
                                        _cor_radar = _color_map_comp.get(atleta, cor_atleta(atleta))
                                        fig_radar.add_trace(go.Scatterpolar(
                                            r=valores_norm + [valores_norm[0]],
                                            theta=metricas_comp + [metricas_comp[0]],
                                            fill='toself', opacity=0.4,
                                            name=atleta,
                                            line=dict(color=_cor_radar, width=2),
                                            fillcolor=_cor_radar,
                                        ))
                                    fig_radar.update_layout(
                                        polar=dict(
                                            bgcolor='#1e2130',
                                            radialaxis=dict(visible=True, range=[0,1],
                                                           gridcolor='#444', tickfont=dict(color='#aaa',size=8)),
                                            angularaxis=dict(gridcolor='#444', tickfont=dict(color='white',size=10)),
                                        ),
                                        paper_bgcolor='#0e1117', font=dict(color='white'),
                                        title=dict(text='Comparação Radial (Normalizado 0–1)', font=dict(color='white')),
                                        legend=dict(font=dict(color='white')),
                                        height=420, margin=dict(t=50,b=10),
                                    )
                                    st.plotly_chart(fig_radar, use_container_width=True)

                            # ── Tabela com fundo colorido por grupo de posição ─────────
                            _cols_show = [c for c in ['Atleta', 'Equipe', 'Posição'] + metricas_comp if c in df_filtrado.columns]
                            _df_show = df_filtrado[_cols_show].reset_index(drop=True)

                            def _style_gc_row(row):
                                _pos_r = (_atleta_pos_map_gc.get(row.get('Atleta', ''), '') or '')
                                _grp_r, _ = _get_pos_grupo(_pos_r)
                                _base_r = _POSICAO_COR_LEGENDA.get(_grp_r, '#9C27B0')
                                _hr2 = int(_base_r[1:3], 16)
                                _hg2 = int(_base_r[3:5], 16)
                                _hb2 = int(_base_r[5:7], 16)
                                _bg2 = f'rgba({_hr2},{_hg2},{_hb2},0.12)'
                                _fg2 = _base_r
                                return [
                                    (f'background-color:{_bg2};color:{_fg2};font-weight:700'
                                     if c == 'Posição'
                                     else f'background-color:{_bg2}')
                                    for c in row.index
                                ]

                            st.dataframe(
                                _df_show.style.apply(_style_gc_row, axis=1),
                                use_container_width=True
                            )
                
                else:
                    primeiro_periodo = list(resultados_por_periodo.keys())[0]
                    if resultados_por_periodo[primeiro_periodo]:
                        df_primeiro = pd.DataFrame(resultados_por_periodo[primeiro_periodo])
                        # Aplica filtro por posição
                        if _gc_atletas_pos_filtro is not None:
                            df_primeiro = df_primeiro[df_primeiro['Atleta'].isin(_gc_atletas_pos_filtro)]
                        _atls_para_comp = df_primeiro['Atleta'].tolist() if not df_primeiro.empty else []
                        atleta_comp = st.selectbox("Selecione o atleta:", _atls_para_comp, key="atleta_comp") if _atls_para_comp else None

                        if atleta_comp is None:
                            st.info("Nenhum atleta disponível com os filtros de posição aplicados.")

                        dados_atleta = {}
                        for periodo, resultados in resultados_por_periodo.items():
                            if resultados and atleta_comp:
                                df_periodo = pd.DataFrame(resultados)
                                atleta_data = df_periodo[df_periodo['Atleta'] == atleta_comp]
                                if not atleta_data.empty:
                                    dados_atleta[periodo] = atleta_data.iloc[0]
                        
                        if dados_atleta:
                            _primeiro_periodo_dados = list(dados_atleta.values())[0]
                            _todas_metricas_p = ['Distância (m)', 'Dist. > 19 km/h (m)', 'Dist. > 24 km/h (m)',
                                                 'PlayerLoad', 'Velocidade Máx (km/h)', 'Velocidade Média (km/h)',
                                                 'FC Média (bpm)', 'Sprints (>24 km/h)', 'Acelerações (>3 m/s²)']
                            metricas_comp = [m for m in _todas_metricas_p if m in _primeiro_periodo_dados.index]
                            df_comparativo = pd.DataFrame({
                                periodo: [dados_atleta[periodo].get(m, 0) for m in metricas_comp]
                                for periodo in dados_atleta.keys()
                            }, index=metricas_comp).T
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                fig = px.bar(df_comparativo, x=df_comparativo.index, y=metricas_comp,
                                            title=f"Comparação de {atleta_comp} entre Períodos", barmode='group')
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                fig_line = go.Figure()
                                for metrica in metricas_comp:
                                    fig_line.add_trace(go.Scatter(
                                        x=list(dados_atleta.keys()),
                                        y=[dados_atleta[p].get(metrica, 0) for p in dados_atleta.keys()],
                                        mode='lines+markers',
                                        name=metrica
                                    ))
                                fig_line.update_layout(title="Evolução das Métricas", xaxis_title="Período", yaxis_title="Valor")
                                st.plotly_chart(fig_line, use_container_width=True)
                            st.dataframe(df_comparativo, use_container_width=True)

                # ── #15: Análise de Duplas / Linhas ───────────────────────────
                st.markdown("---")
                st.markdown("## 🤝 Análise de Duplas e Linhas")
                st.caption(
                    "Selecione 2–6 atletas (uma linha, dupla de zaga, dupla de ataque…) "
                    "para analisar **sincronia de esforços**, **cobertura coletiva** e "
                    "**distância inter-jogadores** ao longo do jogo."
                )
                _duo_periodos = list(resultados_por_periodo.keys())
                if _duo_periodos:
                    _duo_per = st.selectbox("Período:", _duo_periodos, key="duo_periodo")
                    _duo_df_all = pd.DataFrame(resultados_por_periodo.get(_duo_per, []))
                    if not _duo_df_all.empty and 'Atleta' in _duo_df_all.columns:
                        if _gc_atletas_pos_filtro is not None:
                            _duo_df_all = _duo_df_all[_duo_df_all['Atleta'].isin(_gc_atletas_pos_filtro)]
                        _duo_atls_disp = _duo_df_all['Atleta'].tolist()
                        _duo_sel = st.multiselect(
                            "Selecione o grupo (2–6 atletas):",
                            _duo_atls_disp,
                            default=_duo_atls_disp[:min(4, len(_duo_atls_disp))],
                            max_selections=6,
                            key="duo_atletas",
                        )

                        if len(_duo_sel) >= 2:
                            _duo_df = _duo_df_all[_duo_df_all['Atleta'].isin(_duo_sel)].set_index('Atleta')
                            _duo_metricas = [m for m in [
                                'Distância (m)', 'Dist. > 19 km/h (m)', 'Dist. > 24 km/h (m)',
                                'PlayerLoad', 'Sprints (>24 km/h)', 'Acelerações (>3 m/s²)',
                                'Velocidade Máx (km/h)', 'FC Média (bpm)', 'TRIMP',
                            ] if m in _duo_df.columns]

                            # ── 1. Spider chart do grupo ──────────────────────
                            _duo_c1, _duo_c2 = st.columns(2)
                            with _duo_c1:
                                st.markdown("#### 🕸️ Perfil do Grupo")
                                _fig_duo_radar = go.Figure()
                                _duo_max = _duo_df[_duo_metricas].max().replace(0, 1)
                                for _da in _duo_sel:
                                    if _da in _duo_df.index:
                                        _row = _duo_df.loc[_da, _duo_metricas]
                                        _norm = (_row / _duo_max).tolist()
                                        _fig_duo_radar.add_trace(go.Scatterpolar(
                                            r=_norm + [_norm[0]],
                                            theta=_duo_metricas + [_duo_metricas[0]],
                                            fill='toself', opacity=0.45,
                                            name=_da,
                                            line=dict(color=cor_atleta(_da), width=2),
                                            fillcolor=cor_atleta(_da),
                                            hovertemplate='%{theta}: %{r:.2f}<extra>' + _da + '</extra>',
                                        ))
                                _fig_duo_radar.update_layout(
                                    polar=dict(
                                        bgcolor='#1e2130',
                                        radialaxis=dict(visible=True, range=[0,1], gridcolor='#444',
                                                        tickfont=dict(color='#aaa', size=8)),
                                        angularaxis=dict(gridcolor='#444', tickfont=dict(color='white', size=9)),
                                    ),
                                    paper_bgcolor='#0e1117', font=dict(color='white'),
                                    legend=dict(font=dict(color='white', size=10)),
                                    height=380, margin=dict(t=30,b=10),
                                    showlegend=True,
                                )
                                st.plotly_chart(_fig_duo_radar, use_container_width=True)

                            with _duo_c2:
                                # ── 2. Tabela comparativa com ranking por métrica ──
                                st.markdown("#### 📋 Comparativo do Grupo")
                                _duo_show = _duo_df[_duo_metricas].copy().reset_index()
                                _duo_show.columns = ['Atleta'] + _duo_metricas
                                # Destaca o melhor em cada métrica
                                st.dataframe(
                                    _duo_show.style.highlight_max(
                                        subset=_duo_metricas, color='rgba(42,157,143,0.4)', axis=0
                                    ).highlight_min(
                                        subset=_duo_metricas, color='rgba(230,57,70,0.3)', axis=0
                                    ),
                                    use_container_width=True, hide_index=True,
                                )

                                # ── 3. Sincronia de esforços ─────────────────────
                                st.markdown("#### ⚡ Sincronia de Esforços de Alta Intensidade")
                                _duo_sprints = {
                                    a: int(_duo_df.loc[a, 'Sprints (>24 km/h)'])
                                    if 'Sprints (>24 km/h)' in _duo_df.columns and a in _duo_df.index else 0
                                    for a in _duo_sel
                                }
                                _duo_fig_sync = go.Figure(go.Bar(
                                    x=list(_duo_sprints.keys()),
                                    y=list(_duo_sprints.values()),
                                    marker=dict(color=[cor_atleta(a) for a in _duo_sprints]),
                                    hovertemplate='%{x}: %{y} sprints<extra></extra>',
                                ))
                                _duo_fig_sync.update_layout(
                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(title='Sprints (>24 km/h)', gridcolor='#333'),
                                    height=220, margin=dict(t=10,b=10), showlegend=False,
                                )
                                st.plotly_chart(_duo_fig_sync, use_container_width=True)

                            # ── 4. Distância inter-jogadores (posição) ────────
                            _duo_pos_avail = all(
                                a in dados_posicao_por_periodo.get(_duo_per, {}) for a in _duo_sel
                            )
                            if _duo_pos_avail and len(_duo_sel) == 2:
                                st.markdown("#### 📏 Distância Entre os Dois Atletas ao Longo do Jogo")
                                _da1, _da2 = _duo_sel[0], _duo_sel[1]
                                _d1 = dados_posicao_por_periodo[_duo_per][_da1]
                                _d2 = dados_posicao_por_periodo[_duo_per][_da2]
                                _xs1 = np.array(_d1.get('xs', []))
                                _ys1 = np.array(_d1.get('ys', []))
                                _xs2 = np.array(_d2.get('xs', []))
                                _ys2 = np.array(_d2.get('ys', []))
                                _n_dist = min(len(_xs1), len(_xs2))
                                if _n_dist > 10:
                                    _dists = np.sqrt((_xs1[:_n_dist]-_xs2[:_n_dist])**2 +
                                                     (_ys1[:_n_dist]-_ys2[:_n_dist])**2)
                                    _t_dist = np.arange(_n_dist) / 10 / 60  # minutos
                                    # suavizar
                                    _dists_sm = savgol_filter(_dists, min(51, _n_dist//4*2+1), 3)
                                    _fig_dist = go.Figure()
                                    _fig_dist.add_trace(go.Scatter(
                                        x=_t_dist, y=_dists_sm,
                                        mode='lines', line=dict(color='#2196F3', width=2),
                                        fill='toself', fillcolor='rgba(33,150,243,0.1)',
                                        name='Distância',
                                        hovertemplate='%{x:.1f} min — %{y:.1f} m<extra></extra>',
                                    ))
                                    _fig_dist.add_hline(y=float(np.mean(_dists_sm)),
                                        line=dict(color='#FFD700', dash='dash', width=1.5),
                                        annotation_text=f"Média: {np.mean(_dists_sm):.1f}m",
                                        annotation_font=dict(color='#FFD700'))
                                    _fig_dist.update_layout(
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='Distância (m)', gridcolor='#333'),
                                        height=280, margin=dict(t=15,b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_dist, use_container_width=True)
                                    _d_k1, _d_k2, _d_k3 = st.columns(3)
                                    _d_k1.metric("Distância Média", f"{np.mean(_dists_sm):.1f} m")
                                    _d_k2.metric("Distância Máxima", f"{np.max(_dists_sm):.1f} m")
                                    _d_k3.metric("% do tempo > 20m", f"{np.sum(_dists_sm>20)/_n_dist*100:.1f}%")
                        else:
                            st.info("Selecione 2 a 6 atletas para ativar a análise de grupo.")
                    else:
                        st.info("Nenhum dado disponível para este período.")
                else:
                    st.info("Carregue dados para usar a análise de duplas/linhas.")

            # ==================== ABA 2: CAMPO DE FUTEBOL ====================
            with abas[1]:
                st.subheader("🗺️ Campo de Futebol — Análise de Movimentação")
                st.caption(REFERENCIAS["campo"])

                # ── Split View toggle ──────────────────────────────────────────
                _split_mode = st.toggle("⚖️ Modo Comparação (Split View)", value=False, key="split_view_mode")

                if _split_mode and dados_posicao_por_periodo:
                    _split_col_A, _split_col_B = st.columns(2)
                    for _split_idx, _split_col in enumerate([_split_col_A, _split_col_B]):
                        with _split_col:
                            _lbl = "A" if _split_idx == 0 else "B"
                            st.markdown(f"**Lado {_lbl}**")
                            _sp_per = st.selectbox(f"Período {_lbl}:", list(dados_posicao_por_periodo.keys()), key=f"split_per_{_lbl}")
                            _sp_ats = list(dados_posicao_por_periodo.get(_sp_per, {}).keys())
                            _sp_atl = st.selectbox(f"Atleta {_lbl}:", _sp_ats, key=f"split_atl_{_lbl}") if _sp_ats else None
                            if _sp_atl:
                                _sp_d = dados_posicao_por_periodo.get(_sp_per, {}).get(_sp_atl, {})
                                _sp_xs = list(_sp_d.get('xs', []))
                                _sp_ys = list(_sp_d.get('ys', []))
                                _sp_vs = list(_sp_d.get('vel', _sp_d.get('vels_gps', [])))

                                # ── Fallback GPS: converte lat/lon se não há x/y ────
                                _sp_gps_used = False
                                if not _sp_xs:
                                    _sp_lats = _sp_d.get('lats', [])
                                    _sp_lons = _sp_d.get('lons', [])
                                    if _sp_lats and _sp_lons:
                                        # Busca qualquer cfg de campo salvo em session_state
                                        _sp_cfg = None
                                        for _ssk, _ssv in st.session_state.items():
                                            if (isinstance(_ssk, str)
                                                    and _ssk.startswith('campo_cfg__')
                                                    and isinstance(_ssv, dict)
                                                    and _ssv.get('fl')):
                                                _sp_cfg = _ssv
                                                break
                                        if _sp_cfg:
                                            try:
                                                _sp_xs, _sp_ys = gps_para_campo_coords(
                                                    _sp_lats, _sp_lons, _sp_cfg)
                                                _sp_vs = list(_sp_d.get('vels_gps', [0.0]*len(_sp_xs)))
                                                _sp_gps_used = True
                                            except Exception:
                                                pass

                                if _sp_xs:
                                    # Direção de ataque para este lado do split view
                                    _sp_atk_opts = ["— Não definido", "➡️  Esq → Dir", "⬅️  Dir → Esq"]
                                    _sp_atk_key = f"attack_dir_split_{_lbl}"
                                    _sp_atk_default = 0
                                    if st.session_state.get(_sp_atk_key) == 'left_to_right':
                                        _sp_atk_default = 1
                                    elif st.session_state.get(_sp_atk_key) == 'right_to_left':
                                        _sp_atk_default = 2
                                    _sp_atk_sel = st.radio(
                                        "⚽ Ataque:",
                                        _sp_atk_opts,
                                        index=_sp_atk_default,
                                        horizontal=True,
                                        key=f"_sp_atk_radio_{_lbl}",
                                    )
                                    if "➡️" in _sp_atk_sel:
                                        st.session_state[_sp_atk_key] = 'left_to_right'
                                    elif "⬅️" in _sp_atk_sel:
                                        st.session_state[_sp_atk_key] = 'right_to_left'
                                    else:
                                        st.session_state[_sp_atk_key] = None
                                    _sp_fig = desenhar_campo_futebol_bonito(
                                        title=f"{_sp_atl} — {_sp_per}"
                                              + (" (GPS)" if _sp_gps_used else ""),
                                        attack_direction=st.session_state[_sp_atk_key],
                                    )
                                    _sp_step = max(1, len(_sp_xs) // 5000)
                                    _sp_vs_sub = ([_sp_vs[i] for i in range(0, len(_sp_vs), _sp_step)]
                                                  if _sp_vs else [0] * len(_sp_xs[::_sp_step]))
                                    _sp_fig.add_trace(go.Scatter(
                                        x=_sp_xs[::_sp_step], y=_sp_ys[::_sp_step],
                                        mode='markers',
                                        marker=dict(
                                            size=3,
                                            color=_sp_vs_sub,
                                            colorscale=[[0,'#2196F3'],[0.4,'#4CAF50'],[0.7,'#FF9800'],[1,'#F44336']],
                                            showscale=False,
                                        ),
                                        hoverinfo='skip', showlegend=False,
                                    ))
                                    _sp_fig.update_layout(height=400, margin=dict(t=30,b=10,l=10,r=10))
                                    st.plotly_chart(_sp_fig, use_container_width=True)
                                else:
                                    _sp_has_gps = bool(_sp_d.get('lats'))
                                    if _sp_has_gps:
                                        st.warning(
                                            "📡 Dados GPS disponíveis mas campo não calibrado.\n\n"
                                            "Vá até a aba **Campo & GPS** no modo normal, aplique o campo "
                                            "no mapa e retorne ao Split View."
                                        )
                                    else:
                                        st.info("Sem dados de posição (x/y ou GPS) para este atleta/período.")
                elif dados_posicao_por_periodo:
                    _todos_periodos = list(dados_posicao_por_periodo.keys())
                    col1, col2 = st.columns(2)
                    with col1:
                        periodos_mapa_sel = st.multiselect(
                            "Selecione o(s) período(s):",
                            options=_todos_periodos,
                            default=_todos_periodos[:1],
                            key="periodos_mapa_sel"
                        )
                    atleta_mapa = None
                    # Período de referência (primeiro selecionado) — usado para config do campo
                    periodo_mapa = periodos_mapa_sel[0] if periodos_mapa_sel else _todos_periodos[0]
                    with col2:
                        # União de atletas disponíveis em todos os períodos selecionados
                        _ats_disponiveis = []
                        for _pm in (periodos_mapa_sel or [periodo_mapa]):
                            for _a in dados_posicao_por_periodo.get(_pm, {}).keys():
                                if _a not in _ats_disponiveis:
                                    _ats_disponiveis.append(_a)
                        if _ats_disponiveis:
                            atletas_mapa = st.multiselect(
                                "Selecione o(s) atleta(s):",
                                _ats_disponiveis,
                                default=_ats_disponiveis[:1],
                                key="atletas_mapa"
                            )
                        else:
                            atletas_mapa = []
                    # Atleta primário: usado para GPS, calibração do campo e ETAPA 2
                    atleta_mapa = atletas_mapa[0] if atletas_mapa else None

                    if not periodos_mapa_sel:
                        st.info("Selecione pelo menos um período.")
                    elif atleta_mapa:
                        # Combina dados de todos os períodos selecionados
                        dados = dados_posicao_por_periodo.get(periodo_mapa, {}).get(atleta_mapa, {})

                        # GPS combinado (todos os períodos)
                        lats_gps, lons_gps, vels_gps, ts_gps = [], [], [], []
                        for _pm in periodos_mapa_sel:
                            _d = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                            lats_gps  += _d.get('lats', [])
                            lons_gps  += _d.get('lons', [])
                            vels_gps  += _d.get('vels_gps', [])
                            ts_gps    += _d.get('ts_gps', [])

                        n_xy  = sum(dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {}).get('n_pontos', 0) for _pm in periodos_mapa_sel)
                        n_gps = len(lats_gps)
                        _label_periodos = " + ".join(periodos_mapa_sel)
                        st.caption(f"📡 Pontos campo (x/y): **{n_xy}** &nbsp;|&nbsp; 🌍 Pontos GPS: **{n_gps}** &nbsp;|&nbsp; 📅 **{_label_periodos}**")

                        # ── Item 8: Badge de qualidade GPS ──────────────────────
                        st.caption("ℹ️ **HDOP** — Horizontal Dilution of Precision: qualidade do sinal GPS. <1.5 = excelente, 1.5–3 = bom, >3 = ruim", unsafe_allow_html=False)
                        _gps_q_parts = []
                        for _pm in periodos_mapa_sel:
                            _dq = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                            _pq_m  = _dq.get('pq_mean')
                            _hdop_m = _dq.get('hdop_mean')
                            _ref_m = _dq.get('ref_mean')
                            if _pq_m is not None or _hdop_m is not None:
                                _gps_q_parts.append((_pm, _pq_m, _hdop_m, _ref_m))
                        if _gps_q_parts:
                            _gps_badge_html = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px'>"
                            for _pm, _pq, _hdop, _ref in _gps_q_parts:
                                # pq: >80% good, 60-80% ok, <60% poor; hdop: <1.5 excellent, <3 good, >3 poor
                                _pq_color  = ('#4CAF50' if (_pq  or 0) >= 80 else '#FF9800' if (_pq  or 0) >= 60 else '#F44336') if _pq  is not None else '#888'
                                _hdop_color= ('#4CAF50' if (_hdop or 0) <= 1.5 else '#FF9800' if (_hdop or 0) <= 3.0 else '#F44336') if _hdop is not None else '#888'
                                _pq_txt    = f"PQ: <b style='color:{_pq_color}'>{_pq:.0f}%</b>" if _pq  is not None else ""
                                _hdop_txt  = f" HDOP: <b style='color:{_hdop_color}'>{_hdop:.2f}</b>" if _hdop is not None else ""
                                _ref_txt   = f" Satélites: <b>{_ref:.0f}</b>" if _ref is not None else ""
                                _gps_badge_html += (
                                    f"<span style='background:#1e2d3d;border:1px solid #2d4a6a;border-radius:8px;"
                                    f"padding:4px 10px;font-size:11px;color:#ccc'>"
                                    f"📡 {_pm} — {_pq_txt}{_hdop_txt}{_ref_txt}</span>"
                                )
                            _gps_badge_html += "</div>"
                            st.markdown(_gps_badge_html, unsafe_allow_html=True)

                        # ── Item 12: Odômetro vs distância integrada ─────────────
                        _odo_vals, _dist_vals = [], []
                        for _pm in periodos_mapa_sel:
                            _dq = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                            _odo = _dq.get('odometro_m')
                            if _odo is not None:
                                _odo_vals.append(_odo)
                        _res_rows = resultados_por_periodo.get(periodos_mapa_sel[0], []) if periodos_mapa_sel else []
                        _dist_integrada = next((r.get('Distância (m)', 0) for r in _res_rows if r.get('Atleta') == atleta_mapa), None)
                        if _odo_vals and _dist_integrada:
                            _odo_total = sum(_odo_vals)
                            _div = abs(_odo_total - _dist_integrada)
                            _div_pct = _div / max(_dist_integrada, 1) * 100
                            _odo_col = '#4CAF50' if _div_pct < 5 else '#FF9800' if _div_pct < 15 else '#F44336'
                            st.markdown(
                                f"<div style='display:flex;gap:12px;align-items:center;margin-bottom:6px;"
                                f"background:#0d1f0d;border-left:3px solid #4CAF50;padding:6px 12px;border-radius:0 6px 6px 0'>"
                                f"<span style='color:#86efac;font-size:12px'>🛣️ <b>Odômetro Catapult</b></span>"
                                f"<span style='color:white;font-size:13px'><b>{_odo_total:,.0f} m</b></span>"
                                f"<span style='color:#aaa;font-size:11px'>vs. GPS integrado: {_dist_integrada:,.0f} m</span>"
                                f"<span style='color:{_odo_col};font-size:11px'>Δ {_div_pct:.1f}%</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        # Chave por atleta (campo físico não muda entre períodos)
                        campo_key = f"campo_cfg__{atleta_mapa}"

                        # ── Auto-carga do banco compartilhado de venues ──────────
                        # Se o venue da atividade já foi configurado por algum usuário,
                        # aplica automaticamente — sem precisar passar pela FASE 1.
                        _venues_db   = _carregar_venues()
                        _venue_name  = st.session_state.get('venue', {}).get('name', '')
                        if (_venue_name and _venue_name in _venues_db
                                and campo_key not in st.session_state):
                            _vs = _venues_db[_venue_name]
                            st.session_state[campo_key] = {
                                'lat':          float(_vs.get('lat') or 0),
                                'lon':          float(_vs.get('lon') or 0),
                                'rot':          int(_vs.get('rot',   0)),
                                'fl':           int(_vs.get('fl',  105)),
                                'fw':           int(_vs.get('fw',   68)),
                                'ig':           int(_vs.get('ig',    3)),
                                '_from_venues': _venue_name,
                                '_saved_at':    _vs.get('saved_at', ''),
                            }

                        campo_aplicado = campo_key in st.session_state

                        # ══════════════════════════════════════════════════════
                        # FASE 1 — POSICIONAMENTO INTERATIVO NO SATÉLITE
                        # ══════════════════════════════════════════════════════
                        if not campo_aplicado:
                            st.markdown("### 1️⃣ Posicionamento no Campo Físico")
                            st.markdown(
                                "Ajuste o campo de futebol sobre a imagem de satélite e clique "
                                "**✅ Aplicar Campo** no painel inferior do mapa."
                            )

                            if lats_gps and lons_gps:
                                st.info(
                                    "📌 **Edite Lat/Lon** (ou use ↑↓) para mover o ⊙ amarelo · "
                                    "⚽ **Mostrar Campo** para ativar o overlay · "
                                    "Sliders ajustam rotação e dimensões **em tempo real** · "
                                    "**✅ Aplicar Campo** quando estiver satisfeito"
                                )

                                # Subamostrar GPS para o componente (máx 3000 pts)
                                _n = len(lats_gps)
                                _step = max(1, _n // 3000)
                                _pts = [
                                    {"lat": round(lats_gps[i], 7),
                                     "lon": round(lons_gps[i], 7),
                                     "v":   round(float(vels_gps[i]) if i < len(vels_gps) else 0.0, 1)}
                                    for i in range(0, _n, _step)
                                ]

                                # ── Venue da atividade → pré-popula campo ────────────
                                # Centro do mapa: sempre usa mediana GPS real dos atletas
                                # (o venue.lat/lng da API pode estar cadastrado num local errado).
                                # Dimensões e rotação do venue são usadas quando disponíveis.
                                _venue_info = st.session_state.get('venue', {})
                                _venue_rot  = int(_venue_info.get('rotation') or 0)
                                _venue_fl   = float(_venue_info.get('length')   or 105)
                                _venue_fw   = float(_venue_info.get('width')    or 68)

                                # Posição do mapa: mediana GPS > venue lat/lng > 0,0
                                if lats_gps:
                                    _lat_c = round(float(np.median(lats_gps)), 7)
                                    _lon_c = round(float(np.median(lons_gps)), 7)
                                else:
                                    _vl = _venue_info.get('lat')
                                    _vg = _venue_info.get('lng')
                                    _lat_c = round(float(_vl), 7) if _vl else 0.0
                                    _lon_c = round(float(_vg), 7) if _vg else 0.0

                                if _venue_info:
                                    st.caption(
                                        f"📡 Venue detectado: **{_venue_fl:.0f}×{_venue_fw:.0f}m** · "
                                        f"rot **{_venue_rot}°** · "
                                        f"centro **{_lat_c:.5f}, {_lon_c:.5f}** "
                                        f"_(pré-populado automaticamente)_"
                                    )

                                # Componente bidirecional: retorna {lat,lon,rot,fl,fw,ig}
                                # quando o usuário clica "✅ Aplicar Campo" no painel do mapa
                                resultado_campo = _campo_component(
                                    pts=_pts,
                                    lat_c=_lat_c,
                                    lon_c=_lon_c,
                                    rot=_venue_rot,
                                    fl=_venue_fl,
                                    fw=_venue_fw,
                                    key=f"campo_mapa_{atleta_mapa}",
                                    default=None
                                )

                                if resultado_campo is not None:
                                    _novo_cfg = {
                                        'lat': float(resultado_campo['lat']),
                                        'lon': float(resultado_campo['lon']),
                                        'rot': int(resultado_campo['rot']),
                                        'fl':  int(resultado_campo['fl']),
                                        'fw':  int(resultado_campo['fw']),
                                        'ig':  int(resultado_campo['ig']),
                                    }
                                    st.session_state[campo_key] = _novo_cfg
                                    st.success("✅ Campo aplicado com sucesso!")

                                    # ── Oferecer salvar no banco compartilhado ──────
                                    _vdb_atual  = _carregar_venues()
                                    _def_vname  = (_venue_name
                                                   or st.session_state.get('venue', {}).get('name', '')
                                                   or '')
                                    st.markdown("---")
                                    st.markdown("#### 💾 Deseja salvar no banco compartilhado de venues?")
                                    st.caption(
                                        "Outros usuários que abrirem uma atividade **neste mesmo venue** "
                                        "terão o campo configurado automaticamente."
                                    )
                                    with st.form("_form_save_venue"):
                                        _vname_inp = st.text_input(
                                            "Nome do venue:",
                                            value=_def_vname,
                                            placeholder="Ex: Estádio Municipal / CT do Clube",
                                        )
                                        if _vname_inp and _vname_inp.strip() in _vdb_atual:
                                            st.warning(
                                                f"⚠️ Já existe um venue salvo com este nome "
                                                f"(salvo em {_vdb_atual[_vname_inp.strip()].get('saved_at','?')}). "
                                                f"Confirme para sobrescrever."
                                            )
                                        _c1, _c2 = st.columns(2)
                                        with _c1:
                                            _btn_salvar = st.form_submit_button(
                                                "💾 Salvar para todos", type="primary")
                                        with _c2:
                                            _btn_pular  = st.form_submit_button("⏭ Pular")

                                    if _btn_salvar and _vname_inp.strip():
                                        _salvar_venue(_vname_inp.strip(), _novo_cfg)
                                        st.success(
                                            f"✅ Venue **{_vname_inp.strip()}** salvo! "
                                            f"Outros usuários serão beneficiados automaticamente."
                                        )
                                        st.session_state[campo_key]['_from_venues'] = _vname_inp.strip()
                                        st.session_state[campo_key]['_saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                                    if _btn_salvar or _btn_pular:
                                        st.rerun()
                            else:
                                st.warning(
                                    "⚠️ Nenhum ponto GPS real (lat/lon) encontrado para este atleta.\n\n"
                                    "Isso pode ocorrer se o sensor não obteve lock GPS durante a sessão."
                                )

                        # ══════════════════════════════════════════════════════
                        # FASE 2 — CAMPO APLICADO + ANÁLISE DE ESFORÇOS
                        # ══════════════════════════════════════════════════════
                        else:
                            cfg = st.session_state[campo_key]
                            _cfg_from_vn  = cfg.get('_from_venues', '')
                            _cfg_saved_at = cfg.get('_saved_at', '')

                            # ── Banner de origem ─────────────────────────────────
                            if _cfg_from_vn:
                                st.success(
                                    f"📦 Campo carregado automaticamente do banco compartilhado: "
                                    f"**{_cfg_from_vn}**"
                                    + (f" · salvo em {_cfg_saved_at}" if _cfg_saved_at else "")
                                )

                            # Botão para reajustar (volta à Fase 1)
                            col_hdr, col_btn, col_mgr = st.columns([4, 1, 1])
                            with col_hdr:
                                st.markdown(
                                    f"### 1️⃣ Campo Aplicado  "
                                    f"<span style='font-size:13px;color:#90CAF9'>"
                                    f"📍 {cfg['lat']:.5f}, {cfg['lon']:.5f} &nbsp; "
                                    f"🧭 {cfg['rot']}° &nbsp; "
                                    f"📏 {cfg['fl']}×{cfg['fw']}m</span>",
                                    unsafe_allow_html=True
                                )
                            with col_btn:
                                if st.button("🔄 Reajustar", key="btn_reajustar"):
                                    del st.session_state[campo_key]
                                    st.rerun()
                            with col_mgr:
                                if st.button("🗄️ Venues", key="btn_mgr_venues"):
                                    st.session_state['_show_venue_mgr'] = not st.session_state.get('_show_venue_mgr', False)

                            # ── Gerenciador de venues ────────────────────────────
                            if st.session_state.get('_show_venue_mgr', False):
                                with st.expander("🗄️ Banco Compartilhado de Venues", expanded=True):
                                    _vdb = _carregar_venues()
                                    if not _vdb:
                                        st.info("Nenhum venue salvo ainda. Configure um campo e clique em 💾 Salvar.")
                                    else:
                                        for _vn, _vc in _vdb.items():
                                            _v1, _v2, _v3, _v4 = st.columns([3, 2, 2, 1])
                                            with _v1:
                                                st.markdown(f"**{_vn}**")
                                            with _v2:
                                                st.caption(f"📏 {_vc.get('fl',105)}×{_vc.get('fw',68)}m · 🧭 {_vc.get('rot',0)}°")
                                            with _v3:
                                                st.caption(f"💾 {_vc.get('saved_at','?')}")
                                            with _v4:
                                                if st.button("🗑️", key=f"_del_vn_{_vn}",
                                                             help=f"Excluir {_vn}"):
                                                    _excluir_venue(_vn)
                                                    st.rerun()
                                    # Salvar config atual com novo nome
                                    st.markdown("---")
                                    st.markdown("**Salvar configuração atual:**")
                                    with st.form("_form_mgr_save"):
                                        _mgr_nome = st.text_input(
                                            "Nome:", value=_venue_name or '',
                                            key="_mgr_vname")
                                        if st.form_submit_button("💾 Salvar / Atualizar", type="primary"):
                                            if _mgr_nome.strip():
                                                _salvar_venue(_mgr_nome.strip(), cfg)
                                                st.success(f"✅ **{_mgr_nome.strip()}** salvo!")
                                                st.rerun()

                            # ── Esforço selecionado (para filtrar GPS no mapa e animação) ──
                            lats_eff, lons_eff, vels_eff, eff_desc = [], [], [], ""
                            _anim_start_ts  = 0.0
                            _anim_end_ts    = 0.0
                            _anim_effort_row = None

                            # ── Arrays de coordenadas (fonte única para ETAPA 2 e ETAPA 3) ──
                            # Construídos aqui para que detecção de esforços e visualização
                            # do campo usem exatamente os mesmos dados — garantindo que
                            # _seg_start_idx dos esforços indexe corretamente xn/yn.
                            xn, yn, vel_raw_campo, acc_raw_campo, ts_pos_campo = [], [], [], [], []
                            _fonte_xy = False
                            for _pm in periodos_mapa_sel:
                                _dc = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                                if _dc.get('xs') and _dc.get('ys') and _dc.get('vel'):
                                    xn  += list(_dc['xs'])
                                    yn  += list(_dc['ys'])
                                    vel_raw_campo += list(_dc['vel'])
                                    acc_raw_campo += list(_dc.get('acc', [0.0]*len(_dc['xs'])))
                                    ts_pos_campo  += list(_dc.get('ts_pos', []))
                                    _fonte_xy = True
                                elif lats_gps and cfg:
                                    _lats_pm = _dc.get('lats', [])
                                    _lons_pm = _dc.get('lons', [])
                                    if _lats_pm:
                                        _gx, _gy = gps_para_campo_coords(_lats_pm, _lons_pm, cfg)
                                        xn  += _gx
                                        yn  += _gy
                                        vel_raw_campo += _dc.get('vels_gps', [0.0]*len(_gx))
                                        acc_raw_campo += [0.0]*len(_gx)
                            _has_xy  = bool(xn and yn)
                            _has_gps = bool(lats_gps and lons_gps and cfg)

                            # ── Coordenadas de campo + GPS por atleta (ETAPA 2) ──
                            # Necessário para calcular esforços de todos os atletas e
                            # recuperar o segmento correto ao selecionar uma linha.
                            _coords_atletas = {}
                            for _atl in atletas_mapa:
                                _xna, _yna, _vela, _acca, _tsa = [], [], [], [], []
                                _latsa, _lonsa, _vgpsa, _tgpsa = [], [], [], []
                                for _pm in periodos_mapa_sel:
                                    _dca = dados_posicao_por_periodo.get(_pm, {}).get(_atl, {})
                                    if _dca.get('xs') and _dca.get('ys') and _dca.get('vel'):
                                        _xna  += list(_dca['xs'])
                                        _yna  += list(_dca['ys'])
                                        _vela += list(_dca['vel'])
                                        _acca += list(_dca.get('acc', [0.0]*len(_dca['xs'])))
                                        _tsa  += list(_dca.get('ts_pos', []))
                                    elif cfg:
                                        _lp = _dca.get('lats', [])
                                        _lo = _dca.get('lons', [])
                                        if _lp:
                                            _gx, _gy = gps_para_campo_coords(_lp, _lo, cfg)
                                            _xna  += _gx;  _yna  += _gy
                                            _vela += _dca.get('vels_gps', [0.0]*len(_gx))
                                            _acca += [0.0]*len(_gx)
                                    _latsa  += _dca.get('lats', [])
                                    _lonsa  += _dca.get('lons', [])
                                    _vgpsa  += _dca.get('vels_gps', [])
                                    _tgpsa  += _dca.get('ts_gps', [])
                                _coords_atletas[_atl] = dict(
                                    xn=_xna, yn=_yna, vel=_vela, acc=_acca, ts=_tsa,
                                    lats=_latsa, lons=_lonsa, vels_gps=_vgpsa, ts_gps=_tgpsa,
                                )

                            # Atleta do esforço selecionado (atualizado na seleção de linha)
                            _sel_atleta = atleta_mapa

                            # Mapa fixo (atualizado abaixo se houver esforço selecionado)
                            mapa_placeholder = st.empty()

                            st.divider()

                            # ── ETAPA 2: Análise de esforços ─────────────────────
                            st.markdown("### 2️⃣ Análise de Esforços no Campo")

                            _min_dur_vel_s = get_min_dur_vel_s()
                            _min_dur_acc_s = get_min_dur_s()

                            tipo_esf = st.radio(
                                "Tipo de esforço:",
                                ["⚡ Velocidade", "🔁 Aceleração"],
                                horizontal=True, key="tipo_esf_campo"
                            )

                            # ── Detecção de esforços para TODOS os atletas ──────────
                            efforts_df_full = pd.DataFrame()
                            _esf_usou_api = False
                            _frames_esf = []

                            for _atl in atletas_mapa:
                                _c = _coords_atletas[_atl]
                                _df_atl = pd.DataFrame()

                                # ── Tentativa 1: Esforços da API Catapult (fonte primária) ──
                                # Mais precisos — calculados pelos algoritmos oficiais
                                # da Catapult com os mesmos limiares do OpenField.
                                _raw_api: list = []
                                for _pm in periodos_mapa_sel:
                                    if tipo_esf == "⚡ Velocidade":
                                        _raw_api += dados_efforts_vel_por_periodo.get(
                                            _pm, {}).get(_atl, [])
                                    else:
                                        _raw_api += dados_efforts_acc_por_periodo.get(
                                            _pm, {}).get(_atl, [])
                                if _raw_api:
                                    _esf_usou_api = True
                                    _df_atl = (processar_efforts_velocidade(_raw_api)
                                               if tipo_esf == "⚡ Velocidade"
                                               else processar_efforts_aceleracao(_raw_api))

                                # ── Fallback: cálculo local com dados do sensor ──────
                                # Usado apenas se a API não retornou esforços.
                                if _df_atl.empty and _c['xn']:
                                    if tipo_esf == "⚡ Velocidade":
                                        _df_atl = calcular_efforts_velocidade_sensor(
                                            _c['xn'], _c['yn'], _c['vel'], _c['ts'],
                                            min_dur_s=_min_dur_vel_s)
                                    else:
                                        _df_atl = calcular_efforts_aceleracao_sensor(
                                            _c['xn'], _c['yn'], _c['acc'], _c['vel'], _c['ts'],
                                            min_dur_s=_min_dur_acc_s)

                                if not _df_atl.empty:
                                    _df_atl['Atleta'] = _atl
                                    _frames_esf.append(_df_atl)

                            if _frames_esf:
                                efforts_df_full = pd.concat(_frames_esf, ignore_index=True)
                                if '_start_ts' in efforts_df_full.columns and \
                                        efforts_df_full['_start_ts'].sum() > 0:
                                    efforts_df_full = efforts_df_full.sort_values(
                                        '_start_ts').reset_index(drop=True)
                                efforts_df_full['Esforço'] = range(1, len(efforts_df_full) + 1)

                                # ── FEATURE 13: Zona e Direção tática ────────────
                                _fl_tac = float(
                                    st.session_state.get('venue', {}).get('length') or 105)
                                efforts_df_full = enriquecer_esforcos_taticos(
                                    efforts_df_full, _coords_atletas, field_length=_fl_tac)

                            if not efforts_df_full.empty:
                                # Indicador de fonte de dados
                                if _esf_usou_api:
                                    st.caption(
                                        "✅ Esforços calculados pela **API Catapult** (algoritmos oficiais OpenField). "
                                        "Atletas sem dados de API usam cálculo local como fallback."
                                    )
                                else:
                                    st.caption(
                                        "⚙️ Esforços calculados **localmente** a partir dos dados do sensor "
                                        "(API Catapult não retornou esforços para estes atletas)."
                                    )

                                # Filtro de bandas
                                if 'Banda' in efforts_df_full.columns:
                                    bandas_disp = sorted(efforts_df_full['Banda'].dropna().unique())
                                    if bandas_disp:
                                        bandas_sel = st.multiselect(
                                            "Filtrar por bandas:", bandas_disp,
                                            default=bandas_disp, key="bandas_campo"
                                        )
                                        efforts_df_full = efforts_df_full[
                                            efforts_df_full['Banda'].isin(bandas_sel)
                                        ] if bandas_sel else efforts_df_full

                                if efforts_df_full.empty:
                                    st.info("Nenhum esforço encontrado após aplicar os filtros.")
                                else:
                                    # Colunas visíveis — "Atleta" primeiro, depois as demais
                                    _cols_ocultas = {c for c in efforts_df_full.columns
                                                     if c.startswith('_')}
                                    _cols_base = [c for c in efforts_df_full.columns
                                                  if c not in _cols_ocultas and c != 'Atleta']
                                    cols_show = (['Atleta'] + _cols_base
                                                 if 'Atleta' in efforts_df_full.columns
                                                 else _cols_base)
                                    efforts_df_show = efforts_df_full[cols_show]

                                    # Métricas resumidas
                                    mc1, mc2, mc3, mc4 = st.columns(4)
                                    with mc1: st.metric("Total de esforços", len(efforts_df_full))
                                    with mc2: st.metric("Duração total (s)", round(efforts_df_full['Duração (s)'].sum(), 1))
                                    with mc3:
                                        if 'Distância (m)' in efforts_df_full.columns:
                                            st.metric("Distância total (m)", round(efforts_df_full['Distância (m)'].sum(), 1))
                                    with mc4: st.metric("Média % máximo", round(efforts_df_full['% do Máximo'].mean(), 1))

                                    st.markdown(
                                        "**Selecione uma linha** para visualizar esse esforço no mapa de satélite acima. "
                                        "Clique novamente na linha selecionada para desfazer a seleção."
                                    )

                                    # Tabela interativa com seleção de linha
                                    evt = st.dataframe(
                                        efforts_df_show,
                                        use_container_width=True,
                                        height=360,
                                        on_select="rerun",
                                        selection_mode="single-row",
                                        key="tbl_esforcos_campo"
                                    )

                                    sel_rows = evt.selection.rows if evt.selection else []

                                    # Captura esforço selecionado + atleta correto
                                    if sel_rows:
                                        _ae_row = efforts_df_full.iloc[sel_rows[0]]
                                        _anim_start_ts   = float(_ae_row.get('_start_ts') or 0)
                                        _anim_end_ts     = float(_ae_row.get('_end_ts')   or 0)
                                        _anim_effort_row = _ae_row
                                        # Atleta dono do esforço selecionado
                                        _sel_atleta = str(_ae_row.get('Atleta', atleta_mapa))

                                    # Highlight GPS no mapa satélite (usa GPS do atleta correto)
                                    if sel_rows:
                                        sel_idx  = sel_rows[0]
                                        row_full = efforts_df_full.iloc[sel_idx]
                                        start_ts = row_full.get('_start_ts', 0)
                                        end_ts   = row_full.get('_end_ts', 0)
                                        _c_sel   = _coords_atletas.get(_sel_atleta,
                                                       _coords_atletas[atleta_mapa])
                                        _ts_gps_sel   = _c_sel['ts_gps']
                                        _lats_gps_sel = _c_sel['lats']
                                        _lons_gps_sel = _c_sel['lons']
                                        _vels_gps_sel = _c_sel['vels_gps']

                                        if start_ts and end_ts and \
                                                len(_ts_gps_sel) == len(_lats_gps_sel):
                                            filtered = [
                                                (la, lo, ve)
                                                for la, lo, ve, ts in zip(
                                                    _lats_gps_sel, _lons_gps_sel,
                                                    _vels_gps_sel, _ts_gps_sel)
                                                if start_ts <= ts <= end_ts
                                            ]
                                            if filtered:
                                                lats_eff = [p[0] for p in filtered]
                                                lons_eff = [p[1] for p in filtered]
                                                vels_eff = [p[2] for p in filtered]
                                                inicio_str = row_full.get('Início', '')
                                                dur_str    = row_full.get('Duração (s)', '')
                                                eff_desc   = (
                                                    f"Esforço #{row_full['Esforço']} "
                                                    f"({_sel_atleta}) — {inicio_str} — {dur_str}s"
                                                )
                                            else:
                                                st.warning("⚠️ Nenhum ponto GPS encontrado na janela de tempo deste esforço.")
                                        elif not _ts_gps_sel:
                                            st.info("ℹ️ Timestamps GPS não disponíveis.")

                                    # Download da tabela
                                    _atls_str = "_".join(a.replace(' ', '') for a in atletas_mapa)
                                    st.download_button(
                                        "📥 Exportar esforços",
                                        efforts_df_show.to_csv(index=False),
                                        file_name=f"esforcos_{_atls_str}_{_label_periodos.replace(' + ','_')}.csv"
                                    )
                            else:
                                st.info("ℹ️ Sem dados de posição (x/y) para calcular esforços. "
                                        "Os dados de sensor precisam incluir coordenadas de campo.")

                            # Renderiza o mapa fixo (com ou sem esforço destacado)
                            with mapa_placeholder:
                                if lats_gps and lons_gps:
                                    html_fixo = criar_html_campo_fixo(
                                        lats_gps, lons_gps, vels_gps, cfg,
                                        lats_eff=lats_eff or None,
                                        lons_eff=lons_eff or None,
                                        vels_eff=vels_eff or None,
                                        atleta_nome=atleta_mapa,
                                        esforco_desc=eff_desc,
                                        height=580
                                    )
                                    st.components.v1.html(html_fixo, height=580, scrolling=False)
                                else:
                                    st.warning("⚠️ Dados GPS não disponíveis para exibição do mapa.")

                            st.divider()

                            # ── ETAPA 3: Campo bonito + análise avançada ─────────
                            st.markdown("### 3️⃣ Análise de Movimentação no Campo")
                            # xn, yn, vel_raw_campo, etc. já foram construídos antes
                            # de ETAPA 2 para garantir consistência de índices.

                            # ── Computar coords do esforço ANTECIPADO ──────────────
                            # (usado no highlight do fig_campo E na animação abaixo)
                            # Usa arrays do atleta dono do esforço selecionado.
                            _c_eff  = _coords_atletas.get(_sel_atleta, _coords_atletas[atleta_mapa])
                            _xn_eff = _c_eff['xn'] or xn
                            _yn_eff = _c_eff['yn'] or yn
                            _vel_eff = _c_eff['vel'] or vel_raw_campo
                            _acc_eff = _c_eff['acc'] or acc_raw_campo
                            _ts_eff  = _c_eff['ts']  or ts_pos_campo

                            xs_a, ys_a, vel_a, acc_a = [], [], [], []
                            if _anim_effort_row is not None and _xn_eff:
                                # ── Tier 0: índices de segmento armazenados pelas
                                #    funções de sensor — indexam o array do atleta correto.
                                _seg_si = int(_anim_effort_row.get('_seg_start_idx', -1))
                                _seg_ei = int(_anim_effort_row.get('_seg_end_idx',   -1))
                                if 0 <= _seg_si < _seg_ei <= len(_xn_eff):
                                    xs_a  = _xn_eff[_seg_si:_seg_ei]
                                    ys_a  = _yn_eff[_seg_si:_seg_ei]
                                    vel_a = (_vel_eff[_seg_si:_seg_ei]
                                             if len(_vel_eff) == len(_xn_eff)
                                             else [0.0] * (_seg_ei - _seg_si))
                                    acc_a = (_acc_eff[_seg_si:_seg_ei]
                                             if len(_acc_eff) == len(_xn_eff)
                                             else [0.0] * (_seg_ei - _seg_si))

                                # ── Tier 1: matching por timestamp exato
                                # Tenta timestamps de campo primeiro; cai para GPS
                                # (campo x/y convertido de GPS → mesma indexação de ts_gps)
                                if len(xs_a) < 2 and _anim_start_ts > 0 and _anim_end_ts > 0:
                                    _ts1_src = []
                                    if _ts_eff and len(_ts_eff) == len(_xn_eff):
                                        _ts1_src = _ts_eff
                                    elif (_c_eff.get('ts_gps')
                                          and len(_c_eff['ts_gps']) == len(_xn_eff)):
                                        _ts1_src = _c_eff['ts_gps']
                                    if _ts1_src:
                                        _ts_c = np.array(_ts1_src, dtype=float)
                                        _m    = (_ts_c >= _anim_start_ts) & (_ts_c <= _anim_end_ts)
                                        if _m.any():
                                            xs_a  = np.array(_xn_eff)[_m].tolist()
                                            ys_a  = np.array(_yn_eff)[_m].tolist()
                                            vel_a = (np.array(_vel_eff)[_m].tolist()
                                                     if len(_vel_eff) == len(_xn_eff) else [0]*int(_m.sum()))
                                            acc_a = (np.array(_acc_eff)[_m].tolist()
                                                     if len(_acc_eff) == len(_xn_eff) else [0]*int(_m.sum()))

                                # ── Tier 2: fallback proporcional por timestamp
                                # Mesma lógica de source: campo → GPS
                                if len(xs_a) < 2 and _anim_start_ts > 0 and _anim_end_ts > 0:
                                    _ts2_src = (_ts_eff
                                                or _c_eff.get('ts_gps', []))
                                    if _ts2_src:
                                        _ts_min = min(_ts2_src)
                                        _ts_max = max(_ts2_src)
                                        if _ts_max > _ts_min:
                                            _sp = max(0.0, (_anim_start_ts - _ts_min) / (_ts_max - _ts_min))
                                            _ep = min(1.0, (_anim_end_ts   - _ts_min) / (_ts_max - _ts_min))
                                            _si = int(_sp * len(_xn_eff))
                                            _ei = min(len(_xn_eff), int(_ep * len(_xn_eff)) + 1)
                                            if _ei > _si + 1:
                                                xs_a  = _xn_eff[_si:_ei]
                                                ys_a  = _yn_eff[_si:_ei]
                                                vel_a = (_vel_eff[_si:_ei]
                                                         if len(_vel_eff) == len(_xn_eff) else [0]*(_ei-_si))
                                                acc_a = (_acc_eff[_si:_ei]
                                                         if len(_acc_eff) == len(_xn_eff) else [0]*(_ei-_si))

                                # ── Tier 3: fallback por string de início + duração
                                if len(xs_a) < 2:
                                    try:
                                        _dur_s   = float(_anim_effort_row.get('Duração (s)', 5))
                                        _ini_s   = 0.0
                                        _ini_str = str(_anim_effort_row.get('Início', ''))
                                        if ':' in _ini_str:
                                            _parts = _ini_str.replace('s', '').split(':')
                                            if len(_parts) == 3:
                                                _ini_s = int(_parts[0])*3600 + int(_parts[1])*60 + float(_parts[2])
                                            else:
                                                _ini_s = int(_parts[0])*60 + float(_parts[1])
                                        elif _ini_str.endswith('s'):
                                            _ini_s = float(_ini_str[:-1])
                                        _total_s = len(_xn_eff) / 10.0
                                        if _total_s > 0:
                                            _sp = max(0.0, min(1.0, _ini_s / _total_s))
                                            _ep = max(0.0, min(1.0, (_ini_s + _dur_s) / _total_s))
                                            _si = int(_sp * len(_xn_eff))
                                            _ei = min(len(_xn_eff), int(_ep * len(_xn_eff)) + 1)
                                            if _ei > _si + 1:
                                                xs_a  = _xn_eff[_si:_ei]
                                                ys_a  = _yn_eff[_si:_ei]
                                                vel_a = (_vel_eff[_si:_ei]
                                                         if len(_vel_eff) == len(_xn_eff)
                                                         else [0]*(_ei-_si))
                                                acc_a = (_acc_eff[_si:_ei]
                                                         if len(_acc_eff) == len(_xn_eff)
                                                         else [0]*(_ei-_si))
                                    except Exception:
                                        pass

                            if _fonte_xy:
                                st.caption("📡 Coordenadas x/y Catapult OpenField")
                            elif not _fonte_xy and _has_gps:
                                st.caption("🌍 Coordenadas derivadas do GPS + campo aplicado.")

                            vel_raw = vel_raw_campo
                            acc_raw = acc_raw_campo

                            if _has_xy or _has_gps:

                                if len(xn) > 0:
                                    # ── Linha 1: modo de visualização ────────────────
                                    col_modo, col_ov = st.columns([2, 3])
                                    with col_modo:
                                        modo_viz = st.radio(
                                            "🎨 Modo de visualização",
                                            ["🗺️ Trajetória",
                                             "⚡ Bandas de Velocidade",
                                             "🔁 Bandas de Aceleração"],
                                            key="modo_campo_v3"
                                        )
                                    with col_ov:
                                        st.markdown("**Overlays opcionais:**")
                                        oa, ob = st.columns(2)
                                        with oa:
                                            ov_setas   = st.checkbox("🏹 Setas de direção",    key="ov_setas")
                                            ov_hull    = st.checkbox("📐 Área de atuação",     key="ov_hull")
                                            ov_eventos = st.checkbox("⚽ Eventos Futebol",      key="ov_eventos")
                                        with ob:
                                            ov_tercos  = st.checkbox("📊 Terços do campo",     key="ov_tercos")
                                            ov_grade   = st.checkbox("🔲 Grade de quadrantes", key="ov_grade")
                                            ov_heatmap_comp = st.checkbox("🔥 Heatmap por Período",  key="ov_hcmp")
                                            ov_voronoi_comp = st.checkbox("🔷 Voronoi por Período",  key="ov_vcmp")

                                    # ── Direção de ataque por período ─────────────────
                                    _atk_key = f"attack_dir_{periodo_mapa}"
                                    _atk_opts = [
                                        "— Não definido",
                                        "➡️  Esq → Dir  (ataque para a direita)",
                                        "⬅️  Dir → Esq  (ataque para a esquerda)",
                                    ]
                                    _atk_default_idx = 0
                                    if st.session_state.get(_atk_key) == 'left_to_right':
                                        _atk_default_idx = 1
                                    elif st.session_state.get(_atk_key) == 'right_to_left':
                                        _atk_default_idx = 2
                                    _atk_sel = st.radio(
                                        f"⚽ Direção de ataque — {periodo_mapa}:",
                                        options=_atk_opts,
                                        index=_atk_default_idx,
                                        horizontal=True,
                                        key=f"_atk_radio_{periodo_mapa}",
                                    )
                                    if "➡️" in _atk_sel:
                                        st.session_state[_atk_key] = 'left_to_right'
                                    elif "⬅️" in _atk_sel:
                                        st.session_state[_atk_key] = 'right_to_left'
                                    else:
                                        st.session_state[_atk_key] = None
                                    _attack_dir_val = st.session_state[_atk_key]

                                    # ── Seletores de bandas (dependem do modo) ────────
                                    bandas_vel_sel = list(BANDAS_VEL.keys())
                                    bandas_acc_sel = list(BANDAS_ACC.keys())

                                    if modo_viz == "⚡ Bandas de Velocidade":
                                        bandas_vel_sel = st.multiselect(
                                            "Bandas de velocidade a exibir:",
                                            options=list(BANDAS_VEL.keys()),
                                            default=list(BANDAS_VEL.keys()),
                                            format_func=lambda k: BANDAS_VEL[k]['label'],
                                            key="ms_bv"
                                        )
                                    elif modo_viz == "🔁 Bandas de Aceleração":
                                        bandas_acc_sel = st.multiselect(
                                            "Bandas de aceleração a exibir:",
                                            options=list(BANDAS_ACC.keys()),
                                            default=list(BANDAS_ACC.keys()),
                                            format_func=lambda k: BANDAS_ACC[k]['label'],
                                            key="ms_ba"
                                        )

                                    # ── Configuração da grade ─────────────────────────
                                    # ── Seletor de eventos para o campo ──────────────
                                    _dados_ev_campo = {}
                                    _ev_tipos_sel   = []
                                    if ov_eventos:
                                        # Combina eventos de todos os períodos selecionados
                                        _ev_raw = {}
                                        for _pm in periodos_mapa_sel:
                                            _ev_pm = dados_eventos_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                                            for _et, _evlist in _ev_pm.items():
                                                _ev_raw.setdefault(_et, [])
                                                _ev_raw[_et] += _evlist
                                        if _ev_raw:
                                            # Enriquecer com posição no campo agora que temos cfg
                                            _ev_rich = enriquecer_eventos_com_posicao(
                                                _ev_raw,
                                                dados.get('ts_gps', []),
                                                dados.get('lats', []),
                                                dados.get('lons', []),
                                                dados.get('vels_gps', []),
                                                campo_config=cfg,
                                            )
                                            _ev_tipos_disp = [
                                                k for k in _ev_rich if _ev_rich[k]]
                                            if _ev_tipos_disp:
                                                _ev_tipos_sel = st.multiselect(
                                                    "Tipos de evento no campo:",
                                                    options=_ev_tipos_disp,
                                                    default=_ev_tipos_disp,
                                                    format_func=lambda k: FUTEBOL_EVENTS_CONFIG[k]['label'],
                                                    key="ev_campo_tipos"
                                                )
                                                _dados_ev_campo = _ev_rich
                                            else:
                                                st.info("Nenhum evento de futebol carregado para este atleta/período.")
                                        else:
                                            st.info("Nenhum evento de futebol carregado. Recarregue os dados com eventos ativados na sidebar.")

                                    n_cols_g, n_rows_g, zona_sel = 4, 3, None
                                    if ov_grade:
                                        gc1, gc2, gc3 = st.columns(3)
                                        with gc1:
                                            n_cols_g = st.slider("Colunas", 2, 10, 4, key="g_cols")
                                        with gc2:
                                            n_rows_g = st.slider("Linhas",  2, 8,  3, key="g_rows")
                                        with gc3:
                                            zonas_disp = ["— Nenhuma —"] + [
                                                f"{chr(65+r)}{c+1}"
                                                for r in range(n_rows_g)
                                                for c in range(n_cols_g)
                                            ]
                                            zona_sel = st.selectbox(
                                                "🔍 Detalhar zona:", zonas_disp, key="zona_det")
                                            if zona_sel == "— Nenhuma —":
                                                zona_sel = None

                                    # ── Construir figura ──────────────────────────────
                                    _titulo_campo = (
                                        f"📍 {' + '.join(atletas_mapa)} — {_label_periodos}"
                                        if len(atletas_mapa) > 1
                                        else f"📍 {atleta_mapa} — {_label_periodos}"
                                    )
                                    fig_campo = desenhar_campo_futebol_bonito(
                                        title=_titulo_campo,
                                        attack_direction=_attack_dir_val,
                                    )

                                    # ── Plota TODOS os atletas com o mesmo modo ───────
                                    # Usa _coords_atletas para garantir que cada atleta
                                    # receba exatamente o mesmo tratamento visual.
                                    # Prefixo do nome = atleta (visível na legenda).
                                    _n_atls = len(atletas_mapa)
                                    for _atl_i, _atl_viz in enumerate(atletas_mapa):
                                        _cv = _coords_atletas.get(_atl_viz, {})
                                        _xv = _cv.get('xn', [])
                                        _yv = _cv.get('yn', [])
                                        _velv = _cv.get('vel', [])
                                        _accv = _cv.get('acc', [])
                                        if not _xv:
                                            continue
                                        # Prefixo apenas quando há mais de 1 atleta
                                        _pfx = _atl_viz if _n_atls > 1 else ''
                                        if modo_viz == "🗺️ Trajetória":
                                            adicionar_trajetoria_campo(
                                                fig_campo, _xv, _yv, _velv, _atl_viz)
                                        elif modo_viz == "⚡ Bandas de Velocidade" and bandas_vel_sel:
                                            adicionar_pontos_velocidade_bandas(
                                                fig_campo, _xv, _yv, _velv, bandas_vel_sel,
                                                mostrar_setas=ov_setas,
                                                atleta_prefix=_pfx)
                                        elif modo_viz == "🔁 Bandas de Aceleração" and bandas_acc_sel:
                                            adicionar_pontos_aceleracao_bandas(
                                                fig_campo, _xv, _yv, _accv, bandas_acc_sel,
                                                mostrar_setas=ov_setas,
                                                atleta_prefix=_pfx)
                                        # Seta de trajetória por atleta
                                        if ov_setas and modo_viz == "🗺️ Trajetória":
                                            adicionar_setas_direcao(fig_campo, _xv, _yv)

                                    # Seta do esforço destacado (laranja, sobre tudo)
                                    if ov_setas and xs_a and len(xs_a) >= 2:
                                        adicionar_setas_direcao(
                                            fig_campo, xn, yn,
                                            xs_effort=xs_a, ys_effort=ys_a,
                                        )
                                    if ov_hull:
                                        adicionar_convex_hull(fig_campo, xn, yn)
                                    if ov_tercos:
                                        adicionar_tercos_campo(fig_campo, xn, yn)
                                    if ov_grade:
                                        adicionar_grade_quadrantes(
                                            fig_campo, xn, yn, n_cols_g, n_rows_g)
                                    if ov_eventos and _dados_ev_campo and _ev_tipos_sel:
                                        adicionar_eventos_campo(
                                            fig_campo, _dados_ev_campo, _ev_tipos_sel)

                                    # ── Highlight do esforço selecionado ─────────────
                                    if xs_a and len(xs_a) >= 2:
                                        # Halo laranja pulsante (linha mais grossa por baixo)
                                        fig_campo.add_trace(go.Scatter(
                                            x=xs_a, y=ys_a, mode='lines',
                                            line=dict(color='rgba(255,152,0,0.35)', width=14),
                                            name='_halo', showlegend=False, hoverinfo='skip'
                                        ))
                                        # Trajetória do esforço em laranja sólido
                                        fig_campo.add_trace(go.Scatter(
                                            x=xs_a, y=ys_a, mode='lines+markers',
                                            line=dict(color='#FF9800', width=4),
                                            marker=dict(size=3, color='#FF9800'),
                                            name=f"Esforço #{int(_anim_effort_row['Esforço'])}",
                                            showlegend=True, hoverinfo='skip'
                                        ))
                                        # Marcador de início (verde)
                                        fig_campo.add_trace(go.Scatter(
                                            x=[xs_a[0]], y=[ys_a[0]], mode='markers+text',
                                            marker=dict(size=14, color='#4CAF50', symbol='circle',
                                                        line=dict(color='white', width=2)),
                                            text=['▶'], textposition='top center',
                                            textfont=dict(color='white', size=10),
                                            name='Início', showlegend=False, hoverinfo='skip'
                                        ))
                                        # Marcador de fim (vermelho)
                                        fig_campo.add_trace(go.Scatter(
                                            x=[xs_a[-1]], y=[ys_a[-1]], mode='markers+text',
                                            marker=dict(size=14, color='#F44336', symbol='x',
                                                        line=dict(color='white', width=2)),
                                            text=['■'], textposition='top center',
                                            textfont=dict(color='white', size=10),
                                            name='Fim', showlegend=False, hoverinfo='skip'
                                        ))
                                        fig_campo.update_layout(
                                            title=dict(
                                                text=(
                                                    f"📍 {atleta_mapa} — {_label_periodos} &nbsp;|&nbsp; "
                                                    f"🟠 Esforço #{int(_anim_effort_row['Esforço'])} destacado"
                                                )
                                            )
                                        )

                                    st.plotly_chart(fig_campo, use_container_width=True)

                                    # ── FEATURE 12: Heatmap comparativo por período ──
                                    if ov_heatmap_comp and len(periodos_mapa_sel) >= 1:
                                        st.markdown("#### 🔥 Heatmap de Posicionamento por Período")
                                        _fl_hm = float(st.session_state.get('venue', {}).get('length') or 105)
                                        _fw_hm = float(st.session_state.get('venue', {}).get('width')  or 68)
                                        _hm_cols = st.columns(min(len(periodos_mapa_sel), 3))
                                        for _hm_i, _hm_pm in enumerate(periodos_mapa_sel[:3]):
                                            with _hm_cols[_hm_i]:
                                                _hm_xs, _hm_ys = [], []
                                                for _hm_atl in atletas_mapa:
                                                    _hm_d = dados_posicao_por_periodo.get(_hm_pm, {}).get(_hm_atl, {})
                                                    _hm_xs += _hm_d.get('xs', [])
                                                    _hm_ys += _hm_d.get('ys', [])
                                                if _hm_xs:
                                                    _fig_hm = desenhar_campo_futebol_bonito(
                                                        _fl_hm, _fw_hm, title=_hm_pm)
                                                    _xhm = np.array(_hm_xs)
                                                    _yhm = np.array(_hm_ys)
                                                    _H, _xe, _ye = np.histogram2d(
                                                        _xhm, _yhm, bins=[52, 34],
                                                        range=[[0, _fl_hm], [0, _fw_hm]])
                                                    _H = _gf(_H, sigma=2.5)
                                                    _xc = (_xe[:-1] + _xe[1:]) / 2
                                                    _yc = (_ye[:-1] + _ye[1:]) / 2
                                                    _fig_hm.add_trace(go.Heatmap(
                                                        x=_xc, y=_yc, z=_H.T,
                                                        colorscale='Hot', opacity=0.60,
                                                        showscale=False, hoverinfo='skip'))
                                                    _fig_hm.update_layout(
                                                        height=320,
                                                        margin=dict(l=5, r=5, t=35, b=5))
                                                    st.plotly_chart(_fig_hm, use_container_width=True)
                                                else:
                                                    st.info(f"Sem dados para {_hm_pm}")

                                    # ── FEATURE 14: Voronoi comparativo por período ──
                                    if ov_voronoi_comp and len(periodos_mapa_sel) >= 1:
                                        st.markdown("#### 🔷 Voronoi — Raio de Ação por Período")
                                        _vc_cols = st.columns(min(len(periodos_mapa_sel), 3))
                                        for _vc_i, _vc_pm in enumerate(periodos_mapa_sel[:3]):
                                            with _vc_cols[_vc_i]:
                                                _vc_pos = {}
                                                for _vc_atl in atletas_mapa:
                                                    _vc_d = dados_posicao_por_periodo.get(_vc_pm, {}).get(_vc_atl, {})
                                                    if _vc_d.get('xs'):
                                                        _vc_pos[_vc_atl] = {'xs': _vc_d['xs'], 'ys': _vc_d['ys']}
                                                if len(_vc_pos) >= 2:
                                                    _fig_vc = calcular_voronoi_campo(_vc_pos)
                                                    if _fig_vc:
                                                        _fig_vc.update_layout(
                                                            title=dict(text=_vc_pm, font=dict(size=13)),
                                                            height=320,
                                                            margin=dict(l=5, r=5, t=35, b=5),
                                                            showlegend=False)
                                                        st.plotly_chart(_fig_vc, use_container_width=True)
                                                    else:
                                                        st.info(f"Sem dados Voronoi para {_vc_pm}")
                                                else:
                                                    st.info(f"Voronoi precisa de ≥2 atletas com dados em {_vc_pm}")

                                    # ══════════════════════════════════════════════
                                    # ANIMAÇÃO DO ESFORÇO SELECIONADO
                                    # ══════════════════════════════════════════════
                                    if _anim_effort_row is not None:
                                        st.markdown("---")
                                        _vel_max_hdr = _anim_effort_row.get(
                                            'Vel. Máx (km/h)',
                                            _anim_effort_row.get('Acc. Máx (m/s²)', '—')
                                        )
                                        _vel_unit_hdr = (
                                            'km/h' if 'Vel. Máx (km/h)' in _anim_effort_row.index
                                            else 'm/s²'
                                        )
                                        st.markdown(
                                            f"### 🎬 Animação — Esforço **#{int(_anim_effort_row['Esforço'])}** &nbsp;|&nbsp; "
                                            f"Início: **{_anim_effort_row['Início']}** &nbsp;|&nbsp; "
                                            f"Duração: **{_anim_effort_row['Duração (s)']}s** &nbsp;|&nbsp; "
                                            f"Vel. Máx: **{_vel_max_hdr} {_vel_unit_hdr}**"
                                        )

                                        # xs_a / ys_a já foram calculados acima com fallbacks
                                        def _vc_anim(v):
                                            if v < 7:  return '#2196F3'
                                            if v < 14: return '#4CAF50'
                                            if v < 19: return '#FFEB3B'
                                            if v < 24: return '#FF9800'
                                            return '#F44336'

                                        if len(xs_a) >= 2:
                                            _n_pts = len(xs_a)
                                            _step_a = max(1, _n_pts // 80)  # máx 80 frames

                                            # ── Slider de velocidade ──────────────────
                                            _vel_opcoes = {
                                                "🐢 0.25×": 320,
                                                "🚶 0.5×":  160,
                                                "🏃 1× (real)": 80,
                                                "⚡ 2×":    40,
                                                "🚀 4×":    20,
                                            }
                                            _vel_sel = st.select_slider(
                                                "⏩ Velocidade da animação",
                                                options=list(_vel_opcoes.keys()),
                                                value="🏃 1× (real)",
                                                key="anim_speed_slider"
                                            )
                                            _frame_dur = _vel_opcoes[_vel_sel]

                                            # ── Campo base (estático) ─────────────────
                                            _fig_a = desenhar_campo_futebol_bonito(
                                                title=(
                                                    f"Esforço #{int(_anim_effort_row['Esforço'])} | "
                                                    f"{_anim_effort_row['Início']} | "
                                                    f"Vel. Máx {_anim_effort_row['Vel. Máx (km/h)']} km/h"
                                                )
                                            )
                                            _n_base_a = len(_fig_a.data)

                                            # Trajetória completa (fundo desbotado)
                                            _sbg = max(1, len(xn) // 5000)
                                            _fig_a.add_trace(go.Scatter(
                                                x=xn[::_sbg], y=yn[::_sbg], mode='markers',
                                                marker=dict(size=1.5, color='rgba(255,255,255,0.10)'),
                                                name='Trajetória', showlegend=False, hoverinfo='skip'
                                            ))
                                            # Marcadores início / fim do esforço (estáticos)
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[0]], y=[ys_a[0]], mode='markers+text',
                                                marker=dict(size=13, color='#4CAF50', symbol='circle',
                                                            line=dict(color='white', width=2)),
                                                text=['▶'], textposition='top center',
                                                textfont=dict(color='#4CAF50', size=11),
                                                name='Início', showlegend=False, hoverinfo='skip'
                                            ))
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[-1]], y=[ys_a[-1]], mode='markers+text',
                                                marker=dict(size=13, color='#F44336', symbol='x',
                                                            line=dict(color='white', width=2)),
                                                text=['■'], textposition='top center',
                                                textfont=dict(color='#F44336', size=11),
                                                name='Fim', showlegend=False, hoverinfo='skip'
                                            ))
                                            # Traço do esforço (animado)
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[0]], y=[ys_a[0]], mode='lines',
                                                line=dict(color='#FF9800', width=4),
                                                name='Esforço', showlegend=False
                                            ))
                                            # Marcador de posição atual (animado)
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[0]], y=[ys_a[0]], mode='markers',
                                                marker=dict(
                                                    size=18, color=_vc_anim(vel_a[0]),
                                                    symbol='circle',
                                                    line=dict(color='white', width=3)
                                                ),
                                                name='Posição', showlegend=False
                                            ))

                                            _idx_trace  = _n_base_a + 3  # traço do esforço
                                            _idx_dot    = _n_base_a + 4  # marcador posição

                                            # ── Frames de animação ────────────────────
                                            _frames_a = []
                                            for _k in range(0, _n_pts, _step_a):
                                                _t  = _k * 0.1
                                                _v  = vel_a[_k] if _k < len(vel_a) else 0
                                                _ac = acc_a[_k] if _k < len(acc_a) else 0
                                                _col = _vc_anim(_v)
                                                _frames_a.append(go.Frame(
                                                    data=[
                                                        go.Scatter(
                                                            x=xs_a[:_k+1], y=ys_a[:_k+1],
                                                            mode='lines',
                                                            line=dict(color=_col, width=4),
                                                        ),
                                                        go.Scatter(
                                                            x=[xs_a[_k]], y=[ys_a[_k]],
                                                            mode='markers',
                                                            marker=dict(
                                                                size=18, color=_col, symbol='circle',
                                                                line=dict(color='white', width=3)
                                                            ),
                                                        ),
                                                    ],
                                                    traces=[_idx_trace, _idx_dot],
                                                    name=str(_k),
                                                    layout=go.Layout(title=dict(
                                                        text=(
                                                            f"⏱️ {_t:.1f}s / {_anim_effort_row['Duração (s)']}s"
                                                            f"   |   💨 {_v:.1f} km/h"
                                                            f"   |   ⚡ {_ac:+.2f} m/s²"
                                                            f"   |   📍 x={xs_a[_k]:.1f}m y={ys_a[_k]:.1f}m"
                                                        ),
                                                        font=dict(color='white', size=12)
                                                    ))
                                                ))
                                            _fig_a.frames = _frames_a

                                            # ── Controles Play/Pause + Slider ─────────
                                            _slider_steps = [
                                                {
                                                    'args': [[f.name],
                                                             {'frame': {'duration': 0, 'redraw': True},
                                                              'mode': 'immediate',
                                                              'transition': {'duration': 0}}],
                                                    'label': f"{int(f.name)*0.1:.0f}s",
                                                    'method': 'animate',
                                                }
                                                for f in _frames_a[::max(1, len(_frames_a)//15)]
                                            ]
                                            _fig_a.update_layout(
                                                height=560,
                                                margin=dict(t=55, b=110, l=10, r=10),
                                                updatemenus=[{
                                                    'buttons': [
                                                        {
                                                            'args': [None, {
                                                                'frame': {'duration': _frame_dur, 'redraw': True},
                                                                'fromcurrent': True,
                                                                'transition': {'duration': 0},
                                                            }],
                                                            'label': '▶  Play',
                                                            'method': 'animate',
                                                        },
                                                        {
                                                            'args': [[None], {
                                                                'frame': {'duration': 0, 'redraw': False},
                                                                'mode': 'immediate',
                                                                'transition': {'duration': 0},
                                                            }],
                                                            'label': '⏸  Pause',
                                                            'method': 'animate',
                                                        },
                                                    ],
                                                    'direction': 'left',
                                                    'pad': {'r': 10, 't': 10},
                                                    'showactive': True,
                                                    'type': 'buttons',
                                                    'x': 0.0, 'y': -0.07,
                                                    'bgcolor': '#1565C0',
                                                    'font': {'color': 'white', 'size': 13},
                                                    'bordercolor': '#0D47A1',
                                                }],
                                                sliders=[{
                                                    'active': 0,
                                                    'steps': _slider_steps,
                                                    'currentvalue': {
                                                        'prefix': '⏱️ ',
                                                        'visible': True,
                                                        'font': {'color': 'white', 'size': 12},
                                                    },
                                                    'tickcolor': 'white',
                                                    'font': {'color': 'white'},
                                                    'y': -0.02,
                                                    'len': 0.88,
                                                    'x': 0.1,
                                                    'bgcolor': '#1a1a2e',
                                                    'bordercolor': '#555',
                                                }],
                                            )
                                            st.plotly_chart(_fig_a, use_container_width=True)

                                            # Painel de info abaixo
                                            _ai1, _ai2, _ai3, _ai4, _ai5 = st.columns(5)
                                            _ai1.metric("⏱️ Duração",       f"{_anim_effort_row['Duração (s)']}s")
                                            _ai2.metric("💨 Vel. Máx",      f"{_anim_effort_row['Vel. Máx (km/h)']} km/h")
                                            _ai3.metric("🏁 Vel. Inicial",  f"{_anim_effort_row['Vel. Inicial (km/h)']} km/h")
                                            _ai4.metric("📏 Distância",     f"{_anim_effort_row['Distância (m)']} m")
                                            _ai5.metric("📊 % do Máximo",   f"{_anim_effort_row['% do Máximo']}%")

                                            # ──────────────────────────────────────────────
                                            # PERFIL DE SPRINT — FASES E EFICIÊNCIA (item 6)
                                            # ──────────────────────────────────────────────
                                            st.markdown("---")
                                            st.markdown("#### 🏃 Perfil do Sprint — Fases e Eficiência")
                                            st.caption(
                                                "Decomposição em 3 fases: 🟠 Aceleração (derivada >+0.5 m/s²) · "
                                                "🔴 Pico (≥95% da vel. máxima) · 🔵 Desaceleração (derivada <−0.5 m/s²)"
                                            )

                                            _sp_vel = np.array(vel_a, dtype=float)
                                            _sp_acc = np.array(acc_a, dtype=float)
                                            _sp_t   = np.arange(len(_sp_vel)) * 0.1

                                            # Suavização para detecção de fases
                                            _sp_wl = min(11, len(_sp_vel) - (1 - len(_sp_vel) % 2))
                                            if len(_sp_vel) >= 5:
                                                from scipy.signal import savgol_filter as _svgf
                                                _sp_sm = np.clip(_svgf(_sp_vel, max(5, _sp_wl), 2), 0, None)
                                            else:
                                                _sp_sm = np.clip(_sp_vel, 0, None)

                                            _sp_grad = np.gradient(_sp_sm, 0.1)
                                            _sp_vmax = float(_sp_sm.max()) if _sp_sm.max() > 0 else 1.0

                                            _sp_phases = np.where(
                                                _sp_sm >= _sp_vmax * 0.95, 2,           # pico
                                                np.where(_sp_grad >= 0.5, 1,            # aceleração
                                                np.where(_sp_grad <= -0.5, 3, 2))       # desaceleração / pico
                                            )

                                            _ph_clrs = {1: '#FFA726', 2: '#EF5350', 3: '#42A5F5'}
                                            _ph_lbls = {1: 'Aceleração', 2: 'Pico', 3: 'Desaceleração'}

                                            _fig_sp = go.Figure()
                                            # Linha de velocidade (fundo)
                                            _fig_sp.add_trace(go.Scatter(
                                                x=_sp_t, y=_sp_sm, mode='lines',
                                                line=dict(color='rgba(255,255,255,0.25)', width=1.5),
                                                showlegend=False, hoverinfo='skip'
                                            ))
                                            # Pontos coloridos por fase
                                            for _ph in [1, 2, 3]:
                                                _msk_ph = _sp_phases == _ph
                                                if _msk_ph.any():
                                                    _fig_sp.add_trace(go.Scatter(
                                                        x=_sp_t[_msk_ph],
                                                        y=_sp_sm[_msk_ph],
                                                        mode='markers',
                                                        name=_ph_lbls[_ph],
                                                        marker=dict(size=6,
                                                                    color=_ph_clrs[_ph],
                                                                    line=dict(width=0)),
                                                    ))
                                            # Linha de vel. máxima
                                            _fig_sp.add_hline(
                                                y=_sp_vmax, line_dash='dot',
                                                line_color='rgba(255,255,255,0.5)', line_width=1,
                                                annotation_text=f"Vmáx {_sp_vmax:.1f} km/h",
                                                annotation_font_color='white',
                                                annotation_font_size=10,
                                            )
                                            _fig_sp.update_layout(
                                                xaxis=dict(title='Tempo (s)', color='white',
                                                           gridcolor='rgba(255,255,255,0.1)'),
                                                yaxis=dict(title='Velocidade (km/h)', color='white',
                                                           gridcolor='rgba(255,255,255,0.1)'),
                                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                                font=dict(color='white'), height=270,
                                                legend=dict(font=dict(color='white'),
                                                            orientation='h', y=1.10),
                                                margin=dict(t=10, b=40, l=55, r=10),
                                            )
                                            st.plotly_chart(_fig_sp, use_container_width=True)

                                            # Métricas por fase
                                            _sp_mc = st.columns(3)
                                            _ph_icons = {1: '🟠', 2: '🔴', 3: '🔵'}
                                            for _phi in [1, 2, 3]:
                                                _msk_ph = _sp_phases == _phi
                                                _ph_dur  = float(_msk_ph.sum()) * 0.1
                                                _ph_vavg = (float(_sp_sm[_msk_ph].mean())
                                                            if _msk_ph.any() else 0.0)
                                                _ph_dist = (float((_sp_sm[_msk_ph] / 3.6 / 10).sum())
                                                            if _msk_ph.any() else 0.0)
                                                _ph_gacc = (float(_sp_acc[_msk_ph].mean())
                                                            if _msk_ph.any() and len(_sp_acc) == len(_sp_vel)
                                                            else 0.0)
                                                with _sp_mc[_phi - 1]:
                                                    st.markdown(
                                                        f"**{_ph_icons[_phi]} {_ph_lbls[_phi]}**"
                                                    )
                                                    st.metric("Duração", f"{_ph_dur:.1f} s")
                                                    st.metric("Vel. Média", f"{_ph_vavg:.1f} km/h")
                                                    st.metric("Distância", f"{_ph_dist:.1f} m")
                                                    if abs(_ph_gacc) > 0.01:
                                                        st.metric(
                                                            "Acc. Média",
                                                            f"{_ph_gacc:+.2f} m/s²"
                                                        )
                                        else:
                                            st.warning(
                                                "⚠️ Não foi possível localizar os pontos de campo para este esforço. "
                                                "Verifique se o campo está configurado corretamente na aba **🗺️ Campo de Futebol** "
                                                "e se os timestamps do esforço correspondem aos dados de posição carregados."
                                            )
                                    else:
                                        st.info("💡 Selecione um esforço na tabela acima para ver a **animação no campo**.")

                                    # ── Detalhes de zona selecionada ──────────────────
                                    if zona_sel and ov_grade:
                                        r_idx = ord(zona_sel[0]) - 65
                                        c_idx = int(zona_sel[1:]) - 1
                                        cw_g  = 105.0 / n_cols_g
                                        rh_g  = 68.0  / n_rows_g
                                        st_z  = stats_quadrante(
                                            xn, yn, vel_raw, acc_raw,
                                            c_idx*cw_g, (c_idx+1)*cw_g,
                                            r_idx*rh_g, (r_idx+1)*rh_g)
                                        st.markdown(f"#### 🔍 Zona **{zona_sel}** — Detalhes")
                                        z1,z2,z3,z4,z5,z6 = st.columns(6)
                                        with z1: st.metric("% do Tempo",     f"{st_z['pct']:.1f}%")
                                        with z2: st.metric("Pontos",          st_z['n_pontos'])
                                        with z3: st.metric("Vel. Média",     f"{st_z['vel_media']:.1f} km/h")
                                        with z4: st.metric("Vel. Máx",       f"{st_z['vel_max']:.1f} km/h")
                                        with z5: st.metric("Acc. Média",     f"{st_z['acc_media']:.2f} m/s²")
                                        with z6: st.metric("|Acc| Máx",      f"{st_z['acc_max']:.2f} m/s²")

                                    # ── Estatísticas gerais ───────────────────────────
                                    st.markdown("#### 📊 Estatísticas de Movimentação")
                                    sc1, sc2, sc3, sc4 = st.columns(4)
                                    with sc1:
                                        dist_km = sum(vel_raw) * 0.1 / 3600
                                        st.metric("Distância Total", f"{dist_km:.2f} km")
                                    with sc2:
                                        xr = max(xn) - min(xn)
                                        st.metric("Largura Atuação", f"{xr:.0f} m")
                                    with sc3:
                                        yr = max(yn) - min(yn)
                                        st.metric("Profundidade", f"{yr:.0f} m")
                                    with sc4:
                                        try:
                                            from scipy.spatial import ConvexHull as _CH
                                            _area = _CH(np.column_stack([xn, yn])).volume
                                            st.metric("Área de Atuação", f"{_area:.0f} m²")
                                        except Exception:
                                            st.metric("Área (bbox)", f"{xr*yr:.0f} m²")

                                    # ── Comparação (Períodos ou Atletas) ─────────────
                                    st.markdown("---")
                                    st.markdown("#### 🔄 Comparação")
                                    _cmp_modo = st.radio(
                                        "Comparar:",
                                        ["📅 Períodos", "👤 Atletas"],
                                        horizontal=True, key="cmp_modo"
                                    )

                                    if _cmp_modo == "📅 Períodos":
                                        periodos_lista = list(dados_posicao_por_periodo.keys())
                                        if len(periodos_lista) < 2:
                                            st.info("Carregue pelo menos 2 períodos para usar esta comparação.")
                                        else:
                                            cp1, cp2 = st.columns(2)
                                            with cp1:
                                                per1 = st.selectbox("Período A:", periodos_lista,
                                                                     index=0, key="cmp_p1")
                                            with cp2:
                                                per2 = st.selectbox("Período B:", periodos_lista,
                                                                     index=min(1, len(periodos_lista)-1),
                                                                     key="cmp_p2")

                                            if per1 != per2:
                                                d1c = dados_posicao_por_periodo.get(per1, {}).get(atleta_mapa, {})
                                                d2c = dados_posicao_por_periodo.get(per2, {}).get(atleta_mapa, {})
                                                cfg_ref = st.session_state.get(f"campo_cfg__{atleta_mapa}", cfg)
                                                if d1c.get('xs') and d2c.get('xs'):
                                                    x1c, y1c = list(d1c['xs']), list(d1c['ys'])
                                                    x2c, y2c = list(d2c['xs']), list(d2c['ys'])
                                                elif d1c.get('lats') and d2c.get('lats') and cfg_ref:
                                                    x1c, y1c = gps_para_campo_coords(d1c['lats'], d1c['lons'], cfg_ref)
                                                    x2c, y2c = gps_para_campo_coords(d2c['lats'], d2c['lons'], cfg_ref)
                                                else:
                                                    x1c = x2c = []
                                                if x1c and x2c:
                                                    fig_cmp = desenhar_campo_futebol_bonito(
                                                        title=f"Comparação: {per1} (azul) vs {per2} (rosa)")
                                                    _s1 = max(1, len(x1c)//3000)
                                                    _s2 = max(1, len(x2c)//3000)
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=x1c[::_s1], y=y1c[::_s1], mode='markers', name=per1,
                                                        marker=dict(size=2, color='#00E5FF', opacity=0.5)))
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=x2c[::_s2], y=y2c[::_s2], mode='markers', name=per2,
                                                        marker=dict(size=2, color='#FF4081', opacity=0.5)))
                                                    st.plotly_chart(fig_cmp, use_container_width=True)
                                                else:
                                                    st.info("Dados de posição não disponíveis em um dos períodos.")
                                            else:
                                                st.info("Selecione dois períodos diferentes para comparar.")

                                    else:  # Comparar Atletas
                                        _ats_cmp = _ats_disponiveis if _ats_disponiveis else [atleta_mapa]
                                        if len(_ats_cmp) < 2:
                                            st.info("Carregue pelo menos 2 atletas para usar esta comparação.")
                                        else:
                                            ca1, ca2 = st.columns(2)
                                            with ca1:
                                                atl_A = st.selectbox("Atleta A:", _ats_cmp,
                                                                      index=0, key="cmp_a1")
                                            with ca2:
                                                atl_B = st.selectbox("Atleta B:", _ats_cmp,
                                                                      index=min(1, len(_ats_cmp)-1),
                                                                      key="cmp_a2")

                                            if atl_A != atl_B:
                                                # Combina todos os períodos selecionados para cada atleta
                                                xa, ya, xb, yb = [], [], [], []
                                                for _pm in periodos_mapa_sel:
                                                    _dA = dados_posicao_por_periodo.get(_pm, {}).get(atl_A, {})
                                                    _dB = dados_posicao_por_periodo.get(_pm, {}).get(atl_B, {})
                                                    cfg_ref = st.session_state.get(f"campo_cfg__{atleta_mapa}", cfg)
                                                    if _dA.get('xs'):
                                                        xa += list(_dA['xs']); ya += list(_dA['ys'])
                                                    elif _dA.get('lats') and cfg_ref:
                                                        _gxa, _gya = gps_para_campo_coords(_dA['lats'], _dA['lons'], cfg_ref)
                                                        xa += _gxa; ya += _gya
                                                    if _dB.get('xs'):
                                                        xb += list(_dB['xs']); yb += list(_dB['ys'])
                                                    elif _dB.get('lats') and cfg_ref:
                                                        _gxb, _gyb = gps_para_campo_coords(_dB['lats'], _dB['lons'], cfg_ref)
                                                        xb += _gxb; yb += _gyb

                                                if xa and xb:
                                                    fig_cmp = desenhar_campo_futebol_bonito(
                                                        title=f"Comparação: {atl_A} (azul) vs {atl_B} (rosa)")
                                                    _sa = max(1, len(xa)//3000)
                                                    _sb = max(1, len(xb)//3000)
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=xa[::_sa], y=ya[::_sa], mode='markers', name=atl_A,
                                                        marker=dict(size=2, color='#00E5FF', opacity=0.5)))
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=xb[::_sb], y=yb[::_sb], mode='markers', name=atl_B,
                                                        marker=dict(size=2, color='#FF4081', opacity=0.5)))
                                                    st.plotly_chart(fig_cmp, use_container_width=True)
                                                else:
                                                    st.info("Dados de posição não disponíveis para um dos atletas selecionados.")
                                            else:
                                                st.info("Selecione dois atletas diferentes para comparar.")

                                    # ══════════════════════════════════════════════
                                    # ANÁLISE — PICO DE DEMANDA (PEAK DEMAND WINDOWS)
                                    # ══════════════════════════════════════════════
                                    st.markdown("---")
                                    with st.expander("⚡ Pico de Demanda por Janela Rolante (Peak Demand Windows)", expanded=False):
                                        st.markdown(
                                            "Identifica o intervalo de tempo onde o atleta atingiu a **maior demanda "
                                            "física** da sessão — o chamado **worst-case scenario** (WCS). "
                                            "É o teto que o treino deve superar para garantir preparação adequada. "
                                            "_(Delaney et al., 2018; Martín-García et al., 2018)_"
                                        )
                                        _pdw_hz  = 10.0
                                        _pdw_vel = np.array(vel_raw_campo, dtype=float)
                                        _pdw_acc = np.array(acc_raw_campo, dtype=float)
                                        _pdw_xn  = np.array(xn, dtype=float)
                                        _pdw_yn  = np.array(yn, dtype=float)

                                        if len(_pdw_vel) < 60:
                                            st.info("Dados insuficientes para análise de janela rolante (mínimo: 60 pontos).")
                                        else:
                                            _pw_col1, _pw_col2 = st.columns([1, 2])
                                            with _pw_col1:
                                                _pdw_wins_sel = st.multiselect(
                                                    "Janelas de tempo:",
                                                    [1, 3, 5, 10], default=[1, 5],
                                                    format_func=lambda x: f"{x} min",
                                                    key="pdw_windows"
                                                )
                                            with _pw_col2:
                                                _pdw_metric = st.radio(
                                                    "Métrica de demanda:",
                                                    ["Distância (m)", "Dist. >14 km/h (m)",
                                                     "Dist. >21 km/h (m)", "Acc. >2 m/s² (n)"],
                                                    horizontal=True, key="pdw_metric"
                                                )

                                            _pdw_results, _pdw_segs = [], {}
                                            _pdw_metric_col = _pdw_metric

                                            for _wm in sorted(_pdw_wins_sel or [1, 5]):
                                                _wn = int(_wm * 60 * _pdw_hz)
                                                if _wn >= len(_pdw_vel):
                                                    continue
                                                _dt = 1.0 / _pdw_hz / 3.6  # m per sample per km/h
                                                if _pdw_metric == "Distância (m)":
                                                    _series = _pdw_vel * _dt
                                                elif _pdw_metric == "Dist. >14 km/h (m)":
                                                    _series = np.where(_pdw_vel >= 14, _pdw_vel, 0) * _dt
                                                elif _pdw_metric == "Dist. >21 km/h (m)":
                                                    _series = np.where(_pdw_vel >= 21, _pdw_vel, 0) * _dt
                                                else:
                                                    _series = (_pdw_acc >= 2.0).astype(float)

                                                _rolling = np.convolve(_series, np.ones(_wn), 'valid')
                                                _pk_i    = int(np.argmax(_rolling))
                                                _pk_val  = float(_rolling[_pk_i])
                                                _t_ini   = _pk_i / (_pdw_hz * 60)
                                                _t_fim   = (_pk_i + _wn) / (_pdw_hz * 60)

                                                _pdw_results.append({
                                                    'Janela': f"{_wm} min",
                                                    _pdw_metric_col: round(_pk_val, 1),
                                                    'Início (min)': round(_t_ini, 1),
                                                    'Fim (min)': round(_t_fim, 1),
                                                })
                                                _pdw_segs[_wm] = (_pk_i, _pk_i + _wn)

                                            if _pdw_results:
                                                _pw_clrs = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                                                _fig_pdw = go.Figure()
                                                for _ri, _rr in enumerate(_pdw_results):
                                                    _fig_pdw.add_trace(go.Bar(
                                                        x=[_rr['Janela']], y=[_rr[_pdw_metric_col]],
                                                        name=_rr['Janela'],
                                                        marker_color=_pw_clrs[_ri % 4],
                                                        text=[f"{_rr[_pdw_metric_col]:.1f}"],
                                                        textposition='outside',
                                                        textfont=dict(color='white', size=12),
                                                    ))
                                                _fig_pdw.update_layout(
                                                    title=dict(text=f"WCS — {_pdw_metric_col}",
                                                               font=dict(color='white', size=13)),
                                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                                    font=dict(color='white'), height=290,
                                                    showlegend=False,
                                                    yaxis=dict(title=_pdw_metric_col,
                                                               gridcolor='rgba(255,255,255,0.1)',
                                                               color='white'),
                                                    xaxis=dict(color='white'),
                                                    margin=dict(t=40, b=30, l=60, r=10),
                                                )
                                                st.plotly_chart(_fig_pdw, use_container_width=True)
                                                st.dataframe(pd.DataFrame(_pdw_results),
                                                             use_container_width=True, hide_index=True)

                                                # Campo da janela de pico selecionada
                                                _pdw_show = st.selectbox(
                                                    "Visualizar no campo:",
                                                    [r['Janela'] for r in _pdw_results],
                                                    key="pdw_show_win"
                                                )
                                                _sel_wm = int(_pdw_show.split()[0])
                                                if _sel_wm in _pdw_segs:
                                                    _si_p, _ei_p = _pdw_segs[_sel_wm]
                                                    _ei_p = min(_ei_p, len(_pdw_xn))
                                                    _rr_show = next(r for r in _pdw_results if r['Janela'] == _pdw_show)
                                                    _fig_pdw_c = desenhar_campo_futebol_bonito(
                                                        title=(f"WCS — {_pdw_show} "
                                                               f"(início: {_rr_show['Início (min)']} min)")
                                                    )
                                                    # ── Animação WCS — mesmo estilo do esforço selecionado ──────────
                                                    _xpk  = list(_pdw_xn[_si_p:_ei_p])
                                                    _ypk  = list(_pdw_yn[_si_p:_ei_p])
                                                    _vpk  = list(_pdw_vel[_si_p:_ei_p])
                                                    _n_fr = len(_xpk)

                                                    def _vc_wcs(v):
                                                        if v < 7:  return '#2196F3'
                                                        if v < 14: return '#4CAF50'
                                                        if v < 19: return '#FFEB3B'
                                                        if v < 24: return '#FF9800'
                                                        return '#F44336'

                                                    _step_fr = max(1, _n_fr // 120)
                                                    _fr_idxs = list(range(0, _n_fr, _step_fr))
                                                    if _fr_idxs[-1] != _n_fr - 1:
                                                        _fr_idxs.append(_n_fr - 1)

                                                    _n_base_wcs = len(_fig_pdw_c.data)

                                                    # Marcadores início / fim (estáticos) — igual ao esforço
                                                    _fig_pdw_c.add_trace(go.Scatter(
                                                        x=[_xpk[0]], y=[_ypk[0]], mode='markers+text',
                                                        marker=dict(size=13, color='#4CAF50', symbol='circle',
                                                                    line=dict(color='white', width=2)),
                                                        text=['▶'], textposition='top center',
                                                        textfont=dict(color='#4CAF50', size=11),
                                                        name='Início', showlegend=False, hoverinfo='skip'
                                                    ))
                                                    _fig_pdw_c.add_trace(go.Scatter(
                                                        x=[_xpk[-1]], y=[_ypk[-1]], mode='markers+text',
                                                        marker=dict(size=13, color='#F44336', symbol='x',
                                                                    line=dict(color='white', width=2)),
                                                        text=['■'], textposition='top center',
                                                        textfont=dict(color='#F44336', size=11),
                                                        name='Fim', showlegend=False, hoverinfo='skip'
                                                    ))
                                                    # Traço animado (linha sólida)
                                                    _fig_pdw_c.add_trace(go.Scatter(
                                                        x=[_xpk[0]], y=[_ypk[0]], mode='lines',
                                                        line=dict(color=_vc_wcs(_vpk[0] if _vpk else 0), width=4),
                                                        name='WCS', showlegend=False
                                                    ))
                                                    # Marcador de posição atual (animado)
                                                    _fig_pdw_c.add_trace(go.Scatter(
                                                        x=[_xpk[0]], y=[_ypk[0]], mode='markers',
                                                        marker=dict(size=18, color=_vc_wcs(_vpk[0] if _vpk else 0),
                                                                    symbol='circle', line=dict(color='white', width=3)),
                                                        name='Posicao', showlegend=False
                                                    ))

                                                    _idx_wcs_trail = _n_base_wcs + 2
                                                    _idx_wcs_dot   = _n_base_wcs + 3

                                                    # Gerar frames
                                                    _frames_wcs = []
                                                    for _fi in _fr_idxs:
                                                        _dur_s = (_fi / max(_n_fr - 1, 1)) * _sel_wm * 60
                                                        _dur_m = int(_dur_s // 60)
                                                        _dur_s2 = int(_dur_s % 60)
                                                        _v_fi   = float(_vpk[_fi]) if _fi < len(_vpk) else 0.0
                                                        _col_fi = _vc_wcs(_v_fi)
                                                        _frames_wcs.append(go.Frame(
                                                            data=[
                                                                go.Scatter(
                                                                    x=_xpk[:_fi + 1], y=_ypk[:_fi + 1],
                                                                    mode='lines',
                                                                    line=dict(color=_col_fi, width=4),
                                                                ),
                                                                go.Scatter(
                                                                    x=[_xpk[_fi]], y=[_ypk[_fi]],
                                                                    mode='markers',
                                                                    marker=dict(size=18, color=_col_fi, symbol='circle',
                                                                                line=dict(color='white', width=3)),
                                                                ),
                                                            ],
                                                            traces=[_idx_wcs_trail, _idx_wcs_dot],
                                                            name=str(_fi),
                                                            layout=go.Layout(title=dict(
                                                                text=(
                                                                    f'WCS {_pdw_show} | '
                                                                    f'⏱️ {_dur_m}:{_dur_s2:02d} / {int(_sel_wm)}:00'
                                                                    f'   |   💨 {_v_fi:.1f} km/h'
                                                                ),
                                                                font=dict(color='white', size=12)
                                                            ))
                                                        ))

                                                    _fig_pdw_c.frames = _frames_wcs

                                                    # Slider de frames e botões play/pause
                                                    _sliders_wcs = [{
                                                        'steps': [
                                                            {
                                                                'args': [[str(_fi)],
                                                                         {'frame': {'duration': 0, 'redraw': True},
                                                                          'mode': 'immediate'}],
                                                                'label': '',
                                                                'method': 'animate',
                                                            }
                                                            for _fi in _fr_idxs
                                                        ],
                                                        'transition': {'duration': 0},
                                                        'x': 0.05, 'len': 0.90,
                                                        'currentvalue': {
                                                            'prefix': 'Progresso: ',
                                                            'visible': False,
                                                        },
                                                        'bgcolor': '#1e3a5f',
                                                        'bordercolor': '#2196F3',
                                                        'tickcolor': 'white',
                                                        'font': {'color': 'white', 'size': 9},
                                                    }]

                                                    _fig_pdw_c.update_layout(
                                                        updatemenus=[{
                                                            'type': 'buttons',
                                                            'showactive': False,
                                                            'y': -0.08, 'x': 0.5,
                                                            'xanchor': 'center', 'yanchor': 'top',
                                                            'buttons': [
                                                                {
                                                                    'label': '▶ Play',
                                                                    'method': 'animate',
                                                                    'args': [None, {
                                                                        'frame': {'duration': 60, 'redraw': True},
                                                                        'fromcurrent': True,
                                                                        'transition': {'duration': 0},
                                                                    }],
                                                                },
                                                                {
                                                                    'label': '⏸ Pause',
                                                                    'method': 'animate',
                                                                    'args': [[None], {
                                                                        'frame': {'duration': 0, 'redraw': False},
                                                                        'mode': 'immediate',
                                                                        'transition': {'duration': 0},
                                                                    }],
                                                                },
                                                            ],
                                                            'font': {'color': 'white'},
                                                            'bgcolor': '#1e3a5f',
                                                            'bordercolor': '#2196F3',
                                                        }],
                                                        sliders=_sliders_wcs,
                                                        margin=dict(b=80),
                                                    )
                                                    st.plotly_chart(_fig_pdw_c, use_container_width=True)
                                                st.caption(
                                                    "⚡ Worst-Case Scenario: identifica a janela de maior exigência "
                                                    "para prescrição de cargas de treino acima do jogo. "
                                                    "Clique ▶ Play para ver o percurso animado."
                                                )

                                    # ══════════════════════════════════════════════
                                    # ANÁLISE — MAPA DE ALTA INTENSIDADE POSICIONAL
                                    # ══════════════════════════════════════════════
                                    st.markdown("---")
                                    with st.expander("🗺️ Mapa de Alta Intensidade Posicional (HSR Zone Map)", expanded=False):
                                        st.markdown(
                                            "Mostra **onde no campo** o atleta realiza ações de alta intensidade. "
                                            "Vai além do heatmap global — filtra apenas os momentos acima do "
                                            "limiar configurado, revelando corredores de sprint, zonas de "
                                            "pressing e rotas de recuperação."
                                        )
                                        # ── Bandas de velocidade da conta Catapult ──────
                                        _hsr_zones_acc = (
                                            st.session_state.get('velocity_zones_account')
                                            or _DEFAULT_VELOCITY_ZONES
                                        )
                                        # Monta lista de opções: apenas bandas com min >= 5 km/h
                                        _hsr_band_opts = []
                                        _hsr_band_map  = {}  # label → min_kmh
                                        for _z in _hsr_zones_acc:
                                            _z_min_kmh = round(float(_z.get('min_ms') or 0) * 3.6, 1)
                                            _z_max_ms  = float(_z.get('max_ms', 9999))
                                            _z_max_str = (
                                                '∞'
                                                if _z_max_ms >= 9000
                                                else f"{round(_z_max_ms*3.6,1)} km/h"
                                            )
                                            if _z_min_kmh >= 5:
                                                _lbl = f"{_z['name']}  ({_z_min_kmh}–{_z_max_str})"
                                                _hsr_band_opts.append(_lbl)
                                                _hsr_band_map[_lbl] = _z_min_kmh

                                        # Default: seleciona automaticamente zonas ≥ 14 km/h
                                        _hsr_default = [
                                            l for l, v in _hsr_band_map.items() if v >= 14.0
                                        ] or (_hsr_band_opts[-2:] if len(_hsr_band_opts) >= 2 else _hsr_band_opts)

                                        _hsr_sels = st.multiselect(
                                            "Bandas HSR (High Speed Running):",
                                            _hsr_band_opts,
                                            default=_hsr_default,
                                            key="hsr_zones_sel",
                                            help=(
                                                "Selecione as bandas que representam alta intensidade. "
                                                "O limiar será o **menor valor** entre as bandas escolhidas."
                                            ),
                                        )

                                        # Limiar = mínimo das bandas selecionadas
                                        if _hsr_sels:
                                            _hsr_vel_thr = min(_hsr_band_map[l] for l in _hsr_sels)
                                        else:
                                            _hsr_vel_thr = 14.0   # fallback

                                        st.caption(
                                            f"🎯 Limiar calculado: **>{_hsr_vel_thr} km/h** "
                                            f"(mínimo das bandas selecionadas)"
                                        )
                                        _hsr_vel_arr = np.array(vel_raw_campo, dtype=float)
                                        _hsr_xn = np.array(xn, dtype=float)
                                        _hsr_yn = np.array(yn, dtype=float)
                                        _hsr_mask = _hsr_vel_arr >= _hsr_vel_thr

                                        _hm1, _hm2, _hm3 = st.columns(3)
                                        _hm1.metric(f"Pontos >{_hsr_vel_thr} km/h", f"{int(_hsr_mask.sum()):,}")
                                        _hm2.metric("% tempo em HSR",
                                                    f"{float(_hsr_mask.mean())*100:.1f}%")
                                        _hm3.metric("Distância HSR (m)",
                                                    f"{float((_hsr_vel_arr[_hsr_mask]/3.6/10).sum()):.0f}")

                                        if _hsr_mask.sum() >= 10:
                                            _hsr_fl = float(st.session_state.get('venue', {}).get('length') or 105)
                                            _hsr_fw = float(st.session_state.get('venue', {}).get('width')  or 68)
                                            _n_lng, _n_lat = 5, 3
                                            _z_lng = ['Def. Prof.', 'Defensivo', 'Meio-Campo',
                                                      'Ofensivo', 'Ataque Prof.']
                                            _z_lat = ['Flanco Esq.', 'Centro', 'Flanco Dir.']

                                            _hsr_xz = _hsr_xn[_hsr_mask]
                                            _hsr_yz = _hsr_yn[_hsr_mask]
                                            _hsr_vz = _hsr_vel_arr[_hsr_mask]

                                            _zone_cnt = np.zeros((_n_lat, _n_lng))
                                            _zone_vel = np.zeros((_n_lat, _n_lng))
                                            for _px, _py, _pv in zip(_hsr_xz, _hsr_yz, _hsr_vz):
                                                _ci = min(int(_px / _hsr_fl * _n_lng), _n_lng - 1)
                                                _ri = min(int(_py / _hsr_fw * _n_lat), _n_lat - 1)
                                                _zone_cnt[_ri, _ci] += 1
                                                _zone_vel[_ri, _ci] += _pv
                                            _zone_vavg = np.where(
                                                _zone_cnt > 0, _zone_vel / _zone_cnt, 0)

                                            _hc1, _hc2 = st.columns([2, 1])
                                            with _hc1:
                                                _fig_hsr = desenhar_campo_futebol_bonito(
                                                    title=(f"HSR >{_hsr_vel_thr} km/h "
                                                           f"— {atleta_mapa}")
                                                )
                                                _sbg_h = max(1, len(_hsr_xn) // 3000)
                                                _fig_hsr.add_trace(go.Scatter(
                                                    x=_hsr_xn[::_sbg_h], y=_hsr_yn[::_sbg_h],
                                                    mode='markers',
                                                    marker=dict(size=1.5,
                                                                color='rgba(255,255,255,0.06)'),
                                                    showlegend=False, hoverinfo='skip'
                                                ))
                                                _sbg_hs = max(1, int(_hsr_mask.sum()) // 3000)
                                                _fig_hsr.add_trace(go.Scatter(
                                                    x=_hsr_xz[::_sbg_hs],
                                                    y=_hsr_yz[::_sbg_hs],
                                                    mode='markers',
                                                    marker=dict(
                                                        size=4,
                                                        color=_hsr_vz[::_sbg_hs],
                                                        colorscale=[
                                                            [0, '#66BB6A'], [0.4, '#FFA726'],
                                                            [0.7, '#EF5350'], [1, '#B71C1C']],
                                                        cmin=float(_hsr_vel_thr),
                                                        cmax=float(min(_hsr_vz.max(), 40)),
                                                        showscale=True,
                                                        colorbar=dict(
                                                            title=dict(text='km/h',
                                                                       font=dict(color='white')),
                                                            tickfont=dict(color='white'), len=0.55,
                                                        ),
                                                        opacity=0.78,
                                                    ),
                                                    showlegend=False,
                                                ))
                                                st.plotly_chart(_fig_hsr, use_container_width=True)

                                            with _hc2:
                                                _fig_zmap = go.Figure(go.Heatmap(
                                                    z=_zone_cnt.tolist(),
                                                    x=_z_lng, y=_z_lat,
                                                    colorscale='YlOrRd',
                                                    text=[[f"{int(_zone_cnt[r, c])}"
                                                           for c in range(_n_lng)]
                                                          for r in range(_n_lat)],
                                                    texttemplate='%{text}',
                                                    textfont=dict(size=10, color='black'),
                                                    showscale=False,
                                                    hovertemplate=(
                                                        '%{x} / %{y}<br>'
                                                        'Pontos HSR: %{z}<extra></extra>'
                                                    ),
                                                ))
                                                _fig_zmap.update_layout(
                                                    title=dict(text='Freq. por Zona',
                                                               font=dict(color='white', size=12)),
                                                    plot_bgcolor='#0e1117',
                                                    paper_bgcolor='#0e1117',
                                                    font=dict(color='white'), height=290,
                                                    margin=dict(t=40, b=60, l=90, r=10),
                                                    xaxis=dict(tickfont=dict(color='white', size=8),
                                                               tickangle=-30),
                                                    yaxis=dict(tickfont=dict(color='white', size=8)),
                                                )
                                                st.plotly_chart(_fig_zmap, use_container_width=True)
                                                _max_ri2, _max_ci2 = np.unravel_index(
                                                    _zone_cnt.argmax(), _zone_cnt.shape)
                                                st.caption(
                                                    f"📍 Zona mais ativa: **{_z_lat[_max_ri2]}** × "
                                                    f"**{_z_lng[_max_ci2]}** "
                                                    f"({int(_zone_cnt[_max_ri2, _max_ci2])} ações)"
                                                )
                                        else:
                                            st.info(
                                                f"Nenhum ponto com velocidade >{_hsr_vel_thr} km/h. "
                                                "Tente reduzir o limiar."
                                            )

                                    # ══════════════════════════════════════════════
                                    # ANÁLISE 5 — DISTÂNCIA ENTRE ATLETAS
                                    # ══════════════════════════════════════════════
                                    if len(atletas_mapa) >= 2:
                                        st.markdown("---")
                                        with st.expander(
                                            f"📏 Distância Entre Atletas "
                                            f"({len(atletas_mapa)} selecionados)",
                                            expanded=False
                                        ):
                                            st.markdown(
                                                "Distância Euclidiana entre os atletas selecionados "
                                                "ao longo do tempo, a partir das coordenadas de campo. "
                                                "Útil para analisar **compactação defensiva**, "
                                                "**pressing em dupla** e **cobertura de espaço**."
                                            )
                                            _cfg_dist = st.session_state.get(
                                                f"campo_cfg__{atleta_mapa}", cfg)
                                            _DCORES = ['#00E5FF','#FF4081','#FFEB3B',
                                                       '#69F0AE','#FF9800','#CE93D8']

                                            # Coleta arrays por atleta
                                            _dist_dados = {}
                                            for _atl_d in atletas_mapa:
                                                _xd, _yd, _tsd = [], [], []
                                                for _pm in periodos_mapa_sel:
                                                    _dd = dados_posicao_por_periodo.get(
                                                        _pm, {}).get(_atl_d, {})
                                                    if _dd.get('xs') and _dd.get('ys'):
                                                        _xd  += list(_dd['xs'])
                                                        _yd  += list(_dd['ys'])
                                                        _tsd += list(_dd.get('ts_pos', []))
                                                    elif _dd.get('lats') and _cfg_dist:
                                                        _gxd, _gyd = gps_para_campo_coords(
                                                            _dd['lats'], _dd['lons'], _cfg_dist)
                                                        _xd  += _gxd
                                                        _yd  += _gyd
                                                        _tsd += [0.0] * len(_gxd)
                                                if _xd:
                                                    _dist_dados[_atl_d] = {
                                                        'x': np.array(_xd, dtype=float),
                                                        'y': np.array(_yd, dtype=float),
                                                        'ts': np.array(_tsd, dtype=float)
                                                    }

                                            _pares_d = [
                                                (atletas_mapa[_ii], atletas_mapa[_jj])
                                                for _ii in range(len(atletas_mapa))
                                                for _jj in range(_ii + 1, len(atletas_mapa))
                                            ]

                                            if len(_dist_dados) >= 2:
                                                _fig_dist = go.Figure()
                                                _resumo_d = []

                                                for _pi_d, (_na, _nb) in enumerate(_pares_d):
                                                    if _na not in _dist_dados or _nb not in _dist_dados:
                                                        continue
                                                    _dA = _dist_dados[_na]
                                                    _dB = _dist_dados[_nb]
                                                    _tsA, _tsB = _dA['ts'], _dB['ts']
                                                    _has_ts_d = (
                                                        _tsA.sum() > 0 and _tsB.sum() > 0)

                                                    if _has_ts_d:
                                                        _tsA_v = _tsA[_tsA > 0]
                                                        _tsB_v = _tsB[_tsB > 0]
                                                        _tc = np.arange(
                                                            max(_tsA_v.min(), _tsB_v.min()),
                                                            min(_tsA_v.max(), _tsB_v.max()),
                                                            0.1
                                                        )
                                                        if len(_tc) < 10:
                                                            _has_ts_d = False

                                                    if _has_ts_d:
                                                        _xA_i = np.interp(_tc, _tsA_v, _dA['x'][_tsA > 0])
                                                        _yA_i = np.interp(_tc, _tsA_v, _dA['y'][_tsA > 0])
                                                        _xB_i = np.interp(_tc, _tsB_v, _dB['x'][_tsB > 0])
                                                        _yB_i = np.interp(_tc, _tsB_v, _dB['y'][_tsB > 0])
                                                        _t_ax = _tc - _tc[0]
                                                    else:
                                                        _nn = min(len(_dA['x']), len(_dB['x']))
                                                        _xA_i = _dA['x'][:_nn]
                                                        _yA_i = _dA['y'][:_nn]
                                                        _xB_i = _dB['x'][:_nn]
                                                        _yB_i = _dB['y'][:_nn]
                                                        _t_ax = np.arange(_nn) / 10.0

                                                    _dist_v = np.sqrt(
                                                        (_xA_i - _xB_i)**2 +
                                                        (_yA_i - _yB_i)**2
                                                    )
                                                    _sbgd = max(1, len(_dist_v) // 3000)
                                                    _lbl = f"{_na.split()[0]} ↔ {_nb.split()[0]}"
                                                    _cor_d = _DCORES[_pi_d % len(_DCORES)]

                                                    _fig_dist.add_trace(go.Scatter(
                                                        x=_t_ax[::_sbgd] / 60,
                                                        y=_dist_v[::_sbgd],
                                                        mode='lines',
                                                        name=_lbl,
                                                        line=dict(color=_cor_d, width=1.5),
                                                        hovertemplate=(
                                                            '%{x:.1f} min | '
                                                            '%{y:.1f} m<extra>'
                                                            + _lbl + '</extra>'
                                                        )
                                                    ))
                                                    _resumo_d.append({
                                                        'Par': _lbl,
                                                        'Dist. Média (m)':
                                                            round(float(_dist_v.mean()), 1),
                                                        'Mediana (m)':
                                                            round(float(np.median(_dist_v)), 1),
                                                        'Mín (m)':
                                                            round(float(_dist_v.min()), 1),
                                                        'Máx (m)':
                                                            round(float(_dist_v.max()), 1),
                                                        '% Tempo < 5 m':
                                                            round(100 * ((_dist_v < 5).sum()
                                                                         / len(_dist_v)), 1),
                                                        '% Tempo < 10 m':
                                                            round(100 * ((_dist_v < 10).sum()
                                                                         / len(_dist_v)), 1),
                                                    })

                                                    # Campo de proximidade (apenas 1º par)
                                                    if _pi_d == 0 and _dist_v.min() < 15:
                                                        _prox_mk = _dist_v < 10
                                                        if _prox_mk.sum() > 5:
                                                            _st_prox = _dist_dados
                                                            _xmid = ((_xA_i[_prox_mk] +
                                                                       _xB_i[_prox_mk]) / 2)
                                                            _ymid = ((_yA_i[_prox_mk] +
                                                                       _yB_i[_prox_mk]) / 2)
                                                            _fig_prox = desenhar_campo_futebol_bonito(
                                                                title=(f"Zonas de Proximidade "
                                                                       f"(< 10 m) — {_lbl}")
                                                            )
                                                            _sbgp = max(1, len(_xA_i) // 3000)
                                                            _fig_prox.add_trace(go.Scatter(
                                                                x=_xA_i[::_sbgp],
                                                                y=_yA_i[::_sbgp],
                                                                mode='markers',
                                                                marker=dict(size=1.5,
                                                                    color='rgba(0,229,255,0.18)'),
                                                                name=_na, showlegend=True
                                                            ))
                                                            _fig_prox.add_trace(go.Scatter(
                                                                x=_xB_i[::_sbgp],
                                                                y=_yB_i[::_sbgp],
                                                                mode='markers',
                                                                marker=dict(size=1.5,
                                                                    color='rgba(255,64,129,0.18)'),
                                                                name=_nb, showlegend=True
                                                            ))
                                                            _sbgpr = max(1, len(_xmid) // 1500)
                                                            _fig_prox.add_trace(go.Scatter(
                                                                x=_xmid[::_sbgpr],
                                                                y=_ymid[::_sbgpr],
                                                                mode='markers',
                                                                marker=dict(
                                                                    size=5,
                                                                    color='rgba(255,82,82,0.65)',
                                                                    symbol='circle'
                                                                ),
                                                                name='Proximidade < 10 m',
                                                                showlegend=True
                                                            ))
                                                            _fig_prox.update_layout(height=430)
                                                            _prox_fig_final = _fig_prox

                                                # Linhas de referência
                                                _fig_dist.add_hline(
                                                    y=5, line_dash='dot',
                                                    line_color='rgba(255,235,59,0.45)',
                                                    annotation_text='5 m',
                                                    annotation_font_color='#FFEB3B',
                                                    annotation_position='right'
                                                )
                                                _fig_dist.add_hline(
                                                    y=10, line_dash='dot',
                                                    line_color='rgba(255,152,0,0.45)',
                                                    annotation_text='10 m',
                                                    annotation_font_color='#FF9800',
                                                    annotation_position='right'
                                                )
                                                _fig_dist.update_layout(
                                                    title=dict(
                                                        text="Distância Entre Atletas ao Longo do Tempo",
                                                        font=dict(color='white', size=14)
                                                    ),
                                                    xaxis=dict(
                                                        title='Tempo (min)',
                                                        gridcolor='rgba(255,255,255,0.1)',
                                                        color='white'
                                                    ),
                                                    yaxis=dict(
                                                        title='Distância (m)',
                                                        gridcolor='rgba(255,255,255,0.1)',
                                                        color='white'
                                                    ),
                                                    paper_bgcolor='rgba(0,0,0,0)',
                                                    plot_bgcolor='rgba(20,20,30,0.85)',
                                                    legend=dict(
                                                        font=dict(color='white'),
                                                        bgcolor='rgba(0,0,0,0.45)'
                                                    ),
                                                    height=390
                                                )
                                                st.plotly_chart(_fig_dist,
                                                                use_container_width=True)

                                                if _resumo_d:
                                                    st.markdown("**📋 Resumo por Par**")
                                                    st.dataframe(
                                                        pd.DataFrame(_resumo_d).set_index('Par'),
                                                        use_container_width=True
                                                    )

                                                if 'prox_fig_final' in dir() and _prox_fig_final:
                                                    st.markdown("---")
                                                    st.plotly_chart(_prox_fig_final,
                                                                    use_container_width=True)
                                                    st.caption(
                                                        "🔴 Pontos vermelhos = posição média "
                                                        "entre os atletas quando distância < 10 m"
                                                    )
                                            else:
                                                st.warning(
                                                    "Dados de posição insuficientes "
                                                    "para os atletas selecionados.")
                                    else:
                                        pass  # análise de distância requer 2+ atletas

                                else:
                                    st.error("❌ Não foi possível calcular as coordenadas de campo.")
                            else:
                                st.warning(
                                    "⚠️ Sem dados de posição disponíveis para análise de campo.\n\n"
                                    "**Para ativar esta seção:**\n"
                                    "- Certifique-se de que o campo GPS está **aplicado** "
                                    "(clique ✅ Aplicar Campo no mapa acima)\n"
                                    "- Verifique se o sensor GPS estava ativo durante a sessão"
                                )

                    elif periodos_mapa_sel:
                        st.info("Selecione pelo menos um atleta para continuar.")

                else:
                    st.info("Dados de posição não disponíveis. Verifique se o sensor GPS estava ativo durante a sessão.")

                # ── FEATURE 5: Anotações Táticas ─────────────────────────────
                st.markdown("---")
                st.markdown("### 📌 Anotações Táticas")
                _ann_api = st.session_state.get('api')
                _ann_act_id = st.session_state.get('activity_id')

                if _ann_api and _ann_act_id:
                    # Carregar anotações existentes
                    if st.button("🔄 Carregar anotações", key="btn_load_ann"):
                        try:
                            _ann_raw = _ann_api.get_activity_annotations(_ann_act_id)
                            st.session_state['annotations'] = _ann_raw
                        except Exception as _ae:
                            st.error(f"Erro: {_ae}")

                    _annotations = st.session_state.get('annotations')
                    if _annotations:
                        try:
                            _ann_list = (_annotations if isinstance(_annotations, list)
                                         else _annotations.get('data', []))
                            if _ann_list:
                                st.markdown("#### Anotações existentes")
                                # Timeline visual
                                _ann_ts0 = min(
                                    (float(a.get('start_time') or 0) for a in _ann_list if a.get('start_time')),
                                    default=0
                                )
                                _fig_ann = go.Figure()
                                for _ia, _ann in enumerate(_ann_list):
                                    _t0 = float(_ann.get('start_time') or 0) - _ann_ts0
                                    _t1 = float(_ann.get('end_time', _ann.get('start_time', 0))) - _ann_ts0
                                    _ann_name = _ann.get('name', f'Anotação {_ia+1}')
                                    _fig_ann.add_shape(
                                        type='rect',
                                        x0=_t0, x1=max(_t1, _t0 + 1),
                                        y0=_ia, y1=_ia + 0.8,
                                        fillcolor='#2196F3', opacity=0.6,
                                        line_width=0,
                                    )
                                    _fig_ann.add_annotation(
                                        x=(_t0 + _t1) / 2, y=_ia + 0.4,
                                        text=_ann_name, showarrow=False,
                                        font=dict(color='white', size=10),
                                    )
                                _fig_ann.update_layout(
                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(title='Tempo (s)', gridcolor='#333'),
                                    yaxis=dict(visible=False),
                                    height=max(120, len(_ann_list) * 40 + 40),
                                    margin=dict(t=10, b=30),
                                )
                                st.plotly_chart(_fig_ann, use_container_width=True)

                                # Lista com botão de deletar
                                for _ann in _ann_list:
                                    _ann_id = _ann.get('id', '')
                                    _ann_nm = _ann.get('name', '—')
                                    _c1, _c2 = st.columns([4, 1])
                                    _c1.write(f"**{_ann_nm}** (id: {_ann_id})")
                                    if _c2.button("🗑️", key=f"del_ann_{_ann_id}"):
                                        try:
                                            _del_ok = _ann_api.delete_annotation(_ann_id)
                                            if _del_ok:
                                                st.success("Anotação removida.")
                                                st.session_state.pop('annotations', None)
                                                st.rerun()
                                        except Exception as _de:
                                            st.error(f"Erro: {_de}")
                        except Exception as _ep:
                            st.error(f"Erro ao processar anotações: {_ep}")

                    # Criar nova anotação
                    with st.expander("➕ Nova Anotação", expanded=False):
                        _ann_c1, _ann_c2 = st.columns(2)
                        with _ann_c1:
                            _ann_new_name = st.text_input("Nome:", key="ann_new_name")
                            _ann_new_start = st.number_input("Início (s Unix):", value=0, key="ann_new_start")
                        with _ann_c2:
                            _ann_new_type = st.selectbox(
                                "Tipo:", ["phase", "event", "highlight"],
                                key="ann_new_type"
                            )
                            _ann_new_end = st.number_input("Fim (s Unix):", value=0, key="ann_new_end")
                        if st.button("✅ Criar Anotação", key="btn_create_ann"):
                            if _ann_new_name:
                                try:
                                    _cr_resp = _ann_api.create_activity_annotation(
                                        _ann_act_id, _ann_new_name,
                                        _ann_new_start, _ann_new_end,
                                        _ann_new_type,
                                    )
                                    if _cr_resp:
                                        st.success("Anotação criada!")
                                        st.session_state.pop('annotations', None)
                                    else:
                                        st.warning("Endpoint de anotações pode não estar disponível nesta licença.")
                                except Exception as _cae:
                                    st.error(f"Erro: {_cae}")
                            else:
                                st.warning("Informe um nome para a anotação.")
                else:
                    st.info("Carregue os dados de uma atividade para usar anotações.")

            # ── #6: ANÁLISE DE FORMA TÁTICA DO TIME ──────────────────────────
            # (inserida no final da aba Campo & GPS)
            with abas[1]:
                st.markdown("---")
                st.markdown("## 🔷 Forma Tática Coletiva")
                st.caption(
                    "Usa as posições simultâneas de todos os atletas para calcular a "
                    "**forma geométrica do time**: centroide, largura, profundidade e "
                    "hull convexo (polígono que engloba o grupo)."
                )
                _shape_periodos = list(dados_posicao_por_periodo.keys()) if dados_posicao_por_periodo else []
                if _shape_periodos:
                    _shape_per = st.selectbox("Período:", _shape_periodos, key="shape_periodo")
                    _shape_all_atls = list(dados_posicao_por_periodo.get(_shape_per, {}).keys())
                    _shape_sel = st.multiselect(
                        "Atletas do grupo (selecione a linha/equipe):",
                        _shape_all_atls,
                        default=_shape_all_atls[:min(11, len(_shape_all_atls))],
                        key="shape_atletas",
                    )
                    _shape_janela = st.select_slider(
                        "Janela temporal:", options=[30, 60, 120, 300, 600, 0],
                        format_func=lambda v: "Completo" if v == 0 else f"{v//60} min" if v >= 60 else f"{v} s",
                        value=0, key="shape_janela",
                    )

                    if len(_shape_sel) >= 3:
                        from scipy.spatial import ConvexHull as _CVH

                        # Coleta posições
                        _shape_xs_dict = {}
                        _shape_ys_dict = {}
                        for _sa in _shape_sel:
                            _sd = dados_posicao_por_periodo.get(_shape_per, {}).get(_sa, {})
                            _sxs = list(_sd.get('xs', []))
                            _sys = list(_sd.get('ys', []))
                            if _sxs:
                                _shape_xs_dict[_sa] = np.array(_sxs)
                                _shape_ys_dict[_sa] = np.array(_sys)

                        if len(_shape_xs_dict) >= 3:
                            # Janela: usar últimos N pontos (ou todos)
                            if _shape_janela > 0:
                                _n_pts = int(_shape_janela * 10)  # 10 Hz
                                _xs_trim = {a: v[-_n_pts:] for a, v in _shape_xs_dict.items()}
                                _ys_trim = {a: v[-_n_pts:] for a, v in _shape_ys_dict.items()}
                            else:
                                _xs_trim = _shape_xs_dict
                                _ys_trim = _shape_ys_dict

                            # Posição média de cada atleta
                            _mean_xs = np.array([float(np.mean(v)) for v in _xs_trim.values()])
                            _mean_ys = np.array([float(np.mean(v)) for v in _ys_trim.values()])
                            _atl_names = list(_xs_trim.keys())

                            # Métricas coletivas
                            _centroid_x = float(np.mean(_mean_xs))
                            _centroid_y = float(np.mean(_mean_ys))
                            _largura  = float(np.max(_mean_xs) - np.min(_mean_xs))
                            _profund  = float(np.max(_mean_ys) - np.min(_mean_ys))
                            _compac   = float(np.mean([
                                np.sqrt((x - _centroid_x)**2 + (y - _centroid_y)**2)
                                for x, y in zip(_mean_xs, _mean_ys)
                            ]))

                            # KPIs
                            _sh1, _sh2, _sh3, _sh4 = st.columns(4)
                            _sh1.metric("Centroide X", f"{_centroid_x:.1f} m")
                            _sh2.metric("Centroide Y", f"{_centroid_y:.1f} m")
                            _sh3.metric("Largura efetiva", f"{_largura:.1f} m")
                            _sh4.metric("Profundidade efetiva", f"{_profund:.1f} m")
                            st.metric("Compacidade (dist. média ao centroide)", f"{_compac:.1f} m",
                                      help="Quanto menor, mais compacto o time.")

                            # Campo com hull convexo
                            _shape_cfg = None
                            for _sk, _sv in st.session_state.items():
                                if isinstance(_sk, str) and _sk.startswith('campo_cfg__') and isinstance(_sv, dict) and _sv.get('fl'):
                                    _shape_cfg = _sv; break
                            _fl = float(_shape_cfg.get('fl', 105)) if _shape_cfg else 105.0
                            _fw = float(_shape_cfg.get('fw', 68))  if _shape_cfg else 68.0

                            _fig_shape = desenhar_campo_futebol_bonito(field_length=_fl, field_width=_fw, margin=3)

                            # Hull convexo
                            _pts_hull = np.column_stack([_mean_xs, _mean_ys])
                            if len(_pts_hull) >= 3:
                                try:
                                    _hull = _CVH(_pts_hull)
                                    _hv = _pts_hull[_hull.vertices]
                                    _hv = np.vstack([_hv, _hv[0]])  # fechar
                                    _fig_shape.add_trace(go.Scatter(
                                        x=_hv[:,0], y=_hv[:,1],
                                        fill='toself', fillcolor='rgba(255,215,0,0.12)',
                                        line=dict(color='#FFD700', width=2, dash='dash'),
                                        name='Hull Convexo', hoverinfo='skip',
                                    ))
                                except Exception:
                                    pass

                            # Centroide
                            _fig_shape.add_trace(go.Scatter(
                                x=[_centroid_x], y=[_centroid_y],
                                mode='markers+text', text=['⊕'], textfont=dict(size=18, color='#FFD700'),
                                marker=dict(size=16, color='#FFD700', symbol='circle',
                                            line=dict(color='white', width=2)),
                                name='Centroide', hovertemplate=f'Centroide: ({_centroid_x:.1f}, {_centroid_y:.1f})<extra></extra>',
                            ))

                            # Posição média de cada atleta
                            for _sa, _mx, _my in zip(_atl_names, _mean_xs, _mean_ys):
                                _fig_shape.add_trace(go.Scatter(
                                    x=[_mx], y=[_my],
                                    mode='markers+text',
                                    text=[_sa.split()[-1][:8]],
                                    textposition='top center',
                                    textfont=dict(color='white', size=9),
                                    marker=dict(size=14, color=cor_atleta(_sa),
                                                line=dict(color='white', width=1.5)),
                                    name=_sa,
                                    hovertemplate=f'{_sa}<br>X: {_mx:.1f}m  Y: {_my:.1f}m<extra></extra>',
                                ))

                            st.plotly_chart(_fig_shape, use_container_width=True)

                            # Evolução do centroide e largura ao longo do tempo
                            with st.expander("📈 Evolução temporal da forma", expanded=False):
                                _evo_window = 300  # 30 s de dados (300 pontos a 10 Hz)
                                _evo_step   = 100
                                _evo_cx, _evo_cy, _evo_larg, _evo_prof, _evo_t = [], [], [], [], []
                                _min_len = min(len(v) for v in _xs_trim.values())
                                _i_evo = 0
                                while _i_evo + _evo_window <= _min_len:
                                    _cx = float(np.mean([v[_i_evo:_i_evo+_evo_window].mean() for v in _xs_trim.values()]))
                                    _cy = float(np.mean([v[_i_evo:_i_evo+_evo_window].mean() for v in _ys_trim.values()]))
                                    _slices_x = np.array([v[_i_evo:_i_evo+_evo_window].mean() for v in _xs_trim.values()])
                                    _slices_y = np.array([v[_i_evo:_i_evo+_evo_window].mean() for v in _ys_trim.values()])
                                    _evo_cx.append(_cx); _evo_cy.append(_cy)
                                    _evo_larg.append(float(_slices_x.max()-_slices_x.min()))
                                    _evo_prof.append(float(_slices_y.max()-_slices_y.min()))
                                    _evo_t.append(_i_evo / 10 / 60)  # minutos
                                    _i_evo += _evo_step

                                if _evo_t:
                                    _fig_evo = make_subplots(rows=2, cols=1,
                                        subplot_titles=['Largura Efetiva ao Longo do Jogo',
                                                        'Profundidade Efetiva ao Longo do Jogo'],
                                        shared_xaxes=True)
                                    _fig_evo.add_trace(go.Scatter(x=_evo_t, y=_evo_larg,
                                        mode='lines', line=dict(color='#2196F3', width=2),
                                        name='Largura'), row=1, col=1)
                                    _fig_evo.add_trace(go.Scatter(x=_evo_t, y=_evo_prof,
                                        mode='lines', line=dict(color='#4CAF50', width=2),
                                        name='Profundidade'), row=2, col=1)
                                    _fig_evo.update_layout(
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'), height=380,
                                        xaxis2=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='m', gridcolor='#333'),
                                        yaxis2=dict(title='m', gridcolor='#333'),
                                        margin=dict(t=30,b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_evo, use_container_width=True)
                    else:
                        st.info("Selecione pelo menos 3 atletas para calcular a forma tática.")
                else:
                    st.info("Carregue dados de posição para usar a análise de forma tática.")

            # ==================== ABA 3: ESFORÇOS AO LONGO DO TEMPO ====================
            with abas[2]:
                st.subheader("⏱️ Esforços ao Longo do Tempo")
                
                if dados_sensor_por_atleta_por_periodo:
                    _ESF_TODOS = "🔀 Todos os períodos (combinado)"
                    _esf_opcoes = [_ESF_TODOS] + list(dados_sensor_por_atleta_por_periodo.keys())
                    periodo_escolhido = st.selectbox("Selecione o período:", _esf_opcoes, key="periodo_esforcos")
                    _esf_modo_todos = (periodo_escolhido == _ESF_TODOS)

                    if _esf_modo_todos:
                        _esf_ats_set = set()
                        for _pv in dados_sensor_por_atleta_por_periodo.values():
                            _esf_ats_set.update(_pv.keys())
                        _esf_atletas = sorted(_esf_ats_set)
                    else:
                        _esf_atletas = list(dados_sensor_por_atleta_por_periodo.get(periodo_escolhido, {}).keys())

                    if _esf_atletas:
                        atleta_escolhido = st.selectbox("Selecione o atleta:", _esf_atletas, key="atleta_esforcos")
                        if _esf_modo_todos:
                            sensor_points = []
                            for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                sensor_points += _pv2.get(atleta_escolhido, [])
                            st.caption(
                                f"📊 Combinando **{len(dados_sensor_por_atleta_por_periodo)} períodos** "
                                f"→ {len(sensor_points):,} amostras para **{atleta_escolhido}**."
                            )
                        else:
                            sensor_points = dados_sensor_por_atleta_por_periodo[periodo_escolhido].get(atleta_escolhido, [])
                        
                        col_config1, col_config2 = st.columns(2)
                        with col_config1:
                            mostrar_tendencia = st.checkbox("Mostrar linha de tendência", value=True)
                            window_size = st.slider("Janela de suavização:", 5, 101, 31, step=2)
                        with col_config2:
                            usar_filtro = st.checkbox("Filtrar por intensidade", value=False)
                            if usar_filtro:
                                intensidade_min = st.slider("Intensidade mínima (km/h):", 0.0, 30.0, 5.0, 0.5)
                            else:
                                intensidade_min = None
                        
                        _dur_s_aba3 = get_min_dur_s()
                        st.caption(
                            f"⚙️ Duração mínima de acc/dec: **{_dur_s_aba3:.1f} s** "
                            f"({max(1, round(_dur_s_aba3 * _SENSOR_HZ))} frames) — "
                            "ajuste na sidebar."
                        )
                        st.markdown("### 🏃‍♂️ Velocidade ao Longo do Tempo")
                        fig_vel = criar_grafico_velocidade_tempo(sensor_points, atleta_escolhido, window_size, mostrar_tendencia, intensidade_min)
                        if fig_vel:
                            st.plotly_chart(fig_vel, use_container_width=True)
                        
                        st.markdown("### 🔄 Aceleração ao Longo do Tempo")
                        fig_acc = criar_grafico_aceleracao_tempo(sensor_points, atleta_escolhido, window_size, mostrar_tendencia, intensidade_min if usar_filtro else None)
                        if fig_acc:
                            st.plotly_chart(fig_acc, use_container_width=True)
                        
                        st.markdown("## 📋 Tabela de Esforços")
                        tipo_esforco = st.selectbox(
                            "Tipo de esforço:",
                            ["Velocidade", "Aceleração", "Frequência Cardíaca", "Saltos", "Simetria de Passada"],
                            key="tipo_esforco_sel",
                        )

                        # Recuperar esforços do tipo escolhido
                        efforts_df = pd.DataFrame()
                        _hist_vmax_ms = st.session_state.get('hist_vmax', {}).get(atleta_escolhido)
                        if _esf_modo_todos:
                            _esf_raw = []
                            for _pk in dados_sensor_por_atleta_por_periodo.keys():
                                if tipo_esforco == "Velocidade":
                                    _esf_raw += dados_efforts_vel_por_periodo.get(_pk, {}).get(atleta_escolhido, [])
                                elif tipo_esforco == "Aceleração":
                                    _esf_raw += dados_efforts_acc_por_periodo.get(_pk, {}).get(atleta_escolhido, [])
                                elif tipo_esforco == "Frequência Cardíaca":
                                    _esf_raw += dados_hr_efforts_por_periodo.get(_pk, {}).get(atleta_escolhido, [])
                                elif tipo_esforco == "Saltos":
                                    _esf_raw += dados_jump_efforts_por_periodo.get(_pk, {}).get(atleta_escolhido, [])
                                else:
                                    _esf_raw += dados_step_efforts_por_periodo.get(_pk, {}).get(atleta_escolhido, [])
                            if _esf_raw:
                                if tipo_esforco == "Velocidade":
                                    efforts_df = processar_efforts_velocidade(_esf_raw, _hist_vmax_ms)
                                elif tipo_esforco == "Aceleração":
                                    efforts_df = processar_efforts_aceleracao(_esf_raw)
                                else:
                                    try:
                                        efforts_df = pd.DataFrame(_esf_raw)
                                    except Exception:
                                        pass
                        else:
                            if tipo_esforco == "Velocidade":
                                _rd = dados_efforts_vel_por_periodo.get(periodo_escolhido, {})
                                if atleta_escolhido in _rd:
                                    efforts_df = processar_efforts_velocidade(_rd[atleta_escolhido], _hist_vmax_ms)
                            elif tipo_esforco == "Aceleração":
                                _rd = dados_efforts_acc_por_periodo.get(periodo_escolhido, {})
                                if atleta_escolhido in _rd:
                                    efforts_df = processar_efforts_aceleracao(_rd[atleta_escolhido])
                            elif tipo_esforco == "Frequência Cardíaca":
                                _rd = dados_hr_efforts_por_periodo.get(periodo_escolhido, {})
                                if atleta_escolhido in _rd:
                                    try:
                                        efforts_df = pd.DataFrame(_rd[atleta_escolhido])
                                    except Exception:
                                        pass
                            elif tipo_esforco == "Saltos":
                                _rd = dados_jump_efforts_por_periodo.get(periodo_escolhido, {})
                                if atleta_escolhido in _rd:
                                    try:
                                        _jdf = pd.DataFrame(_rd[atleta_escolhido])
                                        efforts_df = _jdf
                                        # Mostrar distribuição de altura de salto
                                        if 'height' in _jdf.columns and not _jdf.empty:
                                            st.markdown("#### 📊 Distribuição de Altura de Salto")
                                            _fig_jump = px.histogram(
                                                _jdf, x='height', nbins=20,
                                                title="Distribuição de Altura dos Saltos",
                                                labels={'height': 'Altura (m)'},
                                            )
                                            _fig_jump.update_layout(
                                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                                font=dict(color='white'),
                                            )
                                            st.plotly_chart(_fig_jump, use_container_width=True)
                                    except Exception:
                                        pass
                            else:  # Simetria de Passada
                                _rd = dados_step_efforts_por_periodo.get(periodo_escolhido, {})
                                if atleta_escolhido in _rd:
                                    try:
                                        _sdf = pd.DataFrame(_rd[atleta_escolhido])
                                        efforts_df = _sdf
                                        # Mostrar assimetria esquerda/direita
                                        if not _sdf.empty:
                                            _left_col = next((c for c in _sdf.columns if 'left' in c.lower()), None)
                                            _right_col = next((c for c in _sdf.columns if 'right' in c.lower()), None)
                                            if _left_col and _right_col:
                                                st.markdown("#### ↔️ Assimetria Esquerda/Direita")
                                                _sdf['asymmetry'] = (_sdf[_left_col] - _sdf[_right_col]).abs()
                                                _fig_asym = px.line(
                                                    _sdf.reset_index(), x='index',
                                                    y=['asymmetry'],
                                                    title="Assimetria de Passada ao Longo do Tempo",
                                                )
                                                _fig_asym.update_layout(
                                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                                    font=dict(color='white'),
                                                )
                                                st.plotly_chart(_fig_asym, use_container_width=True)
                                    except Exception:
                                        pass
                        
                        if not efforts_df.empty:
                            if 'Banda' in efforts_df.columns:
                                bandas = sorted(efforts_df['Banda'].dropna().unique())
                                if bandas:
                                    bandas_sel = st.multiselect("Filtrar por bandas:", bandas, default=bandas)
                                    if bandas_sel:
                                        efforts_df = efforts_df[efforts_df['Banda'].isin(bandas_sel)]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1: st.metric("Total", len(efforts_df))
                            with col2: st.metric("Duração Total (s)", round(efforts_df['Duração (s)'].sum(), 1))
                            with col3:
                                if 'Distância (m)' in efforts_df.columns:
                                    st.metric("Distância Total (m)", round(efforts_df['Distância (m)'].sum(), 1))
                            with col4: st.metric("Média % Máximo", round(efforts_df['% do Máximo'].mean(), 1))
                            
                            st.dataframe(efforts_df, use_container_width=True, height=400)
                            csv_eff = efforts_df.to_csv(index=False)
                            st.download_button("📥 Exportar Esforços", csv_eff, f"esforcos_{atleta_escolhido}.csv")

                            # ── Item 17: Distribuição por Banda + Densidade Temporal ──
                            st.markdown("---")
                            _eff_c1, _eff_c2 = st.columns(2)
                            with _eff_c1:
                                st.markdown("#### 📊 Distribuição por Banda")
                                if 'Banda' in efforts_df.columns and efforts_df['Banda'].notna().any():
                                    _banda_grp = efforts_df.groupby('Banda').agg(
                                        Contagem=('Banda', 'count'),
                                        Distancia_Total=('Distância (m)', 'sum'),
                                        Duracao_Total=('Duração (s)', 'sum'),
                                    ).reset_index()
                                    _BAND_COLORS = {
                                        'B1': '#4CAF50', 'B2': '#8BC34A', 'B3': '#FFEB3B',
                                        'B4': '#FF9800', 'B5': '#F44336', 'B6': '#9C27B0', 'B7': '#E91E63',
                                    }
                                    _bc = [_BAND_COLORS.get(str(b).strip(), '#607D8B') for b in _banda_grp['Banda']]
                                    _fig_band = go.Figure(go.Bar(
                                        x=_banda_grp['Banda'], y=_banda_grp['Contagem'],
                                        marker_color=_bc,
                                        text=_banda_grp['Contagem'], textposition='outside',
                                        customdata=np.stack([
                                            _banda_grp['Distancia_Total'].round(1),
                                            _banda_grp['Duracao_Total'].round(1),
                                        ], axis=-1),
                                        hovertemplate='Banda %{x}<br>Esforços: %{y}<br>Dist: %{customdata[0]} m<br>Dur: %{customdata[1]} s<extra></extra>',
                                    ))
                                    _fig_band.update_layout(
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Banda de velocidade', gridcolor='#333'),
                                        yaxis=dict(title='Número de esforços', gridcolor='#333'),
                                        height=300, margin=dict(t=15, b=10, l=10, r=10),
                                        showlegend=False,
                                    )
                                    st.plotly_chart(_fig_band, use_container_width=True)
                                else:
                                    st.info("Dados de banda não disponíveis para este atleta.")

                            with _eff_c2:
                                st.markdown("#### 📈 Densidade de Esforços no Tempo")
                                if '_start_ts' in efforts_df.columns and efforts_df['_start_ts'].sum() > 0:
                                    _ts_eff = efforts_df['_start_ts'].dropna().sort_values()
                                    _t0_eff = _ts_eff.min()
                                    _ts_min = (_ts_eff - _t0_eff) / 60  # minutos
                                    # Histograma de densidade: janelas de 5 min
                                    _dur_tot_min = _ts_min.max()
                                    _n_bins = max(6, int(_dur_tot_min / 5))
                                    _counts, _edges = np.histogram(_ts_min, bins=_n_bins)
                                    _bin_centers = (_edges[:-1] + _edges[1:]) / 2
                                    _fig_dens = go.Figure(go.Bar(
                                        x=_bin_centers, y=_counts,
                                        marker_color='#2196F3',
                                        opacity=0.8,
                                        hovertemplate='%{x:.1f} min: %{y} esforços<extra></extra>',
                                    ))
                                    _fig_dens.update_layout(
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='Esforços / janela', gridcolor='#333'),
                                        height=300, margin=dict(t=15, b=10, l=10, r=10),
                                        showlegend=False,
                                        bargap=0.1,
                                    )
                                    st.plotly_chart(_fig_dens, use_container_width=True)

                                    # Duração + distância por esforço — box plot
                                    with st.expander("📦 Distribuição de Duração e Distância por Esforço"):
                                        _box_c1, _box_c2 = st.columns(2)
                                        with _box_c1:
                                            _fig_box_dur = go.Figure(go.Box(
                                                y=efforts_df['Duração (s)'],
                                                name='Duração (s)',
                                                marker_color='#FF9800',
                                                boxpoints='outliers',
                                            ))
                                            _fig_box_dur.update_layout(
                                                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                                font=dict(color='white'),
                                                yaxis=dict(title='Duração (s)', gridcolor='#333'),
                                                height=280, margin=dict(t=10, b=10),
                                            )
                                            st.plotly_chart(_fig_box_dur, use_container_width=True)
                                        with _box_c2:
                                            if 'Distância (m)' in efforts_df.columns:
                                                _fig_box_dist = go.Figure(go.Box(
                                                    y=efforts_df['Distância (m)'],
                                                    name='Distância (m)',
                                                    marker_color='#4CAF50',
                                                    boxpoints='outliers',
                                                ))
                                                _fig_box_dist.update_layout(
                                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                                    font=dict(color='white'),
                                                    yaxis=dict(title='Distância (m)', gridcolor='#333'),
                                                    height=280, margin=dict(t=10, b=10),
                                                )
                                                st.plotly_chart(_fig_box_dist, use_container_width=True)
                                else:
                                    st.info("Timestamps de esforços não disponíveis para análise de densidade.")

                        # ── Metabolic Power chart (FEATURE 3) ───────────────────
                        st.markdown("---")
                        _mp_vals_esf = [
                            float(p['mp'])
                            for p in sensor_points
                            if p.get('mp') and float(p.get('mp') or 0) > 0
                        ]
                        if _mp_vals_esf:
                            st.markdown("### ⚡ Potência Metabólica ao Longo do Tempo")
                            _ts0_mp = float(sensor_points[0].get('ts') or 0)
                            _mp_ts = [
                                (float(p.get('ts') or 0) - _ts0_mp)
                                for p in sensor_points
                                if p.get('mp') and float(p.get('mp') or 0) > 0
                            ]
                            _fig_mp = go.Figure()
                            _fig_mp.add_trace(go.Scatter(
                                x=_mp_ts, y=_mp_vals_esf,
                                mode='lines',
                                name='MP (W/kg)',
                                line=dict(color='#FF9800', width=1.5),
                                hovertemplate='%{x:.0f}s — %{y:.1f} W/kg<extra></extra>',
                            ))
                            _fig_mp.add_hline(y=20, line=dict(color='#F44336', dash='dash'),
                                              annotation_text='20 W/kg', annotation_font=dict(color='#F44336', size=9))
                            _fig_mp.add_hline(y=25, line=dict(color='#9C27B0', dash='dash'),
                                              annotation_text='25 W/kg', annotation_font=dict(color='#9C27B0', size=9))
                            _fig_mp.update_layout(
                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                font=dict(color='white'),
                                xaxis=dict(title='Tempo (s)', gridcolor='#333'),
                                yaxis=dict(title='MP (W/kg)', gridcolor='#333'),
                                height=320,
                            )
                            st.plotly_chart(_fig_mp, use_container_width=True)
                            _mp_above20 = sum(1 for v in _mp_vals_esf if v > 20) / max(1, len(_mp_vals_esf)) * 100
                            _mp_above25_s = sum(1 for v in _mp_vals_esf if v > 25) * 0.1
                            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                            _mc1.metric("MP Médio (W/kg)", f"{float(np.mean(_mp_vals_esf)):.1f}")
                            _mc2.metric("MP Máx (W/kg)", f"{float(np.max(_mp_vals_esf)):.1f}")
                            _mc3.metric("MP > 20 W/kg (%)", f"{_mp_above20:.1f}%")
                            _mc4.metric("Tempo > 25 W/kg (s)", f"{_mp_above25_s:.0f}s")

                else:
                    st.info("Dados de sensor não disponíveis")

                # ── FEATURE 10: Exportação em Lote (Async Export) ────────────
                st.markdown("---")
                with st.expander("📦 Exportação em Lote (sessões longas)", expanded=False):
                    st.info(
                        "Para sessões > 90 min, use exportação assíncrona para evitar timeout. "
                        "A API Catapult processará o arquivo em background."
                    )
                    _exp_api = st.session_state.get('api')
                    _exp_act_id = st.session_state.get('activity_id')

                    if _exp_api and _exp_act_id:
                        if st.button("📤 Solicitar exportação", key="btn_submit_export"):
                            try:
                                _exp_payload = {
                                    "activity_id": _exp_act_id,
                                    "format": "csv",
                                    "parameters": ["ts", "v", "a", "hr", "pl", "mp"],
                                }
                                _exp_resp = _exp_api.submit_export(_exp_payload)
                                if _exp_resp:
                                    _job_id = (
                                        _exp_resp.get('id') or
                                        _exp_resp.get('job_id') or
                                        str(_exp_resp)
                                    )
                                    st.session_state['export_job_id'] = _job_id
                                    st.success(f"Job submetido! ID: {_job_id}")
                                else:
                                    st.warning("Nenhuma resposta do servidor de exportação.")
                            except Exception as _ee:
                                st.error(f"Erro: {_ee}")

                        _job_id = st.session_state.get('export_job_id')
                        if _job_id:
                            st.caption(f"Job ID: `{_job_id}`")
                            if st.button("🔄 Verificar status", key="btn_check_export"):
                                try:
                                    _status_resp = _exp_api.get_export_status(_job_id)
                                    if _status_resp:
                                        _status = (_status_resp.get('status') or
                                                   _status_resp.get('state', 'DESCONHECIDO'))
                                        st.info(f"Status: **{_status}**")
                                        if str(_status).upper() in ('COMPLETE', 'COMPLETED', 'DONE'):
                                            if st.button("⬇️ Download", key="btn_dl_export"):
                                                try:
                                                    _dl = _exp_api.download_export(_job_id)
                                                    if _dl:
                                                        st.download_button(
                                                            "📥 Baixar arquivo exportado",
                                                            _dl,
                                                            file_name=f"export_{_job_id}.csv",
                                                        )
                                                except Exception as _de:
                                                    st.error(f"Erro no download: {_de}")
                                    else:
                                        st.warning("Sem resposta de status.")
                                except Exception as _se:
                                    st.error(f"Erro ao verificar status: {_se}")
                    else:
                        st.info("Carregue dados de uma atividade primeiro.")

                # ── FEATURE 8: PDF Export ────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📄 Exportar Relatório PDF")
                st.caption(
                    "Gera um relatório A3 com heatmap, perfil de velocidade, aceleração e métricas biométricas."
                )
                if dados_sensor_por_atleta_por_periodo:
                    _pdf_per_opts = list(dados_sensor_por_atleta_por_periodo.keys())
                    _col_pdf1, _col_pdf2 = st.columns(2)
                    with _col_pdf1:
                        _pdf_per = st.selectbox("Período:", _pdf_per_opts, key="pdf_periodo")
                    with _col_pdf2:
                        _pdf_ats = list(dados_sensor_por_atleta_por_periodo.get(_pdf_per, {}).keys())
                        _pdf_atl = st.selectbox("Atleta:", _pdf_ats, key="pdf_atleta") if _pdf_ats else None

                    if _pdf_atl and st.button("📄 Gerar Relatório PDF", type="primary", key="btn_pdf"):
                        with st.spinner("Gerando relatório PDF..."):
                            _pdf_sp = dados_sensor_por_atleta_por_periodo[_pdf_per].get(_pdf_atl, [])
                            _pdf_dp = dados_posicao_por_periodo.get(_pdf_per, {}).get(_pdf_atl, {})
                            _pdf_met = {}
                            for _r in resultados_por_periodo.get(_pdf_per, []):
                                if _r.get('Atleta') == _pdf_atl:
                                    _pdf_met = _r
                                    break
                            try:
                                _pdf_bytes = gerar_pdf_relatorio(
                                    atleta_nome=_pdf_atl,
                                    periodo_nome=_pdf_per,
                                    metricas=_pdf_met,
                                    sensor_points=_pdf_sp,
                                    dados_pos=_pdf_dp,
                                )
                                st.download_button(
                                    label="⬇️ Baixar PDF",
                                    data=_pdf_bytes,
                                    file_name=f"relatorio_{_pdf_atl.replace(' ','_')}_{_pdf_per}.pdf",
                                    mime="application/pdf",
                                    key="dl_pdf"
                                )
                                st.success("✅ PDF gerado com sucesso! Clique em **⬇️ Baixar PDF** acima.")
                            except Exception as _e:
                                st.error(f"Erro ao gerar PDF: {_e}")

            # ==================== ABA 4: JANELAS TEMPORAIS MÓVEIS ====================
            with abas[3]:
                st.subheader("📊 Análise de Intensidade - Janelas Temporais Móveis")
                st.markdown("""
                Esta análise calcula a **média da métrica** em janelas discretas de tempo. 
                Cada janela é dividida em blocos de **10 segundos**, e o valor final é a média desses blocos.
                """)
                
                if dados_sensor_por_atleta_por_periodo:
                    _JAN_TODOS = "🔀 Todos os períodos (combinado)"
                    _jan_opcoes = [_JAN_TODOS] + list(dados_sensor_por_atleta_por_periodo.keys())
                    periodo_janela = st.selectbox("Selecione o período:", _jan_opcoes, key="periodo_janela")
                    _jan_modo_todos = (periodo_janela == _JAN_TODOS)

                    if _jan_modo_todos:
                        _jan_ats_set = set()
                        for _pv in dados_sensor_por_atleta_por_periodo.values():
                            _jan_ats_set.update(_pv.keys())
                        _jan_atletas = sorted(_jan_ats_set)
                    else:
                        _jan_atletas = list(dados_sensor_por_atleta_por_periodo.get(periodo_janela, {}).keys())

                    if _jan_atletas:
                        atleta_janela = st.selectbox("Selecione o atleta:", _jan_atletas, key="atleta_janela")
                        if _jan_modo_todos:
                            sensor_points = []
                            for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                sensor_points += _pv2.get(atleta_janela, [])
                            st.caption(
                                f"📊 Combinando **{len(dados_sensor_por_atleta_por_periodo)} períodos** "
                                f"→ {len(sensor_points):,} amostras para **{atleta_janela}**."
                            )
                        else:
                            sensor_points = dados_sensor_por_atleta_por_periodo[periodo_janela].get(atleta_janela, [])
                        
                        if sensor_points:
                            _dur_s_aba4 = get_min_dur_s()
                            st.caption(
                                f"⚙️ Duração mínima de acc/dec: **{_dur_s_aba4:.1f} s** "
                                f"({max(1, round(_dur_s_aba4 * _SENSOR_HZ))} frames) — "
                                "ajuste na sidebar."
                            )
                            bandas_vel = [3, 4, 5, 6, 7, 8]
                            bandas_acc = [1, 2, 3]

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                window_minutes = st.slider(
                                    "Tamanho da janela temporal (minutos):",
                                    min_value=0.5, max_value=10.0, value=1.0, step=0.5
                                )
                            with col2:
                                tipo_metrica = st.selectbox(
                                    "Selecione a métrica:",
                                    ['Distância', 'PlayerLoad', 'Velocidade', 'Aceleração']
                                )
                            with col3:
                                if tipo_metrica == 'Velocidade':
                                    bandas_vel = st.multiselect(
                                        "Bandas de Velocidade:",
                                        options=[1, 2, 3, 4, 5, 6, 7, 8],
                                        default=[3, 4, 5, 6, 7, 8]
                                    )
                                elif tipo_metrica == 'Aceleração':
                                    bandas_acc = st.multiselect(
                                        "Bandas de Aceleração:",
                                        options=[-3, -2, -1, 0, 1, 2, 3],
                                        default=[1, 2, 3]
                                    )

                            with st.spinner("Calculando janelas temporais..."):
                                if tipo_metrica == 'Distância':
                                    tempos_janela, valores_janela = calcular_distancia_janelas_discretas_10s(sensor_points, window_minutes)
                                    exibir_resultados_janela(tempos_janela, valores_janela, "Distância", atleta_janela, window_minutes, "m/min")

                                elif tipo_metrica == 'PlayerLoad':
                                    tempos_janela, valores_janela = calcular_janelas_discretas_10s(sensor_points, window_minutes, 'pl', None)
                                    exibir_resultados_janela(tempos_janela, valores_janela, "PlayerLoad", atleta_janela, window_minutes, "PL/min")

                                elif tipo_metrica == 'Velocidade':
                                    band_filter = {'velocity_bands': bandas_vel}
                                    tempos_janela, valores_janela = calcular_janelas_discretas_10s(sensor_points, window_minutes, 'v', band_filter)
                                    exibir_resultados_janela(tempos_janela, valores_janela, "Velocidade", atleta_janela, window_minutes, "km/h")

                                elif tipo_metrica == 'Aceleração':
                                    band_filter = {'acceleration_bands': bandas_acc}
                                    tempos_janela, valores_janela = calcular_janelas_discretas_10s(sensor_points, window_minutes, 'a', band_filter)
                                    exibir_resultados_janela(tempos_janela, valores_janela, "Aceleração", atleta_janela, window_minutes, "m/s²")

                            st.markdown(REFERENCIAS["janelas"])
                        else:
                            st.info("Dados de sensor não disponíveis")
                else:
                    st.info("Selecione um atleta para análise")

            # ==================== ABA 5: CARGA NEUROMUSCULAR ====================
            with abas[4]:
                st.subheader("💪 Análise de Carga Neuromuscular")
                st.markdown("""
                Esforços de **aceleração** e **desaceleração** intensa são indicadores críticos de
                carga neuromuscular/excêntrica — desacelerações superiores a 2 m/s² geram impacto
                muscular frequentemente maior que sprints. Esta aba quantifica esses esforços por
                minuto e acumula a carga ao longo da sessão.
                """)

                if dados_sensor_por_atleta_por_periodo:
                    _NM_TODOS = "🔀 Todos os períodos (combinado)"
                    _nm_opcoes_per = [_NM_TODOS] + list(dados_sensor_por_atleta_por_periodo.keys())
                    _nm_per = st.selectbox("Período:", _nm_opcoes_per, key="nm_periodo")

                    # Monta lista de atletas disponíveis
                    if _nm_per == _NM_TODOS:
                        _nm_ats_set = set()
                        for _pv in dados_sensor_por_atleta_por_periodo.values():
                            _nm_ats_set.update(_pv.keys())
                        _nm_ats = sorted(_nm_ats_set)
                    else:
                        _nm_ats = list(dados_sensor_por_atleta_por_periodo.get(_nm_per, {}).keys())

                    if _nm_ats:
                        _nm_atl = st.selectbox("Atleta:", _nm_ats, key="nm_atleta")
                        _nm_lim = st.slider("Limiar de intensidade (m/s²):", 1.0, 4.0, 2.0, 0.5,
                                            key="nm_limiar",
                                            help="Acelerações/desacelerações acima deste valor são classificadas como intensas.")

                        _nm_dur_s = get_min_dur_s()
                        st.caption(
                            f"⚙️ Duração mínima de acc/dec: **{_nm_dur_s:.1f} s** "
                            f"({max(1, round(_nm_dur_s * _SENSOR_HZ))} frames a 10 Hz) — "
                            "ajuste na sidebar."
                        )

                        # Combina sensor_points de todos os períodos se necessário
                        if _nm_per == _NM_TODOS:
                            _nm_sp = []
                            for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                _nm_sp += _pv2.get(_nm_atl, [])
                            if len(dados_sensor_por_atleta_por_periodo) > 1:
                                st.caption(
                                    f"📊 Combinando **{len(dados_sensor_por_atleta_por_periodo)} períodos** "
                                    f"→ {len(_nm_sp):,} amostras para **{_nm_atl}**."
                                )
                        else:
                            _nm_sp = dados_sensor_por_atleta_por_periodo[_nm_per].get(_nm_atl, [])

                        _nm_dados = calcular_carga_neuromuscular(_nm_sp, limiar=_nm_lim, min_dur_s=_nm_dur_s)

                        if _nm_dados:
                            # Métricas resumo
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric(f"🟢 Acels. ≥{_nm_lim} m/s²", _nm_dados['total_hi_acc'])
                            c2.metric(f"🔴 Desacels. ≥{_nm_lim} m/s²", _nm_dados['total_hi_dec'])
                            c3.metric("⚡ Total Acels. (todas)", _nm_dados['total_hi_acc'] + _nm_dados['total_med_acc'])
                            c4.metric("⚡ Total Desacels. (todas)", _nm_dados['total_hi_dec'] + _nm_dados['total_med_dec'])

                            _nm_razao = ((_nm_dados['total_hi_dec'] / _nm_dados['total_hi_acc'])
                                         if _nm_dados['total_hi_acc'] > 0 else 0)
                            if _nm_razao > 1.4:
                                st.warning(f"⚠️ Razão Dec/Acc = **{_nm_razao:.2f}** — alto componente excêntrico. "
                                           "Monitorar recuperação muscular dos membros inferiores.")
                            elif _nm_razao > 0.9:
                                st.info(f"ℹ️ Razão Dec/Acc = **{_nm_razao:.2f}** — carga excêntrica equilibrada.")
                            else:
                                st.success(f"✅ Razão Dec/Acc = **{_nm_razao:.2f}** — perfil predominantemente acelerativo.")

                            _nm_fig = plotar_carga_neuromuscular(_nm_dados, _nm_atl)
                            st.plotly_chart(_nm_fig, use_container_width=True)

                            # Exportar
                            _nm_df = pd.DataFrame({
                                'Tempo (min)': _nm_dados['t_mid'],
                                'Acels. Intensas/min': _nm_dados['hi_acc_min'],
                                'Acels. Médias/min': _nm_dados['med_acc_min'],
                                'Desacels. Intensas/min': _nm_dados['hi_dec_min'],
                                'Desacels. Médias/min': _nm_dados['med_dec_min'],
                            })
                            st.download_button(
                                "📥 Exportar Carga Neuromuscular (CSV)",
                                _nm_df.to_csv(index=False),
                                f"carga_neuro_{_nm_atl.replace(' ','_')}.csv"
                            )
                        else:
                            st.info("Dados de aceleração insuficientes para este atleta/período.")

                        # ── FEATURE 3: Potência Metabólica na aba Carga Neuromuscular ──
                        st.markdown("---")
                        st.markdown("### ⚡ Potência Metabólica (W/kg)")
                        _nm_mp_pts = _nm_sp
                        _nm_mp_vals = [
                            float(p['mp']) for p in _nm_mp_pts
                            if p.get('mp') and float(p.get('mp') or 0) > 0
                        ]
                        if _nm_mp_vals:
                            _nm_mp_mean = float(np.mean(_nm_mp_vals))
                            _nm_mp_max  = float(np.max(_nm_mp_vals))
                            _nm_mp_pct20 = sum(1 for v in _nm_mp_vals if v > 20) / max(1, len(_nm_mp_vals)) * 100
                            _nm_mp_t25   = sum(1 for v in _nm_mp_vals if v > 25) * 0.1
                            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                            _mc1.metric("MP Médio (W/kg)", f"{_nm_mp_mean:.1f}")
                            _mc2.metric("MP Máx (W/kg)", f"{_nm_mp_max:.1f}")
                            _mc3.metric("MP > 20 W/kg (%)", f"{_nm_mp_pct20:.1f}%")
                            _mc4.metric("Tempo > 25 W/kg (s)", f"{_nm_mp_t25:.0f}s")
                        else:
                            st.info(
                                "Dados de potência metabólica (mp) não disponíveis. "
                                "Verifique se o dispositivo suporta este parâmetro."
                            )

                else:
                    st.info("Carregue os dados de um atleta para analisar a carga neuromuscular.")

            # ==================== ABA 6: PERFIL ACELERAÇÃO-VELOCIDADE ====================
            with abas[5]:
                st.subheader("🏎️ Perfil Aceleração × Velocidade")
                st.caption(
                    "Baseado no modelo de Samozino & Morin (2016) — relação linear entre aceleração e velocidade "
                    "para extrair o perfil mecânico individual de sprint."
                )
                st.markdown("---")

                _av_periodos = list(dados_sensor_por_atleta_por_periodo.keys())
                if _av_periodos and resultados_por_periodo:
                    _AV_TODOS = "🔀 Todos os períodos (combinado)"
                    _av_opcoes_per = [_AV_TODOS] + _av_periodos
                    _av_col1, _av_col2 = st.columns([2, 1])
                    with _av_col1:
                        _av_per = st.selectbox("Período:", _av_opcoes_per, key="av_periodo")
                    with _av_col2:
                        if _av_per == _AV_TODOS:
                            _av_ats_set = set()
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _av_ats_set.update(_pv.keys())
                            _av_atletas_disp = sorted(_av_ats_set)
                        else:
                            _av_atletas_disp = list(dados_sensor_por_atleta_por_periodo.get(_av_per, {}).keys())
                        _av_atls_sel = st.multiselect(
                            "Atletas (até 6):", _av_atletas_disp,
                            default=_av_atletas_disp[:min(3, len(_av_atletas_disp))],
                            key="av_atletas_sel"
                        )

                    if _av_atls_sel:
                        # ── Paleta de cores por atleta (usa paleta global persistente) ─────
                        _AV_PALETTE = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4']
                        _av_cores = {
                            a: st.session_state.get('athlete_colors', {}).get(a, _AV_PALETTE[i % len(_AV_PALETTE)])
                            for i, a in enumerate(_av_atls_sel)
                        }

                        # ── Extrai (v, a) de todos os atletas selecionados ────
                        _av_dados = {}
                        for _av_atl in _av_atls_sel:
                            if _av_per == _AV_TODOS:
                                _spts = []
                                for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                    _spts += _pv2.get(_av_atl, [])
                            else:
                                _spts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_av_atl, [])
                            if not _spts:
                                continue
                            _vels, _accs = [], []
                            for _p in _spts:
                                _v = _p.get('v')
                                _a = _p.get('a')
                                if _v is not None and _a is not None:
                                    _vels.append(float(_v) * 3.6)
                                    _accs.append(float(_a))
                            if _vels:
                                _av_dados[_av_atl] = {
                                    'vel': np.array(_vels),
                                    'acc': np.array(_accs),
                                }

                        if not _av_dados:
                            st.warning("Dados de aceleração não disponíveis para os atletas selecionados.")
                        else:
                            # ── FEATURE 3: toggle para velocidade bruta (rv) ─
                            _av_use_rv = st.toggle(
                                "Usar velocidade bruta (rv) para F-V profiling",
                                value=False, key="av_use_rv",
                                help="rv = velocidade bruta do sensor (mais preciso para F-V). "
                                     "Requer que o dispositivo suporte o parâmetro rv.",
                            )
                            if _av_use_rv:
                                st.caption("📍 **Perfil F-V (velocidade bruta — mais preciso)**")
                                # Recalcula usando rv ao invés de v
                                for _av_atl in list(_av_dados.keys()):
                                    if _av_per == _AV_TODOS:
                                        _spts_rv = []
                                        for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                            _spts_rv += _pv2.get(_av_atl, [])
                                    else:
                                        _spts_rv = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_av_atl, [])
                                    _vels_rv, _accs_rv = [], []
                                    for _p in _spts_rv:
                                        _rv = _p.get('rv')
                                        _a = _p.get('a')
                                        if _rv is not None and _a is not None:
                                            _vels_rv.append(float(_rv) * 3.6)
                                            _accs_rv.append(float(_a))
                                    if _vels_rv:
                                        _av_dados[_av_atl]['vel'] = np.array(_vels_rv)
                                        _av_dados[_av_atl]['acc'] = np.array(_accs_rv)

                            # ════════════════════════════════════════════════
                            # SEÇÃO 1 — SCATTER ACC × VEL (multi-atleta)
                            # ════════════════════════════════════════════════
                            _sc_title = ("📍 Perfil F-V (velocidade bruta — mais preciso)"
                                         if _av_use_rv else "📍 Scatter Aceleração × Velocidade")
                            st.markdown(f"### {_sc_title}")
                            _fig_sc = go.Figure()

                            for _av_atl, _d in _av_dados.items():
                                # Sub-amostra para não sobrecarregar o gráfico
                                _step = max(1, len(_d['vel']) // 3000)
                                _v_s  = _d['vel'][::_step]
                                _a_s  = _d['acc'][::_step]
                                _fig_sc.add_trace(go.Scatter(
                                    x=_v_s, y=_a_s, mode='markers',
                                    marker=dict(color=_av_cores[_av_atl], size=3, opacity=0.45),
                                    name=_av_atl,
                                    hovertemplate=f'{_av_atl}<br>Vel: %{{x:.1f}} km/h<br>Acc: %{{y:.2f}} m/s²<extra></extra>',
                                ))

                            # Linhas de limiar
                            for _lim, _cor, _txt in [(3.0,'#F44336','Acc >3 m/s²'),
                                                      (2.0,'#FF9800','Acc >2 m/s²'),
                                                      (-2.0,'#FF9800','Dcc <-2 m/s²'),
                                                      (-3.0,'#F44336','Dcc <-3 m/s²')]:
                                _fig_sc.add_hline(y=_lim, line=dict(color=_cor, dash='dash', width=1),
                                                  annotation_text=_txt,
                                                  annotation_font=dict(color=_cor, size=10))

                            _fig_sc.add_hline(y=0, line=dict(color='white', width=1, dash='dot'))
                            _fig_sc.update_layout(
                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                font=dict(color='white'),
                                xaxis=dict(title='Velocidade (km/h)', gridcolor='#333', range=[-1, None]),
                                yaxis=dict(title='Aceleração (m/s²)', gridcolor='#333'),
                                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)'),
                                height=420, margin=dict(t=20, b=10),
                            )
                            st.plotly_chart(_fig_sc, use_container_width=True)

                            st.markdown("---")

                            # ════════════════════════════════════════════════
                            # SEÇÃO 2 — HISTOGRAMA DE ACELERAÇÕES POR ZONA
                            # ════════════════════════════════════════════════
                            st.markdown("### 📊 Distribuição de Acelerações por Zona")
                            _av_c3, _av_c4 = st.columns(2)

                            _ZONAS_ACC = [
                                ('Muito intensa (>3)', 3.0, 99, '#F44336'),
                                ('Intensa (2–3)',       2.0, 3.0,'#FF9800'),
                                ('Moderada (1–2)',      1.0, 2.0,'#FFEB3B'),
                                ('Leve (0–1)',          0.0, 1.0,'#4CAF50'),
                                ('Desac. leve (0–-1)', -1.0, 0.0,'#26C6DA'),
                                ('Desac. mod. (-1–-2)',-2.0,-1.0,'#1565C0'),
                                ('Desac. int. (-2–-3)',-3.0,-2.0,'#7B1FA2'),
                                ('Desac. muito int. (<-3)',-99,-3.0,'#880E4F'),
                            ]

                            with _av_c3:
                                _fig_hist = go.Figure()
                                for _av_atl, _d in _av_dados.items():
                                    _fig_hist.add_trace(go.Histogram(
                                        x=_d['acc'], name=_av_atl,
                                        marker_color=_av_cores[_av_atl],
                                        opacity=0.65, nbinsx=60,
                                        hovertemplate='Acc: %{x:.2f} m/s²<br>Freq: %{y}<extra></extra>',
                                    ))
                                _fig_hist.update_layout(
                                    title=dict(text='Histograma de Aceleração', font=dict(color='white',size=13)),
                                    barmode='overlay',
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(title='Aceleração (m/s²)', gridcolor='#333'),
                                    yaxis=dict(title='Frequência', gridcolor='#333'),
                                    legend=dict(font=dict(color='white')),
                                    height=320, margin=dict(t=45,b=10,l=10,r=10),
                                )
                                st.plotly_chart(_fig_hist, use_container_width=True)

                            with _av_c4:
                                # Proporção por zona para o primeiro atleta selecionado
                                _av_atl_zona = _av_atls_sel[0]
                                if _av_atl_zona in _av_dados:
                                    _acc_z = _av_dados[_av_atl_zona]['acc']
                                    _zona_counts, _zona_labels, _zona_cores = [], [], []
                                    for _zl, _zmin, _zmax, _zc in _ZONAS_ACC:
                                        _n = int(np.sum((_acc_z >= _zmin) & (_acc_z < _zmax)))
                                        _zona_counts.append(_n)
                                        _zona_labels.append(_zl)
                                        _zona_cores.append(_zc)
                                    _fig_pie = go.Figure(go.Pie(
                                        labels=_zona_labels, values=_zona_counts,
                                        marker=dict(colors=_zona_cores),
                                        textfont=dict(color='white', size=10),
                                        hole=0.4,
                                    ))
                                    _fig_pie.update_layout(
                                        title=dict(text=f'Distribuição por Zona — {_av_atl_zona}',
                                                   font=dict(color='white',size=13)),
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        legend=dict(font=dict(color='white',size=9)),
                                        height=320, margin=dict(t=45,b=10,l=10,r=10),
                                    )
                                    st.plotly_chart(_fig_pie, use_container_width=True)

                            st.markdown("---")

                            # ════════════════════════════════════════════════
                            # SEÇÃO 5 — DENSIDADE NO ESPAÇO ACC × VEL (heatmap 2D)
                            # ════════════════════════════════════════════════
                            st.markdown("### 🌡️ Mapa de Densidade Acc × Vel")
                            st.caption(
                                "Mostra onde o atleta passa a maior parte do tempo no espaço aceleração-velocidade. "
                                "Regiões quentes = maior acúmulo de esforço."
                            )
                            _av_atl_hm = st.selectbox("Atleta para mapa de densidade:",
                                                       _av_atls_sel, key="av_hm_atl")
                            if _av_atl_hm in _av_dados:
                                _v_hm = _av_dados[_av_atl_hm]['vel']
                                _a_hm = _av_dados[_av_atl_hm]['acc']
                                _H2d, _xe2, _ye2 = np.histogram2d(
                                    _v_hm, _a_hm, bins=[50, 40],
                                    range=[[0, max(35, float(_v_hm.max()))], [-6, 6]]
                                )
                                _H2d = _gf(_H2d, sigma=1.5)
                                _xc2 = (_xe2[:-1] + _xe2[1:]) / 2
                                _yc2 = (_ye2[:-1] + _ye2[1:]) / 2
                                _fig_hm2 = go.Figure(go.Heatmap(
                                    x=_xc2, y=_yc2, z=_H2d.T,
                                    colorscale=[[0,'rgba(0,0,0,0)'],[0.0001,'#0D47A1'],
                                                [0.3,'#1565C0'],[0.6,'#FFEB3B'],
                                                [0.85,'#FF9800'],[1,'#F44336']],
                                    opacity=0.85, showscale=True,
                                    colorbar=dict(
                                        title=dict(text='Densidade', font=dict(color='white')),
                                        tickfont=dict(color='white'),
                                    ),
                                    hovertemplate='Vel: %{x:.1f} km/h<br>Acc: %{y:.2f} m/s²<br>Freq: %{z:.0f}<extra></extra>',
                                ))
                                _fig_hm2.add_hline(y=0, line=dict(color='white', width=1, dash='dot'))
                                _fig_hm2.add_hline(y=3,  line=dict(color='#F44336', width=1, dash='dash'))
                                _fig_hm2.add_hline(y=-3, line=dict(color='#F44336', width=1, dash='dash'))
                                _fig_hm2.update_layout(
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(title='Velocidade (km/h)', gridcolor='#333'),
                                    yaxis=dict(title='Aceleração (m/s²)', gridcolor='#333'),
                                    height=400, margin=dict(t=20,b=10),
                                )
                                st.plotly_chart(_fig_hm2, use_container_width=True)

                    else:
                        st.info("Selecione pelo menos 1 atleta para ver o perfil Acc × Vel.")

                    # ════════════════════════════════════════════════════════
                    # #14 — QUALIDADE DE ACELERAÇÃO + #11 — PL DIRECIONAL
                    # ════════════════════════════════════════════════════════
                    if _av_atletas_disp:
                        st.markdown("---")

                        # ── #14: Qualidade dos Eventos de Aceleração ──────────
                        st.markdown("### 🔬 Qualidade de Aceleração")
                        st.caption(
                            "Cada evento de aceleração (cruzar +2 m/s² por ≥0.3 s) é "
                            "caracterizado por: pico, impulso (área sob a curva), taxa de "
                            "desenvolvimento e velocidade de entrada. Mostra SE os esforços "
                            "mantêm qualidade ao longo do jogo."
                        )
                        _qa_atl = st.selectbox("Atleta:", _av_atletas_disp, key="qa_atleta")
                        if _av_per == _AV_TODOS:
                            _qa_pts: list = []
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _qa_pts += _pv.get(_qa_atl, [])
                        else:
                            _qa_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_qa_atl, [])

                        _qa_vs = np.array([float(p.get('v') or 0) * 3.6 for p in _qa_pts])
                        _qa_as = np.array([float(p.get('a') or 0) for p in _qa_pts])
                        _qa_ts = np.array([float(p.get('ts') or 0) for p in _qa_pts])

                        if len(_qa_as) > 10:
                            # Detectar eventos de aceleração: a > 2 m/s² por ≥ 3 amostras (0.3 s a 10 Hz)
                            _qa_eventos = []
                            _qa_in_event = False
                            _qa_start = 0
                            _qa_THRESH = 2.0
                            _qa_MIN_DUR = 3  # amostras
                            for _qi in range(len(_qa_as)):
                                if not _qa_in_event and _qa_as[_qi] >= _qa_THRESH:
                                    _qa_in_event = True
                                    _qa_start = _qi
                                elif _qa_in_event and (_qa_as[_qi] < _qa_THRESH or _qi == len(_qa_as)-1):
                                    _qa_end = _qi
                                    if _qa_end - _qa_start >= _qa_MIN_DUR:
                                        _seg_a = _qa_as[_qa_start:_qa_end]
                                        _seg_v = _qa_vs[_qa_start:_qa_end]
                                        _seg_t = _qa_ts[_qa_start:_qa_end]
                                        _qa_eventos.append({
                                            'inicio_min': (_seg_t[0] - _qa_ts[0]) / 60,
                                            'pico_a': float(np.max(_seg_a)),
                                            'impulso': float(np.trapezoid(_seg_a, dx=0.1)),
                                            'tdr': float((np.max(_seg_a) - _seg_a[0]) / max(0.1, (_seg_t[np.argmax(_seg_a)] - _seg_t[0]))),
                                            'vel_entrada': float(_seg_v[0]),
                                            'duracao_s': float(len(_seg_a) * 0.1),
                                        })
                                    _qa_in_event = False

                            if _qa_eventos:
                                _df_qa = pd.DataFrame(_qa_eventos)
                                _qa_kc1, _qa_kc2, _qa_kc3, _qa_kc4 = st.columns(4)
                                _qa_kc1.metric("Total de Eventos", len(_qa_eventos))
                                _qa_kc2.metric("Pico de Aceleração", f"{_df_qa['pico_a'].max():.2f} m/s²")
                                _qa_kc3.metric("Impulso Médio", f"{_df_qa['impulso'].mean():.2f} m/s")
                                _qa_kc4.metric("Vel. Entrada Média", f"{_df_qa['vel_entrada'].mean():.1f} km/h")

                                _qa_c1, _qa_c2 = st.columns(2)
                                with _qa_c1:
                                    # Scatter: tempo x pico_a colorido por impulso
                                    _fig_qa_sc = go.Figure()
                                    _fig_qa_sc.add_trace(go.Scatter(
                                        x=_df_qa['inicio_min'], y=_df_qa['pico_a'],
                                        mode='markers',
                                        marker=dict(
                                            size=_df_qa['impulso'].clip(3, 20),
                                            color=_df_qa['impulso'],
                                            colorscale='RdYlGn_r',
                                            showscale=True,
                                            colorbar=dict(title=dict(text='Impulso (m/s)', font=dict(color='white')),
                                                          tickfont=dict(color='white')),
                                        ),
                                        customdata=_df_qa[['impulso','vel_entrada','duracao_s']].values,
                                        hovertemplate=(
                                            'Min: %{x:.1f}<br>'
                                            'Pico Acc: %{y:.2f} m/s²<br>'
                                            'Impulso: %{customdata[0]:.2f} m/s<br>'
                                            'Vel entrada: %{customdata[1]:.1f} km/h<br>'
                                            'Duração: %{customdata[2]:.2f} s<extra></extra>'
                                        ),
                                    ))
                                    # linha de tendência do pico_a ao longo do jogo
                                    if len(_df_qa) >= 4:
                                        _qa_z = np.polyfit(_df_qa['inicio_min'], _df_qa['pico_a'], 1)
                                        _qa_xfit = np.linspace(_df_qa['inicio_min'].min(), _df_qa['inicio_min'].max(), 50)
                                        _fig_qa_sc.add_trace(go.Scatter(
                                            x=_qa_xfit, y=np.polyval(_qa_z, _qa_xfit),
                                            mode='lines', name='Tendência',
                                            line=dict(color='#FFD700', width=2, dash='dash'),
                                        ))
                                    _fig_qa_sc.update_layout(
                                        title=dict(text='Pico de Aceleração ao Longo do Jogo', font=dict(color='white', size=13)),
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='Pico de Aceleração (m/s²)', gridcolor='#333'),
                                        height=340, margin=dict(t=45,b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_qa_sc, use_container_width=True)

                                with _qa_c2:
                                    # Histograma de velocidade de entrada nos sprints
                                    _fig_qa_hist = go.Figure()
                                    _fig_qa_hist.add_trace(go.Histogram(
                                        x=_df_qa['vel_entrada'], nbinsx=12,
                                        marker=dict(color=cor_atleta(_qa_atl), opacity=0.8,
                                                    line=dict(color='white', width=0.5)),
                                        name='Vel. Entrada',
                                        hovertemplate='%{x:.1f}–%{x:.1f} km/h: %{y} eventos<extra></extra>',
                                    ))
                                    _fig_qa_hist.update_layout(
                                        title=dict(text='Velocidade de Entrada nos Eventos de Acc.', font=dict(color='white', size=13)),
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Velocidade (km/h)', gridcolor='#333'),
                                        yaxis=dict(title='Nº de Eventos', gridcolor='#333'),
                                        height=340, margin=dict(t=45,b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_qa_hist, use_container_width=True)
                            else:
                                st.info("Nenhum evento de aceleração >2 m/s² detectado para este atleta/período.")

                        st.markdown("---")

                        # ── #11: Player Load Direcional ───────────────────────
                        st.markdown("### 📐 Player Load Direcional")
                        st.caption(
                            "Decompõe o PlayerLoad nos 3 eixos: **Anteroposterior** (frente/trás — corrida "
                            "em linha), **Mediolateral** (esquerda/direita — movimentos defensivos/laterais) "
                            "e **Vertical** (saltos, duelos aéreos, contatos). "
                            "Parâmetros: `pla`, `plml`, `plv` do sensor Catapult."
                        )
                        _pld_atl = st.selectbox("Atleta:", _av_atletas_disp, key="pld_atleta")
                        if _av_per == _AV_TODOS:
                            _pld_pts: list = []
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _pld_pts += _pv.get(_pld_atl, [])
                        else:
                            _pld_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_pld_atl, [])

                        _pld_ap  = sum(float(p.get('pla')  or 0) for p in _pld_pts)
                        _pld_ml  = sum(float(p.get('plml') or 0) for p in _pld_pts)
                        _pld_vt  = sum(float(p.get('plv')  or 0) for p in _pld_pts)
                        _pld_tot = _pld_ap + _pld_ml + _pld_vt

                        # fallback: estimar AP/ML/V a partir de aceleração se sensor não tiver eixos
                        if _pld_tot < 0.01 and _pld_pts:
                            _est_note = True
                            _ap_sum = _ml_sum = _vt_sum = 0.0
                            for _pp in _pld_pts:
                                _av_vel = abs(float(_pp.get('v') or 0))
                                _av_acc = abs(float(_pp.get('a') or 0))
                                # heurística: se velocidade alta → contribuição AP; se baixo → ML
                                _ap_sum += _av_vel * 0.1
                                _ml_sum += max(0, _av_acc - _av_vel * 0.05) * 0.1
                            _pld_ap  = _ap_sum
                            _pld_ml  = _ml_sum
                            _pld_vt  = max(0.0, _pld_ap * 0.15)
                            _pld_tot = _pld_ap + _pld_ml + _pld_vt
                        else:
                            _est_note = False

                        if _pld_tot > 0:
                            if _est_note:
                                st.caption("⚠️ Sensor sem dados de eixo — estimativa heurística.")
                            _pld_c1, _pld_c2 = st.columns(2)
                            with _pld_c1:
                                # KPIs
                                _p1, _p2, _p3 = st.columns(3)
                                _p1.metric("Anteroposterior", f"{_pld_ap:.1f}", f"{_pld_ap/_pld_tot*100:.0f}%")
                                _p2.metric("Mediolateral",   f"{_pld_ml:.1f}", f"{_pld_ml/_pld_tot*100:.0f}%")
                                _p3.metric("Vertical",       f"{_pld_vt:.1f}", f"{_pld_vt/_pld_tot*100:.0f}%")
                                # Pizza
                                _fig_pld_pie = go.Figure(go.Pie(
                                    labels=['Anteroposterior', 'Mediolateral', 'Vertical'],
                                    values=[round(_pld_ap,1), round(_pld_ml,1), round(_pld_vt,1)],
                                    marker=dict(colors=['#2196F3','#4CAF50','#FF9800']),
                                    textinfo='label+percent',
                                    hole=0.42,
                                    hovertemplate='%{label}<br>PL: %{value:.1f}<br>%{percent}<extra></extra>',
                                ))
                                _fig_pld_pie.update_layout(
                                    title=dict(text=f'Distribuição de PL — {_pld_atl}', font=dict(color='white', size=13)),
                                    paper_bgcolor='#0e1117', font=dict(color='white'),
                                    legend=dict(font=dict(color='white', size=10)),
                                    height=320, margin=dict(t=50,b=10,l=10,r=10),
                                )
                                st.plotly_chart(_fig_pld_pie, use_container_width=True)

                            with _pld_c2:
                                # Comparativo entre todos os atletas
                                _pld_rows = []
                                for _pa in _av_atletas_disp:
                                    if _av_per == _AV_TODOS:
                                        _pa_pts: list = []
                                        for _ppv in dados_sensor_por_atleta_por_periodo.values():
                                            _pa_pts += _ppv.get(_pa, [])
                                    else:
                                        _pa_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_pa, [])
                                    _a = sum(float(p.get('pla') or 0)  for p in _pa_pts)
                                    _m = sum(float(p.get('plml') or 0) for p in _pa_pts)
                                    _v = sum(float(p.get('plv') or 0)  for p in _pa_pts)
                                    _t = _a+_m+_v
                                    if _t < 0.01 and _pa_pts:
                                        _a = sum(abs(float(p.get('v') or 0))*0.1 for p in _pa_pts)
                                        _m = sum(max(0,abs(float(p.get('a') or 0))-abs(float(p.get('v') or 0))*0.05)*0.1 for p in _pa_pts)
                                        _v = _a*0.15; _t = _a+_m+_v
                                    if _t > 0:
                                        _pld_rows.append({'Atleta':_pa,'AP':round(_a/_t*100,1),'ML':round(_m/_t*100,1),'VT':round(_v/_t*100,1)})
                                if _pld_rows:
                                    _df_pld = pd.DataFrame(_pld_rows)
                                    _fig_pld_bar = go.Figure()
                                    for _col_lbl, _col_color in [('AP','#2196F3'),('ML','#4CAF50'),('VT','#FF9800')]:
                                        _fig_pld_bar.add_trace(go.Bar(
                                            x=_df_pld['Atleta'], y=_df_pld[_col_lbl],
                                            name=_col_lbl, marker_color=_col_color,
                                            hovertemplate=f'{_col_lbl}: %{{y:.1f}}%<extra></extra>',
                                        ))
                                    _fig_pld_bar.update_layout(
                                        title=dict(text='% PL Direcional por Atleta', font=dict(color='white',size=13)),
                                        barmode='stack', paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(gridcolor='#333', tickangle=-30),
                                        yaxis=dict(title='% PL', gridcolor='#333'),
                                        legend=dict(font=dict(color='white')),
                                        height=320, margin=dict(t=50,b=10),
                                    )
                                    st.plotly_chart(_fig_pld_bar, use_container_width=True)
                        else:
                            st.info("Dados de PL direcional não disponíveis para este atleta/período.")
                else:
                    st.info("Busque os dados de atletas na sessão antes de usar esta análise.")

            # ══════════════════════════════════════════════════════════════
            # ABA 6: FC — TRIMP & ZONAS DE FREQUÊNCIA CARDÍACA
            # ══════════════════════════════════════════════════════════════
            with abas[6]:
                st.subheader("❤️ Frequência Cardíaca & TRIMP")
                st.caption(
                    "**TRIMP** (Training Impulse — Edwards): carga interna calculada pelo tempo em cada zona de "
                    "FC (50–60 / 60–70 / 70–80 / 80–90 / 90–100% HRmax) com multiplicadores 1–2–3–4–5. "
                    "Referência: FC máx = 220 − idade (padrão 180 bpm se não configurada)."
                )

                _fc_periodos = list(dados_sensor_por_atleta_por_periodo.keys())
                if _fc_periodos and resultados_por_periodo:
                    _FC_TODOS = "🔀 Todos os períodos (combinado)"
                    _fc_opcoes_per = [_FC_TODOS] + _fc_periodos

                    _fc_c1, _fc_c2, _fc_c3 = st.columns([2, 1, 1])
                    with _fc_c1:
                        _fc_per = st.selectbox("Período:", _fc_opcoes_per, key="fc_periodo")
                    with _fc_c2:
                        if _fc_per == _FC_TODOS:
                            _fc_ats_set = set()
                            for _fpv in dados_sensor_por_atleta_por_periodo.values():
                                _fc_ats_set.update(_fpv.keys())
                            _fc_atletas_disp = sorted(_fc_ats_set)
                        else:
                            _fc_atletas_disp = list(dados_sensor_por_atleta_por_periodo.get(_fc_per, {}).keys())
                        _fc_atl = st.selectbox("Atleta:", _fc_atletas_disp, key="fc_atleta_sel") if _fc_atletas_disp else None
                    with _fc_c3:
                        _fc_hrmax = st.number_input(
                            "FC Máx (bpm):", min_value=150, max_value=220,
                            value=180, step=1, key="fc_hrmax",
                            help="Padrão: 180 bpm. Ajuste para o valor individual do atleta."
                        )

                    if _fc_atl:
                        # Coletar pontos de sensor
                        if _fc_per == _FC_TODOS:
                            _fc_pts: list = []
                            for _fpv in dados_sensor_por_atleta_por_periodo.values():
                                _fc_pts += _fpv.get(_fc_atl, [])
                        else:
                            _fc_pts = dados_sensor_por_atleta_por_periodo.get(_fc_per, {}).get(_fc_atl, [])

                        _hr_vals = [float(p['hr']) for p in _fc_pts if p.get('hr') and float(p.get('hr') or 0) > 30]

                        if _hr_vals:
                            # ── Zonas Edwards ──────────────────────────────────────
                            _fc_zona_bounds = [
                                (0.50, 0.60, 1, 'Z1  50–60%', '#4CAF50'),
                                (0.60, 0.70, 2, 'Z2  60–70%', '#8BC34A'),
                                (0.70, 0.80, 3, 'Z3  70–80%', '#FFEB3B'),
                                (0.80, 0.90, 4, 'Z4  80–90%', '#FF9800'),
                                (0.90, 1.00, 5, 'Z5  90–100%', '#F44336'),
                            ]
                            _fc_zona_counts = {z[3]: 0 for z in _fc_zona_bounds}
                            _fc_trimp_total = 0.0
                            _fc_dt_s = 0.1  # 10 Hz → 0.1 s por ponto
                            for _hv in _hr_vals:
                                _hpct = _hv / _fc_hrmax
                                for _lo, _hi, _mult, _lbl, _col in _fc_zona_bounds:
                                    if _lo <= _hpct < _hi or (_hi == 1.00 and _hpct >= _lo):
                                        _fc_zona_counts[_lbl] += 1
                                        _fc_trimp_total += _fc_dt_s / 60 * _mult
                                        break

                            # ── KPIs ─────────────────────────────────────────────
                            _fk1, _fk2, _fk3, _fk4, _fk5 = st.columns(5)
                            _fk1.metric("TRIMP Total", f"{_fc_trimp_total:.1f}",
                                        help="Training Impulse (Edwards, 1993)")
                            _fk2.metric("FC Média (bpm)", f"{np.mean(_hr_vals):.0f}")
                            _fk3.metric("FC Máx Atingida", f"{max(_hr_vals):.0f} bpm")
                            _fk4.metric("% FCmax", f"{max(_hr_vals)/_fc_hrmax*100:.1f}%")
                            _fc_tempo_acima_80 = sum(
                                1 for hv in _hr_vals if hv / _fc_hrmax >= 0.80
                            ) * _fc_dt_s / 60
                            _fk5.metric("Tempo > 80% FCmax", f"{_fc_tempo_acima_80:.1f} min")

                            st.markdown("---")
                            _fc_chart_c1, _fc_chart_c2 = st.columns(2)

                            # ── Pizza — distribuição de zonas ─────────────────────
                            with _fc_chart_c1:
                                _fz_lbls = [z[3] for z in _fc_zona_bounds]
                                _fz_vals = [_fc_zona_counts[l] * _fc_dt_s / 60 for l in _fz_lbls]
                                _fz_cols = [z[4] for z in _fc_zona_bounds]
                                _fig_fz = go.Figure(go.Pie(
                                    labels=_fz_lbls, values=_fz_vals,
                                    marker=dict(colors=_fz_cols),
                                    textinfo='label+percent',
                                    hovertemplate='%{label}<br>%{value:.1f} min<extra></extra>',
                                    hole=0.38,
                                ))
                                _fig_fz.update_layout(
                                    title=dict(text=f'Distribuição de Zonas de FC — {_fc_atl}',
                                               font=dict(color='white', size=13)),
                                    paper_bgcolor='#0e1117', font=dict(color='white'),
                                    legend=dict(font=dict(color='white', size=9)),
                                    height=340, margin=dict(t=50, b=10, l=10, r=10),
                                )
                                st.plotly_chart(_fig_fz, use_container_width=True)

                            # ── Barras empilhadas TRIMP por zona ─────────────────
                            with _fc_chart_c2:
                                _fz_trimp = [_fc_zona_counts[l] * _fc_dt_s / 60 * _mult
                                             for l, (_, _, _mult, _, _) in zip(_fz_lbls, _fc_zona_bounds)]
                                _fig_trimp_fc = go.Figure()
                                for _fl, _ft, _fc_col in zip(_fz_lbls, _fz_trimp, _fz_cols):
                                    _fig_trimp_fc.add_trace(go.Bar(
                                        x=[_fc_atl], y=[_ft], name=_fl,
                                        marker_color=_fc_col,
                                        hovertemplate=f'{_fl}: %{{y:.1f}} TRIMP<extra></extra>',
                                    ))
                                _fig_trimp_fc.update_layout(
                                    title=dict(text=f'TRIMP por Zona — {_fc_atl}',
                                               font=dict(color='white', size=13)),
                                    barmode='stack',
                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(title='TRIMP (u.a.)', gridcolor='#333'),
                                    legend=dict(font=dict(color='white', size=9)),
                                    height=340, margin=dict(t=50, b=10, l=10, r=10),
                                )
                                st.plotly_chart(_fig_trimp_fc, use_container_width=True)

                            # ── Curva de FC ao longo do tempo ────────────────────
                            with st.expander("📈 Curva de FC ao longo do tempo", expanded=True):
                                _fc_ts = [float(p.get('ts') or 0) for p in _fc_pts
                                          if p.get('hr') and float(p.get('hr') or 0) > 30]
                                if _fc_ts:
                                    _fc_t0 = _fc_ts[0]
                                    _fc_ts_rel = [(t - _fc_t0) / 60 for t in _fc_ts]
                                    _fig_fc_curve = go.Figure()
                                    _fig_fc_curve.add_trace(go.Scatter(
                                        x=_fc_ts_rel, y=_hr_vals,
                                        mode='lines', line=dict(color='#F44336', width=1.5),
                                        name='FC (bpm)',
                                        hovertemplate='%{x:.1f} min — %{y:.0f} bpm<extra></extra>',
                                    ))
                                    # Linhas de zona
                                    for _lo, _hi, _mult, _lbl, _fcol in _fc_zona_bounds:
                                        _fig_fc_curve.add_hline(
                                            y=_lo * _fc_hrmax,
                                            line=dict(color=_fcol, width=1, dash='dot'),
                                            annotation_text=_lbl,
                                            annotation_font=dict(color=_fcol, size=9),
                                        )
                                    _fig_fc_curve.update_layout(
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='FC (bpm)', gridcolor='#333',
                                                   range=[40, _fc_hrmax * 1.05]),
                                        height=320, margin=dict(t=20, b=10),
                                        showlegend=False,
                                    )
                                    st.plotly_chart(_fig_fc_curve, use_container_width=True)

                            # ── Comparativo TRIMP entre todos os atletas ──────────
                            st.markdown("---")
                            st.markdown("### 📊 Comparativo de TRIMP — Todos os Atletas")
                            _fc_comp_rows = []
                            for _fc_cp_atl in _fc_atletas_disp:
                                if _fc_per == _FC_TODOS:
                                    _fc_cp_pts: list = []
                                    for _fpv in dados_sensor_por_atleta_por_periodo.values():
                                        _fc_cp_pts += _fpv.get(_fc_cp_atl, [])
                                else:
                                    _fc_cp_pts = dados_sensor_por_atleta_por_periodo.get(_fc_per, {}).get(_fc_cp_atl, [])
                                _fc_cp_hr = [float(p['hr']) for p in _fc_cp_pts if p.get('hr') and float(p.get('hr') or 0) > 30]
                                if _fc_cp_hr:
                                    _fc_cp_trimp = 0.0
                                    _fc_cp_z = {z[3]: 0 for z in _fc_zona_bounds}
                                    for _hv2 in _fc_cp_hr:
                                        _hp2 = _hv2 / _fc_hrmax
                                        for _lo2, _hi2, _m2, _l2, _c2 in _fc_zona_bounds:
                                            if _lo2 <= _hp2 < _hi2 or (_hi2 == 1.00 and _hp2 >= _lo2):
                                                _fc_cp_z[_l2] += 1
                                                _fc_cp_trimp += _fc_dt_s / 60 * _m2
                                                break
                                    _fc_comp_rows.append({
                                        'Atleta': _fc_cp_atl,
                                        'TRIMP': round(_fc_cp_trimp, 1),
                                        'FC Média (bpm)': round(float(np.mean(_fc_cp_hr)), 0),
                                        'FC Máx (bpm)': round(float(max(_fc_cp_hr)), 0),
                                        **{l: round(_fc_cp_z[l] * _fc_dt_s / 60, 2) for l in _fc_cp_z}
                                    })
                            if _fc_comp_rows:
                                _df_fc_comp = pd.DataFrame(_fc_comp_rows).sort_values('TRIMP', ascending=False)
                                st.dataframe(_df_fc_comp, use_container_width=True, hide_index=True)
                                # Gráfico de barras TRIMP comparativo
                                _fig_fc_comp_bar = px.bar(
                                    _df_fc_comp, x='Atleta', y='TRIMP',
                                    color='TRIMP', color_continuous_scale='Reds',
                                    title='TRIMP por Atleta',
                                )
                                _fig_fc_comp_bar.update_layout(
                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(title='TRIMP (u.a.)', gridcolor='#333'),
                                    height=350, margin=dict(t=40, b=10),
                                    showlegend=False,
                                )
                                st.plotly_chart(_fig_fc_comp_bar, use_container_width=True)
                        else:
                            st.info("ℹ️ Nenhum dado de FC disponível para este atleta/período.")
                    else:
                        st.info("Selecione um atleta para analisar a FC.")
                else:
                    st.info("Carregue dados da sessão para visualizar a análise de FC.")

                # ── #3: Curva de Fadiga Intra-Jogo ───────────────────────────
                st.markdown("---")
                st.markdown("## 📉 Curva de Fadiga Intra-Jogo")
                st.caption(
                    "Divide cada período em janelas de **5 minutos** e calcula a intensidade "
                    "relativa (m/min e PL/min). Uma linha de tendência negativa indica fadiga "
                    "progressiva. O minuto de início da queda é detectado automaticamente."
                )
                _fat_periodos = [p for p in dados_sensor_por_atleta_por_periodo if p != _CHAVE_COMBINADO]
                if _fat_periodos and resultados_por_periodo:
                    _fat_per = st.selectbox("Período:", _fat_periodos, key="fat_periodo")
                    _fat_atls_disp = list(dados_sensor_por_atleta_por_periodo.get(_fat_per, {}).keys())
                    if _fat_atls_disp:
                        _fat_sel = st.multiselect(
                            "Atletas (até 8):", _fat_atls_disp,
                            default=_fat_atls_disp[:min(5, len(_fat_atls_disp))],
                            key="fat_atletas",
                        )
                        _fat_janela_min = st.select_slider(
                            "Janela:", options=[2, 3, 5, 10], value=3, key="fat_janela",
                            help="Tamanho da janela temporal para cálculo de intensidade"
                        )
                        _fat_janela_s = _fat_janela_min * 60

                        if _fat_sel:
                            _fig_fat = go.Figure()
                            _fat_declive_info = []

                            for _fa in _fat_sel:
                                _fa_pts = dados_sensor_por_atleta_por_periodo.get(_fat_per, {}).get(_fa, [])
                                if not _fa_pts:
                                    continue
                                _fa_ts = np.array([float(p.get('ts') or 0) for p in _fa_pts])
                                _fa_vs = np.array([float(p.get('v')  or 0) * 3.6 for p in _fa_pts])
                                _fa_pl = np.array([float(p.get('pl') or 0) for p in _fa_pts])
                                if len(_fa_ts) < 2:
                                    continue
                                _fa_t0 = _fa_ts[0]
                                _fa_dur = _fa_ts[-1] - _fa_t0
                                if _fa_dur < _fat_janela_s * 2:
                                    continue

                                # janelas deslizantes
                                _win_centers_min = []
                                _win_mmin = []
                                _win_plmin = []
                                _t_start = _fa_t0
                                while _t_start + _fat_janela_s <= _fa_ts[-1]:
                                    _mask = (_fa_ts >= _t_start) & (_fa_ts < _t_start + _fat_janela_s)
                                    if _mask.sum() > 1:
                                        _dist_win = float(np.sum(np.abs(np.diff(_fa_vs[_mask])) * 0.1 / 3.6 * 3.6))
                                        # distância real via integral v×dt
                                        _d_real = float(np.trapezoid(_fa_vs[_mask] / 3.6, _fa_ts[_mask]))
                                        _pl_sum = float(np.sum(_fa_pl[_mask]))
                                        _win_mmin.append(_d_real / _fat_janela_min)
                                        _win_plmin.append(_pl_sum / _fat_janela_min)
                                        _win_centers_min.append((_t_start - _fa_t0 + _fat_janela_s / 2) / 60)
                                    _t_start += _fat_janela_s / 2  # sobreposição 50%

                                if len(_win_centers_min) < 3:
                                    continue

                                _wc = np.array(_win_centers_min)
                                _wm = np.array(_win_mmin)

                                _fa_color = cor_atleta(_fa)
                                _fig_fat.add_trace(go.Scatter(
                                    x=_wc, y=_wm, mode='lines+markers',
                                    name=_fa,
                                    line=dict(color=_fa_color, width=2),
                                    marker=dict(size=5, color=_fa_color),
                                    hovertemplate='%{x:.1f} min — %{y:.1f} m/min<extra>' + _fa + '</extra>',
                                ))
                                # Linha de tendência
                                if len(_wc) >= 4:
                                    _z = np.polyfit(_wc, _wm, 1)
                                    _xfit = np.linspace(_wc[0], _wc[-1], 50)
                                    _fig_fat.add_trace(go.Scatter(
                                        x=_xfit, y=np.polyval(_z, _xfit),
                                        mode='lines', showlegend=False,
                                        line=dict(color=_fa_color, width=1, dash='dot'),
                                        hoverinfo='skip',
                                    ))
                                    # Detectar início de queda: janela com variação negativa sustentada
                                    _rolling_diff = np.diff(_wm)
                                    _neg_idx = np.where(_rolling_diff < -5)[0]
                                    if len(_neg_idx) > 0:
                                        _queda_min = _wc[_neg_idx[0]]
                                        _fat_declive_info.append((_fa, round(float(_z[0]),2), round(_queda_min,1)))

                            _fig_fat.update_layout(
                                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                font=dict(color='white'),
                                xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                yaxis=dict(title='Intensidade (m/min)', gridcolor='#333'),
                                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.4)'),
                                height=380, margin=dict(t=20,b=10),
                            )
                            st.plotly_chart(_fig_fat, use_container_width=True)

                            # Tabela de declive e início de queda
                            if _fat_declive_info:
                                st.markdown("#### ⚠️ Atletas com Queda Detectada")
                                _df_fat_info = pd.DataFrame(
                                    _fat_declive_info,
                                    columns=['Atleta', 'Declive (m/min por min)', 'Início de Queda (min)']
                                ).sort_values('Declive (m/min por min)')
                                # badge de intensidade do declive
                                def _fat_badge(slope):
                                    if slope < -3:   return "🔴 Queda Severa"
                                    elif slope < -1: return "🟡 Queda Moderada"
                                    elif slope < 0:  return "🟢 Queda Leve"
                                    else:            return "✅ Estável"
                                _df_fat_info['Classificação'] = _df_fat_info['Declive (m/min por min)'].apply(_fat_badge)
                                st.dataframe(_df_fat_info, use_container_width=True, hide_index=True)
                    else:
                        st.info("Nenhum atleta com dados de sensor para este período.")
                else:
                    st.info("Carregue dados para visualizar a curva de fadiga.")

            # ══════════════════════════════════════════════════════════════
            # ABA 7: TABELA DESCRITIVA
            # ══════════════════════════════════════════════════════════════
            with abas[7]:
                st.subheader("📋 Tabela Descritiva de Desempenho")
                st.caption("Coloração por percentil do grupo: 🟢 Top 33% · 🟡 Médio · 🔴 Bottom 33%")
                st.caption("ℹ️ **RHIE** — Repeated High Intensity Efforts: blocos de ≥2 esforços >19 km/h separados por <21s · **HSR** — High Speed Running: distância acima de 19 km/h · **M/min** — Metros por minuto: intensidade relativa à duração (elite: 110–130 m/min)")

                _td_periodos = list(resultados_por_periodo.keys())
                if _td_periodos:
                    _td_periodo_sel = st.selectbox("Período:", _td_periodos, key="td_periodo_sel")

                    if resultados_por_periodo.get(_td_periodo_sel):
                        _df_td = pd.DataFrame(resultados_por_periodo[_td_periodo_sel])

                        # ── Usar dados OpenField pré-computados (FEATURE 2) ─────
                        _use_of = st.toggle(
                            "Usar dados OpenField (pré-computados)",
                            value=False,
                            key="td_use_openfield",
                            help="Compara os valores calculados localmente com os pré-computados pelo OpenField Cloud.",
                        )

                        # Calcula %Vmax = vel_max_periodo / vel_max_historica × 100
                        if 'Velocidade Máx (km/h)' in _df_td.columns:
                            _hist_vmax_dict = st.session_state.get('hist_vmax', {})

                            def _calc_pct_vmax(row):
                                _ath      = row.get('Atleta', '')
                                _vel_per  = float(row.get('Velocidade Máx (km/h)', 0) or 0)
                                _hist_raw = _hist_vmax_dict.get(_ath, 0) or 0
                                # hist_vmax armazenado em m/s → converte para km/h
                                _hist_kmh = float(_hist_raw) * 3.6 if _hist_raw > 0 else 0.0
                                # Sanidade: aceitar apenas se estiver no intervalo razoável (5–60 km/h)
                                if not (5.0 <= _hist_kmh <= 60.0):
                                    _hist_kmh = 0.0
                                if _hist_kmh > 0:
                                    return round(_vel_per / _hist_kmh * 100, 1)
                                # Sem histórico → % em relação à maior vel do grupo nesta sessão
                                _vmax_grupo = _df_td['Velocidade Máx (km/h)'].max()
                                if _vmax_grupo > 0:
                                    return round(_vel_per / _vmax_grupo * 100, 1)
                                return 0.0

                            _df_td['%Vmax'] = _df_td.apply(_calc_pct_vmax, axis=1)

                        # Colunas exibidas na tabela (ordem igual ao Power BI)
                        _TD_COLS = [c for c in [
                            'Atleta', 'Posição', 'Duração (min)', 'Distância (m)',
                            'Dist. 19-24 km/h (m)', 'Dist. > 24 km/h (m)',
                            'Dist. > 19 km/h (m)', 'Sprints (>24 km/h)',
                            'Velocidade Máx (km/h)', '%Vmax', 'M/min',
                            'Acc 2-3 (m/s²)', 'Dcc 2-3 (m/s²)',
                            'Acelerações (>3 m/s²)', 'Desacelerações (<-3 m/s²)',
                            'Acc Max (m/s²)', 'Dcc Max (m/s²)',
                            'PlayerLoad', 'RHIE Blocos',
                        ] if c in _df_td.columns]

                        _df_td_show = _df_td[_TD_COLS].copy()

                        # Colunas numéricas para coloração por percentil
                        _TD_NUM = [c for c in _TD_COLS if c not in ('Atleta', 'Posição')]

                        def _td_style_percentile(col):
                            if col.name not in _TD_NUM:
                                return [''] * len(col)
                            p33 = col.quantile(0.33)
                            p66 = col.quantile(0.66)
                            out = []
                            for v in col:
                                try:
                                    vf = float(v)
                                    if vf >= p66:
                                        out.append('background-color:#1a5c2e;color:white;font-weight:bold')
                                    elif vf >= p33:
                                        out.append('background-color:#7d6a08;color:white')
                                    else:
                                        out.append('background-color:#7b1a1a;color:white')
                                except Exception:
                                    out.append('')
                            return out

                        # Montar dict de formato num único .format() — dois .format()
                        # encadeados sobrescrevem um ao outro em versões antigas do pandas.
                        _1dec_cols = {
                            'Velocidade Máx (km/h)', 'Velocidade Média (km/h)',
                            'M/min', 'Acc Max (m/s²)', 'Dcc Max (m/s²)',
                        }
                        _td_fmt = {}
                        for _c in _TD_NUM:
                            if _c == '%Vmax':
                                _td_fmt[_c] = '{:.1f}%'
                            elif _c in _1dec_cols:
                                _td_fmt[_c] = '{:.1f}'
                            else:
                                _td_fmt[_c] = '{:.0f}'

                        _styled_td = (
                            _df_td_show.style
                            .apply(_td_style_percentile, axis=0)
                            .format(_td_fmt, na_rep='—')
                            .set_properties(**{'text-align': 'center'})
                            .set_table_styles([
                                {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
                                {'selector': 'td', 'props': [('text-align', 'center')]},
                            ])
                        )
                        # CSS injetado para garantir centralização independente do tema Streamlit
                        st.markdown(
                            "<style>"
                            "[data-testid='stDataFrame'] th {"
                            "  text-align: center !important;"
                            "  justify-content: center !important;"
                            "}"
                            "[data-testid='stDataFrame'] td {"
                            "  text-align: center !important;"
                            "}"
                            "</style>",
                            unsafe_allow_html=True,
                        )
                        st.dataframe(_styled_td, use_container_width=True, hide_index=True)

                        # ── OpenField vs Calculado Localmente (FEATURE 2) ─────────
                        if _use_of:
                            st.markdown("---")
                            st.markdown("### 🔬 OpenField vs Calculado Localmente")
                            _pos_dados_td = dados_posicao_por_periodo.get(_td_periodo_sel, {})
                            _of_comparisons = []
                            for _td_row in resultados_por_periodo.get(_td_periodo_sel, []):
                                _td_atl = _td_row.get('Atleta', '')
                                _of_sum = _pos_dados_td.get(_td_atl, {}).get('openfield_summary')
                                if _of_sum and isinstance(_of_sum, (dict, list)):
                                    _of_data = _of_sum if isinstance(_of_sum, dict) else (_of_sum[0] if _of_sum else {})
                                    _of_params = _of_data.get('parameters', _of_data)
                                    _of_dist = float(_of_params.get('total_distance', 0) or 0)
                                    _loc_dist = _td_row.get('Distância (m)', 0)
                                    _of_vmax = float(_of_params.get('max_velocity', 0) or 0) * 3.6
                                    _loc_vmax = _td_row.get('Velocidade Máx (km/h)', 0)
                                    _of_comparisons.append({
                                        'Atleta': _td_atl,
                                        'Dist. OpenField (m)': round(_of_dist, 0),
                                        'Dist. Local (m)': _loc_dist,
                                        'Δ Distância (m)': round(_of_dist - _loc_dist, 0),
                                        'Vmax OpenField (km/h)': round(_of_vmax, 1),
                                        'Vmax Local (km/h)': _loc_vmax,
                                        'Δ Vmax (km/h)': round(_of_vmax - _loc_vmax, 1),
                                    })
                            if _of_comparisons:
                                _df_of_cmp = pd.DataFrame(_of_comparisons)

                                def _color_delta(val):
                                    try:
                                        v = float(val)
                                        if abs(v) < 5:
                                            return 'color:#4CAF50'
                                        elif abs(v) < 20:
                                            return 'color:#FF9800'
                                        return 'color:#F44336'
                                    except Exception:
                                        return ''

                                _delta_cols = ['Δ Distância (m)', 'Δ Vmax (km/h)']
                                _of_styled = _df_of_cmp.style.applymap(
                                    _color_delta, subset=_delta_cols
                                )
                                st.dataframe(_of_styled, use_container_width=True, hide_index=True)
                            else:
                                st.info(
                                    "Dados OpenField não disponíveis. O endpoint `/activities/{id}/athletes/{id}/summary` "
                                    "pode não estar habilitado na licença desta conta."
                                )

                        # ── Valores Médios do Grupo ──────────────────────────────
                        st.markdown("---")
                        st.markdown("### 📊 Valores Médios do Grupo")
                        _td_med = _df_td[_TD_NUM].mean()
                        _grp_cols = st.columns(6)
                        _grp_kpis = [
                            ('Distância (m)',          '📏', '{:,.0f} m'),
                            ('M/min',                  '⚡', '{:.1f} m/min'),
                            ('Dist. > 19 km/h (m)',    '🏃', '{:,.0f} m'),
                            ('Sprints (>24 km/h)',      '💨', '{:.0f}'),
                            ('Acelerações (>3 m/s²)',   '🔼', '{:.0f}'),
                            ('PlayerLoad',             '⚙️', '{:,.0f}'),
                        ]
                        for _gi, (_mk, _ico, _fmt) in enumerate(_grp_kpis):
                            if _mk in _td_med.index:
                                _grp_cols[_gi].metric(f"{_ico} {_mk}", _fmt.format(_td_med[_mk]))

                        # Segunda linha de médias
                        _grp_cols2 = st.columns(6)
                        _grp_kpis2 = [
                            ('Acc 2-3 (m/s²)',         '🟡', '{:.0f}'),
                            ('Dcc 2-3 (m/s²)',         '🟡', '{:.0f}'),
                            ('Desacelerações (<-3 m/s²)', '🔽', '{:.0f}'),
                            ('RHIE Blocos',             '🔁', '{:.0f}'),
                            ('Velocidade Máx (km/h)',   '💨', '{:.1f} km/h'),
                            ('%Vmax',                   '📈', '{:.1f}%'),
                        ]
                        for _gi2, (_mk2, _ico2, _fmt2) in enumerate(_grp_kpis2):
                            if _mk2 in _td_med.index:
                                _grp_cols2[_gi2].metric(f"{_ico2} {_mk2}", _fmt2.format(_td_med[_mk2]))

                        # ── Download CSV ─────────────────────────────────────────
                        st.download_button(
                            "📥 Exportar Tabela (CSV)",
                            _df_td_show.to_csv(index=False).encode('utf-8'),
                            f"tabela_descritiva_{_td_periodo_sel}.csv",
                            mime='text/csv'
                        )
                    else:
                        st.info("Nenhum dado disponível para este período.")
                else:
                    st.info("Carregue os dados para visualizar a tabela descritiva.")

            # ══════════════════════════════════════════════════════════════
            # ABA 8: POR POSIÇÃO
            # ══════════════════════════════════════════════════════════════
            with abas[8]:
                st.subheader("📊 Análise por Posição")
                st.caption("Média das métricas agrupada por posição tática dos atletas.")

                _pos_periodos = list(resultados_por_periodo.keys())
                if _pos_periodos:
                    _pos_periodo_sel = st.selectbox("Período:", _pos_periodos, key="pos_periodo_sel")

                    if resultados_por_periodo.get(_pos_periodo_sel):
                        _df_pos_raw = pd.DataFrame(resultados_por_periodo[_pos_periodo_sel])

                        if 'Posição' not in _df_pos_raw.columns or _df_pos_raw['Posição'].replace('', np.nan).isna().all():
                            st.warning("⚠️ Dados de posição não disponíveis. Verifique se as posições foram cadastradas na API Catapult.")
                        else:
                            _df_pos_raw = _df_pos_raw[_df_pos_raw['Posição'].notna() & (_df_pos_raw['Posição'] != '')]
                            _df_pos_grp = _df_pos_raw.groupby('Posição', as_index=False).mean(numeric_only=True)

                            # Paleta de cores por posição (usa sistema de grupos táticos)
                            _cores_pos = [
                                _POSICAO_COR_LEGENDA.get(_get_pos_grupo(p)[0], '#546E7A')
                                for p in _df_pos_grp['Posição']
                            ]

                            def _bar_pos(y_col, title, suffix=''):
                                if y_col not in _df_pos_grp.columns:
                                    return None
                                _f = go.Figure(go.Bar(
                                    x=_df_pos_grp['Posição'], y=_df_pos_grp[y_col],
                                    marker_color=_cores_pos, text=_df_pos_grp[y_col].round(1),
                                    textposition='outside',
                                    hovertemplate=f'%{{x}}<br>{y_col}: %{{y:.1f}}{suffix}<extra></extra>',
                                ))
                                _f.update_layout(
                                    title=dict(text=title, font=dict(color='white', size=13)),
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(gridcolor='#333', title=y_col),
                                    height=320, margin=dict(t=45, b=10, l=10, r=10),
                                    showlegend=False,
                                )
                                return _f

                            # Linha 1: Distância Total + M/min
                            _c1, _c2 = st.columns(2)
                            with _c1:
                                _f = _bar_pos('Distância (m)', 'Média de Distância Total por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c2:
                                _f = _bar_pos('M/min', 'Média de M/min por Posição', ' m/min')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # Linha 2: Zona 4 + Zona 5 (>24 km/h)
                            _c3, _c4 = st.columns(2)
                            with _c3:
                                _f = _bar_pos('Dist. 19-24 km/h (m)', 'Média de Zona 4 (19-24 km/h) por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c4:
                                _f = _bar_pos('Dist. > 24 km/h (m)', 'Média de Zona 5 (>24 km/h) por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # Linha 3: HSR (>19) + Sprints
                            _c5, _c6 = st.columns(2)
                            with _c5:
                                _f = _bar_pos('Dist. > 19 km/h (m)', 'Média de HSR Distance (>19 km/h) por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c6:
                                _f = _bar_pos('Sprints (>24 km/h)', 'Média de Sprints por Posição')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # Linha 4: Acc 2-3 + Dcc 2-3 (gráfico agrupado)
                            _acc23_col = 'Acc 2-3 (m/s²)'
                            _dcc23_col = 'Dcc 2-3 (m/s²)'
                            if _acc23_col in _df_pos_grp.columns and _dcc23_col in _df_pos_grp.columns:
                                _c7, _c8 = st.columns(2)
                                with _c7:
                                    _fig_acc = go.Figure()
                                    _fig_acc.add_trace(go.Bar(
                                        x=_df_pos_grp['Posição'], y=_df_pos_grp[_acc23_col],
                                        name='Acc 2-3', marker_color='#FFA000',
                                        text=_df_pos_grp[_acc23_col].round(1), textposition='outside',
                                    ))
                                    _fig_acc.add_trace(go.Bar(
                                        x=_df_pos_grp['Posição'], y=_df_pos_grp[_dcc23_col],
                                        name='Dcc 2-3', marker_color='#558B2F',
                                        text=_df_pos_grp[_dcc23_col].round(1), textposition='outside',
                                    ))
                                    _fig_acc.update_layout(
                                        title=dict(text='Média de Acc e Dcc 2-3 por Posição', font=dict(color='white', size=13)),
                                        barmode='group', plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), xaxis=dict(gridcolor='#333'),
                                        yaxis=dict(gridcolor='#333'), height=320,
                                        margin=dict(t=45, b=10, l=10, r=10),
                                        legend=dict(font=dict(color='white')),
                                    )
                                    st.plotly_chart(_fig_acc, use_container_width=True)

                                with _c8:
                                    # Acc >3 + Dcc >3 agrupados
                                    _fig_acc3 = go.Figure()
                                    if 'Acelerações (>3 m/s²)' in _df_pos_grp.columns:
                                        _fig_acc3.add_trace(go.Bar(
                                            x=_df_pos_grp['Posição'], y=_df_pos_grp['Acelerações (>3 m/s²)'],
                                            name='Acc >3', marker_color='#E53935',
                                            text=_df_pos_grp['Acelerações (>3 m/s²)'].round(1), textposition='outside',
                                        ))
                                    if 'Desacelerações (<-3 m/s²)' in _df_pos_grp.columns:
                                        _fig_acc3.add_trace(go.Bar(
                                            x=_df_pos_grp['Posição'], y=_df_pos_grp['Desacelerações (<-3 m/s²)'],
                                            name='Dcc >3', marker_color='#1565C0',
                                            text=_df_pos_grp['Desacelerações (<-3 m/s²)'].round(1), textposition='outside',
                                        ))
                                    _fig_acc3.update_layout(
                                        title=dict(text='Média de Acc e Dcc >3 por Posição', font=dict(color='white', size=13)),
                                        barmode='group', plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), xaxis=dict(gridcolor='#333'),
                                        yaxis=dict(gridcolor='#333'), height=320,
                                        margin=dict(t=45, b=10, l=10, r=10),
                                        legend=dict(font=dict(color='white')),
                                    )
                                    st.plotly_chart(_fig_acc3, use_container_width=True)

                            # Linha 5: PlayerLoad + RHIE Blocos
                            _c9, _c10 = st.columns(2)
                            with _c9:
                                _f = _bar_pos('PlayerLoad', 'Média de PlayerLoad por Posição')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c10:
                                _f = _bar_pos('RHIE Blocos', 'Média de RHIE Blocos por Posição')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # ── Item 16: Radar de Perfil por Posição ─────────────
                            st.markdown("---")
                            st.markdown("### 🕸️ Radar de Perfil Físico por Posição")
                            st.caption(
                                "Normalização min-max do grupo: 100% = maior valor do grupo. "
                                "Permite comparar perfis multidimensionais entre posições."
                            )
                            _radar_metrics = [c for c in [
                                'Distância (m)', 'M/min', 'Dist. > 19 km/h (m)',
                                'Sprints (>24 km/h)', 'Acelerações (>3 m/s²)',
                                'Desacelerações (<-3 m/s²)', 'PlayerLoad', 'RHIE Blocos',
                            ] if c in _df_pos_grp.columns]

                            if len(_radar_metrics) >= 4 and len(_df_pos_grp) >= 2:
                                # Normalização min-max por métrica (0-100%)
                                _radar_norm = _df_pos_grp[_radar_metrics].copy()
                                for _rc in _radar_metrics:
                                    _rmin = _radar_norm[_rc].min()
                                    _rmax = _radar_norm[_rc].max()
                                    _radar_norm[_rc] = ((_radar_norm[_rc] - _rmin) / max(_rmax - _rmin, 1e-9) * 100).round(1)

                                _POS_RADAR_PALETTE = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4','#F44336']
                                _fig_radar = go.Figure()
                                for _ri, (_rpos, _rrow) in enumerate(_radar_norm.iterrows()):
                                    _pos_name = _df_pos_grp['Posição'].iloc[_ri]
                                    _vals = list(_rrow[_radar_metrics]) + [_rrow[_radar_metrics[0]]]
                                    _lbls = _radar_metrics + [_radar_metrics[0]]
                                    _col_r = _POS_RADAR_PALETTE[_ri % len(_POS_RADAR_PALETTE)]
                                    _fig_radar.add_trace(go.Scatterpolar(
                                        r=_vals, theta=_lbls, fill='toself',
                                        name=_pos_name,
                                        line=dict(color=_col_r, width=2),
                                        fillcolor=(lambda c: f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)")(_col_r),
                                        opacity=0.85,
                                        hovertemplate='%{theta}: %{r:.1f}%<extra>' + _pos_name + '</extra>',
                                    ))
                                _fig_radar.update_layout(
                                    polar=dict(
                                        bgcolor='#0e1117',
                                        radialaxis=dict(
                                            visible=True, range=[0, 100],
                                            gridcolor='rgba(255,255,255,0.15)',
                                            tickfont=dict(color='rgba(200,200,200,0.7)', size=9),
                                            ticksuffix='%',
                                        ),
                                        angularaxis=dict(
                                            gridcolor='rgba(255,255,255,0.15)',
                                            tickfont=dict(color='white', size=10),
                                        ),
                                    ),
                                    paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    legend=dict(font=dict(color='white', size=10)),
                                    height=460, margin=dict(t=20, b=20, l=20, r=20),
                                )
                                st.plotly_chart(_fig_radar, use_container_width=True)
                            else:
                                st.info("Necessário ≥ 2 posições e ≥ 4 métricas para gerar o radar.")

                            # ── Tabela resumo por posição ─────────────────────────
                            st.markdown("---")
                            st.markdown("### 📋 Resumo por Posição")
                            _TD_POS_COLS = [c for c in [
                                'Posição', 'Distância (m)', 'M/min', 'Dist. 19-24 km/h (m)',
                                'Dist. > 24 km/h (m)', 'Sprints (>24 km/h)', 'Velocidade Máx (km/h)',
                                'Acc 2-3 (m/s²)', 'Dcc 2-3 (m/s²)', 'Acelerações (>3 m/s²)',
                                'Desacelerações (<-3 m/s²)', 'PlayerLoad', 'RHIE Blocos',
                            ] if c in _df_pos_grp.columns]
                            st.dataframe(
                                _df_pos_grp[_TD_POS_COLS].round(1),
                                use_container_width=True, hide_index=True
                            )

                            # ── FEATURE 11: Consulta /stats por posição (longitudinal) ──
                            st.markdown("---")
                            with st.expander("📊 Perfil Longitudinal por Posição via /stats (multi-sessão)", expanded=False):
                                st.caption(
                                    "Consulta o endpoint POST /stats da Catapult para agregar métricas "
                                    "por posição em múltiplas sessões. Requer dados históricos na plataforma."
                                )
                                _stats_api = st.session_state.get('api')

                                # Date range filter
                                _dr_c1, _dr_c2, _dr_c3 = st.columns([2, 2, 1])
                                with _dr_c1:
                                    _stats_start = st.date_input(
                                        "Data inicial:", key="stats_pos_start_date",
                                        value=None,
                                    )
                                with _dr_c2:
                                    _stats_end = st.date_input(
                                        "Data final:", key="stats_pos_end_date",
                                        value=None,
                                    )
                                with _dr_c3:
                                    _stats_grp_week = st.checkbox(
                                        "Agrupar por semana", value=False, key="stats_pos_byweek"
                                    )

                                if _stats_api and st.button("🔄 Consultar /stats por posição", key="btn_stats_pos"):
                                    with st.spinner("Consultando /stats..."):
                                        _stats_payload = {
                                            "group_by": (
                                                ["position", "week"] if _stats_grp_week else ["position"]
                                            ),
                                            "parameters": [
                                                "total_distance", "hsr_distance", "sprint_distance",
                                                "player_load", "max_velocity",
                                            ],
                                            "source": "cached_stats",
                                        }
                                        if _stats_start:
                                            import time as _tm_s
                                            _stats_payload["start_time"] = int(
                                                _tm_s.mktime(_stats_start.timetuple())
                                            )
                                        if _stats_end:
                                            import time as _tm_e
                                            _stats_payload["end_time"] = int(
                                                _tm_e.mktime(_stats_end.timetuple())
                                            )
                                        _stats_resp = _stats_api.get_stats(_stats_payload)
                                        st.session_state['stats_pos_resp'] = _stats_resp

                                _stats_resp = st.session_state.get('stats_pos_resp')
                                if _stats_resp:
                                    try:
                                        _sd = _stats_resp if isinstance(_stats_resp, list) else _stats_resp.get('data', [])
                                        if _sd:
                                            _df_stats = pd.DataFrame(_sd)
                                            st.dataframe(_df_stats, use_container_width=True, hide_index=True)

                                            # Radar chart por posição dos dados históricos
                                            st.markdown("#### 🕸️ Radar por Posição (Dados Históricos)")
                                            _stats_pos_col = next(
                                                (c for c in _df_stats.columns
                                                 if 'position' in c.lower() or 'pos' == c.lower()), None
                                            )
                                            if _stats_pos_col:
                                                _radar_hist_metrics = [
                                                    c for c in _df_stats.columns
                                                    if c != _stats_pos_col and pd.api.types.is_numeric_dtype(_df_stats[c])
                                                ]
                                                if len(_radar_hist_metrics) >= 3:
                                                    _fig_radar_hist = go.Figure()
                                                    _df_stats_norm = _df_stats.copy()
                                                    for _rm in _radar_hist_metrics:
                                                        _col_max = _df_stats_norm[_rm].max()
                                                        if _col_max > 0:
                                                            _df_stats_norm[_rm] = (_df_stats_norm[_rm] / _col_max * 100)
                                                    for _, _prow in _df_stats_norm.iterrows():
                                                        _r_vals = [float(_prow.get(_rm, 0) or 0) for _rm in _radar_hist_metrics]
                                                        _fig_radar_hist.add_trace(go.Scatterpolar(
                                                            r=_r_vals + [_r_vals[0]],
                                                            theta=_radar_hist_metrics + [_radar_hist_metrics[0]],
                                                            fill='toself',
                                                            name=str(_prow.get(_stats_pos_col, '—')),
                                                            opacity=0.7,
                                                        ))
                                                    _fig_radar_hist.update_layout(
                                                        polar=dict(
                                                            radialaxis=dict(visible=True, range=[0, 105]),
                                                            bgcolor='#0e1117',
                                                        ),
                                                        paper_bgcolor='#0e1117',
                                                        font=dict(color='white'),
                                                        height=420,
                                                        legend=dict(font=dict(color='white')),
                                                    )
                                                    st.plotly_chart(_fig_radar_hist, use_container_width=True)

                                            # Week-over-week evolution if grouped by week
                                            if _stats_grp_week and 'week' in ' '.join(_df_stats.columns).lower():
                                                _wk_col = next(
                                                    (c for c in _df_stats.columns if 'week' in c.lower()), None
                                                )
                                                if _wk_col and _stats_pos_col and 'total_distance' in _df_stats.columns:
                                                    st.markdown("#### 📈 Evolução Semanal por Posição")
                                                    _fig_wk = px.line(
                                                        _df_stats, x=_wk_col, y='total_distance',
                                                        color=_stats_pos_col,
                                                        title="Distância Total por Semana e Posição",
                                                        markers=True,
                                                    )
                                                    _fig_wk.update_layout(
                                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                                        font=dict(color='white'),
                                                    )
                                                    st.plotly_chart(_fig_wk, use_container_width=True)
                                        else:
                                            st.info("Nenhum dado retornado pelo /stats.")
                                    except Exception as _e:
                                        st.error(f"Erro ao processar /stats: {_e}")
                                else:
                                    st.warning("⚠️ Endpoint /stats não respondeu. Verifique permissões da API ou se há dados históricos suficientes.")
                    else:
                        st.info("Nenhum dado disponível para este período.")
                else:
                    st.info("Carregue os dados para visualizar a análise por posição.")

            # ══════════════════════════════════════════════════════════════
            # ABA 9: HISTÓRIA DO JOGO
            # ══════════════════════════════════════════════════════════════
            with abas[9]:
                st.subheader("🎬 História do Jogo")
                st.caption("Animação top-down de todos os atletas movendo-se pelo campo ao longo do tempo.")

                # ── Obter config do campo ─────────────────────────────────
                _hist_cfg = None
                for _hk in list(st.session_state.keys()):
                    if _hk.startswith("campo_cfg__") and isinstance(st.session_state[_hk], dict) and 'fl' in st.session_state[_hk]:
                        _hist_cfg = st.session_state[_hk]
                        break

                if _hist_cfg:
                    _hist_fl = float(_hist_cfg.get('fl', 105))
                    _hist_fw = float(_hist_cfg.get('fw', 68))
                else:
                    _hist_venue = st.session_state.get('venue', {})
                    _hist_fl = float(_hist_venue.get('length') or 105)
                    _hist_fw = float(_hist_venue.get('width') or 68)
                    st.info("💡 Configure o campo na aba **🗺️ Campo de Futebol** para usar as dimensões exatas do campo.")

                if not dados_posicao_por_periodo:
                    st.info("Carregue os dados para visualizar a história do jogo.")
                else:
                    # ── Seletores ──────────────────────────────────────────
                    _hist_periodos_disp = list(dados_posicao_por_periodo.keys())
                    _col_hp, _col_hsp = st.columns(2)
                    with _col_hp:
                        _hist_periodo_sel = st.selectbox(
                            "Período:", _hist_periodos_disp, key="hist_periodo_sel"
                        )

                    # Mostra todos os atletas com qualquer dado de posição (xs OU GPS)
                    _hist_atletas_disp = [
                        a for a, d in dados_posicao_por_periodo.get(_hist_periodo_sel, {}).items()
                        if d.get('xs') or d.get('lats')
                    ]
                    # Sub-lista com xs+ts_pos (necessário para animação)
                    _hist_atletas_xy = [
                        a for a, d in dados_posicao_por_periodo.get(_hist_periodo_sel, {}).items()
                        if d.get('xs') and d.get('ts_pos')
                    ]
                    with _col_hsp:
                        # Seleciona todos os atletas do período por padrão
                        _hist_default = _hist_atletas_disp
                        _hist_atletas_sel = st.multiselect(
                            "Atletas:", _hist_atletas_disp,
                            default=_hist_default,
                            key="hist_atletas_sel"
                        )

                    if not _hist_atletas_disp:
                        st.info("Nenhum dado de posição encontrado para este período.")
                    elif not _hist_atletas_sel:
                        st.info("Selecione pelo menos um atleta.")
                    else:
                        # ── Controles de animação ──────────────────────────
                        _col_hc1, _col_hc2, _col_hc3 = st.columns(3)
                        with _col_hc1:
                            _hist_speed_opts = [round(i * 0.25, 2) for i in range(1, 21)]
                            _hist_speed = st.select_slider(
                                "⚡ Velocidade:",
                                options=_hist_speed_opts,
                                value=1.0,
                                format_func=lambda x: f"{x:.2g}×",
                                key="hist_speed"
                            )
                        with _col_hc2:
                            _hist_trail_s = st.select_slider(
                                "🌊 Rastro (s):", options=[3, 10, 20, 30],
                                value=3, key="hist_trail"
                            )
                        with _col_hc3:
                            _hist_max_frames = st.select_slider(
                                "🎞️ Qualidade (jogo completo):", options=[300, 500, 1000, 2000],
                                value=500, format_func=lambda x: f"{x} frames",
                                key="hist_quality"
                            )

                        # ── Janela de tempo (para animação suave em tempo real) ─
                        _col_hw1, _col_hw2 = st.columns([1, 2])
                        with _col_hw1:
                            _hist_window_sel = st.select_slider(
                                "🪟 Janela:",
                                options=["Completo", "10 min", "5 min", "3 min", "2 min", "1 min"],
                                value="3 min",
                                key="hist_window",
                                help=(
                                    "**Completo**: anima o período inteiro com downsample "
                                    "(menos suave, visão geral).\n\n"
                                    "**Janela**: usa TODOS os pontos GPS da janela selecionada "
                                    "→ animação em tempo real verdadeiro, milissegundo a milissegundo."
                                )
                            )
                        with _col_hw2:
                            if _hist_window_sel != "Completo":
                                # Duração do período calculada no render anterior
                                _prev_dur_s = st.session_state.get(
                                    f'_hist_dur_{_hist_periodo_sel}', 0.0
                                )
                                _win_s_prev_map = {
                                    "10 min": 600, "5 min": 300,
                                    "3 min": 180,  "2 min": 120,  "1 min": 60,
                                }
                                _win_s_prev = _win_s_prev_map.get(_hist_window_sel, 180)

                                if _prev_dur_s > 0:
                                    _max_start_min = max(
                                        1, int((_prev_dur_s - _win_s_prev) / 60)
                                    )
                                    # Garante que o valor armazenado não excede o novo máximo
                                    if st.session_state.get('hist_start_min', 0) > _max_start_min:
                                        st.session_state['hist_start_min'] = 0

                                    _hist_start_min = st.slider(
                                        "▶ Início:",
                                        min_value=0,
                                        max_value=_max_start_min,
                                        step=1,
                                        format="%d min",
                                        key="hist_start_min",
                                    )
                                    _hist_start_s = _hist_start_min * 60
                                    _end_s_prev   = _hist_start_s + _win_s_prev
                                    _tot_m = int(_prev_dur_s // 60)
                                    _tot_s = int(_prev_dur_s  % 60)
                                    st.caption(
                                        f"🕐 {_hist_start_min:02d}:00"
                                        f" → {int(_end_s_prev)//60:02d}:{int(_end_s_prev)%60:02d}"
                                        f"  (período: {_tot_m:02d}:{_tot_s:02d})"
                                    )
                                else:
                                    _hist_start_s = 0
                                    st.caption(
                                        "ℹ️ Gere a animação uma vez para habilitar "
                                        "a navegação por minuto."
                                    )
                            else:
                                _hist_start_s = 0

                        # ── Construir dados por atleta ─────────────────────
                        _HIST_COLORS = [
                            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                            '#FFEAA7', '#DDA0DD', '#98FB98', '#FFB347',
                            '#87CEEB', '#F08080', '#90EE90', '#FFD700',
                        ]

                        _hist_coords = {}
                        _hist_gps_fallback_used = False
                        for _ha in _hist_atletas_sel:
                            _hd = dados_posicao_por_periodo[_hist_periodo_sel].get(_ha, {})
                            _hxs = list(_hd.get('xs', []))
                            _hys = list(_hd.get('ys', []))
                            _hvl = list(_hd.get('vel', []))
                            _hts = list(_hd.get('ts_pos', []))
                            if _hxs and _hts and len(_hxs) == len(_hts):
                                # Coordenadas de campo (x,y) disponíveis diretamente
                                _hist_coords[_ha] = {
                                    'xs': _hxs, 'ys': _hys,
                                    'vel': _hvl if len(_hvl) == len(_hxs) else [0.0]*len(_hxs),
                                    'ts': _hts,
                                }
                            elif _hd.get('lats') and _hd.get('ts_gps') and _hist_cfg:
                                # Fallback: converter GPS → coordenadas de campo
                                _hlats   = _hd['lats']
                                _hlons   = _hd.get('lons', [])
                                _hts_gps = _hd['ts_gps']
                                _hvl_gps = _hd.get('vels_gps', [])
                                if _hlons and len(_hlats) == len(_hlons) == len(_hts_gps):
                                    try:
                                        _hfx, _hfy = gps_para_campo_coords(
                                            _hlats, _hlons, _hist_cfg
                                        )
                                        if _hfx:
                                            _hist_coords[_ha] = {
                                                'xs': _hfx, 'ys': _hfy,
                                                'vel': (list(_hvl_gps)
                                                        if len(_hvl_gps) == len(_hfx)
                                                        else [0.0] * len(_hfx)),
                                                'ts': list(_hts_gps),
                                            }
                                            _hist_gps_fallback_used = True
                                    except Exception:
                                        pass

                        if _hist_gps_fallback_used:
                            st.info(
                                "📡 Animação gerada a partir de dados **GPS** convertidos para coordenadas "
                                "de campo. Para usar dados de campo nativos (x,y), verifique a "
                                "configuração do sistema Catapult."
                            )

                        if not _hist_coords:
                            _n_sem_xy = len([a for a in _hist_atletas_sel
                                             if not dados_posicao_por_periodo
                                                .get(_hist_periodo_sel, {})
                                                .get(a, {}).get('xs')])
                            _n_sem_gps = len([a for a in _hist_atletas_sel
                                              if not dados_posicao_por_periodo
                                                 .get(_hist_periodo_sel, {})
                                                 .get(a, {}).get('lats')])
                            _cfg_aviso = (
                                "" if _hist_cfg else
                                "\n- ⚠️ **Campo não configurado** — acesse a aba "
                                "**🗺️ Campo de Futebol** e configure o campo para "
                                "habilitar a conversão GPS→campo"
                            )
                            st.warning(
                                f"⚠️ **{_n_sem_xy}/{len(_hist_atletas_sel)} atleta(s) sem dados de campo (x,y)** "
                                f"para o período **{_hist_periodo_sel}**.\n\n"
                                f"{_n_sem_gps}/{len(_hist_atletas_sel)} atleta(s) também sem dados GPS.\n\n"
                                "Possíveis causas:\n"
                                "- Os sensores não capturaram dados de posição neste período"
                                + _cfg_aviso
                            )
                        else:
                            # ── Normalizar timestamps ──────────────────────
                            _all_ts_raw = [t for _hc in _hist_coords.values() for t in _hc['ts']]
                            _ts_min_raw = min(_all_ts_raw)
                            # Detecta ms (>1e10) vs s
                            _ts_scale = 1000.0 if _ts_min_raw > 1e10 else 1.0
                            for _ha in _hist_coords:
                                _hist_coords[_ha]['ts_norm'] = [
                                    (t - _ts_min_raw) / _ts_scale
                                    for t in _hist_coords[_ha]['ts']
                                ]

                            # ── Timeline: janela ou jogo completo ─────────────
                            _all_ts_norm = sorted({
                                t for _hc in _hist_coords.values() for t in _hc['ts_norm']
                            })
                            _total_s_all = max(1.0, _all_ts_norm[-1] - _all_ts_norm[0])
                            # Persiste duração para o slider de início usar no próximo render
                            st.session_state[f'_hist_dur_{_hist_periodo_sel}'] = _total_s_all

                            _win_s_map = {
                                "Completo": None,
                                "10 min": 600,
                                "5 min":  300,
                                "3 min":  180,
                                "2 min":  120,
                                "1 min":   60,
                            }
                            _win_s = _win_s_map.get(_hist_window_sel)

                            if _win_s is not None:
                                # ── Janela de tempo real ───────────────────────
                                # Usa TODOS os pontos GPS na janela → sem downsample
                                _win_start_s = min(float(_hist_start_s),
                                                   max(0.0, _total_s_all - _win_s))
                                _win_end_s   = _win_start_s + _win_s

                                _frame_ts_raw = [
                                    t for t in _all_ts_norm
                                    if _win_start_s <= t <= _win_end_s
                                ]
                                # Proteção: máximo 3000 frames para não travar o browser
                                _MAX_WIN_FRAMES = 3000
                                if len(_frame_ts_raw) > _MAX_WIN_FRAMES:
                                    _st_w     = max(1, len(_frame_ts_raw) // _MAX_WIN_FRAMES)
                                    _frame_ts = _frame_ts_raw[::_st_w]
                                else:
                                    _frame_ts = _frame_ts_raw or _all_ts_norm[:1]

                                _n_frames = len(_frame_ts)
                                _dur_win_s = (
                                    (_frame_ts[-1] - _frame_ts[0])
                                    if _n_frames > 1 else float(_win_s)
                                )
                                # 1× = tempo real: cada frame dura exatamente o
                                # intervalo real que ele representa na partida
                                _real_ms_per_frame = (
                                    (_dur_win_s * 1000) / max(1, _n_frames - 1)
                                )
                                _frame_dur_ms = max(30, int(_real_ms_per_frame / _hist_speed))

                                _ws_m = int(_win_start_s // 60)
                                _ws_s = int(_win_start_s % 60)
                                _we_m = int(_win_end_s   // 60)
                                _we_s = int(_win_end_s   % 60)
                                st.caption(
                                    f"🪟 Janela {_ws_m:02d}:{_ws_s:02d}→{_we_m:02d}:{_we_s:02d}"
                                    f" · {_n_frames} frames"
                                    f" · {_real_ms_per_frame:.0f} ms reais/frame"
                                    f" · exibição: {_frame_dur_ms} ms/frame a {_hist_speed:.2g}×"
                                )
                            else:
                                # ── Jogo completo: downsample via quality slider ─
                                _step_fr  = max(1, len(_all_ts_norm) // _hist_max_frames)
                                _frame_ts = _all_ts_norm[::_step_fr]
                                _n_frames = len(_frame_ts)
                                _dur_all_s = max(1.0, _frame_ts[-1] - _frame_ts[0])
                                _real_ms_per_frame = (_dur_all_s * 1000) / max(1, _n_frames)
                                _frame_dur_ms = max(30, int(_real_ms_per_frame / _hist_speed))

                            # ── Figura base (campo) ────────────────────────
                            _fig_hist = desenhar_campo_futebol_bonito(
                                field_length=_hist_fl, field_width=_hist_fw,
                                title="🎬 História do Jogo"
                            )
                            _n_static = len(_fig_hist.data)

                            _hist_atl_list = list(_hist_coords.keys())
                            _n_ha = len(_hist_atl_list)

                            # Ghost: trajetória completa, ténue (estática)
                            for _i, _ha in enumerate(_hist_atl_list):
                                _hc = _hist_coords[_ha]
                                _col = _HIST_COLORS[_i % len(_HIST_COLORS)]
                                _fig_hist.add_trace(go.Scatter(
                                    x=_hc['xs'], y=_hc['ys'],
                                    mode='lines',
                                    line=dict(color=_col, width=1),
                                    opacity=0.12,
                                    name=_ha, showlegend=False, hoverinfo='skip',
                                ))

                            # Trail cinemático (broadcast): gradiente de opacidade e tamanho
                            for _i, _ha in enumerate(_hist_atl_list):
                                _col = _HIST_COLORS[_i % len(_HIST_COLORS)]
                                _fig_hist.add_trace(go.Scatter(
                                    x=[], y=[],
                                    mode='markers',
                                    marker=dict(color=[], size=[], opacity=1.0),
                                    name=_ha, showlegend=False, hoverinfo='skip',
                                ))

                            # Marker: posição atual animada (cor por velocidade, borda = cor do atleta)
                            for _i, _ha in enumerate(_hist_atl_list):
                                _hc = _hist_coords[_ha]
                                _col = _HIST_COLORS[_i % len(_HIST_COLORS)]
                                _pos_lbl = dados_posicao_por_periodo[_hist_periodo_sel].get(_ha, {}).get('posicao', '')
                                _eq_lbl  = dados_posicao_por_periodo[_hist_periodo_sel].get(_ha, {}).get('equipe', '')
                                _fig_hist.add_trace(go.Scatter(
                                    x=[_hc['xs'][0]], y=[_hc['ys'][0]],
                                    mode='markers+text',
                                    marker=dict(
                                        size=18,
                                        color='#64B5F6',        # azul = andando (padrão inicial)
                                        line=dict(color=_col, width=2.5),  # borda = identidade
                                        symbol='circle',
                                    ),
                                    text=[_ha.split(' ')[0]],
                                    textposition='top center',
                                    textfont=dict(size=9, color='white'),
                                    name=_ha,
                                    showlegend=True,
                                    hovertemplate=(
                                        f"<b>{_ha}</b><br>"
                                        f"Posição: {_pos_lbl}<br>"
                                        f"Equipe: {_eq_lbl}<extra></extra>"
                                    ),
                                ))

                            # Convex hull da equipa (começa vazio, atualizado por frame)
                            _fig_hist.add_trace(go.Scatter(
                                x=[], y=[],
                                mode='lines',
                                fill='toself',
                                fillcolor='rgba(255,255,255,0.06)',
                                line=dict(color='rgba(255,255,255,0.22)', width=1.5, dash='dot'),
                                name='Área equipa',
                                showlegend=True,
                                hoverinfo='skip',
                            ))
                            _hull_trace_idx = _n_static + 3 * _n_ha

                            # ── Índices dos traces animados ────────────────
                            # ghost  → n_static + i
                            # trail  → n_static + n_ha + i
                            # marker → n_static + 2*n_ha + i
                            # hull   → n_static + 3*n_ha

                            # Paleta de velocidade (precomputada)
                            def _vel_color(v):
                                if v < 7:   return '#64B5F6'  # andando — azul
                                if v < 14:  return '#66BB6A'  # trotando — verde
                                if v < 19:  return '#FFA726'  # correndo — laranja
                                return      '#EF5350'          # sprint   — vermelho

                            # ── Montar frames ──────────────────────────────
                            _frames_list = []
                            for _fi, _fts in enumerate(_frame_ts):
                                _frame_data   = []
                                _frame_traces = []
                                _hull_pts_x   = []
                                _hull_pts_y   = []
                                for _i, _ha in enumerate(_hist_atl_list):
                                    _hc = _hist_coords[_ha]
                                    _col = _HIST_COLORS[_i % len(_HIST_COLORS)]
                                    _ts_arr = np.array(_hc['ts_norm'])
                                    _xs_arr = np.array(_hc['xs'])
                                    _ys_arr = np.array(_hc['ys'])
                                    _vl_arr = np.array(_hc['vel'])

                                    # Índice mais próximo até _fts
                                    _cur_idx = int(np.searchsorted(_ts_arr, _fts, side='right')) - 1
                                    _cur_idx = max(0, min(_cur_idx, len(_ts_arr) - 1))

                                    # Trail: pontos nos últimos _hist_trail_s segundos
                                    _tr_start_ts = _fts - float(_hist_trail_s)
                                    _tr_idx = int(np.searchsorted(_ts_arr, _tr_start_ts, side='left'))
                                    _trail_x = _xs_arr[_tr_idx:_cur_idx + 1].tolist()
                                    _trail_y = _ys_arr[_tr_idx:_cur_idx + 1].tolist()

                                    _cur_x = float(_xs_arr[_cur_idx])
                                    _cur_y = float(_ys_arr[_cur_idx])
                                    _cur_v = float(_vl_arr[_cur_idx]) if _cur_idx < len(_vl_arr) else 0.0
                                    _hull_pts_x.append(_cur_x)
                                    _hull_pts_y.append(_cur_y)

                                    # Trail cinemático: gradiente broadcast (antigo=transparente, novo=opaco)
                                    _n_tr = max(1, len(_trail_x))
                                    _tr_r = int(_col.lstrip('#')[0:2], 16)
                                    _tr_g = int(_col.lstrip('#')[2:4], 16)
                                    _tr_b = int(_col.lstrip('#')[4:6], 16)
                                    _tr_alphas = np.linspace(0.04, 0.94, _n_tr)
                                    _tr_sizes  = np.linspace(2.0, 9.5, _n_tr)
                                    _tr_colors = [
                                        f'rgba({_tr_r},{_tr_g},{_tr_b},{a:.2f})'
                                        for a in _tr_alphas
                                    ]
                                    _frame_data.append(go.Scatter(
                                        x=_trail_x, y=_trail_y,
                                        mode='markers',
                                        marker=dict(
                                            color=_tr_colors,
                                            size=_tr_sizes.tolist(),
                                            opacity=1.0,
                                        ),
                                    ))
                                    _frame_traces.append(_n_static + _n_ha + _i)

                                    # Marker trace update — cor por velocidade, borda = identidade
                                    _frame_data.append(go.Scatter(
                                        x=[_cur_x], y=[_cur_y],
                                        mode='markers+text',
                                        marker=dict(
                                            size=18,
                                            color=_vel_color(_cur_v),
                                            line=dict(color=_col, width=2.5),
                                        ),
                                        text=[_ha.split(' ')[0]],
                                        textfont=dict(size=9, color='white'),
                                        textposition='top center',
                                        hovertemplate=(
                                            f"<b>{_ha}</b><br>"
                                            f"Vel: {_cur_v:.1f} km/h<extra></extra>"
                                        ),
                                    ))
                                    _frame_traces.append(_n_static + 2 * _n_ha + _i)

                                # Convex hull da equipa
                                if len(_hull_pts_x) >= 3:
                                    try:
                                        from scipy.spatial import ConvexHull as _CH
                                        _hpts = np.column_stack([_hull_pts_x, _hull_pts_y])
                                        _chull = _CH(_hpts)
                                        _hv = _chull.vertices.tolist() + [_chull.vertices[0]]
                                        _hx = [_hull_pts_x[j] for j in _hv]
                                        _hy = [_hull_pts_y[j] for j in _hv]
                                    except Exception:
                                        _hx, _hy = _hull_pts_x + [_hull_pts_x[0]], _hull_pts_y + [_hull_pts_y[0]]
                                    _frame_data.append(go.Scatter(x=_hx, y=_hy))
                                    _frame_traces.append(_hull_trace_idx)

                                _mins = int(_fts // 60)
                                _secs = int(_fts % 60)
                                _clock = f"{_mins:02d}:{_secs:02d}"
                                _frames_list.append(go.Frame(
                                    data=_frame_data,
                                    traces=_frame_traces,
                                    name=str(_fi),
                                    layout=go.Layout(annotations=[dict(
                                        x=0.02, y=0.97, xref='paper', yref='paper',
                                        text=f"⏱ {_clock}",
                                        showarrow=False,
                                        font=dict(size=22, color='white'),
                                        bgcolor='rgba(0,0,0,0.55)',
                                        bordercolor='rgba(255,255,255,0.6)',
                                        borderwidth=1, borderpad=5,
                                    )]),
                                ))

                            _fig_hist.frames = _frames_list

                            # ── Layout final ───────────────────────────────
                            _fig_hist.update_layout(
                                height=620,
                                plot_bgcolor='#0e1117',
                                paper_bgcolor='#0e1117',
                                font=dict(color='white'),
                                margin=dict(t=20, b=140, l=10, r=10),
                                xaxis=dict(
                                    range=[-4, _hist_fl + 4],
                                    showgrid=False, zeroline=False,
                                    showticklabels=False, fixedrange=True,
                                ),
                                yaxis=dict(
                                    range=[-4, _hist_fw + 4],
                                    showgrid=False, zeroline=False,
                                    showticklabels=False, fixedrange=True,
                                    scaleanchor='x', scaleratio=1,
                                ),
                                legend=dict(
                                    font=dict(color='white', size=10),
                                    bgcolor='rgba(30,30,30,0.85)',
                                    bordercolor='rgba(150,150,150,0.4)',
                                    borderwidth=1,
                                    x=1.01, y=1, xanchor='left',
                                ),
                                annotations=[dict(
                                    x=0.02, y=0.97, xref='paper', yref='paper',
                                    text="⏱ 00:00",
                                    showarrow=False,
                                    font=dict(size=22, color='white'),
                                    bgcolor='rgba(0,0,0,0.55)',
                                    bordercolor='rgba(255,255,255,0.6)',
                                    borderwidth=1, borderpad=5,
                                )],
                                updatemenus=[dict(
                                    type='buttons', showactive=False,
                                    direction='down',
                                    x=0.83, y=0.97,
                                    xanchor='center', yanchor='top',
                                    buttons=[
                                        dict(
                                            label="▶ Play",
                                            method="animate",
                                            args=[None, {
                                                "frame": {"duration": _frame_dur_ms, "redraw": True},
                                                "fromcurrent": True,
                                                "transition": {"duration": 0},
                                            }],
                                        ),
                                        dict(
                                            label="⏸ Pausar",
                                            method="animate",
                                            args=[[None], {
                                                "frame": {"duration": 0, "redraw": False},
                                                "mode": "immediate",
                                                "transition": {"duration": 0},
                                            }],
                                        ),
                                    ],
                                    font=dict(color='black'),
                                    bgcolor='rgba(200,200,200,0.9)',
                                    bordercolor='rgba(100,100,100,0.5)',
                                )],
                                sliders=[dict(
                                    active=0,
                                    steps=[dict(
                                        args=[[str(_fi)], {
                                            "frame": {"duration": _frame_dur_ms, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0},
                                        }],
                                        label=f"{int(_frame_ts[_fi] // 60):02d}:{int(_frame_ts[_fi] % 60):02d}",
                                        method="animate",
                                    ) for _fi in range(_n_frames)],
                                    x=0.05, len=0.9,
                                    pad={"b": 10, "t": 10},
                                    currentvalue=dict(
                                        prefix="⏱ ",
                                        font=dict(size=14, color='white'),
                                        visible=True,
                                        xanchor='center',
                                    ),
                                    transition=dict(duration=0),
                                    bgcolor='rgba(50,50,50,0.7)',
                                    bordercolor='rgba(120,120,120,0.5)',
                                    font=dict(color='white', size=9),
                                )],
                            )

                            st.plotly_chart(_fig_hist, use_container_width=True)

                            # ── Resumo estatístico ─────────────────────────
                            st.markdown("---")
                            st.markdown("### 📊 Resumo dos Atletas")
                            _hist_stats = []
                            for _i, _ha in enumerate(_hist_atl_list):
                                _hc   = _hist_coords[_ha]
                                _hd_m = dados_posicao_por_periodo[_hist_periodo_sel].get(_ha, {})
                                _col  = _HIST_COLORS[_i % len(_HIST_COLORS)]
                                _vmax = max(_hc['vel']) if _hc['vel'] else 0.0
                                _hxnp = np.array(_hc['xs'])
                                _hynp = np.array(_hc['ys'])
                                _hdist = float(np.sum(
                                    np.sqrt(np.diff(_hxnp)**2 + np.diff(_hynp)**2)
                                )) if len(_hxnp) > 1 else 0.0
                                _dur_s = (
                                    _hc['ts_norm'][-1] - _hc['ts_norm'][0]
                                    if len(_hc['ts_norm']) > 1 else 0.0
                                )
                                _hist_stats.append({
                                    '🎨': f'<span style="color:{_col}">■</span>',
                                    'Atleta': _ha,
                                    'Posição': _hd_m.get('posicao', '—'),
                                    'Equipe': _hd_m.get('equipe', '—'),
                                    'Distância (m)': round(_hdist),
                                    'Vel Máx (km/h)': round(_vmax, 1),
                                    'Duração (s)': round(_dur_s),
                                    'Pontos GPS': len(_hxnp),
                                })
                            if _hist_stats:
                                _df_hist = pd.DataFrame(_hist_stats).drop(columns=['🎨'])
                                st.dataframe(
                                    _df_hist, use_container_width=True, hide_index=True
                                )
                            st.caption(
                                f"⚙️ {_n_frames} frames · {_hist_speed:.2g}× velocidade · "
                                f"rastro {_hist_trail_s}s · campo {_hist_fl:.0f}×{_hist_fw:.0f}m"
                            )

                            # ══════════════════════════════════════════════════════
                            # ANÁLISES TÁTICAS COLETIVAS
                            # ══════════════════════════════════════════════════════
                            if _n_ha >= 2:
                                st.markdown("---")
                                st.markdown("## 🧠 Análises Táticas Coletivas")
                                st.caption("Baseado nas posições e velocidades sincronizadas de todos os atletas selecionados.")

                                # Interpolar todos os atletas para a grade _frame_ts
                                _ft_arr  = np.array(_frame_ts)
                                _sync_xs = np.zeros((_n_ha, _n_frames))
                                _sync_ys = np.zeros((_n_ha, _n_frames))
                                _sync_vl = np.zeros((_n_ha, _n_frames))
                                for _si, _sha in enumerate(_hist_atl_list):
                                    _shc  = _hist_coords[_sha]
                                    _s_ts = np.array(_shc['ts_norm'])
                                    _sync_xs[_si] = np.interp(_ft_arr, _s_ts, np.array(_shc['xs']))
                                    _sync_ys[_si] = np.interp(_ft_arr, _s_ts, np.array(_shc['ys']))
                                    _sync_vl[_si] = np.interp(_ft_arr, _s_ts, np.array(_shc['vel']))

                                # Métricas coletivas por frame
                                _ctr_x    = _sync_xs.mean(axis=0)
                                _ctr_y    = _sync_ys.mean(axis=0)
                                _width    = _sync_ys.max(axis=0) - _sync_ys.min(axis=0)
                                _depth    = _sync_xs.max(axis=0) - _sync_xs.min(axis=0)
                                _compact  = np.sqrt(_sync_xs.std(axis=0)**2 + _sync_ys.std(axis=0)**2)
                                _team_spd = _sync_vl.mean(axis=0)
                                _lat_asym = _ctr_y - _hist_fw / 2.0

                                _tac_tabs = st.tabs([
                                    "🏗️ Estrutura",
                                    "🗺️ Espaço",
                                    "⚡ Dinâmica",
                                ])

                                # ──────────────────────────────────────────────
                                # TAB ESTRUTURA  (1, 2, 3, 5)
                                # ──────────────────────────────────────────────
                                with _tac_tabs[0]:
                                    st.markdown("### 🏗️ Estrutura da Equipa")

                                    # 5 — Linhas Táticas
                                    st.markdown("#### 5 — Linhas Táticas (Defesa / Meio / Ataque)")

                                    # Sugestão automática por percentil 25 de X
                                    # (posição mais recuada típica, robusta a transições)
                                    _avg_x_atl  = np.percentile(_sync_xs, 25, axis=1)
                                    _sorted_idx = np.argsort(_avg_x_atl)
                                    _n3         = max(1, _n_ha // 3)
                                    _def_default = [_hist_atl_list[i] for i in _sorted_idx[:_n3]]
                                    _mid_default = [_hist_atl_list[i] for i in _sorted_idx[_n3:2*_n3]]
                                    _att_default = [_hist_atl_list[i] for i in _sorted_idx[2*_n3:]]

                                    # Seletor manual de linhas
                                    st.caption("💡 Atribuição automática por posição X média — ajuste conforme necessário.")
                                    _lcol1, _lcol2, _lcol3 = st.columns(3)
                                    with _lcol1:
                                        _def_sel = st.multiselect(
                                            "🔴 Defesa:", _hist_atl_list,
                                            default=_def_default, key="tac_linha_def"
                                        )
                                    with _lcol2:
                                        _mid_sel = st.multiselect(
                                            "🟡 Meio:", _hist_atl_list,
                                            default=_mid_default, key="tac_linha_mid"
                                        )
                                    with _lcol3:
                                        _att_sel = st.multiselect(
                                            "🟢 Ataque:", _hist_atl_list,
                                            default=_att_default, key="tac_linha_att"
                                        )

                                    _def_idx = [_hist_atl_list.index(a) for a in _def_sel if a in _hist_atl_list]
                                    _mid_idx = [_hist_atl_list.index(a) for a in _mid_sel if a in _hist_atl_list]
                                    _att_idx = [_hist_atl_list.index(a) for a in _att_sel if a in _hist_atl_list]
                                    _line_def = _sync_xs[_def_idx].mean(axis=0) if len(_def_idx) else np.zeros(_n_frames)
                                    _line_mid = _sync_xs[_mid_idx].mean(axis=0) if len(_mid_idx) else np.full(_n_frames, _hist_fl/2)
                                    _line_att = _sync_xs[_att_idx].mean(axis=0) if len(_att_idx) else np.full(_n_frames, _hist_fl)
                                    _dist_iL  = _line_att - _line_def
                                    _fig_ln = go.Figure()
                                    _fig_ln.add_trace(go.Scatter(x=list(_ft_arr), y=list(_line_def),
                                        name='Linha Def.', line=dict(color='#FF6B6B', width=2)))
                                    _fig_ln.add_trace(go.Scatter(x=list(_ft_arr), y=list(_line_mid),
                                        name='Linha Meio', line=dict(color='#FFEAA7', width=2)))
                                    _fig_ln.add_trace(go.Scatter(x=list(_ft_arr), y=list(_line_att),
                                        name='Linha Atq.', line=dict(color='#96CEB4', width=2)))
                                    _fig_ln.add_trace(go.Scatter(x=list(_ft_arr), y=list(_dist_iL),
                                        name='Dist. inter-linhas', line=dict(color='#DDA0DD', width=1.5, dash='dot')))
                                    _fig_ln.update_layout(
                                        xaxis_title='Tempo (s)', yaxis_title='Posição X (m)',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=290,
                                        legend=dict(font=dict(color='white')),
                                        margin=dict(t=10, b=30, l=40, r=10),
                                    )
                                    st.plotly_chart(_fig_ln, use_container_width=True)
                                    _def_n = ', '.join([_hist_atl_list[i].split(' ')[0] for i in _def_idx])
                                    _mid_n = ', '.join([_hist_atl_list[i].split(' ')[0] for i in _mid_idx])
                                    _att_n = ', '.join([_hist_atl_list[i].split(' ')[0] for i in _att_idx])
                                    st.caption(f"🔴 Defesa: {_def_n}  |  🟡 Meio: {_mid_n}  |  🟢 Ataque: {_att_n}")
                                    _ml1, _ml2, _ml3 = st.columns(3)
                                    _ml1.metric("Altura linha def.", f"{float(_line_def.mean()):.1f} m")
                                    _ml2.metric("Dist. inter-linhas", f"{float(_dist_iL.mean()):.1f} m")
                                    _ml3.metric("Altura linha atq.", f"{float(_line_att.mean()):.1f} m")

                                # ──────────────────────────────────────────────
                                # TAB ESPAÇO  (7, 8, 9)
                                # ──────────────────────────────────────────────
                                with _tac_tabs[1]:
                                    st.markdown("### 🗺️ Controlo do Espaço")

                                    # 9 — Gaps Táticos (Delaunay)
                                    st.markdown("#### 9 — Espaços Descobertos (Gaps Táticos)")
                                    _gap_thr = st.slider("Threshold de gap (m²):", 50, 500, 150, step=25, key="tac_gap")
                                    try:
                                        from scipy.spatial import Delaunay as _Del
                                        _gap_frames = [_n_frames//4, _n_frames//2, 3*_n_frames//4]
                                        _gcols = st.columns(len(_gap_frames))
                                        for _gi, _gframe in enumerate(_gap_frames):
                                            _gx = _sync_xs[:, _gframe]
                                            _gy = _sync_ys[:, _gframe]
                                            _gts_lbl = float(_frame_ts[_gframe])
                                            with _gcols[_gi]:
                                                _fig_gap = desenhar_campo_futebol_bonito(
                                                    field_length=_hist_fl, field_width=_hist_fw,
                                                    title=f"{int(_gts_lbl//60):02d}:{int(_gts_lbl%60):02d}"
                                                )
                                                if len(_gx) >= 3:
                                                    _pts_g = np.column_stack([_gx, _gy])
                                                    _tri_g = _Del(_pts_g)
                                                    for _s in _tri_g.simplices:
                                                        _tx = _pts_g[_s,0].tolist() + [float(_pts_g[_s[0],0])]
                                                        _ty = _pts_g[_s,1].tolist() + [float(_pts_g[_s[0],1])]
                                                        _area = abs(
                                                            (_pts_g[_s[1],0]-_pts_g[_s[0],0])*(_pts_g[_s[2],1]-_pts_g[_s[0],1]) -
                                                            (_pts_g[_s[2],0]-_pts_g[_s[0],0])*(_pts_g[_s[1],1]-_pts_g[_s[0],1])
                                                        ) / 2.0
                                                        _clr = 'rgba(255,50,50,0.45)' if _area > _gap_thr else 'rgba(50,200,50,0.15)'
                                                        _fig_gap.add_trace(go.Scatter(
                                                            x=_tx, y=_ty, fill='toself',
                                                            fillcolor=_clr,
                                                            line=dict(color='rgba(255,255,255,0.25)', width=0.5),
                                                            showlegend=False, hoverinfo='skip',
                                                        ))
                                                _fig_gap.add_trace(go.Scatter(
                                                    x=list(_gx), y=list(_gy), mode='markers',
                                                    marker=dict(size=10, color='white', line=dict(color='black', width=1)),
                                                    showlegend=False, hoverinfo='skip',
                                                ))
                                                _fig_gap.update_layout(
                                                    height=260,
                                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                                    font=dict(color='white', size=9),
                                                    margin=dict(t=35, b=5, l=5, r=5),
                                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, _hist_fl+3]),
                                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, _hist_fw+3], scaleanchor='x', scaleratio=1),
                                                )
                                                st.plotly_chart(_fig_gap, use_container_width=True)
                                        st.caption("🔴 Vermelho = gap > threshold  ·  🟢 Verde = cobertura adequada")
                                    except Exception as _eg:
                                        st.warning(f"scipy indisponível para gaps: {_eg}")

                                # ──────────────────────────────────────────────
                                # TAB DINÂMICA  (11, 12, 13, 14, 15)
                                # ──────────────────────────────────────────────
                                with _tac_tabs[2]:
                                    st.markdown("### ⚡ Dinâmica Coletiva")

                                    # 11 — Pressing
                                    st.markdown("#### 11 — Índice de Pressing Coletivo")
                                    with st.expander("ℹ️ Como é calculado?"):
                                        st.markdown("""
**Metodologia:**
1. A cada frame sincronizado, conta-se quantos atletas possuem velocidade ≥ **Vel. mín.** (ex: 14 km/h) — atletas em alta intensidade (HSR)
2. Um **momento de pressing coletivo** é identificado quando o número de atletas em HSR atinge ou supera o limiar de **Atletas simultâneos**
3. O gráfico exibe essa contagem ao longo do tempo; a linha tracejada amarela marca o limiar configurado
4. **Tempo em pressing (%)** = proporção de frames onde o pressing coletivo foi ativado
5. **Duração total** = soma dos frames em pressing × intervalo real entre frames (em segundos)
6. **Média atletas HSR** = média de atletas acima da velocidade mínima em todos os frames

> 💡 Pressing coletivo requer simultaneidade: um único atleta em alta velocidade não ativa o índice.
                                        """)
                                    _pr_v = st.slider("Vel. mín. de pressing (km/h):", 10, 21, 14, key="tac_prv")
                                    _pr_n = st.slider("Atletas simultâneos (mín.):", 2, min(_n_ha,8), min(3,_n_ha), key="tac_prn")
                                    _press_idx = (_sync_vl >= _pr_v).sum(axis=0).astype(float)
                                    _press_bin = (_press_idx >= _pr_n).astype(float)
                                    _fig_pr = go.Figure()
                                    _fig_pr.add_trace(go.Scatter(
                                        x=list(_ft_arr), y=list(_press_idx),
                                        name='Atletas em alta vel.',
                                        line=dict(color='#FF6B6B', width=1.5),
                                        fill='tozeroy', fillcolor='rgba(255,107,107,0.2)',
                                    ))
                                    _fig_pr.add_hline(y=_pr_n, line_dash='dash', line_color='yellow',
                                                      annotation_text=f"Threshold ({_pr_n})", annotation_font_color='yellow')
                                    _fig_pr.update_layout(
                                        xaxis_title='Tempo (s)', yaxis_title='Nº Atletas',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=255,
                                        margin=dict(t=15, b=30, l=40, r=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_pr, use_container_width=True)
                                    _dt_frame = float(np.diff(_ft_arr).mean()) if _n_frames > 1 else 1.0
                                    _pr1, _pr2, _pr3 = st.columns(3)
                                    _pr1.metric("Tempo em pressing", f"{float(_press_bin.mean())*100:.1f}%")
                                    _pr2.metric("Duração total", f"{float(_press_bin.sum())*_dt_frame:.0f} s")
                                    _pr3.metric("Média atletas HSR", f"{float(_press_idx.mean()):.1f}")

                                    st.markdown("---")

                                    # 12 — Transições
                                    st.markdown("#### 12 — Detecção de Transições")
                                    with st.expander("ℹ️ Como é calculado?"):
                                        st.markdown("""
**Metodologia:**
1. Calcula-se o **centróide da equipa** em X (média da posição longitudinal de todos os atletas) a cada frame
2. Aplica-se o **gradiente temporal** sobre o centróide X → obtém-se a velocidade de deslocamento coletivo (m/s), positiva para avanço e negativa para recuo
3. Suavização por **média móvel de 5 frames** para eliminar oscilações de GPS
4. **Transição ofensiva**: o centróide avança com velocidade > Threshold → equipa progride coletivamente no campo
5. **Transição defensiva**: o centróide recua com velocidade < −Threshold → equipa recua coletivamente
6. Contagem de episódios: cada cruzamento do limiar que se mantém por pelo menos um frame conta como uma transição

> 💡 O threshold controla a sensibilidade: valores baixos (0.5 m/s) detectam micro-transições; valores altos (3+ m/s) capturam apenas transições explosivas.
                                        """)
                                    _dctr = np.gradient(_ctr_x, _ft_arr)
                                    _sm   = np.convolve(_dctr, np.ones(5)/5, mode='same')
                                    _tr_thr = st.slider("Threshold (m/s):", 0.5, 5.0, 1.5, step=0.5, key="tac_trthr")
                                    _fig_tr = go.Figure()
                                    _fig_tr.add_trace(go.Scatter(
                                        x=list(_ft_arr), y=list(_sm),
                                        line=dict(color='#4ECDC4', width=2), name='Vel. centróide X',
                                    ))
                                    _smmax = float(max(abs(_sm.max()), abs(_sm.min()), _tr_thr) + 0.1)
                                    _fig_tr.add_hrect(y0=_tr_thr, y1=_smmax,
                                        fillcolor='rgba(150,206,180,0.2)', line_width=0,
                                        annotation_text='Transição Ofensiva', annotation_font_color='#96CEB4')
                                    _fig_tr.add_hrect(y0=-_smmax, y1=-_tr_thr,
                                        fillcolor='rgba(255,107,107,0.2)', line_width=0,
                                        annotation_text='Transição Defensiva', annotation_font_color='#FF6B6B')
                                    _fig_tr.update_layout(
                                        xaxis_title='Tempo (s)', yaxis_title='m/s',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=255,
                                        margin=dict(t=20, b=30, l=50, r=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_tr, use_container_width=True)
                                    _t_off = (_sm > _tr_thr).astype(int)
                                    _t_def = (_sm < -_tr_thr).astype(int)
                                    _n_toff = int(np.diff(np.concatenate([[0],_t_off,[0]]) ).clip(0).sum())
                                    _n_tdef = int(np.diff(np.concatenate([[0],_t_def,[0]]) ).clip(0).sum())
                                    _tr1, _tr2, _tr3 = st.columns(3)
                                    _tr1.metric("Transições ofensivas", f"{_n_toff}")
                                    _tr2.metric("Transições defensivas", f"{_n_tdef}")
                                    _tr3.metric("Ratio O/D", f"{_n_toff/_n_tdef:.2f}" if _n_tdef else "—")

                                    st.markdown("---")

            # ══════════════════════════════════════════════════════════════
            # WCS: WORST-CASE SCENARIO (sub-tab Campo & GPS)
            # ══════════════════════════════════════════════════════════════
            with _sub_campo[2]:
                st.subheader("⚡ Worst-Case Scenario")
                st.caption(
                    "Identifica a **janela temporal de maior exigência física** de cada atleta "
                    "buscando o pior cenário em todos os períodos da atividade. "
                    "Base para prescrição de cargas de treino acima do jogo. "
                    "_(Delaney et al., 2018; Martín-García et al., 2018)_"
                )

                if _ok_ld == 0 or not dados_posicao_por_periodo:
                    st.info("Carregue os dados para usar a análise de Worst-Case Scenario.")
                else:
                    # ── Controles ──────────────────────────────────────────────────
                    _wcs2_c1, _wcs2_c2 = st.columns([1, 2])
                    with _wcs2_c1:
                        _wcs2_min = st.slider(
                            "⏱️ Janela temporal (min)", 1, 15, 5,
                            key="wcs2_janela",
                            help="Duração da janela rolante para identificar o pior cenário"
                        )
                    with _wcs2_c2:
                        _wcs2_metric_opts = [
                            "Distância (m)",
                            "Dist. >14 km/h (m)",
                            "Dist. >19 km/h (m)  — Alta Intensidade",
                            "Dist. >24 km/h (m)  — Sprint",
                            "Velocidade Máx (km/h)",
                            "PlayerLoad",
                            "Acelerações >2 m/s² (n)",
                            "Acelerações >3 m/s² (n)",
                            "Desacelerações <-2 m/s² (n)",
                            "Desacelerações <-3 m/s² (n)",
                        ]
                        _wcs2_metric = st.selectbox(
                            "📊 Variável", _wcs2_metric_opts,
                            key="wcs2_metric_sel",
                            help="Métrica para identificar a janela de maior exigência"
                        )

                    _wcs2_hz = 10.0
                    _wcs2_n  = int(_wcs2_min * 60 * _wcs2_hz)

                    # ── Cálculo do WCS por atleta ───────────────────────────────────
                    _wcs2_rows = []
                    _wcs2_segs = {}  # atleta → {xn, yn, vel} para animação

                    _wcs2_periodos = [
                        k for k in dados_posicao_por_periodo
                        if k != _CHAVE_COMBINADO
                    ]
                    _wcs2_athletes = sorted(set(
                        a for _pn in _wcs2_periodos
                        for a in dados_posicao_por_periodo.get(_pn, {}).keys()
                    ))

                    def _wcalc_wcs(_sv, _n, _is_vm):
                        if len(_sv) < max(_n, 2):
                            return 0.0
                        if _is_vm:
                            from collections import deque as _DqW
                            _dqW = _DqW(); _bvW = -1.0
                            for _iW in range(len(_sv)):
                                while _dqW and _sv[_dqW[-1]] <= _sv[_iW]:
                                    _dqW.pop()
                                _dqW.append(_iW)
                                if _dqW[0] <= _iW - _n:
                                    _dqW.popleft()
                                if _iW >= _n - 1:
                                    _cW = _sv[_dqW[0]]
                                    if _cW > _bvW:
                                        _bvW = _cW
                            return _bvW
                        else:
                            _cs = sum(_sv[:_n]); _bv = _cs
                            for _i in range(1, len(_sv) - _n + 1):
                                _cs += _sv[_i + _n - 1] - _sv[_i - 1]
                                if _cs > _bv:
                                    _bv = _cs
                            return _bv

                    for _wa in _wcs2_athletes:
                        _wx, _wy, _wv, _wac, _wts, _wper = [], [], [], [], [], []
                        for _pn in _wcs2_periodos:
                            _da  = dados_posicao_por_periodo.get(_pn, {}).get(_wa, {})
                            _xs  = _da.get('xs', [])
                            _ys  = _da.get('ys', [])
                            _vl  = _da.get('vel', [])         # km/h
                            _ac  = _da.get('acc', [0.0] * len(_xs))
                            _ts  = _da.get('ts_pos', [0.0] * len(_xs))
                            _nn  = min(len(_xs), len(_ys), len(_vl))
                            if _nn > 0:
                                _wx  += list(_xs[:_nn])
                                _wy  += list(_ys[:_nn])
                                _wv  += list(_vl[:_nn])
                                _wac += list(_ac[:_nn])
                                _wts += list(_ts[:_nn])
                                _wper += [_pn] * _nn

                        if len(_wx) < max(_wcs2_n, 2):
                            continue

                        # Valor por amostra para a métrica selecionada
                        _Hz = _wcs2_hz
                        _m  = _wcs2_metric
                        if _m == "Distância (m)":
                            _sv = [v / (3.6 * _Hz) for v in _wv]
                        elif _m == "Dist. >14 km/h (m)":
                            _sv = [v / (3.6 * _Hz) if v > 14 else 0.0 for v in _wv]
                        elif "19" in _m:
                            _sv = [v / (3.6 * _Hz) if v > 19 else 0.0 for v in _wv]
                        elif "24" in _m:
                            _sv = [v / (3.6 * _Hz) if v > 24 else 0.0 for v in _wv]
                        elif _m == "Velocidade Máx (km/h)":
                            _sv = list(_wv)   # rolling max — tratado abaixo
                        elif _m == "PlayerLoad":
                            _pl_raw = []
                            for _ppn in _wcs2_periodos:
                                _pl_raw += dados_sensor_por_atleta_por_periodo.get(_ppn, {}).get(_wa, [])
                            if len(_pl_raw) >= len(_wx):
                                _sv = [float(p.get('pl') or 0) for p in _pl_raw[:len(_wx)]]
                            else:
                                _sv = [float(p.get('pl') or 0) for p in _pl_raw] + [0.0] * (len(_wx) - len(_pl_raw))
                        elif ">2" in _m and "Acel" in _m:
                            _sv = [1.0 if a > 2 else 0.0 for a in _wac]
                        elif ">3" in _m and "Acel" in _m:
                            _sv = [1.0 if a > 3 else 0.0 for a in _wac]
                        elif "<-2" in _m:
                            _sv = [1.0 if a < -2 else 0.0 for a in _wac]
                        elif "<-3" in _m:
                            _sv = [1.0 if a < -3 else 0.0 for a in _wac]
                        else:
                            _sv = [v / (3.6 * _Hz) for v in _wv]

                        # Multi-janela (1, 3, 5 min) para comparativo na tabela
                        _is_vm2 = (_m == "Velocidade Máx (km/h)")
                        _mw_vals = {}
                        for _mwname, _mwmin in [('1 min', 1), ('3 min', 3), ('5 min', 5)]:
                            _mwn = int(_mwmin * 60 * _wcs2_hz)
                            if _mwn != _wcs2_n and len(_sv) >= max(_mwn, 2):
                                _mw_vals[_mwname] = round(_wcalc_wcs(_sv, _mwn, _is_vm2), 1)

                        # Janela rolante
                        if _m == "Velocidade Máx (km/h)":
                            from collections import deque as _Dq
                            _dq3 = _Dq()
                            _bv3, _bsi3, _bei3 = -1.0, 0, _wcs2_n
                            for _i3 in range(len(_sv)):
                                while _dq3 and _sv[_dq3[-1]] <= _sv[_i3]:
                                    _dq3.pop()
                                _dq3.append(_i3)
                                if _dq3[0] <= _i3 - _wcs2_n:
                                    _dq3.popleft()
                                if _i3 >= _wcs2_n - 1:
                                    _c3 = _sv[_dq3[0]]
                                    if _c3 > _bv3:
                                        _bv3  = _c3
                                        _bei3 = _i3 + 1
                                        _bsi3 = _bei3 - _wcs2_n
                            _best_val2, _best_si2, _best_ei2 = _bv3, _bsi3, _bei3
                        else:
                            _csum = sum(_sv[:_wcs2_n])
                            _best_val2, _best_si2, _best_ei2 = _csum, 0, _wcs2_n
                            for _i3 in range(1, len(_sv) - _wcs2_n + 1):
                                _csum += _sv[_i3 + _wcs2_n - 1] - _sv[_i3 - 1]
                                if _csum > _best_val2:
                                    _best_val2 = _csum
                                    _best_si2  = _i3
                                    _best_ei2  = _i3 + _wcs2_n

                        # Timestamps
                        _ts0 = _wts[_best_si2] if _best_si2 < len(_wts) else 0
                        _ts1 = _wts[min(_best_ei2 - 1, len(_wts) - 1)] if _wts else 0
                        try:
                            from datetime import datetime as _dtc
                            _ini_str = _dtc.fromtimestamp(float(_ts0)).strftime('%H:%M:%S') if float(_ts0) > 1e6 else f"{int(_best_si2/_Hz//60):02d}:{int(_best_si2/_Hz%60):02d}"
                            _fim_str = _dtc.fromtimestamp(float(_ts1)).strftime('%H:%M:%S') if float(_ts1) > 1e6 else f"{int(_best_ei2/_Hz//60):02d}:{int(_best_ei2/_Hz%60):02d}"
                        except Exception:
                            _ini_str = f"{int(_best_si2/_Hz//60):02d}:{int(_best_si2/_Hz%60):02d}"
                            _fim_str = f"{int(_best_ei2/_Hz//60):02d}:{int(_best_ei2/_Hz%60):02d}"

                        # Posição do atleta
                        _posicao2 = '—'
                        for _rpn2, _rpl2 in resultados_por_periodo.items():
                            for _rrow2 in _rpl2:
                                if str(_rrow2.get('Atleta', '')) == _wa:
                                    _posicao2 = str(_rrow2.get('Posição', '—'))
                                    break
                            if _posicao2 != '—':
                                break

                        _row_d = {
                            '_atl_orig':   _wa,
                            'Atleta':      _wa,
                            'Posição':     _posicao2,
                            'Período':     _wper[_best_si2] if _best_si2 < len(_wper) else '—',
                            _wcs2_metric:  round(_best_val2, 1),
                            'Início':      _ini_str,
                            'Fim':         _fim_str,
                        }
                        _row_d.update(_mw_vals)
                        _wcs2_rows.append(_row_d)
                        _wcs2_segs[_wa] = {
                            'xn':  _wx[_best_si2:_best_ei2],
                            'yn':  _wy[_best_si2:_best_ei2],
                            'vel': _wv[_best_si2:_best_ei2],
                        }

                    _wcs2_rows.sort(key=lambda r: r.get(_wcs2_metric, 0), reverse=True)
                    _rank_icons = ['🔴', '🟠', '🟡']   # vermelho = maior carga/fadiga
                    for _ri2, _wr2 in enumerate(_wcs2_rows):
                        _wr2['#'] = _rank_icons[_ri2] if _ri2 < 3 else f'#{_ri2 + 1}'

                    if not _wcs2_rows:
                        st.warning(
                            "Dados insuficientes para calcular WCS com essa janela temporal. "
                            "Reduza a janela ou carregue mais períodos."
                        )
                    else:
                        # % do Máximo do grupo
                        _wcs2_top = _wcs2_rows[0].get(_wcs2_metric, 0) or 1.0
                        _wcs2_avg = sum(r.get(_wcs2_metric, 0) for r in _wcs2_rows) / len(_wcs2_rows)
                        for _wr in _wcs2_rows:
                            _wr['% Máx Grupo'] = round(_wr.get(_wcs2_metric, 0) / _wcs2_top * 100, 1)

                        # KPIs resumo
                        _wk1, _wk2, _wk3, _wk4 = st.columns(4)
                        _wk1.metric(
                            "🔴 Maior Fadiga",
                            _wcs2_rows[0]['Atleta'],
                            f"{_wcs2_top:.1f} — {_wcs2_rows[0].get('Período', '—')}",
                        )
                        _wk2.metric(
                            "📊 Média Grupo",
                            f"{_wcs2_avg:.1f}",
                            f"Δ {_wcs2_top - _wcs2_avg:.1f} vs líder",
                        )
                        _wk3.metric("👥 Atletas", str(len(_wcs2_rows)))
                        _wk4.metric(
                            "⏱️ Janela",
                            f"{_wcs2_min} min",
                            f"{_wcs2_rows[0].get('Início','—')} → {_wcs2_rows[0].get('Fim','—')}",
                        )

                        st.markdown("---")

                        # Tabela
                        _df_all_tmp = pd.DataFrame(_wcs2_rows)
                        _mw_avail   = [c for c in ['1 min', '3 min', '5 min']
                                       if c in _df_all_tmp.columns
                                       and _df_all_tmp[c].notna().any()]
                        _wcs2_col_order = (
                            ['#', 'Atleta', 'Posição', 'Período',
                             _wcs2_metric, '% Máx Grupo']
                            + _mw_avail
                            + ['Início', 'Fim']
                        )
                        _wcs2_col_order = [c for c in _wcs2_col_order if c in _df_all_tmp.columns]
                        _df_wcs2 = _df_all_tmp[_wcs2_col_order]

                        _col_cfg_wcs = {
                            '% Máx Grupo': st.column_config.ProgressColumn(
                                '% Máx', min_value=0, max_value=100, format='%.1f%%'
                            ),
                        }
                        _wcs2_evt = st.dataframe(
                            _df_wcs2,
                            use_container_width=True,
                            hide_index=True,
                            on_select='rerun',
                            selection_mode='single-row',
                            key='wcs2_table_sel',
                            column_config=_col_cfg_wcs,
                        )
                        st.caption(
                            "💡 Clique em uma linha para visualizar o percurso animado no campo abaixo. "
                            "As colunas 1/3/5 min mostram o valor WCS para janelas fixas de comparação."
                        )

                        # ── Animação no campo ao selecionar linha ──────────────────
                        _wcs2_sel = (_wcs2_evt.selection.rows
                                     if _wcs2_evt.selection else [])
                        if _wcs2_sel:
                            _wsel_atl = _wcs2_rows[_wcs2_sel[0]].get(
                                '_atl_orig', _wcs2_rows[_wcs2_sel[0]]['Atleta']
                            )
                            _wsel_row = _wcs2_rows[_wcs2_sel[0]]
                            _wseg     = _wcs2_segs.get(_wsel_atl, {})
                            _wcs2_xn  = _wseg.get('xn', [])
                            _wcs2_yn  = _wseg.get('yn', [])
                            _wcs2_vel = _wseg.get('vel', [])

                            if len(_wcs2_xn) >= 2:
                                st.markdown(f"### 🏃 {_wsel_atl} — WCS {_wcs2_min} min")
                                _wval_str = f"{_wsel_row.get(_wcs2_metric, 0):.1f}"
                                _wm1, _wm2, _wm3 = st.columns(3)
                                _wm1.metric(_wcs2_metric, _wval_str)
                                _wm2.metric("⏰ Início",  _wsel_row.get('Início', '—'))
                                _wm3.metric("🏁 Fim",     _wsel_row.get('Fim',    '—'))

                                # Campo config
                                _wcs2_cfg = None
                                for _hk2 in list(st.session_state.keys()):
                                    if (_hk2.startswith("campo_cfg__")
                                            and isinstance(st.session_state[_hk2], dict)
                                            and 'fl' in st.session_state[_hk2]):
                                        _wcs2_cfg = st.session_state[_hk2]
                                        break
                                _wcs2_fl = float(_wcs2_cfg.get('fl', 105)) if _wcs2_cfg else 105.0
                                _wcs2_fw = float(_wcs2_cfg.get('fw', 68))  if _wcs2_cfg else 68.0

                                def _vc_wcs2(v):
                                    if v < 7:  return '#2196F3'
                                    if v < 14: return '#4CAF50'
                                    if v < 19: return '#FFEB3B'
                                    if v < 24: return '#FF9800'
                                    return '#F44336'

                                _fig_wcs2 = desenhar_campo_futebol_bonito(
                                    field_length=_wcs2_fl,
                                    field_width=_wcs2_fw,
                                    title=(
                                        f"WCS {_wcs2_min} min — {_wsel_atl}  |  "
                                        f"{_wsel_row.get('Início','—')} → {_wsel_row.get('Fim','—')}"
                                    )
                                )
                                _n_base_wcs2 = len(_fig_wcs2.data)

                                # Downsampling
                                _nf2   = len(_wcs2_xn)
                                _step2 = max(1, _nf2 // 120)
                                _fr2   = list(range(0, _nf2, _step2))
                                if _fr2[-1] != _nf2 - 1:
                                    _fr2.append(_nf2 - 1)

                                # Traces estáticos: início / fim
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=[_wcs2_xn[0]], y=[_wcs2_yn[0]],
                                    mode='markers+text',
                                    marker=dict(size=13, color='#4CAF50', symbol='circle',
                                                line=dict(color='white', width=2)),
                                    text=['▶'], textposition='top center',
                                    textfont=dict(color='#4CAF50', size=11),
                                    name='Início', showlegend=False, hoverinfo='skip'
                                ))
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=[_wcs2_xn[-1]], y=[_wcs2_yn[-1]],
                                    mode='markers+text',
                                    marker=dict(size=13, color='#F44336', symbol='x',
                                                line=dict(color='white', width=2)),
                                    text=['■'], textposition='top center',
                                    textfont=dict(color='#F44336', size=11),
                                    name='Fim', showlegend=False, hoverinfo='skip'
                                ))
                                # Traço animado
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=[_wcs2_xn[0]], y=[_wcs2_yn[0]], mode='lines',
                                    line=dict(color=_vc_wcs2(_wcs2_vel[0] if _wcs2_vel else 0), width=4),
                                    name='Percurso', showlegend=False
                                ))
                                # Dot animado
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=[_wcs2_xn[0]], y=[_wcs2_yn[0]], mode='markers',
                                    marker=dict(
                                        size=18,
                                        color=_vc_wcs2(_wcs2_vel[0] if _wcs2_vel else 0),
                                        symbol='circle',
                                        line=dict(color='white', width=3)
                                    ),
                                    name='Posição', showlegend=False
                                ))

                                _idx_t2 = _n_base_wcs2 + 2
                                _idx_d2 = _n_base_wcs2 + 3

                                # Frames
                                _frames2 = []
                                for _fk2 in _fr2:
                                    _ds3  = (_fk2 / max(_nf2 - 1, 1)) * _wcs2_min * 60
                                    _dm3  = int(_ds3 // 60)
                                    _dsr3 = int(_ds3 % 60)
                                    _v3   = float(_wcs2_vel[_fk2]) if _fk2 < len(_wcs2_vel) else 0.0
                                    _c3   = _vc_wcs2(_v3)
                                    _frames2.append(go.Frame(
                                        data=[
                                            go.Scatter(
                                                x=_wcs2_xn[:_fk2 + 1],
                                                y=_wcs2_yn[:_fk2 + 1],
                                                mode='lines',
                                                line=dict(color=_c3, width=4),
                                            ),
                                            go.Scatter(
                                                x=[_wcs2_xn[_fk2]],
                                                y=[_wcs2_yn[_fk2]],
                                                mode='markers',
                                                marker=dict(
                                                    size=18, color=_c3,
                                                    symbol='circle',
                                                    line=dict(color='white', width=3)
                                                ),
                                            ),
                                        ],
                                        traces=[_idx_t2, _idx_d2],
                                        name=str(_fk2),
                                        layout=go.Layout(title=dict(
                                            text=(
                                                f'WCS {_wcs2_min} min — {_wsel_atl} | '
                                                f'⏱️ {_dm3}:{_dsr3:02d} / {_wcs2_min}:00'
                                                f'   |   💨 {_v3:.1f} km/h'
                                            ),
                                            font=dict(color='white', size=12)
                                        ))
                                    ))

                                _fig_wcs2.frames = _frames2

                                _sliders_wcs2 = [{
                                    'steps': [
                                        {
                                            'args': [[str(_fk2)],
                                                     {'frame': {'duration': 0, 'redraw': True},
                                                      'mode': 'immediate'}],
                                            'label': '',
                                            'method': 'animate',
                                        }
                                        for _fk2 in _fr2
                                    ],
                                    'transition': {'duration': 0},
                                    'x': 0.05, 'len': 0.90,
                                    'currentvalue': {'visible': False},
                                    'bgcolor': '#1e3a5f',
                                    'bordercolor': '#2196F3',
                                    'tickcolor': 'white',
                                    'font': {'color': 'white', 'size': 9},
                                }]
                                _fig_wcs2.update_layout(
                                    height=560,
                                    updatemenus=[{
                                        'type': 'buttons',
                                        'showactive': False,
                                        'y': -0.08, 'x': 0.5,
                                        'xanchor': 'center', 'yanchor': 'top',
                                        'buttons': [
                                            {
                                                'label': '▶ Play',
                                                'method': 'animate',
                                                'args': [None, {
                                                    'frame': {'duration': 60, 'redraw': True},
                                                    'fromcurrent': True,
                                                    'transition': {'duration': 0},
                                                }],
                                            },
                                            {
                                                'label': '⏸ Pause',
                                                'method': 'animate',
                                                'args': [[None], {
                                                    'frame': {'duration': 0, 'redraw': False},
                                                    'mode': 'immediate',
                                                    'transition': {'duration': 0},
                                                }],
                                            },
                                        ],
                                        'font': {'color': 'white'},
                                        'bgcolor': '#1e3a5f',
                                        'bordercolor': '#2196F3',
                                    }],
                                    sliders=_sliders_wcs2,
                                    margin=dict(b=80),
                                )
                                st.plotly_chart(_fig_wcs2, use_container_width=True)
                            else:
                                st.info("Dados GPS insuficientes para animação deste atleta.")

                        # ── Pior Momento Posicional Coletivo ───────────────────────
                        st.markdown("---")
                        with st.expander("🗺️ Pior Momento Posicional Coletivo", expanded=False):
                            st.caption(
                                "Identifica o **instante crítico** em que o grupo apresentou a maior "
                                "exigência posicional coletiva no período selecionado — "
                                "revelando padrões de organização ou desorganização tática. "
                                "Selecione o período e o critério de 'pior momento'."
                            )
                            if not _wcs2_periodos:
                                st.info("Nenhum período disponível.")
                            else:
                                _colc1, _colc2 = st.columns(2)
                                with _colc1:
                                    _col_per_sel = st.selectbox(
                                        "📅 Período",
                                        _wcs2_periodos,
                                        key="wcol_per_sel",
                                        help="Período para a análise coletiva",
                                    )
                                with _colc2:
                                    _col_metric_opts = [
                                        "🔴 Maior Dispersão  (time mais espalhado)",
                                        "🔵 Maior Compactidade  (time mais compacto)",
                                        "📏 Maior Amplitude  (largura máxima)",
                                        "📐 Maior Profundidade  (comprimento máximo)",
                                        "⚡ Maior Espaço entre Linhas  (gap def-ata)",
                                    ]
                                    _col_metric = st.selectbox(
                                        "📊 Métrica Coletiva",
                                        _col_metric_opts,
                                        key="wcol_metric_sel",
                                        help="Critério do 'pior momento' coletivo",
                                    )

                                # Montar dados do período selecionado
                                _col_period_data = dados_posicao_por_periodo.get(_col_per_sel, {})
                                _col_athl_map = {}
                                for _ca, _cad in _col_period_data.items():
                                    _cxs = list(_cad.get('xs', []))
                                    _cys = list(_cad.get('ys', []))
                                    _cvs = list(_cad.get('vel', []))
                                    _cts = list(_cad.get('ts_pos', []))
                                    _nn2 = min(len(_cxs), len(_cys))
                                    if _nn2 > 1:
                                        _col_athl_map[_ca] = {
                                            'xs':  _cxs[:_nn2],
                                            'ys':  _cys[:_nn2],
                                            'vel': _cvs[:_nn2] if len(_cvs) >= _nn2 else [0.0] * _nn2,
                                            'ts':  _cts[:_nn2] if len(_cts) >= _nn2 else [0.0] * _nn2,
                                            'pos': str(_cad.get('posicao', '—')),
                                        }

                                if len(_col_athl_map) < 2:
                                    st.info("Poucos atletas com GPS neste período para análise coletiva.")
                                else:
                                    _col_athletes = list(_col_athl_map.keys())
                                    _col_n        = min(len(_col_athl_map[a]['xs']) for a in _col_athletes)

                                    # Pontuação por frame
                                    _col_scores = []
                                    for _ci in range(_col_n):
                                        _cxi = [_col_athl_map[a]['xs'][_ci] for a in _col_athletes]
                                        _cyi = [_col_athl_map[a]['ys'][_ci] for a in _col_athletes]
                                        _nm2 = len(_cxi)
                                        _xm2 = sum(_cxi) / _nm2
                                        _ym2 = sum(_cyi) / _nm2
                                        _cm  = _col_metric
                                        if "Dispersão" in _cm:
                                            _sc = sum(
                                                ((x - _xm2) ** 2 + (y - _ym2) ** 2) ** 0.5
                                                for x, y in zip(_cxi, _cyi)
                                            ) / _nm2
                                        elif "Compactidade" in _cm:
                                            _sc = -(
                                                sum(
                                                    ((x - _xm2) ** 2 + (y - _ym2) ** 2) ** 0.5
                                                    for x, y in zip(_cxi, _cyi)
                                                ) / _nm2
                                            )
                                        elif "Amplitude" in _cm:
                                            _sc = max(_cxi) - min(_cxi)
                                        elif "Profundidade" in _cm:
                                            _sc = max(_cyi) - min(_cyi)
                                        elif "Espaço" in _cm:
                                            _sy2 = sorted(_cyi)
                                            _h2  = _nm2 // 2
                                            _sc  = (
                                                sum(_sy2[_h2:]) / len(_sy2[_h2:]) - sum(_sy2[:_h2]) / len(_sy2[:_h2])
                                            ) if _h2 else 0.0
                                        else:
                                            _sc = 0.0
                                        _col_scores.append(_sc)

                                    _col_bi  = max(range(len(_col_scores)), key=lambda i: _col_scores[i])
                                    _col_bv  = abs(_col_scores[_col_bi])

                                    # Timestamp
                                    _col_ts0 = _col_athl_map[_col_athletes[0]]['ts'][_col_bi]
                                    try:
                                        from datetime import datetime as _dtcol
                                        _col_tstr = (
                                            _dtcol.fromtimestamp(float(_col_ts0)).strftime('%H:%M:%S')
                                            if float(_col_ts0) > 1e6
                                            else f"{int(_col_bi / 10 // 60):02d}:{int(_col_bi / 10 % 60):02d}"
                                        )
                                    except Exception:
                                        _col_tstr = f"{int(_col_bi / 10 // 60):02d}:{int(_col_bi / 10 % 60):02d}"

                                    _ck1, _ck2, _ck3 = st.columns(3)
                                    _ck1.metric("⏱️ Instante", _col_tstr)
                                    _ck2.metric("📊 Valor", f"{_col_bv:.1f} m")
                                    _ck3.metric("👥 Atletas", str(len(_col_athl_map)))

                                    # Criar campo
                                    _col_cfg3 = None
                                    for _hkc3 in list(st.session_state.keys()):
                                        if (
                                            _hkc3.startswith("campo_cfg__")
                                            and isinstance(st.session_state[_hkc3], dict)
                                            and 'fl' in st.session_state[_hkc3]
                                        ):
                                            _col_cfg3 = st.session_state[_hkc3]
                                            break
                                    _col_fl3 = float(_col_cfg3.get('fl', 105)) if _col_cfg3 else 105.0
                                    _col_fw3 = float(_col_cfg3.get('fw', 68))  if _col_cfg3 else 68.0

                                    _fig_col = desenhar_campo_futebol_bonito(
                                        field_length=_col_fl3,
                                        field_width=_col_fw3,
                                        title=(
                                            f"{_col_metric.split('  ')[0].strip()} | "
                                            f"{_col_per_sel} | ⏱️ {_col_tstr} | "
                                            f"Valor: {_col_bv:.1f} m"
                                        ),
                                    )

                                    _col_palette = [
                                        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                                        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#76D7C4',
                                        '#F1948A', '#85C1E9', '#82E0AA', '#F8C471', '#AED6F1',
                                    ]
                                    _cx_all3, _cy_all3 = [], []
                                    for _ci3, _ca3 in enumerate(_col_athletes):
                                        _cd3  = _col_athl_map[_ca3]
                                        _cx3  = _cd3['xs'][_col_bi]
                                        _cy3  = _cd3['ys'][_col_bi]
                                        _cv3  = _cd3['vel'][_col_bi] if _col_bi < len(_cd3['vel']) else 0.0
                                        _ccl3 = _col_palette[_ci3 % len(_col_palette)]
                                        _cx_all3.append(_cx3)
                                        _cy_all3.append(_cy3)
                                        _lbl3 = (_ca3.split()[-1][:10] if _ca3.split() else _ca3[:10])
                                        _fig_col.add_trace(go.Scatter(
                                            x=[_cx3], y=[_cy3],
                                            mode='markers+text',
                                            marker=dict(
                                                size=22, color=_ccl3, symbol='circle',
                                                line=dict(color='white', width=2),
                                            ),
                                            text=[_lbl3],
                                            textposition='top center',
                                            textfont=dict(color='white', size=8),
                                            name=_ca3,
                                            customdata=[[_cd3.get('pos', '—'), round(_cv3, 1)]],
                                            hovertemplate=(
                                                '<b>%{meta}</b><br>'
                                                'Posição: %{customdata[0]}<br>'
                                                'Velocidade: %{customdata[1]} km/h'
                                                '<extra></extra>'
                                            ),
                                            meta=_ca3,
                                            showlegend=True,
                                        ))

                                    # Centróide do grupo
                                    if _cx_all3:
                                        _cxm3 = sum(_cx_all3) / len(_cx_all3)
                                        _cym3 = sum(_cy_all3) / len(_cy_all3)
                                        _fig_col.add_trace(go.Scatter(
                                            x=[_cxm3], y=[_cym3],
                                            mode='markers',
                                            marker=dict(
                                                size=16, color='white', symbol='x',
                                                line=dict(color='#333', width=2),
                                            ),
                                            name='Centróide',
                                            hoverinfo='name',
                                            showlegend=True,
                                        ))

                                    _fig_col.update_layout(
                                        height=580,
                                        legend=dict(
                                            orientation='h',
                                            yanchor='bottom', y=-0.30,
                                            xanchor='center', x=0.5,
                                            font=dict(color='white', size=8),
                                        ),
                                    )
                                    st.plotly_chart(_fig_col, use_container_width=True)

            # ==================== ABA 10: MONITORAMENTO AO VIVO ====================
            with abas[10]:
                st.subheader("📡 Monitoramento em Tempo Real")
                st.caption(
                    "Conecta ao endpoint `/live` da API Catapult para acompanhar métricas "
                    "de atletas durante uma sessão ativa. Requer uma sessão aberta no OpenField."
                )

                # ── Inicializar session state do live ─────────────────────
                if 'live_active' not in st.session_state:
                    st.session_state['live_active'] = False
                if 'live_alert_log' not in st.session_state:
                    st.session_state['live_alert_log'] = []
                if 'live_snapshot' not in st.session_state:
                    st.session_state['live_snapshot'] = None
                if 'live_info_snapshot' not in st.session_state:
                    st.session_state['live_info_snapshot'] = None

                # ── Layout de controles ────────────────────────────────────
                _lv_c1, _lv_c2, _lv_c3 = st.columns([2, 2, 1])
                with _lv_c1:
                    _lv_interval = st.select_slider(
                        "⏱️ Intervalo de atualização:",
                        options=[5, 10, 15, 30, 60],
                        value=10,
                        format_func=lambda x: f"{x}s",
                        key="live_interval",
                    )
                with _lv_c2:
                    _lv_cols = st.multiselect(
                        "📊 Métricas a exibir:",
                        options=["velocity", "heart_rate", "player_load",
                                 "acceleration", "total_distance", "odometer"],
                        default=["velocity", "heart_rate", "player_load"],
                        key="live_metrics_sel",
                    )
                with _lv_c3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    _lv_refresh_now = st.button(
                        "🔄 Agora", key="live_refresh_now",
                        help="Atualizar imediatamente uma vez"
                    )

                # ── Configuração de limiares (alertas) ────────────────────
                with st.expander("🚨 Configurar Limiares de Alerta", expanded=True):
                    st.caption(
                        "O app dispara um alerta visual quando o atleta **atingir ou ultrapassar** "
                        "o valor configurado. Deixe 0 para desativar."
                    )
                    _lv_th_c1, _lv_th_c2, _lv_th_c3 = st.columns(3)
                    with _lv_th_c1:
                        _lv_th_vel = st.number_input(
                            "🚀 Vel. máx (km/h)", min_value=0.0,
                            max_value=40.0, value=25.0, step=0.5,
                            key="live_th_vel",
                        )
                        _lv_th_dist = st.number_input(
                            "📏 Distância total (m)", min_value=0.0,
                            max_value=15000.0, value=0.0, step=100.0,
                            key="live_th_dist",
                            help="Alerta quando atleta passar desta distância acumulada. 0 = desativado."
                        )
                    with _lv_th_c2:
                        _lv_th_hr = st.number_input(
                            "❤️ FC máx (bpm)", min_value=0.0,
                            max_value=220.0, value=180.0, step=1.0,
                            key="live_th_hr",
                        )
                        _lv_th_pl = st.number_input(
                            "⚡ PlayerLoad", min_value=0.0,
                            max_value=1500.0, value=0.0, step=10.0,
                            key="live_th_pl",
                            help="Alerta quando PlayerLoad acumulado ultrapassar este valor. 0 = desativado."
                        )
                    with _lv_th_c3:
                        _lv_th_acc = st.number_input(
                            "💥 Aceleração (m/s²)", min_value=0.0,
                            max_value=10.0, value=3.5, step=0.1,
                            key="live_th_acc",
                        )
                        _lv_sound = st.checkbox(
                            "🔔 Exibir badge de alerta",
                            value=True, key="live_sound",
                        )

                    _lv_thresholds = {
                        "velocity":       (_lv_th_vel,  "km/h", "🚀"),
                        "heart_rate":     (_lv_th_hr,   "bpm",  "❤️"),
                        "acceleration":   (_lv_th_acc,  "m/s²", "💥"),
                        "total_distance": (_lv_th_dist, "m",    "📏"),
                        "player_load":    (_lv_th_pl,   "UA",   "⚡"),
                    }

                # ── Botões de controle ─────────────────────────────────────
                _lv_btn_c1, _lv_btn_c2, _lv_btn_c3 = st.columns([2, 2, 3])
                with _lv_btn_c1:
                    if not st.session_state['live_active']:
                        if st.button("▶ Iniciar Monitoramento", type="primary",
                                     key="live_start"):
                            st.session_state['live_active'] = True
                            st.session_state['live_alert_log'] = []
                            st.rerun()
                    else:
                        if st.button("⏹ Parar Monitoramento", type="secondary",
                                     key="live_stop"):
                            st.session_state['live_active'] = False
                            st.rerun()
                with _lv_btn_c2:
                    if st.button("🗑️ Limpar Alertas", key="live_clear_alerts"):
                        st.session_state['live_alert_log'] = []
                        st.rerun()
                with _lv_btn_c3:
                    _lv_status_ph = st.empty()

                # ── Indicador de status ────────────────────────────────────
                if st.session_state['live_active']:
                    _lv_status_ph.markdown(
                        "<span style='display:inline-flex;align-items:center;gap:6px;"
                        "background:#064e3b;border:1px solid #10b981;border-radius:20px;"
                        "padding:4px 12px;font-size:13px;color:#34d399'>"
                        "<span style='width:8px;height:8px;border-radius:50%;"
                        "background:#10b981;animation:pulse 1s infinite'></span>"
                        " AO VIVO</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    _lv_status_ph.markdown(
                        "<span style='display:inline-flex;align-items:center;gap:6px;"
                        "background:#1f2937;border:1px solid #4b5563;border-radius:20px;"
                        "padding:4px 12px;font-size:13px;color:#9ca3af'>"
                        "⏸ PAUSADO</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # ── Containers de exibição ─────────────────────────────────
                _lv_session_ph  = st.empty()   # info da sessão
                _lv_athletes_ph = st.empty()   # cards dos atletas
                _lv_alerts_ph   = st.empty()   # log de alertas

                # ── Função para renderizar os dados ───────────────────────
                def _render_live(info_data, athletes_data):
                    """Renderiza info da sessão + cards de atletas."""
                    import time as _tm

                    # ── Painel de sessão ──────────────────────────────────
                    with _lv_session_ph.container():
                        if info_data and isinstance(info_data, dict):
                            _si_c1, _si_c2, _si_c3, _si_c4 = st.columns(4)
                            _act_name = (
                                info_data.get('activity_name') or
                                info_data.get('name') or
                                info_data.get('id', '—')
                            )
                            _start_ts = info_data.get('start_time') or info_data.get('started_at') or 0
                            _elapsed  = ""
                            if _start_ts:
                                try:
                                    _el_s = int(_tm.time() - float(_start_ts))
                                    _elapsed = f"{_el_s//3600:02d}:{(_el_s%3600)//60:02d}:{_el_s%60:02d}"
                                except Exception:
                                    _elapsed = "—"
                            _n_atl = len(athletes_data) if isinstance(athletes_data, list) else 0
                            with _si_c1:
                                st.metric("🎯 Sessão", str(_act_name)[:28] or "Ativa")
                            with _si_c2:
                                st.metric("⏱️ Tempo decorrido", _elapsed or "—")
                            with _si_c3:
                                st.metric("👥 Atletas ativos", str(_n_atl))
                            with _si_c4:
                                st.metric("🔁 Última atualização",
                                          _tm.strftime('%H:%M:%S'))
                        elif athletes_data:
                            st.metric("👥 Atletas ativos",
                                      len(athletes_data) if isinstance(athletes_data, list) else "—")

                    # ── Cards dos atletas ──────────────────────────────────
                    with _lv_athletes_ph.container():
                        if not athletes_data:
                            st.info(
                                "📭 Nenhuma sessão ao vivo detectada.\n\n"
                                "**Verifique:**\n"
                                "- Existe uma atividade aberta no OpenField agora\n"
                                "- Os dispositivos estão transmitindo dados\n"
                                "- O token tem permissão para dados ao vivo"
                            )
                            return

                        # Normalizar resposta (pode ser lista ou dict com key 'athletes')
                        _atl_list = athletes_data
                        if isinstance(athletes_data, dict):
                            _atl_list = (
                                athletes_data.get('athletes') or
                                athletes_data.get('data') or
                                list(athletes_data.values())
                            )
                        if not isinstance(_atl_list, list):
                            st.warning("⚠️ Formato de resposta inesperado da API /live.")
                            st.json(athletes_data)
                            return

                        # Paleta de cores por atleta
                        _LV_COLORS = [
                            '#FF6B6B','#4ECDC4','#45B7D1','#96CEB4',
                            '#FFEAA7','#DDA0DD','#98FB98','#FFB347',
                        ]

                        # ── Mapeamento de campos (diferentes versões da API) ──
                        def _get_field(d, *keys, default=None):
                            for k in keys:
                                if k in d:
                                    return d[k]
                            return default

                        # ── Grid de cards (3 por linha) ───────────────────
                        _n_cols_lv = min(3, len(_atl_list))
                        _lv_rows   = [
                            _atl_list[i:i+_n_cols_lv]
                            for i in range(0, len(_atl_list), _n_cols_lv)
                        ]

                        _new_alerts = []
                        import time as _tm2

                        for _row in _lv_rows:
                            _cols = st.columns(len(_row))
                            for _ci, (_col, _atl) in enumerate(zip(_cols, _row)):
                                if not isinstance(_atl, dict):
                                    continue
                                _color = _LV_COLORS[_ci % len(_LV_COLORS)]
                                _name  = (
                                    _get_field(_atl, 'name', 'athlete_name',
                                               'display_name', default='Atleta')
                                )

                                # Extrair métricas (tentativa em vários campos)
                                _vel  = _get_field(_atl, 'velocity', 'current_velocity',
                                                   'speed', 'v', default=None)
                                _hr   = _get_field(_atl, 'heart_rate', 'hr',
                                                   'current_heart_rate', default=None)
                                _pl   = _get_field(_atl, 'player_load', 'playerload',
                                                   'total_player_load', 'pl', default=None)
                                _acc  = _get_field(_atl, 'acceleration', 'acc',
                                                   'current_acceleration', 'a', default=None)
                                _dist = _get_field(_atl, 'total_distance', 'distance',
                                                   'odometer', 'o', default=None)

                                # Verificar limiares
                                _card_alerts = []
                                _metric_map = {
                                    "velocity":       (_vel,  _lv_thresholds["velocity"]),
                                    "heart_rate":     (_hr,   _lv_thresholds["heart_rate"]),
                                    "acceleration":   (_acc,  _lv_thresholds["acceleration"]),
                                    "total_distance": (_dist, _lv_thresholds["total_distance"]),
                                    "player_load":    (_pl,   _lv_thresholds["player_load"]),
                                }
                                for _mk, (_mv, (_thr, _unit, _ico)) in _metric_map.items():
                                    if _mv is not None and _thr > 0:
                                        try:
                                            if float(_mv) >= float(_thr):
                                                _card_alerts.append(
                                                    f"{_ico} {_mk.replace('_',' ').title()}: "
                                                    f"**{float(_mv):.1f}** {_unit} "
                                                    f"(limiar: {_thr})"
                                                )
                                                _new_alerts.append({
                                                    'ts': _tm2.strftime('%H:%M:%S'),
                                                    'atleta': _name,
                                                    'metrica': _mk,
                                                    'valor': float(_mv),
                                                    'limiar': _thr,
                                                    'unidade': _unit,
                                                    'ico': _ico,
                                                })
                                        except Exception:
                                            pass

                                _has_alert = bool(_card_alerts)

                                # Cor do card por velocidade
                                try:
                                    _v_num = float(_vel) if _vel is not None else 0
                                    _vel_zone_color = (
                                        '#1e3a5f' if _v_num < 7 else
                                        '#1a3d2b' if _v_num < 14 else
                                        '#3d2b00' if _v_num < 19 else
                                        '#3d1a1a'
                                    )
                                except Exception:
                                    _vel_zone_color = '#1e293b'

                                _border_color = '#ef4444' if _has_alert else _color

                                with _col:
                                    # Card HTML
                                    st.markdown(
                                        f"<div style='background:{_vel_zone_color};"
                                        f"border:2px solid {_border_color};"
                                        f"border-radius:10px;padding:14px 12px;"
                                        f"margin-bottom:8px;min-height:160px'>"
                                        f"<div style='display:flex;justify-content:space-between;"
                                        f"align-items:center;margin-bottom:8px'>"
                                        f"<span style='font-weight:700;font-size:14px;"
                                        f"color:{_color}'>{_name[:18]}</span>"
                                        + (
                                            "<span style='background:#ef4444;color:white;"
                                            "font-size:10px;padding:2px 6px;border-radius:10px'>"
                                            "⚠️ ALERTA</span>"
                                            if _has_alert else
                                            "<span style='background:#10b981;color:white;"
                                            "font-size:10px;padding:2px 6px;border-radius:10px'>"
                                            "✅ OK</span>"
                                        ) +
                                        "</div>"
                                        # Métricas
                                        + (
                                            f"<div style='font-size:22px;font-weight:800;"
                                            f"color:white;margin:4px 0'>"
                                            f"🚀 {float(_vel):.1f} <span style='font-size:11px;"
                                            f"color:#94a3b8'>km/h</span></div>"
                                            if _vel is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"❤️ {float(_hr):.0f} bpm</div>"
                                            if _hr is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"⚡ PL {float(_pl):.1f}</div>"
                                            if _pl is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"💥 {float(_acc):.2f} m/s²</div>"
                                            if _acc is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"📏 {float(_dist):.0f} m</div>"
                                            if _dist is not None else ""
                                        )
                                        + "</div>",
                                        unsafe_allow_html=True,
                                    )
                                    # Detalhes do alerta abaixo do card
                                    if _card_alerts and _lv_sound:
                                        for _al in _card_alerts:
                                            st.markdown(
                                                f"<div style='background:#450a0a;border-left:"
                                                f"3px solid #ef4444;border-radius:4px;"
                                                f"padding:4px 8px;font-size:11px;color:#fca5a5;"
                                                f"margin-bottom:3px'>{_al}</div>",
                                                unsafe_allow_html=True,
                                            )

                        # Adicionar novos alertas ao log (sem duplicar os últimos 5s)
                        if _new_alerts:
                            _existing_keys = {
                                (a['atleta'], a['metrica'], a['ts'])
                                for a in st.session_state['live_alert_log'][-20:]
                            }
                            for _na in _new_alerts:
                                _key = (_na['atleta'], _na['metrica'], _na['ts'])
                                if _key not in _existing_keys:
                                    st.session_state['live_alert_log'].append(_na)

                    # ── Log de alertas ────────────────────────────────────
                    with _lv_alerts_ph.container():
                        _log = st.session_state['live_alert_log']
                        if _log:
                            st.markdown("#### 🔴 Histórico de Alertas")
                            _df_log = pd.DataFrame(list(reversed(_log[-50:])))
                            _df_log.columns = [
                                'Hora', 'Atleta', 'Métrica',
                                'Valor', 'Limiar', 'Unidade', '—'
                            ]
                            st.dataframe(
                                _df_log[['Hora','Atleta','Métrica','Valor','Limiar','Unidade']],
                                use_container_width=True,
                                hide_index=True,
                            )

                # ── Loop de polling ────────────────────────────────────────
                _lv_api = st.session_state.get('api')

                if _lv_api is None:
                    st.warning(
                        "⚠️ API não inicializada. Carregue os dados na sidebar primeiro."
                    )
                elif st.session_state['live_active'] or _lv_refresh_now:
                    import time as _lv_time

                    # Buscar dados
                    _lv_info  = _lv_api.get_live_info()
                    _lv_data  = _lv_api.get_live_athletes()

                    # Guardar snapshot para exibição após parar
                    st.session_state['live_snapshot']      = _lv_data
                    st.session_state['live_info_snapshot'] = _lv_info

                    _render_live(_lv_info, _lv_data)

                    # Continuar polling se ativo
                    if st.session_state['live_active'] and not _lv_refresh_now:
                        _lv_time.sleep(int(st.session_state.get('live_interval', 10)))
                        st.rerun()

                elif st.session_state.get('live_snapshot') is not None:
                    # Mostrar último snapshot após pausar
                    st.caption("📸 Último snapshot (monitoramento pausado)")
                    _render_live(
                        st.session_state['live_info_snapshot'],
                        st.session_state['live_snapshot'],
                    )
                else:
                    # Estado inicial
                    st.markdown(
                        "<div style='text-align:center;padding:48px 0;"
                        "color:#64748b'>"
                        "<div style='font-size:48px;margin-bottom:12px'>📡</div>"
                        "<div style='font-size:18px;font-weight:600;color:#94a3b8'>"
                        "Monitoramento ao vivo pronto</div>"
                        "<div style='font-size:14px;margin-top:8px'>"
                        "Configure os limiares acima e clique em "
                        "<b style='color:#10b981'>▶ Iniciar Monitoramento</b><br>"
                        "para começar a acompanhar a sessão em tempo real."
                        "</div></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "> **Pré-requisito:** uma sessão deve estar aberta e transmitindo "
                        "dados no OpenField Cloud agora. O endpoint `/live` só retorna dados "
                        "durante sessões ativas."
                    )

        else:
            st.warning("Nenhum dado encontrado")

    elif 'df_athletes' in st.session_state and not st.session_state.df_athletes.empty:
        st.info("👈 Selecione uma atividade, período(s) e clique em 'Buscar Atletas da Atividade'")

if __name__ == "__main__":
    main()
    