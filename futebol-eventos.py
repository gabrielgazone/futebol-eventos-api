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
        return r.json() if r.status_code == 200 else None
    except Exception:
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
        ("parameters", "ts,lat,long,v,a,hr,pl,xy"),
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
        return [], []
    
    velocity_efforts = []
    acceleration_efforts = []
    
    if isinstance(response_data, list) and len(response_data) > 0:
        item = response_data[0]
        if isinstance(item, dict) and 'data' in item:
            data_obj = item['data']
            if isinstance(data_obj, dict):
                velocity_efforts = data_obj.get('velocity_efforts', [])
                acceleration_efforts = data_obj.get('acceleration_efforts', [])
    
    return velocity_efforts, acceleration_efforts

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
                   showscale=True, colorbar=dict(title="Velocidade (km/h)", x=1.05, len=0.5)),
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
        colorbar=dict(title="Velocidade (km/h)", x=1.05, len=0.5),
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
        colorbar=dict(title="Frequência", x=1.05, len=0.5),
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
            t = float(ev.get('start_time', 0))
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
            t_ev  = float(ev.get('start_time', 0)) - ts0
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
            t = float(ev.get('start_time', 0))
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


def desenhar_campo_futebol_bonito(field_length=105, field_width=68, margin=3, title=""):
    """Campo de futebol com faixas verdes alternadas e marcações oficiais FIFA (top-down)."""
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

def calcular_metricas(sensor_points, athlete_name, min_dur_s=None):
    if not sensor_points:
        return None

    if min_dur_s is None:
        min_dur_s = get_min_dur_s()

    distancia_total = 0
    dist_hi = 0
    dist_sprint = 0
    dist_z4 = 0          # Zone 4: 19-24 km/h
    player_load = 0
    velocidades = []
    fcs = []
    acc_list = []

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
                if v_kmh > 19:
                    dist_hi += dist_seg
                if v_kmh > 24:
                    dist_sprint += dist_seg
                if 19 < v_kmh <= 24:
                    dist_z4 += dist_seg

            if v_kmh > 24 and not in_sprint:
                sprints += 1
                in_sprint = True
            elif v_kmh <= 24:
                in_sprint = False

            if v_kmh > 19 and not in_hi:
                n_esforcos_hi += 1
                in_hi = True
            elif v_kmh <= 19:
                in_hi = False

            # Registra frame de entrada em alta intensidade para RHIE
            if v_kmh > 19 and not _in_hi_rhie:
                rhie_effort_frames.append(_frame_idx)
                _in_hi_rhie = True
            elif v_kmh <= 19:
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
        'Total Pontos': len(sensor_points)
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

def processar_efforts_velocidade(efforts_data):
    if not efforts_data:
        return pd.DataFrame()
    
    records = []
    max_vel_encontrada = 0
    
    for effort in efforts_data:
        max_vel = effort.get('max_velocity', 0)
        if max_vel:
            max_vel_encontrada = max(max_vel_encontrada, max_vel)
    
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
            except:
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
            '_end_ts': end_time
        })
    
    return pd.DataFrame(records)

def processar_efforts_aceleracao(efforts_data):
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
                            'posicao': position_name, 'equipe': team_name
                        })
                    st.session_state.df_athletes = pd.DataFrame(atletas)
                    st.success(f"✅ {len(atletas)} atletas carregados")
                
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
                    with st.spinner("Buscando atletas..."):
                        api = st.session_state.api
                        primeiro_periodo = periodos_selecionados[0] if periodos_selecionados else (st.session_state.period_options[0] if st.session_state.period_options else 'Atividade Completa')
                        period_id = st.session_state.period_ids.get(primeiro_periodo)
                        
                        if period_id:
                            response_data = api.get_athletes_in_period(period_id)
                        else:
                            response_data = api.get_activity_athletes(activity_id)
                        
                        if response_data:
                            athletes_in = None
                            if isinstance(response_data, list):
                                athletes_in = response_data
                            elif isinstance(response_data, dict):
                                for key in ['data', 'items', 'athletes']:
                                    if key in response_data and isinstance(response_data[key], list):
                                        athletes_in = response_data[key]
                                        break
                                if athletes_in is None and 'id' in response_data:
                                    athletes_in = [response_data]
                            
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
                                st.success(f"✅ {len(df_atletas_temp)} atletas encontrados")
                
                if 'atletas_filtrados' in st.session_state and not st.session_state.atletas_filtrados.empty:
                    st.subheader("🏃 Selecionar Atletas")
                    atletas_sel = st.multiselect("Selecione os atletas para análise:", st.session_state.atletas_filtrados['nome'].tolist())
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
                
                if period_id:
                    response         = api.get_period_sensor_data(period_id, athlete_id)
                    efforts_response = api.get_period_efforts(period_id, athlete_id, "velocity,acceleration")
                    events_response  = api.get_period_events(period_id, athlete_id, eventos_futebol_str) if eventos_futebol_str else None
                else:
                    response         = api.get_sensor_data(activity_id, athlete_id)
                    efforts_response = api.get_activity_efforts(activity_id, athlete_id, "velocity,acceleration")
                    events_response  = api.get_activity_events(activity_id, athlete_id, eventos_futebol_str) if eventos_futebol_str else None
                
                sensor_points = extrair_dados_sensor(response)
                
                if sensor_points:
                    dados_sensor_por_atleta[atleta_nome] = sensor_points
                    
                    metricas = calcular_metricas(sensor_points, atleta_nome)
                    if metricas:
                        metricas['Posição'] = athlete_posicao
                        metricas['Equipe'] = athlete_equipe
                        resultados.append(metricas)
                    
                    if efforts_response:
                        vel_efforts, acc_efforts = extrair_efforts_data(efforts_response)
                        if vel_efforts:
                            dados_efforts_vel[atleta_nome] = vel_efforts
                        if acc_efforts:
                            dados_efforts_acc[atleta_nome] = acc_efforts
                    
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

        # ── Gerar "Períodos Combinados" quando há mais de 1 período ──────────
        _CHAVE_COMBINADO = '📊 Períodos Combinados'
        _periodos_reais = [k for k in resultados_por_periodo if k != _CHAVE_COMBINADO]
        if len(_periodos_reais) > 1:
            _res_combinado = combinar_periodos(
                {k: resultados_por_periodo[k] for k in _periodos_reais}
            )
            if _res_combinado:
                resultados_por_periodo[_CHAVE_COMBINADO] = _res_combinado

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
                st.metric("⚡ PlayerLoad Total", f"{total_pl:,.0f}")
            with col4:
                max_vel = 0
                for resultados in resultados_por_periodo.values():
                    for r in resultados:
                        max_vel = max(max_vel, r.get('Velocidade Máx (km/h)', 0))
                st.metric("💨 Velocidade Máx", f"{max_vel:.1f} km/h")
            
            st.markdown("---")
            
            abas = st.tabs([
                "📈 Gráficos Comparativos",
                "🗺️ Campo de Futebol",
                "⏱️ Esforços ao Longo do Tempo",
                "📊 Janelas Temporais Móveis",
                "💪 Carga Neuromuscular",
                "🏎️ Perfil Acc-Vel",
                "📋 Tabela Descritiva",
                "📊 Por Posição",
                "🎬 História do Jogo",
            ])
            
            # ==================== ABA 1: GRÁFICOS COMPARATIVOS ====================
            with abas[0]:
                st.subheader("📈 Gráficos Comparativos")
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
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                df_melted = df_filtrado.melt(id_vars=['Atleta'], value_vars=metricas_comp, var_name='Métrica', value_name='Valor')
                                fig = px.bar(df_melted, x='Atleta', y='Valor', color='Métrica', barmode='group',
                                            title=f"Comparação de Métricas - {periodo_comp}")
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                if len(atletas_selecionados_comp) <= 5:
                                    fig_radar = go.Figure()
                                    # Normaliza cada métrica para 0-1 para comparação justa
                                    _radar_df = df_filtrado[df_filtrado['Atleta'].isin(atletas_selecionados_comp[:5])][['Atleta'] + metricas_comp].copy()
                                    _radar_max = _radar_df[metricas_comp].max()
                                    _radar_max = _radar_max.replace(0, 1)  # evita divisão por zero
                                    for atleta in atletas_selecionados_comp[:5]:
                                        _row = _radar_df[_radar_df['Atleta'] == atleta][metricas_comp].iloc[0]
                                        valores_norm = (_row / _radar_max).tolist()
                                        fig_radar.add_trace(go.Scatterpolar(
                                            r=valores_norm,
                                            theta=metricas_comp,
                                            fill='toself',
                                            name=atleta
                                        ))
                                    fig_radar.update_layout(
                                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                        title="Comparação Radial (Normalizado 0–1)"
                                    )
                                    st.plotly_chart(fig_radar, use_container_width=True)
                            _cols_show = [c for c in ['Atleta', 'Equipe', 'Posição'] + metricas_comp if c in df_filtrado.columns]
                            st.dataframe(df_filtrado[_cols_show], use_container_width=True)
                
                else:
                    primeiro_periodo = list(resultados_por_periodo.keys())[0]
                    if resultados_por_periodo[primeiro_periodo]:
                        df_primeiro = pd.DataFrame(resultados_por_periodo[primeiro_periodo])
                        atleta_comp = st.selectbox("Selecione o atleta:", df_primeiro['Atleta'].tolist(), key="atleta_comp")
                        
                        dados_atleta = {}
                        for periodo, resultados in resultados_por_periodo.items():
                            if resultados:
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
            
            # ==================== ABA 2: CAMPO DE FUTEBOL ====================
            with abas[1]:
                st.subheader("🗺️ Campo de Futebol — Análise de Movimentação")
                st.caption(REFERENCIAS["campo"])

                if dados_posicao_por_periodo:
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
                                'lat':          float(_vs.get('lat', 0)),
                                'lon':          float(_vs.get('lon', 0)),
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

                                # ── Tier 1: matching por timestamp exato (fallback)
                                if len(xs_a) < 2 and _anim_start_ts > 0 and _anim_end_ts > 0:
                                    if _ts_eff and len(_ts_eff) == len(_xn_eff):
                                        _ts_c = np.array(_ts_eff)
                                        _m    = (_ts_c >= _anim_start_ts) & (_ts_c <= _anim_end_ts)
                                        if _m.any():
                                            xs_a  = np.array(_xn_eff)[_m].tolist()
                                            ys_a  = np.array(_yn_eff)[_m].tolist()
                                            vel_a = (np.array(_vel_eff)[_m].tolist()
                                                     if len(_vel_eff) == len(_xn_eff) else [0]*int(_m.sum()))
                                            acc_a = (np.array(_acc_eff)[_m].tolist()
                                                     if len(_acc_eff) == len(_xn_eff) else [0]*int(_m.sum()))

                                # ── Tier 2: fallback proporcional por timestamp
                                if (len(xs_a) < 2 and _anim_start_ts > 0
                                        and _anim_end_ts > 0 and _ts_eff):
                                    _ts_min = min(_ts_eff)
                                    _ts_max = max(_ts_eff)
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
                                    fig_campo = desenhar_campo_futebol_bonito(title=_titulo_campo)

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
                                    if _anim_effort_row is not None and len(xs_a) >= 2:
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
                                        else:
                                            st.warning(
                                                "⚠️ Pontos de campo insuficientes para animar este esforço. "
                                                "Verifique se o campo foi aplicado e se os timestamps correspondem."
                                            )
                                    elif _anim_effort_row is None:
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
                                    # ANÁLISE 4 — PERFIL FORÇA-VELOCIDADE
                                    # ══════════════════════════════════════════════
                                    st.markdown("---")
                                    with st.expander("📊 Perfil Força-Velocidade Individual", expanded=False):
                                        st.markdown(
                                            "Estima a **capacidade de geração de força** em cada faixa de velocidade. "
                                            "Para cada bin de velocidade, calcula o **pico de aceleração** (95º percentil). "
                                            "A regressão linear revela o perfil F-V: "
                                            "inclinação acentuada = atleta de **força**; "
                                            "mais plana = atleta de **velocidade**."
                                        )
                                        _fv_vel = np.array(vel_raw_campo, dtype=float)
                                        _fv_acc = np.array(acc_raw_campo, dtype=float)

                                        if len(_fv_vel) > 200 and _fv_vel.max() > 5:
                                            _bin_w   = 2.0
                                            _v_top   = min(float(_fv_vel.max()), 36.0)
                                            _bins    = np.arange(0, _v_top + _bin_w, _bin_w)
                                            _fv_pv, _fv_pa, _fv_pn = [], [], []
                                            for _bi in range(len(_bins) - 1):
                                                _lo2, _hi2 = _bins[_bi], _bins[_bi + 1]
                                                _mk = (_fv_vel >= _lo2) & (_fv_vel < _hi2) & (_fv_acc > 0)
                                                if _mk.sum() >= 5:
                                                    _fv_pv.append((_lo2 + _hi2) / 2)
                                                    _fv_pa.append(float(np.percentile(_fv_acc[_mk], 95)))
                                                    _fv_pn.append(int(_mk.sum()))

                                            if len(_fv_pv) >= 3:
                                                _fv_pv  = np.array(_fv_pv)
                                                _fv_pa  = np.array(_fv_pa)
                                                _fv_cf  = np.polyfit(_fv_pv, _fv_pa, 1)
                                                _sl, _ic = float(_fv_cf[0]), float(_fv_cf[1])
                                                _F0  = max(_ic, 0.0)
                                                _V0  = ((-_ic / _sl) if _sl < -1e-6
                                                        else float(_fv_pv[-1]) * 1.3)
                                                _V0  = max(_V0, float(_fv_pv[-1]))
                                                _Pmax = (_F0 * (_V0 / 3.6)) / 4.0

                                                if _sl < -0.12:
                                                    _pfil = "💪 Força-Dominante"
                                                    _pdesc = "Alta acc. inicial, queda acentuada com a velocidade."
                                                elif _sl > -0.06:
                                                    _pfil = "⚡ Velocidade-Dominante"
                                                    _pdesc = "Mantém aceleração nas altas velocidades."
                                                else:
                                                    _pfil = "⚖️ Equilibrado"
                                                    _pdesc = "Balanço entre força inicial e velocidade de pico."

                                                _v_line = np.linspace(0, _V0 * 1.05, 120)
                                                _a_line = np.clip(np.polyval(_fv_cf, _v_line), 0, None)

                                                _fig_fv = go.Figure()
                                                _fig_fv.add_trace(go.Scatter(
                                                    x=_fv_pv, y=_fv_pa,
                                                    mode='markers+text',
                                                    marker=dict(
                                                        size=[max(9, min(22, n // 40)) for n in _fv_pn],
                                                        color='#FF9800',
                                                        line=dict(color='white', width=1.5)
                                                    ),
                                                    text=[f"{v:.0f}" for v in _fv_pv],
                                                    textposition='top center',
                                                    textfont=dict(color='rgba(255,255,255,0.75)', size=9),
                                                    name='Pico acc (95p) / bin',
                                                    hovertemplate=(
                                                        'Vel: %{x:.1f} km/h<br>'
                                                        'Acc 95p: %{y:.2f} m/s²<extra></extra>'
                                                    )
                                                ))
                                                _fig_fv.add_trace(go.Scatter(
                                                    x=_v_line, y=_a_line,
                                                    mode='lines',
                                                    line=dict(color='#00E5FF', width=2.5, dash='dash'),
                                                    name='Regressão F-V',
                                                    hoverinfo='skip'
                                                ))
                                                _fig_fv.add_annotation(
                                                    x=0, y=_F0,
                                                    text=f"<b>F₀ = {_F0:.2f} m/s²</b>",
                                                    showarrow=True, arrowhead=2,
                                                    arrowcolor='#4CAF50',
                                                    font=dict(color='#4CAF50', size=11),
                                                    ax=55, ay=-35
                                                )
                                                _fig_fv.add_annotation(
                                                    x=_V0, y=0,
                                                    text=f"<b>V₀ = {_V0:.1f} km/h</b>",
                                                    showarrow=True, arrowhead=2,
                                                    arrowcolor='#FF4081',
                                                    font=dict(color='#FF4081', size=11),
                                                    ax=-55, ay=-35
                                                )
                                                _fig_fv.update_layout(
                                                    title=dict(
                                                        text=(f"Perfil F-V — {atleta_mapa} "
                                                              f"| {_pfil}"),
                                                        font=dict(color='white', size=14)
                                                    ),
                                                    xaxis=dict(
                                                        title='Velocidade (km/h)',
                                                        range=[0, _V0 * 1.1],
                                                        gridcolor='rgba(255,255,255,0.1)',
                                                        color='white'
                                                    ),
                                                    yaxis=dict(
                                                        title='Pico de Aceleração (m/s²)',
                                                        range=[0, _F0 * 1.25],
                                                        gridcolor='rgba(255,255,255,0.1)',
                                                        color='white'
                                                    ),
                                                    paper_bgcolor='rgba(0,0,0,0)',
                                                    plot_bgcolor='rgba(20,20,30,0.85)',
                                                    legend=dict(font=dict(color='white')),
                                                    height=390
                                                )
                                                st.plotly_chart(_fig_fv, use_container_width=True)

                                                _fvc1, _fvc2, _fvc3, _fvc4 = st.columns(4)
                                                with _fvc1:
                                                    st.metric("F₀ — acc teórica em v=0",
                                                              f"{_F0:.2f} m/s²",
                                                              help="Aceleração máxima extrapolada para velocidade zero.")
                                                with _fvc2:
                                                    st.metric("V₀ — vel. máx teórica",
                                                              f"{_V0:.1f} km/h",
                                                              help="Velocidade onde a força propulsora cai a zero.")
                                                with _fvc3:
                                                    st.metric("Pmax (potência rel.)",
                                                              f"{_Pmax:.2f} m²/s³",
                                                              help="Potência máxima relativa: F₀ × V₀ / 4.")
                                                with _fvc4:
                                                    st.metric("Perfil", _pfil)
                                                st.caption(
                                                    f"{_pdesc}  |  "
                                                    f"Slope F-V: {_sl:.3f} (m/s²)/(km/h)  |  "
                                                    f"{len(_fv_pv)} bins usados"
                                                )
                                            else:
                                                st.info("Dados insuficientes para o perfil F-V "
                                                        "(mínimo: 3 bins de velocidade com aceleração positiva).")
                                        else:
                                            st.info("Dados de velocidade/aceleração insuficientes "
                                                    "para este período.")

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
                                        st.markdown("---")
                                        st.info(
                                            "ℹ️ Selecione **2 ou mais atletas** no topo desta "
                                            "seção para ativar a análise de distância entre eles."
                                        )

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
                        tipo_esforco = st.radio("Tipo de esforço:", ["Velocidade", "Aceleração"], horizontal=True)
                        
                        efforts_df = pd.DataFrame()
                        if _esf_modo_todos:
                            _esf_raw = []
                            for _pk in dados_sensor_por_atleta_por_periodo.keys():
                                if tipo_esforco == "Velocidade":
                                    _esf_raw += dados_efforts_vel_por_periodo.get(_pk, {}).get(atleta_escolhido, [])
                                else:
                                    _esf_raw += dados_efforts_acc_por_periodo.get(_pk, {}).get(atleta_escolhido, [])
                            if _esf_raw:
                                efforts_df = (processar_efforts_velocidade(_esf_raw)
                                              if tipo_esforco == "Velocidade"
                                              else processar_efforts_aceleracao(_esf_raw))
                        else:
                            if tipo_esforco == "Velocidade":
                                if atleta_escolhido in dados_efforts_vel_por_periodo.get(periodo_escolhido, {}):
                                    efforts_df = processar_efforts_velocidade(dados_efforts_vel_por_periodo[periodo_escolhido][atleta_escolhido])
                            else:
                                if atleta_escolhido in dados_efforts_acc_por_periodo.get(periodo_escolhido, {}):
                                    efforts_df = processar_efforts_aceleracao(dados_efforts_acc_por_periodo[periodo_escolhido][atleta_escolhido])
                        
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

                        # ── Timeline técnico-físico com eventos futebol ───────
                        st.markdown("---")
                        st.markdown("### ⚽ Timeline Técnico-Físico")
                        if _esf_modo_todos:
                            _ev_atleta = {}
                            for _pk in dados_sensor_por_atleta_por_periodo.keys():
                                for _tipo_ev, _lista_ev in dados_eventos_por_periodo.get(_pk, {}).get(atleta_escolhido, {}).items():
                                    _ev_atleta.setdefault(_tipo_ev, [])
                                    _ev_atleta[_tipo_ev] += _lista_ev
                        else:
                            _ev_atleta = dados_eventos_por_periodo.get(
                                periodo_escolhido, {}).get(atleta_escolhido, {})

                        if _ev_atleta:
                            _todos_tipos = list(_ev_atleta.keys())
                            _tipos_timeline = st.multiselect(
                                "Eventos a marcar na timeline:",
                                options=_todos_tipos,
                                default=_todos_tipos,
                                format_func=lambda k: FUTEBOL_EVENTS_CONFIG[k]['label'],
                                key="tipos_timeline"
                            )
                            if _tipos_timeline:
                                fig_tl = criar_timeline_eventos(
                                    sensor_points, _ev_atleta, atleta_escolhido, _tipos_timeline)
                                if fig_tl:
                                    st.plotly_chart(fig_tl, use_container_width=True)

                                # ── Fadiga + recuperação pós-evento ───────────
                                st.markdown("### 💪 Fadiga & Recuperação por Evento")
                                janela_s = st.slider(
                                    "Janela de análise (segundos antes/depois):",
                                    5, 30, 10, key="janela_fadiga"
                                )
                                df_fadiga = analisar_fadiga_eventos(
                                    sensor_points, _ev_atleta, _tipos_timeline, janela_s)
                                if not df_fadiga.empty:
                                    # Destaque colorido: variação negativa = vermelho
                                    def _cor_variacao(v):
                                        if v is None or pd.isna(v):
                                            return ''
                                        return 'color: #F44336' if v < 0 else 'color: #4CAF50'
                                    st.caption(
                                        "🟢 Variação positiva = atleta acelerou após o evento  "
                                        "🔴 Variação negativa = atleta desacelerou (fadiga/impacto)"
                                    )
                                    styled = df_fadiga.style.map(
                                        _cor_variacao, subset=['Variação (km/h)'])
                                    st.dataframe(styled, use_container_width=True, height=360)
                                    st.download_button(
                                        "📥 Exportar análise de fadiga",
                                        df_fadiga.to_csv(index=False),
                                        f"fadiga_{atleta_escolhido}_{'todos' if _esf_modo_todos else periodo_escolhido}.csv"
                                    )
                                else:
                                    st.info("Nenhum dado de fadiga calculado para os eventos selecionados.")
                        else:
                            st.info("Nenhum evento de futebol carregado para este atleta. "
                                    "Ative os eventos na sidebar e recarregue os dados.")
                else:
                    st.info("Dados de sensor não disponíveis")

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
                        # ── Paleta de cores por atleta ────────────────────────
                        _AV_PALETTE = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4']
                        _av_cores = {a: _AV_PALETTE[i % len(_AV_PALETTE)] for i, a in enumerate(_av_atls_sel)}

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
                            # ════════════════════════════════════════════════
                            # SEÇÃO 1 — SCATTER ACC × VEL (multi-atleta)
                            # ════════════════════════════════════════════════
                            st.markdown("### 📍 Scatter Aceleração × Velocidade")
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
                else:
                    st.info("Busque os dados de atletas na sessão antes de usar esta análise.")

            # ══════════════════════════════════════════════════════════════
            # ABA 7: TABELA DESCRITIVA
            # ══════════════════════════════════════════════════════════════
            with abas[6]:
                st.subheader("📋 Tabela Descritiva de Desempenho")
                st.caption("Coloração por percentil do grupo: 🟢 Top 33% · 🟡 Médio · 🔴 Bottom 33%")

                _td_periodos = list(resultados_por_periodo.keys())
                if _td_periodos:
                    _td_periodo_sel = st.selectbox("Período:", _td_periodos, key="td_periodo_sel")

                    if resultados_por_periodo.get(_td_periodo_sel):
                        _df_td = pd.DataFrame(resultados_por_periodo[_td_periodo_sel])

                        # Calcula %Vmax relativo ao máximo do grupo
                        if 'Velocidade Máx (km/h)' in _df_td.columns:
                            _vmax_grupo = _df_td['Velocidade Máx (km/h)'].max()
                            _df_td['%Vmax'] = (_df_td['Velocidade Máx (km/h)'] / _vmax_grupo * 100).round(1) if _vmax_grupo > 0 else 0

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
                        )
                        st.dataframe(_styled_td, use_container_width=True, hide_index=True)

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
            with abas[7]:
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

                            # Paleta de cores por posição
                            _POS_CORES = {
                                'Zagueiro':    '#1565C0', 'Lateral':    '#0288D1',
                                'Volante':     '#2E7D32', 'Meio campo': '#558B2F',
                                'Atacante':    '#E53935', 'Goleiro':    '#6A1B9A',
                            }
                            _cores_pos = [_POS_CORES.get(p, '#546E7A') for p in _df_pos_grp['Posição']]

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

                            # Tabela resumo por posição
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
                    else:
                        st.info("Nenhum dado disponível para este período.")
                else:
                    st.info("Carregue os dados para visualizar a análise por posição.")

            # ══════════════════════════════════════════════════════════════
            # ABA 9: HISTÓRIA DO JOGO
            # ══════════════════════════════════════════════════════════════
            with abas[8]:
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
                        _hist_default = [a for a in (_hist_atletas_xy[:8] or _hist_atletas_disp[:8])]
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
                            _hist_speed = st.select_slider(
                                "⚡ Velocidade:", options=[10, 30, 60, 120],
                                value=30, format_func=lambda x: f"{x}×",
                                key="hist_speed"
                            )
                        with _col_hc2:
                            _hist_trail_s = st.select_slider(
                                "🌊 Rastro (s):", options=[3, 10, 20, 30],
                                value=10, key="hist_trail"
                            )
                        with _col_hc3:
                            _hist_max_frames = st.select_slider(
                                "🎞️ Qualidade:", options=[150, 300, 500],
                                value=300, format_func=lambda x: f"{x} frames",
                                key="hist_quality"
                            )

                        # ── Construir dados por atleta ─────────────────────
                        _HIST_COLORS = [
                            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                            '#FFEAA7', '#DDA0DD', '#98FB98', '#FFB347',
                            '#87CEEB', '#F08080', '#90EE90', '#FFD700',
                        ]

                        _hist_coords = {}
                        for _ha in _hist_atletas_sel:
                            _hd = dados_posicao_por_periodo[_hist_periodo_sel].get(_ha, {})
                            _hxs = list(_hd.get('xs', []))
                            _hys = list(_hd.get('ys', []))
                            _hvl = list(_hd.get('vel', []))
                            _hts = list(_hd.get('ts_pos', []))
                            if _hxs and _hts and len(_hxs) == len(_hts):
                                _hist_coords[_ha] = {
                                    'xs': _hxs, 'ys': _hys,
                                    'vel': _hvl if len(_hvl) == len(_hxs) else [0.0]*len(_hxs),
                                    'ts': _hts,
                                }

                        if not _hist_coords:
                            _n_sem_xy = len([a for a in _hist_atletas_sel
                                             if not dados_posicao_por_periodo
                                                .get(_hist_periodo_sel, {})
                                                .get(a, {}).get('xs')])
                            st.warning(
                                f"⚠️ **{_n_sem_xy}/{len(_hist_atletas_sel)} atleta(s) sem dados de campo (x,y)** "
                                f"para o período **{_hist_periodo_sel}**.\n\n"
                                "A animação requer coordenadas de campo. Possíveis causas:\n"
                                "- O campo não está configurado no sistema Catapult para este período\n"
                                "- Os sensores não capturaram dados de posição de campo\n"
                                "- Tente selecionar o período **1 Tempo** — geralmente tem mais dados xy"
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

                            # ── Timeline comum e downsample ────────────────
                            _all_ts_norm = sorted({
                                t for _hc in _hist_coords.values() for t in _hc['ts_norm']
                            })
                            _n_all = len(_all_ts_norm)
                            _step_fr = max(1, _n_all // _hist_max_frames)
                            _frame_ts = _all_ts_norm[::_step_fr]
                            _n_frames = len(_frame_ts)
                            _frame_dur_ms = max(30, int(_step_fr * 100 // _hist_speed))

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

                            # Trail: rastro animado (começa vazio)
                            for _i, _ha in enumerate(_hist_atl_list):
                                _col = _HIST_COLORS[_i % len(_HIST_COLORS)]
                                _fig_hist.add_trace(go.Scatter(
                                    x=[], y=[],
                                    mode='lines',
                                    line=dict(color=_col, width=3),
                                    opacity=0.75,
                                    name=_ha, showlegend=False, hoverinfo='skip',
                                ))

                            # Marker: posição atual animada
                            for _i, _ha in enumerate(_hist_atl_list):
                                _hc = _hist_coords[_ha]
                                _col = _HIST_COLORS[_i % len(_HIST_COLORS)]
                                _pos_lbl = dados_posicao_por_periodo[_hist_periodo_sel].get(_ha, {}).get('posicao', '')
                                _eq_lbl  = dados_posicao_por_periodo[_hist_periodo_sel].get(_ha, {}).get('equipe', '')
                                _fig_hist.add_trace(go.Scatter(
                                    x=[_hc['xs'][0]], y=[_hc['ys'][0]],
                                    mode='markers+text',
                                    marker=dict(
                                        size=14, color=_col,
                                        line=dict(color='white', width=1.5),
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

                            # ── Índices dos traces animados ────────────────
                            # ghost  → n_static + i
                            # trail  → n_static + n_ha + i
                            # marker → n_static + 2*n_ha + i

                            # ── Montar frames ──────────────────────────────
                            _frames_list = []
                            for _fi, _fts in enumerate(_frame_ts):
                                _frame_data   = []
                                _frame_traces = []
                                for _i, _ha in enumerate(_hist_atl_list):
                                    _hc = _hist_coords[_ha]
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

                                    # Trail trace update
                                    _frame_data.append(go.Scatter(x=_trail_x, y=_trail_y))
                                    _frame_traces.append(_n_static + _n_ha + _i)

                                    # Marker trace update
                                    _frame_data.append(go.Scatter(
                                        x=[_cur_x], y=[_cur_y],
                                        text=[_ha.split(' ')[0]],
                                        hovertemplate=(
                                            f"<b>{_ha}</b><br>"
                                            f"Vel: {_cur_v:.1f} km/h<extra></extra>"
                                        ),
                                    ))
                                    _frame_traces.append(_n_static + 2 * _n_ha + _i)

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
                                margin=dict(t=60, b=130, l=10, r=10),
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
                                    y=1.06, x=0.5, xanchor='center',
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
                                f"⚙️ {_n_frames} frames · {_hist_speed}× velocidade · "
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
                                    "🔗 Relações",
                                    "🤖 Automática",
                                ])

                                # ──────────────────────────────────────────────
                                # TAB ESTRUTURA  (1, 2, 3, 5)
                                # ──────────────────────────────────────────────
                                with _tac_tabs[0]:
                                    st.markdown("### 🏗️ Estrutura da Equipa")

                                    # 1 — Centróide
                                    st.markdown("#### 1 — Centróide da Equipa ao Longo do Tempo")
                                    _ccol1, _ccol2 = st.columns(2)
                                    with _ccol1:
                                        _fig_ct = go.Figure()
                                        _fig_ct.add_trace(go.Scatter(
                                            x=list(_ft_arr), y=list(_ctr_x),
                                            name='X', line=dict(color='#4ECDC4', width=2),
                                        ))
                                        _fig_ct.add_trace(go.Scatter(
                                            x=list(_ft_arr), y=list(_ctr_y),
                                            name='Y', line=dict(color='#FFB347', width=2),
                                        ))
                                        _fig_ct.update_layout(
                                            xaxis_title='Tempo (s)', yaxis_title='Posição (m)',
                                            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                            font=dict(color='white'), height=270,
                                            legend=dict(font=dict(color='white')),
                                            margin=dict(t=20, b=30, l=40, r=10),
                                        )
                                        st.plotly_chart(_fig_ct, use_container_width=True)
                                    with _ccol2:
                                        _fig_ctf = desenhar_campo_futebol_bonito(
                                            field_length=_hist_fl, field_width=_hist_fw,
                                            title="Trajeto do Centróide"
                                        )
                                        _fig_ctf.add_trace(go.Scatter(
                                            x=list(_ctr_x), y=list(_ctr_y),
                                            mode='lines', line=dict(color='yellow', width=2),
                                            opacity=0.8, showlegend=False, hoverinfo='skip',
                                        ))
                                        _fig_ctf.add_trace(go.Scatter(
                                            x=[float(_ctr_x[0])], y=[float(_ctr_y[0])],
                                            mode='markers', marker=dict(color='lime', size=11, symbol='star'),
                                            name='Início', showlegend=False,
                                        ))
                                        _fig_ctf.add_trace(go.Scatter(
                                            x=[float(_ctr_x[-1])], y=[float(_ctr_y[-1])],
                                            mode='markers', marker=dict(color='red', size=11, symbol='square'),
                                            name='Fim', showlegend=False,
                                        ))
                                        _fig_ctf.update_layout(
                                            height=290,
                                            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                            font=dict(color='white'),
                                            margin=dict(t=35, b=5, l=5, r=5),
                                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4, _hist_fl+4]),
                                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-4, _hist_fw+4], scaleanchor='x', scaleratio=1),
                                        )
                                        st.plotly_chart(_fig_ctf, use_container_width=True)
                                    _mc1, _mc2, _mc3 = st.columns(3)
                                    _mc1.metric("Pos. X média", f"{float(_ctr_x.mean()):.1f} m",
                                                f"{float(_ctr_x.mean())/_hist_fl*100:.0f}% do campo")
                                    _mc2.metric("Pos. Y média", f"{float(_ctr_y.mean()):.1f} m")
                                    _mc3.metric("Variação X (std)", f"{float(_ctr_x.std()):.1f} m")

                                    st.markdown("---")

                                    # 2 — Largura e Profundidade
                                    st.markdown("#### 2 — Largura e Profundidade Dinâmica")
                                    _fig_wd = go.Figure()
                                    _fig_wd.add_trace(go.Scatter(
                                        x=list(_ft_arr), y=list(_width),
                                        name='Largura (Y)', line=dict(color='#FF6B6B', width=2),
                                        fill='tozeroy', fillcolor='rgba(255,107,107,0.15)',
                                    ))
                                    _fig_wd.add_trace(go.Scatter(
                                        x=list(_ft_arr), y=list(_depth),
                                        name='Profundidade (X)', line=dict(color='#45B7D1', width=2),
                                        fill='tozeroy', fillcolor='rgba(69,183,209,0.15)',
                                    ))
                                    _fig_wd.update_layout(
                                        xaxis_title='Tempo (s)', yaxis_title='Metros',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=270,
                                        legend=dict(font=dict(color='white')),
                                        margin=dict(t=10, b=30, l=40, r=10),
                                    )
                                    st.plotly_chart(_fig_wd, use_container_width=True)
                                    _mw1, _mw2, _mw3, _mw4 = st.columns(4)
                                    _mw1.metric("Largura média", f"{float(_width.mean()):.1f} m")
                                    _mw2.metric("Profundidade média", f"{float(_depth.mean()):.1f} m")
                                    _mw3.metric("Largura máx", f"{float(_width.max()):.1f} m")
                                    _mw4.metric("Profundidade máx", f"{float(_depth.max()):.1f} m")

                                    st.markdown("---")

                                    # 3 — Compacidade
                                    st.markdown("#### 3 — Índice de Compacidade")
                                    _fig_cmp = go.Figure()
                                    _cmp_avg = float(_compact.mean())
                                    _fig_cmp.add_trace(go.Scatter(
                                        x=list(_ft_arr), y=list(_compact),
                                        mode='lines', line=dict(color='#96CEB4', width=2),
                                        fill='tozeroy', fillcolor='rgba(150,206,180,0.2)',
                                    ))
                                    _fig_cmp.add_hline(y=_cmp_avg, line_dash='dash', line_color='yellow',
                                                       annotation_text=f"Média: {_cmp_avg:.1f}m",
                                                       annotation_font_color='yellow')
                                    _fig_cmp.update_layout(
                                        xaxis_title='Tempo (s)', yaxis_title='Dispersão (m)',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=260,
                                        margin=dict(t=20, b=30, l=40, r=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_cmp, use_container_width=True)
                                    _mp1, _mp2, _mp3 = st.columns(3)
                                    _mp1.metric("Compacidade média", f"{_cmp_avg:.1f} m")
                                    _mp2.metric("P10 (mais compacto)", f"{float(np.percentile(_compact,10)):.1f} m")
                                    _mp3.metric("P90 (mais disperso)", f"{float(np.percentile(_compact,90)):.1f} m")

                                    st.markdown("---")

                                    # 5 — Linhas Táticas
                                    st.markdown("#### 5 — Linhas Táticas (Defesa / Meio / Ataque)")
                                    _avg_x_atl = _sync_xs.mean(axis=1)
                                    _sorted_idx = np.argsort(_avg_x_atl)
                                    _n3 = max(1, _n_ha // 3)
                                    _def_idx = _sorted_idx[:_n3]
                                    _mid_idx = _sorted_idx[_n3:2*_n3]
                                    _att_idx = _sorted_idx[2*_n3:]
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

                                    # 7 — Heatmap coletivo por fase
                                    st.markdown("#### 7 — Mapa de Calor Coletivo por Fase")
                                    _n_fases = st.select_slider("Fases:", options=[2,3,4,5], value=3, key="tac_fases")
                                    _fase_sz  = _n_frames // _n_fases
                                    _hm_cols  = st.columns(_n_fases)
                                    for _fi_ph in range(_n_fases):
                                        _st_ph = _fi_ph * _fase_sz
                                        _en_ph = (_fi_ph + 1)*_fase_sz if _fi_ph < _n_fases-1 else _n_frames
                                        _ph_xs = _sync_xs[:, _st_ph:_en_ph].flatten().tolist()
                                        _ph_ys = _sync_ys[:, _st_ph:_en_ph].flatten().tolist()
                                        _t0_ph = _frame_ts[_st_ph]
                                        _t1_ph = _frame_ts[min(_en_ph-1, _n_frames-1)]
                                        with _hm_cols[_fi_ph]:
                                            _fig_ph = desenhar_campo_futebol_bonito(
                                                field_length=_hist_fl, field_width=_hist_fw,
                                                title=f"Fase {_fi_ph+1}  {int(_t0_ph//60):02d}:{int(_t0_ph%60):02d}–{int(_t1_ph//60):02d}:{int(_t1_ph%60):02d}"
                                            )
                                            if _ph_xs:
                                                _fig_ph.add_trace(go.Histogram2dContour(
                                                    x=_ph_xs, y=_ph_ys,
                                                    colorscale='Hot', reversescale=True,
                                                    showscale=False, ncontours=12,
                                                    line=dict(width=0),
                                                    contours=dict(coloring='fill'),
                                                    opacity=0.65,
                                                    xbins=dict(start=0, end=_hist_fl, size=5),
                                                    ybins=dict(start=0, end=_hist_fw, size=5),
                                                ))
                                            _fig_ph.update_layout(
                                                height=270,
                                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                                font=dict(color='white', size=9),
                                                margin=dict(t=35, b=5, l=5, r=5),
                                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, _hist_fl+3]),
                                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, _hist_fw+3], scaleanchor='x', scaleratio=1),
                                            )
                                            st.plotly_chart(_fig_ph, use_container_width=True)

                                    st.markdown("---")

                                    # 8 — Corredor dominante
                                    st.markdown("#### 8 — Corredor Dominante")
                                    _y3 = _hist_fw / 3.0
                                    _n_left   = (_sync_ys < _y3).sum(axis=0).astype(float)
                                    _n_center = ((_sync_ys >= _y3) & (_sync_ys < 2*_y3)).sum(axis=0).astype(float)
                                    _n_right  = (_sync_ys >= 2*_y3).sum(axis=0).astype(float)
                                    _fig_cor  = go.Figure()
                                    _fig_cor.add_trace(go.Scatter(x=list(_ft_arr), y=list(_n_left),
                                        name='Corredor Esq.', stackgroup='cor',
                                        line=dict(color='#FF6B6B'), fillcolor='rgba(255,107,107,0.5)'))
                                    _fig_cor.add_trace(go.Scatter(x=list(_ft_arr), y=list(_n_center),
                                        name='Corredor Central', stackgroup='cor',
                                        line=dict(color='#4ECDC4'), fillcolor='rgba(78,205,196,0.5)'))
                                    _fig_cor.add_trace(go.Scatter(x=list(_ft_arr), y=list(_n_right),
                                        name='Corredor Dir.', stackgroup='cor',
                                        line=dict(color='#45B7D1'), fillcolor='rgba(69,183,209,0.5)'))
                                    _fig_cor.update_layout(
                                        xaxis_title='Tempo (s)', yaxis_title='Nº Atletas',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=270,
                                        legend=dict(font=dict(color='white')),
                                        margin=dict(t=10, b=30, l=40, r=10),
                                    )
                                    st.plotly_chart(_fig_cor, use_container_width=True)
                                    _tot_c = float(_n_left.sum() + _n_center.sum() + _n_right.sum())
                                    if _tot_c > 0:
                                        _pL = _n_left.sum()/_tot_c*100
                                        _pC = _n_center.sum()/_tot_c*100
                                        _pR = _n_right.sum()/_tot_c*100
                                        _dom = "Esquerdo" if _pL==max(_pL,_pC,_pR) else ("Central" if _pC==max(_pL,_pC,_pR) else "Direito")
                                        _cor1, _cor2, _cor3, _cor4 = st.columns(4)
                                        _cor1.metric("Esquerdo", f"{_pL:.0f}%")
                                        _cor2.metric("Central", f"{_pC:.0f}%")
                                        _cor3.metric("Direito", f"{_pR:.0f}%")
                                        _cor4.metric("Dominante", _dom)

                                    st.markdown("---")

                                    # 9 — Gaps Táticos (Delaunay)
                                    st.markdown("#### 9 — Espaços Descobertos (Gaps Táticos)")
                                    _gap_thr = st.slider("Threshold de gap (m²):", 50, 500, 200, step=50, key="tac_gap")
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

                                    # 13 — Sincronização de Velocidade
                                    st.markdown("#### 13 — Sincronização de Velocidade da Equipa")
                                    _corr = np.corrcoef(_sync_vl)
                                    _short_names = [a.split(' ')[0] for a in _hist_atl_list]
                                    _fig_cr = go.Figure(go.Heatmap(
                                        z=_corr.tolist(),
                                        x=_short_names, y=_short_names,
                                        colorscale='RdBu', zmin=-1, zmax=1,
                                        text=[[f"{_corr[r,c]:.2f}" for c in range(_n_ha)] for r in range(_n_ha)],
                                        texttemplate='%{text}', textfont=dict(size=9),
                                        colorbar=dict(title='r', tickfont=dict(color='white'), titlefont=dict(color='white')),
                                    ))
                                    _fig_cr.update_layout(
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=max(300, _n_ha*35+80),
                                        margin=dict(t=20, b=30, l=80, r=30),
                                    )
                                    st.plotly_chart(_fig_cr, use_container_width=True)
                                    _upper = np.triu(np.ones_like(_corr, dtype=bool), k=1)
                                    _sync_score = float(_corr[_upper].mean())
                                    _best = np.unravel_index(np.argmax(_corr * _upper), _corr.shape)
                                    _sy1, _sy2 = st.columns(2)
                                    _sy1.metric("Índice de sincronia global", f"{_sync_score:.3f}")
                                    _sy2.metric("Par mais sincronizado",
                                        f"{_short_names[_best[0]]} & {_short_names[_best[1]]}",
                                        f"r = {_corr[_best]:.2f}")

                                    st.markdown("---")

                                    # 14 — Team Speed
                                    st.markdown("#### 14 — Velocidade do Bloco (Team Speed)")
                                    _fig_ts = go.Figure()
                                    _fig_ts.add_trace(go.Scatter(
                                        x=list(_ft_arr), y=list(_team_spd),
                                        line=dict(color='#FFB347', width=2),
                                        fill='tozeroy', fillcolor='rgba(255,179,71,0.2)',
                                    ))
                                    _p75ts = float(np.percentile(_team_spd, 75))
                                    _fig_ts.add_hline(y=_p75ts, line_dash='dot', line_color='#FFD700',
                                                      annotation_text=f"P75: {_p75ts:.1f} km/h",
                                                      annotation_font_color='#FFD700')
                                    _fig_ts.update_layout(
                                        xaxis_title='Tempo (s)', yaxis_title='Vel. média (km/h)',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=255,
                                        margin=dict(t=10, b=30, l=40, r=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_ts, use_container_width=True)
                                    _peak_t = float(_ft_arr[int(np.argmax(_team_spd))])
                                    _ts1, _ts2, _ts3 = st.columns(3)
                                    _ts1.metric("Vel. média global", f"{float(_team_spd.mean()):.1f} km/h")
                                    _ts2.metric("Pico coletivo", f"{float(_team_spd.max()):.1f} km/h")
                                    _ts3.metric("Pico em", f"{int(_peak_t//60):02d}:{int(_peak_t%60):02d}")

                                    st.markdown("---")

                                    # 15 — Compressão Defensiva
                                    st.markdown("#### 15 — Índice de Compressão Defensiva")
                                    _cd_d = st.slider("Profundidade máx (m):", 10, 40, 20, key="tac_cdd")
                                    _cd_w = st.slider("Largura máx (m):", 15, 55, 35, key="tac_cdw")
                                    _cbin = ((_depth <= _cd_d) & (_width <= _cd_w)).astype(float)
                                    _fig_cd = go.Figure()
                                    _fig_cd.add_trace(go.Scatter(x=list(_ft_arr), y=list(_depth),
                                        name='Profundidade', line=dict(color='#45B7D1', width=1.5)))
                                    _fig_cd.add_trace(go.Scatter(x=list(_ft_arr), y=list(_width),
                                        name='Largura', line=dict(color='#FF6B6B', width=1.5)))
                                    _in_c = False
                                    _c_regs = []
                                    for _ci in range(_n_frames):
                                        if _cbin[_ci] and not _in_c:
                                            _cs = float(_ft_arr[_ci]); _in_c = True
                                        elif not _cbin[_ci] and _in_c:
                                            _c_regs.append((_cs, float(_ft_arr[_ci]))); _in_c = False
                                    if _in_c: _c_regs.append((_cs, float(_ft_arr[-1])))
                                    for _cr0, _cr1 in _c_regs[:25]:
                                        _fig_cd.add_vrect(x0=_cr0, x1=_cr1,
                                            fillcolor='rgba(150,206,180,0.25)', line_width=0)
                                    _fig_cd.add_hline(y=_cd_d, line_dash='dash', line_color='#45B7D1', line_width=1)
                                    _fig_cd.add_hline(y=_cd_w, line_dash='dash', line_color='#FF6B6B', line_width=1)
                                    _fig_cd.update_layout(
                                        xaxis_title='Tempo (s)', yaxis_title='Metros',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=270,
                                        legend=dict(font=dict(color='white')),
                                        margin=dict(t=10, b=30, l=40, r=10),
                                    )
                                    st.plotly_chart(_fig_cd, use_container_width=True)
                                    _cd1, _cd2, _cd3 = st.columns(3)
                                    _cd1.metric("Tempo em bloco compacto", f"{float(_cbin.mean())*100:.1f}%")
                                    _cd2.metric("Episódios de compressão", f"{len(_c_regs)}")
                                    _cd3.metric("Profundidade média", f"{float(_depth.mean()):.1f} m")

                                # ──────────────────────────────────────────────
                                # TAB RELAÇÕES  (16, 17, 18, 19)
                                # ──────────────────────────────────────────────
                                with _tac_tabs[3]:
                                    st.markdown("### 🔗 Relações Entre Jogadores")

                                    # 16 — Rede de Proximidade
                                    st.markdown("#### 16 — Rede de Proximidade")
                                    _net_d = st.slider("Distância de ligação (m):", 5, 30, 15, key="tac_netd")
                                    _net_f = st.select_slider(
                                        "Frame (tempo):",
                                        options=list(range(_n_frames)), value=_n_frames//2,
                                        format_func=lambda x: f"{int(_frame_ts[x]//60):02d}:{int(_frame_ts[x]%60):02d}",
                                        key="tac_netf"
                                    )
                                    _nfx = _sync_xs[:, _net_f]
                                    _nfy = _sync_ys[:, _net_f]
                                    _fig_net = desenhar_campo_futebol_bonito(
                                        field_length=_hist_fl, field_width=_hist_fw,
                                        title=f"Rede de Proximidade — {int(_frame_ts[_net_f]//60):02d}:{int(_frame_ts[_net_f]%60):02d}"
                                    )
                                    _degs = [0]*_n_ha
                                    for _na in range(_n_ha):
                                        for _nb in range(_na+1, _n_ha):
                                            _dab = float(np.sqrt((_nfx[_na]-_nfx[_nb])**2+(_nfy[_na]-_nfy[_nb])**2))
                                            if _dab <= _net_d:
                                                _degs[_na] += 1; _degs[_nb] += 1
                                                _fig_net.add_trace(go.Scatter(
                                                    x=[float(_nfx[_na]), float(_nfx[_nb])],
                                                    y=[float(_nfy[_na]), float(_nfy[_nb])],
                                                    mode='lines',
                                                    line=dict(color='rgba(255,255,100,0.45)', width=2),
                                                    showlegend=False, hoverinfo='skip',
                                                ))
                                    for _ni in range(_n_ha):
                                        _nc = _HIST_COLORS[_ni % len(_HIST_COLORS)]
                                        _fig_net.add_trace(go.Scatter(
                                            x=[float(_nfx[_ni])], y=[float(_nfy[_ni])],
                                            mode='markers+text',
                                            marker=dict(size=12+_degs[_ni]*4, color=_nc,
                                                        line=dict(color='white', width=1.5)),
                                            text=[f"{_hist_atl_list[_ni].split(' ')[0]}({_degs[_ni]})"],
                                            textposition='top center',
                                            textfont=dict(size=8, color='white'),
                                            name=_hist_atl_list[_ni], showlegend=False,
                                        ))
                                    _fig_net.update_layout(
                                        height=400, plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), margin=dict(t=40, b=10, l=10, r=10),
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, _hist_fl+3]),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, _hist_fw+3], scaleanchor='x', scaleratio=1),
                                    )
                                    st.plotly_chart(_fig_net, use_container_width=True)
                                    st.caption("Tamanho proporcional ao grau (nº de conexões).")
                                    st.dataframe(
                                        pd.DataFrame({'Atleta': _hist_atl_list, 'Grau': _degs}).sort_values('Grau', ascending=False),
                                        use_container_width=True, hide_index=True
                                    )

                                    st.markdown("---")

                                    # 17 — Distância ao Companheiro Mais Próximo
                                    st.markdown("#### 17 — Distância ao Companheiro Mais Próximo")
                                    _mdists = np.full((_n_ha, _n_frames), np.inf)
                                    for _di in range(_n_ha):
                                        for _dj in range(_n_ha):
                                            if _di == _dj: continue
                                            _d = np.sqrt((_sync_xs[_di]-_sync_xs[_dj])**2+(_sync_ys[_di]-_sync_ys[_dj])**2)
                                            _mdists[_di] = np.minimum(_mdists[_di], _d)
                                    _mdists = np.where(np.isinf(_mdists), 0, _mdists)
                                    _fig_md = go.Figure()
                                    for _di2, _ha2 in enumerate(_hist_atl_list):
                                        _fig_md.add_trace(go.Scatter(
                                            x=list(_ft_arr), y=list(_mdists[_di2]),
                                            name=_ha2.split(' ')[0],
                                            line=dict(color=_HIST_COLORS[_di2%len(_HIST_COLORS)], width=1.5),
                                            opacity=0.8,
                                        ))
                                    _fig_md.add_trace(go.Scatter(
                                        x=list(_ft_arr), y=list(_mdists.mean(axis=0)),
                                        name='Média equipa', line=dict(color='white', width=2.5, dash='dash'),
                                    ))
                                    _fig_md.update_layout(
                                        xaxis_title='Tempo (s)', yaxis_title='Distância (m)',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=290,
                                        legend=dict(font=dict(color='white', size=9)),
                                        margin=dict(t=10, b=30, l=40, r=10),
                                    )
                                    st.plotly_chart(_fig_md, use_container_width=True)
                                    st.dataframe(pd.DataFrame({
                                        'Atleta': _hist_atl_list,
                                        'Dist. Média (m)': [round(float(v),1) for v in _mdists.mean(axis=1)],
                                        'Dist. Máx (m)':   [round(float(v),1) for v in _mdists.max(axis=1)],
                                        'Dist. Mín (m)':   [round(float(v),1) for v in _mdists.min(axis=1)],
                                    }).sort_values('Dist. Média (m)', ascending=False),
                                    use_container_width=True, hide_index=True)

                                    st.markdown("---")

                                    # 18 — Triângulos de Passe Naturais
                                    st.markdown("#### 18 — Triângulos de Passe Naturais")
                                    try:
                                        from scipy.spatial import Delaunay as _Del2
                                        _tnx = int(max(10, _hist_fl//5))
                                        _tny = int(max(8, _hist_fw//5))
                                        _tfreq = np.zeros((_tny, _tnx))
                                        _txr = np.linspace(0, _hist_fl, _tnx+1)
                                        _tyr = np.linspace(0, _hist_fw, _tny+1)
                                        _ts_step = max(1, _n_frames//100)
                                        for _tfi2 in range(0, _n_frames, _ts_step):
                                            _tp = np.column_stack([_sync_xs[:,_tfi2], _sync_ys[:,_tfi2]])
                                            if len(_tp) >= 3:
                                                _tt = _Del2(_tp)
                                                for _ts2 in _tt.simplices:
                                                    _cx2 = float(_tp[_ts2,0].mean())
                                                    _cy2 = float(_tp[_ts2,1].mean())
                                                    _ix2 = int(np.clip(np.searchsorted(_txr,_cx2)-1, 0, _tnx-1))
                                                    _iy2 = int(np.clip(np.searchsorted(_tyr,_cy2)-1, 0, _tny-1))
                                                    _tfreq[_iy2,_ix2] += 1
                                        _fig_trf = go.Figure(go.Heatmap(
                                            z=_tfreq.tolist(),
                                            x=[(_txr[i]+_txr[i+1])/2 for i in range(_tnx)],
                                            y=[(_tyr[i]+_tyr[i+1])/2 for i in range(_tny)],
                                            colorscale='YlOrRd',
                                            colorbar=dict(title='Freq.', tickfont=dict(color='white'), titlefont=dict(color='white')),
                                        ))
                                        _fig_trf.add_shape(type='rect', x0=0, y0=0, x1=_hist_fl, y1=_hist_fw,
                                                           line=dict(color='white', width=2))
                                        _fig_trf.add_shape(type='line', x0=_hist_fl/2, y0=0, x1=_hist_fl/2, y1=_hist_fw,
                                                           line=dict(color='white', width=1))
                                        _fig_trf.update_layout(
                                            title='Frequência de triângulos de passe por zona',
                                            plot_bgcolor='#1a3a18', paper_bgcolor='#0e1117',
                                            font=dict(color='white'), height=320,
                                            margin=dict(t=45, b=20, l=20, r=60),
                                            xaxis=dict(showgrid=False, zeroline=False, range=[-3, _hist_fl+3]),
                                            yaxis=dict(showgrid=False, zeroline=False, range=[-3, _hist_fw+3], scaleanchor='x', scaleratio=1),
                                        )
                                        st.plotly_chart(_fig_trf, use_container_width=True)
                                        st.caption("Zonas quentes = onde a equipa forma triângulos com maior frequência.")
                                    except Exception as _etri:
                                        st.warning(f"scipy indisponível para triângulos: {_etri}")

                                    st.markdown("---")

                                    # 19 — Assimetria Lateral
                                    st.markdown("#### 19 — Assimetria Lateral da Equipa")
                                    _fig_as = go.Figure()
                                    _fig_as.add_trace(go.Scatter(
                                        x=list(_ft_arr), y=list(_lat_asym),
                                        mode='lines', line=dict(color='#DDA0DD', width=2),
                                        fill='tozeroy', fillcolor='rgba(221,160,221,0.2)',
                                    ))
                                    _fig_as.add_hline(y=0, line_dash='dash', line_color='white', line_width=1)
                                    _fig_as.update_layout(
                                        xaxis_title='Tempo (s)',
                                        yaxis_title='Desvio do eixo central (m)',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=255,
                                        margin=dict(t=10, b=30, l=60, r=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_as, use_container_width=True)
                                    _asym_avg = float(_lat_asym.mean())
                                    _a_side   = "Esquerdo 👈" if _asym_avg < -1 else ("Direito 👉" if _asym_avg > 1 else "Equilibrado ⚖️")
                                    _pL2 = float((_lat_asym < -1).mean())*100
                                    _pR2 = float((_lat_asym >  1).mean())*100
                                    _as1, _as2, _as3, _as4 = st.columns(4)
                                    _as1.metric("Lado dominante", _a_side)
                                    _as2.metric("Flanco Esquerdo", f"{_pL2:.0f}%")
                                    _as3.metric("Equilibrado", f"{100-_pL2-_pR2:.0f}%")
                                    _as4.metric("Flanco Direito", f"{_pR2:.0f}%")

                                # ──────────────────────────────────────────────
                                # TAB AUTOMÁTICA  (20)
                                # ──────────────────────────────────────────────
                                with _tac_tabs[4]:
                                    st.markdown("### 🤖 Fase de Jogo Automática")
                                    st.markdown("#### 20 — Clustering Temporal (K-Means)")
                                    _km_k = st.select_slider("Nº de fases (k):", options=[2,3,4,5,6], value=4, key="tac_kmk")

                                    def _mm(arr):
                                        _r = arr.max() - arr.min()
                                        return (arr - arr.min()) / _r if _r > 0 else arr * 0.0

                                    _km_X = np.column_stack([_mm(_ctr_x), _mm(_ctr_y), _mm(_compact), _mm(_team_spd)])
                                    _rng_km = np.random.default_rng(42)
                                    _km_cents = _km_X[_rng_km.choice(_n_frames, size=_km_k, replace=False)].copy()
                                    _km_lbl   = np.zeros(_n_frames, dtype=int)
                                    for _ in range(60):
                                        _km_d   = np.array([np.linalg.norm(_km_X - _km_cents[k], axis=1) for k in range(_km_k)])
                                        _km_new = _km_d.argmin(axis=0)
                                        if np.all(_km_new == _km_lbl): break
                                        _km_lbl = _km_new
                                        for k in range(_km_k):
                                            _mk = _km_lbl == k
                                            if _mk.any(): _km_cents[k] = _km_X[_mk].mean(axis=0)

                                    _km_cx_avg  = [float(_ctr_x[_km_lbl==k].mean()) if (_km_lbl==k).any() else 0.0 for k in range(_km_k)]
                                    _km_ord     = np.argsort(_km_cx_avg)
                                    _km_ph_map  = {}
                                    if _km_k >= 2:
                                        _km_ph_map[int(_km_ord[0])]  = "🔴 Bloco Baixo"
                                        _km_ph_map[int(_km_ord[-1])] = "🟢 Pressão Alta"
                                    if _km_k >= 3:
                                        _km_ph_map[int(_km_ord[_km_k//2])] = "🟡 Fase Média"
                                    _km_colors  = ['#FF6B6B','#FFEAA7','#4ECDC4','#96CEB4','#DDA0DD','#45B7D1']
                                    _km_lbl_map = {k: _km_ph_map.get(k, f"🔵 Fase {k+1}") for k in range(_km_k)}

                                    _fig_km = go.Figure()
                                    for k in range(_km_k):
                                        _mk = _km_lbl == k
                                        if not _mk.any(): continue
                                        _fig_km.add_trace(go.Scatter(
                                            x=list(_ft_arr[_mk]), y=[_km_cx_avg[k]]*int(_mk.sum()),
                                            mode='markers',
                                            marker=dict(color=_km_colors[k%len(_km_colors)], size=7, symbol='square'),
                                            name=_km_lbl_map[k],
                                        ))
                                    _fig_km.update_layout(
                                        title='Fases de Jogo ao Longo do Tempo',
                                        xaxis_title='Tempo (s)', yaxis_title='Centróide X (m)',
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), height=270,
                                        legend=dict(font=dict(color='white')),
                                        margin=dict(t=40, b=30, l=50, r=10),
                                    )
                                    st.plotly_chart(_fig_km, use_container_width=True)

                                    _km_counts = [int((_km_lbl==k).sum()) for k in range(_km_k)]
                                    _kc1, _kc2 = st.columns([1,2])
                                    with _kc1:
                                        _fig_pie = go.Figure(go.Pie(
                                            labels=[_km_lbl_map[k] for k in range(_km_k)],
                                            values=_km_counts,
                                            marker=dict(colors=[_km_colors[k%len(_km_colors)] for k in range(_km_k)]),
                                            textfont=dict(color='white'),
                                        ))
                                        _fig_pie.update_layout(
                                            plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                            font=dict(color='white'), height=260,
                                            margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                                        )
                                        st.plotly_chart(_fig_pie, use_container_width=True)
                                    with _kc2:
                                        _km_rows = []
                                        for k in range(_km_k):
                                            _mk = _km_lbl == k
                                            if not _mk.any(): continue
                                            _km_rows.append({
                                                'Fase': _km_lbl_map[k],
                                                '% Tempo': f"{_mk.sum()/_n_frames*100:.0f}%",
                                                'Duração (s)': f"{int(_mk.sum()*_dt_frame)}",
                                                'Centróide X': f"{float(_ctr_x[_mk].mean()):.1f} m",
                                                'Compacidade': f"{float(_compact[_mk].mean()):.1f} m",
                                                'Vel. Equipa': f"{float(_team_spd[_mk].mean()):.1f} km/h",
                                            })
                                        if _km_rows:
                                            st.dataframe(pd.DataFrame(_km_rows), use_container_width=True, hide_index=True)
                                    st.caption("K-Means sobre: centróide X/Y, compacidade e velocidade média da equipa.")

        else:
            st.warning("Nenhum dado encontrado")
    
    elif 'df_athletes' in st.session_state and not st.session_state.df_athletes.empty:
        st.info("👈 Selecione uma atividade, período(s) e clique em 'Buscar Atletas da Atividade'")

if __name__ == "__main__":
    main()
    