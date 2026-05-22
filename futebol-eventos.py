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

class CatapultAPI:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}
    
    def get_athletes(self):
        r = requests.get(f"{self.base_url}/athletes", headers=self.headers)
        return r.json() if r.status_code == 200 else None
    
    def get_teams(self):
        r = requests.get(f"{self.base_url}/teams", headers=self.headers)
        return r.json() if r.status_code == 200 else None
    
    def get_team_athletes(self, team_id):
        r = requests.get(f"{self.base_url}/teams/{team_id}/athletes", headers=self.headers)
        return r.json() if r.status_code == 200 else None
    
    def get_activities(self):
        r = requests.get(f"{self.base_url}/activities?page_size=500", headers=self.headers)
        return r.json() if r.status_code == 200 else None
    
    def get_activity_athletes(self, activity_id):
        r = requests.get(f"{self.base_url}/activities/{activity_id}/athletes", headers=self.headers)
        return r.json() if r.status_code == 200 else None
    
    def get_activity_periods(self, activity_id):
        r = requests.get(f"{self.base_url}/activities/{activity_id}/periods", headers=self.headers)
        return r.json() if r.status_code == 200 else None
    
    def get_all_periods(self):
        r = requests.get(f"{self.base_url}/periods", headers=self.headers)
        return r.json() if r.status_code == 200 else None
    
    def get_athletes_in_period(self, period_id):
        r = requests.get(f"{self.base_url}/periods/{period_id}/athletes", headers=self.headers)
        return r.json() if r.status_code == 200 else None
    
    def get_positions(self):
        r = requests.get(f"{self.base_url}/positions", headers=self.headers)
        return r.json() if r.status_code == 200 else None
    
    def get_parameters(self):
        r = requests.get(f"{self.base_url}/parameters", headers=self.headers)
        return r.json() if r.status_code == 200 else None
    
    def get_sensor_data(self, activity_id, athlete_id):
        params = {
            "parameters": "ts,lat,long,v,a,hr,pl,xy",
            "nulls": "1"
        }
        r = requests.get(
            f"{self.base_url}/activities/{activity_id}/athletes/{athlete_id}/sensor",
            headers=self.headers,
            params=params,
            timeout=60
        )
        if r.status_code == 200:
            return r.json()
        return None
    
    def get_period_sensor_data(self, period_id, athlete_id):
        params = {
            "parameters": "ts,lat,long,v,a,hr,pl,xy",
            "nulls": "1"
        }
        r = requests.get(
            f"{self.base_url}/periods/{period_id}/athletes/{athlete_id}/sensor",
            headers=self.headers,
            params=params,
            timeout=60
        )
        if r.status_code == 200:
            return r.json()
        return None
    
    def get_activity_efforts(self, activity_id, athlete_id, effort_types="velocity,acceleration"):
        params = {"effort_types": effort_types}
        r = requests.get(
            f"{self.base_url}/activities/{activity_id}/athletes/{athlete_id}/efforts",
            headers=self.headers,
            params=params,
            timeout=60
        )
        if r.status_code == 200:
            return r.json()
        return None
    
    def get_period_efforts(self, period_id, athlete_id, effort_types="velocity,acceleration"):
        params = {"effort_types": effort_types}
        r = requests.get(
            f"{self.base_url}/periods/{period_id}/athletes/{athlete_id}/efforts",
            headers=self.headers,
            params=params,
            timeout=60
        )
        if r.status_code == 200:
            return r.json()
        return None

    def get_activity_events(self, activity_id, athlete_id, event_types):
        params = {"event_types": event_types}
        r = requests.get(
            f"{self.base_url}/activities/{activity_id}/athletes/{athlete_id}/events",
            headers=self.headers,
            params=params,
            timeout=60
        )
        if r.status_code == 200:
            return r.json()
        return None

    def get_period_events(self, period_id, athlete_id, event_types):
        params = {"event_types": event_types}
        r = requests.get(
            f"{self.base_url}/periods/{period_id}/athletes/{athlete_id}/events",
            headers=self.headers,
            params=params,
            timeout=60
        )
        if r.status_code == 200:
            return r.json()
        return None

    # PARTE 2 - FUNÇÕES DE EXTRAÇÃO, CONVERSÃO E CÁLCULO

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

# ==================== FUNÇÃO CORRIGIDA PARA CAMPO DE RUGBY ====================

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


def adicionar_pontos_velocidade_bandas(fig, x_coords, y_coords, velocidades, bandas_sel):
    """Pontos coloridos por bandas de velocidade selecionadas."""
    if not x_coords or not bandas_sel:
        return
    xs = np.array(x_coords)
    ys = np.array(y_coords)
    vs = np.array(velocidades) if velocidades else np.zeros(len(xs))
    for k in bandas_sel:
        b = BANDAS_VEL[k]
        mask = (vs >= b['min']) & (vs < b['max'])
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=xs[mask], y=ys[mask], mode='markers', name=b['label'],
                marker=dict(size=3, color=b['color'], opacity=0.8),
                hovertemplate='x=%{x:.1f}m y=%{y:.1f}m<extra>' + b['label'] + '</extra>'))


def adicionar_pontos_aceleracao_bandas(fig, x_coords, y_coords, aceleracoes, bandas_sel):
    """Pontos coloridos por bandas de aceleração selecionadas."""
    if not x_coords or not bandas_sel:
        return
    xs  = np.array(x_coords)
    ys  = np.array(y_coords)
    acc = np.array(aceleracoes) if aceleracoes else np.zeros(len(xs))
    for k in bandas_sel:
        b = BANDAS_ACC[k]
        mask = (acc >= b['min']) & (acc < b['max'])
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=xs[mask], y=ys[mask], mode='markers', name=b['label'],
                marker=dict(size=3, color=b['color'], opacity=0.8),
                hovertemplate='x=%{x:.1f}m y=%{y:.1f}m<extra>' + b['label'] + '</extra>'))


def adicionar_setas_direcao(fig, x_coords, y_coords, sample=40):
    """Setas de direção do movimento ao longo da trajetória."""
    xs = np.array(x_coords)
    ys = np.array(y_coords)
    if len(xs) < 2:
        return
    for i in range(0, len(xs)-1, sample):
        j = min(i + sample, len(xs)-1)
        dx, dy = xs[j]-xs[i], ys[j]-ys[i]
        norm = np.hypot(dx, dy)
        if norm < 0.5:
            continue
        dx /= norm; dy /= norm
        fig.add_annotation(
            x=float(xs[i]+dx*2), y=float(ys[i]+dy*2),
            ax=float(xs[i]),     ay=float(ys[i]),
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.4,
            arrowwidth=1.5, arrowcolor='rgba(255,255,255,0.55)')


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

def calcular_metricas(sensor_points, athlete_name):
    if not sensor_points:
        return None
    
    distancia_total = 0
    player_load = 0
    velocidades = []
    fcs = []
    velocidade_anterior = 0
    
    prev_v = None
    for ponto in sensor_points:
        if ponto.get('v') is not None:
            v_ms = float(ponto['v'])
            v_kmh = v_ms * 3.6
            velocidades.append(v_kmh)

            if prev_v is not None:
                distancia_total += ((prev_v + v_ms) / 2) * 0.1
            prev_v = v_ms

        if ponto.get('a') is not None:
            acc = float(ponto['a'])
            player_load += acc ** 2

        if ponto.get('hr') is not None:
            hr = float(ponto['hr'])
            if hr > 0:
                fcs.append(hr)

    duracao_min = len(sensor_points) * 0.1 / 60

    return {
        'Atleta': athlete_name,
        'Duração (min)': round(duracao_min, 1),
        'Distância (m)': round(distancia_total, 0),
        'PlayerLoad': round(player_load, 0),
        'Velocidade Máx (km/h)': round(max(velocidades), 1) if velocidades else 0,
        'Velocidade Média (km/h)': round(np.mean(velocidades), 1) if velocidades else 0,
        'FC Máx (bpm)': round(max(fcs), 0) if fcs else 0,
        'FC Média (bpm)': round(np.mean(fcs), 0) if fcs else 0,
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
    xb = xb[(xb >= 0) & (xb <= field_length)]
    yb = yb[(yb >= 0) & (yb <= field_width)]

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
        colorbar=dict(title='Freq.', tickfont=dict(color='white'),
                      titlefont=dict(color='white'), x=1.01, thickness=12),
        name=f'Bloco {bloco_idx + 1}',
        hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Freq: %{z:.0f}<extra></extra>'
    ))
    return fig, int(mascara.sum()), label


# ==================== FEATURE 2: DIAGRAMA DE VORONOI ====================

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

def calcular_carga_neuromuscular(sensor_points, limiar=2.0):
    """Analisa esforços de acc/dec intensos como indicador de carga neuromuscular."""
    if not sensor_points:
        return None

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

    lm = limiar * 0.65
    mask_hi_acc  = acc_arr >= limiar
    mask_med_acc = (acc_arr >= lm) & (acc_arr < limiar)
    mask_hi_dec  = acc_arr <= -limiar
    mask_med_dec = (acc_arr <= -lm) & (acc_arr > -limiar)

    t_bins = np.arange(0, duracao + 60, 60)
    n_bins = max(1, len(t_bins) - 1)

    def _cpm(mask):
        return np.array([
            mask[(ts_rel >= t_bins[i]) & (ts_rel < t_bins[i + 1])].sum()
            for i in range(n_bins)
        ])

    t_mid = [(t_bins[i] + t_bins[i + 1]) / 2 / 60 for i in range(n_bins)]
    return {
        'ts_rel': ts_rel, 'acc': acc_arr, 'vel': vel_arr, 't_mid': t_mid,
        'hi_acc_min':  _cpm(mask_hi_acc),  'hi_dec_min':  _cpm(mask_hi_dec),
        'med_acc_min': _cpm(mask_med_acc), 'med_dec_min': _cpm(mask_med_dec),
        'total_hi_acc':  int(mask_hi_acc.sum()),  'total_hi_dec':  int(mask_hi_dec.sum()),
        'total_med_acc': int(mask_med_acc.sum()), 'total_med_dec': int(mask_med_dec.sum()),
        'limiar': limiar,
    }


def plotar_carga_neuromuscular(dados, atleta_nome):
    """Painel Plotly 2×2 com análise de carga neuromuscular."""
    lim = dados['limiar']
    t   = dados['t_mid']

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
        title=dict(text=f'💪 Carga Neuromuscular — {atleta_nome}',
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
    
    # Sidebar
    with st.sidebar:
        st.header("🌍 Servidor")
        server = st.selectbox("Selecione:", list(SERVERS.keys()))
        base_url = SERVERS[server]

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
                    for t in teams_raw:
                        teams_data.append({'id': t.get('id'), 'nome': t.get('name'), 'slug': t.get('slug')})
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
                        atividades.append({'id': a.get('id'), 'nome': a.get('name'), 'data': a.get('start_time')})
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
            
            if atividade_sel:
                activity_id = st.session_state.df_activities[st.session_state.df_activities['nome'] == atividade_sel]['id'].values[0]
                st.session_state.activity_id = activity_id
                
                with st.spinner("Buscando períodos da atividade..."):
                    api = st.session_state.api
                    periods_raw = api.get_activity_periods(activity_id)
                    
                    period_options = ['Atividade Completa']
                    period_ids = {'Atividade Completa': None}
                    if periods_raw and isinstance(periods_raw, list):
                        for p in periods_raw:
                            period_options.append(p.get('name', 'Período'))
                            period_ids[p.get('name', 'Período')] = p.get('id')
                        st.session_state.period_options = period_options
                        st.session_state.period_ids = period_ids
                        st.success(f"✅ {len(period_options)-1} períodos encontrados")
                    else:
                        st.session_state.period_options = period_options
                        st.session_state.period_ids = {'Atividade Completa': None}
                
                st.subheader("📊 Selecionar Período(s)")
                periodos_selecionados = st.multiselect(
                    "Selecione um ou mais períodos para análise:",
                    options=st.session_state.period_options,
                    default=['Atividade Completa']
                )
                st.session_state.periodos_selecionados = periodos_selecionados
                
                if st.button("🔍 Buscar Atletas da Atividade"):
                    with st.spinner("Buscando atletas..."):
                        api = st.session_state.api
                        primeiro_periodo = periodos_selecionados[0] if periodos_selecionados else 'Atividade Completa'
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
        
        st.info(f"📌 Atividade: {atividade_sel}")
        st.info(f"📌 Períodos selecionados: {', '.join(periodos_selecionados)}")
        
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

        for periodo_nome in periodos_selecionados:
            period_id = period_ids.get(periodo_nome)

            resultados = []
            dados_sensor_por_atleta = {}
            dados_efforts_vel = {}
            dados_efforts_acc = {}
            dados_posicao = {}
            dados_eventos = {}   # ← eventos futebol deste período
            
            progresso = st.progress(0)
            status_text = st.empty()
            
            for i, atleta_nome in enumerate(st.session_state.atletas_sel):
                status_text.text(f"Processando {atleta_nome} - {periodo_nome}...")
                
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
                    
                    # Usa x,y da API diretamente — coordenadas relativas ao campo (m desde
                    # o canto inferior esquerdo), calculadas pela Catapult a partir da
                    # configuração do campo no OpenField. Filtra nulos e valores absurdos
                    # (lat=0,long=0 sem nulls=1 produz x,y na casa dos milhões).
                    pontos_pos = [
                        (float(p['x']), float(p['y']),
                         (p.get('v') or 0) * 3.6,
                         float(p.get('a') or 0),
                         float(p.get('ts') or 0))
                        for p in sensor_points
                        if p.get('x') is not None and p.get('y') is not None
                        and -50 < float(p['x']) < 250
                        and -50 < float(p['y']) < 200
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
                                'vel': [], 'xs': [], 'ys': [],
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
                            st.success(f"✅ {atleta_nome}: {len(sensor_points)} pontos · {n_ev} eventos futebol")
                        else:
                            st.success(f"✅ {atleta_nome}: {len(sensor_points)} pontos · 0 eventos futebol")
                    else:
                        st.success(f"✅ {atleta_nome}: {len(sensor_points)} pontos")
                
                progresso.progress((i + 1) / len(st.session_state.atletas_sel))
            
            status_text.empty()
            progresso.empty()
            
            resultados_por_periodo[periodo_nome] = resultados
            dados_sensor_por_atleta_por_periodo[periodo_nome] = dados_sensor_por_atleta
            dados_efforts_vel_por_periodo[periodo_nome] = dados_efforts_vel
            dados_efforts_acc_por_periodo[periodo_nome] = dados_efforts_acc
            dados_posicao_por_periodo[periodo_nome] = dados_posicao
            dados_eventos_por_periodo[periodo_nome] = dados_eventos
        
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
                "📅 Microciclo ACWR",
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
                            metricas_comp = ['Distância (m)', 'PlayerLoad', 'Velocidade Máx (km/h)', 'Velocidade Média (km/h)', 'FC Média (bpm)']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                df_melted = df_filtrado.melt(id_vars=['Atleta'], value_vars=metricas_comp, var_name='Métrica', value_name='Valor')
                                fig = px.bar(df_melted, x='Atleta', y='Valor', color='Métrica', barmode='group',
                                            title=f"Comparação de Métricas - {periodo_comp}")
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                if len(atletas_selecionados_comp) <= 5:
                                    fig_radar = go.Figure()
                                    for atleta in atletas_selecionados_comp[:5]:
                                        valores = df_filtrado[df_filtrado['Atleta'] == atleta][metricas_comp].iloc[0].tolist()
                                        fig_radar.add_trace(go.Scatterpolar(
                                            r=valores,
                                            theta=metricas_comp,
                                            fill='toself',
                                            name=atleta
                                        ))
                                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Comparação Radial (Normalizado)")
                                    st.plotly_chart(fig_radar, use_container_width=True)
                            st.dataframe(df_filtrado[['Atleta', 'Equipe', 'Posição'] + metricas_comp], use_container_width=True)
                
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
                            metricas_comp = ['Distância (m)', 'PlayerLoad', 'Velocidade Máx (km/h)', 'Velocidade Média (km/h)', 'FC Média (bpm)']
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
            
            # ==================== ABA 2: CAMPO DE RUGBY ====================
            with abas[1]:
                st.subheader("🗺️ Campo de Futebol — Análise de Movimentação")
                st.caption(REFERENCIAS["campo"])

                if dados_posicao_por_periodo:
                    col1, col2 = st.columns(2)
                    with col1:
                        periodo_mapa = st.selectbox("Selecione o período:", list(dados_posicao_por_periodo.keys()), key="periodo_mapa")
                    atleta_mapa = None
                    with col2:
                        if dados_posicao_por_periodo[periodo_mapa]:
                            atleta_mapa = st.selectbox("Selecione o atleta:", list(dados_posicao_por_periodo[periodo_mapa].keys()), key="atleta_mapa")

                    if atleta_mapa and dados_posicao_por_periodo.get(periodo_mapa, {}).get(atleta_mapa):
                        dados = dados_posicao_por_periodo[periodo_mapa][atleta_mapa]

                        n_xy  = dados.get('n_pontos', len(dados.get('xs', [])))
                        n_gps = len(dados.get('lats', []))
                        st.caption(f"📡 Pontos campo (x/y): **{n_xy}** &nbsp;|&nbsp; 🌍 Pontos GPS reais (lat/lon): **{n_gps}**")

                        # Chave por atleta (campo físico não muda entre períodos)
                        campo_key = f"campo_cfg__{atleta_mapa}"
                        campo_aplicado = campo_key in st.session_state

                        lats_gps  = dados.get('lats', [])
                        lons_gps  = dados.get('lons', [])
                        vels_gps  = dados.get('vels_gps', [])
                        ts_gps    = dados.get('ts_gps', [])

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
                                _lat_c = round(float(np.median(lats_gps)), 7)
                                _lon_c = round(float(np.median(lons_gps)), 7)

                                # Componente bidirecional: retorna {lat,lon,rot,fl,fw,ig}
                                # quando o usuário clica "✅ Aplicar Campo" no painel do mapa
                                resultado_campo = _campo_component(
                                    pts=_pts,
                                    lat_c=_lat_c,
                                    lon_c=_lon_c,
                                    key=f"campo_mapa_{periodo_mapa}_{atleta_mapa}",
                                    default=None
                                )

                                if resultado_campo is not None:
                                    st.session_state[campo_key] = {
                                        'lat': float(resultado_campo['lat']),
                                        'lon': float(resultado_campo['lon']),
                                        'rot': int(resultado_campo['rot']),
                                        'fl':  int(resultado_campo['fl']),
                                        'fw':  int(resultado_campo['fw']),
                                        'ig':  int(resultado_campo['ig'])
                                    }
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

                            # Botão para reajustar (volta à Fase 1)
                            col_hdr, col_btn = st.columns([5, 1])
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

                            # ── Esforço selecionado (para filtrar GPS no mapa) ──
                            lats_eff, lons_eff, vels_eff, eff_desc = [], [], [], ""

                            # Mapa fixo (atualizado abaixo se houver esforço selecionado)
                            mapa_placeholder = st.empty()

                            st.divider()

                            # ── ETAPA 2: Análise de esforços ─────────────────────
                            st.markdown("### 2️⃣ Análise de Esforços no Campo")

                            # Dados de esforços deste atleta/período
                            vel_raw = dados_efforts_vel_por_periodo.get(periodo_mapa, {}).get(atleta_mapa, [])
                            acc_raw = dados_efforts_acc_por_periodo.get(periodo_mapa, {}).get(atleta_mapa, [])

                            tipo_esf = st.radio(
                                "Tipo de esforço:",
                                ["⚡ Velocidade", "🔁 Aceleração"],
                                horizontal=True, key="tipo_esf_campo"
                            )

                            raw_list = vel_raw if tipo_esf == "⚡ Velocidade" else acc_raw

                            if raw_list:
                                efforts_df_full = (processar_efforts_velocidade(raw_list)
                                                   if tipo_esf == "⚡ Velocidade"
                                                   else processar_efforts_aceleracao(raw_list))

                                if not efforts_df_full.empty:
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

                                    # Colunas visíveis (esconde _start_ts / _end_ts)
                                    cols_show = [c for c in efforts_df_full.columns
                                                 if not c.startswith('_')]
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

                                    if sel_rows and ts_gps:
                                        sel_idx = sel_rows[0]
                                        # Recupera os timestamps brutos da linha selecionada
                                        row_full = efforts_df_full.iloc[sel_idx]
                                        start_ts = row_full.get('_start_ts', 0)
                                        end_ts   = row_full.get('_end_ts', 0)

                                        if start_ts and end_ts and len(ts_gps) == len(lats_gps):
                                            filtered = [
                                                (la, lo, ve)
                                                for la, lo, ve, ts in zip(lats_gps, lons_gps, vels_gps, ts_gps)
                                                if start_ts <= ts <= end_ts
                                            ]
                                            if filtered:
                                                lats_eff = [p[0] for p in filtered]
                                                lons_eff = [p[1] for p in filtered]
                                                vels_eff = [p[2] for p in filtered]
                                                inicio_str = row_full.get('Início', '')
                                                dur_str    = row_full.get('Duração (s)', '')
                                                eff_desc   = f"Esforço #{row_full['Esforço']} — {inicio_str} — {dur_str}s"
                                            else:
                                                st.warning("⚠️ Nenhum ponto GPS encontrado na janela de tempo deste esforço.")
                                        elif not ts_gps:
                                            st.info("ℹ️ Timestamps GPS não disponíveis. Recarregue os dados para ativar o filtro de esforço no mapa.")

                                    # Download da tabela
                                    st.download_button(
                                        "📥 Exportar esforços",
                                        efforts_df_show.to_csv(index=False),
                                        file_name=f"esforcos_{atleta_mapa}_{periodo_mapa}.csv"
                                    )
                                else:
                                    st.info("Nenhum esforço encontrado após aplicar os filtros.")
                            else:
                                st.info("Nenhum dado de esforço disponível para este atleta neste período.")

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

                            # Fonte: Catapult x/y (prioritário) ou GPS derivado via campo aplicado
                            _has_xy  = bool(dados.get('xs') and dados.get('ys') and dados.get('vel'))
                            _has_gps = bool(lats_gps and lons_gps and cfg)

                            if not _has_xy and _has_gps:
                                st.caption("🌍 Coordenadas derivadas do GPS + campo aplicado.")
                            elif _has_xy:
                                st.caption("📡 Coordenadas x/y Catapult OpenField")

                            if _has_xy or _has_gps:
                                if _has_xy:
                                    xn, yn  = lat_lon_to_campo_coords(dados['xs'], dados['ys'])
                                    vel_raw = dados['vel']
                                    acc_raw = dados.get('acc', [0.0]*len(xn))
                                else:
                                    xn, yn  = gps_para_campo_coords(lats_gps, lons_gps, cfg)
                                    vel_raw = vels_gps if vels_gps else [0.0]*len(xn)
                                    acc_raw = [0.0]*len(xn)

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
                                        _ev_raw = dados_eventos_por_periodo.get(
                                            periodo_mapa, {}).get(atleta_mapa, {})
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
                                    fig_campo = desenhar_campo_futebol_bonito(
                                        title=f"📍 {atleta_mapa} — {periodo_mapa}")

                                    if modo_viz == "🗺️ Trajetória":
                                        adicionar_trajetoria_campo(fig_campo, xn, yn, vel_raw, atleta_mapa)
                                    elif modo_viz == "⚡ Bandas de Velocidade" and bandas_vel_sel:
                                        adicionar_pontos_velocidade_bandas(
                                            fig_campo, xn, yn, vel_raw, bandas_vel_sel)
                                    elif modo_viz == "🔁 Bandas de Aceleração" and bandas_acc_sel:
                                        adicionar_pontos_aceleracao_bandas(
                                            fig_campo, xn, yn, acc_raw, bandas_acc_sel)

                                    if ov_setas:
                                        adicionar_setas_direcao(fig_campo, xn, yn)
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

                                    st.plotly_chart(fig_campo, use_container_width=True)

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

                                    # ── Comparação entre períodos ─────────────────────
                                    if len(dados_posicao_por_periodo) > 1:
                                        st.markdown("---")
                                        st.markdown("#### 🔄 Comparação entre Períodos")
                                        periodos_lista = list(dados_posicao_por_periodo.keys())
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
                                            # Catapult x/y prioritário; cai para GPS fallback
                                            cfg1 = st.session_state.get(
                                                f"campo_cfg__{atleta_mapa}", cfg)
                                            cfg2 = cfg1  # mesmo campo físico para todos os períodos
                                            if d1c.get('xs') and d2c.get('xs'):
                                                x1c, y1c = lat_lon_to_campo_coords(d1c['xs'], d1c['ys'])
                                                x2c, y2c = lat_lon_to_campo_coords(d2c['xs'], d2c['ys'])
                                            elif d1c.get('lats') and d2c.get('lats') and cfg1 and cfg2:
                                                x1c, y1c = gps_para_campo_coords(
                                                    d1c['lats'], d1c['lons'], cfg1)
                                                x2c, y2c = gps_para_campo_coords(
                                                    d2c['lats'], d2c['lons'], cfg2)
                                            else:
                                                x1c = x2c = []
                                            if x1c and x2c:
                                                fig_cmp = desenhar_campo_futebol_bonito(
                                                    title=f"Comparação: {per1} (azul) vs {per2} (rosa)")
                                                fig_cmp.add_trace(go.Scatter(
                                                    x=x1c, y=y1c, mode='markers', name=per1,
                                                    marker=dict(size=2, color='#00E5FF', opacity=0.5)))
                                                fig_cmp.add_trace(go.Scatter(
                                                    x=x2c, y=y2c, mode='markers', name=per2,
                                                    marker=dict(size=2, color='#FF4081', opacity=0.5)))
                                                st.plotly_chart(fig_cmp, use_container_width=True)
                                            else:
                                                st.info("Dados de posição não disponíveis em um dos períodos selecionados.")
                                        else:
                                            st.info("Selecione dois períodos diferentes para comparar.")
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

                    else:
                        st.info("Selecione um período e atleta")

                    # ══════════════════════════════════════════════════════
                    # FEATURE 1 — HEATMAP TEMPORAL SEGMENTADO
                    # ══════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown("### 🕐 Heatmap por Fase da Partida")
                    st.caption("Visualize onde o atleta atuou em cada bloco de tempo da sessão.")

                    _per_ht = list(dados_posicao_por_periodo.keys())
                    if _per_ht:
                        _col_ht1, _col_ht2, _col_ht3 = st.columns([2, 2, 2])
                        with _col_ht1:
                            _per_ht_sel = st.selectbox("Período:", _per_ht, key="ht_periodo")
                        with _col_ht2:
                            _ats_ht = list(dados_posicao_por_periodo.get(_per_ht_sel, {}).keys())
                            _atl_ht = st.selectbox("Atleta:", _ats_ht, key="ht_atleta") if _ats_ht else None
                        with _col_ht3:
                            _bloco_min_ht = st.selectbox("Duração do bloco:", [5, 10, 15, 20, 30],
                                                          index=2, key="ht_bloco")

                        if _atl_ht:
                            _dp_ht = dados_posicao_por_periodo[_per_ht_sel].get(_atl_ht, {})
                            _xs_ht  = _dp_ht.get('xs', [])
                            _ys_ht  = _dp_ht.get('ys', [])
                            _ts_ht  = _dp_ht.get('ts_pos', [])

                            if _xs_ht and _ts_ht:
                                _ts_rel_ht = np.array(_ts_ht, dtype=float)
                                _ts_rel_ht -= _ts_rel_ht.min()
                                _dur_min    = int(_ts_rel_ht.max() / 60)
                                _n_blocos   = max(1, -(-_dur_min // _bloco_min_ht))  # ceil div

                                _col_sl, _col_bt = st.columns([4, 1])
                                with _col_sl:
                                    _bloco_idx = st.slider(
                                        "Selecione o bloco:", 0, _n_blocos - 1, 0,
                                        format=f"Bloco %d de {_n_blocos}",
                                        key="ht_bloco_idx"
                                    )
                                with _col_bt:
                                    st.metric("Total de blocos", _n_blocos)

                                _fig_ht, _n_pts_ht, _lbl_ht = gerar_heatmap_segmentado(
                                    _xs_ht, _ys_ht, _ts_ht,
                                    _bloco_min_ht, _bloco_idx,
                                    field_length=105, field_width=68
                                )
                                if _fig_ht:
                                    st.caption(f"📍 {_n_pts_ht} pontos GPS neste bloco ({_lbl_ht})")
                                    st.plotly_chart(_fig_ht, use_container_width=True)
                                else:
                                    st.info(f"Nenhum ponto de campo disponível no bloco {_lbl_ht}. "
                                            "Tente outro bloco ou reduza a duração.")
                            else:
                                st.info("Dados de posição com timestamp não disponíveis para este atleta.")

                    # ══════════════════════════════════════════════════════
                    # FEATURE 2 — DIAGRAMA DE VORONOI
                    # ══════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown("### 🔷 Diagrama de Voronoi — Raio de Ação Coletivo")
                    st.caption(
                        "Mostra a zona de domínio espacial de cada atleta com base na sua posição mediana. "
                        "Útil para identificar cobertura coletiva e gaps táticos."
                    )

                    _per_vor = list(dados_posicao_por_periodo.keys())
                    if _per_vor:
                        _per_vor_sel = st.selectbox("Período para Voronoi:", _per_vor, key="vor_periodo")
                        _dp_vor_all  = dados_posicao_por_periodo.get(_per_vor_sel, {})
                        _ats_vor_all = list(_dp_vor_all.keys())

                        if len(_ats_vor_all) >= 2:
                            _ats_vor_sel = st.multiselect(
                                "Atletas a incluir no diagrama:",
                                _ats_vor_all, default=_ats_vor_all,
                                key="vor_atletas"
                            )
                            if len(_ats_vor_sel) >= 2:
                                _pos_vor = {a: _dp_vor_all[a] for a in _ats_vor_sel if a in _dp_vor_all}
                                _fig_vor = calcular_voronoi_campo(_pos_vor)
                                if _fig_vor:
                                    st.plotly_chart(_fig_vor, use_container_width=True)
                                    with st.expander("ℹ️ Como interpretar o Voronoi"):
                                        st.markdown("""
                                        - Cada **cor** representa a zona de domínio espacial de um atleta.
                                        - O **losango** indica a posição mediana do atleta na sessão.
                                        - Zonas **grandes** indicam que o jogador cobriu mais espaço sem apoio próximo.
                                        - Zonas **sobrepostas** (pequenas) sugerem concentração de jogadores em uma área.
                                        """)
                                else:
                                    st.info("Dados de posição insuficientes para gerar o diagrama.")
                            else:
                                st.info("Selecione pelo menos 2 atletas para gerar o Voronoi.")
                        else:
                            st.info("É necessário ter pelo menos 2 atletas com dados de posição no período selecionado.")

                else:
                    st.info("Dados de posição não disponíveis. Verifique se o sensor GPS estava ativo durante a sessão.")

            # ==================== ABA 3: ESFORÇOS AO LONGO DO TEMPO ====================
            with abas[2]:
                st.subheader("⏱️ Esforços ao Longo do Tempo")
                
                if dados_sensor_por_atleta_por_periodo:
                    periodo_escolhido = st.selectbox("Selecione o período:", list(dados_sensor_por_atleta_por_periodo.keys()), key="periodo_esforcos")
                    if dados_sensor_por_atleta_por_periodo[periodo_escolhido]:
                        atleta_escolhido = st.selectbox("Selecione o atleta:", list(dados_sensor_por_atleta_por_periodo[periodo_escolhido].keys()), key="atleta_esforcos")
                        sensor_points = dados_sensor_por_atleta_por_periodo[periodo_escolhido][atleta_escolhido]
                        
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
                                        f"fadiga_{atleta_escolhido}_{periodo_escolhido}.csv"
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
                    periodo_janela = st.selectbox("Selecione o período:", list(dados_sensor_por_atleta_por_periodo.keys()), key="periodo_janela")
                    
                    if periodo_janela in dados_sensor_por_atleta_por_periodo:
                        atleta_janela = st.selectbox("Selecione o atleta:", list(dados_sensor_por_atleta_por_periodo[periodo_janela].keys()), key="atleta_janela")
                        sensor_points = dados_sensor_por_atleta_por_periodo[periodo_janela][atleta_janela]
                        
                        if sensor_points:
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
                    _nm_per = st.selectbox("Período:", list(dados_sensor_por_atleta_por_periodo.keys()),
                                           key="nm_periodo")
                    if dados_sensor_por_atleta_por_periodo[_nm_per]:
                        _nm_ats = list(dados_sensor_por_atleta_por_periodo[_nm_per].keys())
                        _nm_atl = st.selectbox("Atleta:", _nm_ats, key="nm_atleta")
                        _nm_lim = st.slider("Limiar de intensidade (m/s²):", 1.0, 4.0, 2.0, 0.5,
                                            key="nm_limiar",
                                            help="Acelerações/desacelerações acima deste valor são classificadas como intensas.")

                        _nm_sp = dados_sensor_por_atleta_por_periodo[_nm_per][_nm_atl]
                        _nm_dados = calcular_carga_neuromuscular(_nm_sp, limiar=_nm_lim)

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

            # ==================== ABA 6: MICROCICLO ACWR ====================
            with abas[5]:
                st.subheader("📅 Microciclo — Acute:Chronic Workload Ratio (ACWR)")
                st.markdown("""
                O **ACWR** mede a relação entre a carga recente (últimos 7 dias) e a carga habitual (média
                das últimas 4 semanas). É um dos indicadores mais usados na prevenção de lesões em futebol.

                | Zona | ACWR | Interpretação |
                |------|------|---------------|
                | 🔵 Subcarregado | < 0.8 | Atleta abaixo da carga habitual |
                | ✅ Ótimo | 0.8 – 1.3 | Zona segura e de alto rendimento |
                | ⚠️ Atenção | 1.3 – 1.5 | Risco moderado de lesão |
                | 🔴 Risco | > 1.5 | Alto risco — revisar planejamento |
                """)
                st.markdown("---")

                if not st.session_state.get('df_activities', pd.DataFrame()).empty and 'api' in st.session_state:
                    _acwr_atividades = st.session_state.df_activities

                    st.markdown("#### 1️⃣ Selecione as atividades do microciclo")
                    st.caption("Selecione múltiplas atividades de diferentes datas para calcular o ACWR.")

                    _acwr_ats_sel = st.multiselect(
                        "Atividades:", _acwr_atividades['nome'].tolist(),
                        key="acwr_ativs"
                    )

                    if _acwr_ats_sel and 'atletas_filtrados' in st.session_state:
                        _acwr_atl_opts = st.session_state.atletas_filtrados['nome'].tolist()
                        _acwr_atls_sel = st.multiselect(
                            "Atletas para o microciclo:", _acwr_atl_opts,
                            default=_acwr_atl_opts[:min(3, len(_acwr_atl_opts))],
                            key="acwr_atletas"
                        )

                        if _acwr_atls_sel and st.button("📊 Calcular ACWR", type="primary", key="btn_acwr"):
                            _acwr_api = st.session_state.api
                            _acwr_rows = []

                            _acwr_prog = st.progress(0)
                            _acwr_status = st.empty()
                            _n_total = len(_acwr_ats_sel) * len(_acwr_atls_sel)
                            _cnt = 0

                            for _acwr_atv_nome in _acwr_ats_sel:
                                _acwr_row_atv = _acwr_atividades[_acwr_atividades['nome'] == _acwr_atv_nome]
                                if _acwr_row_atv.empty:
                                    continue
                                _acwr_atv_id = _acwr_row_atv['id'].values[0]
                                _acwr_data_raw = _acwr_row_atv['data'].values[0]

                                # Parse data
                                try:
                                    if isinstance(_acwr_data_raw, str):
                                        _acwr_dt = datetime.fromisoformat(
                                            _acwr_data_raw.replace('Z', '+00:00')
                                        ).replace(tzinfo=None)
                                    else:
                                        _acwr_dt = pd.to_datetime(_acwr_data_raw).to_pydatetime()
                                except Exception:
                                    _acwr_dt = datetime.now()

                                for _acwr_atl_nome in _acwr_atls_sel:
                                    _cnt += 1
                                    _acwr_status.text(f"Carregando {_acwr_atl_nome} — {_acwr_atv_nome}...")
                                    _acwr_atl_row = st.session_state.atletas_filtrados[
                                        st.session_state.atletas_filtrados['nome'] == _acwr_atl_nome
                                    ]
                                    if _acwr_atl_row.empty:
                                        continue
                                    _acwr_atl_id = _acwr_atl_row['id'].values[0]

                                    try:
                                        _resp = _acwr_api.get_sensor_data(_acwr_atv_id, _acwr_atl_id)
                                        _spts = extrair_dados_sensor(_resp)
                                        if _spts:
                                            _mets = calcular_metricas(_spts, _acwr_atl_nome)
                                            _pl   = _mets.get('PlayerLoad', 0) if _mets else 0
                                        else:
                                            _pl = 0
                                    except Exception:
                                        _pl = 0

                                    _acwr_rows.append({
                                        'atleta': _acwr_atl_nome,
                                        'data': _acwr_dt,
                                        'atividade': _acwr_atv_nome,
                                        'player_load': float(_pl),
                                    })
                                    _acwr_prog.progress(_cnt / _n_total)

                            _acwr_prog.empty()
                            _acwr_status.empty()

                            if _acwr_rows:
                                _df_cargas = pd.DataFrame(_acwr_rows)
                                _df_acwr   = calcular_acwr_df(_df_cargas)
                                st.session_state['df_acwr_calculado'] = _df_acwr
                                st.success(f"✅ ACWR calculado para {len(_acwr_atls_sel)} atletas em {len(_acwr_ats_sel)} atividades.")
                            else:
                                st.warning("Nenhum dado de PlayerLoad obtido. Verifique se os atletas participaram das atividades selecionadas.")

                        # Mostrar resultado se já calculado
                        if 'df_acwr_calculado' in st.session_state:
                            _df_acwr_show = st.session_state['df_acwr_calculado']
                            _fig_acwr = plotar_acwr(_df_acwr_show)
                            st.plotly_chart(_fig_acwr, use_container_width=True)

                            # Alertas automáticos
                            _df_risco = _df_acwr_show[_df_acwr_show['ACWR'] > 1.5].dropna(subset=['ACWR'])
                            _df_atenc = _df_acwr_show[
                                (_df_acwr_show['ACWR'] > 1.3) & (_df_acwr_show['ACWR'] <= 1.5)
                            ].dropna(subset=['ACWR'])

                            if not _df_risco.empty:
                                st.error(f"🔴 **{len(_df_risco)} sessões com ACWR > 1.5** — risco elevado de lesão:")
                                st.dataframe(_df_risco[['Atleta','Data','Atividade','PlayerLoad','ACWR']],
                                             use_container_width=True)
                            if not _df_atenc.empty:
                                st.warning(f"⚠️ **{len(_df_atenc)} sessões com ACWR entre 1.3–1.5** — monitorar:")
                                st.dataframe(_df_atenc[['Atleta','Data','Atividade','PlayerLoad','ACWR']],
                                             use_container_width=True)

                            # Tabela completa + exportar
                            with st.expander("📋 Tabela completa ACWR"):
                                st.dataframe(_df_acwr_show, use_container_width=True)
                            st.download_button(
                                "📥 Exportar ACWR (CSV)",
                                _df_acwr_show.to_csv(index=False),
                                "microciclo_acwr.csv"
                            )
                    elif not _acwr_ats_sel:
                        st.info("Selecione as atividades do microciclo acima.")
                    else:
                        st.info("Carregue os dados de atletas antes de calcular o ACWR.")
                else:
                    st.info("Carregue os dados da API para usar esta funcionalidade.")

        else:
            st.warning("Nenhum dado encontrado")
    
    elif 'df_athletes' in st.session_state and not st.session_state.df_athletes.empty:
        st.info("👈 Selecione uma atividade, período(s) e clique em 'Buscar Atletas da Atividade'")

if __name__ == "__main__":
    main()
    
    