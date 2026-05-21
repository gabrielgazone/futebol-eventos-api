# rugby_eventos_completo_final.py
# PARTE 1 - IMPORTS, CONSTANTES E CLASSE API

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import base64
import numpy as np
from scipy.signal import savgol_filter

st.set_page_config(page_title="Rugby Eventos - Catapult", layout="wide")

SERVERS = {
    "Américas (US)": "https://connect-us.catapultsports.com/api/v6",
    "Europa/África (EU)": "https://connect-eu.catapultsports.com/api/v6",
    "Ásia-Pacífico (AU)": "https://connect-au.catapultsports.com/api/v6",
}

# ==================== REFERÊNCIAS BIBLIOGRÁFICAS ====================
REFERENCIAS = {
    "janelas": """
    **Referência:** Aughey, R.J. (2011). "Applications of GPS technologies to field sports". 
    *International Journal of Sports Physiology and Performance*, 6(3), 295-310.
    """,
    "campo": """
    **Referência:** World Rugby (2023). "Law 1: The Ground". World Rugby Laws of the Game.
    Campo oficial: 100m de comprimento x 70m de largura (entre linhas de fundo).
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
    
    # Escalar para o campo (comprimento 100m, largura 70m)
    x = lon_norm * 100
    y = lat_norm * 70
    
    return x.tolist(), y.tolist()

def desenhar_campo_rugby(field_length=100, field_width=70, in_goal_depth=10):
    """Desenha o campo de rugby com todas as marcações oficiais"""
    fig = go.Figure()
    
    # Fundo do campo
    fig.add_shape(type="rect", x0=-in_goal_depth-2, y0=-5, x1=field_length+in_goal_depth+2, y1=field_width+5,
                  fillcolor="#1a472a", line=dict(color="rgba(0,0,0,0)", width=0))
    
    # Perímetro do campo
    fig.add_shape(type="rect", x0=0, y0=0, x1=field_length, y1=field_width,
                  line=dict(color="white", width=3), fillcolor="rgba(0,100,0,0.3)")
    
    # In-goal
    fig.add_shape(type="rect", x0=-in_goal_depth, y0=0, x1=0, y1=field_width,
                  line=dict(color="white", width=1), fillcolor="rgba(0,150,0,0.2)")
    fig.add_shape(type="rect", x0=field_length, y0=0, x1=field_length + in_goal_depth, y1=field_width,
                  line=dict(color="white", width=1), fillcolor="rgba(0,150,0,0.2)")
    
    # Linhas de gol
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=field_width, line=dict(color="white", width=3))
    fig.add_shape(type="line", x0=field_length, y0=0, x1=field_length, y1=field_width, line=dict(color="white", width=3))
    
    # Linha de 22 metros
    fig.add_shape(type="line", x0=22, y0=0, x1=22, y1=field_width, line=dict(color="white", width=1.5, dash="dash"))
    fig.add_shape(type="line", x0=field_length - 22, y0=0, x1=field_length - 22, y1=field_width, line=dict(color="white", width=1.5, dash="dash"))
    
    # Linha de 10 metros
    fig.add_shape(type="line", x0=10, y0=0, x1=10, y1=field_width, line=dict(color="white", width=1, dash="dot"))
    fig.add_shape(type="line", x0=field_length - 10, y0=0, x1=field_length - 10, y1=field_width, line=dict(color="white", width=1, dash="dot"))
    
    # Linha central
    fig.add_shape(type="line", x0=field_length/2, y0=0, x1=field_length/2, y1=field_width, line=dict(color="white", width=2))
    
    # Círculo central
    circle_center = (field_length/2, field_width/2)
    circle_radius = 10
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = circle_center[0] + circle_radius * np.cos(theta)
    circle_y = circle_center[1] + circle_radius * np.sin(theta)
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines',
                            line=dict(color="white", width=1.5), name="Círculo Central", showlegend=False))
    
    # Traves
    crossbar_y = field_width/2 - 3
    fig.add_shape(type="line", x0=0, y0=crossbar_y, x1=0, y1=crossbar_y + 6, line=dict(color="#FFD700", width=3))
    fig.add_shape(type="line", x0=0, y0=field_width/2, x1=-5, y1=field_width/2, line=dict(color="#FFD700", width=3))
    fig.add_shape(type="line", x0=field_length, y0=crossbar_y, x1=field_length, y1=crossbar_y + 6, line=dict(color="#FFD700", width=3))
    fig.add_shape(type="line", x0=field_length, y0=field_width/2, x1=field_length + 5, y1=field_width/2, line=dict(color="#FFD700", width=3))
    
    # Linhas de 15 metros
    fig.add_shape(type="line", x0=15, y0=0, x1=15, y1=field_width, line=dict(color="white", width=0.5, dash="dot"))
    fig.add_shape(type="line", x0=field_length - 15, y0=0, x1=field_length - 15, y1=field_width, line=dict(color="white", width=0.5, dash="dot"))
    
    fig.update_layout(
        title="Campo de Rugby Oficial - Trajetória e Mapa de Calor",
        xaxis=dict(range=[-in_goal_depth-5, field_length+in_goal_depth+5],
                   title="Comprimento do Campo (m)", fixedrange=False,
                   gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        yaxis=dict(range=[-10, field_width+10], title="Largura do Campo (m)", fixedrange=False,
                   gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        plot_bgcolor='#1a472a',
        paper_bgcolor='#1a1a2e',
        height=600,
        hovermode='closest'
    )
    
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                       text="⚪ Linhas de campo | 🟡 Traves | 🔵 Trajetória | 🟢 Início | 🔴 Fim",
                       showarrow=False, font=dict(color="white", size=10),
                       bgcolor='rgba(0,0,0,0.6)', borderpad=4)
    
    return fig

def plotar_trajetoria_campo(x_coords, y_coords, velocidades, athlete_name):
    """Plota a trajetória do atleta no campo"""
    fig = desenhar_campo_rugby()
    
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
    fig = desenhar_campo_rugby()
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return fig
    
    x_edges = np.linspace(0, 100, 40)
    y_edges = np.linspace(0, 70, 40)
    
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
            'Banda': band
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
            'Banda': band
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


def main():
    st.title("🏉 Rugby Eventos - Catapult Sports")
    st.markdown("### Análise de Performance com Filtros por Equipe, Posição e Período")
    
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
        
        for periodo_nome in periodos_selecionados:
            period_id = period_ids.get(periodo_nome)
            
            resultados = []
            dados_sensor_por_atleta = {}
            dados_efforts_vel = {}
            dados_efforts_acc = {}
            dados_posicao = {}
            
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
                    response = api.get_period_sensor_data(period_id, athlete_id)
                    efforts_response = api.get_period_efforts(period_id, athlete_id, "velocity,acceleration")
                else:
                    response = api.get_sensor_data(activity_id, athlete_id)
                    efforts_response = api.get_activity_efforts(activity_id, athlete_id, "velocity,acceleration")
                
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
                        (float(p['x']), float(p['y']), p.get('v', 0) * 3.6)
                        for p in sensor_points
                        if p.get('x') is not None and p.get('y') is not None
                        and -50 < float(p['x']) < 250
                        and -50 < float(p['y']) < 200
                    ]
                    if pontos_pos:
                        xs = [pt[0] for pt in pontos_pos]
                        ys = [pt[1] for pt in pontos_pos]
                        velocidades = [pt[2] for pt in pontos_pos]
                        dados_posicao[atleta_nome] = {
                            'vel': velocidades, 'xs': xs, 'ys': ys,
                            'posicao': athlete_posicao, 'equipe': athlete_equipe,
                            'n_pontos': len(pontos_pos)
                        }
                    
                    st.success(f"✅ {atleta_nome}: {len(sensor_points)} pontos")
                
                progresso.progress((i + 1) / len(st.session_state.atletas_sel))
            
            status_text.empty()
            progresso.empty()
            
            resultados_por_periodo[periodo_nome] = resultados
            dados_sensor_por_atleta_por_periodo[periodo_nome] = dados_sensor_por_atleta
            dados_efforts_vel_por_periodo[periodo_nome] = dados_efforts_vel
            dados_efforts_acc_por_periodo[periodo_nome] = dados_efforts_acc
            dados_posicao_por_periodo[periodo_nome] = dados_posicao
        
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
                "🗺️ Campo de Rugby",
                "⏱️ Esforços ao Longo do Tempo",
                "📊 Janelas Temporais Móveis",
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
                st.subheader("🗺️ Trajetória no Campo de Rugby")
                st.markdown("Visualize a movimentação do atleta no campo com medidas oficiais")
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
                        st.caption(f"📡 Pontos de campo (x/y) válidos: **{dados.get('n_pontos', len(dados.get('xs', [])))}**")

                        if dados.get('xs') and dados.get('ys') and dados['vel']:
                            # Normaliza x,y para 0-100 × 0-70 (dimensões do campo desenhado).
                            # Os valores brutos da API estão em metros relativos ao campo mas
                            # podem ter range diferente dependendo da configuração do OpenField
                            # (in-goal incluídos, campo rotacionado, origem deslocada, etc.).
                            x_coords, y_coords = lat_lon_to_campo_coords(dados['xs'], dados['ys'])

                            if len(x_coords) > 0:
                                tipo_vis = st.radio(
                                    "Tipo de visualização:",
                                    ["🗺️ Trajetória", "🔥 Mapa de Calor", "📊 Ambos"],
                                    horizontal=True
                                )

                                if tipo_vis in ["🗺️ Trajetória", "📊 Ambos"]:
                                    fig_traj = plotar_trajetoria_campo(x_coords, y_coords, dados['vel'], atleta_mapa)
                                    st.plotly_chart(fig_traj, use_container_width=True)

                                if tipo_vis in ["🔥 Mapa de Calor", "📊 Ambos"]:
                                    fig_heat = plotar_heatmap_campo(x_coords, y_coords, dados['vel'], atleta_mapa)
                                    st.plotly_chart(fig_heat, use_container_width=True)

                                # Estatísticas
                                st.markdown("#### 📊 Estatísticas de Movimentação")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    dist_km = sum(dados['vel']) * 0.1 / 3600
                                    st.metric("Distância Total", f"{dist_km:.2f} km")
                                with col2:
                                    x_range = max(x_coords) - min(x_coords)
                                    st.metric("Largura Atuação", f"{x_range:.0f} m")
                                with col3:
                                    y_range = max(y_coords) - min(y_coords)
                                    st.metric("Profundidade", f"{y_range:.0f} m")
                                with col4:
                                    area = x_range * y_range
                                    st.metric("Área Percorrida", f"{area:.0f} m²")
                            else:
                                st.error("❌ Não foi possível converter as coordenadas GPS")
                        else:
                            st.error(
                                "❌ Nenhum ponto de campo (x/y) válido encontrado para este atleta.\n\n"
                                "**Causas prováveis:**\n"
                                "- O campo não está configurado no OpenField para esta atividade\n"
                                "- Sensor GPS sem lock (x/y retornados como nulos pela API)\n"
                                "- Parâmetro `xy` não disponível neste perfil de sincronização"
                            )
                    else:
                        st.info("Selecione um período e atleta")
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
                else:
                    st.info("Dados de sensor não disponíveis")
            
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
            
        
        else:
            st.warning("Nenhum dado encontrado")
    
    elif 'df_athletes' in st.session_state and not st.session_state.df_athletes.empty:
        st.info("👈 Selecione uma atividade, período(s) e clique em 'Buscar Atletas da Atividade'")

if __name__ == "__main__":
    main()
    
    