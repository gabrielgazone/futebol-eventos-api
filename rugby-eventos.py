# rugby_eventos_completo_final.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import base64
import numpy as np

st.set_page_config(page_title="Rugby Eventos - Catapult", layout="wide")

SERVERS = {
    "Américas (US)": "https://connect-us.catapultsports.com/api/v6",
    "Europa/África (EU)": "https://connect-eu.catapultsports.com/api/v6",
    "Ásia-Pacífico (AU)": "https://connect-au.catapultsports.com/api/v6",
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
    
    def get_sensor_data(self, activity_id, athlete_id):
        params = {
            "parameters": "ts,lat,long,v,a,hr,pl",
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
            "parameters": "ts,lat,long,v,a,hr,pl",
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

def extrair_dados_sensor(response_data):
    if not response_data:
        return []
    if isinstance(response_data, list):
        for item in response_data:
            if isinstance(item, dict) and 'data' in item:
                return item['data']
    return []

def extrair_efforts_data(response_data):
    """Extrai dados de esforços da resposta da API - Estrutura correta"""
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
    
    for i, ponto in enumerate(sensor_points):
        if 'v' in ponto and ponto['v']:
            v_ms = float(ponto['v'])
            v_kmh = v_ms * 3.6
            velocidades.append(v_kmh)
            
            if i > 0 and velocidade_anterior > 0:
                distancia_intervalo = ((velocidade_anterior + v_ms) / 2) * 0.1
                distancia_total += distancia_intervalo
            velocidade_anterior = v_ms
        
        if 'a' in ponto and ponto['a']:
            acc = float(ponto['a'])
            player_load += acc ** 2
        
        if 'hr' in ponto and ponto['hr']:
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

def processar_efforts_velocidade(efforts_data):
    """
    Processa dados de esforços de velocidade para tabela
    O % do Máximo é calculado com base no maior valor da própria lista
    """
    if not efforts_data:
        return pd.DataFrame()
    
    records = []
    
    # Encontrar a maior velocidade nos próprios esforços
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
    """
    Processa dados de esforços de aceleração para tabela
    O % do Máximo é calculado SEPARADAMENTE para acelerações (positivas) e desacelerações (negativas)
    """
    if not efforts_data:
        return pd.DataFrame()
    
    records = []
    
    # Encontrar a maior aceleração positiva e a maior desaceleração (mais negativa)
    max_acc_positiva = 0
    max_acc_negativa = 0
    
    for effort in efforts_data:
        acc = effort.get('acceleration', 0)
        if acc > 0:
            max_acc_positiva = max(max_acc_positiva, acc)
        elif acc < 0:
            max_acc_negativa = min(max_acc_negativa, acc)  # mais negativo
    
    for i, effort in enumerate(efforts_data, 1):
        start_time = effort.get('start_time', 0)
        acceleration = effort.get('acceleration', 0)
        end_time = effort.get('end_time', 0)
        duration = (end_time - start_time) if end_time else 0
        distance = effort.get('distance', 0)
        band = effort.get('band', '')
        
        # Calcular percentual baseado no tipo (positivo ou negativo)
        if acceleration > 0:
            # Aceleração positiva: % em relação à maior aceleração positiva
            percent_of_max = (acceleration / max_acc_positiva * 100) if max_acc_positiva > 0 else 0
            tipo = 'Aceleração'
        elif acceleration < 0:
            # Desaceleração: % em relação à maior desaceleração (valor mais negativo)
            # Ex: -5.85 é 100%, -4.37 é (4.37/5.85)*100 = 74.7%
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

def criar_grafico_velocidade_tempo(sensor_points, athlete_name):
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
            v_ms = float(ponto['v'])
            v_kmh = v_ms * 3.6
            
            tempos.append(tempo_relativo)
            velocidades.append(v_kmh)
    
    if len(tempos) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tempos,
        y=velocidades,
        mode='lines',
        name='Velocidade',
        line=dict(color='blue', width=1),
        fill='tozeroy',
        fillcolor='rgba(0,0,255,0.1)'
    ))
    
    fig.update_layout(
        title=f"Velocidade ao Longo do Tempo - {athlete_name}",
        xaxis_title="Tempo (minutos)",
        yaxis_title="Velocidade (km/h)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def criar_grafico_aceleracao_tempo(sensor_points, athlete_name):
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
            
            tempos.append(tempo_relativo)
            aceleracoes.append(acc)
    
    if len(tempos) == 0:
        return None
    
    colors = ['green' if a >= 0 else 'red' for a in aceleracoes]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tempos,
        y=aceleracoes,
        mode='lines+markers',
        name='Aceleração',
        line=dict(color='gray', width=1),
        marker=dict(size=2, color=colors),
        fill='tozeroy',
        fillcolor='rgba(128,128,128,0.1)'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    fig.update_layout(
        title=f"Aceleração ao Longo do Tempo - {athlete_name}",
        xaxis_title="Tempo (minutos)",
        yaxis_title="Aceleração (m/s²)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def criar_mapa_calor(x, y, athlete_name):
    fig = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=x, y=y, colorscale='Hot', nbinsx=50, nbinsy=50
    ))
    fig.update_layout(
        title=f"Mapa de Calor - {athlete_name}",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        height=500, template='plotly_dark'
    )
    return fig

def criar_trajetoria(x, y, velocidades, athlete_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers',
        marker=dict(size=2, color=velocidades, colorscale='Viridis',
                   showscale=True, colorbar=dict(title="Velocidade (km/h)")),
        line=dict(color='lightgray', width=1)
    ))
    fig.update_layout(
        title=f"Trajetória - {athlete_name}",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        height=500, template='plotly_white'
    )
    return fig

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
    if 'athlete_position_map' not in st.session_state:
        st.session_state.athlete_position_map = {}
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
                
                # Carregar equipes
                st.subheader("📋 Carregando Equipes...")
                teams_raw = api.get_teams()
                if teams_raw:
                    teams_data = []
                    for t in teams_raw:
                        teams_data.append({
                            'id': t.get('id'),
                            'nome': t.get('name'),
                            'slug': t.get('slug')
                        })
                    st.session_state.df_teams = pd.DataFrame(teams_data)
                    st.success(f"✅ {len(teams_data)} equipes carregadas")
                    
                    # Mapear atletas por equipe
                    st.subheader("📋 Mapeando atletas por equipe...")
                    athlete_team_map = {}
                    for _, team in st.session_state.df_teams.iterrows():
                        team_id = team['id']
                        team_name = team['nome']
                        team_athletes = api.get_team_athletes(team_id)
                        if team_athletes:
                            if isinstance(team_athletes, dict):
                                team_athletes = team_athletes.get('data', team_athletes.get('items', []))
                            for ath in team_athletes:
                                athlete_id = ath.get('id')
                                if athlete_id:
                                    athlete_team_map[athlete_id] = team_name
                    st.session_state.athlete_team_map = athlete_team_map
                    st.success(f"✅ {len(athlete_team_map)} atletas mapeados por equipe")
                
                # Carregar posições
                st.subheader("📋 Carregando Posições...")
                positions_raw = api.get_positions()
                if positions_raw:
                    positions_data = []
                    for p in positions_raw:
                        positions_data.append({
                            'id': p.get('id'),
                            'nome': p.get('name'),
                            'slug': p.get('slug'),
                            'esporte': p.get('sport_name')
                        })
                    st.session_state.df_positions = pd.DataFrame(positions_data)
                    st.success(f"✅ {len(positions_data)} posições carregadas")
                
                # Carregar atletas
                st.subheader("📋 Carregando Atletas...")
                athletes_raw = api.get_athletes()
                if athletes_raw:
                    atletas = []
                    position_map = {}
                    
                    for a in athletes_raw:
                        nome = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
                        if not nome:
                            nome = a.get('name', 'Sem nome')
                        
                        athlete_id = a.get('id')
                        
                        position_id = a.get('position_id')
                        position_name = ''
                        if position_id and not st.session_state.df_positions.empty:
                            pos_row = st.session_state.df_positions[
                                st.session_state.df_positions['id'] == position_id
                            ]
                            if not pos_row.empty:
                                position_name = pos_row.iloc[0]['nome']
                        
                        team_name = st.session_state.athlete_team_map.get(athlete_id, '')
                        
                        atletas.append({
                            'id': athlete_id,
                            'nome': nome,
                            'camisa': a.get('jersey', ''),
                            'posicao': position_name,
                            'posicao_id': position_id,
                            'equipe': team_name
                        })
                        position_map[nome] = position_name
                    
                    st.session_state.df_athletes = pd.DataFrame(atletas)
                    st.session_state.athlete_position_map = position_map
                    st.success(f"✅ {len(atletas)} atletas carregados")
                
                # Carregar atividades
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
                            'id': a.get('id'),
                            'nome': a.get('name'),
                            'data': a.get('start_time')
                        })
                    st.session_state.df_activities = pd.DataFrame(atividades)
                    st.success(f"✅ {len(atividades)} atividades")
                
                st.session_state.api = api
        
        # Filtros
        if not st.session_state.df_activities.empty and token:
            st.markdown("---")
            st.header("🎯 Filtros")
            
            # Filtro por EQUIPE
            if not st.session_state.df_teams.empty:
                st.subheader("🏢 Filtrar por Equipe")
                equipes = ['Todas'] + sorted(st.session_state.df_teams['nome'].unique().tolist())
                equipes_selecionadas = st.multiselect(
                    "Selecione as equipes:",
                    options=equipes,
                    default=['Todas']
                )
                st.session_state.equipes_selecionadas = equipes_selecionadas
            
            # Filtro por POSIÇÃO
            if not st.session_state.df_positions.empty:
                st.subheader("🎯 Filtrar por Posição")
                posicoes = ['Todas'] + sorted(st.session_state.df_positions['nome'].unique().tolist())
                posicoes_selecionadas = st.multiselect(
                    "Selecione as posições:",
                    options=posicoes,
                    default=['Todas']
                )
                st.session_state.posicoes_selecionadas = posicoes_selecionadas
            
            # Selecionar atividade
            st.subheader("📅 Atividade")
            atividade_sel = st.selectbox(
                "Selecione a atividade:", 
                st.session_state.df_activities['nome'].tolist()
            )
            
            if atividade_sel:
                activity_id = st.session_state.df_activities[
                    st.session_state.df_activities['nome'] == atividade_sel
                ]['id'].values[0]
                st.session_state.activity_id = activity_id
                
                # Buscar períodos da atividade
                with st.spinner("Buscando períodos da atividade..."):
                    api = st.session_state.api
                    periods_raw = api.get_activity_periods(activity_id)
                    
                    period_options = {'Atividade Completa': None}
                    if periods_raw and isinstance(periods_raw, list):
                        for p in periods_raw:
                            period_options[p.get('name', 'Período')] = p.get('id')
                        st.session_state.period_options = period_options
                        st.success(f"✅ {len(period_options)-1} períodos encontrados")
                    else:
                        st.session_state.period_options = period_options
                        st.info("ℹ️ Nenhum período específico encontrado")
                
                # Selecionar período
                if 'period_options' in st.session_state:
                    periodo_sel = st.selectbox(
                        "Período:",
                        list(st.session_state.period_options.keys())
                    )
                    st.session_state.periodo_sel = periodo_sel
                    st.session_state.period_id = st.session_state.period_options[periodo_sel]
                
                # Buscar atletas da atividade
                if st.button("🔍 Buscar Atletas da Atividade"):
                    with st.spinner("Buscando atletas..."):
                        api = st.session_state.api
                        period_id = st.session_state.get('period_id')
                        periodo_sel = st.session_state.get('periodo_sel', 'Atividade Completa')
                        
                        if period_id and periodo_sel != 'Atividade Completa':
                            st.info(f"📌 Buscando atletas do período: {periodo_sel}")
                            response_data = api.get_athletes_in_period(period_id)
                        else:
                            st.info("📌 Buscando atletas da atividade completa")
                            response_data = api.get_activity_athletes(activity_id)
                        
                        if response_data:
                            athletes_in = None
                            
                            if isinstance(response_data, list):
                                athletes_in = response_data
                            elif isinstance(response_data, dict):
                                for key in ['data', 'items', 'athletes', 'results']:
                                    if key in response_data and isinstance(response_data[key], list):
                                        athletes_in = response_data[key]
                                        break
                                if athletes_in is None and 'id' in response_data:
                                    athletes_in = [response_data]
                            
                            if athletes_in and len(athletes_in) > 0:
                                st.success(f"📊 Encontrados {len(athletes_in)} atletas na API")
                                
                                atletas_temp = []
                                for a in athletes_in:
                                    nome = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
                                    if not nome:
                                        nome = a.get('name', 'Sem nome')
                                    
                                    athlete_id = a.get('id')
                                    position_name = a.get('position_name', a.get('position', ''))
                                    team_name = st.session_state.athlete_team_map.get(athlete_id, '')
                                    
                                    atletas_temp.append({
                                        'id': athlete_id,
                                        'nome': nome,
                                        'camisa': a.get('jersey', ''),
                                        'posicao': position_name,
                                        'equipe': team_name
                                    })
                                
                                df_atletas_temp = pd.DataFrame(atletas_temp)
                                
                                if 'equipes_selecionadas' in st.session_state:
                                    if 'Todas' not in st.session_state.equipes_selecionadas and st.session_state.equipes_selecionadas:
                                        df_atletas_temp = df_atletas_temp[
                                            df_atletas_temp['equipe'].isin(st.session_state.equipes_selecionadas)
                                        ]
                                
                                if 'posicoes_selecionadas' in st.session_state:
                                    if 'Todas' not in st.session_state.posicoes_selecionadas and st.session_state.posicoes_selecionadas:
                                        df_atletas_temp = df_atletas_temp[
                                            df_atletas_temp['posicao'].isin(st.session_state.posicoes_selecionadas)
                                        ]
                                
                                st.session_state.atletas_filtrados = df_atletas_temp
                                st.success(f"✅ {len(df_atletas_temp)} atletas encontrados no {periodo_sel}")
                                
                                if not df_atletas_temp.empty:
                                    st.subheader("📊 Distribuição")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Por Equipe:**")
                                        for team, qtd in df_atletas_temp['equipe'].value_counts().items():
                                            st.write(f"- {team if team else 'Sem equipe'}: {qtd}")
                                    with col2:
                                        st.write("**Por Posição:**")
                                        for pos, qtd in df_atletas_temp['posicao'].value_counts().items():
                                            st.write(f"- {pos if pos else 'Sem posição'}: {qtd}")
                            else:
                                st.warning(f"⚠️ Nenhum atleta encontrado para {periodo_sel}")
                        else:
                            st.error("Não foi possível buscar atletas. Verifique o token.")
                
                # Selecionar atletas
                if 'atletas_filtrados' in st.session_state and not st.session_state.atletas_filtrados.empty:
                    st.subheader("🏃 Selecionar Atletas")
                    atletas_sel = st.multiselect(
                        "Selecione os atletas para análise:",
                        st.session_state.atletas_filtrados['nome'].tolist()
                    )
                    st.session_state.atletas_sel = atletas_sel
    
    # Área principal
    if ('api' in st.session_state and 'atletas_sel' in st.session_state and 
        st.session_state.atletas_sel and 'activity_id' in st.session_state):
        
        api = st.session_state.api
        activity_id = st.session_state.activity_id
        periodo_sel = st.session_state.get('periodo_sel', 'Atividade Completa')
        period_id = st.session_state.get('period_id')
        
        st.info(f"📌 Atividade: {atividade_sel}")
        st.info(f"📌 Período: {periodo_sel}")
        
        resultados = []
        dados_posicao = {}
        dados_sensor_por_atleta = {}
        dados_efforts_velocidade = {}
        dados_efforts_aceleracao = {}
        
        progresso = st.progress(0)
        status_text = st.empty()
        
        for i, atleta_nome in enumerate(st.session_state.atletas_sel):
            status_text.text(f"Processando {atleta_nome}...")
            
            athlete_row = st.session_state.atletas_filtrados[
                st.session_state.atletas_filtrados['nome'] == atleta_nome
            ]
            if athlete_row.empty:
                continue
                
            athlete_id = athlete_row['id'].values[0]
            athlete_posicao = athlete_row['posicao'].values[0] if 'posicao' in athlete_row.columns else ''
            athlete_equipe = athlete_row['equipe'].values[0] if 'equipe' in athlete_row.columns else ''
            
            if period_id and periodo_sel != 'Atividade Completa':
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
                
                lats = [p.get('lat', 0) for p in sensor_points if p.get('lat')]
                lons = [p.get('long', 0) for p in sensor_points if p.get('long')]
                velocidades = [p.get('v', 0) * 3.6 for p in sensor_points if p.get('v')]
                
                if lats and lons:
                    x, y = lat_lon_to_xy(lats, lons)
                    dados_posicao[atleta_nome] = {
                        'x': x, 'y': y, 'vel': velocidades, 
                        'posicao': athlete_posicao, 'equipe': athlete_equipe
                    }
                
                if efforts_response:
                    vel_efforts, acc_efforts = extrair_efforts_data(efforts_response)
                    if vel_efforts:
                        dados_efforts_velocidade[atleta_nome] = vel_efforts
                    if acc_efforts:
                        dados_efforts_aceleracao[atleta_nome] = acc_efforts
                    
                    st.success(f"✅ {atleta_nome}: {len(sensor_points)} pts sensor | Vel: {len(vel_efforts)} | Acc: {len(acc_efforts)}")
                else:
                    st.success(f"✅ {atleta_nome}: {len(sensor_points)} pontos (sem dados de esforços)")
            
            progresso.progress((i + 1) / len(st.session_state.atletas_sel))
        
        status_text.empty()
        progresso.empty()
        
        if resultados:
            df = pd.DataFrame(resultados)
            
            st.subheader("📊 Métricas Biométricas")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏃 Atletas", len(resultados))
            with col2:
                st.metric("📏 Distância Total", f"{df['Distância (m)'].sum():,.0f} m")
            with col3:
                st.metric("⚡ PlayerLoad Total", f"{df['PlayerLoad'].sum():,.0f}")
            with col4:
                st.metric("💨 Velocidade Máx", f"{df['Velocidade Máx (km/h)'].max():.1f} km/h")
            
            st.markdown("---")
            
            colunas = ['Atleta', 'Equipe', 'Posição', 'Duração (min)', 'Distância (m)', 
                      'PlayerLoad', 'Velocidade Máx (km/h)', 'FC Máx (bpm)']
            colunas_existentes = [c for c in colunas if c in df.columns]
            st.dataframe(df[colunas_existentes], use_container_width=True)
            
            # Abas
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Gráficos Comparativos", 
                "🗺️ Mapas de Calor", 
                "⏱️ Esforços ao Longo do Tempo",
                "📋 Métricas Detalhadas"
            ])
            
            with tab1:
                st.subheader("📈 Gráficos Comparativos")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(df, x='Atleta', y='Distância (m)', color='Equipe',
                                title="Distância por Atleta", color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig, key="dist", use_container_width=True)
                with col2:
                    fig = px.bar(df, x='Atleta', y='PlayerLoad', color='Equipe',
                                title="PlayerLoad por Atleta", color_discrete_sequence=px.colors.qualitative.Set3)
                    st.plotly_chart(fig, key="pl", use_container_width=True)
                
                col3, col4 = st.columns(2)
                with col3:
                    fig = px.bar(df, x='Atleta', y='Velocidade Máx (km/h)', color='Posição',
                                title="Velocidade Máxima por Atleta")
                    st.plotly_chart(fig, key="speed", use_container_width=True)
                with col4:
                    if 'Equipe' in df.columns:
                        team_counts = df['Equipe'].value_counts().reset_index()
                        team_counts.columns = ['Equipe', 'Quantidade']
                        if not team_counts.empty:
                            fig = px.pie(team_counts, values='Quantidade', names='Equipe', title="Atletas por Equipe")
                            st.plotly_chart(fig, key="pie_team", use_container_width=True)
            
            with tab2:
                if dados_posicao:
                    st.subheader("🗺️ Mapas de Calor e Trajetórias")
                    for atleta, dados in dados_posicao.items():
                        with st.expander(f"📍 {atleta} ({dados['equipe']} - {dados['posicao']})", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_heat = criar_mapa_calor(dados['x'], dados['y'], atleta)
                                st.plotly_chart(fig_heat, key=f"heat_{atleta}", use_container_width=True)
                            with col2:
                                fig_traj = criar_trajetoria(dados['x'], dados['y'], dados['vel'], atleta)
                                st.plotly_chart(fig_traj, key=f"traj_{atleta}", use_container_width=True)
                else:
                    st.info("Dados de posição não disponíveis para esta atividade")
            
            with tab3:
                st.subheader("⏱️ Esforços ao Longo do Tempo")
                
                if dados_sensor_por_atleta:
                    atleta_escolhido = st.selectbox(
                        "Selecione o atleta para visualizar os esforços:",
                        list(dados_sensor_por_atleta.keys()),
                        key="esforcos_select"
                    )
                    
                    if atleta_escolhido:
                        sensor_points = dados_sensor_por_atleta[atleta_escolhido]
                        
                        st.markdown("### 🏃‍♂️ Velocidade ao Longo do Tempo")
                        fig_vel = criar_grafico_velocidade_tempo(sensor_points, atleta_escolhido)
                        if fig_vel:
                            st.plotly_chart(fig_vel, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("### 🔄 Aceleração ao Longo do Tempo")
                        fig_acc = criar_grafico_aceleracao_tempo(sensor_points, atleta_escolhido)
                        if fig_acc:
                            st.plotly_chart(fig_acc, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("## 📋 Tabela de Esforços Detalhada")
                        
                        tipo_esforco = st.radio(
                            "Selecione o tipo de esforço:",
                            ["🏃‍♂️ Velocidade", "🔄 Aceleração/Desaceleração"],
                            horizontal=True,
                            key="tipo_esforco"
                        )
                        
                        efforts_df = pd.DataFrame()
                        
                        if "Velocidade" in tipo_esforco:
                            if atleta_escolhido in dados_efforts_velocidade:
                                efforts_data = dados_efforts_velocidade[atleta_escolhido]
                                efforts_df = processar_efforts_velocidade(efforts_data)
                                st.success(f"📊 {len(efforts_data)} esforços de velocidade encontrados")
                            else:
                                st.warning("Dados de esforços de velocidade não disponíveis")
                        else:
                            if atleta_escolhido in dados_efforts_aceleracao:
                                efforts_data = dados_efforts_aceleracao[atleta_escolhido]
                                efforts_df = processar_efforts_aceleracao(efforts_data)
                                st.success(f"📊 {len(efforts_data)} esforços de aceleração encontrados")
                            else:
                                st.warning("Dados de esforços de aceleração não disponíveis")
                        
                        if not efforts_df.empty:
                            # Filtro por bandas
                            if 'Banda' in efforts_df.columns and efforts_df['Banda'].notna().any():
                                bandas_disponiveis = sorted(efforts_df['Banda'].dropna().unique().tolist())
                                if bandas_disponiveis:
                                    st.subheader("🎚️ Filtros")
                                    bandas_selecionadas = st.multiselect(
                                        "Selecione as bandas para filtrar:",
                                        options=bandas_disponiveis,
                                        default=bandas_disponiveis
                                    )
                                    if bandas_selecionadas:
                                        efforts_df = efforts_df[efforts_df['Banda'].isin(bandas_selecionadas)]
                            
                            # Estatísticas
                            st.subheader("📊 Resumo dos Esforços")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total de Esforços", len(efforts_df))
                            with col2:
                                if 'Duração (s)' in efforts_df.columns:
                                    st.metric("Duração Total (s)", round(efforts_df['Duração (s)'].sum(), 1))
                            with col3:
                                if 'Distância (m)' in efforts_df.columns:
                                    st.metric("Distância Total (m)", round(efforts_df['Distância (m)'].sum(), 1))
                            with col4:
                                if '% do Máximo' in efforts_df.columns:
                                    st.metric("Média % do Máximo", round(efforts_df['% do Máximo'].mean(), 1))
                            
                            # Tabela
                            st.subheader("📋 Tabela Detalhada de Esforços")
                            st.dataframe(efforts_df, use_container_width=True, height=400)
                            
                            # Exportar
                            csv_efforts = efforts_df.to_csv(index=False)
                            st.download_button(
                                "📥 Exportar Tabela de Esforços (CSV)",
                                csv_efforts,
                                f"esforcos_{atleta_escolhido}_{tipo_esforco.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
                        else:
                            st.info("Nenhum dado de esforço disponível")
                else:
                    st.info("Dados de sensor não disponíveis")
            
            with tab4:
                st.subheader("📋 Métricas Detalhadas")
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Exportar Dados (CSV)",
                    csv,
                    f"metricas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
        else:
            st.warning("Nenhum dado de sensor encontrado para os atletas selecionados")
    
    elif 'df_athletes' in st.session_state and not st.session_state.df_athletes.empty:
        st.info("👈 Selecione uma atividade, um período e clique em 'Buscar Atletas da Atividade'")
        
        if not st.session_state.df_athletes.empty:
            st.subheader("📊 Distribuição de Posições (Todos os Atletas)")
            pos_counts = st.session_state.df_athletes['posicao'].value_counts().reset_index()
            pos_counts.columns = ['Posição', 'Quantidade']
            if not pos_counts.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(pos_counts, use_container_width=True)
                with col2:
                    fig = px.pie(pos_counts, values='Quantidade', names='Posição', title="Posições")
                    st.plotly_chart(fig, key="pie", use_container_width=True)

if __name__ == "__main__":
    main()