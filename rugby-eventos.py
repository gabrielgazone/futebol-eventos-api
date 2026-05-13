# rugby-eventos.py - VERSÃO COMPLETA COM ESFORÇOS (EFFORTS)
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import base64
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Rugby Eventos - Catapult Sports",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Servidores disponíveis
SERVERS = {
    "Américas (US)": "https://connect-us.catapultsports.com/api/v6",
    "Europa/África (EU)": "https://connect-eu.catapultsports.com/api/v6",
    "Ásia-Pacífico (AU)": "https://connect-au.catapultsports.com/api/v6",
    "China (CN)": "https://connect-cn.catapultsports-cn.com/api/v6"
}

class CatapultAPI:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
    
    def _request(self, endpoint, params=None):
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            return None
    
    def get_athletes(self):
        return self._request("athletes")
    
    def get_activities(self, page=1, page_size=200):
        return self._request(f"activities?page={page}&page_size={page_size}")
    
    def get_teams(self):
        return self._request("teams")
    
    def get_team_athletes(self, team_id):
        return self._request(f"teams/{team_id}/athletes")
    
    def get_activity_sensor(self, activity_id, athlete_id):
        return self._request(f"activities/{activity_id}/athletes/{athlete_id}/sensor")
    
    def get_activity_periods(self, activity_id):
        return self._request(f"activities/{activity_id}/periods")
    
    def get_period_sensor(self, period_id, athlete_id):
        return self._request(f"periods/{period_id}/athletes/{athlete_id}/sensor")
    
    def get_activity_athletes(self, activity_id):
        return self._request(f"activities/{activity_id}/athletes")
    
    # ==================== NOVO: ENDPOINT DE ESFORÇOS (EFFORTS) ====================
    def get_activity_efforts(self, activity_id, athlete_id, effort_types="acceleration,velocity", velocity_bands=None, acceleration_bands=None):
        """
        GET /activities/{id}/athletes/{id}/efforts
        Retorna dados de esforços (aceleração e velocidade) do atleta na atividade
        - effort_types: 'acceleration', 'velocity' ou 'acceleration,velocity'
        - velocity_bands: números 1 a 8 (ex: '1,2,3')
        - acceleration_bands: números -3 a 3 (ex: '-3,-2,-1,0,1,2,3')
        """
        params = {"effort_types": effort_types}
        if velocity_bands:
            params["velocity_bands"] = velocity_bands
        if acceleration_bands:
            params["acceleration_bands"] = acceleration_bands
        return self._request(f"activities/{activity_id}/athletes/{athlete_id}/efforts", params=params)
    
    def get_period_efforts(self, period_id, athlete_id, effort_types="acceleration,velocity", velocity_bands=None, acceleration_bands=None):
        """
        GET /periods/{period_id}/athletes/{athlete_id}/efforts
        Retorna dados de esforços (aceleração e velocidade) do atleta no período
        """
        params = {"effort_types": effort_types}
        if velocity_bands:
            params["velocity_bands"] = velocity_bands
        if acceleration_bands:
            params["acceleration_bands"] = acceleration_bands
        return self._request(f"periods/{period_id}/athletes/{athlete_id}/efforts", params=params)

def decode_token(token):
    try:
        payload = token.split('.')[1]
        payload += '=' * (4 - len(payload) % 4)
        return json.loads(base64.b64decode(payload))
    except:
        return {}

def process_efforts_data(efforts_data, effort_type):
    """Processa dados de esforços (aceleração/velocidade) para DataFrame"""
    if not efforts_data:
        return pd.DataFrame()
    
    if isinstance(efforts_data, dict):
        if 'data' in efforts_data:
            return process_efforts_data(efforts_data['data'], effort_type)
        efforts_data = [efforts_data]
    
    if not isinstance(efforts_data, list):
        return pd.DataFrame()
    
    records = []
    for item in efforts_data:
        record = {
            'Banda': item.get('band', item.get('name', '')),
            'Quantidade': item.get('count', item.get('value', 0)),
            'Duração (s)': item.get('duration', item.get('time', 0)),
            'Percentual (%)': item.get('percentage', 0)
        }
        records.append(record)
    
    return pd.DataFrame(records)

def process_sensor_data(sensor_data):
    if not sensor_data:
        return None, {}
    
    if isinstance(sensor_data, dict):
        if 'data' in sensor_data:
            return process_sensor_data(sensor_data['data'])
        sensor_data = [sensor_data]
    
    if not isinstance(sensor_data, list) or len(sensor_data) == 0:
        return None, {}
    
    records = []
    for point in sensor_data:
        record = {}
        
        if 'ts' in point:
            record['timestamp'] = float(point['ts'])
        elif 'cs' in point:
            record['timestamp'] = float(point['cs']) / 100
        
        if 'v' in point:
            record['velocity_ms'] = float(point['v'])
            record['velocity_kmh'] = float(point['v']) * 3.6
        
        if 'a' in point:
            record['acceleration'] = float(point['a'])
        
        if 'hr' in point:
            record['heart_rate'] = float(point['hr'])
        
        records.append(record)
    
    if not records:
        return None, {}
    
    df = pd.DataFrame(records)
    
    if 'timestamp' in df.columns:
        min_time = df['timestamp'].min()
        df['time_seconds'] = df['timestamp'] - min_time
    else:
        df['time_seconds'] = range(len(df))
    
    metrics = {}
    
    if 'timestamp' in df.columns:
        duration = df['timestamp'].max() - df['timestamp'].min()
        metrics['Duração (s)'] = round(duration, 1)
        metrics['Duração (min)'] = round(duration / 60, 1)
    
    if 'velocity_kmh' in df.columns:
        metrics['Velocidade Máx (km/h)'] = round(df['velocity_kmh'].max(), 2)
        metrics['Velocidade Média (km/h)'] = round(df['velocity_kmh'].mean(), 2)
        if 'velocity_ms' in df.columns:
            metrics['Distância Estimada (m)'] = round(df['velocity_ms'].sum() * 0.1, 0)
    
    if 'heart_rate' in df.columns:
        metrics['FC Máx (bpm)'] = round(df['heart_rate'].max(), 0)
        metrics['FC Média (bpm)'] = round(df['heart_rate'].mean(), 0)
    
    metrics['Total de Pontos'] = len(df)
    
    return df, metrics

def main():
    st.title("🏉 Rugby Eventos - Catapult Sports")
    st.markdown("### Análise de Performance - Dados Sensor 10Hz e Esforços")
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("🌍 Servidor")
        server_name = st.selectbox("Selecione:", list(SERVERS.keys()))
        base_url = SERVERS[server_name]
        
        st.header("🔐 Token")
        token = st.text_area("Token JWT:", height=80, placeholder="Cole seu token...")
        
        if token:
            info = decode_token(token)
            if info.get('exp'):
                exp_date = datetime.fromtimestamp(info['exp'])
                days = (exp_date - datetime.now()).days
                if days > 0:
                    st.success(f"✅ Válido até: {exp_date.strftime('%d/%m/%Y')}")
        
        if token:
            api = CatapultAPI(base_url, token)
            st.session_state.api = api
            
            with st.spinner("Carregando dados..."):
                athletes_raw = api.get_athletes()
                teams_raw = api.get_teams()
                activities_raw = api.get_activities()
                
                # Processar times
                if teams_raw:
                    teams_data = [{'ID': t.get('id', ''), 'Time': t.get('name', 'Sem nome')} for t in teams_raw]
                    st.session_state.df_teams = pd.DataFrame(teams_data)
                    
                    team_athletes_map = {}
                    for _, team in st.session_state.df_teams.iterrows():
                        athletes_in_team = api.get_team_athletes(team['ID'])
                        if athletes_in_team:
                            team_athletes_map[team['Time']] = athletes_in_team
                    st.session_state.team_athletes_map = team_athletes_map
                
                # Processar atletas
                if athletes_raw:
                    athletes_data = []
                    for a in athletes_raw:
                        first = a.get('first_name', '')
                        last = a.get('last_name', '')
                        name = f"{first} {last}".strip()
                        if not name:
                            name = a.get('name', 'Sem nome')
                        
                        vel_max = a.get('velocity_max', '')
                        if vel_max and vel_max != '':
                            try:
                                vel_max_kmh = float(vel_max) * 3.6
                                vel_max_display = round(vel_max_kmh, 1)
                            except:
                                vel_max_display = vel_max
                        else:
                            vel_max_display = ''
                        
                        athletes_data.append({
                            'ID': a.get('id', ''),
                            'Atleta': name,
                            'Camisa': a.get('jersey', ''),
                            'Peso (kg)': a.get('weight', ''),
                            'Altura (cm)': a.get('height', ''),
                            'Velocidade Máx (km/h)': vel_max_display,
                            'FC Máx': a.get('heart_rate_max', '')
                        })
                    st.session_state.df_athletes = pd.DataFrame(athletes_data)
                    
                    athlete_to_team = {}
                    for team_name, athletes_list in st.session_state.team_athletes_map.items():
                        for athlete in athletes_list:
                            athlete_name = athlete.get('name', '')
                            if not athlete_name:
                                athlete_name = f"{athlete.get('first_name', '')} {athlete.get('last_name', '')}".strip()
                            if athlete_name:
                                athlete_to_team[athlete_name] = team_name
                    st.session_state.df_athletes['Equipe'] = st.session_state.df_athletes['Atleta'].map(athlete_to_team).fillna('Sem equipe')
                
                # Processar atividades
                if activities_raw:
                    if isinstance(activities_raw, dict):
                        act_list = activities_raw.get('data', activities_raw.get('items', []))
                    else:
                        act_list = activities_raw
                    
                    activities_data = []
                    for act in act_list:
                        start = act.get('start_time')
                        start_date = ''
                        if start:
                            try:
                                start_dt = datetime.fromtimestamp(float(start))
                                start_date = start_dt.strftime('%d/%m/%Y')
                            except:
                                start_date = str(start)
                        
                        activities_data.append({
                            'ID': act.get('id', ''),
                            'Atividade': act.get('name', 'Sem nome'),
                            'Data': start_date
                        })
                    st.session_state.df_activities = pd.DataFrame(activities_data)
                    
                    st.session_state.df_activities['Time_Associado'] = 'Todos'
                    for team in st.session_state.df_teams['Time'].tolist():
                        mask = st.session_state.df_activities['Atividade'].str.contains(team, case=False, na=False)
                        st.session_state.df_activities.loc[mask, 'Time_Associado'] = team
                
                st.success("✅ Dados carregados!")
        
        st.markdown("---")
        
        # ==================== FILTROS ====================
        if st.session_state.get('df_teams') is not None:
            st.header("🎯 Filtros")
            
            st.subheader("🏢 Time")
            team_options = ['Todos'] + st.session_state.df_teams['Time'].tolist()
            selected_teams = st.multiselect("Selecione os times:", team_options, default=['Todos'])
            
            st.subheader("📅 Data")
            available_dates = st.session_state.df_activities['Data'].dropna().unique()
            available_dates = sorted(available_dates)
            
            if len(available_dates) > 0:
                date_options = ['Todas'] + available_dates
                selected_dates = st.multiselect("Selecione as datas:", date_options, default=['Todas'])
            else:
                selected_dates = ['Todas']
            
            st.subheader("📊 Atividade")
            if 'Todos' not in selected_teams:
                filtered_by_team = st.session_state.df_activities[
                    st.session_state.df_activities['Time_Associado'].isin(selected_teams) | 
                    (st.session_state.df_activities['Time_Associado'] == 'Todos')
                ]
            else:
                filtered_by_team = st.session_state.df_activities
            
            if 'Todas' not in selected_dates:
                filtered_activities = filtered_by_team[filtered_by_team['Data'].isin(selected_dates)]
            else:
                filtered_activities = filtered_by_team
            
            activity_options = filtered_activities['Atividade'].tolist()
            selected_activity = st.selectbox("Selecione uma atividade:", activity_options if activity_options else [''])
            
            st.subheader("🏃 Atletas")
            filtered_athletes = st.session_state.df_athletes.copy()
            if 'Todos' not in selected_teams:
                filtered_athletes = filtered_athletes[filtered_athletes['Equipe'].isin(selected_teams)]
            
            athlete_options = filtered_athletes['Atleta'].tolist()
            selected_athletes = st.multiselect("Selecione os atletas:", athlete_options, default=athlete_options[:1] if athlete_options else [])
            
            st.subheader("⚡ Configuração de Esforços")
            effort_types = st.multiselect(
                "Tipos de esforço:",
                options=["acceleration", "velocity"],
                default=["acceleration", "velocity"]
            )
            
            st.subheader("🏷️ Bandas")
            show_velocity_bands = st.checkbox("Mostrar bandas de velocidade", value=True)
            show_acceleration_bands = st.checkbox("Mostrar bandas de aceleração", value=True)
    
    # ==================== TELA INICIAL ====================
    if st.session_state.get('df_athletes') is not None:
        
        display_athletes = st.session_state.df_athletes.copy()
        if 'Todos' not in selected_teams:
            display_athletes = display_athletes[display_athletes['Equipe'].isin(selected_teams)]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏃 Total de Atletas", len(display_athletes))
        with col2:
            st.metric("🏢 Equipes", display_athletes['Equipe'].nunique())
        with col3:
            st.metric("📊 Atividades", len(filtered_activities) if 'filtered_activities' in locals() else len(st.session_state.df_activities))
        with col4:
            vel_values = pd.to_numeric(display_athletes['Velocidade Máx (km/h)'], errors='coerce')
            st.metric("⚡ Vel. Média", f"{vel_values.mean():.1f} km/h" if not vel_values.isna().all() else "N/A")
        
        st.markdown("---")
        
        # ==================== TABELA DE ATLETAS ====================
        st.subheader(f"📋 Informações dos Atletas ({len(display_athletes)})")
        athlete_cols = ['Atleta', 'Equipe', 'Camisa', 'Altura (cm)', 'Peso (kg)', 'Velocidade Máx (km/h)', 'FC Máx']
        existing_cols = [col for col in athlete_cols if col in display_athletes.columns]
        st.dataframe(display_athletes[existing_cols], use_container_width=True)
        
        csv_athletes = display_athletes[existing_cols].to_csv(index=False)
        st.download_button("📥 Download Dados dos Atletas (CSV)", csv_athletes, f"atletas_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # ==================== GRÁFICOS DOS ATLETAS ====================
        with st.expander("📊 Estatísticas dos Atletas"):
            col1, col2 = st.columns(2)
            with col1:
                if 'Velocidade Máx (km/h)' in display_athletes.columns:
                    vel_data = display_athletes[pd.to_numeric(display_athletes['Velocidade Máx (km/h)'], errors='coerce').notna()]
                    if not vel_data.empty:
                        vel_data['Velocidade Máx (km/h)'] = pd.to_numeric(vel_data['Velocidade Máx (km/h)'], errors='coerce')
                        fig_vel = px.bar(
                            vel_data.nlargest(15, 'Velocidade Máx (km/h)'),
                            x='Atleta', y='Velocidade Máx (km/h)',
                            title="Top 15 - Velocidade Máxima (km/h)",
                            color='Velocidade Máx (km/h)',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_vel, use_container_width=True)
            with col2:
                team_counts = display_athletes['Equipe'].value_counts().reset_index()
                team_counts.columns = ['Equipe', 'Quantidade']
                fig_team = px.pie(team_counts, values='Quantidade', names='Equipe', title="Distribuição por Equipe")
                st.plotly_chart(fig_team, use_container_width=True)
        
        st.markdown("---")
        
        # ==================== ANÁLISE DE SENSOR E ESFORÇOS ====================
        if selected_activity and selected_athletes:
            activity_id = st.session_state.df_activities[st.session_state.df_activities['Atividade'] == selected_activity]['ID'].values[0]
            
            # Buscar períodos da atividade
            periods = st.session_state.api.get_activity_periods(activity_id)
            period_options = {}
            if periods:
                for p in periods:
                    period_options[p.get('name', f'Período')] = p.get('id')
            
            # Abas para diferentes análises
            tab1, tab2, tab3 = st.tabs(["📈 Dados de Sensor", "⚡ Esforços (Velocidade)", "🔄 Esforços (Aceleração)"])
            
            # ==================== TAB 1: DADOS DE SENSOR ====================
            with tab1:
                st.subheader("📈 Dados de Sensor 10Hz")
                
                period_type = st.radio("Período:", ["Atividade Completa", "Período Específico"], horizontal=True, key="sensor_period_type")
                
                all_sensor_data = []
                progress_bar = st.progress(0)
                
                for idx, athlete_name in enumerate(selected_athletes):
                    athlete_id_val = st.session_state.df_athletes[st.session_state.df_athletes['Atleta'] == athlete_name]['ID'].values[0]
                    
                    if period_type == "Atividade Completa":
                        sensor_data = st.session_state.api.get_activity_sensor(activity_id, athlete_id_val)
                    else:
                        if period_options:
                            selected_period = st.selectbox("Período:", list(period_options.keys()), key=f"period_{athlete_name}")
                            if selected_period:
                                period_id = period_options[selected_period]
                                sensor_data = st.session_state.api.get_period_sensor(period_id, athlete_id_val)
                        else:
                            sensor_data = None
                    
                    if sensor_data:
                        df, metrics = process_sensor_data(sensor_data)
                        if metrics:
                            metrics['Atleta'] = athlete_name
                            all_sensor_data.append(metrics)
                    
                    progress_bar.progress((idx + 1) / len(selected_athletes))
                
                progress_bar.empty()
                
                if all_sensor_data:
                    df_sensor = pd.DataFrame(all_sensor_data)
                    st.subheader("📊 Comparativo de Dados de Sensor")
                    
                    sensor_params = ['Atleta', 'Duração (min)', 'Velocidade Máx (km/h)', 'Velocidade Média (km/h)', 'Distância Estimada (m)', 'FC Máx (bpm)', 'Total de Pontos']
                    available_params = [p for p in sensor_params if p in df_sensor.columns]
                    st.dataframe(df_sensor[available_params], use_container_width=True)
                    
                    csv_sensor = df_sensor[available_params].to_csv(index=False)
                    st.download_button("📥 Download Dados de Sensor (CSV)", csv_sensor, f"sensor_{selected_activity}.csv")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'Velocidade Máx (km/h)' in df_sensor.columns:
                            fig = px.bar(df_sensor, x='Atleta', y='Velocidade Máx (km/h)', title="Velocidade Máxima por Atleta")
                            st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        if 'Distância Estimada (m)' in df_sensor.columns:
                            fig = px.bar(df_sensor, x='Atleta', y='Distância Estimada (m)', title="Distância por Atleta")
                            st.plotly_chart(fig, use_container_width=True)
            
            # ==================== TAB 2: ESFORÇOS DE VELOCIDADE ====================
            with tab2:
                st.subheader("⚡ Esforços - Bandas de Velocidade")
                st.caption("Distribuição do tempo em diferentes faixas de velocidade (Bandas 1-8)")
                
                period_type_vel = st.radio("Período:", ["Atividade Completa", "Período Específico"], horizontal=True, key="vel_period_type")
                
                effort_types_str = ",".join([et for et in effort_types if et == "velocity"])
                if not effort_types_str:
                    effort_types_str = "velocity"
                
                all_velocity_data = {}
                progress_bar = st.progress(0)
                
                for idx, athlete_name in enumerate(selected_athletes):
                    athlete_id_val = st.session_state.df_athletes[st.session_state.df_athletes['Atleta'] == athlete_name]['ID'].values[0]
                    
                    if period_type_vel == "Atividade Completa":
                        efforts_data = st.session_state.api.get_activity_efforts(activity_id, athlete_id_val, effort_types=effort_types_str)
                    else:
                        if period_options:
                            selected_period = st.selectbox("Período:", list(period_options.keys()), key=f"vel_period_{athlete_name}")
                            if selected_period:
                                period_id = period_options[selected_period]
                                efforts_data = st.session_state.api.get_period_efforts(period_id, athlete_id_val, effort_types=effort_types_str)
                        else:
                            efforts_data = None
                    
                    if efforts_data:
                        df_efforts = process_efforts_data(efforts_data, "velocity")
                        if not df_efforts.empty:
                            all_velocity_data[athlete_name] = df_efforts
                    
                    progress_bar.progress((idx + 1) / len(selected_athletes))
                
                progress_bar.empty()
                
                if all_velocity_data:
                    for athlete_name, df_efforts in all_velocity_data.items():
                        st.subheader(f"🏃 {athlete_name}")
                        st.dataframe(df_efforts, use_container_width=True)
                        
                        if not df_efforts.empty and 'Percentual (%)' in df_efforts.columns:
                            fig = px.bar(df_efforts, x='Banda', y='Percentual (%)', 
                                        title=f"Distribuição de Velocidade - {athlete_name}",
                                        color='Percentual (%)', color_continuous_scale='Viridis')
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Nenhum dado de esforço de velocidade encontrado")
            
            # ==================== TAB 3: ESFORÇOS DE ACELERAÇÃO ====================
            with tab3:
                st.subheader("🔄 Esforços - Bandas de Aceleração/Desaceleração")
                st.caption("Bandas negativas = desaceleração | Banda 0 = velocidade constante | Bandas positivas = aceleração")
                
                period_type_acc = st.radio("Período:", ["Atividade Completa", "Período Específico"], horizontal=True, key="acc_period_type")
                
                effort_types_acc = ",".join([et for et in effort_types if et == "acceleration"])
                if not effort_types_acc:
                    effort_types_acc = "acceleration"
                
                all_acceleration_data = {}
                progress_bar = st.progress(0)
                
                for idx, athlete_name in enumerate(selected_athletes):
                    athlete_id_val = st.session_state.df_athletes[st.session_state.df_athletes['Atleta'] == athlete_name]['ID'].values[0]
                    
                    if period_type_acc == "Atividade Completa":
                        efforts_data = st.session_state.api.get_activity_efforts(activity_id, athlete_id_val, effort_types=effort_types_acc)
                    else:
                        if period_options:
                            selected_period = st.selectbox("Período:", list(period_options.keys()), key=f"acc_period_{athlete_name}")
                            if selected_period:
                                period_id = period_options[selected_period]
                                efforts_data = st.session_state.api.get_period_efforts(period_id, athlete_id_val, effort_types=effort_types_acc)
                        else:
                            efforts_data = None
                    
                    if efforts_data:
                        df_efforts = process_efforts_data(efforts_data, "acceleration")
                        if not df_efforts.empty:
                            all_acceleration_data[athlete_name] = df_efforts
                    
                    progress_bar.progress((idx + 1) / len(selected_athletes))
                
                progress_bar.empty()
                
                if all_acceleration_data:
                    for athlete_name, df_efforts in all_acceleration_data.items():
                        st.subheader(f"🏃 {athlete_name}")
                        st.dataframe(df_efforts, use_container_width=True)
                        
                        if not df_efforts.empty and 'Percentual (%)' in df_efforts.columns:
                            # Ordenar bandas corretamente (-3 a 3)
                            df_efforts['Banda_Num'] = pd.to_numeric(df_efforts['Banda'], errors='coerce')
                            df_efforts = df_efforts.sort_values('Banda_Num')
                            
                            cores = ['red' if int(b) < 0 else 'gray' if int(b) == 0 else 'green' for b in df_efforts['Banda_Num']]
                            
                            fig = px.bar(df_efforts, x='Banda', y='Percentual (%)', 
                                        title=f"Distribuição de Aceleração/Desaceleração - {athlete_name}",
                                        color='Percentual (%)', color_continuous_scale='RdYlGn')
                            fig.update_layout(xaxis_title="Banda de Aceleração", yaxis_title="Percentual do Tempo (%)")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Nenhum dado de esforço de aceleração encontrado")
    
    else:
        st.info("👈 **Selecione o servidor, cole seu token e aguarde o carregamento dos dados**")
        
        st.markdown("""
        ### 🏉 Catapult Sports - Análise de Sensor 10Hz e Esforços
        
        **Funcionalidades:**
        
        | Filtro | Descrição |
        |--------|-----------|
        | 🏢 Time | Selecione um ou mais times |
        | 📅 Data | Selecione uma ou mais datas |
        | 📊 Atividade | Escolha a atividade/jogo |
        | 🏃 Atletas | Selecione **múltiplos atletas** para comparar |
        
        **Análises disponíveis:**
        
        | Aba | Dados |
        |-----|-------|
        | 📈 Dados de Sensor | Velocidade, distância, frequência cardíaca |
        | ⚡ Esforços (Velocidade) | Bandas de velocidade (1-8) |
        | 🔄 Esforços (Aceleração) | Bandas de aceleração/desaceleração (-3 a 3) |
        """)

if __name__ == "__main__":
    main()