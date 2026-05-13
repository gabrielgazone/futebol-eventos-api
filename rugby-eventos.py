# rugby-eventos.py - VERSÃO CORRIGIDA (km/h e filtro de data na sidebar)
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
    
    def get_activity_athletes(self, activity_id):
        return self._request(f"activities/{activity_id}/athletes")

def decode_token(token):
    try:
        payload = token.split('.')[1]
        payload += '=' * (4 - len(payload) % 4)
        return json.loads(base64.b64decode(payload))
    except:
        return {}

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
        
        # Velocidade - converter para km/h
        if 'v' in point:
            record['velocity_ms'] = float(point['v'])
            record['velocity_kmh'] = float(point['v']) * 3.6  # CONVERSÃO PARA km/h
        
        if 'a' in point:
            record['acceleration'] = float(point['a'])
        
        if 'hr' in point:
            record['heart_rate'] = float(point['hr'])
        
        if 'pl' in point:
            record['player_load'] = float(point['pl'])
        
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
    else:
        metrics['Duração (s)'] = len(df) * 0.1
        metrics['Duração (min)'] = round(metrics['Duração (s)'] / 60, 1)
    
    # Métricas de velocidade (já em km/h)
    if 'velocity_kmh' in df.columns:
        metrics['Velocidade Máx (km/h)'] = round(df['velocity_kmh'].max(), 2)
        metrics['Velocidade Média (km/h)'] = round(df['velocity_kmh'].mean(), 2)
        # Distância em metros (usando velocidade em m/s)
        if 'velocity_ms' in df.columns:
            metrics['Distância Estimada (m)'] = round(df['velocity_ms'].sum() * 0.1, 0)
    
    if 'heart_rate' in df.columns:
        metrics['FC Máx (bpm)'] = round(df['heart_rate'].max(), 0)
        metrics['FC Média (bpm)'] = round(df['heart_rate'].mean(), 0)
    
    if 'player_load' in df.columns:
        metrics['Player Load Total'] = round(df['player_load'].sum(), 2)
    
    metrics['Total de Pontos'] = len(df)
    
    return df, metrics

def main():
    st.title("🏉 Rugby Eventos - Catapult Sports")
    st.markdown("### Análise de Performance - Dados Sensor 10Hz")
    
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
                
                # Processar atletas com velocidade em km/h
                if athletes_raw:
                    athletes_data = []
                    for a in athletes_raw:
                        first = a.get('first_name', '')
                        last = a.get('last_name', '')
                        name = f"{first} {last}".strip()
                        if not name:
                            name = a.get('name', 'Sem nome')
                        
                        # Converter velocidade máxima de m/s para km/h
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
                    
                    # Mapear atleta para time
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
                                activity_date = start_dt.date()
                            except:
                                start_date = str(start)
                                activity_date = None
                        
                        activity_name = act.get('name', 'Sem nome')
                        
                        activities_data.append({
                            'ID': act.get('id', ''),
                            'Atividade': activity_name,
                            'Data': start_date,
                            'Data_Obj': activity_date
                        })
                    st.session_state.df_activities = pd.DataFrame(activities_data)
                    
                    # Associar atividades a times pelo nome
                    st.session_state.df_activities['Time_Associado'] = 'Todos'
                    for team in st.session_state.df_teams['Time'].tolist():
                        mask = st.session_state.df_activities['Atividade'].str.contains(team, case=False, na=False)
                        st.session_state.df_activities.loc[mask, 'Time_Associado'] = team
                
                st.success("✅ Dados carregados!")
        
        st.markdown("---")
        
        # ==================== FILTROS EM CASCATA ====================
        if st.session_state.get('df_teams') is not None:
            st.header("🎯 Filtros")
            
            # 1. FILTRO DE TIME
            st.subheader("🏢 Time")
            team_options = ['Todos'] + st.session_state.df_teams['Time'].tolist()
            selected_teams = st.multiselect("Selecione os times:", team_options, default=['Todos'])
            
            # 2. FILTRO DE DATA (na sidebar como solicitado)
            st.subheader("📅 Data")
            
            # Obter todas as datas disponíveis das atividades
            available_dates = st.session_state.df_activities['Data'].dropna().unique()
            available_dates = sorted(available_dates)
            
            if len(available_dates) > 0:
                date_options = ['Todas'] + available_dates
                selected_dates = st.multiselect(
                    "Selecione as datas:",
                    options=date_options,
                    default=['Todas']
                )
            else:
                selected_dates = ['Todas']
                st.info("Nenhuma data disponível")
            
            # 3. FILTRO DE ATIVIDADE (baseado no time E na data)
            st.subheader("📊 Atividade")
            
            # Filtrar atividades pelo time selecionado
            if 'Todos' not in selected_teams:
                filtered_by_team = st.session_state.df_activities[
                    st.session_state.df_activities['Time_Associado'].isin(selected_teams) | 
                    (st.session_state.df_activities['Time_Associado'] == 'Todos')
                ]
            else:
                filtered_by_team = st.session_state.df_activities
            
            # Filtrar atividades pela data selecionada
            if 'Todas' not in selected_dates:
                filtered_activities = filtered_by_team[filtered_by_team['Data'].isin(selected_dates)]
            else:
                filtered_activities = filtered_by_team
            
            activity_options = filtered_activities['Atividade'].tolist()
            selected_activity = st.selectbox(
                "Selecione uma atividade:",
                options=activity_options if activity_options else ['']
            ) if activity_options else None
            
            # 4. FILTRO DE ATLETA (baseado no time selecionado)
            st.subheader("🏃 Atletas")
            
            filtered_athletes = st.session_state.df_athletes.copy()
            if 'Todos' not in selected_teams:
                filtered_athletes = filtered_athletes[filtered_athletes['Equipe'].isin(selected_teams)]
            
            athlete_options = filtered_athletes['Atleta'].tolist()
            selected_athletes = st.multiselect(
                "Selecione os atletas:",
                options=athlete_options,
                default=athlete_options[:1] if athlete_options else []
            )
    
    # ==================== TELA INICIAL - DADOS DOS ATLETAS ====================
    if st.session_state.get('df_athletes') is not None:
        
        # Filtrar atletas pelo time selecionado
        display_athletes = st.session_state.df_athletes.copy()
        if 'Todos' not in selected_teams:
            display_athletes = display_athletes[display_athletes['Equipe'].isin(selected_teams)]
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏃 Total de Atletas", len(display_athletes))
        with col2:
            st.metric("🏢 Equipes", display_athletes['Equipe'].nunique())
        with col3:
            st.metric("📊 Atividades", len(filtered_activities) if 'filtered_activities' in locals() else len(st.session_state.df_activities))
        with col4:
            # Calcular média de velocidade (convertendo para numérico)
            vel_values = pd.to_numeric(display_athletes['Velocidade Máx (km/h)'], errors='coerce')
            st.metric("⚡ Vel. Média", f"{vel_values.mean():.1f} km/h" if not vel_values.isna().all() else "N/A")
        
        st.markdown("---")
        
        # ==================== TABELA DE ATLETAS ====================
        st.subheader(f"📋 Informações dos Atletas ({len(display_athletes)})")
        
        # Colunas para exibir
        athlete_cols = ['Atleta', 'Equipe', 'Camisa', 'Altura (cm)', 'Peso (kg)', 'Velocidade Máx (km/h)', 'FC Máx']
        existing_cols = [col for col in athlete_cols if col in display_athletes.columns]
        
        st.dataframe(display_athletes[existing_cols], use_container_width=True)
        
        # Download
        csv_athletes = display_athletes[existing_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Dados dos Atletas (CSV)",
            data=csv_athletes,
            file_name=f"atletas_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        
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
                            color_continuous_scale='RdYlGn',
                            text='Velocidade Máx (km/h)'
                        )
                        fig_vel.update_traces(textposition='outside')
                        st.plotly_chart(fig_vel, use_container_width=True)
            
            with col2:
                team_counts = display_athletes['Equipe'].value_counts().reset_index()
                team_counts.columns = ['Equipe', 'Quantidade']
                fig_team = px.pie(
                    team_counts, values='Quantidade', names='Equipe',
                    title="Distribuição de Atletas por Equipe"
                )
                st.plotly_chart(fig_team, use_container_width=True)
        
        st.markdown("---")
        
        # ==================== ANÁLISE DE SENSOR ====================
        if selected_activity and selected_athletes:
            
            activity_id = st.session_state.df_activities[st.session_state.df_activities['Atividade'] == selected_activity]['ID'].values[0]
            
            # Buscar dados para cada atleta selecionado
            all_athletes_data = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, athlete_name in enumerate(selected_athletes):
                status_text.text(f"Carregando dados de {athlete_name}...")
                
                athlete_id = st.session_state.df_athletes[st.session_state.df_athletes['Atleta'] == athlete_name]['ID'].values[0]
                
                sensor_data = st.session_state.api.get_activity_sensor(activity_id, athlete_id)
                
                if sensor_data:
                    df, metrics = process_sensor_data(sensor_data)
                    if metrics:
                        metrics['Atleta'] = athlete_name
                        metrics['Equipe'] = st.session_state.df_athletes[st.session_state.df_athletes['Atleta'] == athlete_name]['Equipe'].values[0]
                        all_athletes_data.append(metrics)
                
                progress_bar.progress((idx + 1) / len(selected_athletes))
            
            status_text.text("✅ Carregamento concluído!")
            progress_bar.empty()
            
            if all_athletes_data:
                st.session_state.comparison_df = pd.DataFrame(all_athletes_data)
                st.success(f"✅ Dados de sensor carregados para {len(all_athletes_data)} atletas!")
                
                # ==================== TABELA COMPARATIVA ====================
                st.subheader("📊 Dados de Sensor por Atleta")
                
                df_compare = st.session_state.comparison_df
                
                exclude_cols = ['Atleta', 'Equipe']
                available_params = [col for col in df_compare.columns if col not in exclude_cols]
                
                st.markdown("**Selecione os parâmetros para exibir:**")
                param_cols = st.columns(4)
                selected_params = ['Atleta', 'Equipe']
                
                for i, param in enumerate(available_params):
                    with param_cols[i % 4]:
                        if st.checkbox(param, value=True, key=f"sensor_{param}"):
                            selected_params.append(param)
                
                display_df = df_compare[selected_params].copy()
                for col in display_df.columns:
                    if col not in ['Atleta', 'Equipe']:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x
                        )
                
                st.dataframe(display_df, use_container_width=True)
                
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Dados de Sensor (CSV)",
                    data=csv,
                    file_name=f"sensor_{selected_activity}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                )
                
                # ==================== GRÁFICOS ====================
                with st.expander("📈 Gráficos dos Dados de Sensor"):
                    
                    if 'Velocidade Máx (km/h)' in df_compare.columns:
                        fig_vel = px.bar(
                            df_compare, x='Atleta', y='Velocidade Máx (km/h)',
                            title="Velocidade Máxima por Atleta (km/h)",
                            color='Velocidade Máx (km/h)',
                            color_continuous_scale='RdYlGn',
                            text='Velocidade Máx (km/h)'
                        )
                        fig_vel.update_traces(textposition='outside')
                        st.plotly_chart(fig_vel, use_container_width=True)
                    
                    if 'Distância Estimada (m)' in df_compare.columns:
                        fig_dist = px.bar(
                            df_compare, x='Atleta', y='Distância Estimada (m)',
                            title="Distância Percorrida por Atleta (metros)",
                            color='Distância Estimada (m)',
                            color_continuous_scale='Blues',
                            text='Distância Estimada (m)'
                        )
                        fig_dist.update_traces(textposition='outside')
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    if 'FC Máx (bpm)' in df_compare.columns:
                        fig_fc = px.bar(
                            df_compare, x='Atleta', y='FC Máx (bpm)',
                            title="Frequência Cardíaca Máxima por Atleta (bpm)",
                            color='FC Máx (bpm)',
                            color_continuous_scale='Reds',
                            text='FC Máx (bpm)'
                        )
                        fig_fc.update_traces(textposition='outside')
                        st.plotly_chart(fig_fc, use_container_width=True)
    
    else:
        st.info("👈 **Selecione o servidor, cole seu token e aguarde o carregamento dos dados**")
        
        st.markdown("""
        ### 🏉 Catapult Sports - Análise de Sensor 10Hz
        
        **Funcionalidades:**
        
        | Filtro | Descrição |
        |--------|-----------|
        | 🏢 Time | Selecione um ou mais times |
        | 📅 Data | Selecione uma ou mais datas (na sidebar) |
        | 📊 Atividade | Atualiza automaticamente baseado no time E na data |
        | 🏃 Atletas | Selecione **múltiplos atletas** para comparar |
        
        **Tela Inicial:**
        - Informações básicas dos atletas (nome, equipe, camisa, peso, altura, **velocidade máxima em km/h**)
        - Gráficos de distribuição por equipe e top velocidade
        
        **Análise de Sensor:**
        - Tabela comparativa entre atletas selecionados
        - Gráficos de velocidade (km/h), distância (m) e FC (bpm)
        - Download dos dados em CSV
        """)

if __name__ == "__main__":
    main()