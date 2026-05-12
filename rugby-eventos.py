import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Carrega o token do arquivo .env
load_dotenv()
TOKEN = os.getenv("CATAPULT_TOKEN")

# Configuração da API (do seu token, a região é 'us')
BASE_URL = "https://backend-us.openfield.catapultsports.com/api/v1"

# Headers para todas as requisições
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Cache para evitar sobrecarregar a API
@st.cache_data(ttl=300)
def fetch_from_api(endpoint):
    """Função genérica para buscar dados da API"""
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and "data" in data:
                return pd.DataFrame(data["data"])
            else:
                return pd.DataFrame([data])
        else:
            st.error(f"Erro {response.status_code}: {response.text[:200]}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro de conexão: {str(e)}")
        return pd.DataFrame()

# Título da aplicação
st.set_page_config(page_title="Rugby Analytics", layout="wide")
st.title("🏉 Rugby Analytics - Catapult Data")
st.markdown("Análise de desempenho de atletas")

# Sidebar com navegação
st.sidebar.title("📁 Navegação")
pagina = st.sidebar.radio(
    "Escolha uma página:",
    ["📊 Dashboard Principal", "👥 Atletas", "📅 Atividades", "📈 Estatísticas"]
)

# ============ PÁGINA PRINCIPAL ============
if pagina == "📊 Dashboard Principal":
    st.header("📊 Dashboard Principal")
    
    # Cards com resumo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.spinner("Contando atletas..."):
            df_athletes = fetch_from_api("athletes")
            st.metric("👥 Total de Atletas", len(df_athletes))
    
    with col2:
        with st.spinner("Contando atividades..."):
            df_activities = fetch_from_api("activities")
            st.metric("📅 Total de Atividades", len(df_activities))
    
    with col3:
        st.metric("📊 Status", "Conectado", delta="API OK")
    
    st.divider()
    
    # Últimas atividades
    st.subheader("📋 Últimas Atividades")
    if not df_activities.empty:
        # Pega as 5 atividades mais recentes
        if "date" in df_activities.columns:
            df_activities["date"] = pd.to_datetime(df_activities["date"])
            df_recent = df_activities.sort_values("date", ascending=False).head(5)
        else:
            df_recent = df_activities.head(5)
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("Nenhuma atividade encontrada.")
    
    # Filtros rápidos
    st.subheader("🔍 Filtros Rápidos")
    
    col1, col2 = st.columns(2)
    with col1:
        if not df_athletes.empty and "name" in df_athletes.columns:
            atleta_filter = st.multiselect(
                "Filtrar por Atleta:",
                options=df_athletes["name"].tolist()
            )
    
    with col2:
        if not df_activities.empty and "name" in df_activities.columns:
            atividade_filter = st.multiselect(
                "Filtrar por Atividade:",
                options=df_activities["name"].tolist()
            )
    
    if st.button("🔄 Atualizar Dados"):
        st.cache_data.clear()
        st.rerun()

# ============ PÁGINA DE ATLETAS ============
elif pagina == "👥 Atletas":
    st.header("👥 Atletas")
    
    with st.spinner("Carregando lista de atletas..."):
        df_athletes = fetch_from_api("athletes")
    
    if not df_athletes.empty:
        # Mostra estatísticas rápidas
        st.subheader("📊 Resumo")
        st.write(f"**Total:** {len(df_athletes)} atletas")
        
        # Tabela completa
        st.subheader("📋 Lista Completa")
        st.dataframe(df_athletes, use_container_width=True)
        
        # Botão para download
        csv = df_athletes.to_csv(index=False)
        st.download_button(
            "📥 Baixar CSV",
            csv,
            "atletas.csv",
            "text/csv"
        )
    else:
        st.warning("Não foi possível carregar os atletas. Verifique seu token e permissões.")

# ============ PÁGINA DE ATIVIDADES ============
elif pagina == "📅 Atividades":
    st.header("📅 Atividades")
    
    with st.spinner("Carregando lista de atividades..."):
        df_activities = fetch_from_api("activities")
    
    if not df_activities.empty:
        st.subheader("📊 Resumo")
        st.write(f"**Total:** {len(df_activities)} atividades")
        
        # Seletor de atividade para detalhes
        if "name" in df_activities.columns:
            selected_activity = st.selectbox(
                "Selecione uma atividade para ver detalhes:",
                options=df_activities["name"].tolist()
            )
            
            if selected_activity:
                # Encontra o ID da atividade selecionada
                if "id" in df_activities.columns:
                    activity_id = df_activities[df_activities["name"] == selected_activity]["id"].iloc[0]
                    with st.spinner("Carregando detalhes..."):
                        df_details = fetch_from_api(f"activities/{activity_id}")
                        if not df_details.empty:
                            st.subheader(f"📌 Detalhes: {selected_activity}")
                            st.dataframe(df_details, use_container_width=True)
        
        st.divider()
        st.subheader("📋 Todas as Atividades")
        st.dataframe(df_activities, use_container_width=True)
        
        csv = df_activities.to_csv(index=False)
        st.download_button("📥 Baixar CSV", csv, "atividades.csv", "text/csv")
    else:
        st.warning("Não foi possível carregar as atividades.")

# ============ PÁGINA DE ESTATÍSTICAS ============
elif pagina == "📈 Estatísticas":
    st.header("📈 Estatísticas de Desempenho")
    st.info("💡 Os dados estatísticos podem requerer permissões específicas na sua conta Catapult.")
    
    # Carrega dados necessários
    with st.spinner("Carregando dados..."):
        df_athletes = fetch_from_api("athletes")
        df_activities = fetch_from_api("activities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not df_athletes.empty and "name" in df_athletes.columns:
            atleta = st.selectbox("👤 Selecione o Atleta:", df_athletes["name"].tolist())
    
    with col2:
        if not df_activities.empty and "name" in df_activities.columns:
            atividade = st.selectbox("📅 Selecione a Atividade:", df_activities["name"].tolist())
    
    if st.button("📊 Buscar Estatísticas", type="primary"):
        if atleta and atividade:
            # Encontra os IDs
            athlete_id = df_athletes[df_athletes["name"] == atleta]["id"].iloc[0]
            activity_id = df_activities[df_activities["name"] == atividade]["id"].iloc[0]
            
            with st.spinner("Buscando dados de desempenho..."):
                # Tenta buscar dados específicos do atleta na atividade
                # Nota: O endpoint exato pode variar conforme sua versão da API
                endpoints_to_try = [
                    f"activities/{activity_id}/athletes/{athlete_id}/stats",
                    f"activities/{activity_id}/stats?athlete_id={athlete_id}",
                    f"athletes/{athlete_id}/activities/{activity_id}"
                ]
                
                df_stats = pd.DataFrame()
                for endpoint in endpoints_to_try:
                    df_stats = fetch_from_api(endpoint)
                    if not df_stats.empty:
                        break
                
                if not df_stats.empty:
                    st.success("✅ Dados encontrados!")
                    
                    # Tenta encontrar métricas específicas
                    numeric_cols = df_stats.select_dtypes(include=['number']).columns.tolist()
                    
                    if numeric_cols:
                        col1, col2, col3 = st.columns(3)
                        for i, col in enumerate(numeric_cols[:3]):
                            with [col1, col2, col3][i % 3]:
                                st.metric(f"📊 {col.replace('_', ' ').title()}", 
                                         f"{df_stats[col].iloc[0]:.1f}")
                    
                    st.subheader("📋 Dados Completos")
                    st.dataframe(df_stats.T, use_container_width=True)
                else:
                    st.warning("""
                        Não foi possível encontrar estatísticas para esta combinação.
                        
                        **Possíveis causas:**
                        1. A atividade pode não ter dados de sensor processados
                        2. Seu token pode não ter permissão para acessar estatísticas
                        3. O atleta pode não ter participado desta atividade específica
                        
                        **Teste:** Verifique se consegue ver os dados diretamente no OpenField Cloud.
                    """)
        else:
            st.warning("Selecione um atleta e uma atividade primeiro.")

# Rodapé
st.sidebar.divider()
st.sidebar.caption("🔐 Dados da API Catapult OpenField")
st.sidebar.caption(f"🌎 Região: US")