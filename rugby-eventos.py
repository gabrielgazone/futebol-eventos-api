import streamlit as st
import pandas as pd
import requests

# ✅ Agora o token vem dos Secrets (NÃO escreva o token aqui!)
TOKEN = st.secrets["CATAPULT_TOKEN"]

BASE_URL = "https://backend-us.openfield.catapultsports.com/api/v1"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

@st.cache_data(ttl=300)
def fetch_from_api(endpoint):
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

st.set_page_config(page_title="Rugby Analytics", layout="wide")
st.title("🏉 Rugby Analytics - Catapult Data")

st.sidebar.title("📁 Navegação")
pagina = st.sidebar.radio(
    "Escolha uma página:",
    ["📊 Dashboard Principal", "👥 Atletas", "📅 Atividades", "📈 Estatísticas"]
)

# Página Dashboard
if pagina == "📊 Dashboard Principal":
    st.header("📊 Dashboard Principal")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.spinner("Carregando..."):
            df_athletes = fetch_from_api("athletes")
            st.metric("👥 Total de Atletas", len(df_athletes))
    
    with col2:
        with st.spinner("Carregando..."):
            df_activities = fetch_from_api("activities")
            st.metric("📅 Total de Atividades", len(df_activities))
    
    with col3:
        st.metric("📊 Status", "Conectado")
    
    if not df_activities.empty:
        st.subheader("📋 Últimas Atividades")
        st.dataframe(df_activities.head(10), use_container_width=True)

# Página Atletas
elif pagina == "👥 Atletas":
    st.header("👥 Atletas")
    with st.spinner("Carregando..."):
        df_athletes = fetch_from_api("athletes")
    if not df_athletes.empty:
        st.dataframe(df_athletes, use_container_width=True)
        csv = df_athletes.to_csv(index=False)
        st.download_button("📥 Baixar CSV", csv, "atletas.csv")

# Página Atividades
elif pagina == "📅 Atividades":
    st.header("📅 Atividades")
    with st.spinner("Carregando..."):
        df_activities = fetch_from_api("activities")
    if not df_activities.empty:
        st.dataframe(df_activities, use_container_width=True)

# Página Estatísticas
elif pagina == "📈 Estatísticas":
    st.header("📈 Estatísticas")
    st.info("Em desenvolvimento...")