# descobrir_ids_simples.py
import streamlit as st
import requests
import json
import base64
from datetime import datetime

st.set_page_config(page_title="Descobrir IDs", layout="wide")

SERVERS = {
    "Américas (US)": "https://connect-us.catapultsports.com/api/v6",
    "Europa/África (EU)": "https://connect-eu.catapultsports.com/api/v6",
    "Ásia-Pacífico (AU)": "https://connect-au.catapultsports.com/api/v6",
}

st.title("🔍 Descobrir IDs - Atividade e Atleta")

with st.sidebar:
    server = st.selectbox("Servidor:", list(SERVERS.keys()))
    base_url = SERVERS[server]
    token = st.text_area("Token JWT:", height=80)

if not token:
    st.info("👈 Cole seu token na sidebar")
    st.stop()

headers = {'Authorization': f'Bearer {token}'}

# Buscar atividades
st.subheader("📋 Atividades disponíveis")

response = requests.get(f"{base_url}/activities?page_size=50", headers=headers)

if response.status_code != 200:
    st.error(f"Erro: {response.status_code}")
    st.stop()

activities = response.json()
if isinstance(activities, dict):
    activities = activities.get('data', activities.get('items', []))

# Mostrar atividades
activity_names = []
activity_ids = {}
for act in activities:
    name = act.get('name', 'Sem nome')
    act_id = act.get('id')
    activity_names.append(name)
    activity_ids[name] = act_id
    st.write(f"- {name}")

# Selecionar atividade
selected_name = st.selectbox("Selecione a atividade:", activity_names)
activity_id = activity_ids[selected_name]

st.success(f"**Activity ID:** `{activity_id}`")

# Buscar atletas da atividade
st.subheader("📋 Atletas da atividade")

resp_ath = requests.get(f"{base_url}/activities/{activity_id}/athletes", headers=headers)

if resp_ath.status_code == 200:
    athletes = resp_ath.json()
    if isinstance(athletes, dict):
        athletes = athletes.get('data', athletes.get('items', []))
else:
    st.warning("Buscando todos os atletas...")
    resp_ath = requests.get(f"{base_url}/athletes", headers=headers)
    athletes = resp_ath.json()
    if isinstance(athletes, dict):
        athletes = athletes.get('data', athletes.get('items', []))

# Mostrar atletas
athlete_names = []
athlete_ids = {}
for ath in athletes:
    first = ath.get('first_name', '')
    last = ath.get('last_name', '')
    name = f"{first} {last}".strip()
    if not name:
        name = ath.get('name', 'Sem nome')
    athlete_names.append(name)
    athlete_ids[name] = ath.get('id')
    st.write(f"- {name}")

# Selecionar atleta
selected_athlete = st.selectbox("Selecione o atleta:", athlete_names)
athlete_id = athlete_ids[selected_athlete]

st.success(f"**Athlete ID:** `{athlete_id}`")

# Testar lat/long
st.subheader("🔍 Testar Latitude/Longitude")

if st.button("Testar"):
    params = {"parameters": "lat,long", "nulls": "1"}
    
    test_url = f"{base_url}/activities/{activity_id}/athletes/{athlete_id}/sensor"
    
    r = requests.get(test_url, headers=headers, params=params, timeout=30)
    
    st.write(f"**Status:** {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        st.success("✅ Requisição funcionou!")
        
        # Extrair pontos
        pontos = []
        if isinstance(data, list) and len(data) > 0:
            if 'data' in data[0]:
                pontos = data[0]['data']
        
        if pontos:
            st.write(f"**Total de pontos:** {len(pontos)}")
            st.write("**Campos no primeiro ponto:**")
            st.write(list(pontos[0].keys()))
            
            if 'lat' in pontos[0]:
                st.success("✅ LATITUDE disponível!")
            else:
                st.error("❌ LATITUDE NÃO disponível")
            
            if 'long' in pontos[0]:
                st.success("✅ LONGITUDE disponível!")
            else:
                st.error("❌ LONGITUDE NÃO disponível")
    else:
        st.error(f"Erro: {r.text[:200]}")