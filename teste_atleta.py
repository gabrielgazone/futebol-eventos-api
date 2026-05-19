# diagnostico_atletas.py
import streamlit as st
import requests
import json

st.set_page_config(page_title="Diagnóstico Atletas", layout="wide")

SERVERS = {
    "Américas (US)": "https://connect-us.catapultsports.com/api/v6",
}

st.title("🔍 Diagnóstico - Busca de Atletas")

with st.sidebar:
    token = st.text_area("Token JWT:", height=80)
    activity_id = st.text_input("Activity ID (RESENDE X CABOFRIENSE):", 
                                value="f0c589fe-36cc-480a-9ff3-2d2eaa23ea3c")

if not token or not activity_id:
    st.info("👈 Cole o token e o Activity ID")
    st.stop()

headers = {'Authorization': f'Bearer {token}'}
base_url = SERVERS["Américas (US)"]

# 1. Buscar períodos da atividade
st.header("1. Buscando períodos da atividade")

response_periods = requests.get(f"{base_url}/activities/{activity_id}/periods", headers=headers)

if response_periods.status_code == 200:
    periods = response_periods.json()
    st.success(f"✅ Períodos encontrados: {len(periods)}")
    for p in periods:
        st.write(f"  - ID: {p.get('id')}, Nome: {p.get('name')}")
else:
    st.error(f"Erro: {response_periods.status_code}")
    st.code(response_periods.text)

# 2. Testar /activities/{id}/athletes
st.header("2. Testando GET /activities/{id}/athletes")

url_activity = f"{base_url}/activities/{activity_id}/athletes"
response_act = requests.get(url_activity, headers=headers)

st.write(f"Status: {response_act.status_code}")
if response_act.status_code == 200:
    data = response_act.json()
    if isinstance(data, list):
        st.write(f"Total de atletas: {len(data)}")
        if len(data) > 0:
            st.json(data[0])
    elif isinstance(data, dict):
        items = data.get('data', data.get('items', []))
        st.write(f"Total de atletas: {len(items)}")
        if items:
            st.json(items[0])
else:
    st.code(response_act.text)

# 3. Testar /periods/{id}/athletes para cada período
if periods:
    st.header("3. Testando GET /periods/{id}/athletes")
    
    for period in periods:
        period_id = period.get('id')
        period_name = period.get('name')
        
        st.subheader(f"Período: {period_name} (ID: {period_id})")
        
        url_period = f"{base_url}/periods/{period_id}/athletes"
        response_per = requests.get(url_period, headers=headers)
        
        st.write(f"Status: {response_per.status_code}")
        
        if response_per.status_code == 200:
            data = response_per.json()
            if isinstance(data, list):
                st.write(f"✅ Atletas encontrados: {len(data)}")
                if len(data) > 0:
                    st.write("Exemplo do primeiro atleta:")
                    st.json(data[0])
            elif isinstance(data, dict):
                items = data.get('data', data.get('items', []))
                st.write(f"✅ Atletas encontrados: {len(items)}")
                if items:
                    st.write("Exemplo do primeiro atleta:")
                    st.json(items[0])
            else:
                st.write(f"Tipo de resposta: {type(data)}")
        else:
            st.error(f"Erro: {response_per.status_code}")
            st.code(response_per.text)
    
    # 4. Sugestão: tentar com include=device_info
    st.header("4. Tentando com include=device_info")
    
    url_with_include = f"{base_url}/activities/{activity_id}/athletes?include=device_info"
    response_inc = requests.get(url_with_include, headers=headers)
    
    st.write(f"Status: {response_inc.status_code}")
    if response_inc.status_code == 200:
        data = response_inc.json()
        if isinstance(data, list):
            st.write(f"Atletas: {len(data)}")
        elif isinstance(data, dict):
            items = data.get('data', data.get('items', []))
            st.write(f"Atletas: {len(items)}")