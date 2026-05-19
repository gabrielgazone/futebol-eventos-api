# testar_latlong.py
import requests
import json

# ========== COLOQUE SEUS DADOS AQUI ==========
base_url = "https://connect-us.catapultsports.com/api/v6"  # ou seu servidor
token = "7gr-JWfaSE03rcxjCo6rLa8YbSovSRTCeiGwJrHnlcdd0kuSk"  # seu token COMPLETO
activity_id = "f0c589fe-1575-4423-9eb5-97f4806d323a"  # ID da RESENDE X CABOFRIENSE
athlete_id = "0472b649-afb9-4d37-b8f7-0dd1c0daa"  # ID do atleta (use o que você tem)
# =============================================

headers = {'Authorization': f'Bearer {token}'}

print("=" * 50)
print("TESTANDO DADOS DE LATITUDE E LONGITUDE")
print("=" * 50)

# Testar se lat/long estão disponíveis
print("\n1. Testando com parameters=lat,long...")
params = {"parameters": "lat,long", "nulls": "1"}

response = requests.get(
    f"{base_url}/activities/{activity_id}/athletes/{athlete_id}/sensor",
    headers=headers,
    params=params,
    timeout=30
)

print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    print("✅ REQUISIÇÃO FUNCIONOU!")
    
    # Extrair pontos
    pontos = []
    if isinstance(data, list) and len(data) > 0:
        if 'data' in data[0]:
            pontos = data[0]['data']
    
    if pontos:
        print(f"Total de pontos: {len(pontos)}")
        print("\nCampos disponíveis no primeiro ponto:")
        print(list(pontos[0].keys()))
        
        # Contar quantos pontos têm lat/long válidos
        lat_count = 0
        long_count = 0
        for p in pontos[:100]:  # verifica primeiros 100
            if p.get('lat') is not None and p.get('lat') != 0:
                lat_count += 1
            if p.get('long') is not None and p.get('long') != 0:
                long_count += 1
        
        print(f"\nPontos com lat válido (nos primeiros 100): {lat_count}")
        print(f"Pontos com long válido (nos primeiros 100): {long_count}")
        
        if lat_count > 0:
            print("\n✅ LATITUDE DISPONÍVEL!")
            print(f"Exemplo: {pontos[0].get('lat')}")
        else:
            print("\n❌ Latitude NÃO disponível ou todos são zero")
        
        if long_count > 0:
            print(f"✅ LONGITUDE DISPONÍVEL!")
            print(f"Exemplo: {pontos[0].get('long')}")
        else:
            print(f"❌ Longitude NÃO disponível ou todos são zero")
            
    else:
        print("❌ Nenhum ponto encontrado")
        
elif response.status_code == 401:
    print("❌ Token inválido - verifique se o token está correto")
elif response.status_code == 404:
    print("❌ Atividade ou atleta não encontrado")
else:
    print(f"❌ Erro: {response.status_code}")
    print(response.text[:200])