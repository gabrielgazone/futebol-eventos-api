# Teste com lat/long em vez de x,y
params = {
    "parameters": "ts,lat,long,v,a,hr,pl",
    "nulls": "1"
}

response = requests.get(
    f"{base_url}/activities/{activity_id}/athletes/{athlete_id}/sensor",
    headers=headers,
    params=params
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print("✅ Lat/Long disponíveis!")