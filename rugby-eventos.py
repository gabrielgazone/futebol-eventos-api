# 1. Primeiro, importamos as ferramentas que vamos usar
import pandas as pd
from CatapultPy import ofCreateToken, ofGetAthletes, ofGetActivities, ofGetStats

# 2. O SEU TOKEN que você tem (copie ele inteiro aqui)
MEU_TOKEN_STRING = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6ImU3NzY0MDAzODU1YjlmZWNlOGMxYzIyY2U0YWIxNjlkIn0.eyJhdWQiOiI0NjFiMTExMS02ZjdhLTRkYmItOWQyOS0yMzAzOWZlMjI4OGUiLCJqdGkiOiI3Nzk0YzBiMTdjYjMyNGFhOGE0ZWU0YWI4ODg1ZGE0OWQwZjdkMjg0MjEyYmRjNzY5MzhiYjIyOTY3Y2ZlZGM1ZDZmNDhmODVhOGFlOGNiNSIsImlhdCI6MTc3NzQ2NTczNS45NDE0OTIsIm5iZiI6MTc3NzQ2NTczNS45NDE0OTMsImV4cCI6NDkzMTA2NTczNS45MzI5MjMsInN1YiI6ImJkODAyMzAxLTk1YzgtNDgxNy05ZTAxLWFjOTI5Y2ZlZGMwNiIsInNjb3BlcyI6WyJjb25uZWN0IiwiY2F0YXB1bHRyIiwic2Vuc29yLXJlYWQtb25seSIsImF0aGxldGVzLXVwZGF0ZSIsImFjdGl2aXRpZXMtdXBkYXRlIiwiYW5ub3RhdGlvbnMtdXBkYXRlIiwicGFyYW1ldGVycy11cGRhdGUiXSwiaXNzIjoiaHR0cHM6Ly9iYWNrZW5kLXVzLm9wZW5maWVsZC5jYXRhcHVsc3BvcnRzLmNvbSIsImNvbS5jYXRhcHVsc3BvcnRzIjp7Im9wZW5maWVsZCI6eyJjdXN0b21lcnMiOlt7InJlbGF0aW9uIjoiYXV0aCIsImlkIjoxMzc3fV19fX0.p8Io8nPlCH3Ih-ckTYc_VNN88eWX4FIOjCJvUIC--6wQIfa_Ga_FsTS3HQ-nL_3TMg3KaV_4FChjmG0a1Gxmsj7J2b026Dl9yeHEg7TrbGcIVxKXSy2fdGs4e0VgSGG1BoVYlfQBkaDUxm6gFtmsevmE5xXWuTFlJkHFA1mHQofMWGtbnKJ-AmI2tAmNKADsNDxvdN-io4aVVaB4QVbuCUPf19zMPX0n6qlo1W8HWMoh5w3qjT6r9easbHskUvfPIN5cURZ3vdfq8WQ8hZ5TSMCI1asgrAGTjHC6ZMKv6p-GhFslcoZzYtuafsb8n0clc4_uRswoESnzbJtxDUnt_gtMP7CcLIboVsWJDnmDIvi2nx_OODLYntT9Yi_sb-NJHEctoe5iPc59-GT4Z7rvlObzHElUBRE8ad-HzimW6AwYPcXC9w_bUYcY91z_KNDguIcQiHFuYsh9mzmAAiK7bbQ0NEwVpXRJF83gYS3qM78ZAfhbjq33txg9F8SIA6SIZDWEA7K1bySZEwBILtKbFY9liJbkx5EAlbmAmU4yMXF5xZI1fZNWDSvQQOAgOG8P0hlOiF1V8QwrCFZ09Z0ewDzfOhecSMd8zNLb3iQY6ilNSkxEj_Au-lYKsgHVnneAiuzXKMgT1xAp8ynqu_jSP8ZMhEW0-dadLDM_zbbphM4"

# 3. Agora, "apresentamos" nosso token para a Catapult.
# O 'region' precisa ser o mesmo do seu token. O seu veio do "backend-us", então é 'us'.
print("🔄 Conectando à Catapult...")
meu_token_oficial = ofCreateToken(MEU_TOKEN_STRING, region="us")
print("✅ Token criado com sucesso!")

# 4. Vamos puxar os dados que você quer!

# --- Dados dos Atletas (nomes) ---
print("\n📊 Buscando lista de atletas...")
df_atletas = ofGetAthletes(meu_token_oficial)
print(f"✅ Encontrados {len(df_atletas)} atletas.")
# Exibe as primeiras linhas da tabela para você ver
print(df_atletas.head())

# --- Dados das Atividades (nomes e informações básicas) ---
print("\n📊 Buscando lista de atividades...")
df_atividades = ofGetActivities(meu_token_oficial)
print(f"✅ Encontradas {len(df_atividades)} atividades.")
print(df_atividades.head())

# --- Dados Estatísticos (A PARTE MAIS LEGAL!) ---
# É aqui que vamos pegar a distância total, velocidade máxima, etc.
print("\n📊 Buscando as estatísticas detalhadas...")
try:
    # Vamos pedir as principais métricas para todos os atletas e atividades disponíveis
    df_stats = ofGetStats(
        meu_token_oficial,
        params=[
            "athlete_id", "athlete_name",      # Identificação do atleta
            "activity_id", "activity_name",    # Identificação da atividade
            "date",                             # Data da atividade
            "total_distance",                   # 🎯 Distância total (o que você quer)
            "max_vel",                          # 🎯 Velocidade máxima (o que você quer)
            "total_player_load",                # Carga do jogador (métrica importante)
            "hsr_efforts"                       # Esforços de alta velocidade
        ],
        # group_by=["athlete", "activity"]     # Se quiser dados por atleta+atividade, descomente
    )
    print(f"✅ Estatísticas capturadas! Tabela com {len(df_stats)} linhas.")
    print("🎉 VEJA SEUS DADOS! 🎉")
    print(df_stats.head(10)) # Mostra as primeiras 10 linhas
    
    # Se quiser salvar tudo em um arquivo Excel para analisar depois:
    # df_stats.to_excel("meus_dados_catapult.xlsx", index=False)
    # print("💾 Dados salvos no arquivo 'meus_dados_catapult.xlsx'")

except Exception as e:
    print(f"⚠️ Aviso: Não foi possível buscar estatísticas. Erro: {e}")
    print("Isso pode acontecer se sua conta não tiver acesso a essa função ou se não houver dados ainda.")
    print("Mas você já tem a lista de atletas e atividades, que é um ótimo começo!")

print("\n✨ Pronto! Agora os dados estão na variável 'df_stats' e você pode analisá-los.")