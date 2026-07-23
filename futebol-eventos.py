# futebol_eventos_completo_final.py
# PARTE 1 - IMPORTS, CONSTANTES E CLASSE API

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import base64
import io
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter as _gf
from scipy.spatial import cKDTree
import folium
from streamlit_folium import st_folium
import os as _os

st.set_page_config(page_title="Futebol Eventos - Catapult", layout="wide")

# Componente bidirecional para o mapa interativo de posicionamento de campo
_CAMPO_COMP_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_campo_component")
_campo_component = st.components.v1.declare_component("campo_interativo_v1", path=_CAMPO_COMP_DIR)

# ── P1/P3: motor único de métricas + validação de concordância ──────────────
# Fonte canônica de cálculo (funções puras, cobertas por tests/): todas as
# abas delegam para cá — o mesmo número em qualquer tela.
import metrics as _mtr          # noqa: E402
import validation as _valmod    # noqa: E402
import storage as _storage      # noqa: E402  (P2: persistência durável)
import applog as _applog        # noqa: E402  (P3: logging estruturado)
from catapult_api import _api_fetch, CatapultAPI  # noqa: E402,F401  (P4: cliente API)

# (Cloud) Após um deploy, o Streamlit Cloud reexecuta o script principal mas
# pode manter os módulos locais ANTIGOS em cache no sys.modules — foi a causa
# do AttributeError 'metrics has no attribute playerload_total'. Se a versão
# do esquema não bater, força o reload; o teste de sincronia no CI garante
# que estes números acompanham os SCHEMA_VERSION dos módulos.
_METRICS_SCHEMA_ESPERADO = 4
_VALIDATION_SCHEMA_ESPERADO = 2
import importlib as _importlib  # noqa: E402
if getattr(_mtr, 'SCHEMA_VERSION', 0) < _METRICS_SCHEMA_ESPERADO:
    _mtr = _importlib.reload(_mtr)
if getattr(_valmod, 'SCHEMA_VERSION', 0) < _VALIDATION_SCHEMA_ESPERADO:
    _valmod = _importlib.reload(_valmod)
if getattr(_mtr, 'SCHEMA_VERSION', 0) < _METRICS_SCHEMA_ESPERADO:
    st.error("⚠️ O módulo `metrics.py` no servidor está desatualizado mesmo "
             "após reload — reinicie o app (Manage app → Reboot).")


# (P4) diagnóstico/proveniência -> diagnostics.py
from diagnostics import _PROV_LABELS, _selo_fonte, _diag_log, _diag_reset  # noqa: E402,F401


# ── P9: bandas relativas à Vmáx individual (modo de análise opcional) ───────
# Faixas em % da velocidade máxima de CADA atleta — recomendado quando o
# elenco tem Vmáx heterogênea (limiares absolutos super/subestimam).
_REL_VEL_BANDAS = {
    'R1 — <40% Vmáx':          (0.0, 0.40),
    'R2 — 40–60% Vmáx':        (0.40, 0.60),
    'R3 — 60–70% Vmáx':        (0.60, 0.70),
    'R4 — 70–80% Vmáx':        (0.70, 0.80),
    'R5 — 80–90% Vmáx (HSR)':  (0.80, 0.90),
    'R6 — ≥90% Vmáx (Sprint)': (0.90, 9.99),
}


# (P4) campo/plotagem/eventos -> field.py
from field import (  # noqa: E402,F401
    cor_atleta,
    cor_atleta_pos,
    _get_pos_grupo,
    _ev_icone,
    _vmax_individual_kmh,
    extrair_dados_sensor,
    extrair_efforts_data,
    lat_lon_to_campo_coords,
    desenhar_campo_futebol,
    plotar_trajetoria_campo,
    plotar_heatmap_campo,
    plotar_heatmap_presenca_campo,
    extrair_eventos_futebol,
    enriquecer_eventos_com_posicao,
    adicionar_eventos_campo,
    criar_timeline_eventos,
    analisar_fadiga_eventos,
    desenhar_campo_futebol_bonito,
    adicionar_trajetoria_campo,
    _segmentos_continuos,
    adicionar_pontos_velocidade_bandas,
    adicionar_pontos_aceleracao_bandas,
    adicionar_setas_direcao,
    adicionar_convex_hull,
    adicionar_tercos_campo,
    adicionar_grade_quadrantes,
    stats_quadrante,
    gps_para_campo_coords,
    campo_para_latlon,
    criar_mapa_satelite_futebol,
    criar_html_campo_interativo,
    criar_html_campo_fixo,
    lat_lon_to_xy,
    gerar_heatmap_segmentado,
    enriquecer_esforcos_taticos,
    calcular_voronoi_campo,
    calcular_carga_neuromuscular,
    _neuro_cached,
    plotar_carga_neuromuscular,
    calcular_acwr_df,
    plotar_acwr,
)

# (P4) constantes -> config.py
from config import (  # noqa: E402
    _CHAVE_COMBINADO,
    _DEFAULT_MIN_DUR_S,
    _DEFAULT_MIN_DUR_VEL_S,
    _ATHLETE_PALETTE,
    SERVERS,
    LANGUAGES,
    BANDAS_VEL,
    BANDAS_ACC,
    _DEFAULT_VELOCITY_ZONES,
    _DEFAULT_ACCELERATION_ZONES,
    _ZONES_SCHEMA_VERSION,
    _NOMES_BANDA_VEL_DEFAULT,
    _CORES_BANDA_VEL_DEFAULT,
    _ACC_BAND_MAP,
    FUTEBOL_EVENTS_CONFIG,
)

# ==================== SISTEMA DE IDIOMAS ====================

# (P4) i18n (TRANSLATIONS + t) -> i18n.py
from i18n import TRANSLATIONS, t  # noqa: E402,F401
# ==================== REFERÊNCIAS BIBLIOGRÁFICAS ====================
REFERENCIAS = {
    "janelas": """
    **Referência:** Aughey, R.J. (2011). "Applications of GPS technologies to field sports". 
    *International Journal of Sports Physiology and Performance*, 6(3), 295-310.
    """,
    "campo": """
    **Referência:** FIFA/IFAB (2023). "Regra 1: O Campo de Jogo". Regras do Jogo de Futebol.
    Campo oficial FIFA: 105m de comprimento × 68m de largura (partidas internacionais).
    Área de penalidade: 40,32m × 16,5m | Área pequena: 18,32m × 5,5m | Círculo central: r = 9,15m.
    """
}

# ==================== SISTEMA GLOBAL DE CORES POR ATLETA ====================


# ==================== CORES POR GRUPO DE POSIÇÃO ====================
import colorsys as _colorsys

# Hue base para cada grupo tático (HSL)

# Cor representativa de cada grupo (para legenda)
_POSICAO_COR_LEGENDA = {
    'Goleiro':     '#F4D03F',
    'Defensor':    '#2196F3',
    'Meio-campo':  '#4CAF50',
    'Ala/Extremo': '#FF9800',
    'Atacante':    '#F44336',
    'Outro':       '#9C27B0',
}



# Ícone emoji por tipo de evento de futebol (para timeline e overlay)


# (P4) _api_fetch movido para catapult_api.py


# ==================== PREFERÊNCIAS DO USUÁRIO ====================
# (P4) prefs -> persistence; parsers de zona -> bands
from persistence import _carregar_prefs, _salvar_prefs  # noqa: E402,F401
from bands import _parse_api_velocity_zones, _parse_api_acceleration_zones, _resp_tem_zonas, _zonas_conta_via_api  # noqa: E402,F401




# (P4) persistência (store/venues/bandas do usuário) -> persistence.py
from persistence import (  # noqa: E402,F401
    _get_store, _org_key, _carregar_venues, _salvar_venue, _excluir_venue,
    _salvar_bandas_usuario, _carregar_bandas_usuario, _excluir_bandas_usuario,
)


# (P4) classe CatapultAPI movida para catapult_api.py
# (P4) _DEFAULT_MIN_DUR_S / _DEFAULT_MIN_DUR_VEL_S -> config.py
_SENSOR_HZ = 10  # frequência de amostragem Catapult (10 Hz)


# (P4) compute de análise -> analysis.py
from analysis import (  # noqa: E402,F401
    detectar_eventos_acc,
    acc_series_from_vel,
    detectar_acoes_acc_idx,
    get_min_dur_s,
    get_min_dur_vel_s,
    get_zones_for_athlete,
    calcular_metricas,
    calcular_janelas_discretas_10s,
    calcular_distancia_janelas_discretas_10s,
    calcular_distancia_janelas_por_vel_posicao,
    combinar_periodos_continuo_posicao,
    obter_limites_periodos_posicao,
    combinar_periodos_continuo,
    encontrar_eventos_nao_sobrepostos,
    processar_efforts_velocidade,
    processar_efforts_aceleracao,
    combinar_periodos,
    _segmentos_de_mask,
    calcular_efforts_velocidade_sensor,
    calcular_efforts_aceleracao_sensor,
    criar_grafico_velocidade_tempo,
    criar_grafico_aceleracao_tempo,
    classificar_intensidade,
    criar_grafico_intensidade,
    criar_tabela_intensidade,
    exibir_resultados_janela,
)











# ==================== CONVERSÃO DE COORDENADAS GPS → CAMPO ====================








# ══════════════════════════════════════════════════════════════════════════════
# BANDAS DE VELOCIDADE E ACELERAÇÃO (referências Catapult Football)
# ══════════════════════════════════════════════════════════════════════════════
# Valores padrão = "Bandas Globais" da conta Catapult OpenField (km/h).
# IMPORTANTE: a API Connect v6 NÃO expõe os limites das bandas de velocidade
# (confirmado na documentação oficial — não há endpoint /velocity_zones em v6,
# e /teams/{id} só traz dwell_time e rhie_bands, não os cortes em km/h).
# Por isso estes valores espelham exatamente a tela "Bandas Globais" e podem
# ser ajustados pelo usuário na barra lateral.
# Espelha as "Bandas Globais → Gen2Acceleration" da conta Catapult (m/s²).
# A API Connect v6 também NÃO expõe estes cortes — são configurados manualmente
# na barra lateral, mesmo raciocínio das bandas de velocidade.
# Estrutura espelhando a tela "Bandas Globais → Gen2Acceleration" da Catapult,
# dividida em ACELERAÇÃO (caixas 6,7,8 da nuvem) e DESACELERAÇÃO (caixas 3,2,1).
# As caixas 4 e 5 (-2 a 2 m/s² · zona leve/neutra) NÃO são exibidas.
#   Aceleração   B1 = caixa 6 (2 a 3)   · B2 = caixa 7 (3 a 4)   · B3 = caixa 8 (4 a 10)
#   Desaceleração B1 = caixa 3 (-3 a -2) · B2 = caixa 2 (-4 a -3) · B3 = caixa 1 (-10 a -4)

# ══════════════════════════════════════════════════════════════════════════════
# BANDAS DE VELOCIDADE — helpers de zonas individuais / da conta
# ══════════════════════════════════════════════════════════════════════════════

# Espelha as "Bandas Globais" da conta Catapult (km/h convertido p/ m/s).






# ── Defaults de aceleração ────────────────────────────────────────────────────
# Espelha "Bandas Globais → Gen2Acceleration" (m/s²), dividido em ACELERAÇÃO e
# DESACELERAÇÃO. Apenas as 6 bandas relevantes da nuvem (caixas 6,7,8 e 3,2,1);
# a zona leve/neutra (-2 a 2 · caixas 4 e 5) é ignorada.

# Versão da estrutura das bandas. Ao mudar a forma das bandas padrão (ex.: de 8
# para 6 bandas de aceleração), incrementar este valor força a reinicialização
# das zonas em session_state, descartando valores antigos em cache de sessões
# que ficaram abertas antes da atualização.








# (P4) helpers de banda -> bands.py
from bands import (  # noqa: E402,F401
    _ACC_KEY_TO_NUM,
    _bandas_vel_ativas,
    _legenda_vel_js,
    _legenda_vel_items,
    _fmt_num_banda,
    _rotulo_banda_vel,
    _rotulo_banda_acc,
    _bandas_acc_ativas,
)


# (Removido) Derivação dos cortes de banda a partir dos efforts. O app usa
# limiares fixos documentados; zonas via API só por leitura de configuração.



# ── Configuração dos tipos de eventos futebol ──────────────────────────────







































# PARTE 3 - FUNÇÕES DE HRV, JANELAS E ESFORÇOS












# ════════════════════════════════════════════════════════════════════════
#  TÁTICA COLETIVA
#  Visões que usam a posição de VÁRIOS atletas no MESMO instante — o time
#  como sistema, não como soma de indivíduos. Tudo reaproveita os xs/ys de
#  campo (mesmo referencial, em metros) já presentes em
#  dados_posicao_por_periodo. 4 visões: Pitch Control (Spearman),
#  Respiração da equipe (centroide + convex hull), Voronoi e Replay 3D.
# ════════════════════════════════════════════════════════════════════════

# Paleta determinística para identificar atletas entre as visões.


# (P4) Tática Coletiva -> viz/tatica_coletiva.py
from viz.tatica_coletiva import render_tatica_coletiva  # noqa: E402


# (P4) Exportação para Artigo -> viz/export_artigo.py
from viz.export_artigo import render_export_artigo  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# MONITORAMENTO LONGITUDINAL (P10) — ACWR, monotonia e strain via POST /stats
# ══════════════════════════════════════════════════════════════════════════
# (P4) render_monitoramento -> viz/monitoramento.py
from viz.monitoramento import render_monitoramento  # noqa: E402






# ── Métricas que devem ser SOMADAS ao combinar períodos ──────────────────────
# ── Métricas que devem manter o MÁXIMO registrado ────────────────────────────










# PARTE 4 - FUNÇÕES DE GRÁFICOS, INTENSIDADE E CLASSIFICAÇÃO



# ── Limiares absolutos por métrica (baseados na literatura) ───────────────────
# Distância: Aughey (2011) IJSPP · PlayerLoad: Casamichana et al. (2013)
# Velocidade: Bangsbo (1994) · Aceleração: Osgnach et al. (2010)
_LIMIARES_JANELA = {
    'Distância':  dict(alta=120.0, media=85.0,  ref='Aughey, 2011'),
    'PlayerLoad': dict(alta=8.0,   media=5.0,   ref='Casamichana et al., 2013'),
    'Velocidade': dict(alta=19.0,  media=14.0,  ref='Bangsbo, 1994'),
    'Aceleração': dict(alta=3.0,   media=2.0,   ref='Osgnach et al., 2010'),
}






# PARTE 5 - FUNÇÃO MAIN COMPLETA



# ==================== FEATURE 1: HEATMAP TEMPORAL SEGMENTADO ====================



# ==================== FEATURE 2: DIAGRAMA DE VORONOI ====================





# ==================== FEATURE 6: CARGA NEUROMUSCULAR ====================







# ==================== FEATURE 8: RELATÓRIO PDF ====================



# ==================== FEATURE 10: ACWR ====================





# (P4) helpers de design + CSS global -> ui_theme.py
from ui_theme import _hr, _badge, inject_global_css  # noqa: E402


def main():
    # ═══════════════════════════════════════════════════════════════════
    # DESIGN SYSTEM — CSS global injetado uma vez por sessão
    # ═══════════════════════════════════════════════════════════════════
    inject_global_css()

    # ── Modo Apresentação — CSS dinâmico ──────────────────────────────────
    if st.session_state.get('modo_apresentacao'):
        st.markdown("""<style>
/* ── Ocultar ruído técnico ── */
[data-testid="stCaptionContainer"],
.stCaption { display: none !important; }
[data-testid="stDownloadButton"],
[data-testid="stFileDownloader"] { display: none !important; }
.element-container:has([data-testid="stDownloadButton"]) { display: none !important; }

/* ── Ampliar métricas ── */
[data-testid="stMetricValue"] {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    opacity: 0.75 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.9rem !important; }
[data-testid="metric-container"] {
    padding: 18px 22px !important;
    border-color: rgba(74,222,128,0.18) !important;
    box-shadow: 0 0 18px rgba(74,222,128,0.06) !important;
}

/* ── Sidebar com acento verde em modo apresentação ── */
[data-testid="stSidebar"] {
    border-right: 2px solid rgba(74,222,128,0.40) !important;
    box-shadow: 4px 0 32px rgba(74,222,128,0.07) !important;
}
</style>""", unsafe_allow_html=True)

        # Banner flutuante no canto superior direito
        st.markdown("""
<div style="
    position: fixed; top: 58px; right: 22px; z-index: 9999;
    background: linear-gradient(135deg, rgba(15,40,25,0.96) 0%, rgba(10,30,18,0.96) 100%);
    border: 1px solid rgba(74,222,128,0.45);
    border-radius: 10px; padding: 7px 16px;
    font-size: 0.76rem; font-weight: 600; color: #4ade80;
    box-shadow: 0 4px 24px rgba(0,0,0,0.55), 0 0 12px rgba(74,222,128,0.12);
    backdrop-filter: blur(10px);
    display: flex; align-items: center; gap: 8px;
    letter-spacing: 0.3px;
">
  <span style="font-size:0.65rem;
               animation:_puls 1.4s ease-in-out infinite;
               display:inline-block">🟢</span>
  Modo Apresentação
  <style>
    @keyframes _puls {
      0%,100% { opacity:1; transform:scale(1); }
      50%      { opacity:.5; transform:scale(0.85); }
    }
  </style>
</div>""", unsafe_allow_html=True)

    # ─── Header branded ──────────────────────────────────────────────────
    st.markdown(f"""
<div style="
    background: linear-gradient(135deg,#0d1b2a 0%,#152235 55%,#0d1b2a 100%);
    border-radius:14px; padding:22px 28px; margin-bottom:24px;
    border:1px solid rgba(46,134,193,0.22);
    box-shadow:0 4px 28px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.04);
    display:flex; align-items:center; gap:18px;
">
  <div style="font-size:2.8rem;line-height:1;
              filter:drop-shadow(0 0 10px rgba(255,255,255,0.25))">⚽</div>
  <div style="flex:1;min-width:0">
    <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;
                color:white;letter-spacing:-0.4px;line-height:1.2">
      Futebol Eventos
      <span style="color:#5dade2;font-weight:400"> — Catapult Sports</span>
    </div>
    <div style="font-family:'Inter',sans-serif;font-size:0.8rem;
                color:rgba(255,255,255,0.4);margin-top:5px;font-weight:400;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
      {t('app_subtitle')}
    </div>
  </div>
  <div style="text-align:right;flex-shrink:0">
    <div style="font-size:0.63rem;color:rgba(255,255,255,0.22);
                font-weight:600;letter-spacing:1.8px;text-transform:uppercase">
      Catapult API v6
    </div>
    <div style="font-size:0.75rem;color:#2ecc71;margin-top:4px;font-weight:600">
      ● Online
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Inicializar session state
    if 'df_athletes' not in st.session_state:
        st.session_state.df_athletes = pd.DataFrame()
    if 'df_activities' not in st.session_state:
        st.session_state.df_activities = pd.DataFrame()
    if 'df_positions' not in st.session_state:
        st.session_state.df_positions = pd.DataFrame()
    if 'df_teams' not in st.session_state:
        st.session_state.df_teams = pd.DataFrame()
    if 'df_parameters' not in st.session_state:
        st.session_state.df_parameters = pd.DataFrame()
    if 'athlete_team_map' not in st.session_state:
        st.session_state.athlete_team_map = {}
    
    # ── Carregar preferências salvas ─────────────────────────────────
    _prefs = _carregar_prefs()

    # Sidebar
    with st.sidebar:

        # ── FEATURE 5: Tour guiado para novos usuários ───────────────
        if not _prefs.get('tour_done'):
            with st.expander("📖 Como usar — Guia Rápido", expanded=True):
                st.markdown("""
**Bem-vindo! Siga estes passos:**

**1️⃣ Servidor & Token**
Selecione o servidor da sua região e cole o token JWT do OpenField.

**2️⃣ Carregar Dados**
Clique em **🔄 Carregar Dados** — equipes, atletas e atividades serão buscados automaticamente.

**3️⃣ Filtrar**
Filtre por equipe e posição, selecione a atividade e os períodos desejados.

**4️⃣ Selecionar Atletas**
Escolha um ou mais atletas para análise simultânea.

**5️⃣ Explorar as ABAs**
- 📈 Gráficos Comparativos
- 🗺️ Campo de Futebol (posição, esforços, animação)
- ⏱️ Esforços ao Longo do Tempo
- 📊 Janelas Temporais Móveis
- 💪 Carga Neuromuscular
- 🏎️ Perfil Aceleração × Velocidade
""")
                if st.button("✅ Entendi, não mostrar novamente", key="_tour_ok"):
                    _prefs['tour_done'] = True
                    _salvar_prefs(_prefs)
                    st.rerun()

        # ── Modo Apresentação ─────────────────────────────────────────
        _pmode_on = st.session_state.get('modo_apresentacao', False)
        _pmode_label = "🖥️ Sair do Modo Apresentação" if _pmode_on else "🖥️ Modo Apresentação"
        if st.button(_pmode_label, use_container_width=True,
                     type="primary" if _pmode_on else "secondary",
                     help="Amplia métricas e oculta detalhes técnicos para exibição em telão / projetor"):
            st.session_state['modo_apresentacao'] = not _pmode_on
            st.rerun()
        st.divider()

        st.header("🌍 Servidor")
        # ── FEATURE 2: pré-seleciona servidor salvo nas prefs ────────
        _server_opts  = list(SERVERS.keys())
        _server_saved = _prefs.get('server', _server_opts[0])
        _server_idx   = _server_opts.index(_server_saved) if _server_saved in _server_opts else 0
        server = st.selectbox("Selecione:", _server_opts, index=_server_idx)
        base_url = SERVERS[server]
        # Salva imediatamente se mudou
        if server != _prefs.get('server'):
            _prefs['server'] = server
            _salvar_prefs(_prefs)

        st.header(t("language_header"))
        st.selectbox(t("language_select"), list(LANGUAGES.keys()), key="lang_selector")

        # (Rastreabilidade) carimbo do build em execução — confirme aqui se o
        # deploy mais recente já está ativo antes de exportar para validação.
        try:
            _build_ts = datetime.fromtimestamp(
                _os.path.getmtime(_os.path.abspath(__file__)))
            st.caption(f"🏗️ Build: **{_build_ts.strftime('%d/%m %H:%M')}** · "
                       f"metrics v{getattr(_mtr, 'SCHEMA_VERSION', '?')}")
        except Exception:
            _applog.log_debug_exc()

        # (P2) Aviso de persistência efêmera — força a inicialização do store.
        _get_store()
        if st.session_state.get('_persist_efemera'):
            st.caption("💾 Armazenamento **local (temporário)**: bandas e campos "
                       "podem se perder em um redeploy. Para torná-los duráveis, "
                       "configure um Supabase em `st.secrets['supabase']` "
                       "(url + key). Veja `SETUP_PERSISTENCIA.md`.")

        st.header("🔐 Token")
        # (P7) token mascarado (não fica visível na tela) + suporte a st.secrets
        _tok_secret = ''
        try:
            _tok_secret = str(st.secrets.get('CATAPULT_TOKEN', '') or '')
        except Exception:
            _tok_secret = ''
        token = st.text_input(
            "Token JWT:", type="password", value=_tok_secret,
            help="O token fica mascarado e não é gravado em disco. Em deploy, "
                 "configure CATAPULT_TOKEN em st.secrets para preenchê-lo "
                 "automaticamente.")
        if _tok_secret and token == _tok_secret:
            st.caption("🔒 Token carregado de `st.secrets`.")
        
        if token and st.button("🔄 Carregar Dados", type="primary"):
            with st.spinner("Carregando..."):
                _diag_reset()   # (P5) novo carregamento → diagnóstico limpo
                api = CatapultAPI(base_url, token)
                
                st.subheader("📋 Carregando Equipes...")
                teams_raw = api.get_teams()
                if teams_raw:
                    teams_data = []
                    for team in teams_raw:
                        teams_data.append({'id': team.get('id'), 'nome': team.get('name'), 'slug': team.get('slug')})
                    st.session_state.df_teams = pd.DataFrame(teams_data)
                    st.success(f"✅ {len(teams_data)} equipes carregadas")

                    # (P7) marcador da organização → escopo do banco de venues.
                    # Menor team_id = determinístico p/ qualquer token do clube.
                    try:
                        _org_ids = sorted(str(_t2.get('id') or '')
                                          for _t2 in teams_raw if _t2.get('id'))
                        if _org_ids:
                            st.session_state['_org_marker'] = ''.join(
                                _c for _c in _org_ids[0] if _c.isalnum())[:16]
                    except Exception:
                        _applog.log_debug_exc()
                    
                    st.subheader("📋 Mapeando atletas por equipe...")
                    athlete_team_map = {}
                    for _, team in st.session_state.df_teams.iterrows():
                        team_athletes = api.get_team_athletes(team['id'])
                        if team_athletes:
                            if isinstance(team_athletes, dict):
                                team_athletes = team_athletes.get('data', team_athletes.get('items', []))
                            for ath in team_athletes:
                                athlete_team_map[ath.get('id')] = team['nome']
                    st.session_state.athlete_team_map = athlete_team_map
                    st.success(f"✅ {len(athlete_team_map)} atletas mapeados")
                
                st.subheader("📋 Carregando Posições...")
                positions_raw = api.get_positions()
                if positions_raw:
                    positions_data = []
                    for p in positions_raw:
                        positions_data.append({'id': p.get('id'), 'nome': p.get('name'), 'slug': p.get('slug')})
                    st.session_state.df_positions = pd.DataFrame(positions_data)
                    st.success(f"✅ {len(positions_data)} posições carregadas")
                
                st.subheader("📋 Carregando Atletas...")
                athletes_raw = api.get_athletes()
                if athletes_raw:
                    atletas = []
                    for a in athletes_raw:
                        nome = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
                        if not nome:
                            nome = a.get('name', 'Sem nome')
                        position_name = ''
                        pos_id = a.get('position_id')
                        if pos_id and not st.session_state.df_positions.empty:
                            pos_row = st.session_state.df_positions[st.session_state.df_positions['id'] == pos_id]
                            if not pos_row.empty:
                                position_name = pos_row.iloc[0]['nome']
                        team_name = st.session_state.athlete_team_map.get(a.get('id'), '')
                        atletas.append({
                            'id': a.get('id'), 'nome': nome, 'camisa': a.get('jersey', ''),
                            'posicao': position_name, 'equipe': team_name,
                        })
                    st.session_state.df_athletes = pd.DataFrame(atletas)

                    # ── Auto-busca Vmax histórico via /stats (cached_stats) ──
                    # É o mesmo endpoint do botão fallback manual, mas chamado
                    # automaticamente logo após carregar a lista de atletas.
                    _hvm_auto = st.session_state.get('hist_vmax', {})
                    _src_auto = st.session_state.get('hist_vmax_source', {})
                    _n_vmax_auto = 0
                    # Índice de nomes dos atletas para matching robusto:
                    # { nome_normalizado: nome_real }
                    def _norm(s):
                        return s.lower().strip()
                    _atleta_nome_idx  = {_norm(a['nome']): a['nome'] for a in atletas}
                    # Também índice com tokens ordenados ("Lima Jorge" → "jorge lima")
                    _atleta_token_idx = {
                        ' '.join(sorted(_norm(a['nome']).split())): a['nome']
                        for a in atletas
                    }
                    try:
                        _stats_auto = api.get_stats({
                            "group_by": ["athlete"],
                            "parameters": ["max_velocity"],
                            "source": "cached_stats",
                        })
                        if _stats_auto:
                            _sa_list = (_stats_auto if isinstance(_stats_auto, list)
                                        else _stats_auto.get('data', []))
                            # Salva resposta bruta para debug
                            st.session_state['_debug_stats_raw'] = _sa_list[:3] if _sa_list else []
                            for _sa in (_sa_list or []):
                                # Extrai nome vindo da API — pode ser "Primeiro Ultimo"
                                # ou "Ultimo Primeiro" dependendo da conta
                                _sa_name_raw = str(
                                    _sa.get('athlete') or _sa.get('athlete_name') or
                                    _sa.get('name') or _sa.get('athlete_id') or ''
                                )
                                # max_velocity em parâmetros aninhados ou direto
                                _sa_params = _sa.get('parameters') or _sa
                                _sa_val = float(
                                    _sa_params.get('max_velocity') or
                                    _sa_params.get('max_speed') or
                                    _sa.get('max_velocity') or
                                    _sa.get('max_speed') or 0
                                )
                                if not _sa_name_raw or _sa_val <= 0:
                                    continue
                                # Matching: 1) exato, 2) tokens ordenados (cobre inversão nome)
                                _sa_ms = _sa_val / 3.6 if _sa_val > 15 else _sa_val
                                _match_nome = (
                                    _atleta_nome_idx.get(_norm(_sa_name_raw)) or
                                    _atleta_token_idx.get(' '.join(sorted(_norm(_sa_name_raw).split())))
                                )
                                if _match_nome and _src_auto.get(_match_nome, '') != 'manual':
                                    _hvm_auto[_match_nome] = _sa_ms
                                    _src_auto[_match_nome] = 'catapult_stats'
                                    _n_vmax_auto += 1
                    except Exception:
                        _applog.log_debug_exc()
                    st.session_state['hist_vmax']        = _hvm_auto
                    st.session_state['hist_vmax_source'] = _src_auto
                    _vmax_msg = f" · {_n_vmax_auto} Vmax hist. detectadas" if _n_vmax_auto else ""
                    st.success(f"✅ {len(atletas)} atletas carregados{_vmax_msg}")
                
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
                            'id':    a.get('id'),
                            'nome':  a.get('name'),
                            'data':  a.get('start_time'),
                            'venue': a.get('venue') or {},   # campo: lat, lng, rotation, length, width
                        })
                    st.session_state.df_activities = pd.DataFrame(atividades)
                    st.success(f"✅ {len(atividades)} atividades")
                
                st.session_state.api = api

                # ── Bandas de velocidade / aceleração ─────────────────────────
                # A API Connect v6 NÃO expõe os cortes das "Bandas Globais"
                # diretamente. Mas os EFFORTS da conta trazem nº da banda +
                # velocidade/aceleração reais, então os cortes são DERIVADOS
                # automaticamente ao carregar uma atividade (ver bloco mais
                # abaixo, após o loop de períodos). Aqui apenas:
                #  1) detectamos troca de TOKEN (conta) e limpamos as bandas
                #     para re-derivar de acordo com a nova conta;
                #  2) inicializamos com os defaults como fallback inicial.
                _tok_mark = str(hash(token)) if token else ''
                if (st.session_state.get('_token_marker') != _tok_mark
                        or st.session_state.get('_zones_schema') != _ZONES_SCHEMA_VERSION):
                    st.session_state['_token_marker'] = _tok_mark
                    st.session_state['_zones_schema'] = _ZONES_SCHEMA_VERSION
                    for _zk in ('velocity_zones_account', 'acceleration_zones_account',
                                'velocity_zones_manual', 'acceleration_zones_manual',
                                'velocity_zones_source', 'acceleration_zones_source',
                                'velocity_zones_from_api', 'acceleration_zones_from_api',
                                '_bandas_deriv_key'):
                        st.session_state.pop(_zk, None)

                if not st.session_state.get('velocity_zones_account'):
                    st.session_state['velocity_zones_account'] = _DEFAULT_VELOCITY_ZONES[:]
                    st.session_state['velocity_zones_source'] = 'default'
                if not st.session_state.get('acceleration_zones_account'):
                    st.session_state['acceleration_zones_account'] = _DEFAULT_ACCELERATION_ZONES[:]
                    st.session_state['acceleration_zones_source'] = 'default'

                # ── PRIMÁRIO: bandas configuradas na conta via API ────────────
                # Se a API expuser as zonas (nível conta ou equipe), elas têm
                # prioridade sobre a derivação por efforts. Caso contrário,
                # mantém-se a derivação (bloco após o loop de períodos).
                try:
                    _team_ids_z = ([t.get('id') for t in teams_raw if t.get('id')]
                                   if teams_raw else [])
                    _vz_api, _az_api = _zonas_conta_via_api(api, _team_ids_z)
                    if _vz_api and not st.session_state.get('velocity_zones_manual'):
                        st.session_state['velocity_zones_account'] = _vz_api
                        st.session_state['velocity_zones_source']  = 'api'
                        st.session_state['velocity_zones_from_api'] = True
                    if _az_api and not st.session_state.get('acceleration_zones_manual'):
                        st.session_state['acceleration_zones_account'] = _az_api
                        st.session_state['acceleration_zones_source']  = 'api'
                        st.session_state['acceleration_zones_from_api'] = True
                except Exception:
                    _applog.log_debug_exc()

                # ── Bandas DEFINIDAS PELO USUÁRIO: prioridade máxima ──────────
                # Se o usuário digitou/salvou os cortes da conta OpenField dele,
                # eles valem sobre API/padrão (é a configuração escolhida por ele).
                try:
                    _bu = _carregar_bandas_usuario()
                    if _bu and _bu.get('velocity_zones'):
                        st.session_state['velocity_zones_account'] = _bu['velocity_zones']
                        st.session_state['velocity_zones_manual'] = True
                        st.session_state['velocity_zones_source'] = 'manual'
                    if _bu and _bu.get('acceleration_zones'):
                        st.session_state['acceleration_zones_account'] = _bu['acceleration_zones']
                        st.session_state['acceleration_zones_manual'] = True
                        st.session_state['acceleration_zones_source'] = 'manual'
                except Exception:
                    _applog.log_debug_exc()

        if not st.session_state.df_activities.empty and token:
            # ── P5: Diagnóstico da sessão — nada falha em silêncio ────────────
            _diag_ev = st.session_state.get('_diag_eventos', [])
            _api_err = st.session_state.get('_api_last_err')
            _diag_n = len(_diag_ev) + (1 if _api_err else 0)
            with st.expander(f"🔍 Diagnóstico da sessão ({_diag_n})",
                             expanded=False):
                if _api_err:
                    st.warning(f"Último erro da API: `{_api_err}`")
                if not _diag_ev:
                    st.caption("✅ Nenhum dado descartado nem fallback registrado.")
                else:
                    st.caption("Eventos registrados no processamento (dados "
                               "descartados, fallbacks de fonte, falhas de "
                               "projeção). Atualiza a cada interação:")
                    for _ev in _diag_ev[-60:]:
                        st.markdown(f"- {_ev}")

            st.markdown("---")
            st.header("🎯 Filtros")
            
            if not st.session_state.df_teams.empty:
                st.subheader("🏢 Filtrar por Equipe")
                equipes = ['Todas'] + sorted(st.session_state.df_teams['nome'].unique().tolist())
                equipes_selecionadas = st.multiselect("Selecione as equipes:", equipes, default=['Todas'])
                st.session_state.equipes_selecionadas = equipes_selecionadas
            
            if not st.session_state.df_positions.empty:
                st.subheader("🎯 Filtrar por Posição")
                posicoes = ['Todas'] + sorted(st.session_state.df_positions['nome'].unique().tolist())
                posicoes_selecionadas = st.multiselect("Selecione as posições:", posicoes, default=['Todas'])
                st.session_state.posicoes_selecionadas = posicoes_selecionadas
            
            st.subheader("📅 Atividade")
            atividade_sel = st.selectbox("Selecione a atividade:", st.session_state.df_activities['nome'].tolist())
            st.session_state['_atividade_sel_cached'] = atividade_sel

            if atividade_sel:
                _act_row = st.session_state.df_activities[
                    st.session_state.df_activities['nome'] == atividade_sel]
                activity_id = _act_row['id'].values[0]
                st.session_state.activity_id = activity_id

                # ── Venue (campo) da atividade ────────────────────────────────
                # Salva lat, lng, rotation, length, width para pré-popular o
                # componente de posicionamento de campo automaticamente.
                if 'venue' in _act_row.columns:
                    _venue_val = _act_row['venue'].values[0]
                    st.session_state.venue = _venue_val if isinstance(_venue_val, dict) else {}
                else:
                    st.session_state.venue = {}

                # ── FEATURE 6: Descoberta dinâmica de parâmetros ─────────────
                _api_params_disc = st.session_state.get('api')
                if _api_params_disc:
                    try:
                        _sess_params_raw = _api_params_disc.get_session_parameters(activity_id)
                        if _sess_params_raw:
                            _p_list = (_sess_params_raw if isinstance(_sess_params_raw, list)
                                       else _sess_params_raw.get('data', []))
                            _param_names = []
                            for _pp in _p_list:
                                _pn = _pp.get('name') or _pp.get('slug') or str(_pp)
                                if _pn:
                                    _param_names.append(_pn)
                            st.session_state['available_params'] = _param_names
                    except Exception:
                        _applog.log_debug_exc()

                # ── FEATURE 7: Venues da conta Catapult ──────────────────────
                _api_venues = st.session_state.get('api')
                if _api_venues:
                    try:
                        if 'venues_catapult' not in st.session_state:
                            _venues_raw = _api_venues.get_venues()
                            if _venues_raw:
                                _venues_list = (_venues_raw if isinstance(_venues_raw, list)
                                                else _venues_raw.get('data', []))
                                st.session_state['venues_catapult'] = _venues_list
                    except Exception:
                        _applog.log_debug_exc()

                # Auto-match venue desta atividade com venues da conta
                _venue_act = st.session_state.get('venue', {})
                _venue_name_act = str(_venue_act.get('name', _venue_act.get('venue_name', '')))
                _venues_cat = st.session_state.get('venues_catapult', [])
                for _vc in _venues_cat:
                    _vc_name = str(_vc.get('name', ''))
                    if _vc_name and _vc_name.lower() == _venue_name_act.lower():
                        st.info(f"✅ Campo carregado automaticamente da conta Catapult: {_vc_name}")
                        # Auto-populate lat/lng/rotation/dimensions if available
                        for _fld in ('lat', 'lng', 'lon', 'rotation', 'length', 'width'):
                            if _vc.get(_fld) is not None:
                                _venue_act[_fld] = _vc[_fld]
                        st.session_state['venue'] = _venue_act
                        break

                # ── Item 14: Tags da Atividade ────────────────────────────────
                _api_tags = st.session_state.get('api')
                if _api_tags:
                    try:
                        _act_tags_raw = _api_tags.get_activity_tags(activity_id)
                        _tag_list = []
                        if _act_tags_raw:
                            if isinstance(_act_tags_raw, list):
                                for _tg in _act_tags_raw:
                                    _tn = _tg.get('name') or _tg.get('label') or str(_tg)
                                    if _tn:
                                        _tag_list.append(_tn)
                            elif isinstance(_act_tags_raw, dict):
                                for _tg in _act_tags_raw.get('data', []):
                                    _tn = _tg.get('name') or _tg.get('label') or ''
                                    if _tn:
                                        _tag_list.append(_tn)
                        if _tag_list:
                            _tags_html = "".join(
                                f"<span style='background:#1e3a5f;color:#90CAF9;border-radius:12px;"
                                f"padding:2px 10px;margin:2px 4px 2px 0;font-size:11px;"
                                f"display:inline-block'>{_t}</span>"
                                for _t in _tag_list
                            )
                            st.markdown(
                                f"<div style='margin-bottom:6px'><strong>🏷️ Tags:</strong><br>{_tags_html}</div>",
                                unsafe_allow_html=True
                            )
                    except Exception:
                        _applog.log_debug_exc()

                with st.spinner("Buscando períodos da atividade..."):
                    api = st.session_state.api
                    periods_raw = api.get_activity_periods(activity_id)
                    
                    # Se a API retornou períodos individuais, usamos apenas eles.
                    # "Atividade Completa" (period_id=None) é mantida APENAS como
                    # fallback quando nenhum período é encontrado.
                    if periods_raw and isinstance(periods_raw, list):
                        period_options = []
                        period_ids = {}
                        for p in periods_raw:
                            period_options.append(p.get('name', 'Período'))
                            period_ids[p.get('name', 'Período')] = p.get('id')
                        st.session_state.period_options = period_options
                        st.session_state.period_ids = period_ids
                        st.success(f"✅ {len(period_options)} períodos encontrados")
                    else:
                        # Sem períodos — fallback para atividade completa
                        period_options = ['Atividade Completa']
                        period_ids = {'Atividade Completa': None}
                        st.session_state.period_options = period_options
                        st.session_state.period_ids = period_ids

                st.subheader("📊 Selecionar Período(s)")
                _default_periodos = st.session_state.period_options  # todos por padrão
                periodos_selecionados = st.multiselect(
                    "Selecione um ou mais períodos para análise:",
                    options=st.session_state.period_options,
                    default=_default_periodos
                )
                st.session_state.periodos_selecionados = periodos_selecionados

                if st.button("🔍 Buscar Atletas da Atividade"):
                    with st.spinner("Buscando atletas em todos os períodos..."):
                        api = st.session_state.api
                        # ── Busca atletas em TODOS os períodos selecionados e faz união ──
                        _periodos_para_busca = periodos_selecionados if periodos_selecionados else (
                            st.session_state.period_options if st.session_state.period_options else []
                        )
                        athletes_by_id = {}  # {athlete_id: athlete_dict} — chave para deduplicar

                        def _extrair_lista_atletas(resp):
                            """Normaliza a resposta da API para lista de dicts de atleta."""
                            if not resp:
                                return []
                            if isinstance(resp, list):
                                return resp
                            if isinstance(resp, dict):
                                for _k in ['data', 'items', 'athletes']:
                                    if _k in resp and isinstance(resp[_k], list):
                                        return resp[_k]
                                if 'id' in resp:
                                    return [resp]
                            return []

                        if _periodos_para_busca:
                            for _per_nome in _periodos_para_busca:
                                _pid = st.session_state.period_ids.get(_per_nome)
                                if _pid:
                                    _resp = api.get_athletes_in_period(_pid)
                                else:
                                    _resp = api.get_activity_athletes(activity_id)
                                for _a in _extrair_lista_atletas(_resp):
                                    _aid = _a.get('id')
                                    if _aid and _aid not in athletes_by_id:
                                        athletes_by_id[_aid] = _a
                        else:
                            # Nenhum período selecionado — busca atletas da atividade inteira
                            _resp = api.get_activity_athletes(activity_id)
                            for _a in _extrair_lista_atletas(_resp):
                                _aid = _a.get('id')
                                if _aid and _aid not in athletes_by_id:
                                    athletes_by_id[_aid] = _a

                        athletes_in = list(athletes_by_id.values())

                        if athletes_in:
                            atletas_temp = []
                            for a in athletes_in:
                                nome = f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
                                if not nome:
                                    nome = a.get('name', 'Sem nome')
                                position_name = a.get('position_name', a.get('position', ''))
                                team_name = st.session_state.athlete_team_map.get(a.get('id'), '')
                                atletas_temp.append({
                                    'id': a.get('id'), 'nome': nome, 'camisa': a.get('jersey', ''),
                                    'posicao': position_name, 'equipe': team_name
                                })

                            df_atletas_temp = pd.DataFrame(atletas_temp)

                            if 'equipes_selecionadas' in st.session_state:
                                if 'Todas' not in st.session_state.equipes_selecionadas:
                                    df_atletas_temp = df_atletas_temp[df_atletas_temp['equipe'].isin(st.session_state.equipes_selecionadas)]

                            if 'posicoes_selecionadas' in st.session_state:
                                if 'Todas' not in st.session_state.posicoes_selecionadas:
                                    df_atletas_temp = df_atletas_temp[df_atletas_temp['posicao'].isin(st.session_state.posicoes_selecionadas)]

                            st.session_state.atletas_filtrados = df_atletas_temp
                            # Auto-seleciona TODOS os atletas encontrados — o usuário
                            # já escolheu os períodos, não precisa selecionar novamente.
                            _todos_nomes = df_atletas_temp['nome'].tolist()
                            st.session_state['atletas_sel'] = _todos_nomes
                            st.session_state['atletas_selecionados'] = _todos_nomes
                            _n_per_buscados = len(_periodos_para_busca) if _periodos_para_busca else 1
                            st.success(f"✅ {len(df_atletas_temp)} atletas encontrados e selecionados em {_n_per_buscados} período(s)")
                
                if 'atletas_filtrados' in st.session_state and not st.session_state.atletas_filtrados.empty:
                    st.subheader("🏃 Selecionar Atletas")

                    # Search box
                    _busca_atl = st.text_input("🔍 Buscar atleta:", placeholder="Nome ou camisa...", key="busca_atleta")

                    # Position preset buttons
                    if not st.session_state.atletas_filtrados.empty and 'posicao' in st.session_state.atletas_filtrados.columns:
                        _posicoes_disponiveis = sorted(
                            st.session_state.atletas_filtrados['posicao'].dropna().unique().tolist()
                        )
                        if _posicoes_disponiveis:
                            _pos_cols = st.columns(min(len(_posicoes_disponiveis) + 1, 4))
                            with _pos_cols[0]:
                                if st.button("👥 Todos", key="preset_todos", use_container_width=True):
                                    st.session_state['_preset_posicao'] = None
                            for _pi, _pn in enumerate(_posicoes_disponiveis[:3]):
                                with _pos_cols[_pi + 1]:
                                    if st.button(_pn[:8], key=f"preset_{_pn}", use_container_width=True):
                                        st.session_state['_preset_posicao'] = _pn

                    # Apply filters
                    _preset_pos = st.session_state.get('_preset_posicao')
                    _df_atl_disp = st.session_state.atletas_filtrados.copy()
                    if _busca_atl:
                        _df_atl_disp = _df_atl_disp[
                            _df_atl_disp['nome'].str.contains(_busca_atl, case=False, na=False) |
                            _df_atl_disp['camisa'].astype(str).str.contains(_busca_atl, case=False, na=False)
                        ]
                    if _preset_pos:
                        _df_atl_disp = _df_atl_disp[_df_atl_disp['posicao'] == _preset_pos]

                    atletas_disponiveis_sidebar = _df_atl_disp['nome'].tolist()

                    # Default: usa atletas já pré-selecionados (pelo botão Buscar)
                    # filtrando apenas os que ainda aparecem na lista atual.
                    _sel_prev = st.session_state.get('atletas_sel', [])
                    _sel_valido = [a for a in _sel_prev if a in atletas_disponiveis_sidebar]
                    if not _sel_valido:
                        _sel_valido = atletas_disponiveis_sidebar  # tudo selecionado por padrão
                    atletas_sel = st.multiselect(
                        "Atletas selecionados:",
                        options=atletas_disponiveis_sidebar,
                        default=_sel_valido,
                        key="atletas_selecionados"
                    )
                    st.session_state.atletas_sel = atletas_sel

        # ── Parâmetros de análise de esforço ─────────────────────────
        if not st.session_state.get('df_activities', pd.DataFrame()).empty and token:
            st.markdown("---")
            st.header("⚙️ Parâmetros de Esforço")
            st.slider(
                "⏱️ Duração mínima de acc/dec (s):",
                min_value=0.1, max_value=1.5,
                value=float(st.session_state.get('min_dur_esforco', _DEFAULT_MIN_DUR_S)),
                step=0.1,
                key="min_dur_esforco",
                help=(
                    "Tempo mínimo contínuo na zona de threshold para contar um evento.\n\n"
                    "🔵 Catapult OpenField: 0.6 s\n"
                    "🟡 Sistema alternativo: 0.4 s\n"
                    "Ajuste conforme o sistema de referência da sua análise."
                )
            )
            _dur_atual = st.session_state.get('min_dur_esforco', _DEFAULT_MIN_DUR_S)
            st.caption(
                f"Mínimo: **{_dur_atual:.1f} s** = "
                f"**{max(1, round(_dur_atual * _SENSOR_HZ))} frames** a 10 Hz"
            )

            st.slider(
                "⏱️ Duração mínima de velocidade (s):",
                min_value=0.1, max_value=3.0,
                value=float(st.session_state.get('min_dur_vel', _DEFAULT_MIN_DUR_VEL_S)),
                step=0.1,
                key="min_dur_vel",
                help=(
                    "Tempo mínimo contínuo dentro de uma banda de velocidade para "
                    "que o segmento seja contabilizado como um esforço.\n\n"
                    "💡 Valores maiores filtram movimentos breves e capturam "
                    "apenas esforços sustentados. Padrão: 1.0 s."
                )
            )
            _dur_vel_atual = st.session_state.get('min_dur_vel', _DEFAULT_MIN_DUR_VEL_S)
            st.caption(
                f"Mínimo: **{_dur_vel_atual:.1f} s** = "
                f"**{max(1, round(_dur_vel_atual * _SENSOR_HZ))} frames** a 10 Hz"
            )

        # ── Seletor de Eventos Futebol ────────────────────────────────
        if not st.session_state.get('df_activities', pd.DataFrame()).empty and token:
            st.markdown("---")
            st.header("⚽ Eventos Futebol")
            _todos_ev = list(FUTEBOL_EVENTS_CONFIG.keys())
            _sel_all  = st.checkbox("Selecionar todos", value=True, key="eventos_sel_all")
            if _sel_all:
                eventos_futebol_sel = _todos_ev
            else:
                eventos_futebol_sel = st.multiselect(
                    "Tipos de evento:",
                    options=_todos_ev,
                    default=_todos_ev[:4],
                    format_func=lambda k: FUTEBOL_EVENTS_CONFIG[k]['label'],
                    key="eventos_futebol_ms"
                )
            st.session_state.eventos_futebol_sel = eventos_futebol_sel
            if eventos_futebol_sel:
                st.caption(f"{len(eventos_futebol_sel)} tipo(s) selecionado(s). "
                           "Os eventos serão carregados junto com os dados.")

        # ── Máximo Histórico (CORRECTION 1) ──────────────────────────────
        if not st.session_state.get('df_activities', pd.DataFrame()).empty and token:
            st.markdown("---")
            with st.expander("📊 Máximo Histórico", expanded=False):
                st.caption(
                    "Vmax histórico buscado automaticamente do cadastro Catapult "
                    "(limiares → perfil → openfield_summary → /stats). "
                    "Usado em '% do Máximo' nas abas de esforços. "
                    "Você pode sobrescrever manualmente por atleta."
                )
                _hist_api = st.session_state.get('api')

                # ── Botão para re-buscar o perfil (limpa cache de perfil/thresholds) ──
                if _hist_api and st.button("🔄 Re-buscar via /stats Catapult", key="btn_refetch_hist_vmax"):
                    # Limpa apenas as fontes automáticas (preserva 'manual')
                    _src_g = st.session_state.get('hist_vmax_source', {})
                    _hvm_g = st.session_state.get('hist_vmax', {})
                    for _an_r in list(_hvm_g.keys()):
                        if _src_g.get(_an_r) != 'manual':
                            _hvm_g.pop(_an_r, None)
                            _src_g.pop(_an_r, None)
                    for _k in list(st.session_state.keys()):
                        if _k.startswith('profile_') or _k.startswith('thresholds_'):
                            del st.session_state[_k]
                    # Re-busca imediatamente via /stats
                    try:
                        _rb_stats = _hist_api.get_stats({
                            "group_by": ["athlete"],
                            "parameters": ["max_velocity"],
                            "source": "cached_stats",
                        })
                        if _rb_stats:
                            _rb_list = _rb_stats if isinstance(_rb_stats, list) else _rb_stats.get('data', [])
                            _n_rb = 0
                            for _rbs in (_rb_list or []):
                                _rbs_name = str(_rbs.get('athlete') or _rbs.get('athlete_name') or _rbs.get('name') or '')
                                _rbs_params = _rbs.get('parameters') or _rbs
                                _rbs_val = float(_rbs_params.get('max_velocity') or _rbs_params.get('max_speed') or
                                                 _rbs.get('max_velocity') or _rbs.get('max_speed') or 0)
                                if _rbs_name and _rbs_val > 0 and _src_g.get(_rbs_name) != 'manual':
                                    _rbs_ms = _rbs_val / 3.6 if _rbs_val > 15 else _rbs_val
                                    _hvm_g[_rbs_name] = _rbs_ms
                                    _src_g[_rbs_name] = 'catapult_stats'
                                    _n_rb += 1
                            st.toast(f"✅ {_n_rb} Vmax atualizadas via /stats")
                    except Exception:
                        _applog.log_debug_exc()
                    st.session_state['hist_vmax'] = _hvm_g
                    st.session_state['hist_vmax_source'] = _src_g
                    st.rerun()

                # ── Debug: inspeciona resposta bruta da API ───────────────────
                with st.expander("🛠️ Debug — resposta bruta da API", expanded=False):
                    st.caption("Use para identificar campos retornados pela API Catapult.")
                    _dbg_raw = st.session_state.get('_debug_stats_raw', [])
                    if _dbg_raw:
                        st.markdown("**Primeiros itens do POST /stats (cached_stats):**")
                        st.json(_dbg_raw)
                    else:
                        st.info("Carregue os atletas para ver a resposta bruta.")
                    _dbg_api = st.session_state.get('api')
                    _dbg_df  = st.session_state.get('df_athletes', pd.DataFrame())
                    if _dbg_api and not _dbg_df.empty:
                        if st.button("Inspecionar GET /athletes/{id}", key="btn_dbg_profile"):
                            _aid_dbg = _dbg_df.iloc[0]['id']
                            _prof_dbg = _dbg_api.get_athlete(_aid_dbg)
                            st.json(_prof_dbg or {})

                # ── Botão fallback: /stats ─────────────────────────────────────
                with st.expander("🔍 Buscar via /stats (fallback)", expanded=False):
                    st.caption("Use se os campos abaixo aparecerem como 'não detectado'.")
                    if _hist_api and st.button("Executar busca /stats", key="btn_fetch_hist_vmax"):
                        try:
                            _stats_resp = _hist_api.get_stats({
                                "group_by": ["athlete"],
                                "parameters": ["max_velocity"],
                                "source": "cached_stats",
                            })
                            if _stats_resp:
                                _sv_list = _stats_resp if isinstance(_stats_resp, list) else _stats_resp.get('data', [])
                                _hvm = st.session_state.get('hist_vmax', {})
                                _src_g2 = st.session_state.get('hist_vmax_source', {})
                                _n_upd = 0
                                for _sv in (_sv_list or []):
                                    _sv_name = str(_sv.get('athlete') or _sv.get('athlete_name', ''))
                                    _sv_val  = (_sv.get('parameters') or _sv).get('max_velocity', 0) or 0
                                    if _sv_name and float(_sv_val) > 0 and _src_g2.get(_sv_name) != 'manual':
                                        _hvm[_sv_name] = float(_sv_val)
                                        _src_g2[_sv_name] = 'stats'
                                        _n_upd += 1
                                st.session_state['hist_vmax'] = _hvm
                                st.session_state['hist_vmax_source'] = _src_g2
                                st.success(f"✅ Atualizado para {_n_upd} atleta(s) via /stats.")
                        except Exception as _hve:
                            st.error(f"Erro: {_hve}")

                # ── Tabela de valores detectados ───────────────────────────────
                _atletas_sidebar = (
                    st.session_state.atletas_sel
                    if st.session_state.get('atletas_sel') else []
                )
                _hvm_dict  = st.session_state.get('hist_vmax', {})
                _src_dict  = st.session_state.get('hist_vmax_source', {})

                _src_labels = {
                    'catapult_stats':    '📡 Catapult /stats',
                    'catapult_athletes': '📡 Cadastro',
                    'thresholds':        '📋 Limiares',
                    'profile':           '👤 Perfil',
                    'zones':             '🏷️ Zonas',
                    'stats':             '📊 /stats',
                    'summary':           '📋 Activity Summary',
                    'manual':            '✏️ Manual',
                    '':                  '❓ Não detectado',
                }

                for _an in _atletas_sidebar:
                    _cur_ms  = _hvm_dict.get(_an, 0.0)
                    _cur_kmh = round(_cur_ms * 3.6, 2) if _cur_ms else 0.0
                    _src_tag = _src_labels.get(_src_dict.get(_an, ''), '❓ Não detectado')
                    _col_v, _col_s = st.columns([3, 2])
                    with _col_v:
                        if _cur_kmh > 0:
                            st.markdown(f"**{_an}**  \n`{_cur_kmh} km/h`")
                        else:
                            st.markdown(f"**{_an}**  \n`— não detectado`")
                    with _col_s:
                        st.caption(_src_tag)

                    # Override manual com toggle
                    if st.toggle("✏️ Editar", key=f"_tgl_hvm_{_an}", value=False):
                        _inp = st.number_input(
                            "Vmax (km/h):",
                            min_value=0.0, max_value=50.0,
                            value=_cur_kmh,
                            step=0.1,
                            key=f"hist_vmax_{_an}",
                        )
                        if _inp > 0:
                            _hvm_dict[_an] = _inp / 3.6
                            _src_dict[_an] = 'manual'
                        elif _inp == 0 and _src_dict.get(_an) == 'manual':
                            # Apagar override manual
                            _hvm_dict.pop(_an, None)
                            _src_dict.pop(_an, None)

                st.session_state['hist_vmax']        = _hvm_dict
                st.session_state['hist_vmax_source'] = _src_dict

        # ── Auto-heal de schema das bandas ────────────────────────────────
        # Se a estrutura das bandas padrão mudou (ex.: aceleração de 8→6 bandas)
        # e a sessão já tinha zonas antigas em cache, descarta-as aqui — sem
        # exigir reconectar — para refletir a nova estrutura/derivação.
        if st.session_state.get('_zones_schema') != _ZONES_SCHEMA_VERSION:
            st.session_state['_zones_schema'] = _ZONES_SCHEMA_VERSION
            for _zk in ('velocity_zones_account', 'acceleration_zones_account',
                        'velocity_zones_manual', 'acceleration_zones_manual',
                        'velocity_zones_source', 'acceleration_zones_source',
                        '_bandas_deriv_key'):
                st.session_state.pop(_zk, None)
            st.session_state['velocity_zones_account'] = _DEFAULT_VELOCITY_ZONES[:]
            st.session_state['velocity_zones_source']  = 'default'
            st.session_state['acceleration_zones_account'] = _DEFAULT_ACCELERATION_ZONES[:]
            st.session_state['acceleration_zones_source']  = 'default'

        # ── Bandas de Velocidade / Aceleração — EDITÁVEIS pelo usuário ─────
        # A API não expõe os cortes das "Bandas Globais"; o usuário DIGITA aqui
        # os limiares da sua conta OpenField. É configuração (não fitagem ao
        # resultado). Após salvar, RECARREGUE a atividade: bandas de distância e
        # esforços (contados por sinal) são recalculados com estes cortes.
        if not st.session_state.get('df_activities', pd.DataFrame()).empty and token:
            with st.expander("🏷️ Bandas de Velocidade (editável)", expanded=False):
                st.caption("Digite os limiares (km/h) da **sua conta OpenField**. "
                           "Deixe o **Máx** da última banda em 45 (aberto). Ao "
                           "**Salvar**, o app recalcula sozinho todas as análises. "
                           "Documente estes valores no artigo.")
                _cur_vz = (st.session_state.get('velocity_zones_account')
                           or _DEFAULT_VELOCITY_ZONES)
                _edited_vz = st.data_editor(
                    pd.DataFrame([
                        {'Banda': z.get('name', f'B{_i+1}'),
                         'Mín (km/h)': round(float(z['min_ms']) * 3.6, 2),
                         'Máx (km/h)': (45.0 if z['max_ms'] >= 9000
                                        else round(float(z['max_ms']) * 3.6, 2))}
                        for _i, z in enumerate(_cur_vz)]),
                    use_container_width=True, hide_index=True, num_rows="dynamic",
                    key="editor_vel_zones",
                    column_config={
                        'Mín (km/h)': st.column_config.NumberColumn(
                            min_value=0.0, max_value=60.0, step=0.1, format="%.2f"),
                        'Máx (km/h)': st.column_config.NumberColumn(
                            min_value=0.0, max_value=60.0, step=0.1, format="%.2f")})
                _c1, _c2 = st.columns(2)
                with _c1:
                    if st.button("💾 Salvar bandas de velocidade",
                                 key="btn_save_vz", use_container_width=True):
                        _nz = []
                        for _, _r in _edited_vz.iterrows():
                            try:
                                _mn, _mx = float(_r['Mín (km/h)']), float(_r['Máx (km/h)'])
                            except (TypeError, ValueError):
                                continue
                            _nz.append({'name': str(_r.get('Banda') or f'B{len(_nz)+1}'),
                                        'min_ms': _mn / 3.6, 'max_ms': _mx / 3.6,
                                        'color': _CORES_BANDA_VEL_DEFAULT.get(len(_nz)+1, '#888888')})
                        if _nz:
                            st.session_state['velocity_zones_account'] = _nz
                            st.session_state['velocity_zones_manual'] = True
                            st.session_state['velocity_zones_source'] = 'manual'
                            _salvar_bandas_usuario(vel=_nz)
                            st.success("✅ Salvo — as análises são recalculadas "
                                       "automaticamente com estas bandas.")
                            st.rerun()
                with _c2:
                    if st.button("↩️ Restaurar padrão", key="btn_reset_vz",
                                 use_container_width=True):
                        _excluir_bandas_usuario('velocity')
                        st.session_state['velocity_zones_account'] = _DEFAULT_VELOCITY_ZONES[:]
                        st.session_state['velocity_zones_source'] = 'default'
                        st.session_state.pop('velocity_zones_manual', None)
                        st.session_state.pop('_ld_done_key', None)
                        st.rerun()

            with st.expander("🏷️ Bandas de Aceleração (editável)", expanded=False):
                st.caption("Gen2Acceleration (m/s²): B1/B2/B3 de aceleração (valores +) "
                           "e desaceleração (valores −). Ao **Salvar**, os esforços "
                           "contados por sinal são recalculados automaticamente.")
                _cur_az = (st.session_state.get('acceleration_zones_account')
                           or _DEFAULT_ACCELERATION_ZONES)
                _edited_az = st.data_editor(
                    pd.DataFrame([
                        {'Banda': z.get('name', f'B{_i+1}'),
                         'Mín (m/s²)': round(float(z['min_ms2']), 2),
                         'Máx (m/s²)': round(float(z['max_ms2']), 2)}
                        for _i, z in enumerate(_cur_az)]),
                    use_container_width=True, hide_index=True, num_rows="dynamic",
                    key="editor_acc_zones",
                    column_config={
                        'Mín (m/s²)': st.column_config.NumberColumn(
                            min_value=-20.0, max_value=20.0, step=0.1, format="%.2f"),
                        'Máx (m/s²)': st.column_config.NumberColumn(
                            min_value=-20.0, max_value=20.0, step=0.1, format="%.2f")})
                _c3, _c4 = st.columns(2)
                with _c3:
                    if st.button("💾 Salvar bandas de aceleração",
                                 key="btn_save_az", use_container_width=True):
                        _na = []
                        for _, _r in _edited_az.iterrows():
                            try:
                                _mn, _mx = float(_r['Mín (m/s²)']), float(_r['Máx (m/s²)'])
                            except (TypeError, ValueError):
                                continue
                            _na.append({'name': str(_r.get('Banda') or f'B{len(_na)+1}'),
                                        'min_ms2': _mn, 'max_ms2': _mx, 'color': '#888888'})
                        if _na:
                            st.session_state['acceleration_zones_account'] = _na
                            st.session_state['acceleration_zones_manual'] = True
                            st.session_state['acceleration_zones_source'] = 'manual'
                            _salvar_bandas_usuario(acc=_na)
                            st.session_state.pop('_ld_done_key', None)
                            st.success("✅ Salvo. Recarregue a atividade para recalcular.")
                            st.rerun()
                with _c4:
                    if st.button("↩️ Restaurar padrão", key="btn_reset_az",
                                 use_container_width=True):
                        _excluir_bandas_usuario('acceleration')
                        st.session_state['acceleration_zones_account'] = _DEFAULT_ACCELERATION_ZONES[:]
                        st.session_state['acceleration_zones_source'] = 'default'
                        st.session_state.pop('acceleration_zones_manual', None)
                        st.session_state.pop('_ld_done_key', None)
                        st.rerun()


        # ── Parâmetros disponíveis (FEATURE 6) ───────────────────────────
        _avail_params = st.session_state.get('available_params')
        if _avail_params:
            with st.expander("🔧 Parâmetros disponíveis", expanded=False):
                _desired = {"ts", "lat", "long", "v", "rv", "a", "hr", "pl",
                            "xy", "pq", "hdop", "ref", "o", "mp"}
                _ap_set = set(_avail_params) if isinstance(_avail_params, list) else set()
                _present = _desired & _ap_set
                _missing = _desired - _ap_set
                st.caption(f"Dispositivo: {len(_present)} parâmetros disponíveis.")
                if _missing:
                    st.warning(
                        f"Parâmetros não disponíveis neste dispositivo: "
                        f"`{'`, `'.join(sorted(_missing))}`"
                    )
                    if 'rv' in _missing:
                        st.info("rv (velocidade bruta) não disponível — perfil F-V usará v.")
                    if 'mp' in _missing:
                        st.info("mp (potência metabólica) não disponível.")
                st.write("Disponíveis:", sorted(_present))

    # Área principal
    if ('api' in st.session_state and 'atletas_sel' in st.session_state and
        st.session_state.atletas_sel and 'activity_id' in st.session_state):

        api = st.session_state.api
        activity_id = st.session_state.activity_id
        periodos_selecionados = st.session_state.get('periodos_selecionados', ['Atividade Completa'])
        period_ids = st.session_state.get('period_ids', {})
        _atividade_nome = locals().get('atividade_sel', st.session_state.get('_atividade_sel_cached', ''))
        if _atividade_nome:
            st.session_state['_atividade_sel_cached'] = _atividade_nome
        else:
            _atividade_nome = st.session_state.get('_atividade_sel_cached', '')

        # Dicionários para armazenar dados por período
        resultados_por_periodo = {}
        dados_sensor_por_atleta_por_periodo = {}
        dados_efforts_vel_por_periodo = {}
        dados_efforts_acc_por_periodo = {}
        dados_hr_efforts_por_periodo = {}
        dados_jump_efforts_por_periodo = {}
        dados_step_efforts_por_periodo = {}
        dados_posicao_por_periodo = {}
        dados_eventos_por_periodo = {}   # ← eventos futebol

        # Tipos de eventos futebol selecionados na sidebar
        eventos_futebol_sel = st.session_state.get('eventos_futebol_sel', list(FUTEBOL_EVENTS_CONFIG.keys()))
        eventos_futebol_str = ','.join(eventos_futebol_sel) if eventos_futebol_sel else ''

        # ── Container único de carregamento (substituído a cada atleta, apagado no fim) ──
        _n_per_ld   = len(periodos_selecionados)
        _n_atl_ld   = len(st.session_state.atletas_sel)
        _total_ld   = max(1, _n_per_ld * _n_atl_ld)
        _done_ld    = 0
        _ok_ld      = 0
        _warn_ld    = []
        _ld_box     = st.empty()

        for periodo_nome in periodos_selecionados:
            period_id = period_ids.get(periodo_nome)

            # ── (Validação/Minutos) participantes oficiais do período ────────
            # O OpenField só atribui o período aos atletas que participaram;
            # dispositivos ligados no banco inflavam Minutos (+30%), m/min e a
            # cauda do PlayerLoad. Carrega apenas quem está na lista oficial.
            _part_ids = set()
            if period_id:
                try:
                    _resp_part = api.get_athletes_in_period(period_id)
                    _lst_part = (_resp_part if isinstance(_resp_part, list)
                                 else (_resp_part or {}).get(
                                     'data', (_resp_part or {}).get('items', [])))
                    for _a_p in (_lst_part or []):
                        if isinstance(_a_p, dict) and _a_p.get('id'):
                            _part_ids.add(str(_a_p['id']))
                except Exception:
                    _part_ids = set()
                # Diagnóstico: quantos participantes a API declara p/ o período?
                # Se == nº total de atletas, o endpoint não distingue banco de
                # campo e o 'Minutos' NÃO baterá com o OpenField (limitação API).
                if _part_ids:
                    _diag_log('Carga', f"Período '{periodo_nome}': "
                                       f"{len(_part_ids)} participantes oficiais "
                                       "na API (filtro de Minutos ativo)")
                else:
                    _diag_log('Carga', f"Período '{periodo_nome}': API sem lista "
                                       "de participantes — todos os atletas "
                                       "carregados (Minutos pode divergir do OF)")

            resultados = []
            dados_sensor_por_atleta = {}
            dados_efforts_vel = {}
            dados_efforts_acc = {}
            dados_hr_efforts = {}
            dados_jump_efforts = {}
            dados_step_efforts = {}
            dados_posicao = {}
            dados_eventos = {}   # ← eventos futebol deste período

            _idx_per_ld = periodos_selecionados.index(periodo_nome) + 1

            for i, atleta_nome in enumerate(st.session_state.atletas_sel):
                _done_ld += 1
                _pct_ld   = _done_ld / _total_ld
                with _ld_box.container():
                    st.markdown(
                        f"<div style='padding:14px 20px;background:#111827;border-radius:10px;"
                        f"border:1px solid #1f2937'>"
                        f"<div style='color:#6b7280;font-size:12px;margin-bottom:6px'>"
                        f"⏳ &nbsp;Período &nbsp;<b style='color:#93c5fd'>{periodo_nome}</b>"
                        f"&nbsp; <span style='opacity:.6'>({_idx_per_ld}/{_n_per_ld})</span>"
                        f"</div>"
                        f"<div style='color:#f9fafb;font-size:15px;font-weight:600'>{atleta_nome}</div>"
                        f"<div style='color:#6b7280;font-size:12px;margin-top:4px'>"
                        f"{_done_ld} / {_total_ld} &nbsp;atletas</div></div>",
                        unsafe_allow_html=True
                    )
                    st.progress(_pct_ld)
                
                athlete_row = st.session_state.atletas_filtrados[st.session_state.atletas_filtrados['nome'] == atleta_nome]
                if athlete_row.empty:
                    continue
                    
                athlete_id = athlete_row['id'].values[0]

                # (Validação/Minutos) atleta fora da lista oficial do período →
                # não participou (dispositivo no banco); não carrega este período.
                if _part_ids and str(athlete_id) not in _part_ids:
                    _diag_log('Carga', f"{atleta_nome}: não participante do período "
                                       f"'{periodo_nome}' — excluído (espelha o "
                                       "Minutos do OpenField)")
                    continue

                athlete_posicao = athlete_row['posicao'].values[0] if 'posicao' in athlete_row.columns else ''
                athlete_equipe = athlete_row['equipe'].values[0] if 'equipe' in athlete_row.columns else ''

                # ── Item 15: limiares individuais do atleta ───────────────────
                # Tenta buscar limiares específicos deste atleta via API.
                # Se disponíveis, usa-os para a análise (armazenado em session_state).
                _thr_key = f"thresholds_{athlete_id}"
                if _thr_key not in st.session_state:
                    try:
                        _thr_raw = api.get_athlete_thresholds(athlete_id)
                        _thr_data = {}
                        if _thr_raw:
                            _thr_items = _thr_raw if isinstance(_thr_raw, list) else _thr_raw.get('data', [])
                            for _thr in (_thr_items if isinstance(_thr_items, list) else []):
                                _tname = _thr.get('name') or _thr.get('type', '')
                                _tval  = _thr.get('value') or _thr.get('threshold')
                                if _tname and _tval is not None:
                                    _thr_data[_tname] = float(_tval)
                        st.session_state[_thr_key] = _thr_data
                    except Exception:
                        st.session_state[_thr_key] = {}

                # ── Auto-popular Vmax histórico a partir do cadastro ──────────
                # Só sobrescreve se o novo valor for MAIOR que o atual
                # (preserva o máximo histórico entre sessões e não destrói
                # valores já captados pelo POST /stats ou openfield_summary).
                # Nunca sobrescreve override manual do usuário.
                _hvm_global = st.session_state.get('hist_vmax', {})
                _src_global = st.session_state.get('hist_vmax_source', {})
                _src_atual  = _src_global.get(atleta_nome, '')
                if _src_atual != 'manual':
                    _vmax_encontrado = 0.0
                    _vmax_fonte      = ''
                    # — Fonte 1: limiares cadastrados (thresholds endpoint) ——
                    _thr_data_now = st.session_state.get(_thr_key, {})
                    _vmax_keys_thr = [
                        'max_velocity', 'velocity_max', 'peak_speed',
                        'max_speed', 'v_max', 'maximum_velocity',
                        'MaxVelocity', 'MaxSpeed', 'velocidade_maxima',
                        'vmax',
                    ]
                    for _vk in _vmax_keys_thr:
                        _vv = _thr_data_now.get(_vk, 0)
                        if _vv:
                            _vvf = float(_vv)
                            # Converte km/h → m/s se plausível
                            if _vvf > 15:
                                _vvf /= 3.6
                            if 0.5 < _vvf < 15.0 and _vvf > _vmax_encontrado:
                                _vmax_encontrado = _vvf
                                _vmax_fonte = 'thresholds'
                    # — Fonte 2: perfil do atleta (GET /athletes/{id}) ————
                    try:
                        _prof_key = f"profile_{athlete_id}"
                        if _prof_key not in st.session_state:
                            _prof_raw = api.get_athlete(athlete_id)
                            st.session_state[_prof_key] = _prof_raw or {}
                        _prof_outer = st.session_state.get(_prof_key, {})
                        _prof = (
                            _prof_outer.get('data', _prof_outer)
                            if isinstance(_prof_outer, dict)
                            else (_prof_outer[0] if isinstance(_prof_outer, list)
                                  and _prof_outer else {})
                        )
                        _vmax_prof_keys = [
                            'max_speed', 'max_velocity', 'maximum_velocity',
                            'maximum_speed', 'peak_speed', 'v_max',
                        ]
                        for _pk in _vmax_prof_keys:
                            _pv = _prof.get(_pk, 0)
                            if _pv:
                                _pvf = float(_pv)
                                if _pvf > 15:
                                    _pvf /= 3.6
                                if 0.5 < _pvf < 15.0 and _pvf > _vmax_encontrado:
                                    _vmax_encontrado = _pvf
                                    _vmax_fonte = 'profile'
                    except Exception:
                        _applog.log_debug_exc()
                    # — Persiste apenas se o valor é MAIOR que o já armazenado ─
                    # (POST /stats já pode ter um valor melhor de outra sessão)
                    _cur_best = _hvm_global.get(atleta_nome, 0.0)
                    if _vmax_encontrado > _cur_best:
                        _hvm_global[atleta_nome] = _vmax_encontrado
                        _src_global[atleta_nome] = _vmax_fonte
                st.session_state['hist_vmax']        = _hvm_global
                st.session_state['hist_vmax_source'] = _src_global

                if period_id:
                    response         = api.get_period_sensor_data(period_id, athlete_id)
                    efforts_response = api.get_period_efforts(period_id, athlete_id,
                                                             "velocity,acceleration,heart_rate,jump,step_balance")
                    events_response  = api.get_period_events(period_id, athlete_id, eventos_futebol_str) if eventos_futebol_str else None
                else:
                    response         = api.get_sensor_data(activity_id, athlete_id)
                    efforts_response = api.get_activity_efforts(activity_id, athlete_id,
                                                               "velocity,acceleration,heart_rate,jump,step_balance")
                    events_response  = api.get_activity_events(activity_id, athlete_id, eventos_futebol_str) if eventos_futebol_str else None
                
                sensor_points = extrair_dados_sensor(response)
                
                if sensor_points:
                    dados_sensor_por_atleta[atleta_nome] = sensor_points

                    _atleta_zones = get_zones_for_athlete(atleta_nome)
                    metricas = calcular_metricas(sensor_points, atleta_nome,
                                                 zones=_atleta_zones)
                    if metricas:
                        metricas['Posição'] = athlete_posicao
                        metricas['Equipe'] = athlete_equipe
                        resultados.append(metricas)

                    if efforts_response:
                        _vel_eff, _acc_eff, _hr_eff, _jmp_eff, _step_eff = extrair_efforts_data(efforts_response)
                        if _vel_eff:
                            dados_efforts_vel[atleta_nome] = _vel_eff
                        if _acc_eff:
                            dados_efforts_acc[atleta_nome] = _acc_eff
                        if _hr_eff:
                            dados_hr_efforts[atleta_nome] = _hr_eff
                        if _jmp_eff:
                            dados_jump_efforts[atleta_nome] = _jmp_eff
                        if _step_eff:
                            dados_step_efforts[atleta_nome] = _step_eff

                    # ── OpenField pre-computed summary ────────────────────────
                    try:
                        if period_id:
                            _of_sum = api.get_athlete_period_summary(period_id, athlete_id)
                        else:
                            _of_sum = api.get_athlete_activity_summary(activity_id, athlete_id)
                        if _of_sum:
                            if atleta_nome not in dados_posicao:
                                dados_posicao[atleta_nome] = {
                                    'vel': [], 'xs': [], 'ys': [], 'acc': [], 'ts_pos': [],
                                    'posicao': athlete_posicao, 'equipe': athlete_equipe,
                                    'n_pontos': 0,
                                }
                            dados_posicao[atleta_nome]['openfield_summary'] = _of_sum
                            # ── Extrai max_velocity da summary para hist_vmax ──
                            # A summary retorna max_velocity em m/s (confirmado no
                            # código de comparação OpenField, linha ~9955).
                            # Guarda o maior valor observado entre os períodos carregados.
                            try:
                                _sum_d = (_of_sum if isinstance(_of_sum, dict)
                                          else (_of_sum[0] if isinstance(_of_sum, list) and _of_sum else {}))
                                _sum_p = _sum_d.get('parameters', _sum_d)
                                _sum_vmax_ms = float(_sum_p.get('max_velocity') or 0)
                                # Sanity check: valores plausíveis para velocidade humana
                                # (0.5 a 15 m/s = ~2 a 54 km/h)
                                if 0.5 < _sum_vmax_ms < 15.0:
                                    _hvm_now = st.session_state.get('hist_vmax', {})
                                    _src_now = st.session_state.get('hist_vmax_source', {})
                                    # Mantém o máximo histórico entre períodos;
                                    # nunca sobrescreve override manual do usuário
                                    if (_src_now.get(atleta_nome, '') != 'manual'
                                            and _sum_vmax_ms > _hvm_now.get(atleta_nome, 0)):
                                        _hvm_now[atleta_nome] = _sum_vmax_ms
                                        _src_now[atleta_nome] = 'summary'
                                        st.session_state['hist_vmax']        = _hvm_now
                                        st.session_state['hist_vmax_source'] = _src_now
                            except Exception:
                                _applog.log_debug_exc()
                    except Exception:
                        _applog.log_debug_exc()
                    
                    # A API devolve x,y com origem no canto inferior esquerdo (0,0).
                    # Filtra nulos e artefactos de projeção GPS (valores absurdamente altos).
                    _venue   = st.session_state.get('venue', {})
                    _fl_v    = float(_venue.get('length') or 105)
                    _fw_v    = float(_venue.get('width')  or 68)
                    pontos_pos = [
                        (float(p['x']), float(p['y']),
                         (p.get('v') or 0) * 3.6,
                         float(p.get('a') or 0),
                         float(p.get('ts') or 0))
                        for p in sensor_points
                        if p.get('x') is not None and p.get('y') is not None
                        and float(p['x']) > -15 and float(p['x']) < _fl_v + 15
                        and float(p['y']) > -15 and float(p['y']) < _fw_v + 15
                    ]
                    if pontos_pos:
                        xs          = [pt[0] for pt in pontos_pos]
                        ys          = [pt[1] for pt in pontos_pos]
                        velocidades = [pt[2] for pt in pontos_pos]
                        aceleracoes = [pt[3] for pt in pontos_pos]
                        ts_pos      = [pt[4] for pt in pontos_pos]

                        # ── Fallback de aceleração (dv/dt) ───────────────────
                        # Muitos dispositivos/exports NÃO trazem o parâmetro 'a'
                        # (aceleração). Sem ele, a WCS por bandas de aceleração
                        # ficaria zerada. Quando 'a' está ausente, derivamos a
                        # aceleração (m/s²) da série de velocidade usando os ts.
                        if not any(abs(_a) > 0.05 for _a in aceleracoes):
                            import statistics as _stacc
                            _vms = [float(v) / 3.6 for v in velocidades]  # km/h→m/s
                            _dts = []
                            for _i in range(1, len(ts_pos)):
                                _d = ts_pos[_i] - ts_pos[_i - 1]
                                _dts.append(_d if (_d and 0 < _d < 2) else None)
                            _valid_dt = [_d for _d in _dts if _d]
                            _dt_med = (_stacc.median(_valid_dt)
                                       if _valid_dt else 0.1)
                            _acc_calc = [0.0] * len(_vms)
                            for _i in range(1, len(_vms)):
                                _dt = (_dts[_i - 1] if _dts[_i - 1] else _dt_med)
                                if _dt and _dt > 0:
                                    _acc_calc[_i] = (_vms[_i] - _vms[_i - 1]) / _dt
                            # Suaviza (média móvel 3) e satura em ±10 m/s².
                            _acc_sm = []
                            for _i in range(len(_acc_calc)):
                                _lo = max(0, _i - 1)
                                _hi = min(len(_acc_calc), _i + 2)
                                _mv = sum(_acc_calc[_lo:_hi]) / (_hi - _lo)
                                _acc_sm.append(max(-10.0, min(10.0, _mv)))
                            aceleracoes = _acc_sm

                        dados_posicao[atleta_nome] = {
                            'vel': velocidades, 'xs': xs, 'ys': ys,
                            'acc': aceleracoes, 'ts_pos': ts_pos,
                            'posicao': athlete_posicao, 'equipe': athlete_equipe,
                            'n_pontos': len(pontos_pos)
                        }

                    # Coleta lat/lon reais (GPS) para o mapa satélite.
                    # Filtra zeros (sem lock de GPS) e valores geograficamente inválidos.
                    # Armazena ts (Unix timestamp) para filtrar pontos por esforço.
                    pontos_gps = [
                        (float(p['lat']), float(p['long']),
                         (p.get('v') or 0) * 3.6,
                         float(p.get('ts') or 0))
                        for p in sensor_points
                        if p.get('lat') is not None and p.get('long') is not None
                        and abs(float(p['lat'])) > 1e-6 and abs(float(p['long'])) > 1e-6
                        and -90 < float(p['lat']) < 90
                        and -180 < float(p['long']) < 180
                    ]
                    if pontos_gps:
                        step_gps = max(1, len(pontos_gps) // 30000)
                        gps_sub = pontos_gps[::step_gps]
                        if atleta_nome not in dados_posicao:
                            dados_posicao[atleta_nome] = {
                                'vel': [], 'xs': [], 'ys': [], 'acc': [], 'ts_pos': [],
                                'posicao': athlete_posicao, 'equipe': athlete_equipe,
                                'n_pontos': 0
                            }
                        dados_posicao[atleta_nome]['lats'] = [pt[0] for pt in gps_sub]
                        dados_posicao[atleta_nome]['lons'] = [pt[1] for pt in gps_sub]
                        dados_posicao[atleta_nome]['vels_gps'] = [pt[2] for pt in gps_sub]
                        dados_posicao[atleta_nome]['ts_gps']  = [pt[3] for pt in gps_sub]

                    # ── GPS Quality (pq, hdop, ref) e Odômetro (o) ────────────
                    # Item 8: qualidade do sinal GPS; Item 12: distância pelo odômetro nativo
                    _pq_vals   = [float(p['pq'])   for p in sensor_points if p.get('pq')   is not None and float(p.get('pq') or 0) > 0]
                    _hdop_vals = [float(p['hdop'])  for p in sensor_points if p.get('hdop') is not None]
                    _ref_vals  = [float(p['ref'])   for p in sensor_points if p.get('ref')  is not None and float(p.get('ref') or 0) > 0]
                    _o_vals    = [float(p['o'])     for p in sensor_points if p.get('o')    is not None]
                    if atleta_nome in dados_posicao:
                        dados_posicao[atleta_nome]['pq_mean']   = round(float(np.mean(_pq_vals)),   1) if _pq_vals   else None
                        dados_posicao[atleta_nome]['hdop_mean'] = round(float(np.mean(_hdop_vals)), 2) if _hdop_vals else None
                        dados_posicao[atleta_nome]['ref_mean']  = round(float(np.mean(_ref_vals)),  1) if _ref_vals  else None
                        # Odometer: distância acumulada nativa do dispositivo (mais preciso que integrar v)
                        if len(_o_vals) >= 2:
                            _o_start = min(_o_vals[0], _o_vals[-1])
                            _o_end   = max(_o_vals[0], _o_vals[-1])
                            dados_posicao[atleta_nome]['odometro_m'] = round(_o_end - _o_start, 1)
                        else:
                            dados_posicao[atleta_nome]['odometro_m'] = None
                        # Série temporal do odômetro para gráfico de evolução
                        if _o_vals:
                            _o_base = _o_vals[0]
                            dados_posicao[atleta_nome]['o_series'] = [v - _o_base for v in _o_vals]
                        else:
                            dados_posicao[atleta_nome]['o_series'] = []

                    # ── Processar eventos futebol ─────────────────────────────
                    if events_response:
                        ev_raw = extrair_eventos_futebol(events_response)
                        if ev_raw:
                            ts_g   = dados_posicao.get(atleta_nome, {}).get('ts_gps', [])
                            lats_g = dados_posicao.get(atleta_nome, {}).get('lats', [])
                            lons_g = dados_posicao.get(atleta_nome, {}).get('lons', [])
                            vels_g = dados_posicao.get(atleta_nome, {}).get('vels_gps', [])
                            dados_eventos[atleta_nome] = enriquecer_eventos_com_posicao(
                                ev_raw, ts_g, lats_g, lons_g, vels_g
                                # campo_config será enriquecido depois, no momento da visualização
                            )
                            n_ev = sum(len(v) for v in ev_raw.values())
                            _ok_ld += 1
                        else:
                            _ok_ld += 1
                    else:
                        _ok_ld += 1
                
            resultados_por_periodo[periodo_nome] = resultados
            dados_sensor_por_atleta_por_periodo[periodo_nome] = dados_sensor_por_atleta
            dados_efforts_vel_por_periodo[periodo_nome] = dados_efforts_vel
            dados_efforts_acc_por_periodo[periodo_nome] = dados_efforts_acc
            dados_hr_efforts_por_periodo[periodo_nome] = dados_hr_efforts
            dados_jump_efforts_por_periodo[periodo_nome] = dados_jump_efforts
            dados_step_efforts_por_periodo[periodo_nome] = dados_step_efforts
            dados_posicao_por_periodo[periodo_nome] = dados_posicao
            dados_eventos_por_periodo[periodo_nome] = dados_eventos

        # (Removido) Calibração automática das bandas de velocidade. Confirmou-se
        # que a Connect API v6 não expõe os limiares nem as distâncias por banda
        # (nem /velocity_zones, nem summary, nem /stats). O app usa limiares
        # FIXOS e documentados (_DEFAULT_VELOCITY_ZONES, padrão da literatura) —
        # instrumento determinístico, requisito para a validação científica.

        # (Removido) Derivação dos cortes a partir dos efforts. Inferir os
        # limiares dos dados é uma forma de auto-calibração — inadequada para um
        # instrumento de validação. O app usa os limiares FIXOS documentados
        # (_DEFAULT_VELOCITY_ZONES / _DEFAULT_ACCELERATION_ZONES); quando a conta
        # expõe as zonas por API (leitura limpa da configuração), essas têm
        # prioridade — ver _zonas_conta_via_api na conexão.

        # Apagar container de loading e mostrar resumo compacto
        _ld_box.empty()
        _warn_ld_n = _n_atl_ld - _ok_ld
        if _warn_ld_n > 0:
            _diag_log('Carga', f"{_warn_ld_n} atleta(s) sem dados de sensor "
                               "nesta atividade/períodos (excluídos das análises)")
        _per_label = ', '.join(periodos_selecionados)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;padding:10px 16px;"
            f"background:#052e16;border-radius:8px;border:1px solid #166534;margin-bottom:8px'>"
            f"<span style='font-size:18px'>✅</span>"
            f"<div><span style='color:#86efac;font-weight:600'>{_ok_ld} atletas carregados</span>"
            f"<span style='color:#4ade80;font-size:12px'>&nbsp;·&nbsp;{_per_label}</span>"
            + (f"<span style='color:#fca5a5;font-size:12px'>&nbsp;·&nbsp;{_warn_ld_n} sem dados</span>" if _warn_ld_n else "")
            + f"</div></div>",
            unsafe_allow_html=True
        )

        # ── Alerta de falha de carregamento ────────────────────────────────
        if _ok_ld == 0 and _n_atl_ld > 0:
            _api_err = st.session_state.pop('_api_last_err', None)
            if _api_err == 401:
                st.error(
                    "⚠️ **Token expirado ou inválido (HTTP 401).** "
                    "Gere um novo token em **Settings → API** no OpenField e cole na sidebar."
                )
            elif _api_err == 403:
                st.error(
                    "⚠️ **Acesso negado (HTTP 403).** "
                    "Seu token não tem permissão para acessar esses dados."
                )
            elif _api_err:
                st.error(
                    f"⚠️ **Erro na API Catapult** ({_api_err}). "
                    "Verifique o token e a conexão com a internet."
                )
            else:
                st.warning(
                    "⚠️ **Nenhum dado retornado.** Possíveis causas: "
                    "token expirado, atividade sem dados de sensor, ou atletas sem GPS ativo."
                )

        # ── Gerar "Períodos Combinados" quando há mais de 1 período ──────────
        _CHAVE_COMBINADO = '📊 Períodos Combinados'
        _periodos_reais = [k for k in resultados_por_periodo if k != _CHAVE_COMBINADO]
        if len(_periodos_reais) > 1:
            _res_combinado = combinar_periodos(
                {k: resultados_por_periodo[k] for k in _periodos_reais}
            )
            if _res_combinado:
                resultados_por_periodo[_CHAVE_COMBINADO] = _res_combinado

        # ── Paleta global persistente por atleta (usa _ATHLETE_PALETTE global) ─
        if 'athlete_colors' not in st.session_state:
            st.session_state['athlete_colors'] = {}
        _all_loaded_athletes = sorted({
            r.get('Atleta', '')
            for _p_res in resultados_por_periodo.values()
            for r in _p_res
            if r.get('Atleta')
        })
        for _i, _aname in enumerate(_all_loaded_athletes):
            if _aname not in st.session_state['athlete_colors']:
                st.session_state['athlete_colors'][_aname] = _ATHLETE_PALETTE[_i % len(_ATHLETE_PALETTE)]


        # ── Onboarding guiado — primeiro uso ─────────────────────────────────────
        if 'onboarding_done' not in st.session_state:
            st.session_state['onboarding_done'] = False
        if 'onboarding_step' not in st.session_state:
            st.session_state['onboarding_step'] = 1

        if not st.session_state['onboarding_done'] and 'api' not in st.session_state:
            with st.container():
                _ob_step = st.session_state['onboarding_step']

                _ob_steps = {
                    1: ("🔐 Token de Acesso", "Insira seu token Catapult Connect na sidebar à esquerda. O token é gerado em **Settings → API** na sua conta."),
                    2: ("🔄 Carregar Dados", "Clique em **🔄 Carregar Dados** na sidebar. Aguarde o carregamento de equipes, atletas e atividades."),
                    3: ("📅 Selecionar Sessão", "Escolha a **atividade** (sessão de treino ou jogo) e os **períodos** que deseja analisar."),
                    4: ("👥 Selecionar Atletas", "Busque atletas pelo nome ou use os **presets de posição**. Clique em ✅ Carregar Dados da Sessão."),
                }

                _ob_title, _ob_text = _ob_steps.get(_ob_step, ("✅ Pronto!", "Seu dashboard está configurado."))

                st.markdown(
                    f"<div style='background:linear-gradient(135deg,#1a3a5c,#0d2137);border:1px solid #2d5a8e;"
                    f"border-radius:12px;padding:20px 24px;margin-bottom:16px'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<div>"
                    f"<div style='color:#90CAF9;font-size:11px;font-weight:600;letter-spacing:1px'>PASSO {_ob_step}/4</div>"
                    f"<div style='color:white;font-size:18px;font-weight:700;margin:4px 0'>{_ob_title}</div>"
                    f"<div style='color:#bcd;font-size:13px'>{_ob_text}</div>"
                    f"</div>"
                    f"<div style='display:flex;gap:6px;align-items:center'>"
                    + "".join(f"<div style='width:8px;height:8px;border-radius:50%;background:{'#2196F3' if i+1==_ob_step else '#2d4a6a'}'></div>" for i in range(4))
                    + f"</div></div></div>",
                    unsafe_allow_html=True
                )

                _ob_col1, _ob_col2 = st.columns([1, 5])
                with _ob_col1:
                    if st.button("⏭️ Pular Tour", key="skip_onboarding", use_container_width=True):
                        st.session_state['onboarding_done'] = True
                        st.rerun()
                # Auto-advance when API is connected
                if _ob_step < 4 and 'df_activities' in st.session_state and not st.session_state.df_activities.empty:
                    st.session_state['onboarding_step'] = min(_ob_step + 1, 4)

        if resultados_por_periodo:
            st.subheader("📊 Métricas Biométricas")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🏃 Atletas", len(st.session_state.atletas_sel))
            # Exclui o período sintético "Períodos Combinados" (que já é a soma
            # dos períodos reais por atleta) para não contar cada atleta 2x e
            # garantir que os totais batam com a Tabela Descritiva (_df_ov).
            _res_iter = [(_p, _rs) for _p, _rs in resultados_por_periodo.items()
                         if _p != _CHAVE_COMBINADO]
            with col2:
                total_dist = 0
                for _p, resultados in _res_iter:
                    for r in resultados:
                        total_dist += r.get('Distância (m)', 0)
                st.metric("📏 Distância Total", f"{total_dist:,.0f} m")
            with col3:
                total_pl = 0
                for _p, resultados in _res_iter:
                    for r in resultados:
                        total_pl += r.get('PlayerLoad', 0)
                st.metric("⚡ PlayerLoad Total", f"{total_pl:,.0f}", help="Catapult PlayerLoad™: raiz quadrada da soma das acelerações ao quadrado nos 3 eixos (inercial)")
            with col4:
                max_vel = 0
                for _p, resultados in _res_iter:
                    for r in resultados:
                        max_vel = max(max_vel, r.get('Velocidade Máx (km/h)', 0))
                st.metric("💨 Velocidade Máx", f"{max_vel:.1f} km/h")

            _hr("ANÁLISE", "📊")

            _main_tabs = st.tabs([
                "🏠 Resumo",
                "🗺️ Campo & GPS",
                "🧠 Tática Coletiva",
                "📡 Ao Vivo",
                "📤 Exportação para Artigo",
                "📈 Monitoramento",
            ])

            # ── Criar sub-tabs dentro de cada aba principal ────────────────────
            with _main_tabs[0]:
                _sub_resumo = st.tabs(["🏠 Visão Geral", "📊 Por Posição"])
            with _main_tabs[1]:
                # Campo & GPS agora abriga também as antigas sub-abas de Carga Física
                _sub_campo = st.tabs(["🗺️ Campo de Futebol", "⚡ WCS",
                                      "💪 Neuromuscular", "📊 Janelas Temporais",
                                      "🏎️ Acc-Vel", "❤️ FC"])
            with _main_tabs[2]:
                render_tatica_coletiva(dados_posicao_por_periodo, periodos_selecionados, st.session_state.atletas_sel)
            with _main_tabs[4]:
                render_export_artigo(resultados_por_periodo,
                                     dados_sensor_por_atleta_por_periodo,
                                     dados_efforts_acc_por_periodo)
            with _main_tabs[5]:
                render_monitoramento()

            # Mapeamento: abas[N] aponta para o container correto na nova estrutura
            abas = [
                _sub_campo[0],    # 0: Campo de Futebol        → Campo & GPS
                _sub_campo[2],    # 1: Esforços                → Esforços Neuromusculares
                _sub_campo[3],    # 2: Janelas Temporais       → Campo & GPS
                _sub_campo[2],    # 3: Neuromuscular           → mesma aba (Esforços Neuromusculares)
                _sub_campo[4],    # 4: Acc-Vel                 → Campo & GPS
                _sub_campo[5],    # 5: FC (TRIMP + Zonas)      → Campo & GPS
                _sub_resumo[1],   # 6: Por Posição             → Resumo ✓
                _sub_campo[0],    # 7: (removido — antiga História do Jogo)
                _main_tabs[3],    # 8: Ao Vivo                → Ao Vivo (tab principal)
            ]

            # ==================== ABA RESUMO: OVERVIEW DASHBOARD ====================
            with _sub_resumo[0]:
                st.markdown("## 📊 Resumo da Sessão")

                if resultados_por_periodo:
                    # ── Coleta todas as linhas de todos os períodos (exceto combinado) ──
                    _ov_rows = []
                    for _p, _rs in resultados_por_periodo.items():
                        if _p == _CHAVE_COMBINADO:
                            continue
                        for _r in _rs:
                            _row = dict(_r)
                            _row['Período'] = _p
                            _ov_rows.append(_row)

                    if _ov_rows:
                        _df_ov_raw = pd.DataFrame(_ov_rows)

                        # ── Agrega por atleta — combina todos os períodos ────────────
                        # Métricas que se SOMAM entre períodos
                        _cols_sum = [c for c in [
                            'Distância (m)', 'Dist. > 19 km/h (m)', 'Dist. > 24 km/h (m)',
                            'Dist. 19-24 km/h (m)',
                            'Sprints (>24 km/h)', 'Acelerações (>3 m/s²)',
                            'Desacelerações (>-3 m/s²)', 'Desacelerações (<-3 m/s²)',
                            'RHIE Blocos', 'PlayerLoad', 'TRIMP',
                            'Acc 2-3 (m/s²)', 'Dcc 2-3 (m/s²)',
                            'Duração (min)',
                        ] if c in _df_ov_raw.columns]
                        # Métricas que se tomam o MÁXIMO entre períodos
                        _cols_max = [c for c in [
                            'Velocidade Máx (km/h)', 'Velocidade Bruta Máx (km/h)',
                            'Aceleração Máx (m/s²)', 'Acc Max (m/s²)', 'Dcc Max (m/s²)',
                            'Potência Met. Máx (W/kg)',
                        ] if c in _df_ov_raw.columns]
                        # Métricas que se tomam a MÉDIA entre períodos
                        _cols_mean = [c for c in [
                            'FC Média (bpm)', 'Velocidade Média (km/h)',
                        ] if c in _df_ov_raw.columns]
                        # Campos textuais: mantém o valor do primeiro período
                        _cols_first = [c for c in ['Posição', 'Equipe'] if c in _df_ov_raw.columns]

                        _agg_dict = {}
                        for _c in _cols_sum:  _agg_dict[_c] = 'sum'
                        for _c in _cols_max:  _agg_dict[_c] = 'max'
                        for _c in _cols_mean: _agg_dict[_c] = 'mean'
                        for _c in _cols_first: _agg_dict[_c] = 'first'

                        if _agg_dict and 'Atleta' in _df_ov_raw.columns:
                            _df_ov = (_df_ov_raw.groupby('Atleta', as_index=False)
                                      .agg(_agg_dict))
                            # Arredonda métricas numéricas
                            for _c in _cols_sum + _cols_max + _cols_mean:
                                if _c in _df_ov.columns:
                                    _df_ov[_c] = _df_ov[_c].round(1)
                        else:
                            _df_ov = _df_ov_raw.copy()

                        _n_periodos_ov = _df_ov_raw['Período'].nunique()

                        # ── KPI cards ────────────────────────────────────────────────
                        _ov_c1, _ov_c2, _ov_c3, _ov_c4, _ov_c5 = st.columns(5)
                        _ov_c1.metric("👥 Atletas", len(_df_ov['Atleta'].unique()) if 'Atleta' in _df_ov.columns else 0)
                        _ov_c2.metric("📏 Dist. Média", f"{_df_ov['Distância (m)'].mean():.0f} m" if 'Distância (m)' in _df_ov.columns else "—",
                                      help=f"Soma de todos os {_n_periodos_ov} período(s) por atleta, depois média do grupo")
                        _ov_c3.metric("💨 Vmax do Dia", f"{_df_ov['Velocidade Máx (km/h)'].max():.1f} km/h" if 'Velocidade Máx (km/h)' in _df_ov.columns else "—")
                        _ov_c4.metric("⚡ PL Total Médio", f"{_df_ov['PlayerLoad'].mean():.0f}" if 'PlayerLoad' in _df_ov.columns else "—",
                                      help="Catapult PlayerLoad™ somado em todos os períodos, depois média do grupo")
                        _hsr_col = 'Dist. > 19 km/h (m)'
                        _ov_c5.metric("🏃 HSR Médio", f"{_df_ov[_hsr_col].mean():.0f} m" if _hsr_col in _df_ov.columns else "—",
                                      help=f"HSR somado em {_n_periodos_ov} período(s) por atleta, depois média do grupo")

                        if _n_periodos_ov > 1:
                            st.caption(f"📋 Valores combinados de **{_n_periodos_ov} períodos** — somas, máximos e médias ponderadas por atleta.")

                        _hr("DISTÂNCIA POR ATLETA", "📏")

                        # ── Gráfico de barras — distância combinada (gradiente %) ─
                        if 'Atleta' in _df_ov.columns and 'Distância (m)' in _df_ov.columns:
                            _fig_ov = go.Figure()
                            _dist_vals = _df_ov['Distância (m)'].values
                            _dmin, _dmax = _dist_vals.min(), _dist_vals.max()
                            _drng = max(_dmax - _dmin, 1)
                            # Gradiente azul-escuro → ciano brilhante baseado no percentil
                            def _bar_color(_v):
                                _t = (_v - _dmin) / _drng          # 0=pior, 1=melhor
                                _r = int(21  + _t * (0   - 21))
                                _g = int(101 + _t * (229 - 101))
                                _b = int(192 + _t * (255 - 192))
                                return f'rgb({_r},{_g},{_b})'
                            _bar_colors = [_bar_color(v) for v in _dist_vals]
                            _fig_ov.add_trace(go.Bar(
                                x=_df_ov['Atleta'], y=_dist_vals,
                                marker=dict(
                                    color=_bar_colors,
                                    line=dict(color='rgba(255,255,255,0.08)', width=1),
                                ),
                                text=_dist_vals.round(0).astype(int),
                                textposition='outside',
                                textfont=dict(color='white', size=10),
                                hovertemplate='<b>%{x}</b><br>Distância: %{y:.0f} m<extra></extra>',
                            ))
                            _fig_ov.update_layout(
                                title=dict(
                                    text=f'Distância Total por Atleta ({_n_periodos_ov} período(s) combinado(s))',
                                    font=dict(color='white', size=14, family='Inter, sans-serif')
                                ),
                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                font=dict(color='white', family='Inter, sans-serif'),
                                xaxis=dict(gridcolor='rgba(255,255,255,0.06)',
                                           tickangle=-30, tickfont=dict(size=10)),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.06)',
                                           title='metros'),
                                height=330, margin=dict(t=50, b=80, l=10, r=10),
                                showlegend=False,
                                bargap=0.28,
                            )
                            st.plotly_chart(_fig_ov, use_container_width=True)

                        # ── M/min calculado da agregação (Dist / Duração) ────────────
                        if ('Duração (min)' in _df_ov.columns
                                and 'Distância (m)' in _df_ov.columns):
                            _dur_ov = _df_ov['Duração (min)'].replace(0, float('nan'))
                            _df_ov['M/min'] = (_df_ov['Distância (m)'] / _dur_ov).round(1)

                        # ── %Vmax ──────────────────────────────────────────────────
                        if 'Velocidade Máx (km/h)' in _df_ov.columns:
                            _hvm_ov = st.session_state.get('hist_vmax', {})
                            def _pct_vmax_ov(row):
                                _vel = float(row.get('Velocidade Máx (km/h)', 0) or 0)
                                _hist = float(_hvm_ov.get(row.get('Atleta', ''), 0) or 0) * 3.6
                                if not (5.0 <= _hist <= 60.0):
                                    _hist = 0.0
                                if _hist > 0:
                                    return round(_vel / _hist * 100, 1)
                                _gmax = _df_ov['Velocidade Máx (km/h)'].max()
                                return round(_vel / _gmax * 100, 1) if _gmax > 0 else 0.0
                            _df_ov['%Vmax'] = _df_ov.apply(_pct_vmax_ov, axis=1)

                        # ── Tabela Descritiva com coloração por percentil ─────────
                        _hr("TABELA DESCRITIVA DE DESEMPENHO", "📋")
                        st.subheader("📋 Tabela Descritiva de Desempenho")
                        st.caption("Coloração por percentil do grupo: 🟢 Top 33% · 🟡 Médio · 🔴 Bottom 33%")
                        st.caption("ℹ️ **RHIE** — Repeated High Intensity Efforts · **HSR** — Dist. >19 km/h · **M/min** — Metros por minuto (elite: 110–130)")

                        _OV_TD_COLS = [c for c in [
                            'Atleta', 'Posição',
                            'Duração (min)', 'Distância (m)',
                            'M/min',                              # ← 5ª coluna
                            'Dist. 19-24 km/h (m)', 'Dist. > 24 km/h (m)',
                            'Dist. > 19 km/h (m)', 'Sprints (>24 km/h)',
                            'Velocidade Máx (km/h)', '%Vmax',
                            # 'Acc 2-3 (m/s²)' e 'Dcc 2-3 (m/s²)' removidos a pedido
                            'Acelerações (>3 m/s²)', 'Desacelerações (<-3 m/s²)',
                            'Acc Max (m/s²)', 'Dcc Max (m/s²)',
                            'PlayerLoad', 'RHIE Blocos',
                            'FC Média (bpm)',
                        ] if c in _df_ov.columns]

                        if _OV_TD_COLS:
                            _df_ov_show = (
                                _df_ov[_OV_TD_COLS]
                                # Remove linhas sem atleta ou inteiramente vazias
                                .dropna(subset=['Atleta'])
                                .loc[lambda d: d['Atleta'].astype(str).str.strip() != '']
                                .sort_values(
                                    'Distância (m)' if 'Distância (m)' in _OV_TD_COLS else _OV_TD_COLS[0],
                                    ascending=False
                                )
                                .reset_index(drop=True)
                            )

                            _OV_NUM = [c for c in _OV_TD_COLS if c not in ('Atleta', 'Posição')]

                            def _ov_style_pct(col):
                                if col.name not in _OV_NUM:
                                    return [''] * len(col)
                                p33 = col.quantile(0.33)
                                p66 = col.quantile(0.66)
                                out = []
                                for v in col:
                                    try:
                                        vf = float(v)
                                        if vf >= p66:
                                            out.append('background-color:#1a5c2e;color:white;font-weight:bold')
                                        elif vf >= p33:
                                            out.append('background-color:#7d6a08;color:white')
                                        else:
                                            out.append('background-color:#7b1a1a;color:white')
                                    except Exception:
                                        out.append('')
                                return out

                            _1dec_ov = {
                                'Velocidade Máx (km/h)', 'Velocidade Média (km/h)',
                                'M/min', 'Acc Max (m/s²)', 'Dcc Max (m/s²)',
                                'FC Média (bpm)',
                            }
                            _ov_fmt = {}
                            for _c in _OV_NUM:
                                if _c == '%Vmax':
                                    _ov_fmt[_c] = '{:.1f}%'
                                elif _c in _1dec_ov:
                                    _ov_fmt[_c] = '{:.1f}'
                                else:
                                    _ov_fmt[_c] = '{:.0f}'

                            st.markdown(
                                "<style>"
                                "[data-testid='stDataFrame'] th {"
                                "  text-align:center !important;"
                                "  justify-content:center !important;"
                                "}"
                                "[data-testid='stDataFrame'] td {"
                                "  text-align:center !important;"
                                "}"
                                "</style>",
                                unsafe_allow_html=True,
                            )
                            # Altura dinâmica: mostra todos os atletas sem scroll vertical
                            # ~38 px por linha + 60 px de header/padding
                            _ov_height = max(200, 38 * len(_df_ov_show) + 60)
                            st.dataframe(
                                _df_ov_show.style
                                .apply(_ov_style_pct, axis=0)
                                .format(_ov_fmt, na_rep='—')
                                .set_properties(**{'text-align': 'center'})
                                .set_table_styles([
                                    {'selector': 'th',
                                     'props': [('text-align', 'center'),
                                               ('font-weight', 'bold')]},
                                    {'selector': 'td',
                                     'props': [('text-align', 'center')]},
                                ]),
                                use_container_width=True,
                                hide_index=True,
                                height=_ov_height,
                            )
                            st.download_button(
                                "📥 Exportar Tabela (CSV)",
                                _df_ov_show.to_csv(index=False).encode('utf-8'),
                                "resumo_sessao.csv",
                                mime='text/csv',
                            )
                else:
                    st.info("⚽ Carregue uma sessão na sidebar para visualizar o resumo.")
                    st.markdown("""
**Como começar:**
1. 🔐 Insira seu token Catapult na sidebar
2. 🔄 Clique em "Carregar Dados"
3. 📅 Selecione a atividade
4. 👥 Escolha os atletas
5. ✅ Clique em "Carregar Dados da Sessão"
                    """)

            # ==================== ABA 1: CAMPO DE FUTEBOL ====================
            with abas[0]:
                st.subheader("🗺️ Campo de Futebol — Análise de Movimentação")
                st.caption(REFERENCIAS["campo"])

                # ── Split View toggle ──────────────────────────────────────────
                _split_mode = st.toggle("⚖️ Modo Comparação (Split View)", value=False, key="split_view_mode")

                if _split_mode and dados_posicao_por_periodo:
                    _split_col_A, _split_col_B = st.columns(2)
                    for _split_idx, _split_col in enumerate([_split_col_A, _split_col_B]):
                        with _split_col:
                            _lbl = "A" if _split_idx == 0 else "B"
                            st.markdown(f"**Lado {_lbl}**")
                            _sp_per = st.selectbox(f"Período {_lbl}:", list(dados_posicao_por_periodo.keys()), key=f"split_per_{_lbl}")
                            _sp_ats = list(dados_posicao_por_periodo.get(_sp_per, {}).keys())
                            _sp_atl = st.selectbox(f"Atleta {_lbl}:", _sp_ats, key=f"split_atl_{_lbl}") if _sp_ats else None
                            if _sp_atl:
                                _sp_d = dados_posicao_por_periodo.get(_sp_per, {}).get(_sp_atl, {})
                                _sp_xs = list(_sp_d.get('xs', []))
                                _sp_ys = list(_sp_d.get('ys', []))
                                _sp_vs = list(_sp_d.get('vel', _sp_d.get('vels_gps', [])))

                                # ── Fallback GPS: converte lat/lon se não há x/y ────
                                _sp_gps_used = False
                                if not _sp_xs:
                                    _sp_lats = _sp_d.get('lats', [])
                                    _sp_lons = _sp_d.get('lons', [])
                                    if _sp_lats and _sp_lons:
                                        # Busca qualquer cfg de campo salvo em session_state
                                        _sp_cfg = None
                                        for _ssk, _ssv in st.session_state.items():
                                            if (isinstance(_ssk, str)
                                                    and _ssk.startswith('campo_cfg__')
                                                    and isinstance(_ssv, dict)
                                                    and _ssv.get('fl')):
                                                _sp_cfg = _ssv
                                                break
                                        if _sp_cfg:
                                            try:
                                                _sp_xs, _sp_ys = gps_para_campo_coords(
                                                    _sp_lats, _sp_lons, _sp_cfg)
                                                _sp_vs = list(_sp_d.get('vels_gps', [0.0]*len(_sp_xs)))
                                                _sp_gps_used = True
                                            except Exception:
                                                _applog.log_debug_exc()

                                if _sp_xs:
                                    # Direção de ataque para este lado do split view
                                    _sp_atk_opts = ["— Não definido", "➡️  Esq → Dir", "⬅️  Dir → Esq"]
                                    _sp_atk_key = f"attack_dir_split_{_lbl}"
                                    _sp_atk_default = 0
                                    if st.session_state.get(_sp_atk_key) == 'left_to_right':
                                        _sp_atk_default = 1
                                    elif st.session_state.get(_sp_atk_key) == 'right_to_left':
                                        _sp_atk_default = 2
                                    _sp_atk_sel = st.radio(
                                        "⚽ Ataque:",
                                        _sp_atk_opts,
                                        index=_sp_atk_default,
                                        horizontal=True,
                                        key=f"_sp_atk_radio_{_lbl}",
                                    )
                                    if "➡️" in _sp_atk_sel:
                                        st.session_state[_sp_atk_key] = 'left_to_right'
                                    elif "⬅️" in _sp_atk_sel:
                                        st.session_state[_sp_atk_key] = 'right_to_left'
                                    else:
                                        st.session_state[_sp_atk_key] = None
                                    _sp_fig = desenhar_campo_futebol_bonito(
                                        title=f"{_sp_atl} — {_sp_per}"
                                              + (" (GPS)" if _sp_gps_used else ""),
                                        attack_direction=st.session_state[_sp_atk_key],
                                    )
                                    _sp_step = max(1, len(_sp_xs) // 5000)
                                    _sp_vs_sub = ([_sp_vs[i] for i in range(0, len(_sp_vs), _sp_step)]
                                                  if _sp_vs else [0] * len(_sp_xs[::_sp_step]))
                                    _sp_fig.add_trace(go.Scatter(
                                        x=_sp_xs[::_sp_step], y=_sp_ys[::_sp_step],
                                        mode='markers',
                                        marker=dict(
                                            size=3,
                                            color=_sp_vs_sub,
                                            colorscale=[[0,'#2196F3'],[0.4,'#4CAF50'],[0.7,'#FF9800'],[1,'#F44336']],
                                            showscale=False,
                                        ),
                                        hoverinfo='skip', showlegend=False,
                                    ))
                                    _sp_fig.update_layout(height=400, margin=dict(t=30,b=10,l=10,r=10))
                                    st.plotly_chart(_sp_fig, use_container_width=True)
                                else:
                                    _sp_has_gps = bool(_sp_d.get('lats'))
                                    if _sp_has_gps:
                                        st.warning(
                                            "📡 Dados GPS disponíveis mas campo não calibrado.\n\n"
                                            "Vá até a aba **Campo & GPS** no modo normal, aplique o campo "
                                            "no mapa e retorne ao Split View."
                                        )
                                    else:
                                        st.info("Sem dados de posição (x/y ou GPS) para este atleta/período.")
                elif dados_posicao_por_periodo:
                    _todos_periodos = list(dados_posicao_por_periodo.keys())
                    col1, col2 = st.columns(2)
                    with col1:
                        periodos_mapa_sel = st.multiselect(
                            "Selecione o(s) período(s):",
                            options=_todos_periodos,
                            default=_todos_periodos[:1],
                            key="periodos_mapa_sel"
                        )
                    atleta_mapa = None
                    # Período de referência (primeiro selecionado) — usado para config do campo
                    periodo_mapa = periodos_mapa_sel[0] if periodos_mapa_sel else _todos_periodos[0]
                    with col2:
                        # União de atletas disponíveis em todos os períodos selecionados
                        _ats_disponiveis = []
                        for _pm in (periodos_mapa_sel or [periodo_mapa]):
                            for _a in dados_posicao_por_periodo.get(_pm, {}).keys():
                                if _a not in _ats_disponiveis:
                                    _ats_disponiveis.append(_a)
                        if _ats_disponiveis:
                            atletas_mapa = st.multiselect(
                                "Selecione o(s) atleta(s):",
                                _ats_disponiveis,
                                default=_ats_disponiveis[:1],
                                key="atletas_mapa"
                            )
                        else:
                            atletas_mapa = []
                    # Atleta primário: usado para GPS, calibração do campo e ETAPA 2
                    atleta_mapa = atletas_mapa[0] if atletas_mapa else None

                    if not periodos_mapa_sel:
                        st.info("Selecione pelo menos um período.")
                    elif atleta_mapa:
                        # Combina dados de todos os períodos selecionados
                        dados = dados_posicao_por_periodo.get(periodo_mapa, {}).get(atleta_mapa, {})

                        # GPS combinado (todos os períodos)
                        lats_gps, lons_gps, vels_gps, ts_gps = [], [], [], []
                        for _pm in periodos_mapa_sel:
                            _d = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                            lats_gps  += _d.get('lats', [])
                            lons_gps  += _d.get('lons', [])
                            vels_gps  += _d.get('vels_gps', [])
                            ts_gps    += _d.get('ts_gps', [])

                        n_xy  = sum(dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {}).get('n_pontos', 0) for _pm in periodos_mapa_sel)
                        n_gps = len(lats_gps)
                        _label_periodos = " + ".join(periodos_mapa_sel)
                        st.caption(f"📡 Pontos campo (x/y): **{n_xy}** &nbsp;|&nbsp; 🌍 Pontos GPS: **{n_gps}** &nbsp;|&nbsp; 📅 **{_label_periodos}**")

                        # ── Item 8: Badge de qualidade GPS ──────────────────────
                        st.caption("ℹ️ **HDOP** — Horizontal Dilution of Precision: qualidade do sinal GPS. <1.5 = excelente, 1.5–3 = bom, >3 = ruim", unsafe_allow_html=False)
                        _gps_q_parts = []
                        for _pm in periodos_mapa_sel:
                            _dq = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                            _pq_m  = _dq.get('pq_mean')
                            _hdop_m = _dq.get('hdop_mean')
                            _ref_m = _dq.get('ref_mean')
                            if _pq_m is not None or _hdop_m is not None:
                                _gps_q_parts.append((_pm, _pq_m, _hdop_m, _ref_m))
                        if _gps_q_parts:
                            _gps_badge_html = "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px'>"
                            for _pm, _pq, _hdop, _ref in _gps_q_parts:
                                # pq: >80% good, 60-80% ok, <60% poor; hdop: <1.5 excellent, <3 good, >3 poor
                                _pq_color  = ('#4CAF50' if (_pq  or 0) >= 80 else '#FF9800' if (_pq  or 0) >= 60 else '#F44336') if _pq  is not None else '#888'
                                _hdop_color= ('#4CAF50' if (_hdop or 0) <= 1.5 else '#FF9800' if (_hdop or 0) <= 3.0 else '#F44336') if _hdop is not None else '#888'
                                _pq_txt    = f"PQ: <b style='color:{_pq_color}'>{_pq:.0f}%</b>" if _pq  is not None else ""
                                _hdop_txt  = f" HDOP: <b style='color:{_hdop_color}'>{_hdop:.2f}</b>" if _hdop is not None else ""
                                _ref_txt   = f" Satélites: <b>{_ref:.0f}</b>" if _ref is not None else ""
                                _gps_badge_html += (
                                    f"<span style='background:#1e2d3d;border:1px solid #2d4a6a;border-radius:8px;"
                                    f"padding:4px 10px;font-size:11px;color:#ccc'>"
                                    f"📡 {_pm} — {_pq_txt}{_hdop_txt}{_ref_txt}</span>"
                                )
                            _gps_badge_html += "</div>"
                            st.markdown(_gps_badge_html, unsafe_allow_html=True)

                        # ── Item 12: Odômetro vs distância integrada ─────────────
                        _odo_vals, _dist_vals = [], []
                        for _pm in periodos_mapa_sel:
                            _dq = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                            _odo = _dq.get('odometro_m')
                            if _odo is not None:
                                _odo_vals.append(_odo)
                        _res_rows = resultados_por_periodo.get(periodos_mapa_sel[0], []) if periodos_mapa_sel else []
                        _dist_integrada = next((r.get('Distância (m)', 0) for r in _res_rows if r.get('Atleta') == atleta_mapa), None)
                        if _odo_vals and _dist_integrada:
                            _odo_total = sum(_odo_vals)
                            _div = abs(_odo_total - _dist_integrada)
                            _div_pct = _div / max(_dist_integrada, 1) * 100
                            _odo_col = '#4CAF50' if _div_pct < 5 else '#FF9800' if _div_pct < 15 else '#F44336'
                            st.markdown(
                                f"<div style='display:flex;gap:12px;align-items:center;margin-bottom:6px;"
                                f"background:#0d1f0d;border-left:3px solid #4CAF50;padding:6px 12px;border-radius:0 6px 6px 0'>"
                                f"<span style='color:#86efac;font-size:12px'>🛣️ <b>Odômetro Catapult</b></span>"
                                f"<span style='color:white;font-size:13px'><b>{_odo_total:,.0f} m</b></span>"
                                f"<span style='color:#aaa;font-size:11px'>vs. GPS integrado: {_dist_integrada:,.0f} m</span>"
                                f"<span style='color:{_odo_col};font-size:11px'>Δ {_div_pct:.1f}%</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        # Chave por atleta (campo físico não muda entre períodos)
                        campo_key = f"campo_cfg__{atleta_mapa}"

                        # ── Auto-carga do banco compartilhado de venues ──────────
                        # Se o venue da atividade já foi configurado por algum usuário,
                        # aplica automaticamente — sem precisar passar pela FASE 1.
                        _venues_db   = _carregar_venues()
                        _venue_name  = st.session_state.get('venue', {}).get('name', '')
                        if (_venue_name and _venue_name in _venues_db
                                and campo_key not in st.session_state):
                            _vs = _venues_db[_venue_name]
                            st.session_state[campo_key] = {
                                'lat':          float(_vs.get('lat') or 0),
                                'lon':          float(_vs.get('lon') or 0),
                                'rot':          int(_vs.get('rot',   0)),
                                'fl':           int(_vs.get('fl',  105)),
                                'fw':           int(_vs.get('fw',   68)),
                                'ig':           int(_vs.get('ig',    1)),
                                '_from_venues': _venue_name,
                                '_saved_at':    _vs.get('saved_at', ''),
                            }

                        campo_aplicado = campo_key in st.session_state

                        # ══════════════════════════════════════════════════════
                        # FASE 1 — POSICIONAMENTO INTERATIVO NO SATÉLITE
                        # ══════════════════════════════════════════════════════
                        if not campo_aplicado:
                            st.markdown("### 1️⃣ Posicionamento no Campo Físico")
                            st.markdown(
                                "Ajuste o campo de futebol sobre a imagem de satélite e clique "
                                "**✅ Aplicar Campo** no painel inferior do mapa."
                            )

                            if lats_gps and lons_gps:
                                st.info(
                                    "📌 **Edite Lat/Lon** (ou use ↑↓) para mover o ⊙ amarelo · "
                                    "⚽ **Mostrar Campo** para ativar o overlay · "
                                    "Sliders ajustam rotação e dimensões **em tempo real** · "
                                    "**✅ Aplicar Campo** quando estiver satisfeito"
                                )

                                # Subamostrar GPS para o componente (máx 3000 pts)
                                _n = len(lats_gps)
                                _step = max(1, _n // 3000)
                                _pts = [
                                    {"lat": round(lats_gps[i], 7),
                                     "lon": round(lons_gps[i], 7),
                                     "v":   round(float(vels_gps[i]) if i < len(vels_gps) else 0.0, 1)}
                                    for i in range(0, _n, _step)
                                ]

                                # ── Venue da atividade → pré-popula campo ────────────
                                # Centro do mapa: sempre usa mediana GPS real dos atletas
                                # (o venue.lat/lng da API pode estar cadastrado num local errado).
                                # Dimensões e rotação do venue são usadas quando disponíveis.
                                _venue_info = st.session_state.get('venue', {})
                                _venue_rot  = int(_venue_info.get('rotation') or 0)
                                # Dimensões padrão fixas (105×68m). O usuário ainda pode
                                # ajustar comprimento/largura no painel do mapa; apenas os
                                # valores de ENTRADA padrão são 105×68 (não os do venue).
                                _venue_fl   = 105.0
                                _venue_fw   = 68.0

                                # Posição do mapa: mediana GPS > venue lat/lng > 0,0
                                if lats_gps:
                                    _lat_c = round(float(np.median(lats_gps)), 7)
                                    _lon_c = round(float(np.median(lons_gps)), 7)
                                else:
                                    _vl = _venue_info.get('lat')
                                    _vg = _venue_info.get('lng')
                                    _lat_c = round(float(_vl), 7) if _vl else 0.0
                                    _lon_c = round(float(_vg), 7) if _vg else 0.0

                                if _venue_info:
                                    st.caption(
                                        f"📡 Campo padrão: **{_venue_fl:.0f}×{_venue_fw:.0f}m** · "
                                        f"rot **{_venue_rot}°** · "
                                        f"centro **{_lat_c:.5f}, {_lon_c:.5f}** "
                                        f"_(ajustável no painel do mapa)_"
                                    )

                                # Componente bidirecional: retorna {lat,lon,rot,fl,fw,ig}
                                # quando o usuário clica "✅ Aplicar Campo" no painel do mapa
                                resultado_campo = _campo_component(
                                    pts=_pts,
                                    lat_c=_lat_c,
                                    lon_c=_lon_c,
                                    rot=_venue_rot,
                                    fl=_venue_fl,
                                    fw=_venue_fw,
                                    legend=_legenda_vel_items(),
                                    key=f"campo_mapa_{atleta_mapa}",
                                    default=None
                                )

                                if resultado_campo is not None:
                                    _novo_cfg = {
                                        'lat': float(resultado_campo['lat']),
                                        'lon': float(resultado_campo['lon']),
                                        'rot': int(resultado_campo['rot']),
                                        'fl':  int(resultado_campo['fl']),
                                        'fw':  int(resultado_campo['fw']),
                                        'ig':  int(resultado_campo['ig']),
                                    }
                                    st.session_state[campo_key] = _novo_cfg
                                    st.success("✅ Campo aplicado com sucesso!")

                                    # ── Oferecer salvar no banco compartilhado ──────
                                    _vdb_atual  = _carregar_venues()
                                    _def_vname  = (_venue_name
                                                   or st.session_state.get('venue', {}).get('name', '')
                                                   or '')
                                    st.markdown("---")
                                    st.markdown("#### 💾 Deseja salvar no banco compartilhado de venues?")
                                    st.caption(
                                        "Outros usuários que abrirem uma atividade **neste mesmo venue** "
                                        "terão o campo configurado automaticamente."
                                    )
                                    with st.form("_form_save_venue"):
                                        _vname_inp = st.text_input(
                                            "Nome do venue:",
                                            value=_def_vname,
                                            placeholder="Ex: Estádio Municipal / CT do Clube",
                                        )
                                        if _vname_inp and _vname_inp.strip() in _vdb_atual:
                                            st.warning(
                                                f"⚠️ Já existe um venue salvo com este nome "
                                                f"(salvo em {_vdb_atual[_vname_inp.strip()].get('saved_at','?')}). "
                                                f"Confirme para sobrescrever."
                                            )
                                        _c1, _c2 = st.columns(2)
                                        with _c1:
                                            _btn_salvar = st.form_submit_button(
                                                "💾 Salvar para todos", type="primary")
                                        with _c2:
                                            _btn_pular  = st.form_submit_button("⏭ Pular")

                                    if _btn_salvar and _vname_inp.strip():
                                        _salvar_venue(_vname_inp.strip(), _novo_cfg)
                                        st.success(
                                            f"✅ Venue **{_vname_inp.strip()}** salvo! "
                                            f"Outros usuários serão beneficiados automaticamente."
                                        )
                                        st.session_state[campo_key]['_from_venues'] = _vname_inp.strip()
                                        st.session_state[campo_key]['_saved_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                                    if _btn_salvar or _btn_pular:
                                        st.rerun()
                            else:
                                st.warning(
                                    "⚠️ Nenhum ponto GPS real (lat/lon) encontrado para este atleta.\n\n"
                                    "Isso pode ocorrer se o sensor não obteve lock GPS durante a sessão."
                                )

                        # ══════════════════════════════════════════════════════
                        # FASE 2 — CAMPO APLICADO + ANÁLISE DE ESFORÇOS
                        # ══════════════════════════════════════════════════════
                        else:
                            cfg = st.session_state[campo_key]
                            _cfg_from_vn  = cfg.get('_from_venues', '')
                            _cfg_saved_at = cfg.get('_saved_at', '')

                            # ── Banner de origem ─────────────────────────────────
                            if _cfg_from_vn:
                                st.success(
                                    f"📦 Campo carregado automaticamente do banco compartilhado: "
                                    f"**{_cfg_from_vn}**"
                                    + (f" · salvo em {_cfg_saved_at}" if _cfg_saved_at else "")
                                )

                            # Botão para reajustar (volta à Fase 1)
                            col_hdr, col_btn, col_mgr = st.columns([4, 1, 1])
                            with col_hdr:
                                st.markdown(
                                    f"### 1️⃣ Campo Aplicado  "
                                    f"<span style='font-size:13px;color:#90CAF9'>"
                                    f"📍 {cfg['lat']:.5f}, {cfg['lon']:.5f} &nbsp; "
                                    f"🧭 {cfg['rot']}° &nbsp; "
                                    f"📏 {cfg['fl']}×{cfg['fw']}m</span>",
                                    unsafe_allow_html=True
                                )
                            with col_btn:
                                if st.button("🔄 Reajustar", key="btn_reajustar"):
                                    del st.session_state[campo_key]
                                    st.rerun()
                            with col_mgr:
                                if st.button("🗄️ Venues", key="btn_mgr_venues"):
                                    st.session_state['_show_venue_mgr'] = not st.session_state.get('_show_venue_mgr', False)

                            # ── Gerenciador de venues ────────────────────────────
                            if st.session_state.get('_show_venue_mgr', False):
                                with st.expander("🗄️ Banco Compartilhado de Venues", expanded=True):
                                    _vdb = _carregar_venues()
                                    if not _vdb:
                                        st.info("Nenhum venue salvo ainda. Configure um campo e clique em 💾 Salvar.")
                                    else:
                                        for _vn, _vc in _vdb.items():
                                            _v1, _v2, _v3, _v4 = st.columns([3, 2, 2, 1])
                                            with _v1:
                                                st.markdown(f"**{_vn}**")
                                            with _v2:
                                                st.caption(f"📏 {_vc.get('fl',105)}×{_vc.get('fw',68)}m · 🧭 {_vc.get('rot',0)}°")
                                            with _v3:
                                                st.caption(f"💾 {_vc.get('saved_at','?')}")
                                            with _v4:
                                                if st.button("🗑️", key=f"_del_vn_{_vn}",
                                                             help=f"Excluir {_vn}"):
                                                    _excluir_venue(_vn)
                                                    st.rerun()
                                    # Salvar config atual com novo nome
                                    st.markdown("---")
                                    st.markdown("**Salvar configuração atual:**")
                                    with st.form("_form_mgr_save"):
                                        _mgr_nome = st.text_input(
                                            "Nome:", value=_venue_name or '',
                                            key="_mgr_vname")
                                        if st.form_submit_button("💾 Salvar / Atualizar", type="primary"):
                                            if _mgr_nome.strip():
                                                _salvar_venue(_mgr_nome.strip(), cfg)
                                                st.success(f"✅ **{_mgr_nome.strip()}** salvo!")
                                                st.rerun()

                            # ── Esforço selecionado (para filtrar GPS no mapa e animação) ──
                            lats_eff, lons_eff, vels_eff, eff_desc = [], [], [], ""
                            _anim_start_ts  = 0.0
                            _anim_end_ts    = 0.0
                            _anim_effort_row = None

                            # ── Arrays de coordenadas (fonte única para ETAPA 2 e ETAPA 3) ──
                            # Construídos aqui para que detecção de esforços e visualização
                            # do campo usem exatamente os mesmos dados — garantindo que
                            # _seg_start_idx dos esforços indexe corretamente xn/yn.
                            xn, yn, vel_raw_campo, acc_raw_campo, ts_pos_campo = [], [], [], [], []
                            _fonte_xy = False
                            for _pm in periodos_mapa_sel:
                                _dc = dados_posicao_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                                if _dc.get('xs') and _dc.get('ys') and _dc.get('vel'):
                                    xn  += list(_dc['xs'])
                                    yn  += list(_dc['ys'])
                                    vel_raw_campo += list(_dc['vel'])
                                    acc_raw_campo += list(_dc.get('acc', [0.0]*len(_dc['xs'])))
                                    ts_pos_campo  += list(_dc.get('ts_pos', []))
                                    _fonte_xy = True
                                elif lats_gps and cfg:
                                    _lats_pm = _dc.get('lats', [])
                                    _lons_pm = _dc.get('lons', [])
                                    if _lats_pm:
                                        _gx, _gy = gps_para_campo_coords(_lats_pm, _lons_pm, cfg)
                                        xn  += _gx
                                        yn  += _gy
                                        vel_raw_campo += _dc.get('vels_gps', [0.0]*len(_gx))
                                        acc_raw_campo += [0.0]*len(_gx)
                            _has_xy  = bool(xn and yn)
                            _has_gps = bool(lats_gps and lons_gps and cfg)

                            # ── Coordenadas de campo + GPS por atleta (ETAPA 2) ──
                            # Necessário para calcular esforços de todos os atletas e
                            # recuperar o segmento correto ao selecionar uma linha.
                            _coords_atletas = {}
                            for _atl in atletas_mapa:
                                _xna, _yna, _vela, _acca, _tsa = [], [], [], [], []
                                _latsa, _lonsa, _vgpsa, _tgpsa = [], [], [], []
                                for _pm in periodos_mapa_sel:
                                    _dca = dados_posicao_por_periodo.get(_pm, {}).get(_atl, {})
                                    if _dca.get('xs') and _dca.get('ys') and _dca.get('vel'):
                                        _xna  += list(_dca['xs'])
                                        _yna  += list(_dca['ys'])
                                        _vela += list(_dca['vel'])
                                        _acca += list(_dca.get('acc', [0.0]*len(_dca['xs'])))
                                        _tsa  += list(_dca.get('ts_pos', []))
                                    elif cfg:
                                        _lp = _dca.get('lats', [])
                                        _lo = _dca.get('lons', [])
                                        if _lp:
                                            _gx, _gy = gps_para_campo_coords(_lp, _lo, cfg)
                                            _xna  += _gx;  _yna  += _gy
                                            _vela += _dca.get('vels_gps', [0.0]*len(_gx))
                                            _acca += [0.0]*len(_gx)
                                    _latsa  += _dca.get('lats', [])
                                    _lonsa  += _dca.get('lons', [])
                                    _vgpsa  += _dca.get('vels_gps', [])
                                    _tgpsa  += _dca.get('ts_gps', [])
                                _coords_atletas[_atl] = dict(
                                    xn=_xna, yn=_yna, vel=_vela, acc=_acca, ts=_tsa,
                                    lats=_latsa, lons=_lonsa, vels_gps=_vgpsa, ts_gps=_tgpsa,
                                )

                            # Atleta do esforço selecionado (atualizado na seleção de linha)
                            _sel_atleta = atleta_mapa

                            # Mapa fixo (atualizado abaixo se houver esforço selecionado)
                            mapa_placeholder = st.empty()

                            st.divider()

                            # ── ETAPA 2: Análise de esforços ─────────────────────
                            st.markdown("### 2️⃣ Análise de Esforços no Campo")

                            _min_dur_vel_s = get_min_dur_vel_s()
                            _min_dur_acc_s = get_min_dur_s()

                            tipo_esf = st.radio(
                                "Tipo de esforço:",
                                ["⚡ Velocidade", "🔁 Aceleração"],
                                horizontal=True, key="tipo_esf_campo"
                            )

                            # ── Detecção de esforços para TODOS os atletas ──────────
                            efforts_df_full = pd.DataFrame()
                            _esf_usou_api = False
                            _frames_esf = []

                            for _atl in atletas_mapa:
                                _c = _coords_atletas[_atl]
                                _df_atl = pd.DataFrame()

                                # ── Tentativa 1: Esforços da API Catapult (fonte primária) ──
                                # Mais precisos — calculados pelos algoritmos oficiais
                                # da Catapult com os mesmos limiares do OpenField.
                                _raw_api: list = []
                                for _pm in periodos_mapa_sel:
                                    if tipo_esf == "⚡ Velocidade":
                                        _raw_api += dados_efforts_vel_por_periodo.get(
                                            _pm, {}).get(_atl, [])
                                    else:
                                        _raw_api += dados_efforts_acc_por_periodo.get(
                                            _pm, {}).get(_atl, [])
                                if _raw_api:
                                    _esf_usou_api = True
                                    _df_atl = (processar_efforts_velocidade(_raw_api)
                                               if tipo_esf == "⚡ Velocidade"
                                               else processar_efforts_aceleracao(_raw_api))

                                # ── Fallback: cálculo local com dados do sensor ──────
                                # Usado apenas se a API não retornou esforços.
                                if _df_atl.empty and _c['xn']:
                                    if tipo_esf == "⚡ Velocidade":
                                        _df_atl = calcular_efforts_velocidade_sensor(
                                            _c['xn'], _c['yn'], _c['vel'], _c['ts'],
                                            min_dur_s=_min_dur_vel_s)
                                    else:
                                        _df_atl = calcular_efforts_aceleracao_sensor(
                                            _c['xn'], _c['yn'], _c['acc'], _c['vel'], _c['ts'],
                                            min_dur_s=_min_dur_acc_s)

                                if not _df_atl.empty:
                                    _df_atl['Atleta'] = _atl
                                    _frames_esf.append(_df_atl)

                            if _frames_esf:
                                efforts_df_full = pd.concat(_frames_esf, ignore_index=True)
                                if '_start_ts' in efforts_df_full.columns and \
                                        efforts_df_full['_start_ts'].sum() > 0:
                                    efforts_df_full = efforts_df_full.sort_values(
                                        '_start_ts').reset_index(drop=True)
                                efforts_df_full['Esforço'] = range(1, len(efforts_df_full) + 1)

                                # ── FEATURE 13: Zona e Direção tática ────────────
                                _fl_tac = float(
                                    st.session_state.get('venue', {}).get('length') or 105)
                                efforts_df_full = enriquecer_esforcos_taticos(
                                    efforts_df_full, _coords_atletas, field_length=_fl_tac)

                            if not efforts_df_full.empty:
                                # Indicador de fonte de dados
                                if _esf_usou_api:
                                    st.caption(
                                        "✅ Esforços calculados pela **API Catapult** (algoritmos oficiais OpenField). "
                                        "Atletas sem dados de API usam cálculo local como fallback."
                                    )
                                else:
                                    st.caption(
                                        "⚙️ Esforços calculados **localmente** a partir dos dados do sensor "
                                        "(API Catapult não retornou esforços para estes atletas)."
                                    )

                                # Filtro de bandas
                                if 'Banda' in efforts_df_full.columns:
                                    bandas_disp = sorted(efforts_df_full['Banda'].dropna().unique())
                                    if bandas_disp:
                                        bandas_sel = st.multiselect(
                                            "Filtrar por bandas:", bandas_disp,
                                            default=bandas_disp, key="bandas_campo"
                                        )
                                        efforts_df_full = efforts_df_full[
                                            efforts_df_full['Banda'].isin(bandas_sel)
                                        ] if bandas_sel else efforts_df_full

                                if efforts_df_full.empty:
                                    st.info("Nenhum esforço encontrado após aplicar os filtros.")
                                else:
                                    # Colunas visíveis — "Atleta" primeiro, depois as demais
                                    _cols_ocultas = {c for c in efforts_df_full.columns
                                                     if c.startswith('_')}
                                    _cols_base = [c for c in efforts_df_full.columns
                                                  if c not in _cols_ocultas and c != 'Atleta']
                                    cols_show = (['Atleta'] + _cols_base
                                                 if 'Atleta' in efforts_df_full.columns
                                                 else _cols_base)
                                    efforts_df_show = efforts_df_full[cols_show]

                                    # Métricas resumidas
                                    mc1, mc2, mc3, mc4 = st.columns(4)
                                    with mc1: st.metric("Total de esforços", len(efforts_df_full))
                                    with mc2: st.metric("Duração total (s)", round(efforts_df_full['Duração (s)'].sum(), 1))
                                    with mc3:
                                        if 'Distância (m)' in efforts_df_full.columns:
                                            st.metric("Distância total (m)", round(efforts_df_full['Distância (m)'].sum(), 1))
                                    with mc4: st.metric("Média % máximo", round(efforts_df_full['% do Máximo'].mean(), 1))

                                    st.markdown(
                                        "**Selecione uma linha** para visualizar esse esforço no mapa de satélite acima. "
                                        "Clique novamente na linha selecionada para desfazer a seleção."
                                    )

                                    # Tabela interativa com seleção de linha
                                    evt = st.dataframe(
                                        efforts_df_show,
                                        use_container_width=True,
                                        height=360,
                                        on_select="rerun",
                                        selection_mode="single-row",
                                        key="tbl_esforcos_campo"
                                    )

                                    sel_rows = evt.selection.rows if evt.selection else []

                                    # Captura esforço selecionado + atleta correto
                                    if sel_rows:
                                        _ae_row = efforts_df_full.iloc[sel_rows[0]]
                                        _anim_start_ts   = float(_ae_row.get('_start_ts') or 0)
                                        _anim_end_ts     = float(_ae_row.get('_end_ts')   or 0)
                                        _anim_effort_row = _ae_row
                                        # Atleta dono do esforço selecionado
                                        _sel_atleta = str(_ae_row.get('Atleta', atleta_mapa))

                                    # Highlight GPS no mapa satélite (usa GPS do atleta correto)
                                    if sel_rows:
                                        sel_idx  = sel_rows[0]
                                        row_full = efforts_df_full.iloc[sel_idx]
                                        start_ts = row_full.get('_start_ts', 0)
                                        end_ts   = row_full.get('_end_ts', 0)
                                        _c_sel   = _coords_atletas.get(_sel_atleta,
                                                       _coords_atletas[atleta_mapa])
                                        _ts_gps_sel   = _c_sel['ts_gps']
                                        _lats_gps_sel = _c_sel['lats']
                                        _lons_gps_sel = _c_sel['lons']
                                        _vels_gps_sel = _c_sel['vels_gps']

                                        if start_ts and end_ts and \
                                                len(_ts_gps_sel) == len(_lats_gps_sel):
                                            filtered = [
                                                (la, lo, ve)
                                                for la, lo, ve, ts in zip(
                                                    _lats_gps_sel, _lons_gps_sel,
                                                    _vels_gps_sel, _ts_gps_sel)
                                                if start_ts <= ts <= end_ts
                                            ]
                                            if filtered:
                                                lats_eff = [p[0] for p in filtered]
                                                lons_eff = [p[1] for p in filtered]
                                                vels_eff = [p[2] for p in filtered]
                                                inicio_str = row_full.get('Início', '')
                                                dur_str    = row_full.get('Duração (s)', '')
                                                eff_desc   = (
                                                    f"Esforço #{row_full['Esforço']} "
                                                    f"({_sel_atleta}) — {inicio_str} — {dur_str}s"
                                                )
                                            else:
                                                st.warning("⚠️ Nenhum ponto GPS encontrado na janela de tempo deste esforço.")
                                        elif not _ts_gps_sel:
                                            st.info("ℹ️ Timestamps GPS não disponíveis.")

                                    # Download da tabela
                                    _atls_str = "_".join(a.replace(' ', '') for a in atletas_mapa)
                                    st.download_button(
                                        "📥 Exportar esforços",
                                        efforts_df_show.to_csv(index=False),
                                        file_name=f"esforcos_{_atls_str}_{_label_periodos.replace(' + ','_')}.csv"
                                    )
                            else:
                                st.info("ℹ️ Sem dados de posição (x/y) para calcular esforços. "
                                        "Os dados de sensor precisam incluir coordenadas de campo.")

                            # Renderiza o mapa fixo (com ou sem esforço destacado)
                            with mapa_placeholder:
                                if lats_gps and lons_gps:
                                    html_fixo = criar_html_campo_fixo(
                                        lats_gps, lons_gps, vels_gps, cfg,
                                        lats_eff=lats_eff or None,
                                        lons_eff=lons_eff or None,
                                        vels_eff=vels_eff or None,
                                        atleta_nome=atleta_mapa,
                                        esforco_desc=eff_desc,
                                        height=580
                                    )
                                    st.components.v1.html(html_fixo, height=580, scrolling=False)
                                else:
                                    st.warning("⚠️ Dados GPS não disponíveis para exibição do mapa.")

                            st.divider()

                            # ── ETAPA 3: Campo bonito + análise avançada ─────────
                            st.markdown("### 3️⃣ Análise de Movimentação no Campo")
                            # xn, yn, vel_raw_campo, etc. já foram construídos antes
                            # de ETAPA 2 para garantir consistência de índices.

                            # ── Computar coords do esforço ANTECIPADO ──────────────
                            # (usado no highlight do fig_campo E na animação abaixo)
                            # Usa arrays do atleta dono do esforço selecionado.
                            _c_eff  = _coords_atletas.get(_sel_atleta, _coords_atletas[atleta_mapa])
                            _xn_eff = _c_eff['xn'] or xn
                            _yn_eff = _c_eff['yn'] or yn
                            _vel_eff = _c_eff['vel'] or vel_raw_campo
                            _acc_eff = _c_eff['acc'] or acc_raw_campo
                            _ts_eff  = _c_eff['ts']  or ts_pos_campo

                            xs_a, ys_a, vel_a, acc_a = [], [], [], []
                            if _anim_effort_row is not None and _xn_eff:
                                # ── Tier 0: índices de segmento armazenados pelas
                                #    funções de sensor — indexam o array do atleta correto.
                                _seg_si = int(_anim_effort_row.get('_seg_start_idx', -1))
                                _seg_ei = int(_anim_effort_row.get('_seg_end_idx',   -1))
                                if 0 <= _seg_si < _seg_ei <= len(_xn_eff):
                                    xs_a  = _xn_eff[_seg_si:_seg_ei]
                                    ys_a  = _yn_eff[_seg_si:_seg_ei]
                                    vel_a = (_vel_eff[_seg_si:_seg_ei]
                                             if len(_vel_eff) == len(_xn_eff)
                                             else [0.0] * (_seg_ei - _seg_si))
                                    acc_a = (_acc_eff[_seg_si:_seg_ei]
                                             if len(_acc_eff) == len(_xn_eff)
                                             else [0.0] * (_seg_ei - _seg_si))

                                # ── Tier 1: matching por timestamp exato
                                # Tenta timestamps de campo primeiro; cai para GPS
                                # (campo x/y convertido de GPS → mesma indexação de ts_gps)
                                if len(xs_a) < 2 and _anim_start_ts > 0 and _anim_end_ts > 0:
                                    _ts1_src = []
                                    if _ts_eff and len(_ts_eff) == len(_xn_eff):
                                        _ts1_src = _ts_eff
                                    elif (_c_eff.get('ts_gps')
                                          and len(_c_eff['ts_gps']) == len(_xn_eff)):
                                        _ts1_src = _c_eff['ts_gps']
                                    if _ts1_src:
                                        _ts_c = np.array(_ts1_src, dtype=float)
                                        _m    = (_ts_c >= _anim_start_ts) & (_ts_c <= _anim_end_ts)
                                        if _m.any():
                                            xs_a  = np.array(_xn_eff)[_m].tolist()
                                            ys_a  = np.array(_yn_eff)[_m].tolist()
                                            vel_a = (np.array(_vel_eff)[_m].tolist()
                                                     if len(_vel_eff) == len(_xn_eff) else [0]*int(_m.sum()))
                                            acc_a = (np.array(_acc_eff)[_m].tolist()
                                                     if len(_acc_eff) == len(_xn_eff) else [0]*int(_m.sum()))

                                # ── Tier 2: fallback proporcional por timestamp
                                # Mesma lógica de source: campo → GPS
                                if len(xs_a) < 2 and _anim_start_ts > 0 and _anim_end_ts > 0:
                                    _ts2_src = (_ts_eff
                                                or _c_eff.get('ts_gps', []))
                                    if _ts2_src:
                                        _ts_min = min(_ts2_src)
                                        _ts_max = max(_ts2_src)
                                        if _ts_max > _ts_min:
                                            _sp = max(0.0, (_anim_start_ts - _ts_min) / (_ts_max - _ts_min))
                                            _ep = min(1.0, (_anim_end_ts   - _ts_min) / (_ts_max - _ts_min))
                                            _si = int(_sp * len(_xn_eff))
                                            _ei = min(len(_xn_eff), int(_ep * len(_xn_eff)) + 1)
                                            if _ei > _si + 1:
                                                xs_a  = _xn_eff[_si:_ei]
                                                ys_a  = _yn_eff[_si:_ei]
                                                vel_a = (_vel_eff[_si:_ei]
                                                         if len(_vel_eff) == len(_xn_eff) else [0]*(_ei-_si))
                                                acc_a = (_acc_eff[_si:_ei]
                                                         if len(_acc_eff) == len(_xn_eff) else [0]*(_ei-_si))

                                # ── Tier 3: fallback por string de início + duração
                                if len(xs_a) < 2:
                                    try:
                                        _dur_s   = float(_anim_effort_row.get('Duração (s)', 5))
                                        _ini_s   = 0.0
                                        _ini_str = str(_anim_effort_row.get('Início', ''))
                                        if ':' in _ini_str:
                                            _parts = _ini_str.replace('s', '').split(':')
                                            if len(_parts) == 3:
                                                _ini_s = int(_parts[0])*3600 + int(_parts[1])*60 + float(_parts[2])
                                            else:
                                                _ini_s = int(_parts[0])*60 + float(_parts[1])
                                        elif _ini_str.endswith('s'):
                                            _ini_s = float(_ini_str[:-1])
                                        _total_s = len(_xn_eff) / 10.0
                                        if _total_s > 0:
                                            _sp = max(0.0, min(1.0, _ini_s / _total_s))
                                            _ep = max(0.0, min(1.0, (_ini_s + _dur_s) / _total_s))
                                            _si = int(_sp * len(_xn_eff))
                                            _ei = min(len(_xn_eff), int(_ep * len(_xn_eff)) + 1)
                                            if _ei > _si + 1:
                                                xs_a  = _xn_eff[_si:_ei]
                                                ys_a  = _yn_eff[_si:_ei]
                                                vel_a = (_vel_eff[_si:_ei]
                                                         if len(_vel_eff) == len(_xn_eff)
                                                         else [0]*(_ei-_si))
                                                acc_a = (_acc_eff[_si:_ei]
                                                         if len(_acc_eff) == len(_xn_eff)
                                                         else [0]*(_ei-_si))
                                    except Exception:
                                        _applog.log_debug_exc()

                            if _fonte_xy:
                                st.caption("📡 Coordenadas x/y Catapult OpenField")
                            elif not _fonte_xy and _has_gps:
                                st.caption("🌍 Coordenadas derivadas do GPS + campo aplicado.")

                            vel_raw = vel_raw_campo
                            acc_raw = acc_raw_campo

                            if _has_xy or _has_gps:

                                if len(xn) > 0:
                                    # ── Linha 1: modo de visualização ────────────────
                                    col_modo, col_ov = st.columns([2, 3])
                                    with col_modo:
                                        modo_viz = st.radio(
                                            "🎨 Modo de visualização",
                                            ["🗺️ Trajetória",
                                             "⚡ Bandas de Velocidade",
                                             "🔁 Bandas de Aceleração"],
                                            key="modo_campo_v3"
                                        )
                                    with col_ov:
                                        st.markdown("**Overlays opcionais:**")
                                        oa, ob = st.columns(2)
                                        with oa:
                                            ov_setas   = st.checkbox("🏹 Setas de direção",    key="ov_setas")
                                            ov_hull    = st.checkbox("📐 Área de atuação",     key="ov_hull")
                                            ov_eventos = st.checkbox("⚽ Eventos Futebol",      key="ov_eventos")
                                        with ob:
                                            ov_tercos  = st.checkbox("📊 Terços do campo",     key="ov_tercos")
                                            ov_grade   = st.checkbox("🔲 Grade de quadrantes", key="ov_grade")
                                            ov_heatmap_comp = st.checkbox("🔥 Heatmap por Período",  key="ov_hcmp")
                                            ov_voronoi_comp = st.checkbox("🔷 Voronoi por Período",  key="ov_vcmp")

                                    # ── Direção de ataque por período ─────────────────
                                    _atk_key = f"attack_dir_{periodo_mapa}"
                                    _atk_opts = [
                                        "— Não definido",
                                        "➡️  Esq → Dir  (ataque para a direita)",
                                        "⬅️  Dir → Esq  (ataque para a esquerda)",
                                    ]
                                    _atk_default_idx = 0
                                    if st.session_state.get(_atk_key) == 'left_to_right':
                                        _atk_default_idx = 1
                                    elif st.session_state.get(_atk_key) == 'right_to_left':
                                        _atk_default_idx = 2
                                    _atk_sel = st.radio(
                                        f"⚽ Direção de ataque — {periodo_mapa}:",
                                        options=_atk_opts,
                                        index=_atk_default_idx,
                                        horizontal=True,
                                        key=f"_atk_radio_{periodo_mapa}",
                                    )
                                    if "➡️" in _atk_sel:
                                        st.session_state[_atk_key] = 'left_to_right'
                                    elif "⬅️" in _atk_sel:
                                        st.session_state[_atk_key] = 'right_to_left'
                                    else:
                                        st.session_state[_atk_key] = None
                                    _attack_dir_val = st.session_state[_atk_key]

                                    # ── Seletores de bandas (dependem do modo) ────────
                                    _bv_ui = _bandas_vel_ativas()
                                    _ba_ui = _bandas_acc_ativas()
                                    bandas_vel_sel = list(_bv_ui.keys())
                                    bandas_acc_sel = list(_ba_ui.keys())

                                    if modo_viz == "⚡ Bandas de Velocidade":
                                        bandas_vel_sel = st.multiselect(
                                            "Bandas de velocidade a exibir:",
                                            options=list(_bv_ui.keys()),
                                            default=list(_bv_ui.keys()),
                                            format_func=lambda k: _bv_ui[k]['label'],
                                            key="ms_bv"
                                        )
                                    elif modo_viz == "🔁 Bandas de Aceleração":
                                        # Duas caixas separadas: Aceleração (A*) e Desaceleração (D*).
                                        _acc_k = [k for k in _ba_ui if str(k).startswith('A')]
                                        _dec_k = [k for k in _ba_ui if str(k).startswith('D')]
                                        _cba, _cbd = st.columns(2)
                                        with _cba:
                                            _sel_a = st.multiselect(
                                                "🚀 Aceleração:",
                                                options=_acc_k,
                                                default=_acc_k,
                                                format_func=lambda k: _ba_ui[k]['label'],
                                                key="ms_ba_pos"
                                            )
                                        with _cbd:
                                            _sel_d = st.multiselect(
                                                "🛑 Desaceleração:",
                                                options=_dec_k,
                                                default=_dec_k,
                                                format_func=lambda k: _ba_ui[k]['label'],
                                                key="ms_ba_neg"
                                            )
                                        bandas_acc_sel = list(_sel_a) + list(_sel_d)

                                    # ── Configuração da grade ─────────────────────────
                                    # ── Seletor de eventos para o campo ──────────────
                                    _dados_ev_campo = {}
                                    _ev_tipos_sel   = []
                                    if ov_eventos:
                                        # Combina eventos de todos os períodos selecionados
                                        _ev_raw = {}
                                        for _pm in periodos_mapa_sel:
                                            _ev_pm = dados_eventos_por_periodo.get(_pm, {}).get(atleta_mapa, {})
                                            for _et, _evlist in _ev_pm.items():
                                                _ev_raw.setdefault(_et, [])
                                                _ev_raw[_et] += _evlist
                                        if _ev_raw:
                                            # Enriquecer com posição no campo agora que temos cfg
                                            _ev_rich = enriquecer_eventos_com_posicao(
                                                _ev_raw,
                                                dados.get('ts_gps', []),
                                                dados.get('lats', []),
                                                dados.get('lons', []),
                                                dados.get('vels_gps', []),
                                                campo_config=cfg,
                                            )
                                            _ev_tipos_disp = [
                                                k for k in _ev_rich if _ev_rich[k]]
                                            if _ev_tipos_disp:
                                                _ev_tipos_sel = st.multiselect(
                                                    "Tipos de evento no campo:",
                                                    options=_ev_tipos_disp,
                                                    default=_ev_tipos_disp,
                                                    format_func=lambda k: FUTEBOL_EVENTS_CONFIG[k]['label'],
                                                    key="ev_campo_tipos"
                                                )
                                                _dados_ev_campo = _ev_rich
                                            else:
                                                st.info("Nenhum evento de futebol carregado para este atleta/período.")
                                        else:
                                            st.info("Nenhum evento de futebol carregado. Recarregue os dados com eventos ativados na sidebar.")

                                    n_cols_g, n_rows_g, zona_sel = 4, 3, None
                                    if ov_grade:
                                        gc1, gc2, gc3 = st.columns(3)
                                        with gc1:
                                            n_cols_g = st.slider("Colunas", 2, 10, 4, key="g_cols")
                                        with gc2:
                                            n_rows_g = st.slider("Linhas",  2, 8,  3, key="g_rows")
                                        with gc3:
                                            zonas_disp = ["— Nenhuma —"] + [
                                                f"{chr(65+r)}{c+1}"
                                                for r in range(n_rows_g)
                                                for c in range(n_cols_g)
                                            ]
                                            zona_sel = st.selectbox(
                                                "🔍 Detalhar zona:", zonas_disp, key="zona_det")
                                            if zona_sel == "— Nenhuma —":
                                                zona_sel = None

                                    # ── Construir figura ──────────────────────────────
                                    _titulo_campo = (
                                        f"📍 {' + '.join(atletas_mapa)} — {_label_periodos}"
                                        if len(atletas_mapa) > 1
                                        else f"📍 {atleta_mapa} — {_label_periodos}"
                                    )
                                    fig_campo = desenhar_campo_futebol_bonito(
                                        title=_titulo_campo,
                                        attack_direction=_attack_dir_val,
                                    )

                                    # ── Plota TODOS os atletas com o mesmo modo ───────
                                    # Usa _coords_atletas para garantir que cada atleta
                                    # receba exatamente o mesmo tratamento visual.
                                    # Prefixo do nome = atleta (visível na legenda).
                                    _n_atls = len(atletas_mapa)
                                    for _atl_i, _atl_viz in enumerate(atletas_mapa):
                                        _cv = _coords_atletas.get(_atl_viz, {})
                                        _xv = _cv.get('xn', [])
                                        _yv = _cv.get('yn', [])
                                        _velv = _cv.get('vel', [])
                                        _accv = _cv.get('acc', [])
                                        if not _xv:
                                            continue
                                        # Prefixo apenas quando há mais de 1 atleta
                                        _pfx = _atl_viz if _n_atls > 1 else ''
                                        if modo_viz == "🗺️ Trajetória":
                                            adicionar_trajetoria_campo(
                                                fig_campo, _xv, _yv, _velv, _atl_viz)
                                        elif modo_viz == "⚡ Bandas de Velocidade" and bandas_vel_sel:
                                            adicionar_pontos_velocidade_bandas(
                                                fig_campo, _xv, _yv, _velv, bandas_vel_sel,
                                                mostrar_setas=ov_setas,
                                                atleta_prefix=_pfx)
                                        elif modo_viz == "🔁 Bandas de Aceleração" and bandas_acc_sel:
                                            adicionar_pontos_aceleracao_bandas(
                                                fig_campo, _xv, _yv, _accv, bandas_acc_sel,
                                                mostrar_setas=ov_setas,
                                                atleta_prefix=_pfx)
                                        # Seta de trajetória por atleta
                                        if ov_setas and modo_viz == "🗺️ Trajetória":
                                            adicionar_setas_direcao(fig_campo, _xv, _yv)

                                    # Seta do esforço destacado (laranja, sobre tudo)
                                    if ov_setas and xs_a and len(xs_a) >= 2:
                                        adicionar_setas_direcao(
                                            fig_campo, xn, yn,
                                            xs_effort=xs_a, ys_effort=ys_a,
                                        )
                                    if ov_hull:
                                        adicionar_convex_hull(fig_campo, xn, yn)
                                    if ov_tercos:
                                        adicionar_tercos_campo(fig_campo, xn, yn)
                                    if ov_grade:
                                        adicionar_grade_quadrantes(
                                            fig_campo, xn, yn, n_cols_g, n_rows_g)
                                    if ov_eventos and _dados_ev_campo and _ev_tipos_sel:
                                        adicionar_eventos_campo(
                                            fig_campo, _dados_ev_campo, _ev_tipos_sel)

                                    # ── Highlight do esforço selecionado ─────────────
                                    if xs_a and len(xs_a) >= 2:
                                        # Halo laranja pulsante (linha mais grossa por baixo)
                                        fig_campo.add_trace(go.Scatter(
                                            x=xs_a, y=ys_a, mode='lines',
                                            line=dict(color='rgba(255,152,0,0.35)', width=14),
                                            name='_halo', showlegend=False, hoverinfo='skip'
                                        ))
                                        # Trajetória do esforço em laranja sólido
                                        fig_campo.add_trace(go.Scatter(
                                            x=xs_a, y=ys_a, mode='lines+markers',
                                            line=dict(color='#FF9800', width=4),
                                            marker=dict(size=3, color='#FF9800'),
                                            name=f"Esforço #{int(_anim_effort_row['Esforço'])}",
                                            showlegend=True, hoverinfo='skip'
                                        ))
                                        # Marcador de início (verde)
                                        fig_campo.add_trace(go.Scatter(
                                            x=[xs_a[0]], y=[ys_a[0]], mode='markers+text',
                                            marker=dict(size=14, color='#4CAF50', symbol='circle',
                                                        line=dict(color='white', width=2)),
                                            text=['▶'], textposition='top center',
                                            textfont=dict(color='white', size=10),
                                            name='Início', showlegend=False, hoverinfo='skip'
                                        ))
                                        # Marcador de fim (vermelho)
                                        fig_campo.add_trace(go.Scatter(
                                            x=[xs_a[-1]], y=[ys_a[-1]], mode='markers+text',
                                            marker=dict(size=14, color='#F44336', symbol='x',
                                                        line=dict(color='white', width=2)),
                                            text=['■'], textposition='top center',
                                            textfont=dict(color='white', size=10),
                                            name='Fim', showlegend=False, hoverinfo='skip'
                                        ))
                                        fig_campo.update_layout(
                                            title=dict(
                                                text=(
                                                    f"📍 {atleta_mapa} — {_label_periodos} &nbsp;|&nbsp; "
                                                    f"🟠 Esforço #{int(_anim_effort_row['Esforço'])} destacado"
                                                )
                                            )
                                        )

                                    st.plotly_chart(fig_campo, use_container_width=True)

                                    # ── FEATURE 12: Heatmap comparativo por período ──
                                    if ov_heatmap_comp and len(periodos_mapa_sel) >= 1:
                                        st.markdown("#### 🔥 Heatmap de Posicionamento por Período")
                                        _fl_hm = float(st.session_state.get('venue', {}).get('length') or 105)
                                        _fw_hm = float(st.session_state.get('venue', {}).get('width')  or 68)
                                        _hm_cols = st.columns(min(len(periodos_mapa_sel), 3))
                                        for _hm_i, _hm_pm in enumerate(periodos_mapa_sel[:3]):
                                            with _hm_cols[_hm_i]:
                                                _hm_xs, _hm_ys = [], []
                                                for _hm_atl in atletas_mapa:
                                                    _hm_d = dados_posicao_por_periodo.get(_hm_pm, {}).get(_hm_atl, {})
                                                    _hm_xs += _hm_d.get('xs', [])
                                                    _hm_ys += _hm_d.get('ys', [])
                                                if _hm_xs:
                                                    _fig_hm = desenhar_campo_futebol_bonito(
                                                        _fl_hm, _fw_hm, title=_hm_pm)
                                                    _xhm = np.array(_hm_xs)
                                                    _yhm = np.array(_hm_ys)
                                                    _H, _xe, _ye = np.histogram2d(
                                                        _xhm, _yhm, bins=[52, 34],
                                                        range=[[0, _fl_hm], [0, _fw_hm]])
                                                    _H = _gf(_H, sigma=2.5)
                                                    _xc = (_xe[:-1] + _xe[1:]) / 2
                                                    _yc = (_ye[:-1] + _ye[1:]) / 2
                                                    _fig_hm.add_trace(go.Heatmap(
                                                        x=_xc, y=_yc, z=_H.T,
                                                        colorscale='Hot', opacity=0.60,
                                                        showscale=False, hoverinfo='skip'))
                                                    _fig_hm.update_layout(
                                                        height=320,
                                                        margin=dict(l=5, r=5, t=35, b=5))
                                                    st.plotly_chart(_fig_hm, use_container_width=True)
                                                else:
                                                    st.info(f"Sem dados para {_hm_pm}")

                                    # ── FEATURE 14: Voronoi comparativo por período ──
                                    if ov_voronoi_comp and len(periodos_mapa_sel) >= 1:
                                        st.markdown("#### 🔷 Voronoi — Raio de Ação por Período")
                                        _vc_cols = st.columns(min(len(periodos_mapa_sel), 3))
                                        for _vc_i, _vc_pm in enumerate(periodos_mapa_sel[:3]):
                                            with _vc_cols[_vc_i]:
                                                _vc_pos = {}
                                                for _vc_atl in atletas_mapa:
                                                    _vc_d = dados_posicao_por_periodo.get(_vc_pm, {}).get(_vc_atl, {})
                                                    if _vc_d.get('xs'):
                                                        _vc_pos[_vc_atl] = {'xs': _vc_d['xs'], 'ys': _vc_d['ys']}
                                                if len(_vc_pos) >= 2:
                                                    _fig_vc = calcular_voronoi_campo(_vc_pos)
                                                    if _fig_vc:
                                                        _fig_vc.update_layout(
                                                            title=dict(text=_vc_pm, font=dict(size=13)),
                                                            height=320,
                                                            margin=dict(l=5, r=5, t=35, b=5),
                                                            showlegend=False)
                                                        st.plotly_chart(_fig_vc, use_container_width=True)
                                                    else:
                                                        st.info(f"Sem dados Voronoi para {_vc_pm}")
                                                else:
                                                    st.info(f"Voronoi precisa de ≥2 atletas com dados em {_vc_pm}")

                                    # ══════════════════════════════════════════════
                                    # ANIMAÇÃO DO ESFORÇO SELECIONADO
                                    # ══════════════════════════════════════════════
                                    if _anim_effort_row is not None:
                                        st.markdown("---")
                                        _vel_max_hdr = _anim_effort_row.get(
                                            'Vel. Máx (km/h)',
                                            _anim_effort_row.get('Acc. Máx (m/s²)', '—')
                                        )
                                        _vel_unit_hdr = (
                                            'km/h' if 'Vel. Máx (km/h)' in _anim_effort_row.index
                                            else 'm/s²'
                                        )
                                        st.markdown(
                                            f"### 🎬 Animação — Esforço **#{int(_anim_effort_row['Esforço'])}** &nbsp;|&nbsp; "
                                            f"Início: **{_anim_effort_row['Início']}** &nbsp;|&nbsp; "
                                            f"Duração: **{_anim_effort_row['Duração (s)']}s** &nbsp;|&nbsp; "
                                            f"Vel. Máx: **{_vel_max_hdr} {_vel_unit_hdr}**"
                                        )

                                        # xs_a / ys_a já foram calculados acima com fallbacks
                                        def _vc_anim(v):
                                            if v < 7:  return '#2196F3'
                                            if v < 14: return '#4CAF50'
                                            if v < 19: return '#FFEB3B'
                                            if v < 24: return '#FF9800'
                                            return '#F44336'

                                        if len(xs_a) >= 2:
                                            _n_pts = len(xs_a)
                                            _step_a = max(1, _n_pts // 80)  # máx 80 frames

                                            # ── Slider de velocidade ──────────────────
                                            _vel_opcoes = {
                                                "🐢 0.25×": 320,
                                                "🚶 0.5×":  160,
                                                "🏃 1× (real)": 80,
                                                "⚡ 2×":    40,
                                                "🚀 4×":    20,
                                            }
                                            _vel_sel = st.select_slider(
                                                "⏩ Velocidade da animação",
                                                options=list(_vel_opcoes.keys()),
                                                value="🏃 1× (real)",
                                                key="anim_speed_slider"
                                            )
                                            _frame_dur = _vel_opcoes[_vel_sel]

                                            # ── Campo base (estático) ─────────────────
                                            _fig_a = desenhar_campo_futebol_bonito(
                                                title=(
                                                    f"Esforço #{int(_anim_effort_row['Esforço'])} | "
                                                    f"{_anim_effort_row['Início']} | "
                                                    f"Vel. Máx {_anim_effort_row['Vel. Máx (km/h)']} km/h"
                                                )
                                            )
                                            _n_base_a = len(_fig_a.data)

                                            # Trajetória completa (fundo desbotado)
                                            _sbg = max(1, len(xn) // 5000)
                                            _fig_a.add_trace(go.Scatter(
                                                x=xn[::_sbg], y=yn[::_sbg], mode='markers',
                                                marker=dict(size=1.5, color='rgba(255,255,255,0.10)'),
                                                name='Trajetória', showlegend=False, hoverinfo='skip'
                                            ))
                                            # Marcadores início / fim do esforço (estáticos)
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[0]], y=[ys_a[0]], mode='markers+text',
                                                marker=dict(size=13, color='#4CAF50', symbol='circle',
                                                            line=dict(color='white', width=2)),
                                                text=['▶'], textposition='top center',
                                                textfont=dict(color='#4CAF50', size=11),
                                                name='Início', showlegend=False, hoverinfo='skip'
                                            ))
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[-1]], y=[ys_a[-1]], mode='markers+text',
                                                marker=dict(size=13, color='#F44336', symbol='x',
                                                            line=dict(color='white', width=2)),
                                                text=['■'], textposition='top center',
                                                textfont=dict(color='#F44336', size=11),
                                                name='Fim', showlegend=False, hoverinfo='skip'
                                            ))
                                            # Traço do esforço (animado)
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[0]], y=[ys_a[0]], mode='lines',
                                                line=dict(color='#FF9800', width=4),
                                                name='Esforço', showlegend=False
                                            ))
                                            # Marcador de posição atual (animado)
                                            _fig_a.add_trace(go.Scatter(
                                                x=[xs_a[0]], y=[ys_a[0]], mode='markers',
                                                marker=dict(
                                                    size=18, color=_vc_anim(vel_a[0]),
                                                    symbol='circle',
                                                    line=dict(color='white', width=3)
                                                ),
                                                name='Posição', showlegend=False
                                            ))

                                            _idx_trace  = _n_base_a + 3  # traço do esforço
                                            _idx_dot    = _n_base_a + 4  # marcador posição

                                            # ── Frames de animação ────────────────────
                                            _frames_a = []
                                            for _k in range(0, _n_pts, _step_a):
                                                _t  = _k * 0.1
                                                _v  = vel_a[_k] if _k < len(vel_a) else 0
                                                _ac = acc_a[_k] if _k < len(acc_a) else 0
                                                _col = _vc_anim(_v)
                                                _frames_a.append(go.Frame(
                                                    data=[
                                                        go.Scatter(
                                                            x=xs_a[:_k+1], y=ys_a[:_k+1],
                                                            mode='lines',
                                                            line=dict(color=_col, width=4),
                                                        ),
                                                        go.Scatter(
                                                            x=[xs_a[_k]], y=[ys_a[_k]],
                                                            mode='markers',
                                                            marker=dict(
                                                                size=18, color=_col, symbol='circle',
                                                                line=dict(color='white', width=3)
                                                            ),
                                                        ),
                                                    ],
                                                    traces=[_idx_trace, _idx_dot],
                                                    name=str(_k),
                                                    layout=go.Layout(title=dict(
                                                        text=(
                                                            f"⏱️ {_t:.1f}s / {_anim_effort_row['Duração (s)']}s"
                                                            f"   |   💨 {_v:.1f} km/h"
                                                            f"   |   ⚡ {_ac:+.2f} m/s²"
                                                            f"   |   📍 x={xs_a[_k]:.1f}m y={ys_a[_k]:.1f}m"
                                                        ),
                                                        font=dict(color='white', size=12)
                                                    ))
                                                ))
                                            _fig_a.frames = _frames_a

                                            # ── Controles Play/Pause + Slider ─────────
                                            _slider_steps = [
                                                {
                                                    'args': [[f.name],
                                                             {'frame': {'duration': 0, 'redraw': True},
                                                              'mode': 'immediate',
                                                              'transition': {'duration': 0}}],
                                                    'label': f"{int(f.name)*0.1:.0f}s",
                                                    'method': 'animate',
                                                }
                                                for f in _frames_a[::max(1, len(_frames_a)//15)]
                                            ]
                                            _fig_a.update_layout(
                                                height=560,
                                                margin=dict(t=55, b=110, l=10, r=10),
                                                updatemenus=[{
                                                    'buttons': [
                                                        {
                                                            'args': [None, {
                                                                'frame': {'duration': _frame_dur, 'redraw': True},
                                                                'fromcurrent': True,
                                                                'transition': {'duration': _frame_dur, 'easing': 'linear'},
                                                            }],
                                                            'label': '▶  Play',
                                                            'method': 'animate',
                                                        },
                                                        {
                                                            'args': [[None], {
                                                                'frame': {'duration': 0, 'redraw': False},
                                                                'mode': 'immediate',
                                                                'transition': {'duration': 0},
                                                            }],
                                                            'label': '⏸  Pause',
                                                            'method': 'animate',
                                                        },
                                                    ],
                                                    'direction': 'left',
                                                    'pad': {'r': 10, 't': 10},
                                                    'showactive': True,
                                                    'type': 'buttons',
                                                    'x': 0.0, 'y': -0.07,
                                                    'bgcolor': '#1565C0',
                                                    'font': {'color': 'white', 'size': 13},
                                                    'bordercolor': '#0D47A1',
                                                }],
                                                sliders=[{
                                                    'active': 0,
                                                    'steps': _slider_steps,
                                                    'currentvalue': {
                                                        'prefix': '⏱️ ',
                                                        'visible': True,
                                                        'font': {'color': 'white', 'size': 12},
                                                    },
                                                    'tickcolor': 'white',
                                                    'font': {'color': 'white'},
                                                    'y': -0.02,
                                                    'len': 0.88,
                                                    'x': 0.1,
                                                    'bgcolor': '#1a1a2e',
                                                    'bordercolor': '#555',
                                                }],
                                            )
                                            st.plotly_chart(_fig_a, use_container_width=True)

                                            # Painel de info abaixo
                                            _ai1, _ai2, _ai3, _ai4, _ai5 = st.columns(5)
                                            _ai1.metric("⏱️ Duração",       f"{_anim_effort_row['Duração (s)']}s")
                                            _ai2.metric("💨 Vel. Máx",      f"{_anim_effort_row['Vel. Máx (km/h)']} km/h")
                                            _ai3.metric("🏁 Vel. Inicial",  f"{_anim_effort_row['Vel. Inicial (km/h)']} km/h")
                                            _ai4.metric("📏 Distância",     f"{_anim_effort_row['Distância (m)']} m")
                                            _ai5.metric("📊 % do Máximo",   f"{_anim_effort_row['% do Máximo']}%")

                                            # ──────────────────────────────────────────────
                                            # PERFIL DE SPRINT — FASES E EFICIÊNCIA (item 6)
                                            # ──────────────────────────────────────────────
                                            st.markdown("---")
                                            st.markdown("#### 🏃 Perfil do Sprint — Fases e Eficiência")
                                            st.caption(
                                                "Decomposição em 3 fases: 🟠 Aceleração (derivada >+0.5 m/s²) · "
                                                "🔴 Pico (≥95% da vel. máxima) · 🔵 Desaceleração (derivada <−0.5 m/s²)"
                                            )

                                            _sp_vel = np.array(vel_a, dtype=float)
                                            _sp_acc = np.array(acc_a, dtype=float)
                                            _sp_t   = np.arange(len(_sp_vel)) * 0.1

                                            # Suavização para detecção de fases
                                            _sp_wl = min(11, len(_sp_vel) - (1 - len(_sp_vel) % 2))
                                            if len(_sp_vel) >= 5:
                                                from scipy.signal import savgol_filter as _svgf
                                                _sp_sm = np.clip(_svgf(_sp_vel, max(5, _sp_wl), 2), 0, None)
                                            else:
                                                _sp_sm = np.clip(_sp_vel, 0, None)

                                            _sp_grad = np.gradient(_sp_sm, 0.1)
                                            _sp_vmax = float(_sp_sm.max()) if _sp_sm.max() > 0 else 1.0

                                            _sp_phases = np.where(
                                                _sp_sm >= _sp_vmax * 0.95, 2,           # pico
                                                np.where(_sp_grad >= 0.5, 1,            # aceleração
                                                np.where(_sp_grad <= -0.5, 3, 2))       # desaceleração / pico
                                            )

                                            _ph_clrs = {1: '#FFA726', 2: '#EF5350', 3: '#42A5F5'}
                                            _ph_lbls = {1: 'Aceleração', 2: 'Pico', 3: 'Desaceleração'}

                                            _fig_sp = go.Figure()
                                            # Linha de velocidade (fundo)
                                            _fig_sp.add_trace(go.Scatter(
                                                x=_sp_t, y=_sp_sm, mode='lines',
                                                line=dict(color='rgba(255,255,255,0.25)', width=1.5),
                                                showlegend=False, hoverinfo='skip'
                                            ))
                                            # Pontos coloridos por fase
                                            for _ph in [1, 2, 3]:
                                                _msk_ph = _sp_phases == _ph
                                                if _msk_ph.any():
                                                    _fig_sp.add_trace(go.Scatter(
                                                        x=_sp_t[_msk_ph],
                                                        y=_sp_sm[_msk_ph],
                                                        mode='markers',
                                                        name=_ph_lbls[_ph],
                                                        marker=dict(size=6,
                                                                    color=_ph_clrs[_ph],
                                                                    line=dict(width=0)),
                                                    ))
                                            # Linha de vel. máxima
                                            _fig_sp.add_hline(
                                                y=_sp_vmax, line_dash='dot',
                                                line_color='rgba(255,255,255,0.5)', line_width=1,
                                                annotation_text=f"Vmáx {_sp_vmax:.1f} km/h",
                                                annotation_font_color='white',
                                                annotation_font_size=10,
                                            )
                                            _fig_sp.update_layout(
                                                xaxis=dict(title='Tempo (s)', color='white',
                                                           gridcolor='rgba(255,255,255,0.1)'),
                                                yaxis=dict(title='Velocidade (km/h)', color='white',
                                                           gridcolor='rgba(255,255,255,0.1)'),
                                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                                font=dict(color='white'), height=270,
                                                legend=dict(font=dict(color='white'),
                                                            orientation='h', y=1.10),
                                                margin=dict(t=10, b=40, l=55, r=10),
                                            )
                                            st.plotly_chart(_fig_sp, use_container_width=True)

                                            # Métricas por fase
                                            _sp_mc = st.columns(3)
                                            _ph_icons = {1: '🟠', 2: '🔴', 3: '🔵'}
                                            for _phi in [1, 2, 3]:
                                                _msk_ph = _sp_phases == _phi
                                                _ph_dur  = float(_msk_ph.sum()) * 0.1
                                                _ph_vavg = (float(_sp_sm[_msk_ph].mean())
                                                            if _msk_ph.any() else 0.0)
                                                _ph_dist = (float((_sp_sm[_msk_ph] / 3.6 / 10).sum())
                                                            if _msk_ph.any() else 0.0)
                                                _ph_gacc = (float(_sp_acc[_msk_ph].mean())
                                                            if _msk_ph.any() and len(_sp_acc) == len(_sp_vel)
                                                            else 0.0)
                                                with _sp_mc[_phi - 1]:
                                                    st.markdown(
                                                        f"**{_ph_icons[_phi]} {_ph_lbls[_phi]}**"
                                                    )
                                                    st.metric("Duração", f"{_ph_dur:.1f} s")
                                                    st.metric("Vel. Média", f"{_ph_vavg:.1f} km/h")
                                                    st.metric("Distância", f"{_ph_dist:.1f} m")
                                                    if abs(_ph_gacc) > 0.01:
                                                        st.metric(
                                                            "Acc. Média",
                                                            f"{_ph_gacc:+.2f} m/s²"
                                                        )
                                        else:
                                            st.warning(
                                                "⚠️ Não foi possível localizar os pontos de campo para este esforço. "
                                                "Verifique se o campo está configurado corretamente na aba **🗺️ Campo de Futebol** "
                                                "e se os timestamps do esforço correspondem aos dados de posição carregados."
                                            )
                                    else:
                                        st.info("💡 Selecione um esforço na tabela acima para ver a **animação no campo**.")

                                    # ── Detalhes de zona selecionada ──────────────────
                                    if zona_sel and ov_grade:
                                        r_idx = ord(zona_sel[0]) - 65
                                        c_idx = int(zona_sel[1:]) - 1
                                        cw_g  = 105.0 / n_cols_g
                                        rh_g  = 68.0  / n_rows_g
                                        st_z  = stats_quadrante(
                                            xn, yn, vel_raw, acc_raw,
                                            c_idx*cw_g, (c_idx+1)*cw_g,
                                            r_idx*rh_g, (r_idx+1)*rh_g)
                                        st.markdown(f"#### 🔍 Zona **{zona_sel}** — Detalhes")
                                        z1,z2,z3,z4,z5,z6 = st.columns(6)
                                        with z1: st.metric("% do Tempo",     f"{st_z['pct']:.1f}%")
                                        with z2: st.metric("Pontos",          st_z['n_pontos'])
                                        with z3: st.metric("Vel. Média",     f"{st_z['vel_media']:.1f} km/h")
                                        with z4: st.metric("Vel. Máx",       f"{st_z['vel_max']:.1f} km/h")
                                        with z5: st.metric("Acc. Média",     f"{st_z['acc_media']:.2f} m/s²")
                                        with z6: st.metric("|Acc| Máx",      f"{st_z['acc_max']:.2f} m/s²")

                                    # ── Estatísticas gerais ───────────────────────────
                                    st.markdown("#### 📊 Estatísticas de Movimentação")
                                    sc1, sc2, sc3, sc4 = st.columns(4)
                                    with sc1:
                                        dist_km = sum(vel_raw) * 0.1 / 3600
                                        st.metric("Distância Total", f"{dist_km:.2f} km")
                                    with sc2:
                                        xr = max(xn) - min(xn)
                                        st.metric("Largura Atuação", f"{xr:.0f} m")
                                    with sc3:
                                        yr = max(yn) - min(yn)
                                        st.metric("Profundidade", f"{yr:.0f} m")
                                    with sc4:
                                        try:
                                            from scipy.spatial import ConvexHull as _CH
                                            _area = _CH(np.column_stack([xn, yn])).volume
                                            st.metric("Área de Atuação", f"{_area:.0f} m²")
                                        except Exception:
                                            st.metric("Área (bbox)", f"{xr*yr:.0f} m²")

                                    # ── Comparação (Períodos ou Atletas) ─────────────
                                    st.markdown("---")
                                    st.markdown("#### 🔄 Comparação")
                                    _cmp_modo = st.radio(
                                        "Comparar:",
                                        ["📅 Períodos", "👤 Atletas"],
                                        horizontal=True, key="cmp_modo"
                                    )

                                    if _cmp_modo == "📅 Períodos":
                                        periodos_lista = list(dados_posicao_por_periodo.keys())
                                        if len(periodos_lista) < 2:
                                            st.info("Carregue pelo menos 2 períodos para usar esta comparação.")
                                        else:
                                            cp1, cp2 = st.columns(2)
                                            with cp1:
                                                per1 = st.selectbox("Período A:", periodos_lista,
                                                                     index=0, key="cmp_p1")
                                            with cp2:
                                                per2 = st.selectbox("Período B:", periodos_lista,
                                                                     index=min(1, len(periodos_lista)-1),
                                                                     key="cmp_p2")

                                            if per1 != per2:
                                                d1c = dados_posicao_por_periodo.get(per1, {}).get(atleta_mapa, {})
                                                d2c = dados_posicao_por_periodo.get(per2, {}).get(atleta_mapa, {})
                                                cfg_ref = st.session_state.get(f"campo_cfg__{atleta_mapa}", cfg)
                                                if d1c.get('xs') and d2c.get('xs'):
                                                    x1c, y1c = list(d1c['xs']), list(d1c['ys'])
                                                    x2c, y2c = list(d2c['xs']), list(d2c['ys'])
                                                elif d1c.get('lats') and d2c.get('lats') and cfg_ref:
                                                    x1c, y1c = gps_para_campo_coords(d1c['lats'], d1c['lons'], cfg_ref)
                                                    x2c, y2c = gps_para_campo_coords(d2c['lats'], d2c['lons'], cfg_ref)
                                                else:
                                                    x1c = x2c = []
                                                if x1c and x2c:
                                                    fig_cmp = desenhar_campo_futebol_bonito(
                                                        title=f"Comparação: {per1} (azul) vs {per2} (rosa)")
                                                    _s1 = max(1, len(x1c)//3000)
                                                    _s2 = max(1, len(x2c)//3000)
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=x1c[::_s1], y=y1c[::_s1], mode='markers', name=per1,
                                                        marker=dict(size=2, color='#00E5FF', opacity=0.5)))
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=x2c[::_s2], y=y2c[::_s2], mode='markers', name=per2,
                                                        marker=dict(size=2, color='#FF4081', opacity=0.5)))
                                                    st.plotly_chart(fig_cmp, use_container_width=True)
                                                else:
                                                    st.info("Dados de posição não disponíveis em um dos períodos.")
                                            else:
                                                st.info("Selecione dois períodos diferentes para comparar.")

                                    else:  # Comparar Atletas
                                        _ats_cmp = _ats_disponiveis if _ats_disponiveis else [atleta_mapa]
                                        if len(_ats_cmp) < 2:
                                            st.info("Carregue pelo menos 2 atletas para usar esta comparação.")
                                        else:
                                            ca1, ca2 = st.columns(2)
                                            with ca1:
                                                atl_A = st.selectbox("Atleta A:", _ats_cmp,
                                                                      index=0, key="cmp_a1")
                                            with ca2:
                                                atl_B = st.selectbox("Atleta B:", _ats_cmp,
                                                                      index=min(1, len(_ats_cmp)-1),
                                                                      key="cmp_a2")

                                            if atl_A != atl_B:
                                                # Combina todos os períodos selecionados para cada atleta
                                                xa, ya, xb, yb = [], [], [], []
                                                for _pm in periodos_mapa_sel:
                                                    _dA = dados_posicao_por_periodo.get(_pm, {}).get(atl_A, {})
                                                    _dB = dados_posicao_por_periodo.get(_pm, {}).get(atl_B, {})
                                                    cfg_ref = st.session_state.get(f"campo_cfg__{atleta_mapa}", cfg)
                                                    if _dA.get('xs'):
                                                        xa += list(_dA['xs']); ya += list(_dA['ys'])
                                                    elif _dA.get('lats') and cfg_ref:
                                                        _gxa, _gya = gps_para_campo_coords(_dA['lats'], _dA['lons'], cfg_ref)
                                                        xa += _gxa; ya += _gya
                                                    if _dB.get('xs'):
                                                        xb += list(_dB['xs']); yb += list(_dB['ys'])
                                                    elif _dB.get('lats') and cfg_ref:
                                                        _gxb, _gyb = gps_para_campo_coords(_dB['lats'], _dB['lons'], cfg_ref)
                                                        xb += _gxb; yb += _gyb

                                                if xa and xb:
                                                    fig_cmp = desenhar_campo_futebol_bonito(
                                                        title=f"Comparação: {atl_A} (azul) vs {atl_B} (rosa)")
                                                    _sa = max(1, len(xa)//3000)
                                                    _sb = max(1, len(xb)//3000)
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=xa[::_sa], y=ya[::_sa], mode='markers', name=atl_A,
                                                        marker=dict(size=2, color='#00E5FF', opacity=0.5)))
                                                    fig_cmp.add_trace(go.Scatter(
                                                        x=xb[::_sb], y=yb[::_sb], mode='markers', name=atl_B,
                                                        marker=dict(size=2, color='#FF4081', opacity=0.5)))
                                                    st.plotly_chart(fig_cmp, use_container_width=True)
                                                else:
                                                    st.info("Dados de posição não disponíveis para um dos atletas selecionados.")
                                            else:
                                                st.info("Selecione dois atletas diferentes para comparar.")

                                    # ══════════════════════════════════════════════
                                    # ══════════════════════════════════════════════
                                    # ANÁLISE — MAPA DE ALTA INTENSIDADE POSICIONAL
                                    # ══════════════════════════════════════════════
                                    st.markdown("---")
                                    with st.expander("🗺️ Mapa de Alta Intensidade Posicional (HSR Zone Map)", expanded=False):
                                        st.markdown(
                                            "Mostra **onde no campo** o atleta realiza ações de alta intensidade. "
                                            "Vai além do heatmap global — filtra apenas os momentos acima do "
                                            "limiar configurado, revelando corredores de sprint, zonas de "
                                            "pressing e rotas de recuperação."
                                        )
                                        # ── Bandas de velocidade da conta Catapult ──────
                                        _hsr_zones_acc = (
                                            st.session_state.get('velocity_zones_account')
                                            or _DEFAULT_VELOCITY_ZONES
                                        )
                                        # Monta lista de opções: apenas bandas com min >= 5 km/h
                                        _hsr_band_opts = []
                                        _hsr_band_map  = {}  # label → min_kmh
                                        for _z in _hsr_zones_acc:
                                            _z_min_kmh = round(float(_z.get('min_ms') or 0) * 3.6, 1)
                                            _z_max_ms  = float(_z.get('max_ms', 9999))
                                            _z_max_str = (
                                                '∞'
                                                if _z_max_ms >= 9000
                                                else f"{round(_z_max_ms*3.6,1)} km/h"
                                            )
                                            if _z_min_kmh >= 5:
                                                _lbl = f"{_z['name']}  ({_z_min_kmh}–{_z_max_str})"
                                                _hsr_band_opts.append(_lbl)
                                                _hsr_band_map[_lbl] = _z_min_kmh

                                        # Default: seleciona automaticamente zonas ≥ 14 km/h
                                        _hsr_default = [
                                            l for l, v in _hsr_band_map.items() if v >= 14.0
                                        ] or (_hsr_band_opts[-2:] if len(_hsr_band_opts) >= 2 else _hsr_band_opts)

                                        _hsr_sels = st.multiselect(
                                            "Bandas HSR (High Speed Running):",
                                            _hsr_band_opts,
                                            default=_hsr_default,
                                            key="hsr_zones_sel",
                                            help=(
                                                "Selecione as bandas que representam alta intensidade. "
                                                "O limiar será o **menor valor** entre as bandas escolhidas."
                                            ),
                                        )

                                        # Limiar = mínimo das bandas selecionadas
                                        if _hsr_sels:
                                            _hsr_vel_thr = min(_hsr_band_map[l] for l in _hsr_sels)
                                        else:
                                            _hsr_vel_thr = 14.0   # fallback

                                        st.caption(
                                            f"🎯 Limiar calculado: **>{_hsr_vel_thr} km/h** "
                                            f"(mínimo das bandas selecionadas)"
                                        )
                                        _hsr_vel_arr = np.array(vel_raw_campo, dtype=float)
                                        _hsr_xn = np.array(xn, dtype=float)
                                        _hsr_yn = np.array(yn, dtype=float)
                                        _hsr_mask = _hsr_vel_arr >= _hsr_vel_thr

                                        _hm1, _hm2, _hm3 = st.columns(3)
                                        _hm1.metric(f"Pontos >{_hsr_vel_thr} km/h", f"{int(_hsr_mask.sum()):,}")
                                        _hm2.metric("% tempo em HSR",
                                                    f"{float(_hsr_mask.mean())*100:.1f}%")
                                        _hm3.metric("Distância HSR (m)",
                                                    f"{float((_hsr_vel_arr[_hsr_mask]/3.6/10).sum()):.0f}")

                                        if _hsr_mask.sum() >= 10:
                                            _hsr_fl = float(st.session_state.get('venue', {}).get('length') or 105)
                                            _hsr_fw = float(st.session_state.get('venue', {}).get('width')  or 68)
                                            _n_lng, _n_lat = 5, 3
                                            _z_lng = ['Def. Prof.', 'Defensivo', 'Meio-Campo',
                                                      'Ofensivo', 'Ataque Prof.']
                                            _z_lat = ['Flanco Esq.', 'Centro', 'Flanco Dir.']

                                            _hsr_xz = _hsr_xn[_hsr_mask]
                                            _hsr_yz = _hsr_yn[_hsr_mask]
                                            _hsr_vz = _hsr_vel_arr[_hsr_mask]

                                            _zone_cnt = np.zeros((_n_lat, _n_lng))
                                            _zone_vel = np.zeros((_n_lat, _n_lng))
                                            for _px, _py, _pv in zip(_hsr_xz, _hsr_yz, _hsr_vz):
                                                _ci = min(int(_px / _hsr_fl * _n_lng), _n_lng - 1)
                                                _ri = min(int(_py / _hsr_fw * _n_lat), _n_lat - 1)
                                                _zone_cnt[_ri, _ci] += 1
                                                _zone_vel[_ri, _ci] += _pv
                                            _zone_vavg = np.where(
                                                _zone_cnt > 0, _zone_vel / _zone_cnt, 0)

                                            _hc1, _hc2 = st.columns([2, 1])
                                            with _hc1:
                                                _fig_hsr = desenhar_campo_futebol_bonito(
                                                    title=(f"HSR >{_hsr_vel_thr} km/h "
                                                           f"— {atleta_mapa}")
                                                )
                                                _sbg_h = max(1, len(_hsr_xn) // 3000)
                                                _fig_hsr.add_trace(go.Scatter(
                                                    x=_hsr_xn[::_sbg_h], y=_hsr_yn[::_sbg_h],
                                                    mode='markers',
                                                    marker=dict(size=1.5,
                                                                color='rgba(255,255,255,0.06)'),
                                                    showlegend=False, hoverinfo='skip'
                                                ))
                                                _sbg_hs = max(1, int(_hsr_mask.sum()) // 3000)
                                                _fig_hsr.add_trace(go.Scatter(
                                                    x=_hsr_xz[::_sbg_hs],
                                                    y=_hsr_yz[::_sbg_hs],
                                                    mode='markers',
                                                    marker=dict(
                                                        size=4,
                                                        color=_hsr_vz[::_sbg_hs],
                                                        colorscale=[
                                                            [0, '#66BB6A'], [0.4, '#FFA726'],
                                                            [0.7, '#EF5350'], [1, '#B71C1C']],
                                                        cmin=float(_hsr_vel_thr),
                                                        cmax=float(min(_hsr_vz.max(), 40)),
                                                        showscale=True,
                                                        colorbar=dict(
                                                            title=dict(text='km/h',
                                                                       font=dict(color='white')),
                                                            tickfont=dict(color='white'), len=0.55,
                                                        ),
                                                        opacity=0.78,
                                                    ),
                                                    showlegend=False,
                                                ))
                                                st.plotly_chart(_fig_hsr, use_container_width=True)

                                            with _hc2:
                                                _fig_zmap = go.Figure(go.Heatmap(
                                                    z=_zone_cnt.tolist(),
                                                    x=_z_lng, y=_z_lat,
                                                    colorscale='YlOrRd',
                                                    text=[[f"{int(_zone_cnt[r, c])}"
                                                           for c in range(_n_lng)]
                                                          for r in range(_n_lat)],
                                                    texttemplate='%{text}',
                                                    textfont=dict(size=10, color='black'),
                                                    showscale=False,
                                                    hovertemplate=(
                                                        '%{x} / %{y}<br>'
                                                        'Pontos HSR: %{z}<extra></extra>'
                                                    ),
                                                ))
                                                _fig_zmap.update_layout(
                                                    title=dict(text='Freq. por Zona',
                                                               font=dict(color='white', size=12)),
                                                    plot_bgcolor='#0e1117',
                                                    paper_bgcolor='#0e1117',
                                                    font=dict(color='white'), height=290,
                                                    margin=dict(t=40, b=60, l=90, r=10),
                                                    xaxis=dict(tickfont=dict(color='white', size=8),
                                                               tickangle=-30),
                                                    yaxis=dict(tickfont=dict(color='white', size=8)),
                                                )
                                                st.plotly_chart(_fig_zmap, use_container_width=True)
                                                _max_ri2, _max_ci2 = np.unravel_index(
                                                    _zone_cnt.argmax(), _zone_cnt.shape)
                                                st.caption(
                                                    f"📍 Zona mais ativa: **{_z_lat[_max_ri2]}** × "
                                                    f"**{_z_lng[_max_ci2]}** "
                                                    f"({int(_zone_cnt[_max_ri2, _max_ci2])} ações)"
                                                )
                                        else:
                                            st.info(
                                                f"Nenhum ponto com velocidade >{_hsr_vel_thr} km/h. "
                                                "Tente reduzir o limiar."
                                            )

                                    # ══════════════════════════════════════════════
                                    # ANÁLISE 5 — DISTÂNCIA ENTRE ATLETAS
                                    # ══════════════════════════════════════════════
                                    if len(atletas_mapa) >= 2:
                                        st.markdown("---")
                                        with st.expander(
                                            f"📏 Distância Entre Atletas "
                                            f"({len(atletas_mapa)} selecionados)",
                                            expanded=False
                                        ):
                                            st.markdown(
                                                "Distância Euclidiana entre os atletas selecionados "
                                                "ao longo do tempo, a partir das coordenadas de campo. "
                                                "Útil para analisar **compactação defensiva**, "
                                                "**pressing em dupla** e **cobertura de espaço**."
                                            )
                                            _cfg_dist = st.session_state.get(
                                                f"campo_cfg__{atleta_mapa}", cfg)
                                            _DCORES = ['#00E5FF','#FF4081','#FFEB3B',
                                                       '#69F0AE','#FF9800','#CE93D8']

                                            # Coleta arrays por atleta
                                            _dist_dados = {}
                                            for _atl_d in atletas_mapa:
                                                _xd, _yd, _tsd = [], [], []
                                                for _pm in periodos_mapa_sel:
                                                    _dd = dados_posicao_por_periodo.get(
                                                        _pm, {}).get(_atl_d, {})
                                                    if _dd.get('xs') and _dd.get('ys'):
                                                        _xd  += list(_dd['xs'])
                                                        _yd  += list(_dd['ys'])
                                                        _tsd += list(_dd.get('ts_pos', []))
                                                    elif _dd.get('lats') and _cfg_dist:
                                                        _gxd, _gyd = gps_para_campo_coords(
                                                            _dd['lats'], _dd['lons'], _cfg_dist)
                                                        _xd  += _gxd
                                                        _yd  += _gyd
                                                        _tsd += [0.0] * len(_gxd)
                                                if _xd:
                                                    _dist_dados[_atl_d] = {
                                                        'x': np.array(_xd, dtype=float),
                                                        'y': np.array(_yd, dtype=float),
                                                        'ts': np.array(_tsd, dtype=float)
                                                    }

                                            _pares_d = [
                                                (atletas_mapa[_ii], atletas_mapa[_jj])
                                                for _ii in range(len(atletas_mapa))
                                                for _jj in range(_ii + 1, len(atletas_mapa))
                                            ]

                                            if len(_dist_dados) >= 2:
                                                _fig_dist = go.Figure()
                                                _resumo_d = []

                                                for _pi_d, (_na, _nb) in enumerate(_pares_d):
                                                    if _na not in _dist_dados or _nb not in _dist_dados:
                                                        continue
                                                    _dA = _dist_dados[_na]
                                                    _dB = _dist_dados[_nb]
                                                    _tsA, _tsB = _dA['ts'], _dB['ts']
                                                    _has_ts_d = (
                                                        _tsA.sum() > 0 and _tsB.sum() > 0)

                                                    if _has_ts_d:
                                                        _tsA_v = _tsA[_tsA > 0]
                                                        _tsB_v = _tsB[_tsB > 0]
                                                        _tc = np.arange(
                                                            max(_tsA_v.min(), _tsB_v.min()),
                                                            min(_tsA_v.max(), _tsB_v.max()),
                                                            0.1
                                                        )
                                                        if len(_tc) < 10:
                                                            _has_ts_d = False

                                                    if _has_ts_d:
                                                        _xA_i = np.interp(_tc, _tsA_v, _dA['x'][_tsA > 0])
                                                        _yA_i = np.interp(_tc, _tsA_v, _dA['y'][_tsA > 0])
                                                        _xB_i = np.interp(_tc, _tsB_v, _dB['x'][_tsB > 0])
                                                        _yB_i = np.interp(_tc, _tsB_v, _dB['y'][_tsB > 0])
                                                        _t_ax = _tc - _tc[0]
                                                    else:
                                                        _nn = min(len(_dA['x']), len(_dB['x']))
                                                        _xA_i = _dA['x'][:_nn]
                                                        _yA_i = _dA['y'][:_nn]
                                                        _xB_i = _dB['x'][:_nn]
                                                        _yB_i = _dB['y'][:_nn]
                                                        _t_ax = np.arange(_nn) / 10.0

                                                    _dist_v = np.sqrt(
                                                        (_xA_i - _xB_i)**2 +
                                                        (_yA_i - _yB_i)**2
                                                    )
                                                    _sbgd = max(1, len(_dist_v) // 3000)
                                                    _lbl = f"{_na.split()[0]} ↔ {_nb.split()[0]}"
                                                    _cor_d = _DCORES[_pi_d % len(_DCORES)]

                                                    _fig_dist.add_trace(go.Scatter(
                                                        x=_t_ax[::_sbgd] / 60,
                                                        y=_dist_v[::_sbgd],
                                                        mode='lines',
                                                        name=_lbl,
                                                        line=dict(color=_cor_d, width=1.5),
                                                        hovertemplate=(
                                                            '%{x:.1f} min | '
                                                            '%{y:.1f} m<extra>'
                                                            + _lbl + '</extra>'
                                                        )
                                                    ))
                                                    _resumo_d.append({
                                                        'Par': _lbl,
                                                        'Dist. Média (m)':
                                                            round(float(_dist_v.mean()), 1),
                                                        'Mediana (m)':
                                                            round(float(np.median(_dist_v)), 1),
                                                        'Mín (m)':
                                                            round(float(_dist_v.min()), 1),
                                                        'Máx (m)':
                                                            round(float(_dist_v.max()), 1),
                                                        '% Tempo < 5 m':
                                                            round(100 * ((_dist_v < 5).sum()
                                                                         / len(_dist_v)), 1),
                                                        '% Tempo < 10 m':
                                                            round(100 * ((_dist_v < 10).sum()
                                                                         / len(_dist_v)), 1),
                                                    })

                                                    # Campo de proximidade (apenas 1º par)
                                                    if _pi_d == 0 and _dist_v.min() < 15:
                                                        _prox_mk = _dist_v < 10
                                                        if _prox_mk.sum() > 5:
                                                            _st_prox = _dist_dados
                                                            _xmid = ((_xA_i[_prox_mk] +
                                                                       _xB_i[_prox_mk]) / 2)
                                                            _ymid = ((_yA_i[_prox_mk] +
                                                                       _yB_i[_prox_mk]) / 2)
                                                            _fig_prox = desenhar_campo_futebol_bonito(
                                                                title=(f"Zonas de Proximidade "
                                                                       f"(< 10 m) — {_lbl}")
                                                            )
                                                            _sbgp = max(1, len(_xA_i) // 3000)
                                                            _fig_prox.add_trace(go.Scatter(
                                                                x=_xA_i[::_sbgp],
                                                                y=_yA_i[::_sbgp],
                                                                mode='markers',
                                                                marker=dict(size=1.5,
                                                                    color='rgba(0,229,255,0.18)'),
                                                                name=_na, showlegend=True
                                                            ))
                                                            _fig_prox.add_trace(go.Scatter(
                                                                x=_xB_i[::_sbgp],
                                                                y=_yB_i[::_sbgp],
                                                                mode='markers',
                                                                marker=dict(size=1.5,
                                                                    color='rgba(255,64,129,0.18)'),
                                                                name=_nb, showlegend=True
                                                            ))
                                                            _sbgpr = max(1, len(_xmid) // 1500)
                                                            _fig_prox.add_trace(go.Scatter(
                                                                x=_xmid[::_sbgpr],
                                                                y=_ymid[::_sbgpr],
                                                                mode='markers',
                                                                marker=dict(
                                                                    size=5,
                                                                    color='rgba(255,82,82,0.65)',
                                                                    symbol='circle'
                                                                ),
                                                                name='Proximidade < 10 m',
                                                                showlegend=True
                                                            ))
                                                            _fig_prox.update_layout(height=430)
                                                            _prox_fig_final = _fig_prox

                                                # Linhas de referência
                                                _fig_dist.add_hline(
                                                    y=5, line_dash='dot',
                                                    line_color='rgba(255,235,59,0.45)',
                                                    annotation_text='5 m',
                                                    annotation_font_color='#FFEB3B',
                                                    annotation_position='right'
                                                )
                                                _fig_dist.add_hline(
                                                    y=10, line_dash='dot',
                                                    line_color='rgba(255,152,0,0.45)',
                                                    annotation_text='10 m',
                                                    annotation_font_color='#FF9800',
                                                    annotation_position='right'
                                                )
                                                _fig_dist.update_layout(
                                                    title=dict(
                                                        text="Distância Entre Atletas ao Longo do Tempo",
                                                        font=dict(color='white', size=14)
                                                    ),
                                                    xaxis=dict(
                                                        title='Tempo (min)',
                                                        gridcolor='rgba(255,255,255,0.1)',
                                                        color='white'
                                                    ),
                                                    yaxis=dict(
                                                        title='Distância (m)',
                                                        gridcolor='rgba(255,255,255,0.1)',
                                                        color='white'
                                                    ),
                                                    paper_bgcolor='rgba(0,0,0,0)',
                                                    plot_bgcolor='rgba(20,20,30,0.85)',
                                                    legend=dict(
                                                        font=dict(color='white'),
                                                        bgcolor='rgba(0,0,0,0.45)'
                                                    ),
                                                    height=390
                                                )
                                                st.plotly_chart(_fig_dist,
                                                                use_container_width=True)

                                                if _resumo_d:
                                                    st.markdown("**📋 Resumo por Par**")
                                                    st.dataframe(
                                                        pd.DataFrame(_resumo_d).set_index('Par'),
                                                        use_container_width=True
                                                    )

                                                if 'prox_fig_final' in dir() and _prox_fig_final:
                                                    st.markdown("---")
                                                    st.plotly_chart(_prox_fig_final,
                                                                    use_container_width=True)
                                                    st.caption(
                                                        "🔴 Pontos vermelhos = posição média "
                                                        "entre os atletas quando distância < 10 m"
                                                    )
                                            else:
                                                st.warning(
                                                    "Dados de posição insuficientes "
                                                    "para os atletas selecionados.")
                                    else:
                                        pass  # análise de distância requer 2+ atletas

                                else:
                                    st.error("❌ Não foi possível calcular as coordenadas de campo.")
                            else:
                                st.warning(
                                    "⚠️ Sem dados de posição disponíveis para análise de campo.\n\n"
                                    "**Para ativar esta seção:**\n"
                                    "- Certifique-se de que o campo GPS está **aplicado** "
                                    "(clique ✅ Aplicar Campo no mapa acima)\n"
                                    "- Verifique se o sensor GPS estava ativo durante a sessão"
                                )

                    elif periodos_mapa_sel:
                        st.info("Selecione pelo menos um atleta para continuar.")

                else:
                    st.info("Dados de posição não disponíveis. Verifique se o sensor GPS estava ativo durante a sessão.")

                # ── FEATURE 5: Anotações Táticas ─────────────────────────────
                st.markdown("---")
                st.markdown("### 📌 Anotações Táticas")
                _ann_api = st.session_state.get('api')
                _ann_act_id = st.session_state.get('activity_id')

                if _ann_api and _ann_act_id:
                    # Carregar anotações existentes
                    if st.button("🔄 Carregar anotações", key="btn_load_ann"):
                        try:
                            _ann_raw = _ann_api.get_activity_annotations(_ann_act_id)
                            st.session_state['annotations'] = _ann_raw
                        except Exception as _ae:
                            st.error(f"Erro: {_ae}")

                    _annotations = st.session_state.get('annotations')
                    if _annotations:
                        try:
                            _ann_list = (_annotations if isinstance(_annotations, list)
                                         else _annotations.get('data', []))
                            if _ann_list:
                                st.markdown("#### Anotações existentes")
                                # Timeline visual
                                _ann_ts0 = min(
                                    (float(a.get('start_time') or 0) for a in _ann_list if a.get('start_time')),
                                    default=0
                                )
                                _fig_ann = go.Figure()
                                for _ia, _ann in enumerate(_ann_list):
                                    _t0 = float(_ann.get('start_time') or 0) - _ann_ts0
                                    _t1 = float(_ann.get('end_time', _ann.get('start_time', 0))) - _ann_ts0
                                    _ann_name = _ann.get('name', f'Anotação {_ia+1}')
                                    _fig_ann.add_shape(
                                        type='rect',
                                        x0=_t0, x1=max(_t1, _t0 + 1),
                                        y0=_ia, y1=_ia + 0.8,
                                        fillcolor='#2196F3', opacity=0.6,
                                        line_width=0,
                                    )
                                    _fig_ann.add_annotation(
                                        x=(_t0 + _t1) / 2, y=_ia + 0.4,
                                        text=_ann_name, showarrow=False,
                                        font=dict(color='white', size=10),
                                    )
                                _fig_ann.update_layout(
                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(title='Tempo (s)', gridcolor='#333'),
                                    yaxis=dict(visible=False),
                                    height=max(120, len(_ann_list) * 40 + 40),
                                    margin=dict(t=10, b=30),
                                )
                                st.plotly_chart(_fig_ann, use_container_width=True)

                                # Lista com botão de deletar
                                for _ann in _ann_list:
                                    _ann_id = _ann.get('id', '')
                                    _ann_nm = _ann.get('name', '—')
                                    _c1, _c2 = st.columns([4, 1])
                                    _c1.write(f"**{_ann_nm}** (id: {_ann_id})")
                                    if _c2.button("🗑️", key=f"del_ann_{_ann_id}"):
                                        try:
                                            _del_ok = _ann_api.delete_annotation(_ann_id)
                                            if _del_ok:
                                                st.success("Anotação removida.")
                                                st.session_state.pop('annotations', None)
                                                st.rerun()
                                        except Exception as _de:
                                            st.error(f"Erro: {_de}")
                        except Exception as _ep:
                            st.error(f"Erro ao processar anotações: {_ep}")

                    # Criar nova anotação
                    with st.expander("➕ Nova Anotação", expanded=False):
                        _ann_c1, _ann_c2 = st.columns(2)
                        with _ann_c1:
                            _ann_new_name = st.text_input("Nome:", key="ann_new_name")
                            _ann_new_start = st.number_input("Início (s Unix):", value=0, key="ann_new_start")
                        with _ann_c2:
                            _ann_new_type = st.selectbox(
                                "Tipo:", ["phase", "event", "highlight"],
                                key="ann_new_type"
                            )
                            _ann_new_end = st.number_input("Fim (s Unix):", value=0, key="ann_new_end")
                        if st.button("✅ Criar Anotação", key="btn_create_ann"):
                            if _ann_new_name:
                                try:
                                    _cr_resp = _ann_api.create_activity_annotation(
                                        _ann_act_id, _ann_new_name,
                                        _ann_new_start, _ann_new_end,
                                        _ann_new_type,
                                    )
                                    if _cr_resp:
                                        st.success("Anotação criada!")
                                        st.session_state.pop('annotations', None)
                                    else:
                                        st.warning("Endpoint de anotações pode não estar disponível nesta licença.")
                                except Exception as _cae:
                                    st.error(f"Erro: {_cae}")
                            else:
                                st.warning("Informe um nome para a anotação.")
                else:
                    st.info("Carregue os dados de uma atividade para usar anotações.")

            # ==================== ABA 2: ESFORÇOS AO LONGO DO TEMPO ====================
            with abas[1]:
                st.subheader("⏱️ Esforços ao Longo do Tempo")
                
                if dados_sensor_por_atleta_por_periodo:
                    _ESF_TODOS = "🔀 Todos os períodos (combinado)"
                    _esf_opcoes = [_ESF_TODOS] + list(dados_sensor_por_atleta_por_periodo.keys())
                    periodo_escolhido = st.selectbox("Selecione o período:", _esf_opcoes, key="periodo_esforcos")
                    _esf_modo_todos = (periodo_escolhido == _ESF_TODOS)

                    if _esf_modo_todos:
                        _esf_ats_set = set()
                        for _pv in dados_sensor_por_atleta_por_periodo.values():
                            _esf_ats_set.update(_pv.keys())
                        _esf_atletas = sorted(_esf_ats_set)
                    else:
                        _esf_atletas = list(dados_sensor_por_atleta_por_periodo.get(periodo_escolhido, {}).keys())

                    if _esf_atletas:
                        atleta_escolhido = st.selectbox("Selecione o atleta:", _esf_atletas, key="atleta_esforcos")
                        if _esf_modo_todos:
                            sensor_points = []
                            for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                sensor_points += _pv2.get(atleta_escolhido, [])
                            st.caption(
                                f"📊 Combinando **{len(dados_sensor_por_atleta_por_periodo)} períodos** "
                                f"→ {len(sensor_points):,} amostras para **{atleta_escolhido}**."
                            )
                        else:
                            sensor_points = dados_sensor_por_atleta_por_periodo[periodo_escolhido].get(atleta_escolhido, [])
                        
                        col_config1, col_config2 = st.columns(2)
                        with col_config1:
                            mostrar_tendencia = st.checkbox("Mostrar linha de tendência", value=True)
                            window_size = st.slider("Janela de suavização:", 5, 101, 31, step=2)
                        with col_config2:
                            usar_filtro = st.checkbox("Filtrar por intensidade", value=False)
                            if usar_filtro:
                                intensidade_min = st.slider("Intensidade mínima (km/h):", 0.0, 30.0, 5.0, 0.5)
                            else:
                                intensidade_min = None
                        
                        _dur_s_aba3 = get_min_dur_s()
                        st.caption(
                            f"⚙️ Duração mínima de acc/dec: **{_dur_s_aba3:.1f} s** "
                            f"({max(1, round(_dur_s_aba3 * _SENSOR_HZ))} frames) — "
                            "ajuste na sidebar."
                        )
                        st.markdown("### 🏃‍♂️ Velocidade ao Longo do Tempo")
                        fig_vel = criar_grafico_velocidade_tempo(sensor_points, atleta_escolhido, window_size, mostrar_tendencia, intensidade_min)
                        if fig_vel:
                            st.plotly_chart(fig_vel, use_container_width=True)
                        
                        st.markdown("### 🔄 Aceleração ao Longo do Tempo")
                        fig_acc = criar_grafico_aceleracao_tempo(sensor_points, atleta_escolhido, window_size, mostrar_tendencia, intensidade_min if usar_filtro else None)
                        if fig_acc:
                            st.plotly_chart(fig_acc, use_container_width=True)
                        
                        # ── Metabolic Power chart (FEATURE 3) ───────────────────
                        st.markdown("---")
                        _mp_vals_esf = [
                            float(p['mp'])
                            for p in sensor_points
                            if p.get('mp') and float(p.get('mp') or 0) > 0
                        ]
                        if _mp_vals_esf:
                            st.markdown("### ⚡ Potência Metabólica ao Longo do Tempo")
                            _ts0_mp = float(sensor_points[0].get('ts') or 0)
                            _mp_ts = [
                                (float(p.get('ts') or 0) - _ts0_mp)
                                for p in sensor_points
                                if p.get('mp') and float(p.get('mp') or 0) > 0
                            ]
                            _fig_mp = go.Figure()
                            _fig_mp.add_trace(go.Scatter(
                                x=_mp_ts, y=_mp_vals_esf,
                                mode='lines',
                                name='MP (W/kg)',
                                line=dict(color='#FF9800', width=1.5),
                                hovertemplate='%{x:.0f}s — %{y:.1f} W/kg<extra></extra>',
                            ))
                            _fig_mp.add_hline(y=20, line=dict(color='#F44336', dash='dash'),
                                              annotation_text='20 W/kg', annotation_font=dict(color='#F44336', size=9))
                            _fig_mp.add_hline(y=25, line=dict(color='#9C27B0', dash='dash'),
                                              annotation_text='25 W/kg', annotation_font=dict(color='#9C27B0', size=9))
                            _fig_mp.update_layout(
                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                font=dict(color='white'),
                                xaxis=dict(title='Tempo (s)', gridcolor='#333'),
                                yaxis=dict(title='MP (W/kg)', gridcolor='#333'),
                                height=320,
                            )
                            st.plotly_chart(_fig_mp, use_container_width=True)
                            _mp_above20 = sum(1 for v in _mp_vals_esf if v > 20) / max(1, len(_mp_vals_esf)) * 100
                            _mp_above25_s = sum(1 for v in _mp_vals_esf if v > 25) * 0.1
                            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                            _mc1.metric("MP Médio (W/kg)", f"{float(np.mean(_mp_vals_esf)):.1f}")
                            _mc2.metric("MP Máx (W/kg)", f"{float(np.max(_mp_vals_esf)):.1f}")
                            _mc3.metric("MP > 20 W/kg (%)", f"{_mp_above20:.1f}%")
                            _mc4.metric("Tempo > 25 W/kg (s)", f"{_mp_above25_s:.0f}s")

                else:
                    st.info("Dados de sensor não disponíveis")


                # ── FEATURE 8: PDF Export — REMOVIDO ──────────────────────────

            # ==================== ABA 3: JANELAS TEMPORAIS MÓVEIS ====================
            with abas[2]:
                st.subheader("📊 Análise de Intensidade - Janelas Temporais (Rolling Window)")
                st.markdown("""
                Esta análise usa uma **janela deslizante** (*rolling window*): a janela avança **10 s por passo**
                ao longo de toda a sessão, capturando o pico real de intensidade independentemente de onde ele
                começa — sem o erro de corte das janelas discretas fixas.
                No modo **combinado**, todos os períodos são encadeados em uma linha do tempo contínua
                (sem o gap do intervalo).
                """)

                if dados_sensor_por_atleta_por_periodo:

                    # ── Seletor de modo ────────────────────────────────────────────
                    _jan_modo = st.radio(
                        "Modo de análise:",
                        ["🔵 Individual", "🟡 Por Posição", "🔴 Time Completo"],
                        horizontal=True, key="jan_modo_analise",
                        help="Individual: análise detalhada de um atleta · "
                             "Por Posição: curva média por grupo tático · "
                             "Time Completo: heatmap de intensidade de todos os atletas"
                    )
                    st.divider()

                    # ── Controles comuns: período + janela + métrica ───────────────
                    _JAN_TODOS = "🔀 Todos os períodos (combinado)"
                    _jan_opcoes = [_JAN_TODOS] + list(dados_sensor_por_atleta_por_periodo.keys())
                    periodo_janela = st.selectbox("Selecione o período:", _jan_opcoes, key="periodo_janela")
                    _jan_modo_todos = (periodo_janela == _JAN_TODOS)

                    if _jan_modo_todos:
                        _jan_ats_set = set()
                        for _pv in dados_sensor_por_atleta_por_periodo.values():
                            _jan_ats_set.update(_pv.keys())
                        _jan_atletas = sorted(_jan_ats_set)
                    else:
                        _jan_atletas = list(dados_sensor_por_atleta_por_periodo.get(periodo_janela, {}).keys())

                    bandas_vel = [3, 4, 5, 6, 7, 8]
                    bandas_acc = [1, 2, 3]
                    _col_w, _col_m, _col_extra = st.columns(3)
                    with _col_w:
                        window_minutes = st.slider(
                            "Janela temporal (minutos):",
                            min_value=0.5, max_value=10.0, value=1.0, step=0.5,
                            key="jan_window"
                        )
                    _MET_ACOES = '💥 Ações Acel/Desacel'
                    _MET_VEL_BANDAS = '🏃 Velocidade (bandas)'
                    with _col_m:
                        tipo_metrica = st.selectbox(
                            "Métrica:",
                            ['Distância', 'PlayerLoad', _MET_VEL_BANDAS, _MET_ACOES],
                            key="jan_metrica",
                            help="🏃 Velocidade (bandas) = distância (m) percorrida nas bandas "
                                 "de velocidade selecionadas, por janela (igual ao WCS). "
                                 "💥 Ações Acel/Desacel = nº de ações de acel/desacel nas "
                                 "bandas, detectadas no sinal de aceleração do sensor (mesma "
                                 "fonte da aba Neuromuscular)."
                        )
                    # ── Bandas de VELOCIDADE (para a métrica '🏃 Velocidade (bandas)') ──
                    sel_vel_bands = []   # dicts {min,max} absolutos (km/h)
                    sel_vel_pct = []     # (P9) faixas relativas (fração da Vmáx)
                    jan_vel_rel = False
                    with _col_extra:
                        if tipo_metrica == _MET_VEL_BANDAS:
                            jan_vel_rel = st.checkbox(
                                "% da Vmáx individual", value=False, key="jan_vel_rel",
                                help="(P9) Em vez das bandas absolutas da conta, usa faixas "
                                     "relativas à velocidade máxima de CADA atleta "
                                     "(histórico da conta, com fallback no pico da sessão).")
                            if jan_vel_rel:
                                _rel_pick_j = st.multiselect(
                                    "🎚️ Faixas (% da Vmáx)",
                                    list(_REL_VEL_BANDAS.keys()),
                                    default=[_k for _k in _REL_VEL_BANDAS
                                             if _REL_VEL_BANDAS[_k][0] >= 0.7],
                                    key="jan_vel_rel_bands",
                                    help="A distância (m) é acumulada quando a velocidade "
                                         "está dentro da faixa relativa do próprio atleta.")
                                sel_vel_pct = [_REL_VEL_BANDAS[_s] for _s in _rel_pick_j]
                                if not sel_vel_pct:
                                    st.info("Selecione ao menos uma faixa.")
                            else:
                                _bv_act_j = _bandas_vel_ativas()
                                _bv_lbl_j = {}
                                for _bk, _bd in _bv_act_j.items():
                                    _mx = float(_bd.get('max', 9999))
                                    _faixa = (f">{_fmt_num_banda(_bd.get('min', 0))}"
                                              if _mx >= 9000
                                              else f"{_fmt_num_banda(_bd.get('min', 0))}-"
                                                   f"{_fmt_num_banda(_mx)}")
                                    _bv_lbl_j[f"B{_bk} — {_faixa} km/h"] = _bk
                                _bv_pick_j = st.multiselect(
                                    "🎚️ Bandas de velocidade",
                                    list(_bv_lbl_j.keys()),
                                    default=list(_bv_lbl_j.keys()),
                                    key="jan_vel_bands",
                                    help="A distância (m) é acumulada apenas enquanto a velocidade "
                                         "está dentro das bandas selecionadas — igual ao WCS."
                                )
                                sel_vel_bands = [_bv_act_j[_bv_lbl_j[_s]] for _s in _bv_pick_j]
                                if not sel_vel_bands:
                                    st.info("Selecione ao menos uma banda de velocidade.")

                    _unidade_jan = {
                        'Distância': 'm/min', 'PlayerLoad': 'PL/min',
                        _MET_VEL_BANDAS: 'm', _MET_ACOES: 'ações',
                    }.get(tipo_metrica, '')

                    # ── Bandas de AÇÕES (efforts) — duas caixas accel/decel ────────
                    # Mesma seleção da aba WCS para que os valores batam.
                    sel_acc_bands = []
                    sel_acc_boxes = set()   # caixas Gen2 (1..8) selecionadas
                    if tipo_metrica == _MET_ACOES:
                        _ba_act_j = _bandas_acc_ativas()
                        _acc_lbl_j = {_ba_act_j[k]['label']: k
                                      for k in _ba_act_j if str(k).startswith('A')}
                        _dec_lbl_j = {_ba_act_j[k]['label']: k
                                      for k in _ba_act_j if str(k).startswith('D')}
                        _cja, _cjd = st.columns(2)
                        with _cja:
                            _acc_pick_j = st.multiselect(
                                "🚀 Aceleração",
                                list(_acc_lbl_j.keys()),
                                default=list(_acc_lbl_j.keys()),
                                key="jan_acc_bands_pos",
                                help="Ações de aceleração (Gen2Acceleration · caixas 6,7,8)."
                            )
                        with _cjd:
                            _dec_pick_j = st.multiselect(
                                "🛑 Desaceleração",
                                list(_dec_lbl_j.keys()),
                                default=list(_dec_lbl_j.keys()),
                                key="jan_acc_bands_neg",
                                help="Ações de desaceleração (Gen2Acceleration · caixas 3,2,1)."
                            )
                        sel_acc_bands = (
                            [_ba_act_j[_acc_lbl_j[_s]] for _s in _acc_pick_j]
                            + [_ba_act_j[_dec_lbl_j[_s]] for _s in _dec_pick_j]
                        )
                        # Caixas Gen2 oficiais das bandas escolhidas (A1..A3→6,7,8;
                        # D1..D3→3,2,1). Contar pela caixa é robusto — não depende do
                        # valor médio do effort cair no intervalo derivado.
                        sel_acc_boxes = (
                            {_ACC_KEY_TO_NUM[_acc_lbl_j[_s]] for _s in _acc_pick_j
                             if _acc_lbl_j[_s] in _ACC_KEY_TO_NUM}
                            | {_ACC_KEY_TO_NUM[_dec_lbl_j[_s]] for _s in _dec_pick_j
                               if _dec_lbl_j[_s] in _ACC_KEY_TO_NUM}
                        )
                        st.caption(
                            "Conta o **nº de ações** de acel/desacel nas bandas selecionadas "
                            "por janela — detectadas no **sinal de aceleração do sensor** "
                            "(mesma fonte da aba **Neuromuscular**), sustentadas pela duração "
                            "mínima da sidebar. O pior minuto é a janela com mais ações. "
                            "Quando a API traz *acceleration_efforts* (modo por período), usa "
                            "a contagem oficial por caixa Gen2."
                        )
                        if not sel_acc_bands:
                            st.info("Selecione ao menos uma banda de aceleração ou desaceleração.")

                    # Detecta Hz uma vez — (P1) canônico: metrics.estimate_hz
                    # (nativo usa ts_pos, GPS-only usa ts_gps).
                    _hz_jan = 10.0
                    if dados_posicao_por_periodo:
                        _series_hz = []
                        for _pn_hz in list(dados_posicao_por_periodo.keys())[:5]:
                            for _an_hz in list(dados_posicao_por_periodo[_pn_hz].values())[:5]:
                                _tss_hz = _an_hz.get('ts_pos', []) or _an_hz.get('ts_gps', [])
                                if len(_tss_hz) > 20:
                                    _series_hz.append(_tss_hz)
                        _hz_jan = _mtr.estimate_hz(_series_hz, default=10.0)

                    # ── Helper: AÇÕES (efforts) — espelha exatamente o cálculo WCS ─
                    def _calc_rolling_acoes(_atl):
                        """
                        Conta ações (efforts) de acel/desacel por janela rolante,
                        usando EXATAMENTE a mesma lógica da aba 'Pior Cenário (WCS)':
                        timeline = ts_pos concatenado dos períodos (de
                        dados_posicao_por_periodo); cada effort soma +1 na amostra
                        mais próxima do seu start_time; soma rolante de N amostras.
                        Assim o pico individual coincide com o WCS.
                        Retorna (tempos_min, valores) — valores = nº de ações na janela.
                        """
                        # Períodos reais (para efforts oficiais da API, quando existirem).
                        _ps = ([k for k in dados_posicao_por_periodo
                                if k != _CHAVE_COMBINADO]
                               if _jan_modo_todos else [periodo_janela])

                        # ── FONTE: sinal de aceleração do SENSOR (nativo 'a', 10 Hz) ──
                        # Mesma fonte da aba Neuromuscular (que conta 200+ ações). Antes
                        # usava a trajetória GPS + derivação por velocidade, que zerava
                        # em dispositivos só-GPS. O sinal do sensor é sempre confiável.
                        if _jan_modo_todos:
                            _sp = combinar_periodos_continuo(
                                dados_sensor_por_atleta_por_periodo, _atl)
                        else:
                            _sp = dados_sensor_por_atleta_por_periodo.get(
                                periodo_janela, {}).get(_atl, [])
                        if not _sp or not sel_acc_bands:
                            return [], []

                        _Hz = float(_SENSOR_HZ)                 # sensor uniforme 10 Hz
                        _n = max(2, int(window_minutes * 60 * _Hz))
                        _nsp = len(_sp)
                        if _nsp < _n:
                            return [], []

                        # Sinal de aceleração (m/s²): nativo 'a'; se ausente (só-GPS sem
                        # IMU), deriva de dv/dt da velocidade do próprio sensor.
                        _acc_sig = [float(_p.get('a') or 0.0) for _p in _sp]
                        _ts_raw = [float(_p.get('ts') or 0.0) for _p in _sp]
                        st.session_state['_prov_jan_acoes'] = 'sensor'   # (P4)
                        if not any(abs(_a) > 0.05 for _a in _acc_sig):
                            _vel_sig = [float(_p.get('v') or 0.0) * 3.6 for _p in _sp]
                            _acc_sig = acc_series_from_vel(_vel_sig, _ts_raw, _Hz)
                            st.session_state['_prov_jan_acoes'] = 'derivado'
                            _diag_log('Janelas', f"{_atl}: sem aceleração nativa — "
                                                 "ações derivadas por dv/dt da velocidade")

                        _sv = [0.0] * _nsp
                        _ts_np = np.array(_ts_raw, dtype=float)
                        _ts_unix_ok = (_ts_np.size > 0
                                       and float(np.median(_ts_np)) > 1e6)
                        _has_api_eff = any(
                            len(dados_efforts_acc_por_periodo
                                .get(_pn, {}).get(_atl, []) or []) > 0
                            for _pn in _ps)
                        # Efforts oficiais da API (contagem por caixa Gen2) só quando há
                        # timestamps Unix — o modo combinado reescreve os ts, então usa o
                        # sinal. Ambos contam AÇÕES; o sinal garante que nunca zere.
                        if _has_api_eff and _ts_unix_ok and not _jan_modo_todos:
                            st.session_state['_prov_jan_acoes'] = 'efforts'   # (P4)
                            for _pn in _ps:
                                for _ef in (dados_efforts_acc_por_periodo
                                            .get(_pn, {}).get(_atl, []) or []):
                                    try:
                                        _bx = int(round(float(_ef.get('band'))))
                                        _stt = float(_ef.get('start_time') or 0)
                                    except (TypeError, ValueError):
                                        continue
                                    if _stt <= 0 or _bx not in sel_acc_boxes:
                                        continue
                                    _idx = int(np.argmin(np.abs(_ts_np - _stt)))
                                    if 0 <= _idx < _nsp:
                                        _sv[_idx] += 1.0
                        else:
                            _idxs_acc = detectar_acoes_acc_idx(
                                _acc_sig, sel_acc_bands, freq_hz=_Hz)
                            for _ix in _idxs_acc:
                                if 0 <= _ix < _nsp:
                                    _sv[_ix] += 1.0

                        # Soma rolante — (P1) canônico: metrics.rolling_sum
                        _roll = _mtr.rolling_sum(_sv, _n)
                        if not _roll:
                            return [], []

                        # Downsample (~1 ponto/s) garantindo o pico real (= WCS)
                        _stepd = max(1, int(round(_Hz)))
                        _t_out, _v_out = [], []
                        for _i in range(0, len(_roll), _stepd):
                            _t_out.append(_i / (_Hz * 60.0))
                            _v_out.append(float(_roll[_i]))
                        _imax = int(np.argmax(_roll))
                        if _imax % _stepd != 0:
                            import bisect as _bis
                            _tp = _imax / (_Hz * 60.0)
                            _pos = _bis.bisect_left(_t_out, _tp)
                            _t_out.insert(_pos, _tp)
                            _v_out.insert(_pos, float(_roll[_imax]))
                        return _t_out, _v_out

                    # ── Helper: VELOCIDADE (bandas) — distância (m) nas bandas por janela ─
                    def _calc_rolling_vel_bandas(_atl):
                        """Distância (m) percorrida nas bandas de velocidade selecionadas,
                        por janela rolante — MESMA lógica do WCS '🏃 Velocidade (bandas)':
                        soma v/(3.6·Hz) por amostra quando a velocidade cai nas bandas."""
                        if _jan_modo_todos:
                            _ps = [k for k in dados_posicao_por_periodo
                                   if k != _CHAVE_COMBINADO]
                        else:
                            _ps = [periodo_janela]
                        _wts, _wv = [], []
                        for _pn in _ps:
                            _da = dados_posicao_por_periodo.get(_pn, {}).get(_atl, {})
                            # Nativo: ts_pos/vel · GPS-only: ts_gps/vels_gps
                            _ts = _da.get('ts_pos', []) or _da.get('ts_gps', [])
                            _vl = _da.get('vel', []) or _da.get('vels_gps', [])   # km/h
                            _nn = min(len(_ts), len(_vl))
                            if _nn == 0:
                                continue
                            _wts += list(_ts[:_nn])
                            _wv += list(_vl[:_nn])
                        _Hz = _hz_jan
                        _n = max(2, int(window_minutes * 60 * _Hz))
                        if len(_wv) < _n:
                            return [], []
                        if jan_vel_rel:
                            # (P9) faixas relativas à Vmáx individual do atleta
                            _vmx = _vmax_individual_kmh(_atl, _wv)
                            if _vmx <= 0 or not sel_vel_pct:
                                _diag_log('Janelas', f"{_atl}: sem Vmáx individual "
                                                     "confiável — excluído do modo % Vmáx")
                                return [], []
                            _faixas_v = [(lo * _vmx, hi * _vmx) for lo, hi in sel_vel_pct]
                        else:
                            _faixas_v = [(float(b.get('min', 0)), float(b.get('max', 9999)))
                                         for b in sel_vel_bands]
                        if not _faixas_v:
                            return [], []

                        # (P1) canônico: metrics.per_sample_distance_in_bands + rolling_sum
                        _sv = _mtr.per_sample_distance_in_bands(_wv, _faixas_v, _Hz)
                        _roll = _mtr.rolling_sum(_sv, _n)
                        if not _roll:
                            return [], []
                        _stepd = max(1, int(round(_Hz)))
                        _t_out, _v_out = [], []
                        for _i in range(0, len(_roll), _stepd):
                            _t_out.append(_i / (_Hz * 60.0))
                            _v_out.append(float(_roll[_i]))
                        _imax = int(np.argmax(_roll))
                        if _imax % _stepd != 0:
                            import bisect as _bis
                            _tp = _imax / (_Hz * 60.0)
                            _pos = _bis.bisect_left(_t_out, _tp)
                            _t_out.insert(_pos, _tp)
                            _v_out.insert(_pos, float(_roll[_imax]))
                        return _t_out, _v_out

                    # ── Helper: rolling window para um atleta ──────────────────────
                    def _calc_rolling(_atl):
                        """Retorna (tempos_min, valores) para o atleta e configuração atual."""
                        if tipo_metrica == _MET_ACOES:
                            return _calc_rolling_acoes(_atl)
                        if tipo_metrica == _MET_VEL_BANDAS:
                            return _calc_rolling_vel_bandas(_atl)
                        # ── Sensor helper (reutilizado no fallback de Distância) ────
                        def _get_sp():
                            if _jan_modo_todos:
                                return combinar_periodos_continuo(
                                    dados_sensor_por_atleta_por_periodo, _atl)
                            return dados_sensor_por_atleta_por_periodo.get(
                                periodo_janela, {}).get(_atl, [])

                        if tipo_metrica == 'Distância':
                            # 1ª tentativa: GPS field-filtered (mais preciso)
                            if dados_posicao_por_periodo:
                                if _jan_modo_todos:
                                    _vj, _tj = combinar_periodos_continuo_posicao(
                                        dados_posicao_por_periodo, _atl)
                                else:
                                    _daj = dados_posicao_por_periodo.get(
                                        periodo_janela, {}).get(_atl, {})
                                    _vj = _daj.get('vel', [])
                                    _tj = _daj.get('ts_pos', [])
                                if _vj:
                                    _res_gps = calcular_distancia_janelas_por_vel_posicao(
                                        _vj, _tj, window_minutes, _hz_jan)
                                    if _res_gps[0]:   # GPS devolveu dados válidos
                                        st.session_state['_prov_jan_dist'] = 'gps'   # (P4)
                                        return _res_gps
                            # 2ª tentativa: sensor IMU (fallback)
                            _sp = _get_sp()
                            if _sp:
                                st.session_state['_prov_jan_dist'] = 'sensor'   # (P4)
                                return calcular_distancia_janelas_discretas_10s(
                                    _sp, window_minutes)
                            return [], []

                        # Métricas baseadas em sensor
                        _sp = _get_sp()
                        if not _sp:
                            return [], []
                        if tipo_metrica == 'PlayerLoad':
                            return calcular_janelas_discretas_10s(_sp, window_minutes, 'pl', None)
                        if tipo_metrica == 'Velocidade':
                            return calcular_janelas_discretas_10s(
                                _sp, window_minutes, 'v', {'velocity_bands': bandas_vel})
                        if tipo_metrica == 'Aceleração':
                            return calcular_janelas_discretas_10s(
                                _sp, window_minutes, 'a', {'acceleration_bands': bandas_acc})
                        return [], []

                    # ── Helper: posição do atleta ──────────────────────────────────
                    def _get_pos_atl(_atl):
                        for _pd in dados_posicao_por_periodo.values():
                            if _atl in _pd:
                                return _pd[_atl].get('posicao') or 'Outro'
                        return 'Outro'

                    # ── Paleta por posição ─────────────────────────────────────────
                    _POS_COR = {
                        'Goleiro': '#5dade2',      'Zagueiro': '#2ecc71',
                        'Lateral': '#1abc9c',      'Volante': '#f39c12',
                        'Meia': '#e67e22',         'Meia-atacante': '#d4ac0d',
                        'Atacante': '#e74c3c',     'Extremo': '#c0392b',
                        'Centroavante': '#9b59b6', 'Outro': '#95a5a6',
                    }
                    _POS_RGBA_FILL = {
                        k: f"rgba({int(v[1:3],16)},{int(v[3:5],16)},{int(v[5:7],16)},0.13)"
                        for k, v in _POS_COR.items()
                    }

                    # ══════════════════════════════════════════════════════════════
                    # MODO 1 — INDIVIDUAL
                    # ══════════════════════════════════════════════════════════════
                    if _jan_modo == "🔵 Individual":
                        if _jan_atletas:
                            atleta_janela = st.selectbox(
                                "Selecione o atleta:", _jan_atletas, key="atleta_janela")

                            if _jan_modo_todos:
                                sensor_points = combinar_periodos_continuo(
                                    dados_sensor_por_atleta_por_periodo, atleta_janela)
                                st.caption(
                                    f"📊 Combinando **{len(dados_sensor_por_atleta_por_periodo)} períodos** "
                                    f"em linha do tempo contínua → {len(sensor_points):,} amostras "
                                    f"para **{atleta_janela}**.")
                            else:
                                sensor_points = dados_sensor_por_atleta_por_periodo[
                                    periodo_janela].get(atleta_janela, [])

                            if sensor_points:
                                _dur_s_aba4 = get_min_dur_s()
                                st.caption(
                                    f"⚙️ Duração mínima de acc/dec: **{_dur_s_aba4:.1f} s** "
                                    f"({max(1, round(_dur_s_aba4 * _SENSOR_HZ))} frames) — "
                                    "ajuste na sidebar.")

                                if _jan_modo_todos and dados_posicao_por_periodo:
                                    _period_boundaries = obter_limites_periodos_posicao(
                                        dados_posicao_por_periodo, atleta_janela)
                                elif not _jan_modo_todos:
                                    _period_boundaries = [(0.0, float('inf'), periodo_janela)]
                                else:
                                    _period_boundaries = None

                                with st.spinner("Calculando janelas temporais..."):
                                    _tj, _vj = _calc_rolling(atleta_janela)
                                    if _tj:
                                        # (P4) selo de proveniência da métrica exibida
                                        if tipo_metrica == _MET_ACOES:
                                            _selo_fonte(st.session_state.get(
                                                '_prov_jan_acoes', 'sensor'))
                                        elif tipo_metrica == 'Distância':
                                            _selo_fonte(st.session_state.get(
                                                '_prov_jan_dist', 'gps'))
                                        elif tipo_metrica == _MET_VEL_BANDAS:
                                            _vmx_ref = (_vmax_individual_kmh(atleta_janela)
                                                        if jan_vel_rel else 0.0)
                                            _selo_fonte('gps',
                                                        (f"faixas em **% da Vmáx individual** "
                                                         f"(ref.: {_vmx_ref:.1f} km/h)"
                                                         if jan_vel_rel and _vmx_ref > 0 else
                                                         ("faixas em % da Vmáx individual"
                                                          if jan_vel_rel else "")))
                                        elif tipo_metrica == 'PlayerLoad':
                                            _selo_fonte('sensor')
                                        exibir_resultados_janela(
                                            _tj, _vj, tipo_metrica, atleta_janela,
                                            window_minutes, _unidade_jan, _period_boundaries)
                                    else:
                                        st.warning("Dados insuficientes para calcular janelas.")

                                if not st.session_state.get('modo_apresentacao'):
                                    st.markdown(REFERENCIAS["janelas"])
                            else:
                                st.info("Dados de sensor não disponíveis")
                        else:
                            st.info("Selecione um atleta para análise")

                    # ══════════════════════════════════════════════════════════════
                    # MODO 2 — POR POSIÇÃO
                    # ══════════════════════════════════════════════════════════════
                    elif _jan_modo == "🟡 Por Posição":
                        if not _jan_atletas:
                            st.info("Sem atletas disponíveis.")
                        else:
                            # Agrupa atletas por posição
                            _pos_atls: dict = {}
                            for _a in _jan_atletas:
                                _p = _get_pos_atl(_a)
                                _pos_atls.setdefault(_p, []).append(_a)

                            _pos_sel = st.multiselect(
                                "Posições a comparar:",
                                options=sorted(_pos_atls.keys()),
                                default=sorted(_pos_atls.keys()),
                                key="jan_pos_sel"
                            )
                            if not _pos_sel:
                                st.info("Selecione ao menos uma posição.")
                            else:
                                with st.spinner("Calculando por posição..."):
                                    import plotly.graph_objects as _go_pos

                                    # Rolling window para cada atleta das posições selecionadas
                                    _atl_res: dict = {}
                                    for _ps in _pos_sel:
                                        for _a in _pos_atls.get(_ps, []):
                                            if _a not in _atl_res:
                                                _t_a, _v_a = _calc_rolling(_a)
                                                if _t_a and _v_a:
                                                    _atl_res[_a] = (
                                                        np.array(_t_a), np.array(_v_a))

                                    if not _atl_res:
                                        st.warning(
                                            "Sem dados suficientes para as posições selecionadas.")
                                    else:
                                        _max_t_pos = max(
                                            float(_t[-1]) + window_minutes
                                            for _t, _ in _atl_res.values())
                                        _t_grid_pos = np.arange(0, _max_t_pos + 1/60, 1/60)

                                        fig_pos = _go_pos.Figure()
                                        _pos_summ: list = []

                                        for _ps in _pos_sel:
                                            _atls_ps = [
                                                _a for _a in _pos_atls.get(_ps, [])
                                                if _a in _atl_res]
                                            if not _atls_ps:
                                                continue

                                            _v_mat = np.array([
                                                np.interp(_t_grid_pos,
                                                          _atl_res[_a][0],
                                                          _atl_res[_a][1],
                                                          left=np.nan, right=np.nan)
                                                for _a in _atls_ps
                                            ])
                                            _v_mean = np.nanmean(_v_mat, axis=0)
                                            _cor = _POS_COR.get(_ps, '#95a5a6')
                                            _fil = _POS_RGBA_FILL.get(_ps,
                                                                       'rgba(149,165,166,0.13)')

                                            # Área ± std (se mais de 1 atleta)
                                            if len(_atls_ps) > 1:
                                                _v_std = np.nanstd(_v_mat, axis=0)
                                                _yu = _v_mean + _v_std
                                                _yd = _v_mean - _v_std
                                                fig_pos.add_trace(_go_pos.Scatter(
                                                    x=np.concatenate(
                                                        [_t_grid_pos, _t_grid_pos[::-1]]),
                                                    y=np.concatenate([_yu, _yd[::-1]]),
                                                    fill='toself', fillcolor=_fil,
                                                    line=dict(color='rgba(0,0,0,0)'),
                                                    showlegend=False, hoverinfo='skip',
                                                ))

                                            fig_pos.add_trace(_go_pos.Scatter(
                                                x=_t_grid_pos, y=_v_mean,
                                                name=f"{_ps} (n={len(_atls_ps)})",
                                                line=dict(color=_cor, width=2.5),
                                                mode='lines',
                                                hovertemplate=(
                                                    f"<b>{_ps}</b><br>"
                                                    "Tempo: %{x:.1f} min<br>"
                                                    f"{tipo_metrica}: %{{y:.1f}} {_unidade_jan}"
                                                    "<extra></extra>"
                                                ),
                                            ))

                                            _pk = round(float(np.nanmax(_v_mean)), 1)
                                            _av = round(float(np.nanmean(_v_mean)), 1)
                                            _pos_summ.append({
                                                'Posição': _ps,
                                                'N Atletas': len(_atls_ps),
                                                f'Pico Médio ({_unidade_jan})': _pk,
                                                f'Média Geral ({_unidade_jan})': _av,
                                            })

                                        # Limiares globais
                                        _all_v_pos = np.concatenate(
                                            [v for _, v in _atl_res.values()])
                                        _gmax_pos = float(np.nanmax(_all_v_pos))
                                        _la_pos = round(_gmax_pos * 0.90, 1)
                                        _lm_pos = round(_gmax_pos * 0.75, 1)
                                        fig_pos.add_hline(
                                            y=_la_pos, line_dash='dash',
                                            line_color='rgba(239,68,68,0.50)',
                                            annotation_text=f"Alta ≥{_la_pos} {_unidade_jan}",
                                            annotation_position="right")
                                        fig_pos.add_hline(
                                            y=_lm_pos, line_dash='dot',
                                            line_color='rgba(245,158,11,0.50)',
                                            annotation_text=f"Média-Alta ≥{_lm_pos} {_unidade_jan}",
                                            annotation_position="right")

                                        fig_pos.update_layout(
                                            title=dict(
                                                text=(f"Intensidade de {tipo_metrica} por Posição"
                                                      f" — Rolling Window {window_minutes} min"),
                                                font=dict(color='white', size=14)),
                                            xaxis=dict(
                                                title='Tempo (minutos)',
                                                color='rgba(255,255,255,0.6)',
                                                gridcolor='rgba(255,255,255,0.07)'),
                                            yaxis=dict(
                                                title=f'{tipo_metrica} ({_unidade_jan})',
                                                color='rgba(255,255,255,0.6)',
                                                gridcolor='rgba(255,255,255,0.07)'),
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            legend=dict(
                                                font=dict(color='white'),
                                                bgcolor='rgba(0,0,0,0)'),
                                            hovermode='x unified',
                                            height=440,
                                        )
                                        st.plotly_chart(fig_pos, use_container_width=True)

                                        if _pos_summ:
                                            st.markdown("##### 📊 Resumo por Posição")
                                            _df_pos = (
                                                pd.DataFrame(_pos_summ)
                                                .sort_values(f'Pico Médio ({_unidade_jan})',
                                                             ascending=False)
                                                .reset_index(drop=True)
                                            )
                                            st.dataframe(
                                                _df_pos, use_container_width=True,
                                                hide_index=True,
                                                height=38 * len(_df_pos) + 60)

                    # ══════════════════════════════════════════════════════════════
                    # MODO 3 — TIME COMPLETO
                    # ══════════════════════════════════════════════════════════════
                    elif _jan_modo == "🔴 Time Completo":
                        if not _jan_atletas:
                            st.info("Sem atletas disponíveis.")
                        else:
                            with st.spinner(
                                    f"Calculando rolling window para {len(_jan_atletas)} atletas..."):
                                import plotly.graph_objects as _go_tm

                                # Rolling window para cada atleta
                                _team_res: dict = {}
                                for _a in _jan_atletas:
                                    _ta, _va = _calc_rolling(_a)
                                    if _ta and _va:
                                        _team_res[_a] = (
                                            np.array(_ta), np.array(_va))

                                if not _team_res:
                                    st.warning("Dados insuficientes.")
                                else:
                                    # ── Offset absoluto por atleta ─────────────────
                                    # Lógica: cada "período" é uma actividade gravada
                                    # separadamente. Atletas que continuam no jogo
                                    # aparecem em múltiplos períodos consecutivos.
                                    # O substituto entra apenas a partir do período
                                    # em que foi inserido.
                                    #
                                    # Usamos SEMPRE duração acumulada via sensor IMU
                                    # (ts_last − ts_first dentro do mesmo período),
                                    # que funciona tanto para ts relativo (0-based)
                                    # quanto Unix — porque o intervalo interno cancela
                                    # qualquer origem absoluta.
                                    # NÃO usamos Unix ts direto: combinar_periodos_
                                    # continuo remove os intervalos entre períodos,
                                    # enquanto Unix ts os inclui → dessincronização.
                                    _period_order_tm = (
                                        list(dados_sensor_por_atleta_por_periodo.keys())
                                        if _jan_modo_todos
                                        else [periodo_janela]
                                    )

                                    # ── Timestamps absolutos de cada período ──────────
                                    # Sensor IMU (ts + cs/100) usa Unix absoluto →
                                    # períodos sobrepostos têm ts que se intersectam.
                                    def _period_abs_ts_tm(_pnm: str):
                                        """(first_ts, last_ts) em segundos Unix via sensor IMU."""
                                        _mn, _mx = None, None
                                        for _spl in dados_sensor_por_atleta_por_periodo.get(
                                                _pnm, {}).values():
                                            for _pp in _spl:
                                                _tt = (float(_pp.get('ts') or 0)
                                                       + float(_pp.get('cs') or 0) / 100.0)
                                                if _tt <= 0:
                                                    continue
                                                if _mn is None or _tt < _mn: _mn = _tt
                                                if _mx is None or _tt > _mx: _mx = _tt
                                        return (_mn or 0.0, _mx or 0.0)

                                    _period_abs_tm = {
                                        _pn: _period_abs_ts_tm(_pn)
                                        for _pn in _period_order_tm}

                                    # ── Posição de cada período no tempo de jogo ───────
                                    # Lógica de sobreposição:
                                    #   "2tempo" registra os 10 atletas que continuam
                                    #   por TODO o 2º tempo (ex: 45-95 min).
                                    #   "2tempo1" registra apenas o substituto, que
                                    #   COMEÇA DENTRO do "2tempo" (ex: 65-95 min).
                                    #   → "2tempo1" é sub-período de "2tempo", não
                                    #   um período sequencial após ele.
                                    #
                                    # Detecção: se first_ts(P) está ENTRE first_ts(Q) e
                                    # last_ts(Q) de outro período Q já ativo → P é
                                    # sub-período de Q.
                                    # Offset de P = match_start(Q) + (first_ts(P) - first_ts(Q)) / 60
                                    #
                                    # Períodos principais (sem sobreposição) acumulam
                                    # _cum_min_tm normalmente.
                                    _sorted_by_ts_tm = sorted(
                                        _period_order_tm,
                                        key=lambda _p: _period_abs_tm[_p][0])

                                    _period_start_min_tm: dict = {}
                                    _cum_min_tm = 0.0        # só cresce em períodos principais
                                    _active_mains_tm: list = []  # (nm, ft, lt, match_start)

                                    for _pn_s in _sorted_by_ts_tm:
                                        _ft_s, _lt_s = _period_abs_tm[_pn_s]
                                        _dur_s = (
                                            (_lt_s - _ft_s) / 60.0
                                            if _lt_s > _ft_s else 0.0)

                                        # Descarta períodos principais já encerrados
                                        _active_mains_tm = [
                                            _m for _m in _active_mains_tm
                                            if _m[2] > _ft_s]

                                        # Este período começa DENTRO de algum ativo?
                                        _par_tm = next(
                                            (_m for _m in _active_mains_tm
                                             if _ft_s > _m[1] and _ft_s < _m[2]),
                                            None)

                                        if _par_tm is None:
                                            # Período principal — sem sobreposição
                                            _period_start_min_tm[_pn_s] = _cum_min_tm
                                            _active_mains_tm.append(
                                                (_pn_s, _ft_s, _lt_s, _cum_min_tm))
                                            _cum_min_tm += _dur_s
                                        else:
                                            # Sub-período — entra no meio do pai
                                            _, _par_ft, _, _par_ms = _par_tm
                                            _period_start_min_tm[_pn_s] = (
                                                _par_ms + (_ft_s - _par_ft) / 60.0)

                                    # Ordem cronológica de períodos (por match-time start)
                                    _sorted_period_order_tm = sorted(
                                        _period_order_tm,
                                        key=lambda _p: _period_start_min_tm.get(_p, 0.0))

                                    def _atl_offset_min(_atl_nm: str) -> float:
                                        """Offset = match-time do 1º período (ordem ts) com dados do atleta."""
                                        for _pn_ao in _sorted_by_ts_tm:
                                            if (dados_posicao_por_periodo.get(
                                                    _pn_ao, {}).get(_atl_nm, {}).get('vel')
                                                    or dados_sensor_por_atleta_por_periodo.get(
                                                    _pn_ao, {}).get(_atl_nm)):
                                                return _period_start_min_tm.get(_pn_ao, 0.0)
                                        return 0.0

                                    # Ordena atletas por posição → nome
                                    _atls_ord = sorted(
                                        _team_res.keys(),
                                        key=lambda _a: (_get_pos_atl(_a), _a))

                                    # Pré-calcula offsets uma vez
                                    _offsets_tm = {
                                        _a: _atl_offset_min(_a) for _a in _atls_ord}

                                    # Grade temporal comum no tempo absoluto do jogo
                                    _max_t_tm = max(
                                        float(_t[-1]) + _offsets_tm[_a] + window_minutes
                                        for _a, (_t, _) in _team_res.items()
                                        if _a in _offsets_tm)
                                    _tg = np.arange(0, _max_t_tm + 1/60, 1/60)

                                    _z_mat:   list = []   # normalizado (% máx coletivo)
                                    _raw_mat: list = []   # bruto (para média)
                                    _y_lbl:   list = []

                                    # 1ª passagem — séries brutas no tempo absoluto
                                    for _a in _atls_ord:
                                        _ta, _va = _team_res[_a]
                                        # ← desloca para a linha do tempo real do jogo
                                        _ta_abs = _ta + _offsets_tm[_a]
                                        _vr = np.interp(_tg, _ta_abs, _va,
                                                        left=np.nan, right=np.nan)
                                        _raw_mat.append(_vr)
                                        _y_lbl.append(
                                            f"{_a}  [{_get_pos_atl(_a)}]")

                                    # Máximo coletivo — referência única de todo o time
                                    _col_max = max(
                                        (float(np.nanmax(_vr))
                                         for _vr in _raw_mat
                                         if not np.all(np.isnan(_vr))),
                                        default=1.0,
                                    )
                                    if _col_max <= 0:
                                        _col_max = 1.0

                                    # 2ª passagem — normaliza pelo máximo coletivo
                                    # Linhas inteiramente NaN → mantém NaN (não zeros),
                                    # assim "fora de campo" fica transparente no heatmap
                                    # e distingue-se visualmente de 0% de intensidade.
                                    for _vr in _raw_mat:
                                        if not np.all(np.isnan(_vr)):
                                            # Posições NaN dentro da linha ficam NaN;
                                            # posições ativas → 0–100 % do máx coletivo
                                            _vn = np.where(
                                                np.isnan(_vr),
                                                np.nan,
                                                _vr / _col_max * 100,
                                            )
                                        else:
                                            _vn = np.full_like(_vr, np.nan)
                                        _z_mat.append(_vn)

                                    # ── Pré-calcula bandas de período ──────────────
                                    # Usa _sorted_period_order_tm (ordem por match-time)
                                    # para que as bandas sejam sempre sequenciais.
                                    # Cada banda vai do início do período até o início
                                    # do próximo período (ou até o fim do jogo).
                                    _period_bands_tm = []
                                    for _i_pb, _pn_pb in enumerate(_sorted_period_order_tm):
                                        _ps_pb = _period_start_min_tm[_pn_pb]
                                        _pe_pb = (
                                            _period_start_min_tm[
                                                _sorted_period_order_tm[_i_pb + 1]]
                                            if _i_pb + 1 < len(_sorted_period_order_tm)
                                            else _cum_min_tm
                                        )
                                        _period_bands_tm.append((_pn_pb, _ps_pb, _pe_pb))

                                    def _add_period_bands_tm(_fig, show_labels: bool = True):
                                        """Adiciona bandas alternadas + linhas divisórias + rótulos de período."""
                                        _fc = ['rgba(255,255,255,0.045)', 'rgba(0,0,0,0)']
                                        for _ii, (_nm, _ps, _pe) in enumerate(_period_bands_tm):
                                            _fig.add_vrect(
                                                x0=_ps, x1=_pe,
                                                fillcolor=_fc[_ii % 2],
                                                layer='below', line_width=0,
                                            )
                                            if _ii > 0:  # linha divisória entre períodos
                                                _fig.add_vline(
                                                    x=_ps,
                                                    line_dash='dot',
                                                    line_color='rgba(255,255,255,0.20)',
                                                    line_width=1,
                                                )
                                            if show_labels:
                                                _fig.add_annotation(
                                                    x=(_ps + _pe) / 2,
                                                    y=1.01,
                                                    yref='paper',
                                                    text=_nm,
                                                    showarrow=False,
                                                    font=dict(
                                                        color='rgba(255,255,255,0.40)',
                                                        size=9),
                                                    xanchor='center',
                                                    yanchor='bottom',
                                                )

                                    # ── Paleta compartilhada heatmap / swimlane ─────
                                    # NaN = transparente → mostra o plot_bgcolor (cinza)
                                    # 0 %  = azul-marinho escuro  (ativo, baixa intensidade)
                                    # 50 % = verde médio
                                    # 75 % = âmbar
                                    # 87 % = laranja
                                    # 100% = vermelho
                                    _HT_CS = [
                                        [0.000, 'rgba(15,25,90,1)'],
                                        [0.250, 'rgba(12,90,45,1)'],
                                        [0.500, 'rgba(22,163,74,1)'],
                                        [0.750, 'rgba(234,179,8,1)'],
                                        [0.875, 'rgba(234,88,12,1)'],
                                        [1.000, 'rgba(220,38,38,1)'],
                                    ]
                                    # Fundo cinza-índigo: NaN (transparente) destacado
                                    _HT_BG  = 'rgba(28,28,44,1)'

                                    # ── Heatmap (% do máx coletivo) ────────────────
                                    _fig_ht = _go_tm.Figure(_go_tm.Heatmap(
                                        z=_z_mat,
                                        x=_tg,
                                        y=_y_lbl,
                                        colorscale=_HT_CS,
                                        zmin=0, zmax=100,
                                        colorbar=dict(
                                            title=dict(
                                                text='% do Máx<br>Coletivo',
                                                font=dict(color='white'),
                                            ),
                                            tickfont=dict(color='white'),
                                            tickvals=[0, 25, 50, 75, 100],
                                            ticktext=['0%', '25%', '50%', '75%', '100%'],
                                        ),
                                        hovertemplate=(
                                            "<b>%{y}</b><br>"
                                            "Tempo: %{x:.1f} min<br>"
                                            "Intensidade: %{z:.0f}% do máx coletivo"
                                            "<extra></extra>"
                                        ),
                                    ))
                                    _fig_ht.update_layout(
                                        title=dict(
                                            text=(f"Heatmap de Intensidade — {tipo_metrica}"
                                                  f" (% do Máx Coletivo)"
                                                  f" | Janela {window_minutes} min"),
                                            font=dict(color='white', size=13)),
                                        xaxis=dict(
                                            title='Tempo (minutos)',
                                            color='rgba(255,255,255,0.6)',
                                            gridcolor='rgba(255,255,255,0.05)'),
                                        yaxis=dict(
                                            color='rgba(255,255,255,0.75)',
                                            tickfont=dict(size=10)),
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor=_HT_BG,
                                        height=max(300, 36 * len(_atls_ord) + 120),
                                        margin=dict(l=200, r=100, t=40),
                                    )
                                    _add_period_bands_tm(_fig_ht)
                                    st.plotly_chart(_fig_ht, use_container_width=True)

                                    # ── Swimlane (visualização alternativa opcional) ─
                                    with st.expander(
                                            "🔀 Visualização alternativa — Swimlane por Atleta",
                                            expanded=False):
                                        st.caption(
                                            "Cada faixa horizontal representa um atleta. "
                                            "A cor indica a intensidade relativa ao pico coletivo. "
                                            "Cinza = fora de campo.")

                                        # Reamostrar em bins de 1 min para visual "tile"
                                        _sw_bin   = 1.0
                                        _sw_edges = np.arange(
                                            0, _cum_min_tm + _sw_bin, _sw_bin)
                                        _sw_ctrs  = (_sw_edges[:-1] + _sw_edges[1:]) / 2

                                        # Para cada atleta, média de intensidade por bin.
                                        # Bins sem dado → sentinel -10 (cinza explícito).
                                        _sw_z = []
                                        for _vr_sw in _raw_mat:
                                            _row_sw = []
                                            for _bi in range(len(_sw_ctrs)):
                                                _m = ((_tg >= _sw_edges[_bi]) &
                                                      (_tg <  _sw_edges[_bi + 1]))
                                                _vals = _vr_sw[_m]
                                                _ok   = _vals[~np.isnan(_vals)]
                                                _row_sw.append(
                                                    float(np.mean(_ok))
                                                    if len(_ok) > 0
                                                    else -10.0)
                                            _sw_z.append(_row_sw)

                                        # Colorscale com cinza explícito para -10
                                        # Normalizado: (v+10)/110
                                        # -10 → 0.000  |  0 → 0.091  |  50 → 0.545
                                        # 75  → 0.773  |  100→ 1.000
                                        _sw_cs = [
                                            [0.000, 'rgba(45,45,65,1)'],   # -10: fora
                                            [0.090, 'rgba(45,45,65,1)'],   # limite cinza
                                            [0.091, 'rgba(15,25,90,1)'],   # 0%: azul
                                            [0.364, 'rgba(12,90,45,1)'],   # 30%: verde
                                            [0.545, 'rgba(22,163,74,1)'],  # 50%: verde
                                            [0.773, 'rgba(234,179,8,1)'],  # 75%: âmbar
                                            [0.864, 'rgba(234,88,12,1)'],  # 87%: laranja
                                            [1.000, 'rgba(220,38,38,1)'],  # 100%: vermelho
                                        ]

                                        _fig_sw = _go_tm.Figure(_go_tm.Heatmap(
                                            z=_sw_z,
                                            x=list(_sw_ctrs),
                                            y=_y_lbl,
                                            colorscale=_sw_cs,
                                            zmin=-10, zmax=100,
                                            ygap=4,
                                            xgap=1,
                                            colorbar=dict(
                                                title=dict(
                                                    text='% do Máx<br>Coletivo',
                                                    font=dict(color='white'),
                                                ),
                                                tickfont=dict(color='white'),
                                                tickvals=[0, 25, 50, 75, 100],
                                                ticktext=['0%', '25%', '50%',
                                                          '75%', '100%'],
                                            ),
                                            hovertemplate=(
                                                "<b>%{y}</b><br>"
                                                "Tempo: %{x:.0f}–%{x:.0f} min<br>"
                                                "Intensidade: %{z:.0f}% do máx coletivo"
                                                "<extra></extra>"
                                            ),
                                        ))
                                        _fig_sw.update_layout(
                                            title=dict(
                                                text=(f"Swimlane — {tipo_metrica}"
                                                      f" | bins 1 min"),
                                                font=dict(color='white', size=12)),
                                            xaxis=dict(
                                                title='Tempo (minutos)',
                                                color='rgba(255,255,255,0.6)',
                                                gridcolor='rgba(255,255,255,0.04)'),
                                            yaxis=dict(
                                                color='rgba(255,255,255,0.75)',
                                                tickfont=dict(size=10)),
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor=_HT_BG,
                                            height=max(280,
                                                       28 * len(_atls_ord) + 100),
                                            margin=dict(l=200, r=100, t=36),
                                        )
                                        _add_period_bands_tm(_fig_sw)
                                        st.plotly_chart(
                                            _fig_sw, use_container_width=True)

                                    # ── Média do time (valores brutos, colorida) ────
                                    _tm_mean = np.nanmean(_raw_mat, axis=0)
                                    if not np.all(np.isnan(_tm_mean)):
                                        _tmx = float(np.nanmax(_tm_mean))
                                        _tla = round(_tmx * 0.90, 1)
                                        _tlm = round(_tmx * 0.75, 1)
                                        _tm_cores = [
                                            '#ef4444' if v >= _tla
                                            else '#f59e0b' if v >= _tlm
                                            else '#22c55e'
                                            for v in _tm_mean
                                        ]
                                        _fig_avg = _go_tm.Figure()
                                        _fig_avg.add_trace(_go_tm.Scatter(
                                            x=_tg, y=_tm_mean,
                                            mode='markers',
                                            marker=dict(color=_tm_cores, size=3),
                                            name='Média do Time',
                                            hovertemplate=(
                                                "Tempo: %{x:.1f} min<br>"
                                                f"Média Time: %{{y:.1f}} {_unidade_jan}"
                                                "<extra></extra>"
                                            ),
                                        ))
                                        _fig_avg.add_hline(
                                            y=_tla, line_dash='dash',
                                            line_color='rgba(239,68,68,0.50)',
                                            annotation_text=f"Alta ≥{_tla}")
                                        _fig_avg.add_hline(
                                            y=_tlm, line_dash='dot',
                                            line_color='rgba(245,158,11,0.50)',
                                            annotation_text=f"Média-Alta ≥{_tlm}")
                                        _add_period_bands_tm(_fig_avg)
                                        _fig_avg.update_layout(
                                            title=dict(
                                                text=(f"Intensidade Média do Time — "
                                                      f"{tipo_metrica} ({_unidade_jan})"
                                                      f" | Rolling {window_minutes} min"),
                                                font=dict(color='white', size=13)),
                                            xaxis=dict(
                                                title='Tempo (minutos)',
                                                color='rgba(255,255,255,0.6)',
                                                gridcolor='rgba(255,255,255,0.07)'),
                                            yaxis=dict(
                                                title=f'{tipo_metrica} ({_unidade_jan})',
                                                color='rgba(255,255,255,0.6)',
                                                gridcolor='rgba(255,255,255,0.07)'),
                                            paper_bgcolor='rgba(0,0,0,0)',
                                            plot_bgcolor='rgba(0,0,0,0)',
                                            height=300,
                                        )
                                        st.plotly_chart(_fig_avg, use_container_width=True)

                                        # ── Cards de esforços coletivos ────────────
                                        _tg_v   = _tg[~np.isnan(_tm_mean)]
                                        _tmm_v  = _tm_mean[~np.isnan(_tm_mean)]
                                        _alta_ev_tm, _media_ev_tm = (
                                            encontrar_eventos_nao_sobrepostos(
                                                list(_tg_v), list(_tmm_v),
                                                window_minutes, _tla, _tlm, _tmx,
                                            ) if len(_tg_v) > 1 else ([], [])
                                        )
                                        _alta_cnt_tm  = len(_alta_ev_tm)
                                        _media_cnt_tm = len(_media_ev_tm)

                                        _card_alta_tm = f"""
    <div style="
        background: linear-gradient(135deg,rgba(220,38,38,0.18) 0%,rgba(153,27,27,0.08) 100%);
        border: 1px solid rgba(239,68,68,0.55); border-radius: 18px;
        padding: 32px 24px 26px; text-align: center;
        box-shadow: 0 0 32px rgba(220,38,38,0.22),0 2px 8px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(255,255,255,0.07);
        position: relative; overflow: hidden;">
      <div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;
                  border-radius:50%;background:rgba(220,38,38,0.10);pointer-events:none;"></div>
      <div style="font-size:11px;font-weight:600;letter-spacing:2px;
                  color:rgba(255,255,255,0.5);text-transform:uppercase;margin-bottom:10px;">
        Alta Intensidade — Time</div>
      <div style="font-size:72px;font-weight:800;color:#f87171;line-height:1;
                  text-shadow:0 0 24px rgba(248,113,113,0.5);">{_alta_cnt_tm}</div>
      <div style="font-size:12px;color:rgba(255,255,255,0.38);margin-top:14px;line-height:1.6;">
        janelas coletivas com <strong style="color:rgba(248,113,113,0.8);">
        {tipo_metrica} ≥ {_tla} {_unidade_jan}</strong><br>
        &gt; 90% do pico coletivo ({_tmx:.1f} {_unidade_jan})
      </div>
    </div>"""

                                        _card_media_tm = f"""
    <div style="
        background: linear-gradient(135deg,rgba(202,138,4,0.18) 0%,rgba(133,77,14,0.08) 100%);
        border: 1px solid rgba(234,179,8,0.50); border-radius: 18px;
        padding: 32px 24px 26px; text-align: center;
        box-shadow: 0 0 32px rgba(202,138,4,0.22),0 2px 8px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(255,255,255,0.07);
        position: relative; overflow: hidden;">
      <div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;
                  border-radius:50%;background:rgba(202,138,4,0.10);pointer-events:none;"></div>
      <div style="font-size:11px;font-weight:600;letter-spacing:2px;
                  color:rgba(255,255,255,0.5);text-transform:uppercase;margin-bottom:10px;">
        Média-Alta Intensidade — Time</div>
      <div style="font-size:72px;font-weight:800;color:#fbbf24;line-height:1;
                  text-shadow:0 0 24px rgba(251,191,36,0.5);">{_media_cnt_tm}</div>
      <div style="font-size:12px;color:rgba(255,255,255,0.38);margin-top:14px;line-height:1.6;">
        janelas coletivas com <strong style="color:rgba(251,191,36,0.8);">
        {_tlm} ≤ {tipo_metrica} &lt; {_tla} {_unidade_jan}</strong><br>
        75–90% do pico coletivo ({_tmx:.1f} {_unidade_jan})
      </div>
    </div>"""

                                        _ctm1, _ctm2 = st.columns(2)
                                        with _ctm1:
                                            st.markdown(_card_alta_tm,
                                                        unsafe_allow_html=True)
                                        with _ctm2:
                                            st.markdown(_card_media_tm,
                                                        unsafe_allow_html=True)

                                        # ── Feedback coletivo ───────────────────────
                                        _n_tot_tm  = _alta_cnt_tm + _media_cnt_tm
                                        _s_tot_tm  = "s" if _n_tot_tm    != 1 else ""
                                        _s_alt_tm  = "s" if _alta_cnt_tm != 1 else ""
                                        _s_med_tm  = "s" if _media_cnt_tm!= 1 else ""
                                        _dur_tot_tm = (
                                            float(_tg_v[-1]) + window_minutes
                                            if len(_tg_v) else 0.0)
                                        _dh = int(_dur_tot_tm // 60)
                                        _dm = int(_dur_tot_tm % 60)
                                        _dur_str_tm = (
                                            f"{_dh}h {_dm:02d}min" if _dh
                                            else f"{_dm} min")
                                        st.markdown(f"""
<div style="background:linear-gradient(135deg,rgba(25,35,55,0.65) 0%,rgba(15,25,45,0.45) 100%);
     border:1px solid rgba(255,255,255,0.09);
     border-left:3px solid rgba(93,173,226,0.55);
     border-radius:10px;padding:14px 20px;margin:20px 0 10px 0;
     font-size:0.875rem;line-height:1.75;color:rgba(255,255,255,0.72);">
  💬 &nbsp;<strong style="color:white">O time</strong> apresentou
  <strong style="color:#f87171">{_alta_cnt_tm}</strong> período{_s_alt_tm}
  de <span style="color:#f87171">alta intensidade coletiva</span> e
  <strong style="color:#fbbf24">{_media_cnt_tm}</strong>
  de <span style="color:#fbbf24">média-alta</span> —
  totalizando <strong style="color:white">{_n_tot_tm} janela{_s_tot_tm}
  distinta{_s_tot_tm}</strong> de {window_minutes} min com
  <em>{tipo_metrica}</em> médio ≥
  <strong>{_tlm:.1f} {_unidade_jan}</strong>,
  ao longo de <strong style="color:#5dade2">{_dur_str_tm}</strong>
  analisados. Pico coletivo máximo:
  <strong style="color:white">{_tmx:.1f} {_unidade_jan}</strong>.
</div>""", unsafe_allow_html=True)

                                    # ── Tabela de esforços coletivos ────────────────
                                    st.markdown(
                                        "#### 📋 Esforços Coletivos — Média-Alta e Alta Intensidade")
                                    st.caption(
                                        f"Cada linha é uma janela de **{window_minutes} min** "
                                        "distinta e não-sobreposta da **média do time**, "
                                        "selecionada pelo pico máximo coletivo.")

                                    def _periodo_para_t_tm(_t_m):
                                        for (_nm_p, _ps_p, _pe_p) in _period_bands_tm:
                                            if _ps_p <= _t_m <= _pe_p + 0.1:
                                                return _nm_p
                                        if _period_bands_tm:
                                            return min(
                                                _period_bands_tm,
                                                key=lambda _b: abs(_b[1] - _t_m))[0]
                                        return None

                                    _todos_ev_tm = (
                                        [dict(_e, _cat='alta')  for _e in _alta_ev_tm] +
                                        [dict(_e, _cat='media') for _e in _media_ev_tm]
                                    )
                                    _todos_ev_tm.sort(
                                        key=lambda _e: _e['valor'], reverse=True)

                                    if _todos_ev_tm:
                                        _rows_tm = []
                                        for _rk_tm, _ev_tm in enumerate(_todos_ev_tm, 1):
                                            _per_nm = _periodo_para_t_tm(
                                                _ev_tm.get('t_ini_min', 0.0))
                                            _row_tm = {
                                                '#': _rk_tm,
                                                'Início': _ev_tm['inicio'],
                                                'Fim':    _ev_tm['fim'],
                                            }
                                            if _per_nm:
                                                _row_tm['Período'] = _per_nm
                                            _row_tm.update({
                                                f'{tipo_metrica} Médio ({_unidade_jan})':
                                                    _ev_tm['valor'],
                                                '↓ % do Pico Coletivo':
                                                    _ev_tm['pct_max'],
                                                'Intensidade':
                                                    _ev_tm['intensidade'],
                                            })
                                            _rows_tm.append(_row_tm)

                                        _df_ev_tm = pd.DataFrame(_rows_tm)

                                        def _style_ev_tm(row):
                                            if ('Alta Intensidade' in str(
                                                    row.get('Intensidade', ''))
                                                    and 'Média' not in str(
                                                    row.get('Intensidade', ''))):
                                                return ['background-color:rgba(239,68,68,0.12)'] * len(row)
                                            elif 'Média-Alta' in str(row.get('Intensidade', '')):
                                                return ['background-color:rgba(245,158,11,0.10)'] * len(row)
                                            return [''] * len(row)

                                        _fmt_tm = {
                                            f'{tipo_metrica} Médio ({_unidade_jan})': '{:.1f}',
                                            '↓ % do Pico Coletivo': '{:.1f}%',
                                        }
                                        # ── Tabela com seleção de linha ────────────
                                        # Clicar na linha aciona a animação abaixo.
                                        _ev_tbl_ev = st.dataframe(
                                            _df_ev_tm.style.apply(
                                                _style_ev_tm, axis=1).format(_fmt_tm),
                                            use_container_width=True,
                                            height=min(600, 40 + len(_rows_tm) * 36),
                                            on_select="rerun",
                                            selection_mode="single-row",
                                            key="ev_tm_table_sel",
                                        )
                                        if not st.session_state.get('modo_apresentacao'):
                                            st.download_button(
                                                "📥 Exportar Esforços Coletivos (CSV)",
                                                _df_ev_tm.to_csv(index=False).encode('utf-8'),
                                                f"esforcos_coletivos_{tipo_metrica}"
                                                f"_{window_minutes}min.csv",
                                                mime='text/csv',
                                                key="dl_ef_team",
                                            )

                                        # ── Animação do esforço selecionado ────────
                                        if dados_posicao_por_periodo:
                                            # Índice da linha selecionada (padrão: 0)
                                            _sel_ao_idx = (
                                                _ev_tbl_ev.selection.rows[0]
                                                if (hasattr(_ev_tbl_ev, 'selection')
                                                    and _ev_tbl_ev.selection.rows)
                                                else 0
                                            )
                                            _ev_anim = _todos_ev_tm[_sel_ao_idx]
                                            _t_ini_ao = _ev_anim.get('t_ini_min', 0.0)
                                            _per_ao_lbl = (
                                                _periodo_para_t_tm(_t_ini_ao) or '?')

                                            st.markdown("---")
                                            st.markdown(
                                                "#### 🎬 Animar Esforço Coletivo no Campo")
                                            st.caption(
                                                f"**Clique em uma linha** da tabela para "
                                                f"selecionar o esforço. Exibindo: "
                                                f"**{_ev_anim['inicio']}→"
                                                f"{_ev_anim['fim']}** "
                                                f"({_per_ao_lbl})")

                                            # Config do campo (necessária antes do loop
                                            # para o fallback lats/lons → xs/ys)
                                            _anim_cfg_ao = None
                                            for _hk_ao in list(st.session_state.keys()):
                                                if (_hk_ao.startswith("campo_cfg__")
                                                        and isinstance(
                                                        st.session_state[_hk_ao], dict)):
                                                    _anim_cfg_ao = st.session_state[_hk_ao]
                                                    break
                                            _fl_ao = float(
                                                _anim_cfg_ao.get('fl', 105)
                                                if _anim_cfg_ao else 105)
                                            _fw_ao = float(
                                                _anim_cfg_ao.get('fw', 68)
                                                if _anim_cfg_ao else 68)

                                            # Fim de cada período em match-time
                                            _pend_ao = {
                                                _pn_ao2: (
                                                    _period_start_min_tm[_pn_ao2]
                                                    + (_period_abs_tm[_pn_ao2][1]
                                                       - _period_abs_tm[_pn_ao2][0])
                                                    / 60.0)
                                                for _pn_ao2 in _period_order_tm
                                            }

                                            # Paleta de cores
                                            _pal_ao = [
                                                '#FF6B6B','#4ECDC4','#45B7D1','#96CEB4',
                                                '#FFEAA7','#DDA0DD','#98D8C8','#F7DC6F',
                                                '#BB8FCE','#76D7C4','#F1948A','#85C1E9',
                                                '#82E0AA','#F8C471','#AED6F1',
                                            ]

                                            # Para cada atleta: acha período que cobre
                                            # t_ini_ao e extrai o segmento GPS correto.
                                            # Hz calculado por len(xs)/dur_s — robusto
                                            # mesmo com ts_pos = 0 (não confiável).
                                            # Fallback: lats/lons → gps_para_campo_coords
                                            # Exclui goleiro: animação mostra
                                            # apenas os jogadores de linha
                                            _jan_atls_linha = [
                                                _a for _a in _jan_atletas
                                                if _get_pos_grupo(
                                                    _get_pos_atl(_a))[0] != 'Goleiro'
                                            ]

                                            _anim_map = {}
                                            _ao_hz_ref = 10.0  # Hz de ref para frames
                                            for _ci_ao, _atl_ao in enumerate(
                                                    _jan_atls_linha):
                                                for _pn_ao3 in _sorted_by_ts_tm:
                                                    _pos_ao = dados_posicao_por_periodo.get(
                                                        _pn_ao3, {}).get(_atl_ao, {})
                                                    _xs_ao = list(
                                                        _pos_ao.get('xs', []))
                                                    _ys_ao = list(
                                                        _pos_ao.get('ys', []))

                                                    # Fallback: coordenadas GPS → campo
                                                    if not _xs_ao and _anim_cfg_ao:
                                                        _lts_ao = _pos_ao.get('lats', [])
                                                        _lns_ao = _pos_ao.get('lons', [])
                                                        if _lts_ao and _lns_ao:
                                                            try:
                                                                _xs_ao, _ys_ao = (
                                                                    gps_para_campo_coords(
                                                                        _lts_ao, _lns_ao,
                                                                        _anim_cfg_ao))
                                                            except Exception:
                                                                _applog.log_debug_exc()

                                                    if not _xs_ao:
                                                        continue

                                                    _ps_ao = _period_start_min_tm.get(
                                                        _pn_ao3, 0.0)
                                                    _pe_ao = _pend_ao.get(_pn_ao3, _ps_ao)
                                                    if not (_ps_ao <= _t_ini_ao <= _pe_ao):
                                                        continue

                                                    # Hz por comprimento do array
                                                    _dur_s_ao = (
                                                        (_period_abs_tm[_pn_ao3][1]
                                                         - _period_abs_tm[_pn_ao3][0])
                                                        if _period_abs_tm.get(_pn_ao3)
                                                        else 0.0)
                                                    _hz_ao = (
                                                        len(_xs_ao) / _dur_s_ao
                                                        if _dur_s_ao > 0
                                                        else 10.0)
                                                    _ao_hz_ref = _hz_ao

                                                    _off_s_ao = (
                                                        (_t_ini_ao - _ps_ao) * 60.0)
                                                    _n_smp_ao = max(
                                                        2,
                                                        int(window_minutes * 60 * _hz_ao))
                                                    _is_ao = int(_off_s_ao * _hz_ao)
                                                    _ie_ao = min(
                                                        _is_ao + _n_smp_ao,
                                                        len(_xs_ao))

                                                    if 0 <= _is_ao < len(_xs_ao):
                                                        _vs_ao = list(
                                                            _pos_ao.get('vel', []))
                                                        _anim_map[_atl_ao] = {
                                                            'xs': _xs_ao[_is_ao:_ie_ao],
                                                            'ys': (
                                                                _ys_ao[_is_ao:_ie_ao]
                                                                if _ys_ao
                                                                else [0]*(_ie_ao-_is_ao)),
                                                            'vel': (
                                                                _vs_ao[_is_ao:_ie_ao]
                                                                if _vs_ao
                                                                else [0]*(_ie_ao-_is_ao)),
                                                            'color': _pal_ao[
                                                                _ci_ao % len(_pal_ao)],
                                                            'label': (
                                                                _atl_ao.split()[-1][:10]
                                                                if _atl_ao.split()
                                                                else _atl_ao[:10]),
                                                        }
                                                    break

                                            if len(_anim_map) < 2:
                                                st.info(
                                                    "GPS insuficiente para este esforço "
                                                    f"({len(_anim_map)} atleta(s) com "
                                                    "dados de posição). Verifique se o "
                                                    "campo foi configurado e se os dados "
                                                    "GPS foram importados.")
                                            else:
                                                _per_ao_lbl = _periodo_para_t_tm(
                                                    _t_ini_ao) or '?'
                                                _fig_ao = desenhar_campo_futebol_bonito(
                                                    field_length=_fl_ao,
                                                    field_width=_fw_ao,
                                                    title=(
                                                        f"🎬 {_ev_anim['inicio']}"
                                                        f"→{_ev_anim['fim']}"
                                                        f" | {_per_ao_lbl}"
                                                        f" | {_ev_anim['valor']:.1f}"
                                                        f" {_unidade_jan}"),
                                                )

                                                _atls_ao = list(_anim_map.keys())
                                                _tidxs_ao = []
                                                for _pa_ao in _atls_ao:
                                                    _wd_ao = _anim_map[_pa_ao]
                                                    _fig_ao.add_trace(go.Scatter(
                                                        x=[_wd_ao['xs'][0]]
                                                            if _wd_ao['xs'] else [0],
                                                        y=[_wd_ao['ys'][0]]
                                                            if _wd_ao['ys'] else [0],
                                                        mode='markers+text',
                                                        marker=dict(
                                                            size=20,
                                                            color=_wd_ao['color'],
                                                            symbol='circle',
                                                            line=dict(color='white',
                                                                       width=2)),
                                                        text=[_wd_ao['label']],
                                                        textposition='top center',
                                                        textfont=dict(color='white',
                                                                       size=8),
                                                        name=_pa_ao, showlegend=True,
                                                    ))
                                                    _tidxs_ao.append(
                                                        len(_fig_ao.data) - 1)

                                                _wl_ao = max(
                                                    len(_anim_map[a]['xs'])
                                                    for a in _atls_ao)
                                                _step_ao = max(1, _wl_ao // 80)
                                                _fr_ao = list(
                                                    range(0, _wl_ao, _step_ao))
                                                if (_fr_ao and
                                                        _fr_ao[-1] != _wl_ao - 1):
                                                    _fr_ao.append(_wl_ao - 1)

                                                _frames_ao = []
                                                for _fi_ao in _fr_ao:
                                                    _ts_ao = _fi_ao / _ao_hz_ref
                                                    _mm_ao = int(_ts_ao // 60)
                                                    _ss_ao = int(_ts_ao % 60)
                                                    _fd_ao = []
                                                    for _pa_ao in _atls_ao:
                                                        _wd_ao = _anim_map[_pa_ao]
                                                        _xi_ao = (
                                                            _wd_ao['xs'][_fi_ao]
                                                            if _fi_ao < len(_wd_ao['xs'])
                                                            else (_wd_ao['xs'][-1]
                                                                  if _wd_ao['xs'] else 0))
                                                        _yi_ao = (
                                                            _wd_ao['ys'][_fi_ao]
                                                            if _fi_ao < len(_wd_ao['ys'])
                                                            else (_wd_ao['ys'][-1]
                                                                  if _wd_ao['ys'] else 0))
                                                        _fd_ao.append(go.Scatter(
                                                            x=[_xi_ao], y=[_yi_ao],
                                                            mode='markers+text',
                                                            marker=dict(
                                                                size=20,
                                                                color=_wd_ao['color'],
                                                                symbol='circle',
                                                                line=dict(color='white',
                                                                           width=2)),
                                                            text=[_wd_ao['label']],
                                                            textposition='top center',
                                                            textfont=dict(color='white',
                                                                           size=8),
                                                        ))
                                                    _frames_ao.append(go.Frame(
                                                        data=_fd_ao,
                                                        traces=_tidxs_ao,
                                                        name=str(_fi_ao),
                                                        layout=go.Layout(title=dict(
                                                            text=(
                                                                f"🎬 "
                                                                f"{_ev_anim['inicio']}"
                                                                f"→{_ev_anim['fim']}"
                                                                f" | ⏱️ +"
                                                                f"{_mm_ao}:{_ss_ao:02d}"
                                                                f" min"
                                                            ),
                                                            font=dict(color='white',
                                                                       size=12),
                                                        )),
                                                    ))

                                                _fig_ao.frames = _frames_ao
                                                _fig_ao.update_layout(
                                                    height=580,
                                                    updatemenus=[dict(
                                                        type='buttons',
                                                        showactive=False,
                                                        y=0, x=0.5,
                                                        xanchor='center',
                                                        buttons=[
                                                            dict(
                                                                label='▶ Play',
                                                                method='animate',
                                                                args=[None, dict(
                                                                    frame=dict(
                                                                        duration=100,
                                                                        redraw=True),
                                                                    fromcurrent=True,
                                                                    transition=dict(
                                                                        duration=100,
                                                                        easing='linear'),
                                                                    mode='immediate')]),
                                                            dict(
                                                                label='⏸ Pause',
                                                                method='animate',
                                                                args=[[None], dict(
                                                                    frame=dict(
                                                                        duration=0,
                                                                        redraw=False),
                                                                    mode='immediate')]),
                                                        ],
                                                    )],
                                                    sliders=[dict(
                                                        steps=[
                                                            dict(
                                                                args=[[f.name],
                                                                      dict(
                                                                          frame=dict(
                                                                              duration=0,
                                                                              redraw=True),
                                                                          mode='immediate')],
                                                                method='animate',
                                                                label='',
                                                            )
                                                            for f in _frames_ao
                                                        ],
                                                        x=0.0, y=-0.05, len=1.0,
                                                        currentvalue=dict(visible=False),
                                                    )],
                                                    legend=dict(
                                                        orientation='h',
                                                        yanchor='bottom', y=-0.30,
                                                        xanchor='center', x=0.5,
                                                        font=dict(color='white', size=8),
                                                    ),
                                                )
                                                st.plotly_chart(
                                                    _fig_ao,
                                                    use_container_width=True)
                                    else:
                                        st.info(
                                            "Nenhum esforço coletivo de média-alta "
                                            "ou alta intensidade encontrado.")

            # ==================== ABA 4: CARGA NEUROMUSCULAR ====================
            with abas[3]:
                st.divider()
                st.subheader("💪 Análise de Carga Neuromuscular")
                st.markdown("""
                Esforços de **aceleração** e **desaceleração** intensa são indicadores críticos de
                carga neuromuscular/excêntrica — desacelerações superiores a 2 m/s² geram impacto
                muscular frequentemente maior que sprints. Esta aba quantifica esses esforços por
                minuto e acumula a carga ao longo da sessão.
                """)

                if dados_sensor_por_atleta_por_periodo:
                    _NM_TODOS = "🔀 Todos os períodos (combinado)"
                    _nm_opcoes_per = [_NM_TODOS] + list(dados_sensor_por_atleta_por_periodo.keys())
                    _nm_per = st.selectbox("Período:", _nm_opcoes_per, key="nm_periodo")

                    # Monta lista de atletas disponíveis
                    if _nm_per == _NM_TODOS:
                        _nm_ats_set = set()
                        for _pv in dados_sensor_por_atleta_por_periodo.values():
                            _nm_ats_set.update(_pv.keys())
                        _nm_ats = sorted(_nm_ats_set)
                    else:
                        _nm_ats = list(dados_sensor_por_atleta_por_periodo.get(_nm_per, {}).keys())

                    if _nm_ats:
                        _nm_atl = st.selectbox("Atleta:", _nm_ats, key="nm_atleta")
                        _nm_lim = st.slider("Limiar de intensidade (m/s²):", 1.0, 4.0, 2.0, 0.5,
                                            key="nm_limiar",
                                            help="Acelerações/desacelerações acima deste valor são classificadas como intensas.")

                        _nm_dur_s = get_min_dur_s()
                        st.caption(
                            f"⚙️ Duração mínima de acc/dec: **{_nm_dur_s:.1f} s** "
                            f"({max(1, round(_nm_dur_s * _SENSOR_HZ))} frames a 10 Hz) — "
                            "ajuste na sidebar."
                        )

                        # Combina sensor_points de todos os períodos se necessário
                        if _nm_per == _NM_TODOS:
                            _nm_sp = []
                            for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                _nm_sp += _pv2.get(_nm_atl, [])
                            if len(dados_sensor_por_atleta_por_periodo) > 1:
                                st.caption(
                                    f"📊 Combinando **{len(dados_sensor_por_atleta_por_periodo)} períodos** "
                                    f"→ {len(_nm_sp):,} amostras para **{_nm_atl}**."
                                )
                        else:
                            _nm_sp = dados_sensor_por_atleta_por_periodo[_nm_per].get(_nm_atl, [])

                        # (P6) cache: evita recomputar masks/EPM de ~50k amostras
                        # a cada interação de widget (dados grandes fora da chave).
                        _nm_key = (str(st.session_state.get('_token_marker', '')),
                                   str(st.session_state.get('activity_id', '')),
                                   str(_nm_per), str(_nm_atl),
                                   float(_nm_lim), float(_nm_dur_s), len(_nm_sp))
                        _nm_dados = _neuro_cached(_nm_key, _nm_sp, _nm_lim, _nm_dur_s)

                        if _nm_dados:
                            _selo_fonte('sensor')   # (P4) acc/dec do sinal nativo
                            # Métricas resumo
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric(f"🟢 Acels. ≥{_nm_lim} m/s²", _nm_dados['total_hi_acc'])
                            c2.metric(f"🔴 Desacels. ≥{_nm_lim} m/s²", _nm_dados['total_hi_dec'])
                            c3.metric("⚡ Total Acels. (todas)", _nm_dados['total_hi_acc'] + _nm_dados['total_med_acc'])
                            c4.metric("⚡ Total Desacels. (todas)", _nm_dados['total_hi_dec'] + _nm_dados['total_med_dec'])

                            _nm_razao = ((_nm_dados['total_hi_dec'] / _nm_dados['total_hi_acc'])
                                         if _nm_dados['total_hi_acc'] > 0 else 0)
                            if _nm_razao > 1.4:
                                st.warning(f"⚠️ Razão Dec/Acc = **{_nm_razao:.2f}** — alto componente excêntrico. "
                                           "Monitorar recuperação muscular dos membros inferiores.")
                            elif _nm_razao > 0.9:
                                st.info(f"ℹ️ Razão Dec/Acc = **{_nm_razao:.2f}** — carga excêntrica equilibrada.")
                            else:
                                st.success(f"✅ Razão Dec/Acc = **{_nm_razao:.2f}** — perfil predominantemente acelerativo.")

                            _nm_fig = plotar_carga_neuromuscular(_nm_dados, _nm_atl)
                            st.plotly_chart(_nm_fig, use_container_width=True)

                            # Exportar
                            _nm_df = pd.DataFrame({
                                'Tempo (min)': _nm_dados['t_mid'],
                                'Acels. Intensas/min': _nm_dados['hi_acc_min'],
                                'Acels. Médias/min': _nm_dados['med_acc_min'],
                                'Desacels. Intensas/min': _nm_dados['hi_dec_min'],
                                'Desacels. Médias/min': _nm_dados['med_dec_min'],
                            })
                            st.download_button(
                                "📥 Exportar Carga Neuromuscular (CSV)",
                                _nm_df.to_csv(index=False),
                                f"carga_neuro_{_nm_atl.replace(' ','_')}.csv"
                            )
                        else:
                            st.info("Dados de aceleração insuficientes para este atleta/período.")

                        # ── FEATURE 3: Potência Metabólica na aba Carga Neuromuscular ──
                        st.markdown("---")
                        st.markdown("### ⚡ Potência Metabólica (W/kg)")
                        _nm_mp_pts = _nm_sp
                        _nm_mp_vals = [
                            float(p['mp']) for p in _nm_mp_pts
                            if p.get('mp') and float(p.get('mp') or 0) > 0
                        ]
                        if _nm_mp_vals:
                            _nm_mp_mean = float(np.mean(_nm_mp_vals))
                            _nm_mp_max  = float(np.max(_nm_mp_vals))
                            _nm_mp_pct20 = sum(1 for v in _nm_mp_vals if v > 20) / max(1, len(_nm_mp_vals)) * 100
                            _nm_mp_t25   = sum(1 for v in _nm_mp_vals if v > 25) * 0.1
                            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
                            _mc1.metric("MP Médio (W/kg)", f"{_nm_mp_mean:.1f}")
                            _mc2.metric("MP Máx (W/kg)", f"{_nm_mp_max:.1f}")
                            _mc3.metric("MP > 20 W/kg (%)", f"{_nm_mp_pct20:.1f}%")
                            _mc4.metric("Tempo > 25 W/kg (s)", f"{_nm_mp_t25:.0f}s")
                        else:
                            st.info(
                                "Dados de potência metabólica (mp) não disponíveis. "
                                "Verifique se o dispositivo suporta este parâmetro."
                            )

                else:
                    st.info("Carregue os dados de um atleta para analisar a carga neuromuscular.")

            # ==================== ABA 5: PERFIL ACELERAÇÃO-VELOCIDADE ====================
            with abas[4]:
                st.subheader("🏎️ Perfil Aceleração × Velocidade")
                st.caption(
                    "Baseado no modelo de Samozino & Morin (2016) — relação linear entre aceleração e velocidade "
                    "para extrair o perfil mecânico individual de sprint."
                )
                st.markdown("---")

                _av_periodos = list(dados_sensor_por_atleta_por_periodo.keys())
                if _av_periodos and resultados_por_periodo:
                    _AV_TODOS = "🔀 Todos os períodos (combinado)"
                    _av_opcoes_per = [_AV_TODOS] + _av_periodos
                    _av_col1, _av_col2 = st.columns([2, 1])
                    with _av_col1:
                        _av_per = st.selectbox("Período:", _av_opcoes_per, key="av_periodo")
                    with _av_col2:
                        if _av_per == _AV_TODOS:
                            _av_ats_set = set()
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _av_ats_set.update(_pv.keys())
                            _av_atletas_disp = sorted(_av_ats_set)
                        else:
                            _av_atletas_disp = list(dados_sensor_por_atleta_por_periodo.get(_av_per, {}).keys())
                        _av_atls_sel = st.multiselect(
                            "Atletas (até 6):", _av_atletas_disp,
                            default=_av_atletas_disp[:min(3, len(_av_atletas_disp))],
                            key="av_atletas_sel"
                        )

                    if _av_atls_sel:
                        # ── Paleta de cores por atleta (usa paleta global persistente) ─────
                        _AV_PALETTE = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4']
                        _av_cores = {
                            a: st.session_state.get('athlete_colors', {}).get(a, _AV_PALETTE[i % len(_AV_PALETTE)])
                            for i, a in enumerate(_av_atls_sel)
                        }

                        # ── Extrai (v, a) de todos os atletas selecionados ────
                        _av_dados = {}
                        for _av_atl in _av_atls_sel:
                            if _av_per == _AV_TODOS:
                                _spts = []
                                for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                    _spts += _pv2.get(_av_atl, [])
                            else:
                                _spts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_av_atl, [])
                            if not _spts:
                                continue
                            _vels, _accs = [], []
                            for _p in _spts:
                                _v = _p.get('v')
                                _a = _p.get('a')
                                if _v is not None and _a is not None:
                                    _vels.append(float(_v) * 3.6)
                                    _accs.append(float(_a))
                            if _vels:
                                _av_dados[_av_atl] = {
                                    'vel': np.array(_vels),
                                    'acc': np.array(_accs),
                                }

                        if not _av_dados:
                            st.warning("Dados de aceleração não disponíveis para os atletas selecionados.")
                        else:
                            # ── FEATURE 3: toggle para velocidade bruta (rv) ─
                            _av_use_rv = st.toggle(
                                "Usar velocidade bruta (rv) para F-V profiling",
                                value=False, key="av_use_rv",
                                help="rv = velocidade bruta do sensor (mais preciso para F-V). "
                                     "Requer que o dispositivo suporte o parâmetro rv.",
                            )
                            if _av_use_rv:
                                st.caption("📍 **Perfil F-V (velocidade bruta — mais preciso)**")
                                # Recalcula usando rv ao invés de v
                                for _av_atl in list(_av_dados.keys()):
                                    if _av_per == _AV_TODOS:
                                        _spts_rv = []
                                        for _pv2 in dados_sensor_por_atleta_por_periodo.values():
                                            _spts_rv += _pv2.get(_av_atl, [])
                                    else:
                                        _spts_rv = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_av_atl, [])
                                    _vels_rv, _accs_rv = [], []
                                    for _p in _spts_rv:
                                        _rv = _p.get('rv')
                                        _a = _p.get('a')
                                        if _rv is not None and _a is not None:
                                            _vels_rv.append(float(_rv) * 3.6)
                                            _accs_rv.append(float(_a))
                                    if _vels_rv:
                                        _av_dados[_av_atl]['vel'] = np.array(_vels_rv)
                                        _av_dados[_av_atl]['acc'] = np.array(_accs_rv)

                            # ════════════════════════════════════════════════
                            # SEÇÃO 1 — SCATTER ACC × VEL (multi-atleta)
                            # ════════════════════════════════════════════════
                            _sc_title = ("📍 Perfil F-V (velocidade bruta — mais preciso)"
                                         if _av_use_rv else "📍 Scatter Aceleração × Velocidade")
                            st.markdown(f"### {_sc_title}")
                            _fig_sc = go.Figure()

                            for _av_atl, _d in _av_dados.items():
                                # Sub-amostra para não sobrecarregar o gráfico
                                _step = max(1, len(_d['vel']) // 3000)
                                _v_s  = _d['vel'][::_step]
                                _a_s  = _d['acc'][::_step]
                                _fig_sc.add_trace(go.Scatter(
                                    x=_v_s, y=_a_s, mode='markers',
                                    marker=dict(color=_av_cores[_av_atl], size=3, opacity=0.45),
                                    name=_av_atl,
                                    hovertemplate=f'{_av_atl}<br>Vel: %{{x:.1f}} km/h<br>Acc: %{{y:.2f}} m/s²<extra></extra>',
                                ))

                            # Linhas de limiar
                            for _lim, _cor, _txt in [(3.0,'#F44336','Acc >3 m/s²'),
                                                      (2.0,'#FF9800','Acc >2 m/s²'),
                                                      (-2.0,'#FF9800','Dcc <-2 m/s²'),
                                                      (-3.0,'#F44336','Dcc <-3 m/s²')]:
                                _fig_sc.add_hline(y=_lim, line=dict(color=_cor, dash='dash', width=1),
                                                  annotation_text=_txt,
                                                  annotation_font=dict(color=_cor, size=10))

                            _fig_sc.add_hline(y=0, line=dict(color='white', width=1, dash='dot'))
                            _fig_sc.update_layout(
                                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                font=dict(color='white'),
                                xaxis=dict(title='Velocidade (km/h)', gridcolor='#333', range=[-1, None]),
                                yaxis=dict(title='Aceleração (m/s²)', gridcolor='#333'),
                                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)'),
                                height=420, margin=dict(t=20, b=10),
                            )
                            st.plotly_chart(_fig_sc, use_container_width=True)

                            st.markdown("---")

                            # ════════════════════════════════════════════════
                            # SEÇÃO 2 — HISTOGRAMA DE ACELERAÇÕES POR ZONA
                            # ════════════════════════════════════════════════
                            st.markdown("### 📊 Distribuição de Acelerações por Zona")
                            _av_c3, _av_c4 = st.columns(2)

                            _ZONAS_ACC = [
                                ('Muito intensa (>3)', 3.0, 99, '#F44336'),
                                ('Intensa (2–3)',       2.0, 3.0,'#FF9800'),
                                ('Moderada (1–2)',      1.0, 2.0,'#FFEB3B'),
                                ('Leve (0–1)',          0.0, 1.0,'#4CAF50'),
                                ('Desac. leve (0–-1)', -1.0, 0.0,'#26C6DA'),
                                ('Desac. mod. (-1–-2)',-2.0,-1.0,'#1565C0'),
                                ('Desac. int. (-2–-3)',-3.0,-2.0,'#7B1FA2'),
                                ('Desac. muito int. (<-3)',-99,-3.0,'#880E4F'),
                            ]

                            with _av_c3:
                                _fig_hist = go.Figure()
                                for _av_atl, _d in _av_dados.items():
                                    _fig_hist.add_trace(go.Histogram(
                                        x=_d['acc'], name=_av_atl,
                                        marker_color=_av_cores[_av_atl],
                                        opacity=0.65, nbinsx=60,
                                        hovertemplate='Acc: %{x:.2f} m/s²<br>Freq: %{y}<extra></extra>',
                                    ))
                                _fig_hist.update_layout(
                                    title=dict(text='Histograma de Aceleração', font=dict(color='white',size=13)),
                                    barmode='overlay',
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(title='Aceleração (m/s²)', gridcolor='#333'),
                                    yaxis=dict(title='Frequência', gridcolor='#333'),
                                    legend=dict(font=dict(color='white')),
                                    height=320, margin=dict(t=45,b=10,l=10,r=10),
                                )
                                st.plotly_chart(_fig_hist, use_container_width=True)

                            with _av_c4:
                                # Proporção por zona para o primeiro atleta selecionado
                                _av_atl_zona = _av_atls_sel[0]
                                if _av_atl_zona in _av_dados:
                                    _acc_z = _av_dados[_av_atl_zona]['acc']
                                    _zona_counts, _zona_labels, _zona_cores = [], [], []
                                    for _zl, _zmin, _zmax, _zc in _ZONAS_ACC:
                                        _n = int(np.sum((_acc_z >= _zmin) & (_acc_z < _zmax)))
                                        _zona_counts.append(_n)
                                        _zona_labels.append(_zl)
                                        _zona_cores.append(_zc)
                                    _fig_pie = go.Figure(go.Pie(
                                        labels=_zona_labels, values=_zona_counts,
                                        marker=dict(colors=_zona_cores),
                                        textfont=dict(color='white', size=10),
                                        hole=0.4,
                                    ))
                                    _fig_pie.update_layout(
                                        title=dict(text=f'Distribuição por Zona — {_av_atl_zona}',
                                                   font=dict(color='white',size=13)),
                                        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        legend=dict(font=dict(color='white',size=9)),
                                        height=320, margin=dict(t=45,b=10,l=10,r=10),
                                    )
                                    st.plotly_chart(_fig_pie, use_container_width=True)

                            st.markdown("---")

                            # ════════════════════════════════════════════════
                            # SEÇÃO 5 — DENSIDADE NO ESPAÇO ACC × VEL (heatmap 2D)
                            # ════════════════════════════════════════════════
                            st.markdown("### 🌡️ Mapa de Densidade Acc × Vel")
                            st.caption(
                                "Mostra onde o atleta passa a maior parte do tempo no espaço aceleração-velocidade. "
                                "Regiões quentes = maior acúmulo de esforço."
                            )
                            _av_atl_hm = st.selectbox("Atleta para mapa de densidade:",
                                                       _av_atls_sel, key="av_hm_atl")
                            if _av_atl_hm in _av_dados:
                                _v_hm = _av_dados[_av_atl_hm]['vel']
                                _a_hm = _av_dados[_av_atl_hm]['acc']
                                _H2d, _xe2, _ye2 = np.histogram2d(
                                    _v_hm, _a_hm, bins=[50, 40],
                                    range=[[0, max(35, float(_v_hm.max()))], [-6, 6]]
                                )
                                _H2d = _gf(_H2d, sigma=1.5)
                                _xc2 = (_xe2[:-1] + _xe2[1:]) / 2
                                _yc2 = (_ye2[:-1] + _ye2[1:]) / 2
                                _fig_hm2 = go.Figure(go.Heatmap(
                                    x=_xc2, y=_yc2, z=_H2d.T,
                                    colorscale=[[0,'rgba(0,0,0,0)'],[0.0001,'#0D47A1'],
                                                [0.3,'#1565C0'],[0.6,'#FFEB3B'],
                                                [0.85,'#FF9800'],[1,'#F44336']],
                                    opacity=0.85, showscale=True,
                                    colorbar=dict(
                                        title=dict(text='Densidade', font=dict(color='white')),
                                        tickfont=dict(color='white'),
                                    ),
                                    hovertemplate='Vel: %{x:.1f} km/h<br>Acc: %{y:.2f} m/s²<br>Freq: %{z:.0f}<extra></extra>',
                                ))
                                _fig_hm2.add_hline(y=0, line=dict(color='white', width=1, dash='dot'))
                                _fig_hm2.add_hline(y=3,  line=dict(color='#F44336', width=1, dash='dash'))
                                _fig_hm2.add_hline(y=-3, line=dict(color='#F44336', width=1, dash='dash'))
                                _fig_hm2.update_layout(
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(title='Velocidade (km/h)', gridcolor='#333'),
                                    yaxis=dict(title='Aceleração (m/s²)', gridcolor='#333'),
                                    height=400, margin=dict(t=20,b=10),
                                )
                                st.plotly_chart(_fig_hm2, use_container_width=True)

                    else:
                        st.info("Selecione pelo menos 1 atleta para ver o perfil Acc × Vel.")

                    # ════════════════════════════════════════════════════════
                    # #14 — QUALIDADE DE ACELERAÇÃO + #11 — PL DIRECIONAL
                    # ════════════════════════════════════════════════════════
                    if _av_atletas_disp:
                        st.markdown("---")

                        # ── #14: Qualidade dos Eventos de Aceleração ──────────
                        st.markdown("### 🔬 Qualidade de Aceleração")
                        st.caption(
                            "Cada evento de aceleração (cruzar +2 m/s² por ≥0.3 s) é "
                            "caracterizado por: pico, impulso (área sob a curva), taxa de "
                            "desenvolvimento e velocidade de entrada. Mostra SE os esforços "
                            "mantêm qualidade ao longo do jogo."
                        )
                        _qa_atl = st.selectbox("Atleta:", _av_atletas_disp, key="qa_atleta")
                        if _av_per == _AV_TODOS:
                            _qa_pts: list = []
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _qa_pts += _pv.get(_qa_atl, [])
                        else:
                            _qa_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_qa_atl, [])

                        _qa_vs = np.array([float(p.get('v') or 0) * 3.6 for p in _qa_pts])
                        _qa_as = np.array([float(p.get('a') or 0) for p in _qa_pts])
                        _qa_ts = np.array([float(p.get('ts') or 0) for p in _qa_pts])

                        if len(_qa_as) > 10:
                            # Detectar eventos de aceleração: a > 2 m/s² por ≥ 3 amostras (0.3 s a 10 Hz)
                            _qa_eventos = []
                            _qa_in_event = False
                            _qa_start = 0
                            _qa_THRESH = 2.0
                            _qa_MIN_DUR = 3  # amostras
                            for _qi in range(len(_qa_as)):
                                if not _qa_in_event and _qa_as[_qi] >= _qa_THRESH:
                                    _qa_in_event = True
                                    _qa_start = _qi
                                elif _qa_in_event and (_qa_as[_qi] < _qa_THRESH or _qi == len(_qa_as)-1):
                                    _qa_end = _qi
                                    if _qa_end - _qa_start >= _qa_MIN_DUR:
                                        _seg_a = _qa_as[_qa_start:_qa_end]
                                        _seg_v = _qa_vs[_qa_start:_qa_end]
                                        _seg_t = _qa_ts[_qa_start:_qa_end]
                                        _qa_eventos.append({
                                            'inicio_min': (_seg_t[0] - _qa_ts[0]) / 60,
                                            'pico_a': float(np.max(_seg_a)),
                                            'impulso': float(np.trapezoid(_seg_a, dx=0.1)),
                                            'tdr': float((np.max(_seg_a) - _seg_a[0]) / max(0.1, (_seg_t[np.argmax(_seg_a)] - _seg_t[0]))),
                                            'vel_entrada': float(_seg_v[0]),
                                            'duracao_s': float(len(_seg_a) * 0.1),
                                        })
                                    _qa_in_event = False

                            if _qa_eventos:
                                _df_qa = pd.DataFrame(_qa_eventos)
                                _qa_kc1, _qa_kc2, _qa_kc3, _qa_kc4 = st.columns(4)
                                _qa_kc1.metric("Total de Eventos", len(_qa_eventos))
                                _qa_kc2.metric("Pico de Aceleração", f"{_df_qa['pico_a'].max():.2f} m/s²")
                                _qa_kc3.metric("Impulso Médio", f"{_df_qa['impulso'].mean():.2f} m/s")
                                _qa_kc4.metric("Vel. Entrada Média", f"{_df_qa['vel_entrada'].mean():.1f} km/h")

                                _qa_c1, _qa_c2 = st.columns(2)
                                with _qa_c1:
                                    # Scatter: tempo x pico_a colorido por impulso
                                    _fig_qa_sc = go.Figure()
                                    _fig_qa_sc.add_trace(go.Scatter(
                                        x=_df_qa['inicio_min'], y=_df_qa['pico_a'],
                                        mode='markers',
                                        marker=dict(
                                            size=_df_qa['impulso'].clip(3, 20),
                                            color=_df_qa['impulso'],
                                            colorscale='RdYlGn_r',
                                            showscale=True,
                                            colorbar=dict(title=dict(text='Impulso (m/s)', font=dict(color='white')),
                                                          tickfont=dict(color='white')),
                                        ),
                                        customdata=_df_qa[['impulso','vel_entrada','duracao_s']].values,
                                        hovertemplate=(
                                            'Min: %{x:.1f}<br>'
                                            'Pico Acc: %{y:.2f} m/s²<br>'
                                            'Impulso: %{customdata[0]:.2f} m/s<br>'
                                            'Vel entrada: %{customdata[1]:.1f} km/h<br>'
                                            'Duração: %{customdata[2]:.2f} s<extra></extra>'
                                        ),
                                    ))
                                    # linha de tendência do pico_a ao longo do jogo
                                    if len(_df_qa) >= 4:
                                        _qa_z = np.polyfit(_df_qa['inicio_min'], _df_qa['pico_a'], 1)
                                        _qa_xfit = np.linspace(_df_qa['inicio_min'].min(), _df_qa['inicio_min'].max(), 50)
                                        _fig_qa_sc.add_trace(go.Scatter(
                                            x=_qa_xfit, y=np.polyval(_qa_z, _qa_xfit),
                                            mode='lines', name='Tendência',
                                            line=dict(color='#FFD700', width=2, dash='dash'),
                                        ))
                                    _fig_qa_sc.update_layout(
                                        title=dict(text='Pico de Aceleração ao Longo do Jogo', font=dict(color='white', size=13)),
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='Pico de Aceleração (m/s²)', gridcolor='#333'),
                                        height=340, margin=dict(t=45,b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_qa_sc, use_container_width=True)

                                with _qa_c2:
                                    # Histograma de velocidade de entrada nos sprints
                                    _fig_qa_hist = go.Figure()
                                    _fig_qa_hist.add_trace(go.Histogram(
                                        x=_df_qa['vel_entrada'], nbinsx=12,
                                        marker=dict(color=cor_atleta(_qa_atl), opacity=0.8,
                                                    line=dict(color='white', width=0.5)),
                                        name='Vel. Entrada',
                                        hovertemplate='%{x:.1f}–%{x:.1f} km/h: %{y} eventos<extra></extra>',
                                    ))
                                    _fig_qa_hist.update_layout(
                                        title=dict(text='Velocidade de Entrada nos Eventos de Acc.', font=dict(color='white', size=13)),
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Velocidade (km/h)', gridcolor='#333'),
                                        yaxis=dict(title='Nº de Eventos', gridcolor='#333'),
                                        height=340, margin=dict(t=45,b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_qa_hist, use_container_width=True)
                            else:
                                st.info("Nenhum evento de aceleração >2 m/s² detectado para este atleta/período.")

                        st.markdown("---")

                        # ── Qualidade dos Eventos de Desaceleração ────────────
                        st.markdown("### 🔴 Qualidade de Desaceleração")
                        st.caption(
                            "Cada evento de desaceleração (cruzar −2 m/s² por ≥0.3 s) é "
                            "caracterizado por: pico, impulso (área sob a curva), velocidade "
                            "de entrada e duração. Mostra SE os esforços de frenagem mantêm "
                            "qualidade ao longo do jogo."
                        )
                        _qd_atl = st.selectbox("Atleta:", _av_atletas_disp, key="qd_atleta")
                        if _av_per == _AV_TODOS:
                            _qd_pts: list = []
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _qd_pts += _pv.get(_qd_atl, [])
                        else:
                            _qd_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_qd_atl, [])

                        _qd_vs = np.array([float(p.get('v') or 0) * 3.6 for p in _qd_pts])
                        _qd_as = np.array([float(p.get('a') or 0) for p in _qd_pts])
                        _qd_ts = np.array([float(p.get('ts') or 0) for p in _qd_pts])

                        if len(_qd_as) > 10:
                            # Detectar eventos de desaceleração: a < -2 m/s² por ≥ 3 amostras
                            _qd_eventos = []
                            _qd_in_event = False
                            _qd_start = 0
                            _qd_THRESH = 2.0   # limiar absoluto
                            _qd_MIN_DUR = 3    # amostras mínimas
                            for _qi in range(len(_qd_as)):
                                if not _qd_in_event and _qd_as[_qi] <= -_qd_THRESH:
                                    _qd_in_event = True
                                    _qd_start = _qi
                                elif _qd_in_event and (_qd_as[_qi] > -_qd_THRESH or _qi == len(_qd_as) - 1):
                                    _qd_end = _qi
                                    if _qd_end - _qd_start >= _qd_MIN_DUR:
                                        _seg_ad = _qd_as[_qd_start:_qd_end]
                                        _seg_vd = _qd_vs[_qd_start:_qd_end]
                                        _seg_td = _qd_ts[_qd_start:_qd_end]
                                        _qd_eventos.append({
                                            'inicio_min': (_seg_td[0] - _qd_ts[0]) / 60,
                                            'pico_d': float(abs(np.min(_seg_ad))),   # valor absoluto do pico
                                            'impulso': float(np.trapezoid(np.abs(_seg_ad), dx=0.1)),
                                            'vel_entrada': float(_seg_vd[0]),
                                            'duracao_s': float(len(_seg_ad) * 0.1),
                                        })
                                    _qd_in_event = False

                            if _qd_eventos:
                                _df_qd = pd.DataFrame(_qd_eventos)
                                _qd_kc1, _qd_kc2, _qd_kc3, _qd_kc4 = st.columns(4)
                                _qd_kc1.metric("Total de Eventos",      len(_qd_eventos))
                                _qd_kc2.metric("Pico de Desaceleração", f"{_df_qd['pico_d'].max():.2f} m/s²")
                                _qd_kc3.metric("Impulso Médio",         f"{_df_qd['impulso'].mean():.2f} m/s")
                                _qd_kc4.metric("Vel. Entrada Média",    f"{_df_qd['vel_entrada'].mean():.1f} km/h")

                                _qd_c1, _qd_c2 = st.columns(2)
                                with _qd_c1:
                                    # Scatter: tempo × pico_d colorido por impulso
                                    _fig_qd_sc = go.Figure()
                                    _fig_qd_sc.add_trace(go.Scatter(
                                        x=_df_qd['inicio_min'], y=_df_qd['pico_d'],
                                        mode='markers',
                                        marker=dict(
                                            size=_df_qd['impulso'].clip(3, 20),
                                            color=_df_qd['impulso'],
                                            colorscale='RdYlGn_r',
                                            showscale=True,
                                            colorbar=dict(
                                                title=dict(text='Impulso (m/s)', font=dict(color='white')),
                                                tickfont=dict(color='white'),
                                            ),
                                        ),
                                        customdata=_df_qd[['impulso', 'vel_entrada', 'duracao_s']].values,
                                        hovertemplate=(
                                            'Min: %{x:.1f}<br>'
                                            'Pico Dec: %{y:.2f} m/s²<br>'
                                            'Impulso: %{customdata[0]:.2f} m/s<br>'
                                            'Vel entrada: %{customdata[1]:.1f} km/h<br>'
                                            'Duração: %{customdata[2]:.2f} s<extra></extra>'
                                        ),
                                    ))
                                    # Linha de tendência
                                    if len(_df_qd) >= 4:
                                        _qd_z = np.polyfit(_df_qd['inicio_min'], _df_qd['pico_d'], 1)
                                        _qd_xfit = np.linspace(_df_qd['inicio_min'].min(),
                                                               _df_qd['inicio_min'].max(), 50)
                                        _fig_qd_sc.add_trace(go.Scatter(
                                            x=_qd_xfit, y=np.polyval(_qd_z, _qd_xfit),
                                            mode='lines', name='Tendência',
                                            line=dict(color='#FFD700', width=2, dash='dash'),
                                        ))
                                    _fig_qd_sc.update_layout(
                                        title=dict(text='Pico de Desaceleração ao Longo do Jogo',
                                                   font=dict(color='white', size=13)),
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='Pico de Desaceleração (m/s²)', gridcolor='#333'),
                                        height=340, margin=dict(t=45, b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_qd_sc, use_container_width=True)

                                with _qd_c2:
                                    # Histograma de velocidade de entrada
                                    _fig_qd_hist = go.Figure()
                                    _fig_qd_hist.add_trace(go.Histogram(
                                        x=_df_qd['vel_entrada'], nbinsx=12,
                                        marker=dict(color=cor_atleta(_qd_atl), opacity=0.8,
                                                    line=dict(color='white', width=0.5)),
                                        name='Vel. Entrada',
                                        hovertemplate='%{x:.1f} km/h: %{y} eventos<extra></extra>',
                                    ))
                                    _fig_qd_hist.update_layout(
                                        title=dict(text='Velocidade de Entrada nos Eventos de Dec.',
                                                   font=dict(color='white', size=13)),
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Velocidade (km/h)', gridcolor='#333'),
                                        yaxis=dict(title='Nº de Eventos', gridcolor='#333'),
                                        height=340, margin=dict(t=45, b=10), showlegend=False,
                                    )
                                    st.plotly_chart(_fig_qd_hist, use_container_width=True)
                            else:
                                st.info("Nenhum evento de desaceleração <−2 m/s² detectado para este atleta/período.")

                        st.markdown("---")

                        # ── #11: Player Load Direcional ───────────────────────
                        st.markdown("### 📐 Player Load Direcional")
                        st.caption(
                            "Decompõe o PlayerLoad nos 3 eixos: **Anteroposterior** (frente/trás — corrida "
                            "em linha), **Mediolateral** (esquerda/direita — movimentos defensivos/laterais) "
                            "e **Vertical** (saltos, duelos aéreos, contatos). "
                            "Parâmetros: `pla`, `plml`, `plv` do sensor Catapult."
                        )
                        _pld_atl = st.selectbox("Atleta:", _av_atletas_disp, key="pld_atleta")
                        if _av_per == _AV_TODOS:
                            _pld_pts: list = []
                            for _pv in dados_sensor_por_atleta_por_periodo.values():
                                _pld_pts += _pv.get(_pld_atl, [])
                        else:
                            _pld_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_pld_atl, [])

                        _pld_ap  = sum(float(p.get('pla')  or 0) for p in _pld_pts)
                        _pld_ml  = sum(float(p.get('plml') or 0) for p in _pld_pts)
                        _pld_vt  = sum(float(p.get('plv')  or 0) for p in _pld_pts)
                        _pld_tot = _pld_ap + _pld_ml + _pld_vt

                        # fallback: estimar AP/ML/V a partir de aceleração se sensor não tiver eixos
                        if _pld_tot < 0.01 and _pld_pts:
                            _est_note = True
                            _ap_sum = _ml_sum = _vt_sum = 0.0
                            for _pp in _pld_pts:
                                _av_vel = abs(float(_pp.get('v') or 0))
                                _av_acc = abs(float(_pp.get('a') or 0))
                                # heurística: se velocidade alta → contribuição AP; se baixo → ML
                                _ap_sum += _av_vel * 0.1
                                _ml_sum += max(0, _av_acc - _av_vel * 0.05) * 0.1
                            _pld_ap  = _ap_sum
                            _pld_ml  = _ml_sum
                            _pld_vt  = max(0.0, _pld_ap * 0.15)
                            _pld_tot = _pld_ap + _pld_ml + _pld_vt
                        else:
                            _est_note = False

                        if _pld_tot > 0:
                            if _est_note:
                                st.caption("⚠️ Sensor sem dados de eixo — estimativa heurística.")
                            _pld_c1, _pld_c2 = st.columns(2)
                            with _pld_c1:
                                # KPIs
                                _p1, _p2, _p3 = st.columns(3)
                                _p1.metric("Anteroposterior", f"{_pld_ap:.1f}", f"{_pld_ap/_pld_tot*100:.0f}%")
                                _p2.metric("Mediolateral",   f"{_pld_ml:.1f}", f"{_pld_ml/_pld_tot*100:.0f}%")
                                _p3.metric("Vertical",       f"{_pld_vt:.1f}", f"{_pld_vt/_pld_tot*100:.0f}%")
                                # Pizza
                                _fig_pld_pie = go.Figure(go.Pie(
                                    labels=['Anteroposterior', 'Mediolateral', 'Vertical'],
                                    values=[round(_pld_ap,1), round(_pld_ml,1), round(_pld_vt,1)],
                                    marker=dict(colors=['#2196F3','#4CAF50','#FF9800']),
                                    textinfo='label+percent',
                                    hole=0.42,
                                    hovertemplate='%{label}<br>PL: %{value:.1f}<br>%{percent}<extra></extra>',
                                ))
                                _fig_pld_pie.update_layout(
                                    title=dict(text=f'Distribuição de PL — {_pld_atl}', font=dict(color='white', size=13)),
                                    paper_bgcolor='#0e1117', font=dict(color='white'),
                                    legend=dict(font=dict(color='white', size=10)),
                                    height=320, margin=dict(t=50,b=10,l=10,r=10),
                                )
                                st.plotly_chart(_fig_pld_pie, use_container_width=True)

                            with _pld_c2:
                                # Comparativo entre todos os atletas
                                _pld_rows = []
                                for _pa in _av_atletas_disp:
                                    if _av_per == _AV_TODOS:
                                        _pa_pts: list = []
                                        for _ppv in dados_sensor_por_atleta_por_periodo.values():
                                            _pa_pts += _ppv.get(_pa, [])
                                    else:
                                        _pa_pts = dados_sensor_por_atleta_por_periodo.get(_av_per, {}).get(_pa, [])
                                    _a = sum(float(p.get('pla') or 0)  for p in _pa_pts)
                                    _m = sum(float(p.get('plml') or 0) for p in _pa_pts)
                                    _v = sum(float(p.get('plv') or 0)  for p in _pa_pts)
                                    _t = _a+_m+_v
                                    if _t < 0.01 and _pa_pts:
                                        _a = sum(abs(float(p.get('v') or 0))*0.1 for p in _pa_pts)
                                        _m = sum(max(0,abs(float(p.get('a') or 0))-abs(float(p.get('v') or 0))*0.05)*0.1 for p in _pa_pts)
                                        _v = _a*0.15; _t = _a+_m+_v
                                    if _t > 0:
                                        _pld_rows.append({'Atleta':_pa,'AP':round(_a/_t*100,1),'ML':round(_m/_t*100,1),'VT':round(_v/_t*100,1)})
                                if _pld_rows:
                                    _df_pld = pd.DataFrame(_pld_rows)
                                    _fig_pld_bar = go.Figure()
                                    for _col_lbl, _col_color in [('AP','#2196F3'),('ML','#4CAF50'),('VT','#FF9800')]:
                                        _fig_pld_bar.add_trace(go.Bar(
                                            x=_df_pld['Atleta'], y=_df_pld[_col_lbl],
                                            name=_col_lbl, marker_color=_col_color,
                                            hovertemplate=f'{_col_lbl}: %{{y:.1f}}%<extra></extra>',
                                        ))
                                    _fig_pld_bar.update_layout(
                                        title=dict(text='% PL Direcional por Atleta', font=dict(color='white',size=13)),
                                        barmode='stack', paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(gridcolor='#333', tickangle=-30),
                                        yaxis=dict(title='% PL', gridcolor='#333'),
                                        legend=dict(font=dict(color='white')),
                                        height=320, margin=dict(t=50,b=10),
                                    )
                                    st.plotly_chart(_fig_pld_bar, use_container_width=True)
                        else:
                            st.info("Dados de PL direcional não disponíveis para este atleta/período.")
                else:
                    st.info("Busque os dados de atletas na sessão antes de usar esta análise.")

            # ══════════════════════════════════════════════════════════════
            # ABA 6: FC — TRIMP & ZONAS DE FREQUÊNCIA CARDÍACA
            # ══════════════════════════════════════════════════════════════
            with abas[5]:
                st.subheader("❤️ Frequência Cardíaca & TRIMP")
                st.caption(
                    "**TRIMP** (Training Impulse — Edwards): carga interna calculada pelo tempo em cada zona de "
                    "FC (50–60 / 60–70 / 70–80 / 80–90 / 90–100% HRmax) com multiplicadores 1–2–3–4–5. "
                    "Referência: FC máx = 220 − idade (padrão 180 bpm se não configurada)."
                )

                _fc_periodos = list(dados_sensor_por_atleta_por_periodo.keys())
                if _fc_periodos and resultados_por_periodo:
                    _FC_TODOS = "🔀 Todos os períodos (combinado)"
                    _fc_opcoes_per = [_FC_TODOS] + _fc_periodos

                    _fc_c1, _fc_c2, _fc_c3 = st.columns([2, 1, 1])
                    with _fc_c1:
                        _fc_per = st.selectbox("Período:", _fc_opcoes_per, key="fc_periodo")
                    with _fc_c2:
                        if _fc_per == _FC_TODOS:
                            _fc_ats_set = set()
                            for _fpv in dados_sensor_por_atleta_por_periodo.values():
                                _fc_ats_set.update(_fpv.keys())
                            _fc_atletas_disp = sorted(_fc_ats_set)
                        else:
                            _fc_atletas_disp = list(dados_sensor_por_atleta_por_periodo.get(_fc_per, {}).keys())
                        _fc_atl = st.selectbox("Atleta:", _fc_atletas_disp, key="fc_atleta_sel") if _fc_atletas_disp else None
                    with _fc_c3:
                        _fc_hrmax = st.number_input(
                            "FC Máx (bpm):", min_value=150, max_value=220,
                            value=180, step=1, key="fc_hrmax",
                            help="Padrão: 180 bpm. Ajuste para o valor individual do atleta."
                        )

                    if _fc_atl:
                        # Coletar pontos de sensor
                        if _fc_per == _FC_TODOS:
                            _fc_pts: list = []
                            for _fpv in dados_sensor_por_atleta_por_periodo.values():
                                _fc_pts += _fpv.get(_fc_atl, [])
                        else:
                            _fc_pts = dados_sensor_por_atleta_por_periodo.get(_fc_per, {}).get(_fc_atl, [])

                        _hr_vals = [float(p['hr']) for p in _fc_pts if p.get('hr') and float(p.get('hr') or 0) > 30]

                        if _hr_vals:
                            # ── Zonas Edwards ──────────────────────────────────────
                            _fc_zona_bounds = [
                                (0.50, 0.60, 1, 'Z1  50–60%', '#4CAF50'),
                                (0.60, 0.70, 2, 'Z2  60–70%', '#8BC34A'),
                                (0.70, 0.80, 3, 'Z3  70–80%', '#FFEB3B'),
                                (0.80, 0.90, 4, 'Z4  80–90%', '#FF9800'),
                                (0.90, 1.00, 5, 'Z5  90–100%', '#F44336'),
                            ]
                            _fc_zona_counts = {z[3]: 0 for z in _fc_zona_bounds}
                            _fc_trimp_total = 0.0
                            _fc_dt_s = 0.1  # 10 Hz → 0.1 s por ponto
                            for _hv in _hr_vals:
                                _hpct = _hv / _fc_hrmax
                                for _lo, _hi, _mult, _lbl, _col in _fc_zona_bounds:
                                    if _lo <= _hpct < _hi or (_hi == 1.00 and _hpct >= _lo):
                                        _fc_zona_counts[_lbl] += 1
                                        _fc_trimp_total += _fc_dt_s / 60 * _mult
                                        break

                            # ── KPIs ─────────────────────────────────────────────
                            _fk1, _fk2, _fk3, _fk4, _fk5 = st.columns(5)
                            _fk1.metric("TRIMP Total", f"{_fc_trimp_total:.1f}",
                                        help="Training Impulse (Edwards, 1993)")
                            _fk2.metric("FC Média (bpm)", f"{np.mean(_hr_vals):.0f}")
                            _fk3.metric("FC Máx Atingida", f"{max(_hr_vals):.0f} bpm")
                            _fk4.metric("% FCmax", f"{max(_hr_vals)/_fc_hrmax*100:.1f}%")
                            _fc_tempo_acima_80 = sum(
                                1 for hv in _hr_vals if hv / _fc_hrmax >= 0.80
                            ) * _fc_dt_s / 60
                            _fk5.metric("Tempo > 80% FCmax", f"{_fc_tempo_acima_80:.1f} min")

                            st.markdown("---")
                            _fc_chart_c1, _fc_chart_c2 = st.columns(2)

                            # ── Pizza — distribuição de zonas ─────────────────────
                            with _fc_chart_c1:
                                _fz_lbls = [z[3] for z in _fc_zona_bounds]
                                _fz_vals = [_fc_zona_counts[l] * _fc_dt_s / 60 for l in _fz_lbls]
                                _fz_cols = [z[4] for z in _fc_zona_bounds]
                                _fig_fz = go.Figure(go.Pie(
                                    labels=_fz_lbls, values=_fz_vals,
                                    marker=dict(colors=_fz_cols),
                                    textinfo='label+percent',
                                    hovertemplate='%{label}<br>%{value:.1f} min<extra></extra>',
                                    hole=0.38,
                                ))
                                _fig_fz.update_layout(
                                    title=dict(text=f'Distribuição de Zonas de FC — {_fc_atl}',
                                               font=dict(color='white', size=13)),
                                    paper_bgcolor='#0e1117', font=dict(color='white'),
                                    legend=dict(font=dict(color='white', size=9)),
                                    height=340, margin=dict(t=50, b=10, l=10, r=10),
                                )
                                st.plotly_chart(_fig_fz, use_container_width=True)

                            # ── Barras empilhadas TRIMP por zona ─────────────────
                            with _fc_chart_c2:
                                _fz_trimp = [_fc_zona_counts[l] * _fc_dt_s / 60 * _mult
                                             for l, (_, _, _mult, _, _) in zip(_fz_lbls, _fc_zona_bounds)]
                                _fig_trimp_fc = go.Figure()
                                for _fl, _ft, _fc_col in zip(_fz_lbls, _fz_trimp, _fz_cols):
                                    _fig_trimp_fc.add_trace(go.Bar(
                                        x=[_fc_atl], y=[_ft], name=_fl,
                                        marker_color=_fc_col,
                                        hovertemplate=f'{_fl}: %{{y:.1f}} TRIMP<extra></extra>',
                                    ))
                                _fig_trimp_fc.update_layout(
                                    title=dict(text=f'TRIMP por Zona — {_fc_atl}',
                                               font=dict(color='white', size=13)),
                                    barmode='stack',
                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(title='TRIMP (u.a.)', gridcolor='#333'),
                                    legend=dict(font=dict(color='white', size=9)),
                                    height=340, margin=dict(t=50, b=10, l=10, r=10),
                                )
                                st.plotly_chart(_fig_trimp_fc, use_container_width=True)

                            # ── Curva de FC ao longo do tempo ────────────────────
                            with st.expander("📈 Curva de FC ao longo do tempo", expanded=True):
                                _fc_ts = [float(p.get('ts') or 0) for p in _fc_pts
                                          if p.get('hr') and float(p.get('hr') or 0) > 30]
                                if _fc_ts:
                                    _fc_t0 = _fc_ts[0]
                                    _fc_ts_rel = [(t - _fc_t0) / 60 for t in _fc_ts]
                                    _fig_fc_curve = go.Figure()
                                    _fig_fc_curve.add_trace(go.Scatter(
                                        x=_fc_ts_rel, y=_hr_vals,
                                        mode='lines', line=dict(color='#F44336', width=1.5),
                                        name='FC (bpm)',
                                        hovertemplate='%{x:.1f} min — %{y:.0f} bpm<extra></extra>',
                                    ))
                                    # Linhas de zona
                                    for _lo, _hi, _mult, _lbl, _fcol in _fc_zona_bounds:
                                        _fig_fc_curve.add_hline(
                                            y=_lo * _fc_hrmax,
                                            line=dict(color=_fcol, width=1, dash='dot'),
                                            annotation_text=_lbl,
                                            annotation_font=dict(color=_fcol, size=9),
                                        )
                                    _fig_fc_curve.update_layout(
                                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                        yaxis=dict(title='FC (bpm)', gridcolor='#333',
                                                   range=[40, _fc_hrmax * 1.05]),
                                        height=320, margin=dict(t=20, b=10),
                                        showlegend=False,
                                    )
                                    st.plotly_chart(_fig_fc_curve, use_container_width=True)

                            # ── Comparativo TRIMP entre todos os atletas ──────────
                            st.markdown("---")
                            st.markdown("### 📊 Comparativo de TRIMP — Todos os Atletas")
                            _fc_comp_rows = []
                            for _fc_cp_atl in _fc_atletas_disp:
                                if _fc_per == _FC_TODOS:
                                    _fc_cp_pts: list = []
                                    for _fpv in dados_sensor_por_atleta_por_periodo.values():
                                        _fc_cp_pts += _fpv.get(_fc_cp_atl, [])
                                else:
                                    _fc_cp_pts = dados_sensor_por_atleta_por_periodo.get(_fc_per, {}).get(_fc_cp_atl, [])
                                _fc_cp_hr = [float(p['hr']) for p in _fc_cp_pts if p.get('hr') and float(p.get('hr') or 0) > 30]
                                if _fc_cp_hr:
                                    _fc_cp_trimp = 0.0
                                    _fc_cp_z = {z[3]: 0 for z in _fc_zona_bounds}
                                    for _hv2 in _fc_cp_hr:
                                        _hp2 = _hv2 / _fc_hrmax
                                        for _lo2, _hi2, _m2, _l2, _c2 in _fc_zona_bounds:
                                            if _lo2 <= _hp2 < _hi2 or (_hi2 == 1.00 and _hp2 >= _lo2):
                                                _fc_cp_z[_l2] += 1
                                                _fc_cp_trimp += _fc_dt_s / 60 * _m2
                                                break
                                    _fc_comp_rows.append({
                                        'Atleta': _fc_cp_atl,
                                        'TRIMP': round(_fc_cp_trimp, 1),
                                        'FC Média (bpm)': round(float(np.mean(_fc_cp_hr)), 0),
                                        'FC Máx (bpm)': round(float(max(_fc_cp_hr)), 0),
                                        **{l: round(_fc_cp_z[l] * _fc_dt_s / 60, 2) for l in _fc_cp_z}
                                    })
                            if _fc_comp_rows:
                                _df_fc_comp = pd.DataFrame(_fc_comp_rows).sort_values('TRIMP', ascending=False)
                                st.dataframe(_df_fc_comp, use_container_width=True, hide_index=True)
                                # Gráfico de barras TRIMP comparativo
                                _fig_fc_comp_bar = px.bar(
                                    _df_fc_comp, x='Atleta', y='TRIMP',
                                    color='TRIMP', color_continuous_scale='Reds',
                                    title='TRIMP por Atleta',
                                )
                                _fig_fc_comp_bar.update_layout(
                                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(title='TRIMP (u.a.)', gridcolor='#333'),
                                    height=350, margin=dict(t=40, b=10),
                                    showlegend=False,
                                )
                                st.plotly_chart(_fig_fc_comp_bar, use_container_width=True)
                        else:
                            st.info("ℹ️ Nenhum dado de FC disponível para este atleta/período.")
                    else:
                        st.info("Selecione um atleta para analisar a FC.")
                else:
                    st.info("Carregue dados da sessão para visualizar a análise de FC.")

                # ── #3: Curva de Fadiga Intra-Jogo ───────────────────────────
                st.markdown("---")
                st.markdown("## 📉 Curva de Fadiga Intra-Jogo")
                st.caption(
                    "Divide cada período em janelas de **5 minutos** e calcula a intensidade "
                    "relativa (m/min e PL/min). Uma linha de tendência negativa indica fadiga "
                    "progressiva. O minuto de início da queda é detectado automaticamente."
                )
                _fat_periodos = [p for p in dados_sensor_por_atleta_por_periodo if p != _CHAVE_COMBINADO]
                if _fat_periodos and resultados_por_periodo:
                    _fat_per = st.selectbox("Período:", _fat_periodos, key="fat_periodo")
                    _fat_atls_disp = list(dados_sensor_por_atleta_por_periodo.get(_fat_per, {}).keys())
                    if _fat_atls_disp:
                        _fat_sel = st.multiselect(
                            "Atletas (até 8):", _fat_atls_disp,
                            default=_fat_atls_disp[:min(5, len(_fat_atls_disp))],
                            key="fat_atletas",
                        )
                        _fat_janela_min = st.select_slider(
                            "Janela:", options=[2, 3, 5, 10], value=3, key="fat_janela",
                            help="Tamanho da janela temporal para cálculo de intensidade"
                        )
                        _fat_janela_s = _fat_janela_min * 60

                        if _fat_sel:
                            _fig_fat = go.Figure()
                            _fat_declive_info = []

                            for _fa in _fat_sel:
                                _fa_pts = dados_sensor_por_atleta_por_periodo.get(_fat_per, {}).get(_fa, [])
                                if not _fa_pts:
                                    continue
                                _fa_ts = np.array([float(p.get('ts') or 0) for p in _fa_pts])
                                _fa_vs = np.array([float(p.get('v')  or 0) * 3.6 for p in _fa_pts])
                                _fa_pl = np.array([float(p.get('pl') or 0) for p in _fa_pts])
                                if len(_fa_ts) < 2:
                                    continue
                                _fa_t0 = _fa_ts[0]
                                _fa_dur = _fa_ts[-1] - _fa_t0
                                if _fa_dur < _fat_janela_s * 2:
                                    continue

                                # janelas deslizantes
                                _win_centers_min = []
                                _win_mmin = []
                                _win_plmin = []
                                _t_start = _fa_t0
                                while _t_start + _fat_janela_s <= _fa_ts[-1]:
                                    _mask = (_fa_ts >= _t_start) & (_fa_ts < _t_start + _fat_janela_s)
                                    if _mask.sum() > 1:
                                        _dist_win = float(np.sum(np.abs(np.diff(_fa_vs[_mask])) * 0.1 / 3.6 * 3.6))
                                        # distância real via integral v×dt
                                        _d_real = float(np.trapezoid(_fa_vs[_mask] / 3.6, _fa_ts[_mask]))
                                        _pl_sum = float(np.sum(_fa_pl[_mask]))
                                        _win_mmin.append(_d_real / _fat_janela_min)
                                        _win_plmin.append(_pl_sum / _fat_janela_min)
                                        _win_centers_min.append((_t_start - _fa_t0 + _fat_janela_s / 2) / 60)
                                    _t_start += _fat_janela_s / 2  # sobreposição 50%

                                if len(_win_centers_min) < 3:
                                    continue

                                _wc = np.array(_win_centers_min)
                                _wm = np.array(_win_mmin)

                                _fa_color = cor_atleta(_fa)
                                _fig_fat.add_trace(go.Scatter(
                                    x=_wc, y=_wm, mode='lines+markers',
                                    name=_fa,
                                    line=dict(color=_fa_color, width=2),
                                    marker=dict(size=5, color=_fa_color),
                                    hovertemplate='%{x:.1f} min — %{y:.1f} m/min<extra>' + _fa + '</extra>',
                                ))
                                # Linha de tendência
                                if len(_wc) >= 4:
                                    _z = np.polyfit(_wc, _wm, 1)
                                    _xfit = np.linspace(_wc[0], _wc[-1], 50)
                                    _fig_fat.add_trace(go.Scatter(
                                        x=_xfit, y=np.polyval(_z, _xfit),
                                        mode='lines', showlegend=False,
                                        line=dict(color=_fa_color, width=1, dash='dot'),
                                        hoverinfo='skip',
                                    ))
                                    # Detectar início de queda: janela com variação negativa sustentada
                                    _rolling_diff = np.diff(_wm)
                                    _neg_idx = np.where(_rolling_diff < -5)[0]
                                    if len(_neg_idx) > 0:
                                        _queda_min = _wc[_neg_idx[0]]
                                        _fat_declive_info.append((_fa, round(float(_z[0]),2), round(_queda_min,1)))

                            _fig_fat.update_layout(
                                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                                font=dict(color='white'),
                                xaxis=dict(title='Tempo (min)', gridcolor='#333'),
                                yaxis=dict(title='Intensidade (m/min)', gridcolor='#333'),
                                legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.4)'),
                                height=380, margin=dict(t=20,b=10),
                            )
                            st.plotly_chart(_fig_fat, use_container_width=True)

                            # Tabela de declive e início de queda
                            if _fat_declive_info:
                                st.markdown("#### ⚠️ Atletas com Queda Detectada")
                                _df_fat_info = pd.DataFrame(
                                    _fat_declive_info,
                                    columns=['Atleta', 'Declive (m/min por min)', 'Início de Queda (min)']
                                ).sort_values('Declive (m/min por min)')
                                # badge de intensidade do declive
                                def _fat_badge(slope):
                                    if slope < -3:   return "🔴 Queda Severa"
                                    elif slope < -1: return "🟡 Queda Moderada"
                                    elif slope < 0:  return "🟢 Queda Leve"
                                    else:            return "✅ Estável"
                                _df_fat_info['Classificação'] = _df_fat_info['Declive (m/min por min)'].apply(_fat_badge)
                                st.dataframe(_df_fat_info, use_container_width=True, hide_index=True)
                    else:
                        st.info("Nenhum atleta com dados de sensor para este período.")
                else:
                    st.info("Carregue dados para visualizar a curva de fadiga.")

            # ABA 6: POR POSIÇÃO
            # ══════════════════════════════════════════════════════════════
            with abas[6]:
                st.subheader("📊 Análise por Posição")
                st.caption("Média das métricas agrupada por posição tática dos atletas.")

                _pos_periodos = list(resultados_por_periodo.keys())
                if _pos_periodos:
                    _pos_periodo_sel = st.selectbox("Período:", _pos_periodos, key="pos_periodo_sel")

                    if resultados_por_periodo.get(_pos_periodo_sel):
                        _df_pos_raw = pd.DataFrame(resultados_por_periodo[_pos_periodo_sel])

                        if 'Posição' not in _df_pos_raw.columns or _df_pos_raw['Posição'].replace('', np.nan).isna().all():
                            st.warning("⚠️ Dados de posição não disponíveis. Verifique se as posições foram cadastradas na API Catapult.")
                        else:
                            _df_pos_raw = _df_pos_raw[_df_pos_raw['Posição'].notna() & (_df_pos_raw['Posição'] != '')]
                            _df_pos_grp = _df_pos_raw.groupby('Posição', as_index=False).mean(numeric_only=True)

                            # Paleta de cores por posição (usa sistema de grupos táticos)
                            _cores_pos = [
                                _POSICAO_COR_LEGENDA.get(_get_pos_grupo(p)[0], '#546E7A')
                                for p in _df_pos_grp['Posição']
                            ]

                            def _bar_pos(y_col, title, suffix=''):
                                if y_col not in _df_pos_grp.columns:
                                    return None
                                _f = go.Figure(go.Bar(
                                    x=_df_pos_grp['Posição'], y=_df_pos_grp[y_col],
                                    marker_color=_cores_pos, text=_df_pos_grp[y_col].round(1),
                                    textposition='outside',
                                    hovertemplate=f'%{{x}}<br>{y_col}: %{{y:.1f}}{suffix}<extra></extra>',
                                ))
                                _f.update_layout(
                                    title=dict(text=title, font=dict(color='white', size=13)),
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(gridcolor='#333', title=y_col),
                                    height=320, margin=dict(t=45, b=10, l=10, r=10),
                                    showlegend=False,
                                )
                                return _f

                            # Linha 1: Distância Total + M/min
                            _c1, _c2 = st.columns(2)
                            with _c1:
                                _f = _bar_pos('Distância (m)', 'Média de Distância Total por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c2:
                                _f = _bar_pos('M/min', 'Média de M/min por Posição', ' m/min')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # Linha 2: Zona 4 + Zona 5 (>24 km/h)
                            _c3, _c4 = st.columns(2)
                            with _c3:
                                _f = _bar_pos('Dist. 19-24 km/h (m)', 'Média de Zona 4 (19-24 km/h) por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c4:
                                _f = _bar_pos('Dist. > 24 km/h (m)', 'Média de Zona 5 (>24 km/h) por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # Linha 3: HSR (>19) + Sprints
                            _c5, _c6 = st.columns(2)
                            with _c5:
                                _f = _bar_pos('Dist. > 19 km/h (m)', 'Média de HSR Distance (>19 km/h) por Posição', ' m')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c6:
                                _f = _bar_pos('Sprints (>24 km/h)', 'Média de Sprints por Posição')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # Linha 4: Acc 2-3 + Dcc 2-3 (gráfico agrupado)
                            _acc23_col = 'Acc 2-3 (m/s²)'
                            _dcc23_col = 'Dcc 2-3 (m/s²)'
                            if _acc23_col in _df_pos_grp.columns and _dcc23_col in _df_pos_grp.columns:
                                _c7, _c8 = st.columns(2)
                                with _c7:
                                    _fig_acc = go.Figure()
                                    _fig_acc.add_trace(go.Bar(
                                        x=_df_pos_grp['Posição'], y=_df_pos_grp[_acc23_col],
                                        name='Acc 2-3', marker_color='#FFA000',
                                        text=_df_pos_grp[_acc23_col].round(1), textposition='outside',
                                    ))
                                    _fig_acc.add_trace(go.Bar(
                                        x=_df_pos_grp['Posição'], y=_df_pos_grp[_dcc23_col],
                                        name='Dcc 2-3', marker_color='#558B2F',
                                        text=_df_pos_grp[_dcc23_col].round(1), textposition='outside',
                                    ))
                                    _fig_acc.update_layout(
                                        title=dict(text='Média de Acc e Dcc 2-3 por Posição', font=dict(color='white', size=13)),
                                        barmode='group', plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), xaxis=dict(gridcolor='#333'),
                                        yaxis=dict(gridcolor='#333'), height=320,
                                        margin=dict(t=45, b=10, l=10, r=10),
                                        legend=dict(font=dict(color='white')),
                                    )
                                    st.plotly_chart(_fig_acc, use_container_width=True)

                                with _c8:
                                    # Acc >3 + Dcc >3 agrupados
                                    _fig_acc3 = go.Figure()
                                    if 'Acelerações (>3 m/s²)' in _df_pos_grp.columns:
                                        _fig_acc3.add_trace(go.Bar(
                                            x=_df_pos_grp['Posição'], y=_df_pos_grp['Acelerações (>3 m/s²)'],
                                            name='Acc >3', marker_color='#E53935',
                                            text=_df_pos_grp['Acelerações (>3 m/s²)'].round(1), textposition='outside',
                                        ))
                                    if 'Desacelerações (<-3 m/s²)' in _df_pos_grp.columns:
                                        _fig_acc3.add_trace(go.Bar(
                                            x=_df_pos_grp['Posição'], y=_df_pos_grp['Desacelerações (<-3 m/s²)'],
                                            name='Dcc >3', marker_color='#1565C0',
                                            text=_df_pos_grp['Desacelerações (<-3 m/s²)'].round(1), textposition='outside',
                                        ))
                                    _fig_acc3.update_layout(
                                        title=dict(text='Média de Acc e Dcc >3 por Posição', font=dict(color='white', size=13)),
                                        barmode='group', plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                        font=dict(color='white'), xaxis=dict(gridcolor='#333'),
                                        yaxis=dict(gridcolor='#333'), height=320,
                                        margin=dict(t=45, b=10, l=10, r=10),
                                        legend=dict(font=dict(color='white')),
                                    )
                                    st.plotly_chart(_fig_acc3, use_container_width=True)

                            # Linha 5: PlayerLoad + RHIE Blocos
                            _c9, _c10 = st.columns(2)
                            with _c9:
                                _f = _bar_pos('PlayerLoad', 'Média de PlayerLoad por Posição')
                                if _f: st.plotly_chart(_f, use_container_width=True)
                            with _c10:
                                _f = _bar_pos('RHIE Blocos', 'Média de RHIE Blocos por Posição')
                                if _f: st.plotly_chart(_f, use_container_width=True)

                            # ── Item 16: Radar de Perfil por Posição ─────────────
                            st.markdown("---")
                            st.markdown("### 🕸️ Radar de Perfil Físico por Posição")
                            st.caption(
                                "Normalização min-max do grupo: 100% = maior valor do grupo. "
                                "Permite comparar perfis multidimensionais entre posições."
                            )
                            _radar_metrics = [c for c in [
                                'Distância (m)', 'M/min', 'Dist. > 19 km/h (m)',
                                'Sprints (>24 km/h)', 'Acelerações (>3 m/s²)',
                                'Desacelerações (<-3 m/s²)', 'PlayerLoad', 'RHIE Blocos',
                            ] if c in _df_pos_grp.columns]

                            if len(_radar_metrics) >= 4 and len(_df_pos_grp) >= 2:
                                # Normalização min-max por métrica (0-100%)
                                _radar_norm = _df_pos_grp[_radar_metrics].copy()
                                for _rc in _radar_metrics:
                                    _rmin = _radar_norm[_rc].min()
                                    _rmax = _radar_norm[_rc].max()
                                    _radar_norm[_rc] = ((_radar_norm[_rc] - _rmin) / max(_rmax - _rmin, 1e-9) * 100).round(1)

                                _POS_RADAR_PALETTE = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4','#F44336']
                                _fig_radar = go.Figure()
                                for _ri, (_rpos, _rrow) in enumerate(_radar_norm.iterrows()):
                                    _pos_name = _df_pos_grp['Posição'].iloc[_ri]
                                    _vals = list(_rrow[_radar_metrics]) + [_rrow[_radar_metrics[0]]]
                                    _lbls = _radar_metrics + [_radar_metrics[0]]
                                    _col_r = _POS_RADAR_PALETTE[_ri % len(_POS_RADAR_PALETTE)]
                                    _fig_radar.add_trace(go.Scatterpolar(
                                        r=_vals, theta=_lbls, fill='toself',
                                        name=_pos_name,
                                        line=dict(color=_col_r, width=2),
                                        fillcolor=(lambda c: f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)")(_col_r),
                                        opacity=0.85,
                                        hovertemplate='%{theta}: %{r:.1f}%<extra>' + _pos_name + '</extra>',
                                    ))
                                _fig_radar.update_layout(
                                    polar=dict(
                                        bgcolor='#0e1117',
                                        radialaxis=dict(
                                            visible=True, range=[0, 100],
                                            gridcolor='rgba(255,255,255,0.15)',
                                            tickfont=dict(color='rgba(200,200,200,0.7)', size=9),
                                            ticksuffix='%',
                                        ),
                                        angularaxis=dict(
                                            gridcolor='rgba(255,255,255,0.15)',
                                            tickfont=dict(color='white', size=10),
                                        ),
                                    ),
                                    paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    legend=dict(font=dict(color='white', size=10)),
                                    height=460, margin=dict(t=20, b=20, l=20, r=20),
                                )
                                st.plotly_chart(_fig_radar, use_container_width=True)
                            else:
                                st.info("Necessário ≥ 2 posições e ≥ 4 métricas para gerar o radar.")

                            # ── Tabela resumo por posição ─────────────────────────
                            st.markdown("---")
                            st.markdown("### 📋 Resumo por Posição")
                            _TD_POS_COLS = [c for c in [
                                'Posição', 'Distância (m)', 'M/min', 'Dist. 19-24 km/h (m)',
                                'Dist. > 24 km/h (m)', 'Sprints (>24 km/h)', 'Velocidade Máx (km/h)',
                                'Acc 2-3 (m/s²)', 'Dcc 2-3 (m/s²)', 'Acelerações (>3 m/s²)',
                                'Desacelerações (<-3 m/s²)', 'PlayerLoad', 'RHIE Blocos',
                            ] if c in _df_pos_grp.columns]
                            st.dataframe(
                                _df_pos_grp[_TD_POS_COLS].round(1),
                                use_container_width=True, hide_index=True
                            )

                            # ── Perfil Longitudinal /stats — removido (P8) ──
                    else:
                        st.info("Nenhum dado disponível para este período.")
                else:
                    st.info("Carregue os dados para visualizar a análise por posição.")

            # ══════════════════════════════════════════════════════════════
            # ABA 7: HISTÓRIA DO JOGO — removida (P8)
            # ══════════════════════════════════════════════════════════════

            # ══════════════════════════════════════════════════════════════
            # WCS: WORST-CASE SCENARIO (sub-tab Campo & GPS)
            # ══════════════════════════════════════════════════════════════
            with _sub_campo[1]:
                st.subheader("⚡ Worst-Case Scenario")
                st.caption(
                    "Identifica a **janela temporal de maior exigência física** de cada atleta "
                    "buscando o pior cenário em todos os períodos da atividade. "
                    "Base para prescrição de cargas de treino acima do jogo. "
                    "_(Delaney et al., 2018; Martín-García et al., 2018)_"
                )

                if _ok_ld == 0 or not dados_posicao_por_periodo:
                    st.info("Carregue os dados para usar a análise de Worst-Case Scenario.")
                else:
                    # ── Controles ──────────────────────────────────────────────────
                    _wcs2_c1, _wcs2_c2 = st.columns([1, 2])
                    with _wcs2_c1:
                        _wcs2_min = st.slider(
                            "⏱️ Janela temporal (min)", 1, 15, 5,
                            key="wcs2_janela",
                            help="Duração da janela rolante para identificar o pior cenário"
                        )
                    with _wcs2_c2:
                        _wcs2_metric_opts = [
                            "Distância (m)",
                            "🏃 Velocidade (bandas)",
                            "💥 Ações Acel/Desacel (efforts)",
                            "Velocidade Máx (km/h)",
                            "PlayerLoad",
                        ]
                        _wcs2_metric = st.selectbox(
                            "📊 Variável", _wcs2_metric_opts,
                            key="wcs2_metric_sel",
                            help="Métrica para identificar a janela de maior exigência. "
                                 "🏃 Velocidade = distância nas bandas escolhidas. "
                                 "💥 Ações Acel/Desacel = nº de esforços (ações reais da "
                                 "Catapult) de aceleração/desaceleração no pior minuto."
                        )

                        # ── Seleção de bandas (aparece ao escolher Velocidade/Aceleração) ──
                        _sel_vel_bands = []   # lista de dicts {min,max} das bandas de velocidade marcadas
                        _sel_acc_bands = []   # idem para aceleração
                        _sel_acc_boxes = set()  # caixas Gen2 (1..8) selecionadas
                        _sel_vel_pct = []       # (P9) faixas relativas (fração da Vmáx)
                        _wcs2_vel_rel = False   # (P9) modo % Vmáx individual
                        if _wcs2_metric == "🏃 Velocidade (bandas)":
                            _wcs2_vel_rel = st.checkbox(
                                "% da Vmáx individual", value=False, key="wcs2_vel_rel",
                                help="(P9) Em vez das bandas absolutas da conta, usa faixas "
                                     "relativas à Vmáx de CADA atleta (histórico da conta, "
                                     "fallback no pico da sessão).")
                            if _wcs2_vel_rel:
                                _rel_pick_w = st.multiselect(
                                    "🎚️ Faixas (% da Vmáx)",
                                    list(_REL_VEL_BANDAS.keys()),
                                    default=[_k for _k in _REL_VEL_BANDAS
                                             if _REL_VEL_BANDAS[_k][0] >= 0.7],
                                    key="wcs2_vel_rel_bands",
                                    help="A distância (m) é acumulada quando a velocidade está "
                                         "dentro da faixa relativa do próprio atleta.")
                                _sel_vel_pct = [_REL_VEL_BANDAS[_s] for _s in _rel_pick_w]
                                if not _sel_vel_pct:
                                    st.info("Selecione ao menos uma faixa.")
                            else:
                                _bv_act = _bandas_vel_ativas()
                                _bv_lbl = {}
                                for _bk, _bd in _bv_act.items():
                                    _mx = float(_bd.get('max', 9999))
                                    _faixa = (f"&gt;{_fmt_num_banda(_bd.get('min', 0))}"
                                              if _mx >= 9000
                                              else f"{_fmt_num_banda(_bd.get('min', 0))}-{_fmt_num_banda(_mx)}")
                                    _bv_lbl[f"B{_bk} — {_faixa} km/h"] = _bk
                                _bv_pick = st.multiselect(
                                    "🎚️ Bandas de velocidade a visualizar",
                                    list(_bv_lbl.keys()),
                                    default=list(_bv_lbl.keys()),
                                    key="wcs2_vel_bands",
                                    help="A distância percorrida (m) é acumulada apenas enquanto a "
                                         "velocidade está dentro das bandas selecionadas."
                                )
                                _sel_vel_bands = [_bv_act[_bv_lbl[_s]] for _s in _bv_pick]
                                if not _sel_vel_bands:
                                    st.info("Selecione ao menos uma banda de velocidade.")
                        elif _wcs2_metric == "💥 Ações Acel/Desacel (efforts)":
                            _ba_act = _bandas_acc_ativas()
                            # Duas caixas separadas: Aceleração (A*) e Desaceleração (D*).
                            _acc_lbl = {_ba_act[k]['label']: k
                                        for k in _ba_act if str(k).startswith('A')}
                            _dec_lbl = {_ba_act[k]['label']: k
                                        for k in _ba_act if str(k).startswith('D')}
                            _cwa, _cwd = st.columns(2)
                            with _cwa:
                                _acc_pick = st.multiselect(
                                    "🚀 Aceleração",
                                    list(_acc_lbl.keys()),
                                    default=list(_acc_lbl.keys()),
                                    key="wcs2_acc_bands_pos",
                                    help="Ações de aceleração (Gen2Acceleration · caixas 6,7,8)."
                                )
                            with _cwd:
                                _dec_pick = st.multiselect(
                                    "🛑 Desaceleração",
                                    list(_dec_lbl.keys()),
                                    default=list(_dec_lbl.keys()),
                                    key="wcs2_acc_bands_neg",
                                    help="Ações de desaceleração (Gen2Acceleration · caixas 3,2,1)."
                                )
                            _sel_acc_bands = (
                                [_ba_act[_acc_lbl[_s]] for _s in _acc_pick]
                                + [_ba_act[_dec_lbl[_s]] for _s in _dec_pick]
                            )
                            _sel_acc_boxes = (
                                {_ACC_KEY_TO_NUM[_acc_lbl[_s]] for _s in _acc_pick
                                 if _acc_lbl[_s] in _ACC_KEY_TO_NUM}
                                | {_ACC_KEY_TO_NUM[_dec_lbl[_s]] for _s in _dec_pick
                                   if _dec_lbl[_s] in _ACC_KEY_TO_NUM}
                            )
                            st.caption(
                                "Conta o **nº de ações** de acel/desacel na janela — dos "
                                "*acceleration_efforts* da Catapult quando disponíveis, senão "
                                "detectadas no **sinal de aceleração do sensor** (mesma fonte da "
                                "aba Neuromuscular). O pior minuto é a janela com mais ações."
                            )
                            if not _sel_acc_bands:
                                st.info("Selecione ao menos uma banda de aceleração ou desaceleração.")

                    # ── Detecta Hz real a partir dos timestamps ─────────────────────
                    def _detect_hz(_periodos_list, _dppp):
                        """Estima a frequência de amostragem (Hz) como nº de amostras ÷ duração.

                        Usar contagem/duração (em vez da mediana das diferenças) é robusto
                        quando os timestamps vêm arredondados para segundos inteiros mas há
                        vários pontos por segundo — caso em que a mediana das diferenças daria
                        1 Hz erroneamente e a integração de distância superestimaria ~Nx."""
                        # (P1) Delegado ao motor único: metrics.estimate_hz.
                        _series = []
                        for _pnn in _periodos_list[:5]:
                            for _adat in list(_dppp.get(_pnn, {}).values())[:5]:
                                # Nativo: ts_pos · GPS-only: ts_gps
                                _tss = _adat.get('ts_pos', []) or _adat.get('ts_gps', [])
                                if len(_tss) > 20:
                                    _series.append(_tss)
                        return _mtr.estimate_hz(_series, default=10.0)

                    _wcs2_periodos_tmp = [
                        k for k in dados_posicao_por_periodo
                        if k != _CHAVE_COMBINADO
                    ]
                    _wcs2_hz = _detect_hz(_wcs2_periodos_tmp, dados_posicao_por_periodo)
                    _wcs2_n  = max(2, int(_wcs2_min * 60 * _wcs2_hz))
                    st.caption(f"📡 Frequência GPS detectada: **{_wcs2_hz} Hz** — janela = {_wcs2_n} amostras")

                    # ── Config do campo (necessário para fallback GPS) ──────────────
                    _wcs2_cfg = None
                    for _hkw in list(st.session_state.keys()):
                        if (_hkw.startswith("campo_cfg__")
                                and isinstance(st.session_state[_hkw], dict)
                                and 'fl' in st.session_state[_hkw]):
                            _wcs2_cfg = st.session_state[_hkw]
                            break

                    # ── Cálculo do WCS por atleta ───────────────────────────────────
                    _wcs2_rows = []
                    _wcs2_segs = {}  # atleta → {xn, yn, vel} para animação
                    _wcs2_prov = {}  # (P4) fonte das ações → nº de atletas

                    _wcs2_periodos = _wcs2_periodos_tmp
                    _wcs2_athletes = sorted(set(
                        a for _pn in _wcs2_periodos
                        for a in dados_posicao_por_periodo.get(_pn, {}).keys()
                    ))

                    # ── Diagnóstico: quantos efforts de aceleração existem? ─────────
                    if _wcs2_metric == "💥 Ações Acel/Desacel (efforts)":
                        _tot_acc_eff = sum(
                            len(dados_efforts_acc_por_periodo.get(_pn, {}).get(_a, []) or [])
                            for _pn in _wcs2_periodos for _a in _wcs2_athletes
                        )
                        if _tot_acc_eff > 0:
                            st.caption(
                                f"💥 **{_tot_acc_eff} ações** de acel/desacel encontradas "
                                f"nos efforts da conta — contadas por janela para achar o pior minuto."
                            )
                        else:
                            _dur_fb = get_min_dur_s()
                            st.info(
                                "ℹ️ A API não retornou *acceleration_efforts* para estes "
                                "atletas/períodos (dispositivo sem aceleração nativa). "
                                f"Contando **ações discretas** detectadas no sinal de "
                                f"aceleração (dv/dt): cada entrada sustentada por "
                                f"≥ **{_dur_fb:.1f} s** numa banda conta como 1 ação "
                                "(ajuste a duração mínima na sidebar)."
                            )

                    def _wcalc_wcs(_sv, _n, _is_vm):
                        if len(_sv) < max(_n, 2):
                            return 0.0
                        if _is_vm:
                            from collections import deque as _DqW
                            _dqW = _DqW(); _bvW = -1.0
                            for _iW in range(len(_sv)):
                                while _dqW and _sv[_dqW[-1]] <= _sv[_iW]:
                                    _dqW.pop()
                                _dqW.append(_iW)
                                if _dqW[0] <= _iW - _n:
                                    _dqW.popleft()
                                if _iW >= _n - 1:
                                    _cW = _sv[_dqW[0]]
                                    if _cW > _bvW:
                                        _bvW = _cW
                            return _bvW
                        else:
                            # (P1) canônico: metrics.rolling_sum
                            _rw = _mtr.rolling_sum(_sv, _n)
                            return max(_rw) if _rw else 0.0

                    for _wa in _wcs2_athletes:
                        _wx, _wy, _wv, _wac, _wts, _wper = [], [], [], [], [], []
                        for _pn in _wcs2_periodos:
                            _da  = dados_posicao_por_periodo.get(_pn, {}).get(_wa, {})
                            _xs  = _da.get('xs', [])
                            _ys  = _da.get('ys', [])
                            _vl  = _da.get('vel', [])         # km/h
                            _ac  = _da.get('acc', [])
                            _ts  = _da.get('ts_pos', [])
                            # _nn usa xs/ys como referência — vel pode estar vazia
                            _nn  = min(len(_xs), len(_ys))

                            if _nn == 0 and _wcs2_cfg:
                                # Fallback GPS: converte lats/lons para coordenadas de campo
                                _lp = _da.get('lats', [])
                                _lo = _da.get('lons', [])
                                if _lp and _lo:
                                    try:
                                        _gx2, _gy2 = gps_para_campo_coords(_lp, _lo, _wcs2_cfg)
                                        _xs = _gx2;  _ys = _gy2
                                        _vl = _da.get('vels_gps', [0.0] * len(_gx2))
                                        _ts = _da.get('ts_gps', []) or _ts   # timestamps Unix reais (GPS)
                                        _nn = min(len(_xs), len(_ys))
                                    except Exception:
                                        _nn = 0
                                        _diag_log('WCS', f"{_wa}: falha ao projetar "
                                                         f"GPS→campo no período {_pn}")

                            if _nn > 0:
                                # Preenche vel/acc/ts com zeros se ausentes ou mais curtos
                                _vl_pad  = list(_vl[:_nn])  + [0.0] * max(0, _nn - len(_vl))
                                _ac_pad  = list(_ac[:_nn])  + [0.0] * max(0, _nn - len(_ac))
                                _ts_pad  = list(_ts[:_nn])  + [0.0] * max(0, _nn - len(_ts))
                                _wx  += list(_xs[:_nn])
                                _wy  += list(_ys[:_nn])
                                _wv  += _vl_pad
                                _wac += _ac_pad
                                _wts += _ts_pad
                                _wper += [_pn] * _nn

                        if len(_wx) < max(_wcs2_n, 2):
                            continue

                        # Valor por amostra para a métrica selecionada
                        _Hz = _wcs2_hz
                        _m  = _wcs2_metric
                        if _m == "Distância (m)":
                            _sv = [v / (3.6 * _Hz) for v in _wv]
                        elif _m == "🏃 Velocidade (bandas)":
                            # Distância (m) acumulada apenas nas bandas marcadas.
                            if _wcs2_vel_rel:
                                # (P9) faixas relativas à Vmáx individual do atleta
                                _vmx_w = _vmax_individual_kmh(_wa, _wv)
                                _faixas_v = ([(lo * _vmx_w, hi * _vmx_w)
                                              for lo, hi in _sel_vel_pct]
                                             if _vmx_w > 0 else [])
                                if not _faixas_v:
                                    _diag_log('WCS', f"{_wa}: sem Vmáx individual "
                                                     "confiável — excluído do modo % Vmáx")
                            else:
                                _faixas_v = [(float(b.get('min', 0)), float(b.get('max', 9999)))
                                             for b in _sel_vel_bands]
                            # (P1) canônico: metrics.per_sample_distance_in_bands
                            _sv = (_mtr.per_sample_distance_in_bands(_wv, _faixas_v, _Hz)
                                   if _faixas_v else [0.0] * len(_wv))
                        elif _m == "💥 Ações Acel/Desacel (efforts)":
                            # Nº de AÇÕES (efforts da Catapult) de aceleração/desaceleração
                            # na janela: cada effort cuja aceleração cai nas bandas marcadas
                            # soma +1 na amostra mais próxima do seu start_time. A janela
                            # rolante soma as ações → identifica o pior minuto.
                            _faixas_a = [(float(b.get('min', -9999)), float(b.get('max', 9999)))
                                         for b in _sel_acc_bands]
                            def _in_aband(_aa, _ff=_faixas_a):
                                for _lo, _hi in _ff:
                                    if _lo <= _aa < _hi:
                                        return True
                                return False
                            _sv = [0.0] * len(_wx)
                            _wts_np = np.array(_wts, dtype=float)
                            # Timestamps Unix utilizáveis? (efforts usam start_time Unix)
                            _ts_unix_ok = (_wts_np.size > 0
                                           and float(np.median(_wts_np)) > 1e6)
                            # Existem efforts reais da API para este atleta?
                            _has_api_eff = any(
                                len(dados_efforts_acc_por_periodo
                                    .get(_pn, {}).get(_wa, []) or []) > 0
                                for _pn in _wcs2_periodos)
                            if _faixas_a and _ts_unix_ok and _has_api_eff:
                                # Caminho preferido: AÇÕES reais (efforts da Catapult).
                                _wcs2_prov['efforts'] = _wcs2_prov.get('efforts', 0) + 1  # (P4)
                                for _pn in _wcs2_periodos:
                                    _effs = (dados_efforts_acc_por_periodo
                                             .get(_pn, {}).get(_wa, []) or [])
                                    for _ef in _effs:
                                        try:
                                            _bx  = int(round(float(_ef.get('band'))))
                                            _stt = float(_ef.get('start_time') or 0)
                                        except (TypeError, ValueError):
                                            continue
                                        if _stt <= 0 or _bx not in _sel_acc_boxes:
                                            continue
                                        _idx = int(np.argmin(np.abs(_wts_np - _stt)))
                                        if 0 <= _idx < len(_sv):
                                            _sv[_idx] += 1.0
                            elif _faixas_a:
                                # Fallback (API sem efforts): detecta AÇÕES no SINAL DE
                                # ACELERAÇÃO do sensor (nativo 'a', 10 Hz) — mesma fonte da
                                # aba Neuromuscular/Janelas — e mapeia proporcionalmente para
                                # a timeline de posição do WCS. Nunca zera por falta de 'acc'.
                                _sp_w = (combinar_periodos_continuo(
                                            dados_sensor_por_atleta_por_periodo, _wa)
                                         if len(_wcs2_periodos) > 1 else
                                         dados_sensor_por_atleta_por_periodo
                                            .get(_wcs2_periodos[0], {}).get(_wa, [])) \
                                        if _wcs2_periodos else []
                                _acc_w = [float(_p.get('a') or 0.0) for _p in _sp_w]
                                if _acc_w and not any(abs(_a) > 0.05 for _a in _acc_w):
                                    _vw = [float(_p.get('v') or 0.0) * 3.6 for _p in _sp_w]
                                    _tw = [float(_p.get('ts') or 0.0) for _p in _sp_w]
                                    _acc_w = acc_series_from_vel(_vw, _tw, _SENSOR_HZ)
                                    _wcs2_prov['derivado'] = _wcs2_prov.get('derivado', 0) + 1  # (P4)
                                    _diag_log('WCS', f"{_wa}: sem aceleração nativa — "
                                                     "ações derivadas por dv/dt da velocidade")
                                elif _acc_w:
                                    _wcs2_prov['sensor'] = _wcs2_prov.get('sensor', 0) + 1  # (P4)
                                if _acc_w:
                                    _idxs_acc = detectar_acoes_acc_idx(
                                        _acc_w, _sel_acc_bands, freq_hz=_SENSOR_HZ)
                                    _Ls, _Lp = len(_acc_w), len(_sv)
                                    for _ix in _idxs_acc:
                                        _pi = int(_ix / _Ls * _Lp) if _Ls > 0 else 0
                                        if 0 <= _pi < _Lp:
                                            _sv[_pi] += 1.0
                                else:
                                    # Sem sensor (raro): mantém derivação posicional (dv/dt).
                                    _wac_fb = _wac
                                    if any(abs(_v) > 0.1 for _v in _wv):
                                        _wac_fb = acc_series_from_vel(_wv, _wts, _Hz)
                                    _idxs_acc = detectar_acoes_acc_idx(
                                        _wac_fb, _sel_acc_bands, freq_hz=_Hz)
                                    for _ix in _idxs_acc:
                                        if 0 <= _ix < len(_sv):
                                            _sv[_ix] += 1.0
                        elif _m == "Velocidade Máx (km/h)":
                            _sv = list(_wv)   # rolling max — tratado abaixo
                        elif _m == "PlayerLoad":
                            _pl_raw = []
                            for _ppn in _wcs2_periodos:
                                _pl_raw += dados_sensor_por_atleta_por_periodo.get(_ppn, {}).get(_wa, [])
                            if len(_pl_raw) >= len(_wx):
                                _sv = [float(p.get('pl') or 0) for p in _pl_raw[:len(_wx)]]
                            else:
                                _sv = [float(p.get('pl') or 0) for p in _pl_raw] + [0.0] * (len(_wx) - len(_pl_raw))
                        else:
                            _sv = [v / (3.6 * _Hz) for v in _wv]

                        # Multi-janela (1, 3, 5 min) para comparativo na tabela
                        _is_vm2 = (_m == "Velocidade Máx (km/h)")
                        _mw_vals = {}
                        for _mwname, _mwmin in [('1 min', 1), ('3 min', 3), ('5 min', 5)]:
                            _mwn = int(_mwmin * 60 * _wcs2_hz)
                            if _mwn != _wcs2_n and len(_sv) >= max(_mwn, 2):
                                _mw_vals[_mwname] = round(_wcalc_wcs(_sv, _mwn, _is_vm2), 1)

                        # Janela rolante
                        if _m == "Velocidade Máx (km/h)":
                            from collections import deque as _Dq
                            _dq3 = _Dq()
                            _bv3, _bsi3, _bei3 = -1.0, 0, _wcs2_n
                            for _i3 in range(len(_sv)):
                                while _dq3 and _sv[_dq3[-1]] <= _sv[_i3]:
                                    _dq3.pop()
                                _dq3.append(_i3)
                                if _dq3[0] <= _i3 - _wcs2_n:
                                    _dq3.popleft()
                                if _i3 >= _wcs2_n - 1:
                                    _c3 = _sv[_dq3[0]]
                                    if _c3 > _bv3:
                                        _bv3  = _c3
                                        _bei3 = _i3 + 1
                                        _bsi3 = _bei3 - _wcs2_n
                            _best_val2, _best_si2, _best_ei2 = _bv3, _bsi3, _bei3
                        else:
                            # (P1) canônico: metrics.rolling_sum + argmax
                            _rw2 = _mtr.rolling_sum(_sv, _wcs2_n)
                            if _rw2:
                                _bi2 = int(np.argmax(_rw2))
                                _best_val2 = _rw2[_bi2]
                                _best_si2  = _bi2
                                _best_ei2  = _bi2 + _wcs2_n
                            else:
                                _best_val2, _best_si2, _best_ei2 = 0.0, 0, _wcs2_n

                        # Timestamps
                        _ts0 = _wts[_best_si2] if _best_si2 < len(_wts) else 0
                        _ts1 = _wts[min(_best_ei2 - 1, len(_wts) - 1)] if _wts else 0
                        try:
                            from datetime import datetime as _dtc
                            _ini_str = _dtc.fromtimestamp(float(_ts0)).strftime('%H:%M:%S') if float(_ts0) > 1e6 else f"{int(_best_si2/_Hz//60):02d}:{int(_best_si2/_Hz%60):02d}"
                            _fim_str = _dtc.fromtimestamp(float(_ts1)).strftime('%H:%M:%S') if float(_ts1) > 1e6 else f"{int(_best_ei2/_Hz//60):02d}:{int(_best_ei2/_Hz%60):02d}"
                        except Exception:
                            _ini_str = f"{int(_best_si2/_Hz//60):02d}:{int(_best_si2/_Hz%60):02d}"
                            _fim_str = f"{int(_best_ei2/_Hz//60):02d}:{int(_best_ei2/_Hz%60):02d}"

                        # Posição do atleta
                        _posicao2 = '—'
                        for _rpn2, _rpl2 in resultados_por_periodo.items():
                            for _rrow2 in _rpl2:
                                if str(_rrow2.get('Atleta', '')) == _wa:
                                    _posicao2 = str(_rrow2.get('Posição', '—'))
                                    break
                            if _posicao2 != '—':
                                break

                        # ── Série rolling completa (para timeline) ─────────────
                        _is_vm3 = (_m == "Velocidade Máx (km/h)")
                        if _is_vm3:
                            from collections import deque as _DqTL
                            _dqTL = _DqTL(); _roll_full = []
                            for _iRL in range(len(_sv)):
                                while _dqTL and _sv[_dqTL[-1]] <= _sv[_iRL]:
                                    _dqTL.pop()
                                _dqTL.append(_iRL)
                                if _dqTL[0] <= _iRL - _wcs2_n:
                                    _dqTL.popleft()
                                if _iRL >= _wcs2_n - 1:
                                    _roll_full.append(_sv[_dqTL[0]])
                        else:
                            # (P1) canônico: metrics.rolling_sum
                            _roll_full = _mtr.rolling_sum(_sv, _wcs2_n)

                        # ── Densidade de Pico (janelas ≥ 90% do WCS) ───────────
                        _density_90 = (
                            sum(1 for _rv in _roll_full if _rv >= 0.9 * _best_val2)
                            if _best_val2 > 0 and _roll_full else 0
                        )

                        _row_d = {
                            '_atl_orig':   _wa,
                            'Atleta':      _wa,
                            'Posição':     _posicao2,
                            'Período':     _wper[_best_si2] if _best_si2 < len(_wper) else '—',
                            _wcs2_metric:  round(_best_val2, 1),
                            'Picos ≥90%':  _density_90,
                            'Início':      _ini_str,
                            'Fim':         _fim_str,
                        }
                        _row_d.update(_mw_vals)
                        _wcs2_rows.append(_row_d)
                        # Série de aceleração (m/s²) p/ colorir a trilha por banda
                        # de acel/desacel quando a métrica de AÇÕES está ativa.
                        _acc_full_anim = []
                        if _m == "💥 Ações Acel/Desacel (efforts)":
                            if any(abs(_v) > 0.1 for _v in _wv):
                                _acc_full_anim = acc_series_from_vel(_wv, _wts, _Hz)
                            else:
                                _acc_full_anim = list(_wac)
                        _wcs2_segs[_wa] = {
                            'xn':     _wx[_best_si2:_best_ei2],
                            'yn':     _wy[_best_si2:_best_ei2],
                            'vel':    _wv[_best_si2:_best_ei2],
                            'acc':    (_acc_full_anim[_best_si2:_best_ei2]
                                       if _acc_full_anim else []),
                            'rolling': _roll_full,
                            'vel_all': _wv,        # velocidade de toda a série (trilha colorida)
                            'acc_all': _acc_full_anim,
                            'xn_all':  _wx,
                            'yn_all':  _wy,
                        }

                    _wcs2_rows.sort(key=lambda r: r.get(_wcs2_metric, 0), reverse=True)
                    _rank_icons = ['🔴', '🟠', '🟡']   # vermelho = maior carga/fadiga
                    for _ri2, _wr2 in enumerate(_wcs2_rows):
                        _wr2['#'] = _rank_icons[_ri2] if _ri2 < 3 else f'#{_ri2 + 1}'

                    if not _wcs2_rows:
                        st.warning(
                            "Dados insuficientes para calcular WCS com essa janela temporal. "
                            "Reduza a janela ou carregue mais períodos."
                        )
                        # Diagnóstico para debug
                        with st.expander("🛠️ Diagnóstico de dados", expanded=True):
                            _diag = []
                            for _wa_d in _wcs2_athletes[:8]:
                                _tot_xs, _tot_vel = 0, 0
                                for _pn_d in _wcs2_periodos:
                                    _da_d = dados_posicao_por_periodo.get(_pn_d, {}).get(_wa_d, {})
                                    _tot_xs  += len(_da_d.get('xs', []))
                                    _tot_vel += len(_da_d.get('vel', []))
                                _diag.append({'Atleta': _wa_d,
                                              'Amostras XY': _tot_xs,
                                              'Amostras vel': _tot_vel,
                                              'Necessário': _wcs2_n})
                            if _diag:
                                st.dataframe(pd.DataFrame(_diag), hide_index=True,
                                             use_container_width=True)
                            st.caption(f"Hz detectado: {_wcs2_hz} | Janela: {_wcs2_min} min = {_wcs2_n} amostras | Períodos: {len(_wcs2_periodos)}")
                    else:
                        # % do Máximo do grupo
                        _wcs2_top = _wcs2_rows[0].get(_wcs2_metric, 0) or 1.0
                        _wcs2_avg = sum(r.get(_wcs2_metric, 0) for r in _wcs2_rows) / len(_wcs2_rows)
                        for _wr in _wcs2_rows:
                            _wr['% Máx Grupo'] = round(_wr.get(_wcs2_metric, 0) / _wcs2_top * 100, 1)

                        # KPIs resumo
                        _wk1, _wk2, _wk3, _wk4 = st.columns(4)
                        _wk1.metric(
                            "🔴 Maior Fadiga",
                            _wcs2_rows[0]['Atleta'],
                            f"{_wcs2_top:.1f} — {_wcs2_rows[0].get('Período', '—')}",
                        )
                        _wk2.metric(
                            "📊 Média Grupo",
                            f"{_wcs2_avg:.1f}",
                            f"Δ {_wcs2_top - _wcs2_avg:.1f} vs líder",
                        )
                        _wk3.metric("👥 Atletas", str(len(_wcs2_rows)))
                        _wk4.metric(
                            "⏱️ Janela",
                            f"{_wcs2_min} min",
                            f"{_wcs2_rows[0].get('Início','—')} → {_wcs2_rows[0].get('Fim','—')}",
                        )

                        st.markdown("---")

                        # Tabela
                        # (P4) selo de proveniência das ações (pode variar por atleta)
                        if (_wcs2_metric == "💥 Ações Acel/Desacel (efforts)"
                                and _wcs2_prov):
                            _prov_txt = " · ".join(
                                f"{_PROV_LABELS.get(_k, ('⚪', _k))[0]} "
                                f"{_PROV_LABELS.get(_k, ('', str(_k)))[1]}: "
                                f"**{_v} atleta(s)**"
                                for _k, _v in sorted(_wcs2_prov.items()))
                            st.caption(f"**Fonte das ações** — {_prov_txt}")

                        # (P9) aviso do modo relativo (cortes diferentes por atleta)
                        if _wcs2_metric == "🏃 Velocidade (bandas)" and _wcs2_vel_rel:
                            st.caption("🎚️ **Modo % da Vmáx individual** — os cortes das "
                                       "faixas são calculados por atleta (referência: Vmáx "
                                       "histórica da conta, com fallback no pico da sessão).")

                        _df_all_tmp = pd.DataFrame(_wcs2_rows)
                        _mw_avail   = [c for c in ['1 min', '3 min', '5 min']
                                       if c in _df_all_tmp.columns
                                       and _df_all_tmp[c].notna().any()]
                        _wcs2_col_order = (
                            ['#', 'Atleta', 'Posição', 'Período',
                             _wcs2_metric, '% Máx Grupo', 'Picos ≥90%']
                            + _mw_avail
                            + ['Início', 'Fim']
                        )
                        _wcs2_col_order = [c for c in _wcs2_col_order if c in _df_all_tmp.columns]
                        _df_wcs2 = _df_all_tmp[_wcs2_col_order]

                        _col_cfg_wcs = {
                            '% Máx Grupo': st.column_config.ProgressColumn(
                                '% Máx', min_value=0, max_value=100, format='%.1f%%'
                            ),
                        }
                        _wcs2_evt = st.dataframe(
                            _df_wcs2,
                            use_container_width=True,
                            hide_index=True,
                            on_select='rerun',
                            selection_mode='single-row',
                            key='wcs2_table_sel',
                            column_config=_col_cfg_wcs,
                        )
                        st.caption(
                            "💡 Clique em uma linha para visualizar o percurso animado no campo abaixo. "
                            "As colunas 1/3/5 min mostram o valor WCS para janelas fixas de comparação. "
                            "**Picos ≥90%** = nº de janelas onde o atleta atingiu ≥90% do seu WCS."
                        )

                        # ── Timeline do Rolling Window — removido (P8) ──

                        # ── WCS por Período ─────────────────────────────────────────
                        with st.expander("📊 WCS por Período", expanded=False):
                            st.caption(
                                "WCS de cada atleta calculado **separadamente por período**. "
                                "🔴 Vermelho = maior demanda. Detecta queda de desempenho por fadiga entre tempos."
                            )
                            _ppw_data = {}
                            for _pn_pp in _wcs2_periodos:
                                _ppw_data[_pn_pp] = {}
                                for _wa_pp in _wcs2_athletes:
                                    _da_pp  = dados_posicao_por_periodo.get(_pn_pp, {}).get(_wa_pp, {})
                                    _xs_pp  = list(_da_pp.get('xs', []))
                                    _ys_pp  = list(_da_pp.get('ys', []))
                                    _vl_pp  = list(_da_pp.get('vel', []))
                                    _ac_pp  = list(_da_pp.get('acc', []))
                                    _nn_pp  = min(len(_xs_pp), len(_ys_pp))
                                    if _nn_pp == 0 and _wcs2_cfg:
                                        _lp_pp = _da_pp.get('lats', [])
                                        _lo_pp = _da_pp.get('lons', [])
                                        if _lp_pp and _lo_pp:
                                            try:
                                                _gx_pp, _gy_pp = gps_para_campo_coords(
                                                    _lp_pp, _lo_pp, _wcs2_cfg
                                                )
                                                _xs_pp = _gx_pp; _ys_pp = _gy_pp
                                                _vl_pp = _da_pp.get('vels_gps', [0.0]*len(_gx_pp))
                                                _nn_pp = min(len(_xs_pp), len(_ys_pp))
                                            except Exception:
                                                _nn_pp = 0
                                    if _nn_pp < _wcs2_n:
                                        _ppw_data[_pn_pp][_wa_pp] = None
                                        continue
                                    _vl_pp_p = list(_vl_pp[:_nn_pp]) + [0.0]*max(0,_nn_pp-len(_vl_pp))
                                    _ac_pp_p = list(_ac_pp[:_nn_pp]) + [0.0]*max(0,_nn_pp-len(_ac_pp))
                                    _m_pp = _wcs2_metric
                                    if _m_pp == "Distância (m)":
                                        _sv_pp = [v/(3.6*_wcs2_hz) for v in _vl_pp_p]
                                    elif ">14" in _m_pp:
                                        _sv_pp = [v/(3.6*_wcs2_hz) if v>14 else 0.0 for v in _vl_pp_p]
                                    elif "19" in _m_pp:
                                        _sv_pp = [v/(3.6*_wcs2_hz) if v>19 else 0.0 for v in _vl_pp_p]
                                    elif "24" in _m_pp:
                                        _sv_pp = [v/(3.6*_wcs2_hz) if v>24 else 0.0 for v in _vl_pp_p]
                                    elif "Velocidade Máx" in _m_pp:
                                        _sv_pp = _vl_pp_p
                                    elif "PlayerLoad" in _m_pp:
                                        _pl_pp = dados_sensor_por_atleta_por_periodo.get(_pn_pp,{}).get(_wa_pp,[])
                                        _sv_pp = ([float(p.get('pl') or 0) for p in _pl_pp[:_nn_pp]]
                                                  + [0.0]*max(0,_nn_pp-len(_pl_pp)))
                                    elif ">2" in _m_pp and "Acel" in _m_pp:
                                        _sv_pp = [1.0 if a>2 else 0.0 for a in _ac_pp_p]
                                    elif ">3" in _m_pp and "Acel" in _m_pp:
                                        _sv_pp = [1.0 if a>3 else 0.0 for a in _ac_pp_p]
                                    elif "<-2" in _m_pp:
                                        _sv_pp = [1.0 if a<-2 else 0.0 for a in _ac_pp_p]
                                    elif "<-3" in _m_pp:
                                        _sv_pp = [1.0 if a<-3 else 0.0 for a in _ac_pp_p]
                                    else:
                                        _sv_pp = [v/(3.6*_wcs2_hz) for v in _vl_pp_p]
                                    _is_vm_pp = ("Velocidade Máx" in _m_pp)
                                    _wcs_pp   = _wcalc_wcs(_sv_pp, _wcs2_n, _is_vm_pp)
                                    _ppw_data[_pn_pp][_wa_pp] = round(_wcs_pp, 1) if _wcs_pp > 0 else None

                            _df_ppw = pd.DataFrame(_ppw_data).T
                            _df_ppw.index.name = 'Período'
                            if not _df_ppw.empty and _df_ppw.notna().any().any():
                                _z_ppw      = _df_ppw.values.tolist()
                                _aths_ppw   = _df_ppw.columns.tolist()
                                _pers_ppw   = _df_ppw.index.tolist()

                                # Limites globais para escala
                                _all_ppw  = [v for row in _z_ppw for v in row if v is not None]
                                _zmin_ppw = min(_all_ppw) if _all_ppw else 0
                                _zmax_ppw = max(_all_ppw) if _all_ppw else 1

                                # ── Heatmap principal (sem texttemplate) ──────────────
                                _fig_ppw = go.Figure(data=go.Heatmap(
                                    z=_z_ppw,
                                    x=_aths_ppw,
                                    y=_pers_ppw,
                                    colorscale='RdYlGn_r',
                                    zmin=_zmin_ppw,
                                    zmax=_zmax_ppw,
                                    hovertemplate='%{y} — %{x}<br>WCS: %{z:.1f}<extra></extra>',
                                    showscale=True,
                                    colorbar=dict(
                                        title=dict(text=_wcs2_metric, font=dict(color='white')),
                                        tickfont=dict(color='white'),
                                    ),
                                ))

                                # Anotações por célula com cor de texto adaptativa:
                                # RdYlGn_r: baixo→verde(escuro), meio→amarelo(claro), alto→vermelho(escuro)
                                # Zona amarela (norm 0.25–0.75) → texto preto; demais → branco
                                _rng_ppw = _zmax_ppw - _zmin_ppw if _zmax_ppw > _zmin_ppw else 1
                                for _ri_a, _py_a in enumerate(_pers_ppw):
                                    for _ci_a, _px_a in enumerate(_aths_ppw):
                                        _v_a = _z_ppw[_ri_a][_ci_a]
                                        if _v_a is None:
                                            _lbl_a = '—'
                                            _tc_a  = '#888888'
                                        else:
                                            _lbl_a = f'{_v_a:.1f}'
                                            _norm_a = (_v_a - _zmin_ppw) / _rng_ppw
                                            _tc_a   = ('#111111'
                                                        if 0.25 < _norm_a < 0.75
                                                        else 'white')
                                        _fig_ppw.add_annotation(
                                            x=_px_a, y=_py_a,
                                            text=_lbl_a,
                                            showarrow=False,
                                            font=dict(size=11, color=_tc_a,
                                                      family='monospace'),
                                            xanchor='center', yanchor='middle',
                                        )

                                _fig_ppw.update_layout(
                                    title=dict(
                                        text=f"WCS por Período — {_wcs2_metric} ({_wcs2_min} min)",
                                        font=dict(color='white', size=13)
                                    ),
                                    plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    height=max(260, len(_pers_ppw) * 70 + 120),
                                    margin=dict(t=50, b=100, l=10, r=10),
                                    xaxis=dict(tickangle=-35,
                                               tickfont=dict(size=9, color='white'),
                                               color='white'),
                                    yaxis=dict(color='white'),
                                )
                                st.plotly_chart(_fig_ppw, use_container_width=True)

                                # ── Heatmap de variação % entre períodos consecutivos ─
                                if len(_pers_ppw) >= 2:
                                    _diff_z    = []
                                    _diff_lbls = []
                                    _diff_ys   = []
                                    for _pi_d in range(1, len(_pers_ppw)):
                                        _p1d = _pers_ppw[_pi_d - 1]
                                        _p2d = _pers_ppw[_pi_d]
                                        _diff_ys.append(f'Δ% {_p1d}→{_p2d}')
                                        _row_dz = []; _row_dl = []
                                        for _ath_d in _aths_ppw:
                                            _v1d = _ppw_data.get(_p1d, {}).get(_ath_d)
                                            _v2d = _ppw_data.get(_p2d, {}).get(_ath_d)
                                            if (_v1d is not None and _v2d is not None
                                                    and _v1d > 0):
                                                _dv = round((_v2d - _v1d) / _v1d * 100, 1)
                                                _row_dz.append(_dv)
                                                _row_dl.append(
                                                    f'+{_dv:.1f}%' if _dv >= 0
                                                    else f'{_dv:.1f}%'
                                                )
                                            else:
                                                _row_dz.append(None)
                                                _row_dl.append('—')
                                        _diff_z.append(_row_dz)
                                        _diff_lbls.append(_row_dl)

                                    _all_dv = [v for row in _diff_z
                                               for v in row if v is not None]
                                    if _all_dv:
                                        _absmax_d = max(abs(v) for v in _all_dv) or 10
                                        _fig_diff = go.Figure(data=go.Heatmap(
                                            z=_diff_z,
                                            x=_aths_ppw,
                                            y=_diff_ys,
                                            colorscale='RdYlGn_r',
                                            zmid=0,
                                            zmin=-_absmax_d,
                                            zmax=_absmax_d,
                                            hovertemplate='%{y}<br>%{x}: %{z:+.1f}%<extra></extra>',
                                            showscale=True,
                                            colorbar=dict(
                                                title=dict(text='Δ%',
                                                           font=dict(color='white')),
                                                tickfont=dict(color='white'),
                                                ticksuffix='%',
                                            ),
                                        ))
                                        # Anotações adaptativas para diff
                                        for _ri_d2, _yd2 in enumerate(_diff_ys):
                                            for _ci_d2, _xd2 in enumerate(_aths_ppw):
                                                _vd2  = _diff_z[_ri_d2][_ci_d2]
                                                _ld2  = _diff_lbls[_ri_d2][_ci_d2]
                                                if _vd2 is None:
                                                    _tcd2 = '#888888'
                                                else:
                                                    # norm em [-absmax, +absmax] → [0, 1]
                                                    _nd2  = (_vd2 + _absmax_d) / (2 * _absmax_d)
                                                    _tcd2 = ('#111111'
                                                              if 0.25 < _nd2 < 0.75
                                                              else 'white')
                                                _fig_diff.add_annotation(
                                                    x=_xd2, y=_yd2,
                                                    text=_ld2,
                                                    showarrow=False,
                                                    font=dict(size=11, color=_tcd2,
                                                              family='monospace'),
                                                    xanchor='center', yanchor='middle',
                                                )
                                        # altura: cada linha precisa de ~80px + 160px de margens
                                        _h_diff = max(260, len(_diff_ys) * 80 + 160)
                                        _fig_diff.update_layout(
                                            title=dict(
                                                text='Variação % entre Períodos',
                                                font=dict(color='white', size=13)
                                            ),
                                            plot_bgcolor='#0e1117',
                                            paper_bgcolor='#0e1117',
                                            font=dict(color='white'),
                                            height=_h_diff,
                                            margin=dict(t=45, b=110, l=10, r=10),
                                            xaxis=dict(
                                                tickangle=-35,
                                                tickfont=dict(size=9, color='white'),
                                                color='white',
                                            ),
                                            yaxis=dict(
                                                color='white',
                                                tickfont=dict(size=10),
                                            ),
                                        )
                                        st.plotly_chart(_fig_diff, use_container_width=True)
                                        st.caption(
                                            "🟢 Verde = queda de demanda no período seguinte  "
                                            "🔴 Vermelho = aumento de demanda (atenção à fadiga)"
                                        )
                            else:
                                st.info("Dados insuficientes para comparar períodos com esta janela.")

                        st.markdown("---")

                        # ── Animação no campo ao selecionar linha ──────────────────
                        _wcs2_sel = (_wcs2_evt.selection.rows
                                     if _wcs2_evt.selection else [])
                        if _wcs2_sel:
                            _wsel_atl = _wcs2_rows[_wcs2_sel[0]].get(
                                '_atl_orig', _wcs2_rows[_wcs2_sel[0]]['Atleta']
                            )
                            _wsel_row = _wcs2_rows[_wcs2_sel[0]]
                            _wseg     = _wcs2_segs.get(_wsel_atl, {})
                            _wcs2_xn  = _wseg.get('xn', [])
                            _wcs2_yn  = _wseg.get('yn', [])
                            _wcs2_vel = _wseg.get('vel', [])
                            _wcs2_acc = _wseg.get('acc', [])
                            # Quando a métrica é AÇÕES, a trilha/legenda usam bandas de
                            # acel/desacel (m/s²) em vez de velocidade.
                            _is_acoes_anim = (
                                _wcs2_metric == "💥 Ações Acel/Desacel (efforts)"
                                and len(_wcs2_acc) == len(_wcs2_xn)
                            )

                            if len(_wcs2_xn) >= 2:
                                st.markdown(f"### 🏃 {_wsel_atl} — WCS {_wcs2_min} min")
                                _wval_str = f"{_wsel_row.get(_wcs2_metric, 0):.1f}"
                                _wm1, _wm2, _wm3 = st.columns(3)
                                _wm1.metric(_wcs2_metric, _wval_str)
                                _wm2.metric("⏰ Início",  _wsel_row.get('Início', '—'))
                                _wm3.metric("🏁 Fim",     _wsel_row.get('Fim',    '—'))

                                # Campo config
                                _wcs2_cfg = None
                                for _hk2 in list(st.session_state.keys()):
                                    if (_hk2.startswith("campo_cfg__")
                                            and isinstance(st.session_state[_hk2], dict)
                                            and 'fl' in st.session_state[_hk2]):
                                        _wcs2_cfg = st.session_state[_hk2]
                                        break
                                _wcs2_fl = float(_wcs2_cfg.get('fl', 105)) if _wcs2_cfg else 105.0
                                _wcs2_fw = float(_wcs2_cfg.get('fw', 68))  if _wcs2_cfg else 68.0

                                def _vc_wcs2(v):
                                    if v < 7:  return '#2196F3'
                                    if v < 14: return '#4CAF50'
                                    if v < 19: return '#FFEB3B'
                                    if v < 24: return '#FF9800'
                                    return '#F44336'

                                # ── Cor por banda de aceleração/desaceleração ───────
                                _acc_bands_anim = list(_bandas_acc_ativas().values())
                                _NEUTRO_COR_AC = '#546E7A'   # zona leve/neutra (−2..2)
                                def _ac_color(a):
                                    for _b in _acc_bands_anim:
                                        try:
                                            if float(_b['min']) <= a < float(_b['max']):
                                                return _b.get('color', _NEUTRO_COR_AC)
                                        except (TypeError, ValueError, KeyError):
                                            continue
                                    # satura no topo da banda extrema de aceleração
                                    try:
                                        _tops = [float(_b['max']) for _b in _acc_bands_anim
                                                 if float(_b['min']) >= 0]
                                        if _tops and a >= max(_tops):
                                            return next(_b.get('color', _NEUTRO_COR_AC)
                                                        for _b in _acc_bands_anim
                                                        if float(_b['max']) == max(_tops))
                                    except (TypeError, ValueError, KeyError, StopIteration):
                                        pass
                                    return _NEUTRO_COR_AC

                                _fig_wcs2 = desenhar_campo_futebol_bonito(
                                    field_length=_wcs2_fl,
                                    field_width=_wcs2_fw,
                                    title=(
                                        f"WCS {_wcs2_min} min — {_wsel_atl}  |  "
                                        f"{_wsel_row.get('Início','—')} → {_wsel_row.get('Fim','—')}"
                                    )
                                )
                                _n_base_wcs2 = len(_fig_wcs2.data)

                                # Downsampling
                                _nf2   = len(_wcs2_xn)
                                _step2 = max(1, _nf2 // 120)
                                _fr2   = list(range(0, _nf2, _step2))
                                if _fr2[-1] != _nf2 - 1:
                                    _fr2.append(_nf2 - 1)

                                # ── Trilha completa colorida (estática) ──
                                # Métrica de AÇÕES → cor por banda de acel/desacel;
                                # caso contrário → faixa de velocidade.
                                if _is_acoes_anim:
                                    _trail_colors = [_ac_color(a) for a in _wcs2_acc]
                                else:
                                    _trail_colors = [_vc_wcs2(v) for v in _wcs2_vel]
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=_wcs2_xn, y=_wcs2_yn,
                                    mode='markers',
                                    marker=dict(
                                        color=_trail_colors,
                                        size=7, opacity=0.55,
                                    ),
                                    name='Trilha', showlegend=False, hoverinfo='skip',
                                ))
                                # ── Legenda inline ──
                                if _is_acoes_anim:
                                    # Bandas de acel/desacel (rótulo curto B1/B2/B3)
                                    import re as _re_anim
                                    for _bk, _bd in _bandas_acc_ativas().items():
                                        _lbl_full = _bd.get('label', _bk)
                                        # encurta: "Aceleração B1 — 2 a 3 m/s²"
                                        _emoji = '🚀' if str(_bk).startswith('A') else '🛑'
                                        _fig_wcs2.add_trace(go.Scatter(
                                            x=[None], y=[None], mode='markers',
                                            marker=dict(size=10,
                                                        color=_bd.get('color', '#888')),
                                            name=f"{_emoji} {_lbl_full}",
                                        ))
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color=_NEUTRO_COR_AC),
                                        name='• Neutro (−2 a 2 m/s²)',
                                    ))
                                else:
                                    # Legenda de velocidade inline
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color='#2196F3'), name='< 7 km/h',
                                    ))
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color='#4CAF50'), name='7–14 km/h',
                                    ))
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color='#FFEB3B'), name='14–19 km/h',
                                    ))
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color='#FF9800'), name='19–24 km/h',
                                    ))
                                    _fig_wcs2.add_trace(go.Scatter(
                                        x=[None], y=[None], mode='markers',
                                        marker=dict(size=10, color='#F44336'), name='> 24 km/h',
                                    ))
                                # Marcadores início / fim
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=[_wcs2_xn[0]], y=[_wcs2_yn[0]],
                                    mode='markers+text',
                                    marker=dict(size=14, color='#4CAF50', symbol='circle',
                                                line=dict(color='white', width=2)),
                                    text=['▶'], textposition='top center',
                                    textfont=dict(color='#4CAF50', size=11),
                                    name='Início', showlegend=False, hoverinfo='skip',
                                ))
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=[_wcs2_xn[-1]], y=[_wcs2_yn[-1]],
                                    mode='markers+text',
                                    marker=dict(size=14, color='#F44336', symbol='x',
                                                line=dict(color='white', width=2)),
                                    text=['■'], textposition='top center',
                                    textfont=dict(color='#F44336', size=11),
                                    name='Fim', showlegend=False, hoverinfo='skip',
                                ))
                                # Dot animado (único trace que muda nos frames)
                                _dot_c0 = (_ac_color(_wcs2_acc[0] if _wcs2_acc else 0)
                                           if _is_acoes_anim
                                           else _vc_wcs2(_wcs2_vel[0] if _wcs2_vel else 0))
                                _fig_wcs2.add_trace(go.Scatter(
                                    x=[_wcs2_xn[0]], y=[_wcs2_yn[0]], mode='markers',
                                    marker=dict(
                                        size=20,
                                        color=_dot_c0,
                                        symbol='circle',
                                        line=dict(color='white', width=3),
                                    ),
                                    name='Posição atual', showlegend=False,
                                ))

                                # Índice: dot é o último trace adicionado
                                _idx_d2 = len(_fig_wcs2.data) - 1

                                # Frames — só atualiza o dot
                                _frames2 = []
                                for _fk2 in _fr2:
                                    _ds3  = (_fk2 / max(_nf2 - 1, 1)) * _wcs2_min * 60
                                    _dm3  = int(_ds3 // 60)
                                    _dsr3 = int(_ds3 % 60)
                                    _v3   = float(_wcs2_vel[_fk2]) if _fk2 < len(_wcs2_vel) else 0.0
                                    if _is_acoes_anim:
                                        _a3  = float(_wcs2_acc[_fk2]) if _fk2 < len(_wcs2_acc) else 0.0
                                        _c3  = _ac_color(_a3)
                                        _hud = f'   |   ⚡ {_a3:+.1f} m/s²'
                                    else:
                                        _c3  = _vc_wcs2(_v3)
                                        _hud = f'   |   💨 {_v3:.1f} km/h'
                                    _frames2.append(go.Frame(
                                        data=[go.Scatter(
                                            x=[_wcs2_xn[_fk2]],
                                            y=[_wcs2_yn[_fk2]],
                                            mode='markers',
                                            marker=dict(
                                                size=20, color=_c3,
                                                symbol='circle',
                                                line=dict(color='white', width=3),
                                            ),
                                        )],
                                        traces=[_idx_d2],
                                        name=str(_fk2),
                                        layout=go.Layout(title=dict(
                                            text=(
                                                f'WCS {_wcs2_min} min — {_wsel_atl} | '
                                                f'⏱️ {_dm3}:{_dsr3:02d} / {_wcs2_min}:00'
                                                f'{_hud}'
                                            ),
                                            font=dict(color='white', size=12),
                                        )),
                                    ))

                                _fig_wcs2.frames = _frames2

                                _sliders_wcs2 = [{
                                    'steps': [
                                        {
                                            'args': [[str(_fk2)],
                                                     {'frame': {'duration': 0, 'redraw': True},
                                                      'mode': 'immediate'}],
                                            'label': '',
                                            'method': 'animate',
                                        }
                                        for _fk2 in _fr2
                                    ],
                                    'transition': {'duration': 0},
                                    'x': 0.05, 'len': 0.90,
                                    'currentvalue': {'visible': False},
                                    'bgcolor': '#1e3a5f',
                                    'bordercolor': '#2196F3',
                                    'tickcolor': 'white',
                                    'font': {'color': 'white', 'size': 9},
                                }]
                                _fig_wcs2.update_layout(
                                    height=560,
                                    updatemenus=[{
                                        'type': 'buttons',
                                        'showactive': False,
                                        'y': -0.08, 'x': 0.5,
                                        'xanchor': 'center', 'yanchor': 'top',
                                        'buttons': [
                                            {
                                                'label': '▶ Play',
                                                'method': 'animate',
                                                'args': [None, {
                                                    'frame': {'duration': 60, 'redraw': True},
                                                    'fromcurrent': True,
                                                    'transition': {'duration': 60, 'easing': 'linear'},
                                                }],
                                            },
                                            {
                                                'label': '⏸ Pause',
                                                'method': 'animate',
                                                'args': [[None], {
                                                    'frame': {'duration': 0, 'redraw': False},
                                                    'mode': 'immediate',
                                                    'transition': {'duration': 0},
                                                }],
                                            },
                                        ],
                                        'font': {'color': 'white'},
                                        'bgcolor': '#1e3a5f',
                                        'bordercolor': '#2196F3',
                                    }],
                                    sliders=_sliders_wcs2,
                                    margin=dict(b=80),
                                )
                                st.plotly_chart(_fig_wcs2, use_container_width=True)
                            else:
                                st.info("Dados GPS insuficientes para animação deste atleta.")

            # ==================== ABA 8: MONITORAMENTO AO VIVO ====================
            with abas[8]:
                st.subheader("📡 Monitoramento em Tempo Real")
                st.caption(
                    "Conecta ao endpoint `/live` da API Catapult para acompanhar métricas "
                    "de atletas durante uma sessão ativa. Requer uma sessão aberta no OpenField."
                )

                # ── Inicializar session state do live ─────────────────────
                if 'live_active' not in st.session_state:
                    st.session_state['live_active'] = False
                if 'live_alert_log' not in st.session_state:
                    st.session_state['live_alert_log'] = []
                if 'live_snapshot' not in st.session_state:
                    st.session_state['live_snapshot'] = None
                if 'live_info_snapshot' not in st.session_state:
                    st.session_state['live_info_snapshot'] = None

                # ── Layout de controles ────────────────────────────────────
                _lv_c1, _lv_c2, _lv_c3 = st.columns([2, 2, 1])
                with _lv_c1:
                    _lv_interval = st.select_slider(
                        "⏱️ Intervalo de atualização:",
                        options=[5, 10, 15, 30, 60],
                        value=10,
                        format_func=lambda x: f"{x}s",
                        key="live_interval",
                    )
                with _lv_c2:
                    _lv_cols = st.multiselect(
                        "📊 Métricas a exibir:",
                        options=["velocity", "heart_rate", "player_load",
                                 "acceleration", "total_distance", "odometer"],
                        default=["velocity", "heart_rate", "player_load"],
                        key="live_metrics_sel",
                    )
                with _lv_c3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    _lv_refresh_now = st.button(
                        "🔄 Agora", key="live_refresh_now",
                        help="Atualizar imediatamente uma vez"
                    )

                # ── Configuração de limiares (alertas) ────────────────────
                with st.expander("🚨 Configurar Limiares de Alerta", expanded=True):
                    st.caption(
                        "O app dispara um alerta visual quando o atleta **atingir ou ultrapassar** "
                        "o valor configurado. Deixe 0 para desativar."
                    )
                    _lv_th_c1, _lv_th_c2, _lv_th_c3 = st.columns(3)
                    with _lv_th_c1:
                        _lv_th_vel = st.number_input(
                            "🚀 Vel. máx (km/h)", min_value=0.0,
                            max_value=40.0, value=25.0, step=0.5,
                            key="live_th_vel",
                        )
                        _lv_th_dist = st.number_input(
                            "📏 Distância total (m)", min_value=0.0,
                            max_value=15000.0, value=0.0, step=100.0,
                            key="live_th_dist",
                            help="Alerta quando atleta passar desta distância acumulada. 0 = desativado."
                        )
                    with _lv_th_c2:
                        _lv_th_hr = st.number_input(
                            "❤️ FC máx (bpm)", min_value=0.0,
                            max_value=220.0, value=180.0, step=1.0,
                            key="live_th_hr",
                        )
                        _lv_th_pl = st.number_input(
                            "⚡ PlayerLoad", min_value=0.0,
                            max_value=1500.0, value=0.0, step=10.0,
                            key="live_th_pl",
                            help="Alerta quando PlayerLoad acumulado ultrapassar este valor. 0 = desativado."
                        )
                    with _lv_th_c3:
                        _lv_th_acc = st.number_input(
                            "💥 Aceleração (m/s²)", min_value=0.0,
                            max_value=10.0, value=3.5, step=0.1,
                            key="live_th_acc",
                        )
                        _lv_sound = st.checkbox(
                            "🔔 Exibir badge de alerta",
                            value=True, key="live_sound",
                        )

                    _lv_thresholds = {
                        "velocity":       (_lv_th_vel,  "km/h", "🚀"),
                        "heart_rate":     (_lv_th_hr,   "bpm",  "❤️"),
                        "acceleration":   (_lv_th_acc,  "m/s²", "💥"),
                        "total_distance": (_lv_th_dist, "m",    "📏"),
                        "player_load":    (_lv_th_pl,   "UA",   "⚡"),
                    }

                # ── Botões de controle ─────────────────────────────────────
                _lv_btn_c1, _lv_btn_c2, _lv_btn_c3 = st.columns([2, 2, 3])
                with _lv_btn_c1:
                    if not st.session_state['live_active']:
                        if st.button("▶ Iniciar Monitoramento", type="primary",
                                     key="live_start"):
                            st.session_state['live_active'] = True
                            st.session_state['live_alert_log'] = []
                            st.rerun()
                    else:
                        if st.button("⏹ Parar Monitoramento", type="secondary",
                                     key="live_stop"):
                            st.session_state['live_active'] = False
                            st.rerun()
                with _lv_btn_c2:
                    if st.button("🗑️ Limpar Alertas", key="live_clear_alerts"):
                        st.session_state['live_alert_log'] = []
                        st.rerun()
                with _lv_btn_c3:
                    _lv_status_ph = st.empty()

                # ── Indicador de status ────────────────────────────────────
                if st.session_state['live_active']:
                    _lv_status_ph.markdown(
                        "<span style='display:inline-flex;align-items:center;gap:6px;"
                        "background:#064e3b;border:1px solid #10b981;border-radius:20px;"
                        "padding:4px 12px;font-size:13px;color:#34d399'>"
                        "<span style='width:8px;height:8px;border-radius:50%;"
                        "background:#10b981;animation:pulse 1s infinite'></span>"
                        " AO VIVO</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    _lv_status_ph.markdown(
                        "<span style='display:inline-flex;align-items:center;gap:6px;"
                        "background:#1f2937;border:1px solid #4b5563;border-radius:20px;"
                        "padding:4px 12px;font-size:13px;color:#9ca3af'>"
                        "⏸ PAUSADO</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # ── Containers de exibição ─────────────────────────────────
                _lv_session_ph  = st.empty()   # info da sessão
                _lv_athletes_ph = st.empty()   # cards dos atletas
                _lv_alerts_ph   = st.empty()   # log de alertas

                # ── Função para renderizar os dados ───────────────────────
                def _render_live(info_data, athletes_data):
                    """Renderiza info da sessão + cards de atletas."""
                    import time as _tm

                    # ── Painel de sessão ──────────────────────────────────
                    with _lv_session_ph.container():
                        if info_data and isinstance(info_data, dict):
                            _si_c1, _si_c2, _si_c3, _si_c4 = st.columns(4)
                            _act_name = (
                                info_data.get('activity_name') or
                                info_data.get('name') or
                                info_data.get('id', '—')
                            )
                            _start_ts = info_data.get('start_time') or info_data.get('started_at') or 0
                            _elapsed  = ""
                            if _start_ts:
                                try:
                                    _el_s = int(_tm.time() - float(_start_ts))
                                    _elapsed = f"{_el_s//3600:02d}:{(_el_s%3600)//60:02d}:{_el_s%60:02d}"
                                except Exception:
                                    _elapsed = "—"
                            _n_atl = len(athletes_data) if isinstance(athletes_data, list) else 0
                            with _si_c1:
                                st.metric("🎯 Sessão", str(_act_name)[:28] or "Ativa")
                            with _si_c2:
                                st.metric("⏱️ Tempo decorrido", _elapsed or "—")
                            with _si_c3:
                                st.metric("👥 Atletas ativos", str(_n_atl))
                            with _si_c4:
                                st.metric("🔁 Última atualização",
                                          _tm.strftime('%H:%M:%S'))
                        elif athletes_data:
                            st.metric("👥 Atletas ativos",
                                      len(athletes_data) if isinstance(athletes_data, list) else "—")

                    # ── Cards dos atletas ──────────────────────────────────
                    with _lv_athletes_ph.container():
                        if not athletes_data:
                            st.info(
                                "📭 Nenhuma sessão ao vivo detectada.\n\n"
                                "**Verifique:**\n"
                                "- Existe uma atividade aberta no OpenField agora\n"
                                "- Os dispositivos estão transmitindo dados\n"
                                "- O token tem permissão para dados ao vivo"
                            )
                            return

                        # Normalizar resposta (pode ser lista ou dict com key 'athletes')
                        _atl_list = athletes_data
                        if isinstance(athletes_data, dict):
                            _atl_list = (
                                athletes_data.get('athletes') or
                                athletes_data.get('data') or
                                list(athletes_data.values())
                            )
                        if not isinstance(_atl_list, list):
                            st.warning("⚠️ Formato de resposta inesperado da API /live.")
                            st.json(athletes_data)
                            return

                        # Paleta de cores por atleta
                        _LV_COLORS = [
                            '#FF6B6B','#4ECDC4','#45B7D1','#96CEB4',
                            '#FFEAA7','#DDA0DD','#98FB98','#FFB347',
                        ]

                        # ── Mapeamento de campos (diferentes versões da API) ──
                        def _get_field(d, *keys, default=None):
                            for k in keys:
                                if k in d:
                                    return d[k]
                            return default

                        # ── Grid de cards (3 por linha) ───────────────────
                        _n_cols_lv = min(3, len(_atl_list))
                        _lv_rows   = [
                            _atl_list[i:i+_n_cols_lv]
                            for i in range(0, len(_atl_list), _n_cols_lv)
                        ]

                        _new_alerts = []
                        import time as _tm2

                        for _row in _lv_rows:
                            _cols = st.columns(len(_row))
                            for _ci, (_col, _atl) in enumerate(zip(_cols, _row)):
                                if not isinstance(_atl, dict):
                                    continue
                                _color = _LV_COLORS[_ci % len(_LV_COLORS)]
                                _name  = (
                                    _get_field(_atl, 'name', 'athlete_name',
                                               'display_name', default='Atleta')
                                )

                                # Extrair métricas (tentativa em vários campos)
                                _vel  = _get_field(_atl, 'velocity', 'current_velocity',
                                                   'speed', 'v', default=None)
                                _hr   = _get_field(_atl, 'heart_rate', 'hr',
                                                   'current_heart_rate', default=None)
                                _pl   = _get_field(_atl, 'player_load', 'playerload',
                                                   'total_player_load', 'pl', default=None)
                                _acc  = _get_field(_atl, 'acceleration', 'acc',
                                                   'current_acceleration', 'a', default=None)
                                _dist = _get_field(_atl, 'total_distance', 'distance',
                                                   'odometer', 'o', default=None)

                                # Verificar limiares
                                _card_alerts = []
                                _metric_map = {
                                    "velocity":       (_vel,  _lv_thresholds["velocity"]),
                                    "heart_rate":     (_hr,   _lv_thresholds["heart_rate"]),
                                    "acceleration":   (_acc,  _lv_thresholds["acceleration"]),
                                    "total_distance": (_dist, _lv_thresholds["total_distance"]),
                                    "player_load":    (_pl,   _lv_thresholds["player_load"]),
                                }
                                for _mk, (_mv, (_thr, _unit, _ico)) in _metric_map.items():
                                    if _mv is not None and _thr > 0:
                                        try:
                                            if float(_mv) >= float(_thr):
                                                _card_alerts.append(
                                                    f"{_ico} {_mk.replace('_',' ').title()}: "
                                                    f"**{float(_mv):.1f}** {_unit} "
                                                    f"(limiar: {_thr})"
                                                )
                                                _new_alerts.append({
                                                    'ts': _tm2.strftime('%H:%M:%S'),
                                                    'atleta': _name,
                                                    'metrica': _mk,
                                                    'valor': float(_mv),
                                                    'limiar': _thr,
                                                    'unidade': _unit,
                                                    'ico': _ico,
                                                })
                                        except Exception:
                                            _applog.log_debug_exc()

                                _has_alert = bool(_card_alerts)

                                # Cor do card por velocidade
                                try:
                                    _v_num = float(_vel) if _vel is not None else 0
                                    _vel_zone_color = (
                                        '#1e3a5f' if _v_num < 7 else
                                        '#1a3d2b' if _v_num < 14 else
                                        '#3d2b00' if _v_num < 19 else
                                        '#3d1a1a'
                                    )
                                except Exception:
                                    _vel_zone_color = '#1e293b'

                                _border_color = '#ef4444' if _has_alert else _color

                                with _col:
                                    # Card HTML
                                    st.markdown(
                                        f"<div style='background:{_vel_zone_color};"
                                        f"border:2px solid {_border_color};"
                                        f"border-radius:10px;padding:14px 12px;"
                                        f"margin-bottom:8px;min-height:160px'>"
                                        f"<div style='display:flex;justify-content:space-between;"
                                        f"align-items:center;margin-bottom:8px'>"
                                        f"<span style='font-weight:700;font-size:14px;"
                                        f"color:{_color}'>{_name[:18]}</span>"
                                        + (
                                            "<span style='background:#ef4444;color:white;"
                                            "font-size:10px;padding:2px 6px;border-radius:10px'>"
                                            "⚠️ ALERTA</span>"
                                            if _has_alert else
                                            "<span style='background:#10b981;color:white;"
                                            "font-size:10px;padding:2px 6px;border-radius:10px'>"
                                            "✅ OK</span>"
                                        ) +
                                        "</div>"
                                        # Métricas
                                        + (
                                            f"<div style='font-size:22px;font-weight:800;"
                                            f"color:white;margin:4px 0'>"
                                            f"🚀 {float(_vel):.1f} <span style='font-size:11px;"
                                            f"color:#94a3b8'>km/h</span></div>"
                                            if _vel is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"❤️ {float(_hr):.0f} bpm</div>"
                                            if _hr is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"⚡ PL {float(_pl):.1f}</div>"
                                            if _pl is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"💥 {float(_acc):.2f} m/s²</div>"
                                            if _acc is not None else ""
                                        )
                                        + (
                                            f"<div style='font-size:15px;color:#e2e8f0'>"
                                            f"📏 {float(_dist):.0f} m</div>"
                                            if _dist is not None else ""
                                        )
                                        + "</div>",
                                        unsafe_allow_html=True,
                                    )
                                    # Detalhes do alerta abaixo do card
                                    if _card_alerts and _lv_sound:
                                        for _al in _card_alerts:
                                            st.markdown(
                                                f"<div style='background:#450a0a;border-left:"
                                                f"3px solid #ef4444;border-radius:4px;"
                                                f"padding:4px 8px;font-size:11px;color:#fca5a5;"
                                                f"margin-bottom:3px'>{_al}</div>",
                                                unsafe_allow_html=True,
                                            )

                        # Adicionar novos alertas ao log (sem duplicar os últimos 5s)
                        if _new_alerts:
                            _existing_keys = {
                                (a['atleta'], a['metrica'], a['ts'])
                                for a in st.session_state['live_alert_log'][-20:]
                            }
                            for _na in _new_alerts:
                                _key = (_na['atleta'], _na['metrica'], _na['ts'])
                                if _key not in _existing_keys:
                                    st.session_state['live_alert_log'].append(_na)

                    # ── Log de alertas ────────────────────────────────────
                    with _lv_alerts_ph.container():
                        _log = st.session_state['live_alert_log']
                        if _log:
                            st.markdown("#### 🔴 Histórico de Alertas")
                            _df_log = pd.DataFrame(list(reversed(_log[-50:])))
                            _df_log.columns = [
                                'Hora', 'Atleta', 'Métrica',
                                'Valor', 'Limiar', 'Unidade', '—'
                            ]
                            st.dataframe(
                                _df_log[['Hora','Atleta','Métrica','Valor','Limiar','Unidade']],
                                use_container_width=True,
                                hide_index=True,
                            )

                # ── Loop de polling ────────────────────────────────────────
                _lv_api = st.session_state.get('api')

                if _lv_api is None:
                    st.warning(
                        "⚠️ API não inicializada. Carregue os dados na sidebar primeiro."
                    )
                elif st.session_state['live_active'] or _lv_refresh_now:
                    import time as _lv_time

                    # Buscar dados
                    _lv_info  = _lv_api.get_live_info()
                    _lv_data  = _lv_api.get_live_athletes()

                    # Guardar snapshot para exibição após parar
                    st.session_state['live_snapshot']      = _lv_data
                    st.session_state['live_info_snapshot'] = _lv_info

                    _render_live(_lv_info, _lv_data)

                    # Continuar polling se ativo
                    if st.session_state['live_active'] and not _lv_refresh_now:
                        _lv_time.sleep(int(st.session_state.get('live_interval', 10)))
                        st.rerun()

                elif st.session_state.get('live_snapshot') is not None:
                    # Mostrar último snapshot após pausar
                    st.caption("📸 Último snapshot (monitoramento pausado)")
                    _render_live(
                        st.session_state['live_info_snapshot'],
                        st.session_state['live_snapshot'],
                    )
                else:
                    # Estado inicial
                    st.markdown(
                        "<div style='text-align:center;padding:48px 0;"
                        "color:#64748b'>"
                        "<div style='font-size:48px;margin-bottom:12px'>📡</div>"
                        "<div style='font-size:18px;font-weight:600;color:#94a3b8'>"
                        "Monitoramento ao vivo pronto</div>"
                        "<div style='font-size:14px;margin-top:8px'>"
                        "Configure os limiares acima e clique em "
                        "<b style='color:#10b981'>▶ Iniciar Monitoramento</b><br>"
                        "para começar a acompanhar a sessão em tempo real."
                        "</div></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "> **Pré-requisito:** uma sessão deve estar aberta e transmitindo "
                        "dados no OpenField Cloud agora. O endpoint `/live` só retorna dados "
                        "durante sessões ativas."
                    )

        else:
            st.warning("Nenhum dado encontrado")

    elif 'df_athletes' in st.session_state and not st.session_state.df_athletes.empty:
        st.info("👈 Selecione uma atividade, período(s) e clique em 'Buscar Atletas da Atividade'")

if __name__ == "__main__":
    # (P3) Rede de segurança: qualquer erro não tratado é REGISTRADO com
    # traceback nos logs do servidor e mostrado de forma amigável — em vez de
    # sumir ou virar um traceback cru na tela.
    try:
        main()
    except Exception:
        _applog.log_exc("Erro não tratado em main()")
        try:
            st.error("⚠️ Ocorreu um erro inesperado. A equipe foi notificada "
                     "pelos logs. Recarregue a página; se persistir, reporte ao "
                     "suporte com o horário.")
            with st.expander("Detalhes técnicos"):
                import traceback as _tb
                st.code(_tb.format_exc())
        except Exception:
            raise
    