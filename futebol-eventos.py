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
from viz.visao_geral import render_visao_geral  # noqa: E402
from viz.campo import render_campo  # noqa: E402
from viz.janelas import render_janelas  # noqa: E402
from viz.neuromuscular import render_neuromuscular  # noqa: E402
from viz.acc_vel import render_acc_vel  # noqa: E402
from viz.fc import render_fc  # noqa: E402
from viz.por_posicao import render_por_posicao  # noqa: E402
from viz.wcs import render_wcs  # noqa: E402
from viz.ao_vivo import render_ao_vivo  # noqa: E402
from viz.esforcos import render_esforcos  # noqa: E402


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
            help="O token fica mascarado, NÃO é gravado em disco e é removido "
                 "dos logs (redação automática). Boas práticas: gere com o "
                 "ESCOPO MÍNIMO necessário (Dados resumidos + Sensor 10 Hz) e "
                 "validade CURTA; revogue no OpenField se vazar. Em deploy, "
                 "configure CATAPULT_TOKEN em st.secrets.")
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
                    st.warning(f"Último erro da API: `{_applog.redact(_api_err)}`")
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
                    f"⚠️ **Erro na API Catapult** ({_applog.redact(_api_err)}). "
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
                render_visao_geral(resultados_por_periodo)

            # ==================== ABA 1: CAMPO DE FUTEBOL ====================
            with abas[0]:
                render_campo(REFERENCIAS, _campo_component, dados_efforts_acc_por_periodo, dados_efforts_vel_por_periodo, dados_eventos_por_periodo, dados_posicao_por_periodo, resultados_por_periodo)

            # ==================== ABA 2: ESFORÇOS AO LONGO DO TEMPO ====================
            with abas[1]:
                render_esforcos(_SENSOR_HZ, dados_sensor_por_atleta_por_periodo)

            # ==================== ABA 3: JANELAS TEMPORAIS MÓVEIS ====================
            with abas[2]:
                render_janelas(REFERENCIAS, _REL_VEL_BANDAS, _SENSOR_HZ, dados_efforts_acc_por_periodo, dados_posicao_por_periodo, dados_sensor_por_atleta_por_periodo)

            # ==================== ABA 4: CARGA NEUROMUSCULAR ====================
            with abas[3]:
                render_neuromuscular(_SENSOR_HZ, dados_sensor_por_atleta_por_periodo)

            # ==================== ABA 5: PERFIL ACELERAÇÃO-VELOCIDADE ====================
            with abas[4]:
                render_acc_vel(dados_sensor_por_atleta_por_periodo, resultados_por_periodo)

            # ══════════════════════════════════════════════════════════════
            # ABA 6: FC — TRIMP & ZONAS DE FREQUÊNCIA CARDÍACA
            # ══════════════════════════════════════════════════════════════
            with abas[5]:
                render_fc(dados_sensor_por_atleta_por_periodo, resultados_por_periodo)

            # ABA 6: POR POSIÇÃO
            # ══════════════════════════════════════════════════════════════
            with abas[6]:
                render_por_posicao(_POSICAO_COR_LEGENDA, resultados_por_periodo)

            # ══════════════════════════════════════════════════════════════
            # ABA 7: HISTÓRIA DO JOGO — removida (P8)
            # ══════════════════════════════════════════════════════════════

            # ══════════════════════════════════════════════════════════════
            # WCS: WORST-CASE SCENARIO (sub-tab Campo & GPS)
            # ══════════════════════════════════════════════════════════════
            with _sub_campo[1]:
                render_wcs(_REL_VEL_BANDAS, _SENSOR_HZ, _ok_ld, dados_efforts_acc_por_periodo, dados_posicao_por_periodo, dados_sensor_por_atleta_por_periodo, resultados_por_periodo)

            # ==================== ABA 8: MONITORAMENTO AO VIVO ====================
            with abas[8]:
                render_ao_vivo()

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
    