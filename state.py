# -*- coding: utf-8 -*-
"""Esquema central do estado da sessão (P6).

O app guarda dezenas de valores em `st.session_state`. Este módulo dá a eles um
lugar documentado, uma VERSÃO de esquema e um ponto único de inicialização:

- `KEYS`: os nomes das chaves como constantes (evita erros de digitação).
- `SCHEMA_VERSION`: incremente ao mudar a forma do estado; força a limpeza do
  estado volátil no próximo render (evita "estado velho após deploy", que já
  causou crashes — AttributeError/telas travadas — nesta base).
- `init(st)`: chamado uma vez no topo de `main()`. Aplica defaults seguros e,
  se a versão mudou, descarta as chaves VOLÁTEIS (dados carregados da API,
  re-obteníveis com um novo "Carregar Dados").

Funções puras (recebem o objeto `st`), para poderem ser testadas.
"""
from __future__ import annotations

SCHEMA_VERSION = "2026-07-24-1"


class KEYS:
    # ── Conexão / sessão ────────────────────────────────────────────────
    API            = "api"
    ORG_MARKER     = "_org_marker"
    TOKEN_MARKER   = "_token_marker"
    KV_STORE       = "_kv_store"
    PERSIST_EFEMERA = "_persist_efemera"
    API_LAST_ERR   = "_api_last_err"
    # ── Dados carregados (VOLÁTEIS — re-obtidos a cada "Carregar Dados") ─
    DF_TEAMS       = "df_teams"
    DF_ATHLETES    = "df_athletes"
    DF_ACTIVITIES  = "df_activities"
    DF_POSITIONS   = "df_positions"
    DF_PARAMETERS  = "df_parameters"
    ATLETAS_FILTRADOS = "atletas_filtrados"
    ATLETAS_SEL    = "atletas_sel"
    ACTIVITY_ID    = "activity_id"
    PERIOD_IDS     = "period_ids"
    PERIOD_OPTIONS = "period_options"
    PERIODOS_SEL   = "periodos_selecionados"
    AVAILABLE_PARAMS = "available_params"
    ATHLETE_COLORS = "athlete_colors"
    ATHLETE_TEAM_MAP = "athlete_team_map"
    VENUE          = "venue"
    HIST_VMAX      = "hist_vmax"
    HIST_VMAX_SOURCE = "hist_vmax_source"
    # ── Zonas/bandas (fonte na sessão) ──────────────────────────────────
    VEL_ZONES      = "velocity_zones_account"
    ACC_ZONES      = "acceleration_zones_account"
    # ── Seleções / UI ───────────────────────────────────────────────────
    EQUIPES_SEL    = "equipes_selecionadas"
    POSICOES_SEL   = "posicoes_selecionadas"
    EVENTOS_SEL    = "eventos_futebol_sel"
    MODO_APRES     = "modo_apresentacao"
    ONBOARDING_STEP = "onboarding_step"
    ONBOARDING_DONE = "onboarding_done"
    # ── Diagnóstico ─────────────────────────────────────────────────────
    DIAG_EVENTOS   = "_diag_eventos"


# Estado VOLÁTIL: dados carregados da API + caches derivados. Limpos num bump de
# SCHEMA_VERSION (deploy) para nunca renderizar com estrutura incompatível.
VOLATILE = (
    KEYS.DF_TEAMS, KEYS.DF_ATHLETES, KEYS.DF_ACTIVITIES, KEYS.DF_POSITIONS,
    KEYS.DF_PARAMETERS, KEYS.ATLETAS_FILTRADOS, KEYS.ATLETAS_SEL,
    KEYS.ACTIVITY_ID, KEYS.PERIOD_IDS, KEYS.PERIOD_OPTIONS, KEYS.PERIODOS_SEL,
    KEYS.AVAILABLE_PARAMS, KEYS.ATHLETE_COLORS, KEYS.ATHLETE_TEAM_MAP,
    KEYS.VENUE, KEYS.HIST_VMAX, KEYS.HIST_VMAX_SOURCE, KEYS.API_LAST_ERR,
    KEYS.DIAG_EVENTOS,
)

# Defaults seguros aplicados na inicialização (chave -> valor).
DEFAULTS = {
    KEYS.MODO_APRES: False,
    KEYS.ONBOARDING_STEP: 0,
}


def init(st) -> bool:
    """Inicializa o estado no topo de main(). Retorna True se houve reset de
    esquema (deploy com estado incompatível). Idempotente por render."""
    ss = st.session_state
    resetou = ss.get("_state_schema") != SCHEMA_VERSION
    if resetou:
        for k in VOLATILE:
            ss.pop(k, None)
        ss["_state_schema"] = SCHEMA_VERSION
    for k, v in DEFAULTS.items():
        ss.setdefault(k, v)
    return resetou
