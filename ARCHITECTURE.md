# Arquitetura & Modularização (P4)

O app nasceu como um único arquivo (`futebol-eventos.py`, ~15,6 mil linhas). P4 é
o processo — **incremental e testado** — de extrair responsabilidades em módulos
puros/coesos, reduzindo o monólito sem quebrar produção. **Toda a infraestrutura
foi extraída** (10 módulos); o pacote `viz/` (camada de UI) foi iniciado.

## Módulos extraídos

| Módulo | Responsabilidade | Acoplamento | Testes |
|---|---|---|---|
| `metrics.py` | Motor de cálculo (distância, bandas, PlayerLoad, ACWR, calibração) | Puro (numpy) | `tests/test_metrics.py` |
| `validation.py` | Concordância / Bland-Altman (estudo de validação) | Puro (numpy/pandas) | `tests/test_validation.py` |
| `storage.py` | Persistência chave→valor (local + Supabase) | Puro | `tests/test_storage.py` |
| `applog.py` | Logging estruturado | Puro (stdlib) | `tests/test_applog.py` |
| `catapult_api.py` | Cliente HTTP da Catapult Connect v6 (`_api_fetch` + `CatapultAPI`) | streamlit (cache) + requests | `tests/test_e2e_load.py` |
| `ui_theme.py` | CSS global + helpers de design (`_hr`, `_badge`) | streamlit (markup) | `tests/test_e2e_load.py` |
| `diagnostics.py` | Selo de proveniência + diagnóstico da sessão (`_diag_log`) | streamlit (sessão) | `tests/test_e2e_load.py` |
| `config.py` | Constantes (servidores, i18n LANGUAGES, bandas, Gen2, eventos) | Puro (dados) | — |
| `i18n.py` | Traduções (`TRANSLATIONS`) + `t()` | streamlit (sessão) | `tests/test_e2e_load.py` |
| `viz/monitoramento.py` | Aba Monitoramento (ACWR/monotonia/strain) — 1ª `render_*` | streamlit + metrics/diagnostics/i18n | `tests/test_e2e_load.py` |

Resultado: monólito de ~15,6k → ~14,5k linhas, agora com **fronteiras de módulo
limpas**. O que sobra em `futebol-eventos.py` é a camada de **UI/render**
(demais `render_*` + seções das abas) e a **orquestração de estado**.

## Restante (mesma receita, mecânica e comprovada)

O pacote `viz/` está estabelecido e o padrão validado pelo E2E. As demais
`render_*` seguem idênticas — mover uma por commit, mantendo o E2E verde:

1. **`viz/tatica_coletiva.py`** e **`viz/export_artigo.py`** — as outras duas
   `render_*` autônomas. Antes, extrair os helpers de banda que elas usam
   (`_bandas_vel_ativas`/`_bandas_acc_ativas`) para um `bands.py`.
2. **Seções inline das abas** (Resumo, Campo, WCS, Neuromuscular, Janelas,
   Acc-Vel, FC) — hoje dentro de `main()`; transformar cada uma em
   `render_*(...)` e mover para `viz/`.
3. **`state.py`** (opcional) — modelo central do `st.session_state`
   (dataclass + versão de schema) para reduzir bugs de estado.
4. **`data_layer.py`** (opcional) — repositório sobre `catapult_api` com cache
   TTL, retry/backoff e cache do sinal 10 Hz em parquet (ver P7).

## Regras da modularização

- **Uma extração por commit**, com `tests/` + E2E verdes antes de push.
- Módulos novos **puros sempre que possível** (sem `import streamlit`), para
  serem testáveis isoladamente. Quando o Streamlit for inevitável (cache,
  sessão), isolar num módulo "streamlit-aware" claramente identificado.
- Nada de dependência reversa: um módulo extraído **não** importa
  `futebol-eventos.py`.
