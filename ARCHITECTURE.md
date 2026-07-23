# Arquitetura & Modularização (P4)

O app nasceu como um único arquivo (`futebol-eventos.py`, ~15 mil linhas). P4 é
o processo — **incremental e testado** — de extrair responsabilidades em módulos
puros/coesos, reduzindo o monólito sem quebrar produção.

## Módulos já extraídos

| Módulo | Responsabilidade | Acoplamento | Testes |
|---|---|---|---|
| `metrics.py` | Motor de cálculo (distância, bandas, PlayerLoad, ACWR, calibração) | Puro (numpy) | `tests/test_metrics.py` |
| `validation.py` | Concordância / Bland-Altman (estudo de validação) | Puro (numpy/pandas) | `tests/test_validation.py` |
| `storage.py` | Persistência chave→valor (local + Supabase) | Puro | `tests/test_storage.py` |
| `applog.py` | Logging estruturado | Puro (stdlib) | `tests/test_applog.py` |
| `catapult_api.py` | Cliente HTTP da Catapult Connect v6 (`_api_fetch` + `CatapultAPI`) | streamlit (cache) + requests | `tests/test_e2e_load.py` |

O que sobra em `futebol-eventos.py`: a camada de **UI/render** (Streamlit) e a
orquestração de estado.

## Roadmap (ordem sugerida, por risco/valor)

1. **`ui_theme.py`** — o CSS global e helpers de design (`_hr`, `_badge`,
   cabeçalho). Baixo risco (quase sem lógica), alto ganho de legibilidade.
2. **`state.py`** — um modelo central do `st.session_state` (dataclass + versão
   de schema), substituindo as dezenas de chaves soltas. Reduz bugs de estado.
3. **`data_layer.py`** — repositório sobre `catapult_api` com cache TTL,
   retry/backoff e cache em parquet do sinal 10 Hz (ver P7 da lista de melhorias).
4. **`viz/`** (pacote) — mover as funções `render_*` uma a uma:
   `render_tatica_coletiva`, `render_export_artigo`, `render_monitoramento`,
   e as seções de Resumo / Campo / WCS / Neuromuscular / Janelas / Acc-Vel / FC.
   Cada extração deve manter o E2E (`AppTest`) verde.
5. **`config.py`** — constantes (zonas padrão, mapeamentos Gen2, servidores,
   i18n) num só lugar.

## Regras da modularização

- **Uma extração por commit**, com `tests/` + E2E verdes antes de push.
- Módulos novos **puros sempre que possível** (sem `import streamlit`), para
  serem testáveis isoladamente. Quando o Streamlit for inevitável (cache,
  sessão), isolar num módulo "streamlit-aware" claramente identificado.
- Nada de dependência reversa: um módulo extraído **não** importa
  `futebol-eventos.py`.
