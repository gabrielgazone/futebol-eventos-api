# Arquitetura & Modularização (P4)

O app nasceu como um único arquivo (`futebol-eventos.py`, **~15,6 mil linhas**)
que misturava tudo. P4 extraiu — **incremental, testado, com E2E + pyflakes
verdes a cada commit** — todas as responsabilidades separáveis em módulos.

**Resultado final: 29 módulos; monólito reduzido de ~15,6 mil para ~1,79 mil
linhas (−88,5%). O `futebol-eventos.py` é apenas `main()` (barra lateral de
conexão/seleção + a estrutura de abas chamando os `render_*`). Toda a lógica,
compute, plotagem, carga de dados E a UI das abas vivem em módulos.**

> Contexto: P4 é o 4º item de uma lista de 10 melhorias de arquitetura —
> **P1–P10 estão todos concluídos** (versões fixadas, persistência durável,
> observabilidade, modularização, token/segredos, esquema de estado, camada de
> dados com retry/backoff, cobertura de testes 62→110, performance e portões de
> CI com ruff).

## Módulos extraídos (29)

| Módulo | Responsabilidade |
|---|---|
| `metrics.py` | Motor de cálculo (distância, bandas, PlayerLoad, ACWR, calibração) |
| `validation.py` | Concordância / Bland-Altman (estudo de validação) |
| `storage.py` | Persistência chave→valor (local + Supabase) |
| `persistence.py` | Store + venues + bandas do usuário + prefs (sobre `storage`) |
| `applog.py` | Logging estruturado + redação de segredos (P5) |
| `config.py` | Constantes (servidores, i18n, bandas, Gen2, eventos, paletas) |
| `state.py` | Esquema central do `st.session_state` + versão/reset (P6) |
| `catapult_api.py` | Cliente HTTP da Catapult Connect v6 + retry/backoff (P7) |
| `data_loader.py` | Carga de dados: pré-busca paralela (P9) + `carregar_dados()` |
| `i18n.py` | Traduções + `t()` |
| `diagnostics.py` | Selo de proveniência + diagnóstico da sessão |
| `bands.py` | Cortes de banda + rótulos/formatação + parsers de zona da API |
| `analysis.py` | Compute: métricas por atleta, janelas, esforços, gráficos |
| `field.py` | Campo/plotagem, trajetórias/heatmaps, eventos, Voronoi, neuro, ACWR |
| `ui_theme.py` | CSS global + helpers de design |
| `viz/` (pacote) | **13 abas**: `visao_geral`, `campo`, `janelas`, `neuromuscular`, `acc_vel`, `fc`, `por_posicao`, `wcs`, `ao_vivo`, `tatica_coletiva`, `export_artigo`, `monitoramento`, `esforcos` |

Grafo de dependências acíclico: `viz/*` → `field`/`analysis`/`bands`/… →
`metrics`/`config`/`applog`. Nenhum módulo importa `futebol-eventos.py`.

Cada `render_*(...)` recebe os dados carregados como parâmetros (mesmos nomes
do escopo de `main`), descobertos por análise de variáveis livres + verificação
`pyflakes` (garante zero nomes indefinidos — nenhum import ou dado faltando).

## Verificação

- `tests/test_{metrics,validation,storage,applog}.py` — unitários dos módulos puros.
- `tests/test_e2e_load.py` — `AppTest` que **renderiza o app inteiro** (cobre os
  módulos streamlit-aware e as 3 `render_*`).
- **`pyflakes`** em todo o repo — garante **zero nomes indefinidos** (pegou bugs
  latentes que o E2E não exercitava, ex.: import faltando num ramo condicional).

## `main()` — o que sobrou (orquestração pura)

O `futebol-eventos.py` (~1,79k linhas) é a função `main()`: barra lateral
(conexão, filtros, seleção, editores), a estrutura de abas (`st.tabs`) e a
chamada de cada `render_*` do pacote `viz/`. A carga de dados já vive em
`data_loader.carregar_dados()`. É o **shell de orquestração** — nada de lógica
de negócio, compute ou plotagem.

**P4 concluído.** Todo o código separável — lógica, dados, API, tema, i18n,
persistência, compute, plotagem e as 13 abas — está em 29 módulos, com grafo
acíclico e verificação por testes + E2E + pyflakes.

### Por que paramos em −88,5% (e não perseguimos 100%)

Os 11,5% restantes são o **shell de entrada do Streamlit** — barra lateral
(fluxo de conexão) + estrutura de abas. Extrair a barra lateral para um módulo
**não** foi feito de propósito: ela é UI acoplada ao I/O de conexão, **não é
reutilizável, não é testável isoladamente e não é compartilhada**; movê-la só
relocaria o contrato de `session_state` (mesmo acoplamento atrás de um `import`)
— "teatro de modularização". Além disso, é o fluxo de conexão dos clientes
reais ("zero erro"): risco alto, benefício nulo em testabilidade/reuso. O ponto
de parada certo é quando o próximo passo adiciona risco sem adicionar valor.
