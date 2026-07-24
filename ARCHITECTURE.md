# Arquitetura & Modularização (P4)

O app nasceu como um único arquivo (`futebol-eventos.py`, **~15,6 mil linhas**)
que misturava tudo. P4 extraiu — **incremental, testado, com E2E + pyflakes
verdes a cada commit** — todas as responsabilidades separáveis em módulos.

**Resultado final: 26 módulos; monólito reduzido de ~15,6 mil para ~2,2 mil
linhas (−86%). O `futebol-eventos.py` é apenas `main()` (orquestração: barra
lateral, conexão, loop de carga e a estrutura de abas chamando os `render_*`).
Toda a lógica, compute, plotagem E a UI das abas vivem em módulos.**

## Módulos extraídos (18)

| Módulo | Responsabilidade |
|---|---|
| `metrics.py` | Motor de cálculo (distância, bandas, PlayerLoad, ACWR, calibração) |
| `validation.py` | Concordância / Bland-Altman (estudo de validação) |
| `storage.py` | Persistência chave→valor (local + Supabase) |
| `persistence.py` | Store + venues + bandas do usuário + prefs (sobre `storage`) |
| `applog.py` | Logging estruturado |
| `config.py` | Constantes (servidores, i18n, bandas, Gen2, eventos, paletas) |
| `catapult_api.py` | Cliente HTTP da Catapult Connect v6 |
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

O `futebol-eventos.py` (~2,2k linhas) é a função `main()`: barra lateral
(conexão, filtros, seleção, editores), o loop de carga da API, a estrutura de
abas (`st.tabs`) e a chamada de cada `render_*` do pacote `viz/`. É o **shell de
orquestração** — nada de lógica de negócio, compute ou plotagem.

**P4 concluído.** Todo o código separável — lógica, dados, API, tema, i18n,
persistência, compute, plotagem e as 13 abas — está em 26 módulos, com grafo
acíclico e verificação por testes + E2E + pyflakes.
