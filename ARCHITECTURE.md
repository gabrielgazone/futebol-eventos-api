# Arquitetura & Modularização (P4)

O app nasceu como um único arquivo (`futebol-eventos.py`, **~15,6 mil linhas**)
que misturava tudo. P4 extraiu — **incremental, testado, com E2E + pyflakes
verdes a cada commit** — todas as responsabilidades separáveis em módulos.

**Resultado: 18 módulos; monólito reduzido a ~9,0 mil linhas (−42%). O
`futebol-eventos.py` não contém mais nenhuma função helper top-level — apenas
`main()` (a página/orquestração).**

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
| `viz/monitoramento.py` | Aba Monitoramento (ACWR/monotonia/strain) |
| `viz/tatica_coletiva.py` | Aba Tática Coletiva (Pitch Control, Voronoi, replay 3D) |
| `viz/export_artigo.py` | Aba Exportação para Artigo (tabela + validação) |

Grafo de dependências acíclico: `viz/*` → `field`/`analysis`/`bands`/… →
`metrics`/`config`/`applog`. Nenhum módulo importa `futebol-eventos.py`.

## Verificação

- `tests/test_{metrics,validation,storage,applog}.py` — unitários dos módulos puros.
- `tests/test_e2e_load.py` — `AppTest` que **renderiza o app inteiro** (cobre os
  módulos streamlit-aware e as 3 `render_*`).
- **`pyflakes`** em todo o repo — garante **zero nomes indefinidos** (pegou bugs
  latentes que o E2E não exercitava, ex.: import faltando num ramo condicional).

## O que resta: `main()` (a camada de página)

O `futebol-eventos.py` é agora **só a função `main()`** — barra lateral
(conexão, filtros, seleção, editores), loop de carga, e as **seções inline das
abas** (Resumo, Campo & GPS + subabas, Ao Vivo) renderizadas no corpo de `main`.

Isso é o **fim natural do de-monolito**: todo o código reutilizável/testável
(lógica, dados, API, tema, i18n, persistência, compute, plot) está em módulos; o
que sobra é o **fluxo de página** de um app Streamlit, que idiomaticamente vive
no script principal.

### Decomposição adicional (opcional)

Quebrar as seções inline das abas em `viz/*` exige **refatorar `main()`**: montar
um objeto de contexto com os dados carregados e transformar cada bloco
`with aba[i]:` em `render_x(ctx)`. É page-decomposition (não de-monolito de
helpers), a fatia mais acoplada; deve ser feita **uma aba por commit, com E2E +
pyflakes verdes**. As 3 `render_*` já extraídas são o modelo.
