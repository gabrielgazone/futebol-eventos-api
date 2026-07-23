# Arquitetura & Modularização (P4)

O app nasceu como um único arquivo (`futebol-eventos.py`, **~15,6 mil linhas**)
que misturava tudo. P4 extraiu — **incremental e testado, E2E verde a cada
commit** — todas as responsabilidades separáveis em módulos coesos.

**Resultado: 14 módulos, monólito reduzido a ~12,6 mil linhas (−19%).** Toda a
infraestrutura e as três `render_*` autônomas estão fora do arquivo principal.

## Módulos extraídos

| Módulo | Responsabilidade | Acoplamento |
|---|---|---|
| `metrics.py` | Motor de cálculo (distância, bandas, PlayerLoad, ACWR, calibração) | Puro (numpy) |
| `validation.py` | Concordância / Bland-Altman (estudo de validação) | Puro |
| `storage.py` | Persistência chave→valor (local + Supabase) | Puro |
| `applog.py` | Logging estruturado | Puro (stdlib) |
| `config.py` | Constantes (servidores, i18n LANGUAGES, bandas, Gen2, eventos) | Puro (dados) |
| `catapult_api.py` | Cliente HTTP da Catapult Connect v6 | streamlit (cache) + requests |
| `i18n.py` | Traduções (`TRANSLATIONS`) + `t()` | streamlit (sessão) |
| `diagnostics.py` | Selo de proveniência + diagnóstico da sessão (`_diag_log`) | streamlit (sessão) |
| `persistence.py` | Store + venues + bandas do usuário (sobre `storage`) | streamlit + storage |
| `bands.py` | Cortes ativos de banda + rótulos/formatação | streamlit + config |
| `ui_theme.py` | CSS global + helpers de design | streamlit |
| `viz/monitoramento.py` | Aba Monitoramento (ACWR/monotonia/strain) | viz |
| `viz/tatica_coletiva.py` | Aba Tática Coletiva (Pitch Control, Voronoi, replay 3D) | viz |
| `viz/export_artigo.py` | Aba Exportação para Artigo (tabela + validação) | viz |

Testes: `tests/test_{metrics,validation,storage,applog}.py` (unitários dos
módulos puros) + `tests/test_e2e_load.py` (`AppTest` que renderiza o app inteiro
— cobre os módulos streamlit-aware e as `render_*`).

## O que resta no `futebol-eventos.py`

A função `main()` — **orquestração + as seções inline das abas** que ainda vivem
dentro dela (Resumo, Campo & GPS com 6 subabas, Ao Vivo). Diferente das 3
`render_*` autônomas, essas seções são **código inline no corpo de `main()`**,
lendo dezenas de variáveis locais (dados por período/atleta, ids, seleção…).

## Refinamento final opcional (split das abas inline)

Mover essas seções para `viz/` exige, antes, **refatorar `main()`**: transformar
cada bloco `with aba[i]:` numa função `render_x(dados…)` com parâmetros
explícitos, e só então movê-la. É a maior e mais acoplada fatia; deve ser feita
**uma aba por commit, com o E2E verde** (que já pegou regressões reais nesta
rodada). Não é pré-requisito de boa arquitetura — o `main()` é legitimamente a
camada de página/orquestração de um app Streamlit.

## Regras da modularização (seguidas em todos os 14 commits)

- **Uma extração por commit**, com `tests/` + E2E verdes antes do push.
- Módulos **puros quando possível**; quando o Streamlit é inevitável
  (cache/sessão), o módulo é "streamlit-aware" e declarado como tal.
- **Sem dependência reversa**: nenhum módulo importa `futebol-eventos.py`.
- Constantes/estado compartilhados vão para `config`/`persistence`/`bands`,
  nunca duplicados nem presos num módulo de UI.
