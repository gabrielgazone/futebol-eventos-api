# -*- coding: utf-8 -*-
"""Camada de carga de dados (P9 — performance).

Pré-busca EM PARALELO o sinal 10 Hz + efforts de cada (período × atleta),
aquecendo o cache `@st.cache_data` do cliente. O loop de processamento de
`main()` — que ainda roda sequencialmente para atualizar a barra de progresso —
passa a acertar o cache em cada atleta (chamadas instantâneas), trocando N
idas-e-voltas de rede sequenciais por N concorrentes.

Best-effort: exceções são logadas em DEBUG e ignoradas (o loop principal lida
com dados faltantes). As chamadas usam os métodos do cliente (com retry/backoff
do P7); os writes de erro em session_state dentro do cliente são protegidos, o
que torna seguro chamá-los de threads.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import applog as _applog


def prefetch_sensores(api, activity_id, period_ids, periodos, atletas_ids,
                      effort_types: str = "velocity,acceleration",
                      max_workers: int = 8) -> int:
    """Aquece o cache do sinal+efforts de todos os (período × atleta) em
    paralelo. Retorna o nº de tarefas disparadas."""
    tarefas = []
    for pnome in (periodos or []):
        pid = (period_ids or {}).get(pnome)
        for aid in (atletas_ids or []):
            tarefas.append((pid, aid))
    if not tarefas:
        return 0

    def _um(t):
        pid, aid = t
        try:
            if pid:
                api.get_period_sensor_data(pid, aid)
                api.get_period_efforts(pid, aid, effort_types)
            else:
                api.get_sensor_data(activity_id, aid)
                api.get_activity_efforts(activity_id, aid, effort_types)
        except Exception:
            _applog.log_debug_exc()

    try:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tarefas))) as ex:
            list(ex.map(_um, tarefas))
    except Exception:
        _applog.log_debug_exc()          # sem paralelismo, o loop ainda funciona
    return len(tarefas)
