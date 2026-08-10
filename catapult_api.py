# -*- coding: utf-8 -*-
"""Cliente da API Catapult Connect v6 (P4 — extraído do monólito).

Contém o funil HTTP cacheado (_api_fetch) e a classe CatapultAPI (todos os
endpoints usados pelo app). Depende apenas de streamlit (cache + registro de
erro na sessão), requests e applog. Sem acoplamento com o resto do app.
"""
from __future__ import annotations

import time

import streamlit as st
import requests
import applog as _applog


# ==================== CAMADA DE DADOS — retry/backoff (P7) ====================
# Falhas transitórias que valem uma nova tentativa (rate-limit + erros de
# servidor). 429 respeita o cabeçalho Retry-After quando presente.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


def _http(method: str, url: str, *, headers=None, params=None, json=None,
          timeout: float = 60, max_retries: int = 3, base_delay: float = 0.5,
          sleep=time.sleep):
    """GET/POST com retry exponencial em 429/5xx/erros de rede. Usa
    getattr(requests, method) (compatível com o mock do E2E). Retorna o
    Response (mesmo no erro final) ou relança a última exceção de rede.
    `sleep` é injetável para testes."""
    fn = getattr(requests, method.lower())
    delay = base_delay
    r = None
    for attempt in range(max_retries + 1):
        try:
            kw = {'headers': headers, 'params': params, 'timeout': timeout}
            if json is not None:
                kw['json'] = json
            r = fn(url, **kw)
            if r.status_code in _RETRY_STATUS and attempt < max_retries:
                _ra = (getattr(r, 'headers', None) or {}).get('Retry-After')
                wait = float(_ra) if (isinstance(_ra, str) and _ra.isdigit()) else delay
                _applog.log_warn(f"API {r.status_code} em {method.upper()} — "
                                 f"retry {attempt+1}/{max_retries} em {min(wait,10):.1f}s")
                sleep(min(wait, 10)); delay *= 2
                continue
            return r
        except requests.RequestException:
            if attempt < max_retries:
                _applog.log_warn(f"Rede falhou — retry {attempt+1}/{max_retries} "
                                 f"em {delay:.1f}s")
                sleep(delay); delay *= 2
                continue
            raise
    return r


# ==================== CACHE DE API (TTL 15 min) ====================
# Standalone function cacheada pelo Streamlit. Parâmetros como
# primitivos/tuples para serem hashable. TTL de 900s (15 min).
@st.cache_data(ttl=900, show_spinner=False)
def _api_fetch(base_url: str, token: str, path: str,
               params: tuple = ()) -> object:
    """Chamada HTTP GET à API Catapult com cache (15 min) e retry/backoff."""
    headers = {'Authorization': f'Bearer {token}'}
    try:
        r = _http('get', f"{base_url}/{path}", headers=headers,
                  params=dict(params), timeout=60)
        if r is not None and r.status_code == 200:
            return r.json()
        # Salva o status de erro para exibir ao usuário + loga (P3).
        _applog.log_warn(f"API {r.status_code} em GET /{path}")
        try:
            import streamlit as _st_fetch
            _st_fetch.session_state['_api_last_err'] = r.status_code
        except Exception:
            pass
        return None
    except Exception as _exc:
        _applog.log_exc(f"Falha de rede em GET /{path}")   # traceback nos logs
        try:
            import streamlit as _st_fetch
            _st_fetch.session_state['_api_last_err'] = str(_exc)
        except Exception:
            pass
        return None


class CatapultAPI:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self._token   = token                                  # usado no cache
        self.headers  = {'Authorization': f'Bearer {token}'}
    
    def get_athletes(self):
        return _api_fetch(self.base_url, self._token, "athletes")

    def get_athlete(self, athlete_id):
        """Perfil completo de um atleta (GET /athletes/{id})."""
        return _api_fetch(self.base_url, self._token, f"athletes/{athlete_id}")

    def get_teams(self):
        return _api_fetch(self.base_url, self._token, "teams")

    def get_team_athletes(self, team_id):
        return _api_fetch(self.base_url, self._token, f"teams/{team_id}/athletes")

    def get_activities(self):
        return _api_fetch(self.base_url, self._token, "activities",
                          params=(("page_size", "500"),))

    def get_activity_athletes(self, activity_id):
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/athletes")

    def get_activity_periods(self, activity_id):
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/periods")

    def get_deep_activity(self, activity_id):
        """GET /activities/{id}?include=all — atividade "profunda": períodos com
        start_time/end_time e os atletas de cada período, além de athletes
        (com position) e teams. Substitui ~5 chamadas por atividade e é a fonte
        AUTORITATIVA de duração de período e participação (o que o OpenField usa
        para decidir quem entra no export de cada tempo)."""
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}",
                          (("include", "all"),))

    def get_all_periods(self):
        return _api_fetch(self.base_url, self._token, "periods")

    def get_athletes_in_period(self, period_id):
        return _api_fetch(self.base_url, self._token,
                          f"periods/{period_id}/athletes")

    def get_positions(self):
        return _api_fetch(self.base_url, self._token, "positions")

    def get_parameters(self):
        return _api_fetch(self.base_url, self._token, "parameters")
    
    _SENSOR_PARAMS = (
        ("parameters", "ts,lat,long,v,rv,a,hr,pl,xy,pq,hdop,ref,o,mp"),
        ("nulls",      "1"),
    )

    def get_sensor_data(self, activity_id, athlete_id):
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/athletes/{athlete_id}/sensor",
                          params=self._SENSOR_PARAMS)

    def get_period_sensor_data(self, period_id, athlete_id):
        return _api_fetch(self.base_url, self._token,
                          f"periods/{period_id}/athletes/{athlete_id}/sensor",
                          params=self._SENSOR_PARAMS)

    def get_activity_efforts(self, activity_id, athlete_id,
                             effort_types="velocity,acceleration"):
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/athletes/{athlete_id}/efforts",
                          params=(("effort_types", effort_types),))

    def get_period_efforts(self, period_id, athlete_id,
                           effort_types="velocity,acceleration"):
        return _api_fetch(self.base_url, self._token,
                          f"periods/{period_id}/athletes/{athlete_id}/efforts",
                          params=(("effort_types", effort_types),))

    def get_activity_events(self, activity_id, athlete_id, event_types):
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/athletes/{athlete_id}/events",
                          params=(("event_types", event_types),))

    def get_period_events(self, period_id, athlete_id, event_types):
        return _api_fetch(self.base_url, self._token,
                          f"periods/{period_id}/athletes/{athlete_id}/events",
                          params=(("event_types", event_types),))

    # ── Live endpoints (sem cache — dados em tempo real) ──────────────
    def get_live_info(self):
        """Metadados da sessão ao vivo ativa (GET /live/info)."""
        import requests as _req
        try:
            r = _req.get(
                f"{self.base_url}/live/info",
                headers=self.headers, timeout=8,
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def get_live_athletes(self):
        """Métricas ao vivo de todos os atletas na sessão ativa (GET /live)."""
        import requests as _req
        try:
            r = _req.get(
                f"{self.base_url}/live",
                headers=self.headers, timeout=8,
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    # ── Tags, thresholds e stats (sem cache onde necessário) ─────────────
    def get_tags(self):
        """Todas as tags disponíveis no sistema (GET /tags)."""
        return _api_fetch(self.base_url, self._token, "tags")

    def get_activity_tags(self, activity_id):
        """Tags associadas a uma atividade específica (GET /activities/{id}/tags)."""
        return _api_fetch(self.base_url, self._token, f"activities/{activity_id}/tags")

    def get_athlete_thresholds(self, athlete_id):
        """Limiares individuais de velocidade/acc do atleta (GET /athletes/{id}/thresholds)."""
        return _api_fetch(self.base_url, self._token, f"athletes/{athlete_id}/thresholds")

    def get_stats(self, payload, timeout=20):
        """Estatísticas agregadas por grupo (POST /stats), com retry/backoff.
        Sem cache — dados dinâmicos. `timeout` menor evita travar a UI quando a
        consulta é opcional (ex.: sonda de parâmetro no export)."""
        try:
            r = _http('post', f"{self.base_url}/stats",
                      headers={**self.headers, "Content-Type": "application/json"},
                      json=payload, timeout=timeout)
            return r.json() if (r is not None and r.status_code == 200) else None
        except Exception:
            _applog.log_exc("Falha em POST /stats")
            return None

    # ── Velocity zones ───────────────────────────────────────────────────────
    def get_velocity_zones(self):
        """Bandas de velocidade configuradas na conta (GET /velocity_zones)."""
        return _api_fetch(self.base_url, self._token, "velocity_zones")

    def get_athlete_velocity_zones(self, athlete_id):
        """Bandas de velocidade personalizadas por atleta (GET /athletes/{id}/velocity_zones)."""
        return _api_fetch(self.base_url, self._token, f"athletes/{athlete_id}/velocity_zones")

    def get_team_velocity_zones(self, team_id):
        """Bandas de velocidade da equipe — onde ficam as 'Bandas Globais'
        configuradas na conta (GET /teams/{id}/velocity_zones)."""
        return _api_fetch(self.base_url, self._token, f"teams/{team_id}/velocity_zones")

    def get_team_acceleration_zones(self, team_id):
        """Bandas de aceleração da equipe (GET /teams/{id}/acceleration_zones)."""
        return _api_fetch(self.base_url, self._token, f"teams/{team_id}/acceleration_zones")

    def get_acceleration_zones(self):
        """Bandas de aceleração configuradas na conta (GET /acceleration_zones)."""
        return _api_fetch(self.base_url, self._token, "acceleration_zones")

    def get_athlete_acceleration_zones(self, athlete_id):
        """Bandas de aceleração personalizadas por atleta (GET /athletes/{id}/acceleration_zones)."""
        return _api_fetch(self.base_url, self._token, f"athletes/{athlete_id}/acceleration_zones")

    def get_settings(self):
        """Configurações/preferências do usuário (GET /settings).
        Retorna pares {key, value} — ex.: SpeedUnit, DistanceUnit.
        NÃO contém os cortes das bandas de velocidade (a API v6 não os expõe)."""
        return _api_fetch(self.base_url, self._token, "settings")

    # ── Activity/period summaries (pre-computed by OpenField) ───────────────
    def get_athlete_activity_summary(self, activity_id, athlete_id):
        """Resumo pré-computado pelo OpenField (GET /activities/{id}/athletes/{aid}/summary)."""
        return _api_fetch(self.base_url, self._token,
                          f"activities/{activity_id}/athletes/{athlete_id}/summary")

    def get_athlete_period_summary(self, period_id, athlete_id):
        """Resumo pré-computado por período (GET /periods/{id}/athletes/{aid}/summary)."""
        return _api_fetch(self.base_url, self._token,
                          f"periods/{period_id}/athletes/{athlete_id}/summary")

    # ── Session parameters (dynamic device discovery) ────────────────────────
    def get_session_parameters(self, activity_id):
        """Parâmetros disponíveis nesta sessão (GET /activities/{id}/parameters)."""
        return _api_fetch(self.base_url, self._token, f"activities/{activity_id}/parameters")

    # ── Venues from Catapult account ─────────────────────────────────────────
    def get_venues(self):
        """Venues cadastrados na conta (GET /venues)."""
        return _api_fetch(self.base_url, self._token, "venues")

    # ── Annotations ──────────────────────────────────────────────────────────
    def get_activity_annotations(self, activity_id):
        """Lista anotações de uma atividade (GET /activities/{id}/annotations)."""
        return _api_fetch(self.base_url, self._token, f"activities/{activity_id}/annotations")

    def create_activity_annotation(self, activity_id, name, start_time, end_time,
                                   annotation_type="phase"):
        """Cria nova anotação (POST /activities/{id}/annotations)."""
        import requests as _req
        try:
            payload = {
                "name": name,
                "start_time": start_time,
                "end_time": end_time,
                "annotation_type": annotation_type,
            }
            r = _req.post(
                f"{self.base_url}/activities/{activity_id}/annotations",
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload, timeout=20,
            )
            return r.json() if r.status_code in (200, 201) else None
        except Exception:
            return None

    def delete_annotation(self, annotation_id):
        """Remove uma anotação (DELETE /annotations/{id})."""
        import requests as _req
        try:
            r = _req.delete(
                f"{self.base_url}/annotations/{annotation_id}",
                headers=self.headers, timeout=15,
            )
            return r.status_code in (200, 204)
        except Exception:
            return False

    # ── Async export ─────────────────────────────────────────────────────────
    def submit_export(self, payload):
        """Submete job de exportação assíncrona (POST /export)."""
        import requests as _req
        try:
            r = _req.post(
                f"{self.base_url}/export",
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload, timeout=20,
            )
            return r.json() if r.status_code in (200, 201, 202) else None
        except Exception:
            return None

    def get_export_status(self, job_id):
        """Verifica status de um job de exportação (GET /export/{job_id})."""
        import requests as _req
        try:
            r = _req.get(
                f"{self.base_url}/export/{job_id}",
                headers=self.headers, timeout=15,
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def download_export(self, job_id):
        """Download do arquivo exportado (GET /export/{job_id}/download)."""
        import requests as _req
        try:
            r = _req.get(
                f"{self.base_url}/export/{job_id}/download",
                headers=self.headers, timeout=60,
            )
            return r.content if r.status_code == 200 else None
        except Exception:
            return None

    # PARTE 2 - FUNÇÕES DE EXTRAÇÃO, CONVERSÃO E CÁLCULO

