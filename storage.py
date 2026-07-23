# -*- coding: utf-8 -*-
"""Camada de persistência durável (P2) — armazenamento chave→valor (JSON).

Motivação: no Streamlit Cloud o sistema de arquivos do container é EFÊMERO.
Gravar venues/bandas em arquivos locais faz a configuração do clube desaparecer
a cada redeploy/reboot. Este módulo abstrai o armazenamento com dois backends:

- LocalJSONStore: arquivos JSON (comportamento antigo, NÃO durável). Fallback.
- SupabaseStore: tabela chave→valor via API REST (PostgREST) usando `requests`
  — durável, sem driver de banco (evita risco de wheel no Python novo).

O app escolhe o backend em runtime: se houver credenciais do Supabase em
st.secrets, usa o durável; senão, cai no local com um aviso. As classes aqui
são PURAS (sem Streamlit), para poderem ser testadas isoladamente.

Backend Supabase — SQL da tabela (rode uma vez no editor SQL do Supabase):

    create table if not exists app_kv (
        key   text primary key,
        value jsonb,
        updated_at timestamptz default now()
    );
    -- Opcional (RLS liberado para a service key; NÃO exponha a service key no
    -- cliente — aqui ela vive apenas em st.secrets, no servidor do app):
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

try:
    import applog as _applog
except Exception:                                 # pragma: no cover
    class _applog:                                # fallback silencioso
        @staticmethod
        def log_exc(ctx=""): pass
        @staticmethod
        def log_warn(msg=""): pass


class BaseStore:
    durable: bool = False

    def get(self, key: str) -> Optional[Any]:      # pragma: no cover - interface
        raise NotImplementedError

    def set(self, key: str, value: Any) -> bool:   # pragma: no cover - interface
        raise NotImplementedError

    def delete(self, key: str) -> bool:            # pragma: no cover - interface
        raise NotImplementedError


class LocalJSONStore(BaseStore):
    """Persistência em arquivos JSON (um por chave). NÃO durável no Cloud."""
    durable = False

    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _path(self, key: str) -> str:
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(key))
        return os.path.join(self.base_dir, f"{safe}.json")

    def get(self, key: str) -> Optional[Any]:
        try:
            with open(self._path(key), encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> bool:
        try:
            with open(self._path(key), "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            _applog.log_exc(f"LocalJSONStore.set falhou (key={key})")
            return False

    def delete(self, key: str) -> bool:
        try:
            os.remove(self._path(key))
            return True
        except FileNotFoundError:
            return True
        except Exception:
            return False


class SupabaseStore(BaseStore):
    """KV durável via PostgREST do Supabase (HTTP + `requests`).

    Tabela `app_kv(key text primary key, value jsonb)`. Usa upsert
    (Prefer: resolution=merge-duplicates). Nunca lança exceção — retorna
    None/False em erro, para o app poder cair no fallback com segurança.
    """
    durable = True

    def __init__(self, url: str, apikey: str, table: str = "app_kv",
                 timeout: float = 10.0):
        self.endpoint = url.rstrip("/") + "/rest/v1/" + table
        self.timeout = timeout
        self._headers = {
            "apikey": apikey,
            "Authorization": f"Bearer {apikey}",
            "Content-Type": "application/json",
        }

    def get(self, key: str) -> Optional[Any]:
        import requests
        try:
            r = requests.get(self.endpoint, headers=self._headers,
                             params={"key": f"eq.{key}", "select": "value"},
                             timeout=self.timeout)
            if r.status_code == 200:
                rows = r.json()
                if rows:
                    return rows[0].get("value")
        except Exception:
            pass
        return None

    def set(self, key: str, value: Any) -> bool:
        import requests
        try:
            r = requests.post(
                self.endpoint,
                headers={**self._headers, "Prefer": "resolution=merge-duplicates"},
                json={"key": key, "value": value}, timeout=self.timeout)
            if r.status_code not in (200, 201, 204):
                _applog.log_warn(f"SupabaseStore.set HTTP {r.status_code} (key={key})")
            return r.status_code in (200, 201, 204)
        except Exception:
            _applog.log_exc(f"SupabaseStore.set falhou (key={key})")
            return False

    def delete(self, key: str) -> bool:
        import requests
        try:
            r = requests.delete(self.endpoint, headers=self._headers,
                                params={"key": f"eq.{key}"}, timeout=self.timeout)
            return r.status_code in (200, 204)
        except Exception:
            return False
