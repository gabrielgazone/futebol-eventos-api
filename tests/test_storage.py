# -*- coding: utf-8 -*-
"""Testes do módulo de persistência durável (P2)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import storage as s  # noqa: E402


class TestLocalJSONStore:
    def test_roundtrip(self, tmp_path):
        st = s.LocalJSONStore(str(tmp_path))
        assert st.get('k') is None
        assert st.set('k', {'a': 1, 'b': [1, 2, 3]}) is True
        assert st.get('k') == {'a': 1, 'b': [1, 2, 3]}
        assert st.durable is False

    def test_overwrite_e_delete(self, tmp_path):
        st = s.LocalJSONStore(str(tmp_path))
        st.set('k', {'v': 1})
        st.set('k', {'v': 2})
        assert st.get('k') == {'v': 2}
        assert st.delete('k') is True
        assert st.get('k') is None
        # deletar inexistente é idempotente (True)
        assert st.delete('k') is True

    def test_chave_sanitizada(self, tmp_path):
        st = s.LocalJSONStore(str(tmp_path))
        st.set('venues_12/34', {'x': 1})           # barra não vira subpasta
        assert st.get('venues_12/34') == {'x': 1}
        assert any(f.endswith('.json') for f in os.listdir(tmp_path))

    def test_valor_invalido_nao_quebra(self, tmp_path):
        st = s.LocalJSONStore(str(tmp_path))
        assert st.set('k', {'bad': {1, 2, 3}}) is False   # set não serializa
        assert st.get('k') is None


class TestSupabaseStore:
    """Sem banco real: valida construção e robustez (não lança em falha)."""
    def test_endpoint_e_headers(self):
        st = s.SupabaseStore('https://x.supabase.co/', 'KEY', 'app_kv')
        assert st.endpoint == 'https://x.supabase.co/rest/v1/app_kv'
        assert st._headers['apikey'] == 'KEY'
        assert st._headers['Authorization'] == 'Bearer KEY'
        assert st.durable is True

    def test_falha_de_rede_nao_lanca(self):
        # URL inexistente → get/set/delete retornam None/False sem exceção.
        st = s.SupabaseStore('https://nao-existe.invalid', 'KEY', timeout=0.01)
        assert st.get('k') is None
        assert st.set('k', {'a': 1}) is False
        assert st.delete('k') is False
