# Persistência durável (P2)

O Streamlit Cloud roda em containers com **sistema de arquivos efêmero**: tudo
que é gravado em disco (venues, bandas do usuário) **desaparece a cada
redeploy/reboot**. Para tornar essas configurações duráveis, o app usa um
armazenamento chave→valor externo (Supabase) quando configurado — senão, cai no
armazenamento local (temporário) e exibe um aviso.

## Como ativar (Supabase, gratuito — ~3 minutos)

1. Crie um projeto em <https://supabase.com> (plano free).
2. No **SQL Editor** do projeto, rode:

   ```sql
   create table if not exists app_kv (
     key        text primary key,
     value      jsonb,
     updated_at timestamptz default now()
   );
   ```

3. Em **Project Settings → API**, copie:
   - **Project URL** (ex.: `https://xxxx.supabase.co`)
   - Uma **API key**. Como a chave fica **apenas no servidor do app**
     (`st.secrets`, nunca no navegador), pode usar a `service_role`. Se preferir
     a `anon`, crie uma policy de RLS que permita `select/insert/update/delete`
     na tabela `app_kv`.

4. No Streamlit Cloud, em **Manage app → Settings → Secrets**, adicione:

   ```toml
   [supabase]
   url = "https://xxxx.supabase.co"
   key = "SUA_API_KEY"
   # table = "app_kv"   # opcional (padrão: app_kv)
   ```

5. Salve e reinicie o app. O aviso de "armazenamento temporário" some — as
   configurações passam a sobreviver a redeploys, compartilhadas por
   organização (mesmo clube).

## Como funciona

- Código: [`storage.py`](storage.py) — `LocalJSONStore` (fallback) e
  `SupabaseStore` (durável, via API REST/PostgREST usando `requests`; sem driver
  de banco, então não há risco de wheel em Python novo).
- As chaves têm escopo por organização: `venues_{org}`, `bandas_usuario_{org}`.
- Sem `st.secrets['supabase']`, o app funciona igual a antes (local), apenas com
  o aviso de não-durabilidade.
