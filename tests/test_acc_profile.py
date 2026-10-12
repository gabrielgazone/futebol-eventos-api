# -*- coding: utf-8 -*-
"""Testes do perfil de aceleração por zona de velocidade de entrada (ES1-ES4)."""
import pytest

import acc_profile as ap


# ── Classificação por zona ────────────────────────────────────────────────────

def test_zona_de_entrada_fronteiras():
    _vm = 40.0          # ES1 <10 | ES2 10-20 | ES3 20-30 | ES4 >=30
    assert ap.zona_de_entrada(0.0,  _vm) == 'ES1'
    assert ap.zona_de_entrada(9.9,  _vm) == 'ES1'
    assert ap.zona_de_entrada(10.0, _vm) == 'ES2'     # fronteira fecha à esquerda
    assert ap.zona_de_entrada(19.9, _vm) == 'ES2'
    assert ap.zona_de_entrada(20.0, _vm) == 'ES3'
    assert ap.zona_de_entrada(29.9, _vm) == 'ES3'
    assert ap.zona_de_entrada(30.0, _vm) == 'ES4'
    assert ap.zona_de_entrada(40.0, _vm) == 'ES4'     # na própria Vmax
    assert ap.zona_de_entrada(45.0, _vm) == 'ES4'     # acima da Vmax (não perde)


def test_zona_sem_vmax_valida():
    assert ap.zona_de_entrada(10.0, 0) is None
    assert ap.zona_de_entrada(10.0, None) is None


# ── Detecção de eventos ───────────────────────────────────────────────────────

def _sinal(eventos, n=100, hz=10):
    """Monta (vel_kmh, acc) com eventos em (frame_ini, n_frames, acc, vel)."""
    _acc = [0.0] * n
    _vel = [0.0] * n
    for _ini, _dur, _a, _v in eventos:
        for _k in range(_ini, _ini + _dur):
            _acc[_k] = _a
            _vel[_k] = _v
    return _vel, _acc


def test_detectar_eventos_respeita_duracao_minima():
    # 10 frames (1,0 s) passa; 2 frames (0,2 s) não, com min_dur_s=0.4
    _vel, _acc = _sinal([(10, 10, 3.0, 5.0), (40, 2, 3.0, 5.0)])
    _ev = ap.detectar_eventos(_vel, _acc, hz=10, min_dur_s=0.4)
    assert len(_ev) == 1
    assert _ev[0]['frame_ini'] == 10
    assert _ev[0]['dur_s'] == 1.0
    assert _ev[0]['pico'] == 3.0


def test_detectar_eventos_guarda_velocidade_de_entrada():
    _vel, _acc = _sinal([(20, 6, 2.5, 17.3)])
    _ev = ap.detectar_eventos(_vel, _acc, hz=10, min_dur_s=0.4)
    assert len(_ev) == 1
    assert _ev[0]['vel_entrada'] == 17.3


def test_detectar_eventos_pico_e_o_maximo_do_trecho():
    _acc = [0.0] * 30
    for _k, _v in zip(range(10, 16), [1.0, 2.0, 4.5, 3.0, 2.0, 1.0]):
        _acc[_k] = _v
    _vel = [8.0] * 30
    _ev = ap.detectar_eventos(_vel, _acc, hz=10, min_dur_s=0.4)
    assert len(_ev) == 1
    assert _ev[0]['pico'] == 4.5


def test_detectar_eventos_fecha_evento_aberto_no_fim():
    _vel, _acc = _sinal([(90, 10, 2.0, 5.0)], n=100)
    _ev = ap.detectar_eventos(_vel, _acc, hz=10, min_dur_s=0.4)
    assert len(_ev) == 1 and _ev[0]['frame_ini'] == 90


def test_detectar_eventos_serie_vazia():
    assert ap.detectar_eventos([], [], hz=10) == []


# ── Pipeline completo ─────────────────────────────────────────────────────────

def _pontos(vel_kmh, acc):
    return [{'v': _v / 3.6, 'a': _a, 'ts': _i / 10.0}
            for _i, (_v, _a) in enumerate(zip(vel_kmh, acc))]


def test_perfil_um_evento_por_zona_aplica_os_pesos():
    # Vmax=40 -> ES1<10 | ES2 10-20 | ES3 20-30 | ES4>=30
    _vel, _acc = _sinal([(10, 10, 3.0,  5.0),    # ES1
                         (30, 10, 2.0, 15.0),    # ES2
                         (50, 10, 1.0, 25.0),    # ES3
                         (70, 10, 0.8, 35.0)],   # ES4
                        n=100)
    # piso explícito: este teste exercita os PESOS, não o piso de segmentação
    _p = ap.perfil_atleta(_pontos(_vel, _acc), hz=10, vmax=40.0, min_dur_s=0.4,
                          piso_ms2=0.5)
    _z = _p['zonas']
    assert [_z[_k]['n_total'] for _k in ('ES1', 'ES2', 'ES3', 'ES4')] == [1, 1, 1, 1]
    # evento único: P95 = o próprio pico, corte = 50% dele -> sempre válido
    assert [_z[_k]['n_validos'] for _k in ('ES1', 'ES2', 'ES3', 'ES4')] == [1, 1, 1, 1]
    assert _z['ES1']['score'] == 1.0
    assert _z['ES2']['score'] == 1.5
    assert _z['ES3']['score'] == 2.0
    assert _z['ES4']['score'] == 2.5
    assert _p['total'] == 7.0                      # 1,0+1,5+2,0+2,5
    assert _p['n_eventos'] == 4


def test_corte_p95_descarta_ruido_de_baixo_impacto():
    """Quatro eventos fracos + um forte na MESMA zona: o corte derruba os fracos."""
    _evs = [(10 + _i * 10, 6, 1.0, 5.0) for _i in range(4)]   # picos 1,0
    _evs.append((60, 6, 10.0, 5.0))                            # pico 10,0
    _vel, _acc = _sinal(_evs, n=100)
    _p = ap.perfil_atleta(_pontos(_vel, _acc), hz=10, vmax=40.0, min_dur_s=0.4,
                          piso_ms2=0.5)
    _z1 = _p['zonas']['ES1']
    assert _z1['n_total'] == 5
    # P95 de [1,1,1,1,10] (interpolação linear) = 8,2 -> corte = 4,1
    assert _z1['p95'] == pytest.approx(8.2, abs=1e-6)
    assert _z1['corte'] == pytest.approx(4.1, abs=1e-6)
    assert _z1['n_validos'] == 1                   # só o de pico 10,0
    assert _z1['score'] == 1.0


def test_total_e_a_soma_dos_scores_das_zonas():
    _vel, _acc = _sinal([(10, 8, 4.0, 3.0), (30, 8, 3.0, 12.0),
                         (50, 8, 2.0, 22.0), (70, 8, 1.5, 33.0)], n=100)
    _p = ap.perfil_atleta(_pontos(_vel, _acc), hz=10, vmax=40.0, min_dur_s=0.4,
                          piso_ms2=0.5)
    _soma = sum(_p['zonas'][_z[0]]['score'] for _z in ap.ES_ZONAS)
    assert _p['total'] == pytest.approx(_soma)


def test_pct_corte_mais_alto_reduz_validos():
    _evs = [(10 + _i * 10, 6, 1.0 + _i, 5.0) for _i in range(5)]  # picos 1..5
    _vel, _acc = _sinal(_evs, n=100)
    _base = ap.perfil_atleta(_pontos(_vel, _acc), hz=10, vmax=40.0,
                             min_dur_s=0.4, piso_ms2=0.5, pct_corte=0.10)
    _duro = ap.perfil_atleta(_pontos(_vel, _acc), hz=10, vmax=40.0,
                             min_dur_s=0.4, piso_ms2=0.5, pct_corte=0.95)
    assert _base['zonas']['ES1']['n_validos'] >= _duro['zonas']['ES1']['n_validos']
    assert _duro['zonas']['ES1']['n_validos'] >= 1      # o pico nunca é cortado


def test_marca_zona_pouco_confiavel():
    _vel, _acc = _sinal([(10, 8, 3.0, 5.0)], n=100)
    _p = ap.perfil_atleta(_pontos(_vel, _acc), hz=10, vmax=40.0, min_dur_s=0.4)
    assert _p['zonas']['ES1']['confiavel'] is False      # 1 evento < MIN_EVENTOS_P95
    assert _p['zonas']['ES4']['n_total'] == 0


def test_zona_vazia_nao_quebra_e_pontua_zero():
    _vel, _acc = _sinal([(10, 8, 3.0, 5.0)], n=100)
    _p = ap.perfil_atleta(_pontos(_vel, _acc), hz=10, vmax=40.0, min_dur_s=0.4)
    for _k in ('ES2', 'ES3', 'ES4'):
        assert _p['zonas'][_k]['score'] == 0.0
        assert _p['zonas'][_k]['p95'] == 0.0


# ── Vmax ──────────────────────────────────────────────────────────────────────

def test_vmax_do_sinal_e_percentil_protege_de_spike():
    _v = [10.0] * 999 + [90.0]              # um spike absurdo de GPS
    assert ap.vmax_kmh(_v, 100.0) == 90.0
    assert ap.vmax_kmh(_v, 99.0) < 20.0     # percentil ignora o spike


def test_vmax_derivada_do_proprio_sinal_quando_nao_informada():
    _vel, _acc = _sinal([(10, 8, 3.0, 20.0)], n=100)
    _vel[95] = 36.0                          # pico de velocidade do jogo
    _p = ap.perfil_atleta(_pontos(_vel, _acc), hz=10, min_dur_s=0.4)
    assert _p['vmax'] == pytest.approx(36.0)
    # entrada 20 km/h com Vmax 36 -> 55,6% -> ES3
    assert _p['zonas']['ES3']['n_total'] == 1


# ── Fallback dv/dt ────────────────────────────────────────────────────────────

def test_fallback_derivada_quando_sensor_nao_traz_aceleracao():
    # rampa sustentada de velocidade, campo 'a' zerado
    _vel = [0.0] * 10 + [min(2.0 * _i, 30.0) for _i in range(40)] + [30.0] * 10
    _pts = [{'v': _v / 3.6, 'a': 0.0, 'ts': _i / 10.0}
            for _i, _v in enumerate(_vel)]
    _acc = ap.acc_de_pontos(_pts, hz=10.0)
    assert any(abs(_a) > 0.5 for _a in _acc), "dv/dt deveria gerar aceleração"


# ── Saídas tabulares ──────────────────────────────────────────────────────────

def test_linhas_tabela_uma_linha_por_atleta_x_zona():
    _vel, _acc = _sinal([(10, 8, 3.0, 5.0)], n=100)
    _perfis = {'A': ap.perfil_atleta(_pontos(_vel, _acc), hz=10, vmax=40.0,
                                     min_dur_s=0.4)}
    _rows = ap.linhas_tabela(_perfis)
    assert len(_rows) == 4
    assert {_r['Zona'] for _r in _rows} == {'ES1', 'ES2', 'ES3', 'ES4'}


def test_linhas_resumo_ordena_por_carga_decrescente():
    _v1, _a1 = _sinal([(10, 8, 3.0, 35.0)], n=100)      # ES4 -> peso 2,5
    _v2, _a2 = _sinal([(10, 8, 3.0, 5.0)], n=100)       # ES1 -> peso 1,0
    _perfis = {
        'Fraco': ap.perfil_atleta(_pontos(_v2, _a2), hz=10, vmax=40.0, min_dur_s=0.4),
        'Forte': ap.perfil_atleta(_pontos(_v1, _a1), hz=10, vmax=40.0, min_dur_s=0.4),
    }
    _rows = ap.linhas_resumo(_perfis)
    assert _rows[0]['Atleta'] == 'Forte'
    assert _rows[0]['Carga total ponderada'] == 2.5
    assert _rows[1]['Carga total ponderada'] == 1.0


def test_padroes_calibrados_nao_mudam_em_silencio():
    """O piso de segmentação é o parâmetro mais sensível da análise: com piso
    baixo um jogo gera milhares de 'eventos'. Travado por teste de propósito."""
    assert ap.DEFAULT_PISO_MS2 == 1.50
    assert ap.DEFAULT_MIN_DUR_S == 0.50
    assert ap.DEFAULT_PERCENTIL == 95.0
    assert ap.DEFAULT_PCT_CORTE == 0.50
    assert [(_z[0], _z[3]) for _z in ap.ES_ZONAS] == [
        ('ES1', 1.0), ('ES2', 1.5), ('ES3', 2.0), ('ES4', 2.5)]


def test_piso_mais_alto_reduz_contagem_de_eventos():
    """Garante a direção do efeito do piso (o achado da calibração)."""
    _evs = [(10 + _i * 8, 5, 0.8 + 0.4 * _i, 5.0) for _i in range(10)]
    _vel, _acc = _sinal(_evs, n=120)
    _baixo = ap.detectar_eventos(_vel, _acc, hz=10, piso_ms2=0.5, min_dur_s=0.4)
    _alto = ap.detectar_eventos(_vel, _acc, hz=10, piso_ms2=2.5, min_dur_s=0.4)
    assert len(_baixo) > len(_alto)


def test_perfil_equipe_cobre_todos_os_atletas():
    _vel, _acc = _sinal([(10, 8, 3.0, 5.0)], n=100)
    _ds = {'A': _pontos(_vel, _acc), 'B': _pontos(_vel, _acc), 'C': []}
    _perfis = ap.perfil_equipe(_ds, hz=10, min_dur_s=0.4, vmax_por_atleta={'A': 40.0, 'B': 40.0})
    assert set(_perfis.keys()) == {'A', 'B'}          # 'C' sem dados é omitido
    assert _perfis['A']['total'] == 1.0
