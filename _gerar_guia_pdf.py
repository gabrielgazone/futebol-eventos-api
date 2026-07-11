# -*- coding: utf-8 -*-
"""Gera o Guia do Usuário (PDF) em PT, ES e EN."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ── Paleta ──────────────────────────────────────────────────────────────────
VERDE     = colors.HexColor('#1b5e20')
VERDE_CLR = colors.HexColor('#2e7d32')
AZUL      = colors.HexColor('#1565c0')
AZUL_SOFT = colors.HexColor('#e3f2fd')
LARANJA   = colors.HexColor('#e65100')
CINZA     = colors.HexColor('#444444')
CINZA_CLR = colors.HexColor('#777777')
LINHA     = colors.HexColor('#cfd8cf')
EMAIL     = "gabriel.gazone@catapultsports.com"

ss = getSampleStyleSheet()
st_title = ParagraphStyle('t', parent=ss['Title'], fontName='Helvetica-Bold',
                          fontSize=22, textColor=colors.white, leading=25, spaceAfter=2)
st_sub = ParagraphStyle('s', parent=ss['Normal'], fontName='Helvetica',
                        fontSize=10.5, textColor=colors.HexColor('#dcedc8'), leading=13)
st_h2 = ParagraphStyle('h2', parent=ss['Heading2'], fontName='Helvetica-Bold',
                       fontSize=13, textColor=VERDE, spaceBefore=10, spaceAfter=4, leading=15)
st_h3 = ParagraphStyle('h3', parent=ss['Heading3'], fontName='Helvetica-Bold',
                       fontSize=10.5, textColor=VERDE_CLR, spaceBefore=4, spaceAfter=1, leading=13)
st_body = ParagraphStyle('b', parent=ss['Normal'], fontName='Helvetica',
                         fontSize=9.5, textColor=CINZA, leading=13.5, alignment=TA_JUSTIFY)
st_li = ParagraphStyle('li', parent=st_body, leftIndent=10, spaceAfter=2)
st_card_h = ParagraphStyle('ch', parent=ss['Normal'], fontName='Helvetica-Bold',
                           fontSize=10, textColor=colors.white, leading=12)
st_card_b = ParagraphStyle('cb', parent=ss['Normal'], fontName='Helvetica',
                           fontSize=8.7, textColor=CINZA, leading=11.5)
st_tip_h = ParagraphStyle('th', parent=ss['Normal'], fontName='Helvetica-Bold',
                          fontSize=9.8, textColor=LARANJA, leading=12, spaceAfter=2)
st_step_n = ParagraphStyle('sn', parent=ss['Normal'], fontName='Helvetica-Bold',
                           fontSize=14, textColor=colors.white, alignment=1, leading=16)
st_cell_h = ParagraphStyle('clh', parent=ss['Normal'], fontName='Helvetica-Bold',
                           fontSize=8.8, textColor=colors.white, leading=11)
st_cell = ParagraphStyle('cl', parent=ss['Normal'], fontName='Helvetica',
                         fontSize=8.6, textColor=CINZA, leading=11)
st_cell_b = ParagraphStyle('clb', parent=ss['Normal'], fontName='Helvetica-Bold',
                           fontSize=8.6, textColor=VERDE, leading=11)
st_q = ParagraphStyle('q', parent=ss['Normal'], fontName='Helvetica-Bold',
                      fontSize=9.3, textColor=AZUL, leading=12, spaceBefore=5, spaceAfter=1)
st_a = ParagraphStyle('a', parent=ss['Normal'], fontName='Helvetica',
                      fontSize=9.2, textColor=CINZA, leading=12.5, alignment=TA_JUSTIFY)
st_mail = ParagraphStyle('m', parent=ss['Normal'], fontName='Helvetica-Bold',
                         fontSize=12, textColor=AZUL, leading=15, alignment=1)


def bullet(txt, style=st_li):
    return Paragraph(f'<font color="#2e7d32">&bull;</font>&nbsp; {txt}', style)


def faixa_titulo(S):
    cab = Table(
        [[Paragraph(S['title'], st_title)],
         [Paragraph(S['subtitle'], st_sub)]], colWidths=[170 * mm])
    cab.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), VERDE),
        ('LEFTPADDING', (0, 0), (-1, -1), 14), ('RIGHTPADDING', (0, 0), (-1, -1), 14),
        ('TOPPADDING', (0, 0), (0, 0), 14), ('BOTTOMPADDING', (0, 0), (0, 0), 1),
        ('TOPPADDING', (0, 1), (0, 1), 0), ('BOTTOMPADDING', (0, 1), (0, 1), 14),
        ('ROUNDEDCORNERS', [6, 6, 6, 6]),
    ]))
    return cab


def card(titulo, cor, linhas, w=88):
    head = Table([[Paragraph(titulo, st_card_h)]], colWidths=[(w - 6) * mm])
    head.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), cor),
        ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    items = [Paragraph(f'<font color="#2e7d32">&bull;</font>&nbsp; {l}', st_card_b) for l in linhas]
    box = Table([[[head, Spacer(1, 4)] + items]], colWidths=[(w - 2) * mm])
    box.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.6, LINHA),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('BACKGROUND', (0, 0), (-1, -1), colors.white),
    ]))
    return box


def passo(n, titulo, texto, cor=VERDE_CLR):
    num = Table([[Paragraph(str(n), st_step_n)]], colWidths=[10 * mm], rowHeights=[10 * mm])
    num.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), cor),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ROUNDEDCORNERS', [5, 5, 5, 5]),
    ]))
    txt = [Paragraph(f'<b>{titulo}</b>', st_h3), Paragraph(texto, st_card_b)]
    row = Table([[num, txt]], colWidths=[13 * mm, 157 * mm])
    row.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0), ('TOPPADDING', (0, 0), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    return row


def tabela_controles(header, linhas):
    data = [[Paragraph(header[0], st_cell_h), Paragraph(header[1], st_cell_h)]]
    for ctrl, desc in linhas:
        data.append([Paragraph(ctrl, st_cell_b), Paragraph(desc, st_cell)])
    t = Table(data, colWidths=[42 * mm, 134 * mm])
    sty = [
        ('BACKGROUND', (0, 0), (-1, 0), AZUL),
        ('LINEBELOW', (0, 0), (-1, -1), 0.4, LINHA),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('BOX', (0, 0), (-1, -1), 0.5, LINHA),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            sty.append(('BACKGROUND', (0, r), (-1, r), colors.HexColor('#f5f8f5')))
    t.setStyle(TableStyle(sty))
    return t


def faq(q, a):
    return [Paragraph(f"P:&nbsp; {q}", st_q),
            Paragraph(f'<font color="#2e7d32"><b>R:</b></font>&nbsp; {a}', st_a)]


def caixa(cor_fundo, cor_borda, flowables):
    box = Table([[flowables]], colWidths=[176 * mm])
    box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), cor_fundo),
        ('BOX', (0, 0), (-1, -1), 0.8, cor_borda),
        ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return box


def build_story(S):
    story = []
    story.append(faixa_titulo(S))
    story.append(Spacer(1, 10))
    story.append(Paragraph(S['intro'], st_body))

    story.append(Paragraph(S['h_start'], st_h2))
    story.append(HRFlowable(width="100%", thickness=1, color=LINHA, spaceAfter=6))
    for i, (t, txt) in enumerate(S['steps'], 1):
        story.append(passo(i, t, txt))

    story.append(Paragraph(S['h_modules'], st_h2))
    story.append(HRFlowable(width="100%", thickness=1, color=LINHA, spaceAfter=6))
    cores = [VERDE_CLR, AZUL, LARANJA, VERDE]
    cards = [card(S['mod_titles'][i], cores[i], S['mod_lines'][i]) for i in range(4)]
    grid = Table([[cards[0], cards[1]], [cards[2], cards[3]]], colWidths=[88 * mm, 88 * mm])
    grid.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (0, 0), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(grid)

    # Marcando o campo
    story.append(PageBreak())
    story.append(Paragraph(S['h_field'], st_h2))
    story.append(HRFlowable(width="100%", thickness=1, color=LINHA, spaceAfter=6))
    story.append(Paragraph(S['field_intro'], st_body))
    story.append(Paragraph(S['h_where'], st_h3))
    story.append(Paragraph(S['where_text'], st_body))
    story.append(Paragraph(S['h_steps'], st_h3))
    for i, (t, txt) in enumerate(S['field_steps'], 1):
        story.append(passo(i, t, txt, AZUL))
    story.append(Spacer(1, 4))
    story.append(tabela_controles(S['ctrl_header'], S['ctrl_rows']))
    story.append(Paragraph(S['h_save'], st_h3))
    story.append(Paragraph(S['save_text'], st_body))
    for b in S['save_bullets']:
        story.append(bullet(b))
    story.append(Spacer(1, 6))
    wm = [Paragraph(S['wellmarked_title'], ParagraphStyle('qh', parent=st_tip_h, textColor=AZUL))]
    wm += [bullet(b, st_card_b) for b in S['wellmarked']]
    story.append(caixa(AZUL_SOFT, AZUL, wm))

    # Tática Coletiva
    story.append(PageBreak())
    story.append(Paragraph(S['h_tatica'], st_h2))
    story.append(HRFlowable(width="100%", thickness=1, color=LINHA, spaceAfter=6))
    story.append(Paragraph(S['tatica_intro'], st_body))
    for nome, desc in S['views']:
        story.append(Paragraph(nome, st_h3))
        story.append(bullet(desc))
    story.append(Paragraph(S['h_anim'], st_h3))
    for b in S['anim']:
        story.append(bullet(b))
    story.append(Spacer(1, 6))
    ts = [Paragraph(S['twosliders_title'], st_tip_h)]
    ts += [bullet(b, st_card_b) for b in S['twosliders']]
    story.append(caixa(colors.HexColor('#e8f5e9'), VERDE_CLR, ts))

    # Solução de problemas + dicas + contato
    story.append(PageBreak())
    story.append(Paragraph(S['h_trouble'], st_h2))
    story.append(HRFlowable(width="100%", thickness=1, color=LINHA, spaceAfter=4))
    for q, a in S['faqs']:
        story.extend(faq(q, a))
    story.append(Spacer(1, 8))
    tips = [Paragraph(S['h_tips'], st_tip_h)]
    tips += [bullet(b, st_card_b) for b in S['tips']]
    story.append(caixa(colors.HexColor('#fff3e0'), LARANJA, tips))

    story.append(Spacer(1, 10))
    contato = [
        Paragraph(S['h_contact'], ParagraphStyle('cth', parent=st_h2, spaceBefore=0, spaceAfter=4)),
        Paragraph(S['contact_text'], st_body),
        Spacer(1, 4),
        Paragraph(f'<a href="mailto:{EMAIL}"><font color="#1565c0">{EMAIL}</font></a>', st_mail),
    ]
    story.append(caixa(colors.HexColor('#eef4fb'), AZUL, contato))
    return story


def gerar(S, arquivo):
    rodape = S['footer']

    def _decoracao(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINHA); canvas.setLineWidth(0.6)
        canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
        canvas.setFont('Helvetica', 7.5); canvas.setFillColor(CINZA_CLR)
        canvas.drawString(18 * mm, 9.5 * mm, rodape)
        canvas.drawRightString(192 * mm, 9.5 * mm, f"{S['page']} {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(arquivo, pagesize=A4,
                          leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=15 * mm, bottomMargin=18 * mm,
                          title=S['title'], author="Gabriel Gazone dos Santos")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='main')
    doc.addPageTemplates([PageTemplate(id='all', frames=[frame], onPage=_decoracao)])
    doc.build(build_story(S))
    print("OK:", arquivo)


# ══════════════════════════════════════════════════════════════════════════
# TEXTOS POR IDIOMA
# ══════════════════════════════════════════════════════════════════════════
PT = {
    'page': 'Pagina', 'footer': 'Desenvolvido por Gabriel Gazone dos Santos',
    'title': 'Guia do Usuário',
    'subtitle': 'Plataforma de Análise de Desempenho no Futebol &nbsp;|&nbsp; '
                'GPS, carga física e inteligência tática',
    'intro': "Esta ferramenta transforma os dados de <b>GPS e sensores inerciais</b> dos atletas "
        "(coletados via Catapult Connect) em <b>visões claras e acionáveis</b> sobre desempenho físico "
        "e comportamento tático. Em poucos cliques você sai de planilhas cruas para <b>mapas de calor, "
        "perfis de carga, cenários de pior caso (WCS) e visualizações táticas coletivas</b> — incluindo "
        "um replay 3D do jogo. O objetivo: dar a comissões técnicas e departamentos de performance uma "
        "leitura <b>rápida, visual e cientificamente embasada</b> de cada atleta e do time como sistema.",
    'h_start': 'Começando em 4 passos',
    'steps': [
        ("Conecte-se", "Na barra lateral, informe seu <b>Token</b> da Connect e carregue os atletas e as atividades."),
        ("Escolha a atividade", "Selecione a <b>atividade</b> (treino ou jogo) que deseja analisar na lista carregada."),
        ("Selecione período(s) e atletas", "Marque um ou mais <b>períodos</b> (1º tempo, 2º tempo, etc.) e os "
         "<b>atletas</b> de interesse. Com vários períodos, a ferramenta cria automaticamente uma visão combinada."),
        ("Explore as 5 abas", "Os dados aparecem em cinco módulos. Cada aba traz gráficos interativos "
         "(passe o mouse para detalhes, arraste para dar zoom) e botões de exportação."),
    ],
    'h_modules': 'Os 5 módulos da plataforma',
    'mod_titles': ["RESUMO", "CAMPO & GPS", "CARGA FÍSICA", "TÁTICA COLETIVA"],
    'mod_lines': [
        ["<b>Visão Geral</b>: distância total, PlayerLoad, velocidade máxima e indicadores da sessão.",
         "<b>Por Posição</b>: compara atletas por função tática (zagueiro, meia, atacante...)."],
        ["<b>Campo de Futebol</b>: mapas de calor de ocupação e trajetória — é aqui que se marca o campo.",
         "<b>História do Jogo</b>: heatmaps por fase, mostrando a evolução ao longo do tempo.",
         "<b>WCS</b>: os trechos de maior exigência da partida por janela móvel."],
        ["<b>Esforços</b>: acelerações, desacelerações e sprints, com zona e direção.",
         "<b>Janelas Temporais</b>: intensidade em janela móvel e picos de alta/média-alta.",
         "<b>Neuromuscular, Acc-Vel e FC</b>: fadiga, perfil acc×vel e carga cardíaca (TRIMP + zonas)."],
        ["O time como <b>sistema</b>: cruza a posição de todos os atletas no mesmo instante.",
         "Quatro visões: <b>Pitch Control</b>, <b>Respiração</b>, <b>Voronoi</b> e <b>Replay 3D</b>.",
         "Funciona até com dispositivos <b>só-GPS</b> (reconstrói as coordenadas)."],
    ],
    'h_field': 'Marcando o campo: habilitando os dados de localização',
    'field_intro': "O GPS bruto é apenas uma nuvem de pontos de <b>latitude/longitude</b>. Para que o app "
        "entenda <i>onde</i> no gramado cada ponto está — e assim gere ocupação por zonas, terços do campo, "
        "esforços com direção e toda a <b>Tática Coletiva</b> — é preciso dizer a ele <b>onde fica o campo e "
        "como ele está orientado</b>. Isso se chama <b>marcar (ou calibrar) o campo</b>, e é feito uma única "
        "vez por local, sobre uma imagem de satélite.",
    'h_where': 'Onde fica',
    'where_text': "Aba <b>Campo &amp; GPS &gt; Campo de Futebol</b>. Selecione um <b>atleta de referência que "
        "tenha GPS</b> na sessão (o campo é do local, não do atleta — qualquer um com sinal serve). Se a sessão "
        "tiver pontos GPS, aparece um <b>mapa de satélite</b> com a nuvem de pontos e um <b>alvo amarelo</b> "
        "marcando o centro do campo.",
    'h_steps': 'Passo a passo da marcação',
    'field_steps': [
        ("Mostre o campo", "Clique em <b>Mostrar Campo</b> para sobrepor o desenho do campo (linhas brancas) sobre a imagem de satélite."),
        ("Centralize", "Arraste o <b>alvo amarelo</b> — ou edite <b>Lat/Lon</b> — até o desenho cobrir a nuvem de pontos GPS, sobre o gramado real na imagem."),
        ("Gire (rotação)", "Use os botões de <b>Rot</b> (« =-10°, ‹ =-1°, › =+1°, » =+10°) até as <b>linhas do desenho coincidirem com as linhas reais</b> do campo no satélite. Este é o passo mais importante para a fidelidade tática."),
        ("Ajuste as dimensões", "Acerte <b>Compr.</b> (comprimento, 90–120 m) e <b>Larg.</b> (largura, 55–90 m) para casar com o tamanho real. O padrão é 105×68 m. <b>Marg.</b> adiciona uma folga fora das linhas, se quiser."),
        ("Aplique", "Clique em <b>Aplicar Campo</b>. A partir daí, as coordenadas de localização passam a alimentar os mapas de calor, os esforços com direção e toda a Tática Coletiva."),
    ],
    'ctrl_header': ("Controle no painel do mapa", "O que faz"),
    'ctrl_rows': [
        ("Mostrar Campo", "Liga/desliga o desenho do campo sobre o satélite."),
        ("Lat / Lon", "Move o centro do campo. Arraste o alvo amarelo ou edite os números."),
        ("Rot &nbsp;« ‹ › »", "Gira o campo: « -10°, ‹ -1°, › +1°, » +10°. Alinhe as linhas do desenho às do gramado."),
        ("Compr. / Larg.", "Comprimento (90–120 m) e largura (55–90 m). Padrão 105×68 m."),
        ("Marg.", "Margem extra desenhada fora das linhas (0–15 m)."),
        ("Aplicar Campo", "Confirma a configuração e carrega as coordenadas de campo."),
    ],
    'h_save': 'Salvar para todos (banco de venues)',
    'save_text': "Logo após aplicar, o app oferece <b>salvar a configuração no banco compartilhado de venues</b>, "
        "com um nome (ex.: &ldquo;Estádio Municipal&rdquo;). Vale muito a pena: na próxima vez que <b>qualquer "
        "usuário</b> abrir uma atividade <b>neste mesmo local</b>, o campo já vem <b>configurado "
        "automaticamente</b> — sem repetir a marcação. Você verá o aviso &ldquo;Campo carregado do banco compartilhado&rdquo;.",
    'save_bullets': [
        "<b>Reajustar</b>: refaz a marcação do zero (volta ao mapa de satélite).",
        "<b>Venues</b>: abre o gerenciador para listar, atualizar ou excluir locais salvos.",
    ],
    'wellmarked_title': 'Como saber que ficou bem marcado',
    'wellmarked': [
        "As <b>linhas brancas do desenho</b> coincidem com as linhas do campo na imagem de satélite.",
        "A <b>nuvem de pontos GPS</b> fica contida dentro do campo, sem &ldquo;vazar&rdquo; muito para fora.",
        "Os <b>gols</b> ficam nas pontas certas e o meio-campo cruza o círculo central real.",
        "Dica: comece pela <b>rotação</b>, depois ajuste o <b>centro</b> e por fim as <b>dimensões</b>.",
    ],
    'h_tatica': 'Em destaque: a aba Tática Coletiva',
    'tatica_intro': "É o módulo mais inovador. Sincronizando a posição de todos os atletas selecionados, ele "
        "revela o comportamento do time como um todo — algo impossível de ver olhando atleta por atleta. Exige "
        "<b>pelo menos 2 atletas com posição</b> no mesmo período. As quatro visões:",
    'views': [
        ("Pitch Control", "Modelo de William Spearman (Liverpool FC). Colore o campo pelo <b>domínio de espaço</b> "
         "— quem chegaria primeiro em cada ponto, considerando velocidade e direção."),
        ("Respiração da equipe", "Mostra o bloco <b>comprimindo</b> na marcação e <b>expandindo</b> na posse, com "
         "largura, comprimento, área e dispersão ao longo do tempo."),
        ("Voronoi", "O &ldquo;vitral&rdquo; tático: cada região do campo pertence ao jogador mais próximo, expondo "
         "<b>cobertura de espaço e buracos</b> na estrutura."),
        ("Replay 3D", "Um <b>broadcast sintético</b> navegável (gire a câmera, dê zoom) com gramado listrado, traves "
         "e marcações — reconstruído apenas das coordenadas."),
    ],
    'h_anim': 'Controles da animação',
    'anim': [
        "<b>Janela de análise</b>: de 30 s a 10 min, duração personalizada ou o <b>período inteiro</b>. Janelas "
        "curtas mostram o deslocamento contínuo (tempo real); o período inteiro dá a visão macro de toda a "
        "partida (com frames mais espaçados).",
        "<b>Início da janela</b>: arraste para posicionar a análise em <b>qualquer momento</b> do jogo (inclusive os minutos finais).",
        "<b>Velocidade</b>: <b>1× = tempo real</b> do jogo; abaixo é câmera lenta, acima é acelerado. Use <b>Play/Pause</b> e o slider para navegar pelos frames.",
    ],
    'twosliders_title': 'Dois controles de tempo — não confunda',
    'twosliders': [
        "O slider <b>Início da janela</b> (acima do gráfico) <b>move a janela</b> pela partida toda.",
        "O slider <b>dentro do gráfico</b> percorre apenas os <b>frames da janela atual</b>.",
    ],
    'h_trouble': 'Solução de problemas (dificuldades comuns)',
    'faqs': [
        ("Não aparece o mapa / &ldquo;Nenhum ponto GPS real encontrado&rdquo;.",
         "O sensor daquele atleta não obteve <i>lock</i> de GPS na sessão. Como o campo é do local, escolha <b>outro "
         "atleta de referência</b> que tenha sinal GPS para fazer a marcação."),
        ("O campo fica torto ou desalinhado sobre o satélite.",
         "Ajuste primeiro a <b>rotação</b> (« ‹ › ») e depois o <b>centro</b> (Lat/Lon) até as linhas do desenho "
         "baterem com o gramado. Confira também <b>Compr./Larg.</b> — dimensões erradas distorcem tudo."),
        ("Na Tática Coletiva aparece &ldquo;coordenadas reconstruídas do GPS / alinhamento aproximado&rdquo;.",
         "Seu dispositivo é <b>só-GPS</b> (sem x/y nativo). As <b>posições relativas</b> entre jogadores são fiéis; "
         "para um alinhamento absoluto melhor, <b>marque o campo</b> em Campo &amp; GPS — ele passa a ser usado automaticamente aqui."),
        ("&ldquo;Esta aba precisa de pelo menos 2 atletas com posição no mesmo período&rdquo;.",
         "Selecione, na barra lateral, <b>2 ou mais atletas</b> que tenham GPS naquele período."),
        ("A animação parece &ldquo;fotos&rdquo; com saltos, sem movimento contínuo.",
         "Trechos longos têm frames espaçados. Reduza a <b>Janela de análise</b> para <b>1–2 min</b> e use o "
         "<b>Início da janela</b> para focar o lance. O período inteiro é para a visão macro."),
        ("As distâncias do GPS divergem do esperado.",
         "Verifique se o campo está bem marcado e com as <b>dimensões reais</b> do gramado. O app também compara a "
         "distância filtrada pelo campo com o GPS integrado — diferenças grandes indicam marcação a revisar."),
        ("Configurei o campo, mas quero refazer.",
         "Use <b>Reajustar</b> (volta ao mapa) ou <b>Venues</b> para atualizar/excluir o local salvo."),
    ],
    'h_tips': 'Dicas para tirar o máximo',
    'tips': [
        "<b>Salve o venue</b> após marcar o campo: beneficia toda a equipe automaticamente.",
        "<b>Combine períodos</b> para ver a partida inteira numa linha do tempo contínua, sem o intervalo.",
        "<b>Exporte</b> tabelas e eventos em CSV para relatórios e acompanhamento longitudinal.",
        "Para estudar um lance em tempo real, use janela de <b>1–2 min</b> e mire o início no momento certo.",
        "Passe o mouse sobre os gráficos para ver <b>valores exatos</b>; arraste para dar zoom.",
    ],
    'h_contact': 'Adições, sugestões, comentários',
    'contact_text': "Tem ideias para melhorar a ferramenta, encontrou algo a ajustar ou quer sugerir uma nova "
        "funcionalidade? Seu retorno é muito bem-vindo. Escreva para:",
}

ES = {
    'page': 'Pagina', 'footer': 'Desarrollado por Gabriel Gazone dos Santos',
    'title': 'Guía del Usuario',
    'subtitle': 'Plataforma de Análisis de Rendimiento en el Fútbol &nbsp;|&nbsp; '
                'GPS, carga física e inteligencia táctica',
    'intro': "Esta herramienta transforma los datos de <b>GPS y sensores inerciales</b> de los atletas "
        "(recolectados vía Catapult Connect) en <b>visiones claras y accionables</b> sobre el rendimiento físico "
        "y el comportamiento táctico. En pocos clics usted pasa de planillas crudas a <b>mapas de calor, perfiles "
        "de carga, escenarios de peor caso (WCS) y visualizaciones tácticas colectivas</b> — incluyendo una "
        "repetición 3D del partido. El objetivo: dar a los cuerpos técnicos y departamentos de rendimiento una "
        "lectura <b>rápida, visual y científicamente fundamentada</b> de cada atleta y del equipo como sistema.",
    'h_start': 'Empezando en 4 pasos',
    'steps': [
        ("Conéctese", "En la barra lateral, ingrese su <b>Token</b> de Connect y cargue los atletas y las actividades."),
        ("Elija la actividad", "Seleccione la <b>actividad</b> (entrenamiento o partido) que desea analizar en la lista cargada."),
        ("Seleccione período(s) y atletas", "Marque uno o más <b>períodos</b> (1.er tiempo, 2.º tiempo, etc.) y los "
         "<b>atletas</b> de interés. Con varios períodos, la herramienta crea automáticamente una vista combinada."),
        ("Explore las 5 pestañas", "Los datos aparecen en cinco módulos. Cada pestaña trae gráficos interactivos "
         "(pase el ratón para ver detalles, arrastre para hacer zoom) y botones de exportación."),
    ],
    'h_modules': 'Los 5 módulos de la plataforma',
    'mod_titles': ["RESUMEN", "CAMPO & GPS", "CARGA FÍSICA", "TÁCTICA COLECTIVA"],
    'mod_lines': [
        ["<b>Visión General</b>: distancia total, PlayerLoad, velocidad máxima e indicadores de la sesión.",
         "<b>Por Posición</b>: compara atletas por función táctica (defensa, mediocampista, delantero...)."],
        ["<b>Campo de Fútbol</b>: mapas de calor de ocupación y trayectoria — aquí se marca el campo.",
         "<b>Historia del Partido</b>: mapas de calor por fase, mostrando la evolución a lo largo del tiempo.",
         "<b>WCS</b>: los tramos de mayor exigencia del partido por ventana móvil."],
        ["<b>Esfuerzos</b>: aceleraciones, desaceleraciones y sprints, con zona y dirección.",
         "<b>Ventanas Temporales</b>: intensidad en ventana móvil y picos de alta/media-alta.",
         "<b>Neuromuscular, Acc-Vel y FC</b>: fatiga, perfil acc×vel y carga cardíaca (TRIMP + zonas)."],
        ["El equipo como <b>sistema</b>: cruza la posición de todos los atletas en el mismo instante.",
         "Cuatro vistas: <b>Pitch Control</b>, <b>Respiración</b>, <b>Voronoi</b> y <b>Repetición 3D</b>.",
         "Funciona incluso con dispositivos <b>solo-GPS</b> (reconstruye las coordenadas)."],
    ],
    'h_field': 'Marcando el campo: habilitando los datos de ubicación',
    'field_intro': "El GPS crudo es solo una nube de puntos de <b>latitud/longitud</b>. Para que la app entienda "
        "<i>dónde</i> en el césped está cada punto — y así genere ocupación por zonas, tercios del campo, esfuerzos "
        "con dirección y toda la <b>Táctica Colectiva</b> — hay que indicarle <b>dónde está el campo y cómo está "
        "orientado</b>. Esto se llama <b>marcar (o calibrar) el campo</b>, y se hace una sola vez por lugar, sobre "
        "una imagen de satélite.",
    'h_where': 'Dónde está',
    'where_text': "Pestaña <b>Campo &amp; GPS &gt; Campo de Fútbol</b>. Seleccione un <b>atleta de referencia que "
        "tenga GPS</b> en la sesión (el campo es del lugar, no del atleta — cualquiera con señal sirve). Si la "
        "sesión tiene puntos GPS, aparece un <b>mapa de satélite</b> con la nube de puntos y un <b>objetivo "
        "amarillo</b> marcando el centro del campo.",
    'h_steps': 'Paso a paso de la marcación',
    'field_steps': [
        ("Muestre el campo", "Haga clic en <b>Mostrar Campo</b> para superponer el dibujo del campo (líneas blancas) sobre la imagen de satélite."),
        ("Centre", "Arrastre el <b>objetivo amarillo</b> — o edite <b>Lat/Lon</b> — hasta que el dibujo cubra la nube de puntos GPS, sobre el césped real en la imagen."),
        ("Gire (rotación)", "Use los botones de <b>Rot</b> (« =-10°, ‹ =-1°, › =+1°, » =+10°) hasta que las <b>líneas del dibujo coincidan con las líneas reales</b> del campo en el satélite. Este es el paso más importante para la fidelidad táctica."),
        ("Ajuste las dimensiones", "Ajuste <b>Long.</b> (longitud, 90–120 m) y <b>Anch.</b> (ancho, 55–90 m) para coincidir con el tamaño real. El valor por defecto es 105×68 m. <b>Marg.</b> agrega un margen fuera de las líneas, si lo desea."),
        ("Aplique", "Haga clic en <b>Aplicar Campo</b>. A partir de ahí, las coordenadas de ubicación alimentan los mapas de calor, los esfuerzos con dirección y toda la Táctica Colectiva."),
    ],
    'ctrl_header': ("Control en el panel del mapa", "Qué hace"),
    'ctrl_rows': [
        ("Mostrar Campo", "Activa/desactiva el dibujo del campo sobre el satélite."),
        ("Lat / Lon", "Mueve el centro del campo. Arrastre el objetivo amarillo o edite los números."),
        ("Rot &nbsp;« ‹ › »", "Gira el campo: « -10°, ‹ -1°, › +1°, » +10°. Alinee las líneas del dibujo con las del césped."),
        ("Long. / Anch.", "Longitud (90–120 m) y ancho (55–90 m). Por defecto 105×68 m."),
        ("Marg.", "Margen extra dibujado fuera de las líneas (0–15 m)."),
        ("Aplicar Campo", "Confirma la configuración y carga las coordenadas del campo."),
    ],
    'h_save': 'Guardar para todos (banco de sedes)',
    'save_text': "Justo después de aplicar, la app ofrece <b>guardar la configuración en el banco compartido de "
        "sedes (venues)</b>, con un nombre (ej.: &ldquo;Estadio Municipal&rdquo;). Vale mucho la pena: la próxima "
        "vez que <b>cualquier usuario</b> abra una actividad <b>en este mismo lugar</b>, el campo ya viene "
        "<b>configurado automáticamente</b> — sin repetir la marcación. Verá el aviso &ldquo;Campo cargado del banco compartido&rdquo;.",
    'save_bullets': [
        "<b>Reajustar</b>: rehace la marcación desde cero (vuelve al mapa de satélite).",
        "<b>Sedes (Venues)</b>: abre el gestor para listar, actualizar o eliminar lugares guardados.",
    ],
    'wellmarked_title': 'Cómo saber que quedó bien marcado',
    'wellmarked': [
        "Las <b>líneas blancas del dibujo</b> coinciden con las líneas del campo en la imagen de satélite.",
        "La <b>nube de puntos GPS</b> queda contenida dentro del campo, sin &ldquo;desbordar&rdquo; mucho.",
        "Los <b>arcos</b> quedan en los extremos correctos y el medio campo cruza el círculo central real.",
        "Consejo: empiece por la <b>rotación</b>, luego ajuste el <b>centro</b> y por último las <b>dimensiones</b>.",
    ],
    'h_tatica': 'Destacado: la pestaña Táctica Colectiva',
    'tatica_intro': "Es el módulo más innovador. Sincronizando la posición de todos los atletas seleccionados, "
        "revela el comportamiento del equipo como un todo — algo imposible de ver mirando atleta por atleta. "
        "Requiere <b>al menos 2 atletas con posición</b> en el mismo período. Las cuatro vistas:",
    'views': [
        ("Pitch Control", "Modelo de William Spearman (Liverpool FC). Colorea el campo según el <b>dominio del "
         "espacio</b> — quién llegaría primero a cada punto, considerando velocidad y dirección."),
        ("Respiración del equipo", "Muestra el bloque <b>comprimiéndose</b> en la marca y <b>expandiéndose</b> en "
         "la posesión, con ancho, longitud, área y dispersión a lo largo del tiempo."),
        ("Voronoi", "El &ldquo;vitral&rdquo; táctico: cada región del campo pertenece al jugador más cercano, "
         "exponiendo <b>cobertura de espacio y huecos</b> en la estructura."),
        ("Repetición 3D", "Una <b>retransmisión sintética</b> navegable (gire la cámara, haga zoom) con césped a "
         "rayas, arcos y marcas — reconstruida solo a partir de las coordenadas."),
    ],
    'h_anim': 'Controles de la animación',
    'anim': [
        "<b>Ventana de análisis</b>: de 30 s a 10 min, duración personalizada o el <b>período completo</b>. "
        "Ventanas cortas muestran el desplazamiento continuo (tiempo real); el período completo da la vista macro "
        "de todo el partido (con frames más espaciados).",
        "<b>Inicio de la ventana</b>: arrastre para ubicar el análisis en <b>cualquier momento</b> del partido (incluidos los minutos finales).",
        "<b>Velocidad</b>: <b>1× = tiempo real</b> del partido; por debajo es cámara lenta, por encima acelerado. Use <b>Play/Pausa</b> y el slider para navegar por los frames.",
    ],
    'twosliders_title': 'Dos controles de tiempo — no los confunda',
    'twosliders': [
        "El slider <b>Inicio de la ventana</b> (encima del gráfico) <b>mueve la ventana</b> por todo el partido.",
        "El slider <b>dentro del gráfico</b> recorre solo los <b>frames de la ventana actual</b>.",
    ],
    'h_trouble': 'Solución de problemas (dificultades comunes)',
    'faqs': [
        ("No aparece el mapa / &ldquo;Ningún punto GPS real encontrado&rdquo;.",
         "El sensor de ese atleta no obtuvo <i>lock</i> de GPS en la sesión. Como el campo es del lugar, elija "
         "<b>otro atleta de referencia</b> que tenga señal GPS para hacer la marcación."),
        ("El campo queda torcido o desalineado sobre el satélite.",
         "Ajuste primero la <b>rotación</b> (« ‹ › ») y luego el <b>centro</b> (Lat/Lon) hasta que las líneas del "
         "dibujo coincidan con el césped. Verifique también <b>Long./Anch.</b> — dimensiones erróneas lo distorsionan todo."),
        ("En la Táctica Colectiva aparece &ldquo;coordenadas reconstruidas del GPS / alineación aproximada&rdquo;.",
         "Su dispositivo es <b>solo-GPS</b> (sin x/y nativo). Las <b>posiciones relativas</b> entre jugadores son "
         "fieles; para una alineación absoluta mejor, <b>marque el campo</b> en Campo &amp; GPS — se usará automáticamente aquí."),
        ("&ldquo;Esta pestaña necesita al menos 2 atletas con posición en el mismo período&rdquo;.",
         "Seleccione, en la barra lateral, <b>2 o más atletas</b> que tengan GPS en ese período."),
        ("La animación parece &ldquo;fotos&rdquo; con saltos, sin movimiento continuo.",
         "Los tramos largos tienen frames espaciados. Reduzca la <b>Ventana de análisis</b> a <b>1–2 min</b> y use "
         "el <b>Inicio de la ventana</b> para enfocar la jugada. El período completo es para la vista macro."),
        ("Las distancias del GPS difieren de lo esperado.",
         "Verifique que el campo esté bien marcado y con las <b>dimensiones reales</b> del césped. La app también "
         "compara la distancia filtrada por el campo con el GPS integrado — diferencias grandes indican una marcación a revisar."),
        ("Configuré el campo, pero quiero rehacerlo.",
         "Use <b>Reajustar</b> (vuelve al mapa) o <b>Sedes (Venues)</b> para actualizar/eliminar el lugar guardado."),
    ],
    'h_tips': 'Consejos para sacar el máximo',
    'tips': [
        "<b>Guarde la sede</b> tras marcar el campo: beneficia a todo el equipo automáticamente.",
        "<b>Combine períodos</b> para ver el partido entero en una línea de tiempo continua, sin el entretiempo.",
        "<b>Exporte</b> tablas y eventos en CSV para informes y seguimiento longitudinal.",
        "Para estudiar una jugada en tiempo real, use ventana de <b>1–2 min</b> y apunte el inicio al momento exacto.",
        "Pase el ratón sobre los gráficos para ver <b>valores exactos</b>; arrastre para hacer zoom.",
    ],
    'h_contact': 'Adiciones, sugerencias, comentarios',
    'contact_text': "¿Tiene ideas para mejorar la herramienta, encontró algo que ajustar o quiere sugerir una "
        "nueva funcionalidad? Sus comentarios son muy bienvenidos. Escriba a:",
}

EN = {
    'page': 'Page', 'footer': 'Developed by Gabriel Gazone dos Santos',
    'title': 'User Guide',
    'subtitle': 'Football Performance Analysis Platform &nbsp;|&nbsp; '
                'GPS, physical load and tactical intelligence',
    'intro': "This tool turns athletes' <b>GPS and inertial sensor</b> data (collected via Catapult Connect) into "
        "<b>clear, actionable insights</b> on physical performance and tactical behaviour. In just a few clicks you "
        "go from raw spreadsheets to <b>heat maps, load profiles, worst-case scenarios (WCS) and collective "
        "tactical visualisations</b> — including a 3D replay of the match. The goal: to give coaching and "
        "performance staff a <b>fast, visual and scientifically grounded</b> reading of each athlete and of the "
        "team as a system.",
    'h_start': 'Getting started in 4 steps',
    'steps': [
        ("Connect", "In the sidebar, enter your Connect <b>Token</b> and load the athletes and activities."),
        ("Choose the activity", "Select the <b>activity</b> (training or match) you want to analyse from the loaded list."),
        ("Select period(s) and athletes", "Pick one or more <b>periods</b> (1st half, 2nd half, etc.) and the "
         "<b>athletes</b> of interest. With several periods, the tool automatically creates a combined view."),
        ("Explore the 5 tabs", "Data is organised into five modules. Each tab has interactive charts (hover for "
         "details, drag to zoom) and export buttons."),
    ],
    'h_modules': 'The 5 platform modules',
    'mod_titles': ["SUMMARY", "FIELD & GPS", "PHYSICAL LOAD", "COLLECTIVE TACTICS"],
    'mod_lines': [
        ["<b>Overview</b>: total distance, PlayerLoad, top speed and key session indicators.",
         "<b>By Position</b>: compares athletes by tactical role (defender, midfielder, forward...)."],
        ["<b>Football Field</b>: occupancy heat maps and trajectory — this is where you mark the field.",
         "<b>Match Story</b>: heat maps by phase, showing how presence evolves over time.",
         "<b>WCS</b>: the most demanding passages of the match by rolling window."],
        ["<b>Efforts</b>: accelerations, decelerations and sprints, with zone and direction.",
         "<b>Time Windows</b>: rolling-window intensity and high / medium-high peaks.",
         "<b>Neuromuscular, Acc-Vel and HR</b>: fatigue, acc×speed profile and cardiac load (TRIMP + zones)."],
        ["The team as a <b>system</b>: cross-references every athlete's position at the same instant.",
         "Four views: <b>Pitch Control</b>, <b>Team Breathing</b>, <b>Voronoi</b> and <b>3D Replay</b>.",
         "Works even with <b>GPS-only</b> devices (reconstructs the coordinates)."],
    ],
    'h_field': 'Marking the field: enabling location data',
    'field_intro': "Raw GPS is just a cloud of <b>latitude/longitude</b> points. For the app to understand "
        "<i>where</i> on the pitch each point is — and thus produce zone occupancy, field thirds, directional "
        "efforts and the whole <b>Collective Tactics</b> module — you must tell it <b>where the field is and how it "
        "is oriented</b>. This is called <b>marking (or calibrating) the field</b>, and it is done once per venue, "
        "over a satellite image.",
    'h_where': 'Where it is',
    'where_text': "Tab <b>Field &amp; GPS &gt; Football Field</b>. Select a <b>reference athlete who has GPS</b> in "
        "the session (the field belongs to the venue, not the athlete — anyone with signal works). If the session "
        "has GPS points, a <b>satellite map</b> appears with the point cloud and a <b>yellow target</b> marking the "
        "field centre.",
    'h_steps': 'Step-by-step marking',
    'field_steps': [
        ("Show the field", "Click <b>Show Field</b> to overlay the field drawing (white lines) on the satellite image."),
        ("Centre it", "Drag the <b>yellow target</b> — or edit <b>Lat/Lon</b> — until the drawing covers the GPS point cloud, over the real pitch in the image."),
        ("Rotate", "Use the <b>Rot</b> buttons (« =-10°, ‹ =-1°, › =+1°, » =+10°) until the <b>drawing's lines match the real lines</b> of the field on the satellite. This is the most important step for tactical fidelity."),
        ("Adjust dimensions", "Set <b>Length</b> (90–120 m) and <b>Width</b> (55–90 m) to match the real size. Default is 105×68 m. <b>Marg.</b> adds an outer margin beyond the lines if you wish."),
        ("Apply", "Click <b>Apply Field</b>. From then on, location coordinates feed the heat maps, directional efforts and the whole Collective Tactics module."),
    ],
    'ctrl_header': ("Control in the map panel", "What it does"),
    'ctrl_rows': [
        ("Show Field", "Toggles the field drawing over the satellite."),
        ("Lat / Lon", "Moves the field centre. Drag the yellow target or edit the numbers."),
        ("Rot &nbsp;« ‹ › »", "Rotates the field: « -10°, ‹ -1°, › +1°, » +10°. Align the drawing's lines with the pitch."),
        ("Length / Width", "Length (90–120 m) and width (55–90 m). Default 105×68 m."),
        ("Marg.", "Extra margin drawn outside the lines (0–15 m)."),
        ("Apply Field", "Confirms the setup and loads the field coordinates."),
    ],
    'h_save': 'Save for everyone (venue database)',
    'save_text': "Right after applying, the app offers to <b>save the configuration to the shared venue "
        "database</b>, with a name (e.g. &ldquo;Municipal Stadium&rdquo;). It's well worth it: next time <b>any "
        "user</b> opens an activity <b>at this same venue</b>, the field is already <b>configured automatically</b> "
        "— no need to mark it again. You'll see the notice &ldquo;Field loaded from the shared database&rdquo;.",
    'save_bullets': [
        "<b>Readjust</b>: redo the marking from scratch (back to the satellite map).",
        "<b>Venues</b>: opens the manager to list, update or delete saved venues.",
    ],
    'wellmarked_title': "How to tell it's well marked",
    'wellmarked': [
        "The <b>white drawing lines</b> match the field lines in the satellite image.",
        "The <b>GPS point cloud</b> stays inside the field, without spilling far outside.",
        "The <b>goals</b> sit at the correct ends and the halfway line crosses the real centre circle.",
        "Tip: start with <b>rotation</b>, then adjust the <b>centre</b>, and finally the <b>dimensions</b>.",
    ],
    'h_tatica': 'Spotlight: the Collective Tactics tab',
    'tatica_intro': "It's the most innovative module. By syncing the position of all selected athletes, it reveals "
        "the team's behaviour as a whole — impossible to see athlete by athlete. It requires <b>at least 2 athletes "
        "with position</b> in the same period. The four views:",
    'views': [
        ("Pitch Control", "William Spearman's model (Liverpool FC). Colours the field by <b>space dominance</b> — "
         "who would reach each point first, accounting for speed and direction."),
        ("Team Breathing", "Shows the block <b>compressing</b> when defending and <b>expanding</b> in possession, "
         "with width, length, area and spread over time."),
        ("Voronoi", "The tactical &ldquo;stained glass&rdquo;: each region of the field belongs to the nearest "
         "player, exposing <b>space coverage and gaps</b> in the structure."),
        ("3D Replay", "A navigable <b>synthetic broadcast</b> (rotate the camera, zoom) with striped grass, goals "
         "and markings — reconstructed from coordinates alone."),
    ],
    'h_anim': 'Animation controls',
    'anim': [
        "<b>Analysis window</b>: from 30 s to 10 min, a custom duration or the <b>whole period</b>. Short windows "
        "show continuous movement (real time); the whole period gives the macro view of the entire match (with more "
        "spaced-out frames).",
        "<b>Window start</b>: drag it to place the analysis at <b>any moment</b> of the match (including the final minutes).",
        "<b>Speed</b>: <b>1× = real time</b> of the match; below is slow motion, above is sped up. Use <b>Play/Pause</b> and the slider to navigate the frames.",
    ],
    'twosliders_title': "Two time controls — don't confuse them",
    'twosliders': [
        "The <b>Window start</b> slider (above the chart) <b>moves the window</b> across the whole match.",
        "The slider <b>inside the chart</b> only scrubs the <b>frames of the current window</b>.",
    ],
    'h_trouble': 'Troubleshooting (common difficulties)',
    'faqs': [
        ("The map doesn't appear / &ldquo;No real GPS point found&rdquo;.",
         "That athlete's sensor didn't get a GPS <i>lock</i> in the session. Since the field belongs to the venue, "
         "pick <b>another reference athlete</b> with GPS signal to do the marking."),
        ("The field is crooked or misaligned on the satellite.",
         "First adjust the <b>rotation</b> (« ‹ › ») and then the <b>centre</b> (Lat/Lon) until the drawing's lines "
         "match the pitch. Also check <b>Length/Width</b> — wrong dimensions distort everything."),
        ("Collective Tactics shows &ldquo;coordinates reconstructed from GPS / approximate alignment&rdquo;.",
         "Your device is <b>GPS-only</b> (no native x/y). The <b>relative positions</b> between players are "
         "faithful; for better absolute alignment, <b>mark the field</b> in Field &amp; GPS — it will be used automatically here."),
        ("&ldquo;This tab needs at least 2 athletes with position in the same period&rdquo;.",
         "In the sidebar, select <b>2 or more athletes</b> that have GPS in that period."),
        ("The animation looks like &ldquo;photos&rdquo; jumping, with no continuous motion.",
         "Long passages have spaced-out frames. Reduce the <b>Analysis window</b> to <b>1–2 min</b> and use the "
         "<b>Window start</b> to focus on the play. The whole period is for the macro view."),
        ("GPS distances differ from what I expected.",
         "Check that the field is well marked and with the <b>real dimensions</b> of the pitch. The app also "
         "compares the field-filtered distance with the integrated GPS — big differences point to a marking that needs review."),
        ("I configured the field but want to redo it.",
         "Use <b>Readjust</b> (back to the map) or <b>Venues</b> to update/delete the saved venue."),
    ],
    'h_tips': 'Tips to get the most out of it',
    'tips': [
        "<b>Save the venue</b> after marking the field: it benefits the whole team automatically.",
        "<b>Combine periods</b> to see the whole match on a continuous timeline, without half-time.",
        "<b>Export</b> tables and events to CSV for reports and longitudinal tracking.",
        "To study a play in real time, use a <b>1–2 min</b> window and aim the start at the exact moment.",
        "Hover over the charts to see <b>exact values</b>; drag to zoom.",
    ],
    'h_contact': 'Additions, suggestions, comments',
    'contact_text': "Have ideas to improve the tool, found something to adjust, or want to suggest a new feature? "
        "Your feedback is very welcome. Write to:",
}

if __name__ == '__main__':
    gerar(PT, "Guia_do_Usuario_PT.pdf")
    gerar(ES, "Guia_del_Usuario_ES.pdf")
    gerar(EN, "User_Guide_EN.pdf")
