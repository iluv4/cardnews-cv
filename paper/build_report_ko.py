#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_report_ko.py -- 국문 2단 기말 보고서(PDF) 빌더.

cardnews-cv 논문(main.tex, CVPR 스타일)의 내용을 한국어 2단 보고서 양식으로
재구성한다. 본문은 왼쪽 단(위->아래) 후 오른쪽 단(위->아래)으로 자연스럽게
이어지는 2-Frame 흐름을 사용한다. 폰트는 맑은 고딕(Malgun Gothic).

출력: paper/cardnews_report_ko.pdf
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image,
    Table, TableStyle, FrameBreak, NextPageTemplate, KeepTogether, CondPageBreak,
)
from reportlab.lib.utils import ImageReader

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "cardnews_report_ko.pdf")

# ----------------------------------------------------------------------------
# Fonts (Malgun Gothic)
# ----------------------------------------------------------------------------
pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunBd", r"C:\Windows\Fonts\malgunbd.ttf"))
registerFontFamily("Malgun", normal="Malgun", bold="MalgunBd",
                   italic="Malgun", boldItalic="MalgunBd")

# ----------------------------------------------------------------------------
# Page geometry  (A4, 2-column)
# ----------------------------------------------------------------------------
PW, PH = A4                      # 595 x 842 pt
LM = RM = 17 * mm
TM = 16 * mm
BM = 16 * mm
GUT = 7 * mm                     # column gutter
CONTENT_W = PW - LM - RM
COL_W = (CONTENT_W - GUT) / 2.0
HEADER_H = 120 * mm              # title + authors + full-width abstract (page 1)

INK = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#555555")
ACCENT = colors.HexColor("#11324D")
RULE = colors.HexColor("#888888")

# ----------------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------------
title = ParagraphStyle("title", fontName="MalgunBd", fontSize=16, leading=21,
                       alignment=TA_CENTER, textColor=INK, spaceAfter=3)
subtitle = ParagraphStyle("subtitle", fontName="Malgun", fontSize=9.5,
                          leading=13, alignment=TA_CENTER, textColor=MUTED,
                          spaceAfter=2)
meta = ParagraphStyle("meta", fontName="Malgun", fontSize=9.5, leading=14,
                      alignment=TA_CENTER, textColor=INK, spaceBefore=4)

h1 = ParagraphStyle("h1", fontName="MalgunBd", fontSize=11.5, leading=15,
                    textColor=ACCENT, spaceBefore=9, spaceAfter=4,
                    alignment=TA_LEFT, keepWithNext=True)
h2 = ParagraphStyle("h2", fontName="MalgunBd", fontSize=10, leading=14,
                    textColor=INK, spaceBefore=6, spaceAfter=2,
                    alignment=TA_LEFT, keepWithNext=True)
body = ParagraphStyle("body", fontName="Malgun", fontSize=9.3, leading=14.2,
                      alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5,
                      firstLineIndent=10)
body0 = ParagraphStyle("body0", parent=body, firstLineIndent=0)
abhead = ParagraphStyle("abhead", fontName="MalgunBd", fontSize=11,
                        leading=15, alignment=TA_CENTER, textColor=INK,
                        spaceBefore=8, spaceAfter=4)
abstract = ParagraphStyle("abstract", fontName="Malgun", fontSize=9.2,
                          leading=14.0, alignment=TA_JUSTIFY, textColor=INK,
                          spaceAfter=4, firstLineIndent=10,
                          leftIndent=10, rightIndent=10)
caption = ParagraphStyle("caption", fontName="Malgun", fontSize=8.2,
                         leading=11.5, alignment=TA_CENTER, textColor=MUTED,
                         spaceBefore=3, spaceAfter=8)
ref = ParagraphStyle("ref", fontName="Malgun", fontSize=8.2, leading=11.6,
                     alignment=TA_LEFT, textColor=INK, spaceAfter=3,
                     leftIndent=12, firstLineIndent=-12)


def P(t, st=body):
    return Paragraph(t, st)


def figure(fname, cap, max_h=None):
    full = os.path.join(FIG, fname)
    iw, ih = ImageReader(full).getSize()
    w = COL_W
    h = ih * (w / float(iw))
    if max_h and h > max_h:
        h = max_h
        w = iw * (h / float(ih))
    img = Image(full, width=w, height=h)
    img.hAlign = "CENTER"
    return KeepTogether([Spacer(1, 2), img, P(cap, caption)])


def ablation_table():
    data = [
        ["설정 (Configuration)", "mAP@50-95", "mAP@50"],
        ["scratch (사전학습 없음)", "0.502", "0.736"],
        ["baseline (n, freeze 10)", "0.662", "0.844"],
        ["freeze 0", "0.671", "0.851"],
        ["aug: heavy", "0.644", "0.838"],
        ["aug: card-tuned", "0.696", "0.838"],
        ["img 768", "0.680", "0.841"],
        ["YOLOv8-s (freeze 10)", "0.705", "0.852"],
        ["long300 + card aug (최고)", "0.718", "0.854"],
    ]
    cw = [COL_W * 0.54, COL_W * 0.23, COL_W * 0.23]
    t = Table(data, colWidths=cw, hAlign="CENTER")
    best = len(data) - 1
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "MalgunBd"),
        ("FONTNAME", (0, 1), (-1, -1), "Malgun"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
        ("FONTNAME", (0, best), (-1, best), "MalgunBd"),
        ("BACKGROUND", (0, best), (-1, best), colors.HexColor("#EAF0F6")),
    ]))
    return KeepTogether([Spacer(1, 2), t,
                         P("[표 2] 시드 109장(의사 레이블) 기준 검출기 ablation.", caption)])


def rw_table():
    cell = ParagraphStyle("cell", fontName="Malgun", fontSize=7.2, leading=8.8,
                          alignment=TA_CENTER, textColor=INK)
    cellL = ParagraphStyle("cellL", parent=cell, alignment=TA_LEFT)
    hdr = ParagraphStyle("hdr", parent=cell, fontName="MalgunBd")

    def c(t):
        return Paragraph(t, cell)

    def cl(t):
        return Paragraph(t, cellL)

    def h(t):
        return Paragraph(t, hdr)

    data = [
        [h("방법"), h("한글"), h("다중<br/>페이지"), h("내용<br/>인지"), h("편집<br/>가능")],
        [cl("LayoutGAN [8]"), c("×"), c("×"), c("×"), c("×")],
        [cl("LayoutTransformer [9]"), c("×"), c("×"), c("△"), c("×")],
        [cl("PosterLayout [1] / DS-GAN"), c("×"), c("×"), c("○"), c("×")],
        [cl("<b>본 연구: 템플릿 엔진</b>"), c("○"), c("○"), c("○"), c("○")],
    ]
    cw = [COL_W * 0.40, COL_W * 0.13, COL_W * 0.17, COL_W * 0.15, COL_W * 0.15]
    t = Table(data, colWidths=cw, hAlign="CENTER")
    best = len(data) - 1
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK),
        ("BACKGROUND", (0, best), (-1, best), colors.HexColor("#EAF0F6")),
    ]))
    return KeepTogether([Spacer(1, 2), t,
                         P("[표 1] 레이아웃 생성 접근 비교 (○ 지원 · △ 부분 · × 미지원).", caption)])


# ----------------------------------------------------------------------------
# Page callbacks (footer page number)
# ----------------------------------------------------------------------------
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Malgun", 8)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(PW / 2.0, 9 * mm, "- %d -" % doc.page)
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(OUT, pagesize=A4,
                          leftMargin=LM, rightMargin=RM,
                          topMargin=TM, bottomMargin=BM,
                          title="검출 기반 레이아웃 분석과 템플릿 생성",
                          author="이승주")

    # --- page 1: full-width header frame + two columns below ---
    fh = Frame(LM, PH - TM - HEADER_H, CONTENT_W, HEADER_H, id="header",
               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    body_top = PH - TM - HEADER_H - 4 * mm
    body_h1 = body_top - BM
    f1a = Frame(LM, BM, COL_W, body_h1, id="c1a",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f1b = Frame(LM + COL_W + GUT, BM, COL_W, body_h1, id="c1b",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    # --- later pages: two full-height columns ---
    col_h = PH - TM - BM
    f2a = Frame(LM, BM, COL_W, col_h, id="c2a",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f2b = Frame(LM + COL_W + GUT, BM, COL_W, col_h, id="c2b",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    doc.addPageTemplates([
        PageTemplate(id="first", frames=[fh, f1a, f1b], onPage=footer),
        PageTemplate(id="later", frames=[f2a, f2b], onPage=footer),
    ])

    s = []  # story

    # ===== Title block (header frame) =====
    s.append(P("검출 기반 레이아웃 분석과 템플릿 생성:<br/>"
               "한국어 카드뉴스와 다중 페이지 덱의 시계열 분석", title))
    s.append(P("컴퓨터비전 · 시계열 기말 프로젝트<br/>"
               "202321131 이승주", meta))

    # ===== Abstract (full width, at top) =====
    s.append(P("초록", abhead))
    s.append(P(
        "카드뉴스는 한국 소셜미디어에서 널리 쓰이는 짧은 이미지 기반 슬라이드 묶음으로, "
        "사진 배경 위에 한글 타이포그래피를 정교하게 배치하여 구성된다. 본 연구는 "
        "(i) 기존 카드뉴스의 <b>레이아웃을 분석</b>하는 문제와 (ii) 배포 가능한 품질의 카드를 "
        "<b>새로 생성</b>하는 문제를 다룬다. 우리는 <b>66개의 다중 페이지 덱</b>으로 구성된 "
        "<b>687장</b>의 한국어 카드 코퍼스를 구축하고, 고재현율(high-recall) OCR 파이프라인과 "
        "줄-블록 병합으로 제목/본문 요소 주석을 의사 레이블(pseudo-label)로 자동 생성한다. "
        "이 의사 레이블로 미세조정한 YOLOv8 검출기는 <b>mAP@50-95 = 0.718</b>(mAP@50 = 0.854)에 "
        "도달하며, 전이학습이 결정적이다(scratch: 0.502). 이어 두 가지 생성 경로를 비교한다 — "
        "PosterLayout[1]의 DS-GAN을 적응시킨 연구용 베이스라인과, 실제 레이아웃을 추출하여 "
        "디자이너용 한글 서체로 선명한 벡터 텍스트를 다시 렌더링하는 <b>검출 기반 템플릿 엔진</b>이다. "
        "템플릿 엔진은 배포 환경에서 뚜렷하게 더 안정적이고 가독성이 높은 반면, GAN이 렌더링한 "
        "텍스트는 실사용이 불가능하다. 끝으로 각 덱을 순서가 있는 시퀀스로 보고, 66개 덱에 걸쳐 "
        "페이지 위치에 따른 레이아웃 동역학을 <b>시계열 분석</b>하여 일관된 '표지 대 내지' 구조"
        "(표지는 더 어둡고 채도가 높으며 에지 밀도가 낮음)를 밝힌다. 코드와 데이터 도구, 그림을 공개한다.",
        abstract))
    s.append(NextPageTemplate("later"))  # page 2+ : two columns only
    s.append(FrameBreak())               # drop into the two-column body (page 1)

    # ===== 1. Introduction =====
    s.append(P("1. 서론", h1))
    s.append(P(
        "한국어 카드뉴스(<i>카드뉴스</i>)는 하나의 기사를 디자인된 슬라이드의 연속으로 압축한 "
        "다중 페이지 시각 게시물이다. 이를 잘 만들려면 <b>제목</b>, <b>본문</b>, <b>로고</b>, "
        "<b>언더레이</b> 같은 디자인 요소를 내용 인지적으로(content-aware) 배치하여, 텍스트가 "
        "읽히기 쉽고 이미지의 돌출 영역을 피하며 페이지 전반에서 일관된 시각 리듬을 따르도록 해야 한다. "
        "이는 PosterLayout[1]이 정식화한 <i>내용 인지형 시각-텍스트 레이아웃</i> 문제와 정확히 같되, "
        "(a) 대부분의 레이아웃 코퍼스가 다루지 않는 한글 문자와 (b) 카드뉴스 고유의 다중 페이지 덱 "
        "구조에 특화된 형태이다. 한 카드뉴스 덱의 예시(표지→내지→마무리)는 [그림 1]과 같다.", body0))
    s.append(figure("fig_teaser.png",
                    "[그림 1] 한국어 카드뉴스 덱 예시(한 게시물의 페이지를 순서대로 표시). 표지는 강한 "
                    "대비의 제목으로 주목을 끌고, 내지는 본문·도표가 많아 더 밝고 조밀하다. 본 연구는 "
                    "이런 덱을 분석·생성한다.", max_h=40*mm))
    s.append(P(
        "실무적으로 카드뉴스 자동 생성의 큰 장벽은 <i>픽셀 베이킹(pixel-baking)</i> 방식이다. "
        "텍스트를 배경에 통째로 합성해 굽는 확산(diffusion) 계열 생성은 한 장당 수십 초가 걸리고, "
        "비결정적(non-deterministic)이며, 오타 한 글자를 고치려 해도 전체 이미지를 다시 생성해야 하는 "
        "치명적 비효율을 낳는다. 더욱이 한글 받침이 깨지거나 글자가 뭉개지는 등 텍스트 가독성이 "
        "보장되지 않는다. 이러한 관찰은 본 연구가 지향하는 방향 — 즉 텍스트를 배경과 분리해 "
        "<i>편집 가능한</i> 형태로 유지하고, 실제 레이아웃을 복사·재렌더링하는 검출 기반 접근 — 의 "
        "동기가 된다.", body))
    s.append(P("우리는 다음 세 가지 기여로 문제를 구체적이고 재현 가능하게 만든다.", body0))
    s.append(P("(1) <b>한국어 카드뉴스 코퍼스</b>: 66개 순서 덱으로 묶인 687장의 이미지와, 수작업 "
               "주석 없이 제목/본문 박스를 산출하는 자동 고재현율 라벨링 파이프라인(OCR + 줄-블록 "
               "병합)(3장).", body0))
    s.append(P("(2) <b>레이아웃 시스템</b>: 미세조정 YOLOv8 검출기와 두 생성 경로 — 적응된 DS-GAN "
               "베이스라인과 검출 기반 <i>템플릿 엔진</i> — 를 실사용 품질 관점에서 비교한다(4·6장).", body0))
    s.append(P("(3) <b>순차(시계열) 분석</b>: 페이지 인덱스를 이산 시간축으로 보고, 표지에서 마지막 "
               "페이지까지 시각 구조가 어떻게 변하는지 정량화한다(5장).", body0))

    # ===== 2. Related Work =====
    s.append(P("2. 관련 연구", h1))
    s.append(P("<b>내용 인지형 레이아웃 생성.</b> LayoutGAN[8]과 LayoutTransformer[9]는 노이즈나 "
               "부분 레이아웃으로부터 요소 배치를 생성한다. PosterLayout[1]은 깨끗한 배경과 돌출(saliency) "
               "맵을 조건으로 받아 도메인 정렬 GAN(DS-GAN)을 학습하여 요소를 돌출 영역 밖에 배치하며, "
               "본 연구의 생성 베이스라인이다.", body0))
    s.append(P("<b>객체 검출.</b> 우리는 COCO[12]에 사전학습되고 카드 요소에 미세조정된 YOLOv8[2]을 "
               "사용한다. YOLO 계열[10]의 현대적 후속 모델이다.", body0))
    s.append(P("<b>텍스트 검출 / OCR.</b> 부트스트랩 라벨에는 EasyOCR[3]을 사용하며, CRAFT[7] 같은 "
               "문자 영역 기반 방법은 향후 더 높은 재현율을 위한 대안이다.", body0))
    s.append(P("<b>돌출 및 인페인팅.</b> 생성 경로는 돌출 객체 검출(U²-Net[5], BASNet[6])로 보호 "
               "영역을 찾고, 대형 마스크 인페인팅(LaMa[4])으로 기존 텍스트를 지워 깨끗한 배경을 만든다.", body0))
    s.append(P("[표 1]은 선행 레이아웃 생성 연구와 본 연구를 핵심 축에서 비교한다. 대부분의 선행 연구는 "
               "한글과 다중 페이지 덱을 다루지 않으며, 생성 결과의 텍스트를 사후에 편집할 수 없다 — 본 "
               "연구의 검출 기반 템플릿 엔진은 이 네 축을 모두 만족한다.", body0))
    s.append(rw_table())

    # ===== 3. Dataset =====
    s.append(P("3. 데이터셋", h1))
    s.append(P("<b>데이터 확보.</b> 한국어 카드뉴스는 공개된 정제 코퍼스가 사실상 없으므로, 세 경로를 "
               "병행해 데이터를 확보했다. (1) <b>공공 데이터셋</b> — 공공기관·지자체가 배포한 카드뉴스 등 "
               "공개 자료를 수집해 도메인 다양성과 라이선스 안전성을 확보했다. (2) <b>직접 제작</b> — "
               "레이아웃 원형(에디토리얼·중앙·하단 패널·표지)을 통제하기 위해 일부 카드를 직접 디자인·"
               "제작하여 시드 참조로 사용했다. (3) <b>크롤링</b> — 실제 사용 분포를 반영하기 위해 인스타그램 "
               "공개 게시물을 <b>RapidAPI 기반 인스타그램 API</b>로 크롤링하여 다중 페이지 카드뉴스 덱을 "
               "원본 순서 그대로 수집했다(게시물 단위로 페이지 순서를 보존). 이렇게 모은 687장은 시드 참조 "
               "109장과 새로 수집한 578장으로 구성되며, 후자는 공공·농업/스마트팜·에디토리얼 주제를 아우르고 "
               "주로 인스타그램 RapidAPI 크롤링으로 확보되었다. 새 578장은 66개의 <i>덱</i>(4-11페이지의 "
               "순서 시리즈)으로 구성된다.", body0))
    s.append(P("<b>표준화.</b> 모든 이미지는 ASCII 파일명의 RGB JPEG로 재인코딩하여"
               "(OpenCV/Ultralytics가 OS와 무관하게 읽도록) 긴 변 2048px로 제한하며, 생성 파이프라인에서는 "
               "추가로 513×750 PosterLayout 캔버스로 리스케일한다. 표본은 [그림 2].", body0))
    s.append(figure("fig_dataset.png",
                    "[그림 2] 687장·66덱 한국어 카드뉴스 코퍼스의 대표 페이지."))
    s.append(P("<b>자동 라벨링.</b> 수작업 박스 주석은 확장성이 없으므로 <i>의사 레이블</i>을 생성한다. "
               "EasyOCR[3]을 고재현율 설정(canvas_size=2560, mag_ratio=2.0, low_text=0.3)으로 돌려 작은 "
               "본문 텍스트까지 복원하고, 각 텍스트 줄의 높이가 이미지 높이의 4.5%를 넘으면 <i>제목</i>, "
               "아니면 <i>본문</i>으로 분류한다. 레이아웃 생성에는 개별 줄이 아니라 요소 <i>영역</i>이 "
               "필요하므로, 같은 클래스의 인접 줄을 높이의 일정 비율만큼 확장한 박스에 대한 union-find로 "
               "<b>블록</b>으로 병합한다. 이로써 시드셋에서 제목 284개, 본문 641개 박스를 얻으며, "
               "logo/underlay는 수작업 주석 대상으로 남긴다. 자동 의사 레이블의 예시는 [그림 3]과 같다.", body))
    s.append(figure("fig_pseudolabel.png",
                    "[그림 3] 자동 의사 레이블 예시. 고재현율 OCR과 줄-블록 병합으로 얻은 제목/본문 "
                    "블록(빨간 박스)을 원본 카드 위에 표시했다. 로고/언더레이는 아직 주석하지 않는다."))

    # ===== 4. Method =====
    s.append(P("4. 방법", h1))
    s.append(P("[그림 4]는 시스템을 요약한다: 수집 → 검출 → {템플릿 엔진 | DS-GAN} → 렌더링.", body0))
    s.append(figure("fig_pipeline.png",
                    "[그림 4] 시스템 개요. 공유 검출기가 두 생성 경로에 입력을 공급한다. "
                    "템플릿 엔진(3a)이 배포 경로, DS-GAN(3b)이 연구용 베이스라인."))
    s.append(P("4.1 요소 검출", h2))
    s.append(P("COCO 가중치에서 YOLOv8-nano(및 -small)를 의사 레이블로 미세조정한다. 수평 뒤집기는 "
               "한글을 거울상으로 만들기 때문에 비활성화하며(fliplr=0), 카드에 맞춘 증강 세트(약한 모자이크, "
               "HSV 지터, 작은 스케일/이동, 랜덤 이레이징)를 쓴다. 검출기는 (a) 인페인팅용 마스크와 "
               "DS-GAN이 쓰는 train_csv를 공급하고, (b) 다음에 설명하는 <i>레이아웃 템플릿을 추출</i>한다.",
               body0))
    s.append(P("4.2 레이아웃 생성: 두 경로", h2))
    s.append(P("<b>(3b) DS-GAN 베이스라인.</b> PosterLayout[1]을 따라 필요한 Dataset/(인페인팅된 포스터, "
               "이중 돌출 맵, 정규화된 요소 박스의 train_csv)을 U²-Net/BASNet 돌출과 LaMa 인페인팅으로 "
               "구성하고, DS-GAN을 미세조정하여 깨끗한 한글 배경 위에 (클래스, 박스) 레이아웃을 생성한다. "
               "모델은 그럴듯한 박스를 만들지만 <i>렌더링된</i> 결과는 배포 불가능하다 — 읽을 수 있는 "
               "한글 글리프를 배치하지 못하고 레이아웃 안정성이 실행마다 달라진다.", body0))
    s.append(P("<b>(3a) 검출 기반 템플릿 엔진(제안).</b> 우리는 대신 실제 레이아웃을 <i>복사·재렌더링</i>한다. "
               "검출기가 참조 카드에서 제목/본문 블록을 찾고, 고전적 인페인팅(또는 LaMa)이 원본 텍스트를 "
               "제거하며, 새 내용을 동일한 블록 기하에 Pretendard[11] 서체로 다시 조판한다. 추가로 코퍼스에서 "
               "증류한 소규모 레이아웃 원형 라이브러리(에디토리얼·중앙·하단 패널·표지)와, 텍스트가 놓일 "
               "가장 잔잔한 이미지 띠(평균 그래디언트 크기가 가장 낮은 영역)를 고르고 휘도 적응형 스크림을 "
               "적용하여 <i>어떤</i> 배경에서도 가독성을 보장하는 돌출 인지 배치 단계를 둔다.", body))
    s.append(P("4.3 렌더링", h2))
    s.append(P("카드는 1080×1350(인스타그램 4:5)으로 합성한다. 텍스트 색은 각 블록 아래 배경 휘도에서 "
               "선택하고(어두우면 흰색, 밝으면 거의 검정), 글자 크기는 박스별 자동 맞춤이며, 부드러운 둥근 "
               "스크림이 대비를 보장하고 잔여 텍스처를 덮는다. 타이포그래피는 제목에 Pretendard Bold, 본문에 "
               "Regular을 쓰고 고정된 간격 체계와 강조 규칙을 따른다.", body0))

    # ===== 5. Sequential analysis =====
    s.append(P("5. 덱의 시계열 분석", h1))
    s.append(P("카드뉴스 덱은 순서가 있는 페이지 시퀀스다. 우리는 페이지 인덱스를 이산 시간축으로 보고 "
               "레이아웃/시각 구조가 그 축을 따라 어떻게 변하는지 묻는다. 66개 덱(각 ≥4페이지)에 대해 "
               "페이지마다 네 가지 cv2 특징 — Canny <i>에지 밀도</i>(텍스트/그래픽 밀도), <i>밝기</i>, "
               "Hasler-Süsstrunk <i>색채성(colorfulness)</i>[13], HSV <i>채도</i> — 을 계산하여 10개의 "
               "상대 위치 빈(0=표지, 1=마지막)으로 평균낸다.", body0))
    s.append(P("[그림 5]는 일관된 <b>표지 대 내지</b> 패턴을 보인다: 표지는 뚜렷하게 더 <i>어둡고</i>"
               "(밝기 0.62 대 내지 ≈0.71), 더 <i>채도가 높으며</i>(0.33 대 ≈0.27) 더 <i>색채롭고</i>, "
               "<i>에지 밀도가 가장 낮다</i>(0.097). 내지는 더 밝고 조밀하며(텍스트가 많음), 마지막 "
               "페이지는 표지의 채도 톤으로 부분적으로 회귀한다. 이는 디자인 관습 — 눈길을 끄는 고채도 표지 "
               "뒤에 읽기 쉬운 내용 페이지 — 을 정량화하며, <i>위치 조건부</i> 템플릿(4장)을 정당화한다: "
               "엔진은 1페이지에 표지 원형, 이후에 내용 원형을 선택한다.", body))
    s.append(figure("fig_timeseries.png",
                    "[그림 5] 상대 페이지 위치에 따른 덱 동역학(66덱). 표지는 내지보다 더 어둡고 "
                    "채도/색채가 높으며 에지 밀도가 낮다."))

    # ===== 6. Experiments =====
    s.append(P("6. 실험", h1))
    s.append(P("<b>설정.</b> YOLOv8 미세조정, 640px, 150-300 에폭, GPU에 맞춘 배치이며 지표는 박스 mAP다. "
               "시드 분할은 학습 93 / 검증 16이다. 클라우드 학습은 RTX 4090, 개발/추론은 RTX 3050에서 수행한다.",
               body0))
    s.append(P("6.1 검출기 ablation", h2))
    s.append(P("[표 2]와 [그림 6]이 ablation을 보고한다. 전이학습이 지배적 요인이다(scratch 0.502 대 "
               "baseline 0.662). 카드에 맞춘 증강과 더 긴 스케줄이 최고의 nano 모델을 준다"
               "(long300_card: mAP@50-95 = 0.718, mAP@50 = 0.854). YOLOv8-small도 경쟁력 있다(0.705). "
               "시드 반복 간 편차는 <0.02 mAP로 순위가 안정적이다. 정성적 검출은 [그림 8].", body0))
    s.append(ablation_table())
    s.append(figure("fig_ablation.png",
                    "[그림 6] 주요 ablation 결과(박스 mAP@50-95). 전이학습과 카드 맞춤·장기 스케줄이 "
                    "향상을 견인한다."))
    s.append(P("<b>클래스별·오류 분석.</b> 최고 모델은 정밀도 0.785, 재현율 0.842이며, 클래스별 AP는 "
               "제목 0.909로 본문 0.798보다 높다 — 제목은 크고 대비가 강해 검출이 쉽고, 본문은 작고 밀집해 "
               "더 어렵다. 혼동 행렬([그림 7])은 주된 오류가 클래스 간 혼동이 아니라 <i>배경과의 혼동</i>"
               "(미검출·오검출)임을 보여준다: 본문 19개·제목 7개가 배경으로 누락되며 대부분 작은 텍스트에서 "
               "발생한다. logo/underlay는 의사 레이블이 없어 학습되지 않았고, 이는 향후 수작업 주석 과제다.",
               body0))
    s.append(figure("fig_confusion.png",
                    "[그림 7] 검증셋 혼동 행렬(제목/본문/로고/언더레이/배경). 오류 대부분이 배경 누락이며, "
                    "로고·언더레이는 미주석으로 비어 있다.", max_h=68*mm))
    s.append(figure("fig_detector_pred.jpg",
                    "[그림 8] 검증용 미사용 카드에 대한 검출기 예측(제목/본문).", max_h=86*mm))
    s.append(P("<b>강건성(5-겹 교차검증).</b> 베이스라인의 5-겹 교차검증은 mAP@50-95 = 0.611±0.049로, "
               "소규모 데이터 결과가 분할에 종속적이지 않음을 확인한다.", body0))
    s.append(P("<b>데이터 규모(109 대 687).</b> 새 578장의 효과를 측정하기 위해 <i>누수 없는(leak-free)</i> "
               "프로토콜을 채택한다: 공통 테스트셋(새 이미지의 15%)을 떼어 학습에 전혀 쓰지 않고, 최고 "
               "레시피를 (i) 시드 109장과 (ii) 전체 687장으로 각각 학습하여 동일한 테스트셋에서 평가한다. "
               "도구는 공개하며, 이 비교의 클라우드 학습은 진행 중이므로 미완성 실행에서 결론을 끌어내지 "
               "않도록 완결된 결과가 아니라 프로토콜로 보고한다.", body))
    s.append(P("6.2 생성 품질", h2))
    s.append(P("[그림 9]는 템플릿 엔진의 생성 결과 예시를 보인다. 템플릿 엔진은 1080×1350에서 읽기 쉽고 "
               "일관되게 디자인된 한글 카드를 산출하는 반면, DS-GAN 경로는 쓸 만한 <i>박스</i>는 만들지만 "
               "배포 가능한 렌더링 텍스트는 만들지 못한다. 따라서 실서비스용 카드뉴스에는 템플릿 엔진을 "
               "채택하고 DS-GAN은 연구용 비교로 남긴다.", body0))
    s.append(figure("fig_results_template.png",
                    "[그림 9] 템플릿 엔진 카드(고해상도, Pretendard, 적응형 색상)."))

    # ===== 7. Discussion =====
    s.append(P("7. 논의 및 한계", h1))
    s.append(P("의사 레이블이 검출기 품질의 상한을 정한다: EasyOCR은 매우 작거나 스타일화된 텍스트를 "
               "놓치므로, 데이터 양이 아니라 라벨 재현율이 구속 조건이다. 본 연구는 고재현율 OCR 설정과 "
               "블록 병합으로 이를 완화했고, 향후 CRAFT[7]나 수작업 logo/underlay 주석으로 보완할 수 있다. "
               "DS-GAN 경로는 한글 글리프 렌더링과 소규모 데이터 불안정성에 의해 제한된다. 시계열 분석은 "
               "저수준 프록시를 사용하므로, 페이지별 검출기 기반 요소 통계가 자연스러운 확장이다.", body0))

    # ===== 8. Conclusion =====
    s.append(P("8. 결론", h1))
    s.append(P("우리는 한국어 카드뉴스 레이아웃을 위한 재현 가능한 파이프라인을 제시했다: 자동 블록 라벨을 "
               "갖춘 687장·66덱 코퍼스, 미세조정 YOLOv8 검출기(mAP@50-95 = 0.718), 배포 관점에서 검출 기반 "
               "템플릿 엔진이 우세한 두 생성 경로의 비교, 그리고 정량적 '표지 대 내지' 디자인 관습을 밝히는 "
               "시계열 분석이다. 이 산출물들은 실용적인 한국어 카드뉴스 생성 서비스의 토대를 제공한다.", body0))

    # ===== References =====
    s.append(P("참고문헌", h1))
    refs = [
        '[1] H.-Y. Hsu, X. He, Y. Peng, H. Kong, and Q. Zhang, "PosterLayout: A New '
        'Benchmark and Approach for Content-aware Visual-Textual Presentation Layout," '
        'In <i>CVPR</i>. 2023.',
        '[2] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," 2023. '
        '<i>https://github.com/ultralytics/ultralytics</i>.',
        '[3] JaidedAI, "EasyOCR," 2020. <i>https://github.com/JaidedAI/EasyOCR</i>.',
        '[4] R. Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier '
        'Convolutions," In <i>WACV</i>. 2022.',
        '[5] X. Qin et al., "U2-Net: Going deeper with nested U-structure for salient '
        'object detection," <i>Pattern Recognition</i>, vol. 106. 2020.',
        '[6] X. Qin et al., "BASNet: Boundary-Aware Salient Object Detection," '
        'In <i>CVPR</i>. 2019.',
        '[7] Y. Baek, B. Lee, D. Han, S. Yun, and H. Lee, "Character Region Awareness '
        'for Text Detection," In <i>CVPR</i>. 2019.',
        '[8] J. Li, J. Yang, A. Hertzmann, J. Zhang, and T. Xu, "LayoutGAN: Generating '
        'Graphic Layouts with Wireframe Discriminators," In <i>ICLR</i>. 2019.',
        '[9] K. Gupta et al., "LayoutTransformer: Layout Generation and Completion with '
        'Self-attention," In <i>ICCV</i>. 2021.',
        '[10] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: '
        'Unified, Real-Time Object Detection," In <i>CVPR</i>. 2016.',
        '[11] H. Kil, "Pretendard: A variable, modern Korean/Latin sans-serif typeface," '
        '2021. <i>https://github.com/orioncactus/pretendard</i>.',
        '[12] T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," '
        'In <i>ECCV</i>. 2014.',
        '[13] D. Hasler and S. E. Suesstrunk, "Measuring colorfulness in natural images," '
        'In <i>Human Vision and Electronic Imaging VIII</i>, vol. 5007. 2003.',
    ]
    for r in refs:
        s.append(P(r, ref))

    doc.build(s)


if __name__ == "__main__":
    build()
    sz = os.path.getsize(OUT)
    print("Wrote %s (%.1f KB)" % (OUT, sz / 1024.0))
