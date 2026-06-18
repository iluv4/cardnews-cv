#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_pdf.py -- Local single-column PDF preview of paper/main.tex.

Renders a clean, readable Letter-size preview that mirrors the content of
main.tex (title, authors, abstract, all sections, figures, the detector
ablation table, and a numbered References list built from refs.bib).

This is NOT a LaTeX compiler. It is a faithful platypus/reportlab rendering
intended as a quick local preview. The source of truth for content remains
paper/main.tex; for actual CVPR submission, compile main.tex (see README.md).

Usage:
    py -3 paper/build_pdf.py
Output:
    paper/cardnews_paper.pdf
"""

import os
import re

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    KeepTogether, ListFlowable, ListItem,
)
from reportlab.lib.utils import ImageReader

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
OUT_PDF = os.path.join(HERE, "cardnews_paper.pdf")
REFS_BIB = os.path.join(HERE, "refs.bib")


# --------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------
def build_styles():
    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "PaperTitle", parent=styles["Title"],
        fontName="Times-Bold", fontSize=18, leading=22,
        alignment=TA_CENTER, spaceAfter=10,
    )
    author = ParagraphStyle(
        "PaperAuthor", parent=styles["Normal"],
        fontName="Times-Roman", fontSize=11, leading=15,
        alignment=TA_CENTER, spaceAfter=16,
    )
    abstract_head = ParagraphStyle(
        "AbstractHead", parent=styles["Normal"],
        fontName="Times-Bold", fontSize=12, leading=14,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=6,
    )
    body = ParagraphStyle(
        "PaperBody", parent=styles["BodyText"],
        fontName="Times-Roman", fontSize=10.5, leading=14.5,
        alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=6,
        firstLineIndent=0,
    )
    abstract_body = ParagraphStyle(
        "AbstractBody", parent=body,
        fontName="Times-Italic", fontSize=10, leading=14,
        leftIndent=18, rightIndent=18,
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontName="Times-Bold", fontSize=14, leading=17,
        spaceBefore=14, spaceAfter=6, alignment=TA_LEFT,
        textColor=colors.black,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontName="Times-Bold", fontSize=12, leading=15,
        spaceBefore=8, spaceAfter=4, alignment=TA_LEFT,
        textColor=colors.black,
    )
    caption = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontName="Times-Italic", fontSize=9, leading=12,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=10,
        textColor=colors.HexColor("#333333"),
    )
    listitem = ParagraphStyle(
        "ListItem", parent=body, spaceBefore=1, spaceAfter=3,
    )
    ref = ParagraphStyle(
        "Ref", parent=styles["Normal"],
        fontName="Times-Roman", fontSize=9.5, leading=12.5,
        alignment=TA_LEFT, spaceAfter=4,
    )
    return dict(title=title, author=author, abstract_head=abstract_head,
                body=body, abstract_body=abstract_body, h1=h1, h2=h2,
                caption=caption, listitem=listitem, ref=ref)


# --------------------------------------------------------------------------
# Figure helper -- scale to text width
# --------------------------------------------------------------------------
def fig(path, caption_text, styles, max_w, max_h=4.6 * inch):
    full = os.path.join(FIG_DIR, path)
    if not os.path.exists(full):
        return [Paragraph("[missing figure: %s]" % path, styles["caption"])]
    iw, ih = ImageReader(full).getSize()
    scale = max_w / float(iw)
    w = max_w
    h = ih * scale
    if h > max_h:
        scale = max_h / float(ih)
        h = max_h
        w = iw * scale
    img = Image(full, width=w, height=h)
    img.hAlign = "CENTER"
    cap = Paragraph(caption_text, styles["caption"])
    return [KeepTogether([Spacer(1, 4), img, cap])]


def P(text, styles, style="body"):
    return Paragraph(text, styles[style])


# --------------------------------------------------------------------------
# Detector ablation table
# --------------------------------------------------------------------------
def ablation_table(styles, text_w):
    header = ["Configuration", "mAP@50-95", "mAP@50"]
    rows = [
        ("scratch (no pretrain)", "0.502", "0.736"),
        ("baseline (n, freeze 10)", "0.662", "0.844"),
        ("freeze 0", "0.671", "0.851"),
        ("aug: heavy", "0.644", "0.838"),
        ("aug: card-tuned", "0.696", "0.838"),
        ("img 768", "0.680", "0.841"),
        ("YOLOv8-s (freeze 10)", "0.705", "0.852"),
        ("long300 + card aug (best)", "0.718", "0.854"),
    ]
    data = [header] + [list(r) for r in rows]

    col_w = [text_w * 0.56, text_w * 0.22, text_w * 0.22]
    tbl = Table(data, colWidths=col_w, hAlign="CENTER")

    best_idx = len(data) - 1  # last row is "best"
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
        ("FONTNAME", (0, best_idx), (-1, best_idx), "Times-Bold"),
        ("BACKGROUND", (0, best_idx), (-1, best_idx), colors.HexColor("#eef3fb")),
    ]
    tbl.setStyle(TableStyle(style_cmds))
    cap = Paragraph(
        "Table 1. Detector ablation on the 109-image seed set (pseudo-labels).",
        styles["caption"])
    return KeepTogether([Spacer(1, 4), tbl, cap])


# --------------------------------------------------------------------------
# refs.bib parser -> numbered reference strings
# --------------------------------------------------------------------------
def parse_bib(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    entries = []
    # split on @type{key, ... } blocks
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),", raw):
        start = m.end()
        # find matching close brace for this entry
        depth = 1
        i = start
        while i < len(raw) and depth > 0:
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
            i += 1
        body = raw[start:i - 1]
        key = m.group(2).strip()
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            fname = fm.group(1).lower()
            j = fm.end()
            if j >= len(body):
                continue
            if body[j] == "{":
                depth = 1
                k = j + 1
                while k < len(body) and depth > 0:
                    if body[k] == "{":
                        depth += 1
                    elif body[k] == "}":
                        depth -= 1
                    k += 1
                val = body[j + 1:k - 1]
            elif body[j] == '"':
                k = j + 1
                while k < len(body) and body[k] != '"':
                    k += 1
                val = body[j + 1:k]
            else:
                k = j
                while k < len(body) and body[k] not in ",\n":
                    k += 1
                val = body[j:k]
            fields[fname] = " ".join(val.split())
        entries.append((key, fields))
    return entries


def clean_braces(s):
    return s.replace("{", "").replace("}", "").replace("\\&", "&")


def format_authors(raw):
    if not raw:
        return ""
    raw = clean_braces(raw)
    authors = [a.strip() for a in re.split(r"\s+and\s+", raw) if a.strip()]
    out = []
    for a in authors:
        if "," in a:
            last, first = a.split(",", 1)
            out.append("%s %s" % (first.strip(), last.strip()))
        else:
            out.append(a)
    if len(out) > 6:
        return ", ".join(out[:6]) + " et al."
    if len(out) > 1:
        return ", ".join(out[:-1]) + " and " + out[-1]
    return out[0] if out else ""


def format_reference(fields):
    parts = []
    auth = format_authors(fields.get("author", ""))
    if auth:
        parts.append(auth + ",")
    title = clean_braces(fields.get("title", "")).strip()
    if title:
        parts.append('"%s,"' % title)
    venue = (fields.get("booktitle") or fields.get("journal")
             or fields.get("howpublished") or "")
    venue = clean_braces(venue).strip()
    venue = re.sub(r"\\url\s*", "", venue)
    if venue:
        if "github.com" in venue or venue.startswith("http"):
            parts.append("<i>%s</i>." % venue)
        else:
            parts.append("In <i>%s</i>." % venue if fields.get("booktitle")
                         else "<i>%s</i>." % venue)
    vol = fields.get("volume", "")
    if vol and fields.get("journal"):
        parts.append("vol. %s." % vol)
    year = fields.get("year", "")
    if year:
        parts.append("%s." % year)
    return " ".join(parts)


def references_flow(styles):
    entries = parse_bib(REFS_BIB)
    # main.tex uses ieeetr -> references numbered in citation order.
    # We approximate by bib file order (a reasonable, deterministic ordering).
    items = []
    for idx, (key, fields) in enumerate(entries, start=1):
        txt = "[%d] %s" % (idx, format_reference(fields))
        items.append(Paragraph(txt, styles["ref"]))
    return items


# --------------------------------------------------------------------------
# Build document
# --------------------------------------------------------------------------
def build():
    margin = 0.9 * inch
    doc = SimpleDocTemplate(
        OUT_PDF, pagesize=letter,
        leftMargin=margin, rightMargin=margin,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        title="Detection-Driven Layout Analysis and Template Generation "
              "for Korean Card-News",
        author="Anonymous Submission",
    )
    text_w = letter[0] - 2 * margin
    s = build_styles()
    story = []

    # ---- Title ----
    story.append(P("Detection-Driven Layout Analysis and Template Generation "
                   "for Korean Card-News, with a Sequential Analysis of "
                   "Multi-Page Decks", s, "title"))
    story.append(P("Anonymous Submission<br/>"
                   "Computer Vision &amp; Time-Series Project<br/>"
                   "<font face='Courier'>cardnews-cv</font>", s, "author"))

    # ---- Abstract ----
    story.append(P("Abstract", s, "abstract_head"))
    story.append(P(
        "<i>Card-news</i>&mdash;short, image-based slide decks widely used on "
        "Korean social media&mdash;combine photographic backgrounds with "
        "carefully placed Korean typography. We study the problem of (i) "
        "<b>analyzing</b> the layout of existing card-news and (ii) "
        "<b>generating</b> new, production-quality cards. We build a curated "
        "corpus of <b>687</b> Korean cards organized into <b>66 multi-page "
        "decks</b>, and bootstrap element annotations (title / body) with a "
        "high-recall OCR pipeline and line-to-block merging. A YOLOv8 detector "
        "fine-tuned on these pseudo-labels reaches <b>mAP@50-95 = 0.718</b> "
        "(mAP@50 = 0.854); transfer learning is decisive (scratch: 0.502). We "
        "then compare two layout-generation routes: a research baseline that "
        "adapts the DS-GAN of PosterLayout, and a <b>detection-driven template "
        "engine</b> that extracts real layouts and re-renders crisp vector text "
        "with a designer Korean typeface. The template engine is markedly more "
        "reliable and legible for deployment, where GAN-rendered text is "
        "unusable. Finally, treating each deck as an ordered sequence, we present "
        "a <b>time-series analysis</b> of layout dynamics across page position "
        "over 66 decks, revealing a consistent “cover vs. interior” "
        "structure (covers are darker, more saturated and less edge-dense). "
        "Code, data tooling, and figures are released.",
        s, "abstract_body"))

    # ---- 1 Introduction ----
    story.append(P("1. Introduction", s, "h1"))
    story.append(P(
        "Korean <i>card-news</i> (<i>kadeunyuseu</i>) are multi-page visual "
        "posts that compress an article into a sequence of designed slides. "
        "Producing them well requires content-aware placement of design "
        "elements&mdash;<b>title</b>, <b>body</b>, <b>logo</b>, "
        "<b>underlay</b>&mdash;so that text stays readable, avoids salient image "
        "regions, and follows a consistent visual rhythm across pages. This is "
        "exactly the <i>content-aware visual-textual layout</i> problem "
        "formalized by PosterLayout [1], but specialized to (a) the Korean "
        "script, for which most layout corpora provide no coverage, and (b) the "
        "multi-page deck structure unique to card-news.", s))
    story.append(P("We make the problem concrete and reproducible with three "
                   "contributions:", s))
    story.append(ListFlowable([
        ListItem(P("<b>A curated Korean card-news corpus</b> of 687 images in "
                   "66 ordered decks, with an automatic, high-recall labeling "
                   "pipeline (OCR + line-to-block merging) that yields "
                   "title/body boxes without manual annotation (Sec. 3).",
                   s, "listitem")),
        ListItem(P("<b>A layout system</b> with a fine-tuned YOLOv8 detector "
                   "and two generation routes&mdash;an adapted DS-GAN baseline "
                   "and a detection-driven <i>template engine</i>&mdash;which "
                   "we compare for real-world quality (Sec. 4, 6).",
                   s, "listitem")),
        ListItem(P("<b>A sequential (time-series) analysis</b> of deck "
                   "dynamics: we treat page index as a discrete time axis and "
                   "quantify how visual structure evolves from cover to closing "
                   "page (Sec. 5).", s, "listitem")),
    ], bulletType="1", leftIndent=18))

    # ---- 2 Related Work ----
    story.append(P("2. Related Work", s, "h1"))
    story.append(P(
        "<b>Content-aware layout generation.</b> LayoutGAN [8] and "
        "LayoutTransformer [9] generate element arrangements from noise or "
        "partial layouts. PosterLayout [1] conditions on a clean background and "
        "a saliency map and trains a domain-alignment GAN (DS-GAN) to place "
        "elements off salient regions; it is our generation baseline.", s))
    story.append(P(
        "<b>Object detection.</b> We use YOLOv8 [2], the modern successor of "
        "the YOLO family [10], pre-trained on COCO [12] and fine-tuned for card "
        "elements.", s))
    story.append(P(
        "<b>Text detection / OCR.</b> For bootstrap labels we use EasyOCR [3]; "
        "character-region methods such as CRAFT [7] are a higher-recall "
        "alternative for future work.", s))
    story.append(P(
        "<b>Saliency and inpainting.</b> The generation route uses "
        "salient-object detection (U²-Net [5], BASNet [6]) to find "
        "protected regions and large-mask inpainting (LaMa [4]) to erase "
        "existing text into clean backgrounds.", s))

    # ---- 3 Dataset ----
    story.append(P("3. Dataset", s, "h1"))
    story.append(P(
        "<b>Collection and standardization.</b> We assembled 687 Korean cards: "
        "109 seed references and 578 newly collected pages spanning "
        "public-sector, agriculture/smart-farm, and editorial themes. The 578 "
        "new pages are organized as 66 <i>decks</i> (ordered series of 4-11 "
        "pages). All images are re-encoded to RGB JPEG with ASCII filenames (so "
        "OpenCV/Ultralytics read them on any OS) and capped at 2048&nbsp;px on "
        "the long side; the generation pipeline additionally rescales to the "
        "513&times;750 PosterLayout canvas. Figure 1 shows samples.", s))
    story += fig("fig_dataset.png",
                 "Figure 1. Representative pages from the 687-image, 66-deck "
                 "Korean card-news corpus.", s, text_w)
    story.append(P(
        "<b>Automatic labeling.</b> Manual box annotation does not scale, so we "
        "generate <i>pseudo-labels</i>: EasyOCR [3] is run at high recall "
        "(canvas_size=2560, mag_ratio=2.0, low_text=0.3) to recover small body "
        "text; each text line is typed as <i>title</i> when its height exceeds "
        "4.5% of the image height, else <i>body</i>. Because layout generation "
        "needs element <i>regions</i> rather than individual lines, we merge "
        "neighboring lines of the same class into <b>blocks</b> via a union-find "
        "over boxes expanded by a fraction of their height. This yields 284 "
        "title and 641 body boxes on the seed set; logo/underlay remain for "
        "manual annotation.", s))

    # ---- 4 Method ----
    story.append(P("4. Method", s, "h1"))
    story.append(P(
        "Figure 2 summarizes the system: collect → detect → {template "
        "engine | DS-GAN} → render.", s))
    story += fig("fig_pipeline.png",
                 "Figure 2. System overview. A shared detector feeds two "
                 "generation routes; the template engine (3a) is our deployed "
                 "path, DS-GAN (3b) is the research baseline.", s, text_w)
    story.append(P("4.1. Element detection", s, "h2"))
    story.append(P(
        "We fine-tune YOLOv8-nano (and -small) from COCO weights on the "
        "pseudo-labels. Horizontal flipping is disabled (fliplr=0) because it "
        "mirrors Hangul; we use a card-tuned augmentation set (mild mosaic, HSV "
        "jitter, small scale/translate, random erasing). The detector both (a) "
        "supplies masks for inpainting and the train_csv used by DS-GAN, and (b) "
        "<i>extracts the layout templates</i> described next.", s))
    story.append(P("4.2. Layout generation: two routes", s, "h2"))
    story.append(P(
        "<b>(3b) DS-GAN baseline.</b> Following PosterLayout [1], we construct "
        "the required Dataset/ (inpainted posters, dual saliency maps, "
        "train_csv of normalized element boxes) using U²-Net/BASNet "
        "saliency and LaMa inpainting, and fine-tune DS-GAN to generate (class, "
        "box) layouts on clean Korean backgrounds. While the model produces "
        "plausible boxes, its <i>rendered</i> output is not deployable: it "
        "cannot place legible Korean glyphs and layout stability varies "
        "run-to-run.", s))
    story.append(P(
        "<b>(3a) Detection-driven template engine (ours).</b> We instead "
        "<i>copy and re-render</i> real layouts. The detector localizes "
        "title/body blocks on a reference card; classical inpainting (or LaMa) "
        "removes the original text; new content is typeset back into the same "
        "block geometry with the Pretendard [11] typeface. We additionally "
        "maintain a small library of layout archetypes (editorial, centered, "
        "bottom-panel, cover) distilled from the corpus, and a saliency-aware "
        "placement step that selects the calmest image band (lowest mean "
        "gradient magnitude) for text and applies a luminance-adaptive scrim so "
        "the result is legible over <i>any</i> background.", s))
    story.append(P("4.3. Rendering", s, "h2"))
    story.append(P(
        "Cards are composited at 1080&times;1350 (Instagram 4:5). Text color is "
        "chosen from the background luminance under each block (white on dark, "
        "near-black on light), font size is auto-fit per box, and a soft rounded "
        "scrim guarantees contrast and covers residual texture. Typography uses "
        "Pretendard Bold for titles and Regular for body, with a fixed spacing "
        "system and accent rules.", s))

    # ---- 5 Sequential Analysis ----
    story.append(P("5. Sequential Analysis of Decks", s, "h1"))
    story.append(P(
        "A card-news deck is an ordered sequence of pages; we treat the page "
        "index as a discrete time axis and ask how layout/visual structure "
        "evolves along it. Over the 66 decks (≥ 4 pages each) we compute, "
        "per page, four cv2 features&mdash;Canny <i>edge density</i> "
        "(text/graphic density), <i>brightness</i>, Hasler-Süsstrunk "
        "<i>colorfulness</i> [13], and HSV <i>saturation</i>&mdash;and average "
        "them into ten relative-position bins (0=cover, 1=last).", s))
    story.append(P(
        "Figure 3 shows a consistent <b>cover-vs-interior</b> pattern: covers "
        "are notably <i>darker</i> (brightness 0.62 vs. ≈0.71 for "
        "interiors), more <i>saturated</i> (0.33 vs. ≈0.27) and more "
        "<i>colorful</i>, and have the <i>lowest edge density</i> (0.097). "
        "Interior pages are brighter and denser (more text), and the final page "
        "partly reverts toward the cover's saturated tone. This quantifies a "
        "design convention&mdash;an eye-catching, high-chroma cover followed by "
        "readable content pages&mdash;and motivates <i>position-conditioned</i> "
        "templates (Sec. 4): the engine selects a cover archetype for page 1 and "
        "content archetypes thereafter.", s))
    story += fig("fig_timeseries.png",
                 "Figure 3. Deck dynamics across relative page position (66 "
                 "decks). Covers are darker, more saturated/colorful and less "
                 "edge-dense than interior pages.", s, text_w)

    # ---- 6 Experiments ----
    story.append(P("6. Experiments", s, "h1"))
    story.append(P(
        "<b>Setup.</b> YOLOv8 fine-tuning, 640&nbsp;px, 150-300 epochs, batch "
        "tuned to GPU; metrics are box mAP. The seed split is 93 train / 16 val. "
        "Cloud training uses an RTX 4090; development/inference run on an RTX "
        "3050.", s))
    story.append(P("6.1. Detector ablation", s, "h2"))
    story.append(P(
        "Table 1 and Figure 4 report the ablation. Transfer learning is the "
        "dominant factor (scratch 0.502 vs. baseline 0.662). A card-tuned "
        "augmentation and a longer schedule give the best nano model "
        "(long300_card: mAP@50-95 = 0.718, mAP@50 = 0.854); YOLOv8-small is "
        "competitive (0.705). Seed repeats vary by &lt;0.02 mAP, indicating "
        "stable ranking. Qualitative detections are shown in Figure 5.", s))
    story.append(ablation_table(s, text_w))
    story += fig("fig_ablation.png",
                 "Figure 4. Key ablation results (box mAP@50-95). Transfer "
                 "learning and a card-tuned, longer schedule drive the gains.",
                 s, text_w)
    story += fig("fig_detector_pred.jpg",
                 "Figure 5. Detector predictions on held-out validation cards "
                 "(title/body).", s, text_w)
    story.append(P(
        "<b>Robustness (5-fold CV).</b> Five-fold cross-validation of the "
        "baseline gives mAP@50-95 = 0.611 ± 0.049, confirming the "
        "small-data result is not split-specific.", s))
    story.append(P(
        "<b>Data scale (109 vs. 687).</b> To measure the effect of the 578 new "
        "images we adopt a <i>leak-free</i> protocol: a common test set (15% of "
        "the new images) is held out and never trained on, and the best recipe "
        "is trained on (i) the 109 seed images and (ii) the full 687, with both "
        "evaluated on the identical test set. The tooling is released; cloud "
        "training of this comparison is in progress and we report it as a "
        "protocol rather than a completed result, to avoid drawing conclusions "
        "from unfinished runs.", s))
    story.append(P("6.2. Generation quality", s, "h2"))
    story.append(P(
        "Figure 6 shows template-engine outputs and a copy-and-re-render "
        "example (original → cleaned background → recomposed). The "
        "template engine yields legible, consistently designed Korean cards at "
        "1080&times;1350, whereas the DS-GAN route produces usable <i>boxes</i> "
        "but not deployable rendered text. For a production card-news service we "
        "therefore adopt the template engine and retain DS-GAN as a research "
        "comparison.", s))
    story += fig("fig_results_template.png",
                 "Figure 6a. Template-engine cards (high-res, Pretendard, "
                 "adaptive color).", s, text_w, max_h=3.2 * inch)
    story += fig("fig_results_copy.png",
                 "Figure 6b. Layout copy: original | inpainted background | "
                 "recomposed, with new text in the analyzed layout.", s,
                 text_w, max_h=3.2 * inch)

    # ---- 7 Discussion ----
    story.append(P("7. Discussion and Limitations", s, "h1"))
    story.append(P(
        "Pseudo-labels cap detector quality: EasyOCR misses very small or "
        "stylized text, so label recall&mdash;not data volume&mdash;is the "
        "binding constraint, addressed here by high-recall OCR settings and "
        "block merging, and in future by CRAFT [7] or manual logo/underlay "
        "annotation. The DS-GAN route is limited by Korean glyph rendering and "
        "small-data instability. The sequential analysis uses low-level proxies; "
        "detector-derived element statistics per page are a natural extension.",
        s))

    # ---- 8 Conclusion ----
    story.append(P("8. Conclusion", s, "h1"))
    story.append(P(
        "We presented a reproducible pipeline for Korean card-news layout: a "
        "curated 687-image, 66-deck corpus with automatic block labels, a "
        "fine-tuned YOLOv8 detector (mAP@50-95 = 0.718), two generation routes "
        "whose comparison favors a detection-driven template engine for "
        "deployment, and a time-series analysis that reveals a quantifiable "
        "cover-vs-interior design convention. The artifacts provide a basis for "
        "a practical Korean card-news generation service.", s))

    # ---- References ----
    story.append(P("References", s, "h1"))
    story += references_flow(s)

    doc.build(story)


if __name__ == "__main__":
    build()
    size = os.path.getsize(OUT_PDF)
    print("Wrote %s (%d bytes, %.1f KB)" % (OUT_PDF, size, size / 1024.0))
