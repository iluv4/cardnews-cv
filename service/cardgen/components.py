"""Reusable visual components for the v2 card-news engine.

These are the building blocks distilled from real Korean gov/agri card-news
(e.g. the EPIS tomato-education deck): a dark themed background with a faint
scattered deco layer, a two-tone *glowing* title, a keyword-highlighted
subtitle, and a white rounded panel with an accent checklist + dotted dividers.

Everything is deterministic and CPU-only (Pillow). Geometry is in pixels so the
caller controls layout; colors come from a theme dict (see themes.py).
"""
import math
import random

from PIL import Image, ImageDraw, ImageFilter

from .assets import font


# ---------- low-level text fitting ----------
def _wrap_text(draw, text, fnt, max_w):
    out = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            t = cur + ch
            if draw.textlength(t, font=fnt) <= max_w or not cur:
                cur = t
            else:
                out.append(cur)
                cur = ch
        out.append(cur)
    return out


def _line_h(fnt, lh):
    asc, desc = fnt.getmetrics()
    return (asc + desc) * lh


def fit_lines(draw, lines, box_w, box_h, weight, start_frac, *, min_px=20,
              lh=1.14):
    """Pick the largest font (<= box) that fits the given pre-split lines."""
    size = max(min_px, int(box_h * start_frac))
    while size > min_px:
        fnt = font(size, weight)
        if (_line_h(fnt, lh) * len(lines) <= box_h
                and all(draw.textlength(l, font=fnt) <= box_w for l in lines)):
            return fnt
        size = int(size * 0.94)
    return font(min_px, weight)


# ---------- glowing two-tone title ----------
def glow_title(img, box_px, lines, theme, *, weight="Bold", lh=1.16,
               start_frac=0.42, align="center"):
    """Draw title lines centered in box; even lines use theme title color,
    odd lines use theme title_accent. Each line gets a soft same-color glow.

    `lines` may be a list[str] or a single "\\n"-joined string.
    Returns the y of the bottom of the drawn block (for stacking subtitles)."""
    if isinstance(lines, str):
        lines = [l for l in lines.split("\n")]
    x0, y0, x1, y1 = box_px
    bw, bh = x1 - x0, y1 - y0
    draw = ImageDraw.Draw(img)
    fnt = fit_lines(draw, lines, bw, bh, weight, start_frac, lh=lh)
    line_h = _line_h(fnt, lh)
    total = line_h * len(lines)
    y = y0 + (bh - total) / 2

    title = theme["title"]
    accent = theme.get("title_accent", theme["accent"])
    light_title = sum(title) > 600  # white-ish title on dark bg
    bottoms = []
    for i, ln in enumerate(lines):
        is_accent = i % 2 == 1
        col = accent if is_accent else title
        tw = draw.textlength(ln, font=fnt)
        x = {"center": x0 + (bw - tw) / 2, "left": x0,
             "right": x1 - tw}[align]
        # Per-line glow. A heavy blur of light text on a dark bg smears into a
        # grey rectangle, so white/title lines get a tight, faint halo while
        # accent lines get a stronger coloured one.
        if is_accent:
            g_alpha, g_r, passes = 165, fnt.size * 0.13, 2
        elif light_title:
            g_alpha, g_r, passes = 70, fnt.size * 0.07, 1
        else:                       # dark title on light bg: skip glow
            g_alpha, g_r, passes = 0, 0, 0
        for _ in range(passes):
            glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(glow).text((x, y), ln, font=fnt,
                                      fill=col + (g_alpha,))
            img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(g_r)))
        draw.text((x, y), ln, font=fnt, fill=col + (255,))
        bottoms.append(y + line_h)
        y += line_h
    return bottoms[-1] if bottoms else y0


# ---------- keyword-highlighted subtitle ----------
def _runs(text):
    """Split `text` on *...* markers into (segment, is_highlight) runs."""
    runs, buf, hot = [], "", False
    for ch in text:
        if ch == "*":
            if buf:
                runs.append((buf, hot))
                buf = ""
            hot = not hot
        else:
            buf += ch
    if buf:
        runs.append((buf, hot))
    return runs


def subtitle(img, box_px, text, theme, *, weight="SemiBold", lh=1.34,
             start_frac=0.30, align="center", base_color=None):
    """Wrapped subtitle with *keyword* runs drawn in the accent color.

    Wrapping is char-level (Korean-friendly) across the mixed-color runs."""
    x0, y0, x1, y1 = box_px
    bw, bh = x1 - x0, y1 - y0
    draw = ImageDraw.Draw(img)
    base = base_color or theme.get("body", (235, 235, 235))
    hot = theme.get("title_accent", theme["accent"])
    runs = _runs(text)

    # flatten to (char, color) then greedily wrap into lines
    chars = []
    for seg, is_hot in runs:
        for ch in seg:
            chars.append((ch, hot if is_hot else base))

    # find a font size where the wrapped block fits the box height + width
    size = max(16, int(bh * start_frac))
    while size > 16:
        fnt = font(size, weight)
        lines = _greedy_wrap(draw, chars, fnt, bw)
        if _line_h(fnt, lh) * len(lines) <= bh:
            break
        size = int(size * 0.93)
    fnt = font(size, weight)
    lines = _greedy_wrap(draw, chars, fnt, bw)
    line_h = _line_h(fnt, lh)
    total = line_h * len(lines)
    y = y0 + (bh - total) / 2
    for line in lines:
        tw = _line_width(draw, line, fnt)
        x = {"center": x0 + (bw - tw) / 2, "left": x0, "right": x1 - tw}[align]
        for ch, col in line:
            draw.text((x, y), ch, font=fnt, fill=col)
            x += draw.textlength(ch, font=fnt)
        y += line_h
    return y


def _greedy_wrap(draw, chars, fnt, max_w):
    """Word-aware wrap of a (char, color) stream. Breaks at spaces (Korean
    eojeol) so words stay whole; only char-breaks a word too long to fit."""
    # group into words (each carries its trailing space); '\n' forces a break
    words, cur = [], []
    for ch, col in chars:
        if ch == "\n":
            if cur:
                words.append(cur); cur = []
            words.append("BR")
        else:
            cur.append((ch, col))
            if ch == " ":
                words.append(cur); cur = []
    if cur:
        words.append(cur)

    def wlen(word):
        return sum(draw.textlength(c, font=fnt) for c, _ in word)

    lines, line, w = [], [], 0.0
    for word in words:
        if word == "BR":
            lines.append(line); line, w = [], 0.0
            continue
        ww = wlen(word)
        if w + ww > max_w and line:
            lines.append(line); line, w = [], 0.0
        if ww > max_w:                      # single word wider than the box
            for c, col in word:
                cw = draw.textlength(c, font=fnt)
                if w + cw > max_w and line:
                    lines.append(line); line, w = [], 0.0
                line.append((c, col)); w += cw
        else:
            line.extend(word); w += ww
    if line:
        lines.append(line)
    return lines


def _line_width(draw, line, fnt):
    """Visible width of a wrapped line, ignoring trailing spaces."""
    end = len(line)
    while end and line[end - 1][0] == " ":
        end -= 1
    return sum(draw.textlength(c, font=fnt) for c, _ in line[:end])


# ---------- white rounded checklist panel ----------
def checklist_panel(img, box_px, items, theme, *, radius=34, pad=0.07):
    """White rounded panel with accent check marks, bold dark items, and
    faint dotted dividers between rows. `items` is a list[str]."""
    x0, y0, x1, y1 = box_px
    draw = ImageDraw.Draw(img)
    panel = theme.get("panel", (255, 255, 255))
    text_col = theme.get("panel_text", (28, 28, 30))
    check = theme.get("check", theme["accent"])

    # soft drop shadow then the panel
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([x0, y0 + 10, x1, y1 + 10], radius=radius,
                         fill=(0, 0, 0, 60))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=panel + (255,))

    n = max(1, len(items))
    px = (x1 - x0) * pad
    inner_x0, inner_x1 = x0 + px, x1 - px
    row_h = (y1 - y0 - 2 * px) / n
    check_box = min(row_h * 0.32, (inner_x1 - inner_x0) * 0.075)
    text_x = inner_x0 + check_box + (x1 - x0) * 0.04
    text_w = inner_x1 - text_x

    # size the font so the longest item fits the row on at most two lines
    longest = max(items, key=len) if items else ""
    fnt = font(int(row_h * 0.34), "Bold")
    while fnt.size > 20:
        wrapped = _wrap_words(draw, longest, fnt, text_w)
        if len(wrapped) <= 2 and _line_h(fnt, 1.12) * len(wrapped) <= row_h * 0.82:
            break
        fnt = font(int(fnt.size * 0.94), "Bold")

    tint = tuple(int(c * 0.16 + 255 * 0.84) for c in check)  # opaque light box
    for i, item in enumerate(items):
        cy = y0 + px + row_h * (i + 0.5)
        # check box: light-tinted fill + accent outline + bold accent check
        bx0, by0 = inner_x0, cy - check_box / 2
        box = [bx0, by0, bx0 + check_box, by0 + check_box]
        draw.rounded_rectangle(box, radius=check_box * 0.26,
                               fill=tint + (255,), outline=check + (255,),
                               width=max(2, int(check_box * 0.10)))
        _check_mark(draw, bx0, by0, check_box, check)
        # item text (word-wrapped + balanced, vertically centered in its row)
        lines = _balance(_wrap_words(draw, item, fnt, text_w), draw, fnt,
                         text_w)
        lh_px = _line_h(fnt, 1.12)
        ty = cy - lh_px * len(lines) / 2
        for ln in lines:
            draw.text((text_x, ty), ln, font=fnt, fill=text_col)
            ty += lh_px
        if i < len(items) - 1:
            _dotted_line(draw, inner_x0, y0 + px + row_h * (i + 1),
                         inner_x1, theme.get("panel_sub", (170, 175, 180)))


def _wrap_words(draw, text, fnt, max_w):
    """Word-aware wrap of plain text (breaks at spaces; char-breaks only a word
    that is itself wider than the box)."""
    lines, cur = [], ""
    for para in text.split("\n"):
        for word in para.split(" "):
            cand = (cur + " " + word).strip() if cur else word
            if draw.textlength(cand, font=fnt) <= max_w or not cur:
                cur = cand
            else:
                lines.append(cur); cur = word
        lines.append(cur); cur = ""
    return [l for l in lines if l != ""] or [""]


def _check_mark(draw, bx0, by0, s, color):
    pts = [(bx0 + s * 0.24, by0 + s * 0.52),
           (bx0 + s * 0.43, by0 + s * 0.72),
           (bx0 + s * 0.78, by0 + s * 0.30)]
    draw.line(pts, fill=color + (255,), width=max(3, int(s * 0.15)),
              joint="curve")


def _balance(lines, draw, fnt, max_w):
    """If a wrap produced exactly two lines, re-split at the word boundary that
    most evens their widths — avoids a lone trailing syllable like '가능'.
    Never accepts a split where a line exceeds max_w."""
    if len(lines) != 2:
        return lines
    words = (lines[0] + " " + lines[1]).split(" ")
    if len(words) < 3:
        return lines
    best, best_diff = None, None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        wa = draw.textlength(a, font=fnt)
        wb = draw.textlength(b, font=fnt)
        if wa > max_w or wb > max_w:
            continue
        diff = abs(wa - wb)
        if best_diff is None or diff < best_diff:
            best, best_diff = (a, b), diff
    return [best[0], best[1]] if best else lines


def _dotted_line(draw, x0, y, x1, color, gap=14, r=2):
    x = x0
    while x < x1:
        draw.ellipse([x, y - r, x + r, y + r], fill=color + (180,))
        x += gap


# ---------- brand mark (top-right) ----------
def brand_mark(img, box_px, text, theme, *, align="right"):
    x0, y0, x1, y1 = box_px
    draw = ImageDraw.Draw(img)
    col = theme.get("brand", theme["title"])
    fnt = fit_lines(draw, [text], x1 - x0, y1 - y0, "Bold", 0.9, min_px=18)
    tw = draw.textlength(text, font=fnt)
    x = {"right": x1 - tw, "left": x0, "center": x0 + (x1 - x0 - tw) / 2}[align]
    draw.text((x, y0), text, font=fnt, fill=col + (255,))


# ---------- scattered deco layer (consistent per deck) ----------
def _ic_sparkle(d, cx, cy, s, col, w):
    for a in (0, 90):
        rad = math.radians(a)
        dx, dy = math.cos(rad) * s, math.sin(rad) * s
        d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=col, width=w)
    for a in (45, 135):
        rad = math.radians(a)
        dx, dy = math.cos(rad) * s * 0.6, math.sin(rad) * s * 0.6
        d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=col, width=w)


def _ic_ring(d, cx, cy, s, col, w):
    d.ellipse([cx - s, cy - s, cx + s, cy + s], outline=col, width=w)


def _ic_wave(d, cx, cy, s, col, w):
    n, step = 4, s * 0.5
    pts = []
    for i in range(n * 2 + 1):
        x = cx - s + i * step / 2
        y = cy + (-1) ** i * step * 0.4
        pts.append((x, y))
    d.line(pts, fill=col, width=w, joint="curve")


def _ic_magnifier(d, cx, cy, s, col, w):
    r = s * 0.7
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=w)
    d.line([(cx + r * 0.7, cy + r * 0.7), (cx + s, cy + s)], fill=col, width=w)


def _ic_plus(d, cx, cy, s, col, w):
    d.line([(cx - s, cy), (cx + s, cy)], fill=col, width=w)
    d.line([(cx, cy - s), (cx, cy + s)], fill=col, width=w)


def _ic_dots(d, cx, cy, s, col, w):
    for ox in (-s, 0, s):
        d.ellipse([cx + ox - w, cy - w, cx + ox + w, cy + w], fill=col)


_ICONS = [_ic_sparkle, _ic_ring, _ic_wave, _ic_magnifier, _ic_plus, _ic_dots]

# fixed scatter positions (normalized), kept off the central content column
_SPOTS = [
    (0.08, 0.10), (0.90, 0.07), (0.06, 0.32), (0.94, 0.30),
    (0.10, 0.55), (0.91, 0.52), (0.07, 0.78), (0.93, 0.80),
    (0.20, 0.92), (0.80, 0.93), (0.16, 0.07), (0.86, 0.93),
]


def scatter_deco(img, theme, seed=0, *, alpha=34, scale=1.0):
    """Faint, consistent deco icons around the edges. Same seed -> same layout
    across a deck, which is what makes a real card-news set feel designed."""
    w, h = img.size
    col = theme.get("deco", (255, 255, 255)) + (alpha,)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rng = random.Random(seed * 7919 + 17)
    lw = max(2, int(w * 0.004))
    for i, (nx, ny) in enumerate(_SPOTS):
        icon = _ICONS[(i + seed) % len(_ICONS)]
        jx = (rng.random() - 0.5) * 0.03
        jy = (rng.random() - 0.5) * 0.03
        s = w * 0.022 * scale * (0.8 + rng.random() * 0.6)
        icon(d, (nx + jx) * w, (ny + jy) * h, s, col, lw)
    img.alpha_composite(layer)


# ---------- speech bubble + mascot (optional cover accent) ----------
def speech_bubble(img, center, r, theme, *, mascot=None):
    """White circular speech bubble with a little tail. `mascot` may be
    'tomato' (a drawn vector mascot) or a path to a PNG (pasted, fit inside)."""
    cx, cy = center
    draw = ImageDraw.Draw(img)
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([cx - r, cy - r, cx + r, cy + r],
                                 fill=(255, 255, 255, 90))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(int(r * 0.4))))
    tail = [(cx - r * 0.5, cy + r * 0.5), (cx - r * 0.05, cy + r * 1.18),
            (cx + r * 0.02, cy + r * 0.45)]
    draw.polygon(tail, fill=(255, 255, 255, 255))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 255))
    if mascot == "tomato":
        tomato(img, (cx, cy), r * 0.62, theme)
    elif isinstance(mascot, str) and mascot:
        try:
            m = Image.open(mascot).convert("RGBA")
            s = int(r * 1.3)
            m.thumbnail((s, s), Image.LANCZOS)
            img.alpha_composite(m, (int(cx - m.width / 2),
                                    int(cy - m.height / 2)))
        except Exception:
            pass


def tomato(img, center, r, theme):
    """A small, clean vector tomato mascot (body + highlight + leaf + face)."""
    cx, cy = center
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    red, red_dk = (228, 62, 56), (196, 40, 44)
    grn, grn_dk = (74, 168, 86), (54, 138, 70)
    # body: slightly wide ellipse + a darker base for roundness
    d.ellipse([cx - r, cy - r * 0.92, cx + r, cy + r], fill=red + (255,))
    d.ellipse([cx - r, cy - r * 0.55, cx + r, cy + r], fill=red_dk + (90,))
    d.ellipse([cx - r, cy - r * 0.92, cx + r, cy + r * 0.85], fill=red + (255,))
    # glossy highlight
    d.ellipse([cx - r * 0.55, cy - r * 0.6, cx - r * 0.05, cy - r * 0.15],
              fill=(255, 255, 255, 120))
    # leaf: a small five-point green star/calyx on top
    pts = []
    for k in range(5):
        a = math.radians(-90 + k * 72)
        pts.append((cx + math.cos(a) * r * 0.5,
                    cy - r * 0.78 + math.sin(a) * r * 0.5))
        a2 = math.radians(-90 + k * 72 + 36)
        pts.append((cx + math.cos(a2) * r * 0.22,
                    cy - r * 0.78 + math.sin(a2) * r * 0.22))
    d.polygon(pts, fill=grn + (255,))
    d.ellipse([cx - r * 0.12, cy - r * 0.95, cx + r * 0.12, cy - r * 0.7],
              fill=grn_dk + (255,))
    # simple happy face
    eye = r * 0.12
    for ex in (-r * 0.32, r * 0.32):
        d.ellipse([cx + ex - eye, cy - eye, cx + ex + eye, cy + eye],
                  fill=(40, 24, 24, 255))
    d.arc([cx - r * 0.3, cy + r * 0.0, cx + r * 0.3, cy + r * 0.45],
          start=15, end=165, fill=(40, 24, 24, 255), width=max(2, int(r * 0.09)))
    img.alpha_composite(layer)
