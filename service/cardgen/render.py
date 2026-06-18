"""Card-news rendering engine v2: generate_card / generate_deck.

Two rendering paths:
  * Rich themed cards (no photo) — dark/branded background + a consistent
    scattered deco layer + two-tone glowing title + keyword-highlighted
    subtitle + an optional white rounded checklist panel + brand mark. This is
    the look distilled from real gov/agri card-news (e.g. the EPIS deck) and is
    what makes output read as designer-made rather than "AI-tic".
  * Photo cards — user image with saliency-light placement, adaptive text color
    and a scrim (kept from v1; good enough for photo backgrounds).

Deterministic, CPU-only (Pillow). A deck shares one theme + one deco seed so the
set feels like a single designed series.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import themes as TH
from . import templates as TP
from . import components as C
from .assets import font
from .placement import region_luma

SIZE = (1080, 1350)


# ---------- background ----------
def gradient(size, c0, c1, angle=90):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = np.deg2rad(angle)
    proj = xx * np.cos(rad) + yy * np.sin(rad)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    a = np.array(c0, np.float32)[None, None] * (1 - proj)[..., None] \
        + np.array(c1, np.float32)[None, None] * proj[..., None]
    return Image.fromarray(a.astype("uint8"), "RGB")


def cover_fit(img, size):
    w, h = size
    iw, ih = img.size
    s = max(w / iw, h / ih)
    img = img.resize((max(1, int(iw * s)), max(1, int(ih * s))), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def _vignette(img, strength=0.55):
    """Soft radial darkening toward the corners — adds depth to flat fills."""
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    m = np.clip((d - 0.6) / 0.9, 0, 1) * strength
    layer = Image.fromarray((m * 255).astype("uint8"), "L")
    black = Image.new("RGBA", img.size, (0, 0, 0, 0))
    black.putalpha(layer)
    img.alpha_composite(black)


def themed_background(size, theme, seed):
    base = gradient(size, theme["bg"][0], theme["bg"][-1],
                    theme.get("angle", 90)).convert("RGBA")
    if theme.get("dark", True):
        _vignette(base)
    C.scatter_deco(base, theme, seed,
                   alpha=34 if theme.get("dark", True) else 26)
    return base


# ---------- photo path (kept from v1) ----------
def _wrap(draw, text, fnt, max_w):
    out = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            t = cur + ch
            if draw.textlength(t, font=fnt) <= max_w or not cur:
                cur = t
            else:
                out.append(cur); cur = ch
        out.append(cur)
    return out


def _fit(draw, text, bw, bh, weight, start, min_px=18, lh=1.16):
    size = max(min_px, int(bh * start))
    while size > min_px:
        fnt = font(size, weight)
        lines = _wrap(draw, text, fnt, bw)
        asc, desc = fnt.getmetrics()
        line_h = (asc + desc) * lh
        if line_h * len(lines) <= bh and all(
                draw.textlength(l, font=fnt) <= bw for l in lines):
            return fnt, lines, line_h
        size = int(size * 0.93)
    fnt = font(min_px, weight)
    asc, desc = fnt.getmetrics()
    return fnt, _wrap(draw, text, fnt, bw), (asc + desc) * lh


def _scrim(img, box_px, dark, pad=0.12, radius=26, alpha=140):
    x0, y0, x1, y1 = box_px
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    col = (0, 0, 0, alpha) if dark else (255, 255, 255, int(alpha * 1.3))
    ImageDraw.Draw(layer).rounded_rectangle(
        [x0 - pw, y0 - ph, x1 + pw, y1 + ph], radius=radius, fill=col)
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(3)))


def _photo_text(img, box_px, text, *, weight, color, align, start, lh=1.16,
                scrim=None):
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box_px
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    if scrim is not None:
        _scrim(img, box_px, dark=scrim)
        draw = ImageDraw.Draw(img)
    fnt, lines, line_h = _fit(draw, text, bw, bh, weight, start, lh=lh)
    total = line_h * len(lines)
    y = y0 + (bh - total) / 2
    for ln in lines:
        tw = draw.textlength(ln, font=fnt)
        x = {"left": x0, "center": x0 + (bw - tw) / 2, "right": x1 - tw}[align]
        draw.text((x, y), ln, font=fnt, fill=color)
        y += line_h


def _render_photo(title, body, background, theme, eyebrow, brand, page, total,
                  size):
    th = theme
    bg = background if isinstance(background, Image.Image) \
        else Image.open(background)
    base = cover_fit(bg.convert("RGB"), size).convert("RGBA")
    W, H = size
    slots = TP.get("bottom")
    content = {"eyebrow": eyebrow, "title": title, "body": body, "logo": brand}
    for role, x0, y0, x1, y1, align in slots:
        box = (x0 * W, y0 * H, x1 * W, y1 * H)
        text = content.get(role)
        if not text:
            continue
        dark = region_luma(base.convert("RGB"), box) < 130
        fg = (255, 255, 255) if dark else (24, 24, 24)
        if role == "title":
            _photo_text(base, box, text, weight="Bold", color=fg, align=align,
                        start=0.62, lh=1.12, scrim=dark)
        elif role == "body":
            _photo_text(base, box, text, weight="Regular", color=fg,
                        align=align, start=0.30, lh=1.45, scrim=dark)
        elif role == "logo":
            C.brand_mark(base, box, text, th, align="right")
    if page and total:
        _page_badge(base, page, total, (255, 255, 255))
    return base.convert("RGB")


def _page_badge(img, idx, total, color):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    fnt = font(26, "SemiBold")
    t = f"{idx:02d} / {total:02d}"
    tw = draw.textlength(t, font=fnt)
    draw.text((w - tw - 46, h - 60), t, font=fnt, fill=color)


# ---------- rich themed path ----------
def _eyebrow_label(img, x_center, y, text, theme):
    """Small uppercase label with leading dot (e.g. a 'NEW' kicker)."""
    draw = ImageDraw.Draw(img)
    col = theme.get("title_accent", theme["accent"])
    fnt = font(max(20, int(img.size[0] * 0.026)), "Bold")
    tw = draw.textlength(text, font=fnt)
    dot = fnt.size * 0.5
    total = dot * 1.5 + tw
    x = x_center - total / 2
    asc, desc = fnt.getmetrics()
    cy = y + (asc + desc) / 2
    draw.ellipse([x, cy - dot / 2, x + dot, cy + dot / 2], fill=col + (255,))
    draw.text((x + dot * 1.5, y), text, font=fnt, fill=col + (255,))


def _render_rich(title, subtitle, checklist, theme, kind, eyebrow, brand,
                 mascot, page, total, size, seed):
    W, H = size
    base = themed_background(size, theme, seed)

    # brand mark, top-right
    if brand:
        C.brand_mark(base, (0.60 * W, 0.045 * H, 0.95 * W, 0.095 * H),
                     brand, theme, align="right")

    if kind == "cover":
        if mascot:
            C.speech_bubble(base, (0.80 * W, 0.20 * H), 0.10 * W, theme,
                            mascot=mascot)
        if eyebrow:
            _eyebrow_label(base, W / 2, 0.345 * H, eyebrow, theme)
        bottom = C.glow_title(base, (0.06 * W, 0.40 * H, 0.94 * W, 0.62 * H),
                              title, theme, start_frac=0.5)
        if subtitle:
            C.subtitle(base, (0.14 * W, bottom + 0.025 * H, 0.86 * W,
                              bottom + 0.13 * H),
                       subtitle, theme, base_color=theme["title"],
                       start_frac=0.6)

    elif kind == "checklist":
        bottom = C.glow_title(base, (0.06 * W, 0.07 * H, 0.94 * W, 0.27 * H),
                              title, theme, start_frac=0.5)
        if subtitle:
            bottom = C.subtitle(
                base, (0.10 * W, bottom + 0.015 * H, 0.90 * W, 0.42 * H),
                subtitle, theme, start_frac=0.22)
        top = max(bottom + 0.03 * H, 0.46 * H)
        C.checklist_panel(base, (0.08 * W, top, 0.92 * W, 0.92 * H),
                          checklist, theme)

    else:  # statement: title + subtitle, centered
        bottom = C.glow_title(base, (0.06 * W, 0.24 * H, 0.94 * W, 0.50 * H),
                              title, theme, start_frac=0.5)
        if subtitle:
            C.subtitle(base, (0.12 * W, bottom + 0.03 * H, 0.88 * W, 0.80 * H),
                       subtitle, theme, start_frac=0.22)

    if page and total:
        _page_badge(base, page, total, theme.get("body", (255, 255, 255)))
    return base.convert("RGB")


# ---------- main API ----------
def generate_card(title, body=None, *, subtitle=None, checklist=None,
                  background=None, theme="auto", kind="auto", eyebrow=None,
                  brand="BRAND", logo_text=None, mascot=None, page=None,
                  total=None, size=SIZE, seed=0):
    """Render one card-news image -> PIL RGB.

    title     : may contain "\\n"; even lines render white, odd lines accent.
    subtitle  : supports *keyword* markup (accent-highlighted runs).
    checklist : list[str] -> white rounded panel with accent checks.
    background: a photo path/Image routes through the photo renderer.
    kind      : auto|cover|checklist|statement (auto picks from content).
    """
    th = TH.pick(seed) if theme in (None, "auto") else TH.get(theme)
    brand = logo_text if logo_text is not None else brand  # back-compat
    sub = subtitle if subtitle is not None else body

    if background is not None:
        return _render_photo(title, sub, background, th, eyebrow, brand,
                             page, total, size)

    if kind in (None, "auto"):
        if checklist:
            kind = "checklist"
        elif not sub:
            kind = "cover"
        else:
            kind = "statement"
    return _render_rich(title, sub, checklist, th, kind, eyebrow, brand,
                        mascot, page, total, size, seed)


def generate_deck(items, *, theme="auto", brand="BRAND", logo_text=None,
                  background=None, mascot=None, size=SIZE, seed=0):
    """items: list of dicts {title, subtitle?/body?, checklist?, eyebrow?}.
    First item -> cover. One theme + one deco seed across the deck."""
    th_name = TH.ORDER[seed % len(TH.ORDER)] if theme in (None, "auto") \
        else theme
    brand = logo_text if logo_text is not None else brand
    n = len(items)
    cards = []
    for i, it in enumerate(items):
        kind = "cover" if i == 0 else "auto"
        cards.append(generate_card(
            it.get("title", ""), it.get("body"),
            subtitle=it.get("subtitle"), checklist=it.get("checklist"),
            background=background, theme=th_name, kind=kind,
            eyebrow=it.get("eyebrow"), brand=brand,
            mascot=(mascot if i == 0 else None),
            page=(None if i == 0 else i + 1), total=(None if i == 0 else n),
            size=size, seed=seed))
    return cards
