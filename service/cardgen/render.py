"""Card-news rendering engine: generate_card / generate_deck.

Quality-first, deterministic, no GPU in the hot path. Backgrounds can be a
designer gradient (theme) or a user photo (saliency-aware placement + adaptive
text color + scrim). Typography uses Pretendard with a title/body hierarchy.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import themes as TH
from . import templates as TP
from .assets import font
from .placement import calmest_band, region_luma

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


def accent_blob(img, color, seed=0):
    """Subtle large translucent circle for visual interest on flat backgrounds."""
    w, h = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx = w * (0.85 if seed % 2 else 0.15)
    cy = h * (0.12 if seed % 3 else 0.9)
    r = int(w * 0.42)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (28,))
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(2)))


# ---------- text ----------
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
        if line_h * len(lines) <= bh and all(draw.textlength(l, font=fnt) <= bw for l in lines):
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
    ImageDraw.Draw(layer).rounded_rectangle([x0 - pw, y0 - ph, x1 + pw, y1 + ph],
                                            radius=radius, fill=col)
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(3)))


def _draw_text(img, box_px, text, *, weight, color, align, start, lh=1.16,
               scrim=None, accent_bar=None):
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box_px
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    if scrim is not None:
        _scrim(img, box_px, dark=scrim)
        draw = ImageDraw.Draw(img)
    fnt, lines, line_h = _fit(draw, text, bw, bh, weight, start, lh=lh)
    total = line_h * len(lines)
    y = y0 + (bh - total) / 2
    if accent_bar is not None and lines:
        bw_bar = max(5, int(bw * 0.014))
        draw.rounded_rectangle([x0 - bw_bar * 2, y + 4, x0 - bw_bar, y + total - 4],
                               radius=bw_bar, fill=accent_bar)
    for ln in lines:
        tw = draw.textlength(ln, font=fnt)
        x = {"left": x0, "center": x0 + (bw - tw) / 2, "right": x1 - tw}[align]
        draw.text((x, y), ln, font=fnt, fill=color)
        y += line_h


def _eyebrow(img, box_px, text, color, align):
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box_px
    h = int((y1 - y0) * 0.72)
    fnt = font(max(14, h), "SemiBold")
    # leading accent dot
    dot = h * 0.5
    draw.ellipse([x0, y0 + (y1 - y0 - dot) / 2, x0 + dot, y0 + (y1 - y0 + dot) / 2], fill=color)
    draw.text((x0 + dot * 1.6, y0 + (y1 - y0 - (sum(fnt.getmetrics()))) / 2),
              text, font=fnt, fill=color)


def _rule(img, box_px, color):
    x0, y0, x1, y1 = box_px
    ImageDraw.Draw(img).rounded_rectangle([x0, y0, x1, max(y0 + 4, y1)],
                                          radius=3, fill=color)


def _logo(img, box_px, text, color, align):
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box_px
    fnt = font(max(14, int((y1 - y0) * 0.7)), "Bold")
    tw = draw.textlength(text, font=fnt)
    x = {"left": x0, "center": x0 + ((x1 - x0) - tw) / 2, "right": x1 - tw}[align]
    draw.text((x, y0), text, font=fnt, fill=color)


def _page_badge(img, idx, total, color):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    fnt = font(26, "SemiBold")
    t = f"{idx:02d} / {total:02d}"
    tw = draw.textlength(t, font=fnt)
    draw.text((w - tw - 46, h - 60), t, font=fnt, fill=color)


# ---------- main API ----------
def generate_card(title, body=None, *, background=None, theme="auto",
                  template="auto", eyebrow=None, logo_text="BRAND",
                  page=None, total=None, size=SIZE, seed=0):
    """Render one card-news image. Returns a PIL RGB image (1080x1350)."""
    th = TH.pick(seed) if theme in (None, "auto") else TH.get(theme)
    has_photo = background is not None

    # background layer
    if has_photo:
        bg = background if isinstance(background, Image.Image) else Image.open(background)
        base = cover_fit(bg.convert("RGB"), size).convert("RGBA")
    else:
        base = gradient(size, th["bg"][0], th["bg"][-1], th.get("angle", 90)).convert("RGBA")
        accent_blob(base, th["accent"], seed)

    # template
    slots = TP.get(template) if template not in (None, "auto") else TP.auto(title, body or "", has_photo)
    W, H = size
    content = {"eyebrow": eyebrow, "title": title, "body": body, "logo": logo_text}

    for role, x0, y0, x1, y1, align in slots:
        box = (x0 * W, y0 * H, x1 * W, y1 * H)
        if role == "rule":
            _rule(base, box, th["accent"]); continue
        text = content.get(role)
        if not text:
            continue
        if has_photo:
            dark = region_luma(base.convert("RGB"), box) < 130
            fg = (255, 255, 255) if dark else (24, 24, 24)
            scrim = dark
        else:
            scrim = None
        if role == "eyebrow":
            _eyebrow(base, box, text, th["accent"] if not has_photo else (fg), align)
        elif role == "title":
            _draw_text(base, box, text, weight="Bold",
                       color=(fg if has_photo else th["title"]), align=align,
                       start=0.62, lh=1.12, scrim=scrim,
                       accent_bar=th["accent"] if align == "left" else None)
        elif role == "body":
            _draw_text(base, box, text, weight="Regular",
                       color=(fg if has_photo else th["body"]), align=align,
                       start=0.30, lh=1.45, scrim=scrim)
        elif role == "logo":
            _logo(base, box, text, th["accent"] if not has_photo else fg, align)

    if page and total:
        _page_badge(base, page, total, th["body"] if not has_photo else (255, 255, 255))
    return base.convert("RGB")


def generate_deck(items, *, theme="auto", logo_text="BRAND", background=None, seed=0):
    """items: list of dicts {title, body?, eyebrow?}. First item -> cover.
    Returns list of PIL images. Same theme across the deck for consistency."""
    th_name = TH.ORDER[seed % len(TH.ORDER)] if theme in (None, "auto") else theme
    n = len(items)
    cards = []
    for i, it in enumerate(items):
        tmpl = "cover" if i == 0 else "auto"
        cards.append(generate_card(
            it.get("title", ""), it.get("body"),
            background=background, theme=th_name, template=tmpl,
            eyebrow=it.get("eyebrow"), logo_text=logo_text,
            page=(None if i == 0 else i + 1), total=(None if i == 0 else n), seed=seed))
    return cards
