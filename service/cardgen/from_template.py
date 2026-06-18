"""Render INTO an extracted layout template — the real "copy that layout, refill
with the user's text" step (vs the v2 fixed-kind layouts).

A template (see gen/extract_templates.py) carries the reference's actual
normalized title/body block boxes + palette + dark flag. Here we:
  1. synthesize a theme from the reference palette (so colours match the source),
  2. place the user's title into the reference's title block(s) with a glow,
  3. place subtitle / checklist into the reference's body block(s),
using the v2 components for designer-grade quality. This makes auto-fill copy the
*actual layout* of a chosen reference, not just its tone — locally, no torch,
for every reference that has committed labels.
"""
import numpy as np

from . import components as C
from .render import themed_background


# ---------- theme synthesised from the reference palette ----------
def _luma(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _sat(c):
    return (max(c) - min(c)) / (max(c) + 1e-6)


def _darker(c, f=0.6):
    return tuple(int(v * f) for v in c)


def _pick_accent(palette, bg, dark):
    """Most vivid palette colour that contrasts with the background."""
    best, best_score = None, -1
    for c in palette:
        c = tuple(c)
        contrast = abs(_luma(c) - _luma(bg)) / 255.0
        score = _sat(c) * 0.7 + contrast * 0.3
        if _sat(c) < 0.18:           # skip near-greys
            continue
        if score > best_score:
            best, best_score = c, score
    if best is None:
        best = (244, 120, 150) if dark else (230, 57, 70)
    return best


def synth_theme(t):
    bg = tuple(t["bg"])
    dark = bool(t["is_dark"])
    accent = _pick_accent(t.get("palette", []), bg, dark)
    title = (255, 255, 255) if dark else (26, 26, 28)
    body = (226, 232, 229) if dark else (70, 72, 75)
    return {
        "bg": [bg, _darker(bg, 0.62 if dark else 0.94)], "angle": 90,
        "title": title, "title_accent": accent, "body": body, "accent": accent,
        "panel": (255, 255, 255), "panel_text": (28, 28, 30),
        "panel_sub": (150, 153, 158), "check": accent,
        "deco": title, "brand": title, "dark": dark,
    }


# ---------- geometry helpers ----------
def _union(blocks):
    a = np.array(blocks, float)
    return [a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()]


def _px(box, W, H, pad=0.0):
    x0, y0, x1, y1 = box
    px = (x1 - x0) * pad
    py = (y1 - y0) * pad
    return (max(0, x0 - px) * W, max(0, y0 - py) * H,
            min(1, x1 + px) * W, min(1, y1 + py) * H)


# ---------- main ----------
def render_from_template(t, title, *, subtitle=None, checklist=None,
                         brand=None, width=1080, seed=None):
    """Refill a layout template `t` with new content. Returns a PIL RGB image at
    the reference's aspect ratio."""
    th = synth_theme(t)
    W0, H0 = t["size"]
    W = width
    H = max(1, round(width * H0 / W0))
    sd = seed if seed is not None else (abs(hash(t["id"])) % 997)
    base = themed_background((W, H), th, sd)

    blocks = t["blocks"]
    titles, bodies = blocks.get("title", []), blocks.get("body", [])

    # resolve the body/panel region first, so a multi-line refilled title can be
    # clamped to stop above it (refill text rarely matches the original length).
    body_box = None
    if bodies:
        if checklist:
            body_box = _px(_union(bodies), W, H, pad=0.06)
        elif subtitle:
            body_box = _px(bodies[0], W, H, pad=0.25)

    if titles:
        x0, y0, x1, y1 = _px(_union(titles), W, H, pad=0.12)
        # give a short title block a little more height to breathe
        if (y1 - y0) < 0.12 * H:
            y1 = y0 + 0.12 * H
        if body_box is not None:                 # never overlap the panel/body
            y1 = min(y1, body_box[1] - 0.02 * H)
            if (y1 - y0) < 0.08 * H:              # keep a sane minimum height
                y1 = y0 + 0.08 * H
        C.glow_title(base, (x0, y0, x1, y1), title, th, start_frac=0.62)

    if checklist and bodies:
        C.checklist_panel(base, body_box, checklist, th)
    elif subtitle and bodies:
        C.subtitle(base, body_box, subtitle, th, start_frac=0.5)

    if brand:
        C.brand_mark(base, (0.60 * W, 0.04 * H, 0.95 * W, 0.092 * H), brand, th)

    return base.convert("RGB")
