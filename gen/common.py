"""Shared constants/helpers for the generation pipeline (Stages 3-5)."""
import os
import glob

# PosterLayout canonical canvas (DS-GAN normalizes boxes by these).
W, H = 513, 750

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTS = (".png", ".jpg", ".jpeg")

# class ids in PosterLayout: 0=pad, 1=text, 2=logo, 3=underlay
CLS_TEXT, CLS_LOGO, CLS_UNDERLAY = 1, 2, 3


def list_images(*dirs):
    paths = []
    for d in dirs:
        ad = d if os.path.isabs(d) else os.path.join(ROOT, d)
        if not os.path.isdir(ad):
            continue
        for e in os.scandir(ad):
            if e.is_file() and e.name.lower().endswith(EXTS):
                paths.append(e.path)
    return sorted(paths)


FONT_DIR = os.path.join(ROOT, "assets", "fonts")
# Malgun weights for the bold/regular fallback when Pretendard isn't bundled.
_MALGUN = {"Bold": r"C:\Windows\Fonts\malgunbd.ttf", "Regular": r"C:\Windows\Fonts\malgun.ttf"}


def find_korean_font(size=34, weight="Regular"):
    """PIL ImageFont with Korean glyph coverage. Prefers the bundled designer font
    (Pretendard); falls back to Malgun / Noto / Nanum. weight in
    {Bold, SemiBold, Medium, Regular}."""
    from PIL import ImageFont
    candidates = [
        os.path.join(FONT_DIR, f"Pretendard-{weight}.ttf"),
        os.path.join(FONT_DIR, "Pretendard-Regular.ttf"),
        _MALGUN.get(weight, _MALGUN["Regular"]),
        r"C:\Windows\Fonts\malgun.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" if weight == "Bold"
        else "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def load_yolo(weights=None):
    """Load the trained detector. Defaults to the best comparison/baseline model."""
    from ultralytics import YOLO
    if weights is None:
        for cand in ("results/best_model/best_full687.pt",
                     "results/best_model/best.pt"):
            p = os.path.join(ROOT, cand)
            if os.path.exists(p):
                weights = p
                break
    if weights is None:
        raise SystemExit("no detector weights found (results/best_model/*.pt)")
    return YOLO(weights)


def detect_boxes(model, pil_canvas, conf=0.25, max_elem=8, imgsz=1280):
    """Detect on a 513x750 RGB canvas. Returns list of (cls_elem, x0,y0,x1,y1)
    in canvas pixels. title/body -> text(1); kept top-`max_elem` by confidence.
    imgsz=1280 (high-res inference) so small body text/blocks are still found."""
    import numpy as np
    res = model.predict(np.array(pil_canvas), conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for b in res.boxes:
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
        c = float(b.conf[0])
        out.append((c, CLS_TEXT, x0, y0, x1, y1))
    out.sort(key=lambda r: r[0], reverse=True)
    return [(cls, x0, y0, x1, y1) for _c, cls, x0, y0, x1, y1 in out[:max_elem]]
