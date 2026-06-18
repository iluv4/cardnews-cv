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


def find_korean_font(size=34):
    """Return a PIL ImageFont with Korean glyph coverage, trying common paths."""
    from PIL import ImageFont
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",            # Windows: Malgun Gothic
        r"C:\Windows\Fonts\NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",       # Linux (fonts-nanum)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # last resort (no Hangul)
    ]
    for c in candidates:
        if os.path.exists(c):
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


def detect_boxes(model, pil_canvas, conf=0.25, max_elem=8):
    """Detect on a 513x750 RGB canvas. Returns list of (cls_elem, x0,y0,x1,y1)
    in canvas pixels. title/body -> text(1); kept top-`max_elem` by confidence."""
    import numpy as np
    res = model.predict(np.array(pil_canvas), conf=conf, verbose=False)[0]
    out = []
    for b in res.boxes:
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
        c = float(b.conf[0])
        out.append((c, CLS_TEXT, x0, y0, x1, y1))
    out.sort(key=lambda r: r[0], reverse=True)
    return [(cls, x0, y0, x1, y1) for _c, cls, x0, y0, x1, y1 in out[:max_elem]]
