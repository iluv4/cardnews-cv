"""Build the layout-TEMPLATE LIBRARY from the labeled corpus (no GPU).

Product core, first half: the engine is NOT one hand-coded style — it is *one
reusable layout template per reference in the dataset*. Here we turn each
labeled card (YOLO title/body boxes) into a normalized, refillable template:

  reference image + label  ->  {blocks, archetype, palette, bg, align, ...}

These templates are what a user later searches/selects; the selected template is
copied and refilled with the user's text by the v2 renderer. Right now the boxes
come from the committed labels (109 imgs, runs locally). On RunPod the trained
detector labels the full 687-image corpus the same way, growing the library with
no code change.

  py -3 gen/extract_templates.py            # -> service/library/templates.json
"""
import os
import glob
import json

import numpy as np
from PIL import Image

from common import ROOT

PAIRS = [("dataset/images/train", "dataset/labels/train"),
         ("dataset/images/val", "dataset/labels/val")]
OUT_DIR = os.path.join(ROOT, "service", "library")
CLS = {0: "title", 1: "body"}


# ---------- label parsing + block merging (shared with copy_layout) ----------
def parse_label(path):
    """YOLO label -> {0:[(cx,cy,w,h)...], 1:[...]} in normalized coords."""
    out = {0: [], 1: []}
    if not os.path.exists(path):
        return out
    for ln in open(path, encoding="utf-8-sig"):
        p = ln.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        if c not in out:
            continue
        cx, cy, w, h = map(float, p[1:])
        out[c].append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def merge(rects, gap_v=0.7, gap_h=0.5):
    """Union-find merge of line boxes into blocks (gaps scaled by box height)."""
    n = len(rects)
    if n <= 1:
        return [tuple(r) for r in rects]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def exp(r):
        x0, y0, x1, y1 = r
        h = max(1e-3, y1 - y0)
        return (x0 - gap_h * h, y0 - gap_v * h, x1 + gap_h * h, y1 + gap_v * h)

    er = [exp(r) for r in rects]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = er[i], er[j]
            if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                parent[find(i)] = find(j)
    g = {}
    for i, r in enumerate(rects):
        g.setdefault(find(i), []).append(r)
    return [(min(x[0] for x in v), min(x[1] for x in v),
             max(x[2] for x in v), max(x[3] for x in v)) for v in g.values()]


# ---------- color / style descriptors ----------
def palette(img, k=5):
    sm = img.convert("RGB").resize((120, 120))
    q = sm.quantize(colors=k).convert("RGB")
    cols = sorted(q.getcolors(120 * 120) or [], reverse=True)
    return [[int(c) for c in rgb] for _cnt, rgb in cols[:k]]


def bg_color(img):
    rgb = img.convert("RGB")
    W, H = rgb.size
    s = max(4, int(min(W, H) * 0.08))
    patches = [np.asarray(rgb.crop((x, y, x + s, y + s))).reshape(-1, 3)
               for x, y in [(0, 0), (W - s, 0), (0, H - s), (W - s, H - s)]]
    return [int(v) for v in np.median(np.concatenate(patches, 0), 0)]


def luma(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def aspect_tag(W, H):
    r = W / H
    if r > 1.15:
        return "landscape"
    if r < 0.85:
        return "portrait"
    return "square"


def classify(titles, bodies):
    nt, nb = len(titles), len(bodies)
    if nb == 0 and nt >= 1:
        return "cover"          # title-only hero
    if nb >= 3:
        return "list"           # checklist / multi-point body
    if nt >= 1 and 1 <= nb <= 2:
        return "statement"      # title + short body
    if nt == 0 and nb >= 1:
        return "body"           # body-only (quote / paragraph)
    return "mixed"


def align_of(blocks):
    if not blocks:
        return "center"
    cx = np.mean([(b[0] + b[2]) / 2 for b in blocks])
    return "center" if 0.4 <= cx <= 0.6 else ("left" if cx < 0.4 else "right")


# ---------- main ----------
def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    templates = []
    for img_dir, lbl_dir in PAIRS:
        for ip in sorted(glob.glob(os.path.join(ROOT, img_dir, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(ROOT, lbl_dir, stem + ".txt")
            boxes = parse_label(lp)
            if not boxes[0] and not boxes[1]:
                continue
            img = Image.open(ip).convert("RGB")
            W, H = img.size
            titles = sorted(merge(boxes[0]), key=lambda b: b[1])
            bodies = sorted(merge(boxes[1]), key=lambda b: b[1])
            # promote topmost body to title if none detected
            if not titles and bodies:
                titles, bodies = [bodies[0]], bodies[1:]
            bg = bg_color(img)
            r = lambda b: [round(v, 4) for v in b]
            templates.append({
                "id": stem,
                "source": os.path.relpath(ip, ROOT).replace("\\", "/"),
                "size": [W, H],
                "aspect": aspect_tag(W, H),
                "archetype": classify(titles, bodies),
                "blocks": {"title": [r(b) for b in titles],
                           "body": [r(b) for b in bodies]},
                "n_title": len(titles),
                "n_body": len(bodies),
                "title_align": align_of(titles),
                "bg": bg,
                "is_dark": bool(luma(bg) < 128),
                "palette": palette(img),
            })

    out = os.path.join(OUT_DIR, "templates.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=1)

    # report
    arch = {}
    for t in templates:
        arch[t["archetype"]] = arch.get(t["archetype"], 0) + 1
    dark = sum(t["is_dark"] for t in templates)
    print(f"templates: {len(templates)}  -> {out}")
    print(f"  dark/light: {dark}/{len(templates) - dark}")
    print("  archetypes:", ", ".join(f"{k}={v}" for k, v in sorted(arch.items())))
    return templates


if __name__ == "__main__":
    build()
