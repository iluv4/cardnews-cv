"""Build the layout-template library for the FULL corpus via the trained
detector (torch — RunPod or a local GPU).

This is the RunPod half of step 2: the committed-labels extractor
(`extract_templates.py`) only covers the 95 labeled cards. Here the detector
labels every image in the corpus the SAME way, so EVERY searchable reference
gets a refillable template (template ids == reflib index ids == filename stems).
After this runs, the service's "theme fallback" path disappears — every
selection copies a real per-reference layout.

Output: service/library/templates.json  (same schema as extract_templates.py)

Setup:  pip install ultralytics pillow numpy
Run:    py -3 gen/extract_templates_detector.py [--conf 0.25] [--imgsz 1280]
"""
import os
import sys
import json
import argparse

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))         # gen/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from common import ROOT, load_yolo                                      # gen/common.py
from extract_templates import (merge, palette, bg_color, luma,          # reuse helpers
                               aspect_tag, classify, align_of, OUT_DIR)
from reflib import common as rc                                         # same corpus + ids


def detect_blocks(model, img, conf, imgsz):
    """Detector -> {0:[title boxes], 1:[body boxes]} in normalized xyxy."""
    W, H = img.size
    res = model.predict(np.array(img), conf=conf, imgsz=imgsz, verbose=False)[0]
    out = {0: [], 1: []}
    for b in res.boxes:
        c = int(b.cls[0])
        if c not in out:            # keep only title(0)/body(1) for refill blocks
            continue
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
        out[c].append((x0 / W, y0 / H, x1 / W, y1 / H))
    return out


def build(conf=0.25, imgsz=1280):
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_yolo()
    paths = rc.list_images()
    templates, n = [], len(paths)
    print(f"extracting templates from {n} images (detector) ...")
    for i, ip in enumerate(paths, 1):
        try:
            img = Image.open(ip).convert("RGB")
        except Exception as e:
            print(f"  ! {os.path.basename(ip)}: {e}")
            continue
        W, H = img.size
        boxes = detect_blocks(model, img, conf, imgsz)
        if not boxes[0] and not boxes[1]:
            continue
        titles = sorted(merge(boxes[0]), key=lambda b: b[1])
        bodies = sorted(merge(boxes[1]), key=lambda b: b[1])
        if not titles and bodies:                 # promote topmost body to title
            titles, bodies = [bodies[0]], bodies[1:]
        bg = bg_color(img)
        r = lambda b: [round(v, 4) for v in b]
        templates.append({
            "id": os.path.splitext(os.path.basename(ip))[0],
            "source": os.path.relpath(ip, ROOT).replace("\\", "/"),
            "size": [W, H], "aspect": aspect_tag(W, H),
            "archetype": classify(titles, bodies),
            "blocks": {"title": [r(b) for b in titles], "body": [r(b) for b in bodies]},
            "n_title": len(titles), "n_body": len(bodies),
            "title_align": align_of(titles), "bg": bg,
            "is_dark": bool(luma(bg) < 128), "palette": palette(img),
        })
        if i % 100 == 0 or i == n:
            print(f"  {i}/{n}  ({len(templates)} templates)")

    out = os.path.join(OUT_DIR, "templates.json")
    json.dump(templates, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    arch = {}
    for t in templates:
        arch[t["archetype"]] = arch.get(t["archetype"], 0) + 1
    print(f"\nwrote {len(templates)} templates -> {out}")
    print("  archetypes:", ", ".join(f"{k}={v}" for k, v in sorted(arch.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()
    build(args.conf, args.imgsz)


if __name__ == "__main__":
    main()
