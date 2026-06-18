"""Detector layout tags for each reference  (RunPod / any torch machine).

Runs the trained detector (results/best_model/best.pt, classes:
0=title 1=body 2=logo 3=underlay) over every indexed image and stores a
structural `layout` block per record:

  counts        per-class element counts
  title_pos     normalized (cx, cy) of the dominant title block (None if none)
  text_cov      fraction of canvas covered by title+body boxes
  signature     coarse archetype string, e.g. "title-top|body-mid|logo-tr"

These feed structural filters/search and the layout-copy step (gen/copy_layout.py).

Setup (RunPod):  pip install ultralytics
Run:             python reflib/tag_layout.py --imgsz 1280
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ROOT too, so `from gen.common import load_yolo` (gen/ is a namespace package)
# resolves when this is run as a standalone script.
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

CLASSES = {0: "title", 1: "body", 2: "logo", 3: "underlay"}


def _band(v):
    return "top" if v < 0.34 else ("mid" if v < 0.67 else "bot")


def _hband(v):
    return "l" if v < 0.34 else ("c" if v < 0.67 else "r")


def _signature(boxes):
    """boxes: list of (cls, cx, cy, area_frac) normalized -> archetype string."""
    parts = []
    titles = [b for b in boxes if b[0] == 0] or [b for b in boxes if b[0] == 1]
    if titles:
        t = max(titles, key=lambda b: b[3])
        parts.append(f"title-{_band(t[2])}")
    bodies = [b for b in boxes if b[0] == 1]
    if bodies:
        cy = sum(b[2] for b in bodies) / len(bodies)
        parts.append(f"body-{_band(cy)}")
    logos = [b for b in boxes if b[0] == 2]
    if logos:
        l = logos[0]
        parts.append(f"logo-{_band(l[2])[0]}{_hband(l[1])}")
    return "|".join(parts) or "none"


def run(imgsz=1280, conf=0.25):
    from gen.common import load_yolo  # reuse the canonical loader
    import numpy as np
    from PIL import Image

    index = json.load(open(common.INDEX_PATH, encoding="utf-8"))
    recs = index["records"]
    model = load_yolo()

    for i, r in enumerate(recs, 1):
        p = os.path.join(common.ROOT, r["path"])
        try:
            im = Image.open(p).convert("RGB")
            W, H = im.size
            res = model.predict(np.array(im), conf=conf, imgsz=imgsz, verbose=False)[0]
        except Exception as e:
            print(f"  ! {r['id']}: {e}")
            continue
        boxes, counts = [], {v: 0 for v in CLASSES.values()}
        text_area = 0.0
        for b in res.boxes:
            c = int(b.cls[0])
            x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0)) / (W * H)
            counts[CLASSES[c]] += 1
            boxes.append((c, cx, cy, area))
            if c in (0, 1):
                text_area += area
        r["layout"] = {
            "counts": counts,
            "title_pos": next(([round(b[1], 3), round(b[2], 3)]
                               for b in sorted(boxes, key=lambda x: -x[3]) if b[0] == 0), None),
            "text_cov": round(min(text_area, 1.0), 3),
            "signature": _signature(boxes),
        }
        if i % 100 == 0:
            print(f"  {i}/{len(recs)}")

    # roll up signature popularity for browse/filters
    sigs = {}
    for r in recs:
        s = r.get("layout", {}).get("signature", "none")
        sigs[s] = sigs.get(s, 0) + 1
    index["layout_signatures"] = dict(sorted(sigs.items(), key=lambda x: -x[1]))
    json.dump(index, open(common.INDEX_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print("wrote layout tags. top signatures:")
    for s, n in list(index["layout_signatures"].items())[:10]:
        print(f"  {n:>3}  {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    run(args.imgsz, args.conf)


if __name__ == "__main__":
    main()
