"""Layout-COPY demo (no GPU): analyze an existing card's layout, clean its
background, and refill the SAME layout with new text.

This reflects the product direction ("copy the existing layout"):
  1. read the detected element boxes for a real card (YOLO labels = the analyzed
     layout). title=0, body=1.
  2. merge lines into title/body BLOCKS.
  3. erase the original text with classical inpainting (cv2 Telea) -> clean bg.
  4. render new text back into the SAME block positions (Pretendard, hierarchy).
  5. save a 3-panel comparison: original | cleaned background | recomposed.

Uses only PIL + numpy + cv2 (already installed) — runs locally without torch.
On RunPod the boxes come from the trained detector + LaMa replaces cv2 inpaint;
the render step is identical.

  py -3 gen/copy_layout.py --n 4
"""
import os
import glob
import argparse

import numpy as np
import cv2
from PIL import Image

from common import ROOT
from render import RW, RH, draw_text_block, draw_logo

IMG_DIR = os.path.join(ROOT, "dataset", "images", "train")
LBL_DIR = os.path.join(ROOT, "dataset", "labels", "train")

NEW_TITLE = "스마트팜으로\n바뀌는 농업"
NEW_BODY = "복잡한 내용을 한 장에 정리했습니다.\n핵심만 빠르게 확인하세요."


def parse_label(path, W, H):
    """YOLO label -> dict cls -> list of (x0,y0,x1,y1) pixel boxes."""
    out = {0: [], 1: []}
    if not os.path.exists(path):
        return out
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) != 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
        if c not in out:
            continue
        out[c].append((max(0, (cx - w / 2) * W), max(0, (cy - h / 2) * H),
                       min(W, (cx + w / 2) * W), min(H, (cy + h / 2) * H)))
    return out


def merge(rects, gap_v=0.7, gap_h=0.5):
    n = len(rects)
    if n <= 1:
        return [tuple(r) for r in rects]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def exp(r):
        x0, y0, x1, y1 = r; h = max(1.0, y1 - y0)
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


def inpaint_clean(pil_rgb, boxes_px, dilate=10):
    arr = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)
    H, W = arr.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in boxes_px:
        cv2.rectangle(mask, (int(x0 - dilate), int(y0 - dilate)),
                      (int(x1 + dilate), int(y1 + dilate)), 255, -1)
    out = cv2.inpaint(arr, mask, 4, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def recompose(clean_pil, title_blocks, body_blocks, W, H):
    img = clean_pil.convert("RGB").resize((RW, RH)).convert("RGBA")
    sx, sy = RW / W, RH / H
    P = lambda b: (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)
    for b in title_blocks:
        draw_text_block(img, P(b), NEW_TITLE, weight="Bold", start_frac=0.62,
                        center=True, accent=(255, 90, 60))
    for b in body_blocks:
        draw_text_block(img, P(b), NEW_BODY, weight="Regular", start_frac=0.34,
                        center=False)
    return img.convert("RGB")


def panel(pil, h=720):
    w = int(pil.width * h / pil.height)
    return pil.convert("RGB").resize((w, h))


def montage(images, gap=16, bg=(245, 245, 245)):
    ps = [panel(im) for im in images]
    h = max(p.height for p in ps)
    W = sum(p.width for p in ps) + gap * (len(ps) + 1)
    canvas = Image.new("RGB", (W, h + 2 * gap), bg)
    x = gap
    for p in ps:
        canvas.paste(p, (x, gap)); x += p.width + gap
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "gen_output", "copy")
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    made = 0
    for ip in imgs:
        if made >= args.n:
            break
        lp = os.path.join(LBL_DIR, os.path.splitext(os.path.basename(ip))[0] + ".txt")
        orig = Image.open(ip).convert("RGB")
        W, H = orig.size
        boxes = parse_label(lp, W, H)
        if not boxes[0] and not boxes[1]:
            continue
        titles = merge(boxes[0])
        bodies = merge(boxes[1])
        # if no detected title block, promote the topmost body block to title
        if not titles and bodies:
            bodies = sorted(bodies, key=lambda b: b[1])
            titles, bodies = [bodies[0]], bodies[1:]

        clean = inpaint_clean(orig, boxes[0] + boxes[1])
        recomposed = recompose(clean, titles, bodies, W, H)
        out = montage([orig, clean, recomposed])
        op = os.path.join(out_dir, f"copy_{made:02d}_{os.path.basename(ip)}.png")
        out.save(op)
        print(f"  wrote {op}  (titles {len(titles)} / bodies {len(bodies)})")
        made += 1
    print(f"\nlayout-copy comparisons -> {out_dir}")


if __name__ == "__main__":
    main()
