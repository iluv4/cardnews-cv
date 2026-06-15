"""Draw the generated YOLO labels on a few images so we can eyeball quality.
title=red, body=green.
"""
import os
import glob
from PIL import Image, ImageDraw

ROOT = r"C:\Users\Administrator\Downloads\cardnews-detector\dataset"
OUT = r"C:\Users\Administrator\Downloads\cardnews-detector\label_preview"
os.makedirs(OUT, exist_ok=True)
COLORS = {0: (230, 30, 30), 1: (30, 200, 30)}  # title, body

# pick a few train images that actually have labels
imgs = sorted(glob.glob(os.path.join(ROOT, "images", "train", "*.jpg")))
shown = 0
for ip in imgs:
    name = os.path.splitext(os.path.basename(ip))[0]
    lp = os.path.join(ROOT, "labels", "train", name + ".txt")
    with open(lp) as fh:
        rows = [r.split() for r in fh.read().splitlines() if r.strip()]
    if len(rows) < 3:
        continue
    img = Image.open(ip).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    for cls, cx, cy, w, h in rows:
        cls = int(cls); cx, cy, w, h = float(cx)*W, float(cy)*H, float(w)*W, float(h)*H
        d.rectangle([cx-w/2, cy-h/2, cx+w/2, cy+h/2], outline=COLORS[cls], width=4)
    img.save(os.path.join(OUT, f"{name}.png"))
    shown += 1
    if shown >= 6:
        break
print(f"wrote {shown} previews to {OUT}")
