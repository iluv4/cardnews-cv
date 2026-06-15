"""Auto-label card images into YOLO format using EasyOCR (pseudo-labels).

- Text regions -> class 0 (title) if tall relative to image, else class 1 (body).
- logo (2) / underlay (3) are NOT auto-labelled here (need manual annotation).
- Images are re-encoded to ASCII filenames + RGB JPEG so Ultralytics' cv2.imread
  works on Windows (cv2 can't read non-ASCII paths).
"""
import os
import csv
import glob
import random
import numpy as np
from PIL import Image
import easyocr

SRC = r"C:\Users\Administrator\Downloads\cardnews-detector\images"
ROOT = r"C:\Users\Administrator\Downloads\cardnews-detector\dataset"
CONF = 0.30
TITLE_H_RATIO = 0.045  # text taller than 4.5% of image height -> title
VAL_FRAC = 0.15
SEED = 0

for split in ("train", "val"):
    os.makedirs(os.path.join(ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "labels", split), exist_ok=True)

reader = easyocr.Reader(["ko", "en"], gpu=True)

paths = [p for p in sorted(glob.glob(os.path.join(SRC, "*")))
         if p.lower().endswith((".png", ".jpg", ".jpeg"))]
random.Random(SEED).shuffle(paths)
n_val = max(1, int(len(paths) * VAL_FRAC))
val_set = set(paths[:n_val])

mapping = []
counts = {"title": 0, "body": 0, "empty_imgs": 0}
for i, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    W, H = img.size
    arr = np.array(img)
    results = reader.readtext(arr)

    lines = []
    for bbox, text, conf in results:
        if conf < CONF:
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x0, x1 = max(0, min(xs)), min(W, max(xs))
        y0, y1 = max(0, min(ys)), min(H, max(ys))
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1 or bh <= 1:
            continue
        cls = 0 if (bh / H) >= TITLE_H_RATIO else 1
        cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
        counts["title" if cls == 0 else "body"] += 1

    split = "val" if p in val_set else "train"
    name = f"img_{i:03d}"
    img.save(os.path.join(ROOT, "images", split, name + ".jpg"), quality=92)
    with open(os.path.join(ROOT, "labels", split, name + ".txt"), "w") as fh:
        fh.write("\n".join(lines))
    if not lines:
        counts["empty_imgs"] += 1
    mapping.append([name + ".jpg", os.path.basename(p), split, len(lines)])

with open(os.path.join(ROOT, "mapping.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["yolo_name", "original", "split", "n_boxes"])
    w.writerows(mapping)

yaml = (
    "path: C:/Users/Administrator/Downloads/cardnews-detector/dataset\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n  0: title\n  1: body\n  2: logo\n  3: underlay\n"
)
with open(os.path.join(os.path.dirname(ROOT), "dataset.yaml"), "w", encoding="utf-8") as fh:
    fh.write(yaml)

print(f"images: {len(paths)}  (train {len(paths)-n_val} / val {n_val})")
print(f"boxes -> title: {counts['title']}, body: {counts['body']}")
print(f"images with no text (background): {counts['empty_imgs']}")
print("wrote dataset/ + dataset.yaml + mapping.csv")
