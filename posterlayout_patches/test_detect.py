"""Contrast demo: TEXT DETECTION (recognize the real layout) vs PosterLayout's
generation. Runs EasyOCR (Korean+English) on the same cards and draws a box
around every text region it actually finds.

This is a stand-in for the detector we will TRAIN in Stage 2 (title/body/logo/
underlay classes). Here we only use a pretrained text detector to show that
"recognizing the existing text layout" is a detection task, not a GAN task.
"""
import os
import glob
import easyocr
from PIL import Image, ImageDraw

SRC_DIR = r"C:\Users\Administrator\Downloads\carnews-insta\public\saved-refs"
OUT_DIR = "output/detect_test"
N = 6
W, H = 513, 750

os.makedirs(OUT_DIR, exist_ok=True)
reader = easyocr.Reader(["ko", "en"], gpu=True)

paths = [p for p in sorted(glob.glob(os.path.join(SRC_DIR, "*.*")))
         if p.lower().endswith((".png", ".jpg", ".jpeg"))][:N]
print(f"images={len(paths)}")

for i, p in enumerate(paths):
    img = Image.open(p).convert("RGB").resize((W, H))
    # detail=1 -> (bbox, text, confidence); paragraph keeps lines separate
    results = reader.readtext(os.path.normpath(p))
    draw = ImageDraw.Draw(img)
    n_text = 0
    for bbox, text, conf in results:
        if conf < 0.3:
            continue
        n_text += 1
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        # bbox is in original-image coords -> scale to the resized canvas
        ow, oh = Image.open(p).size
        x0, x1 = min(xs) / ow * W, max(xs) / ow * W
        y0, y1 = min(ys) / oh * H, max(ys) / oh * H
        draw.rectangle([x0, y0, x1, y1], outline=(0, 120, 255), width=4)
    out = os.path.join(OUT_DIR, f"detect_{i}_{os.path.basename(p)}.png")
    img.save(out)
    print(f"[{i}] {os.path.basename(p)}: detected {n_text} text regions")

print(f"\nSaved detection overlays to {os.path.abspath(OUT_DIR)}")
