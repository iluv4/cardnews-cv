"""Generate a tiny synthetic test set so infer.py/eval.py can run end-to-end
without the gated PKU dataset. Produces 4 fake 'clean canvas' backgrounds plus
matching saliency maps in the directory layout the dataloader expects.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

W, H = 513, 750  # PosterLayout's canonical canvas size
N = 4

base = "Dataset/test"
canvas_dir = os.path.join(base, "image_canvas")
pfpn_dir = os.path.join(base, "saliencymaps_pfpn")
basnet_dir = os.path.join(base, "saliencymaps_basnet")
for d in (canvas_dir, pfpn_dir, basnet_dir):
    os.makedirs(d, exist_ok=True)

rng = np.random.default_rng(0)
for i in range(N):
    # --- background: vertical gradient + a colored "product" blob (the salient region) ---
    grad = np.linspace(40, 210, H, dtype=np.uint8)[:, None].repeat(W, axis=1)
    img = np.stack([grad, np.roll(grad, 60), np.roll(grad, 120)], axis=-1)
    canvas = Image.fromarray(img, "RGB")
    draw = ImageDraw.Draw(canvas)
    cx, cy = rng.integers(120, W - 120), rng.integers(180, H - 180)
    r = rng.integers(80, 150)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(230, 90, 60))
    canvas.save(os.path.join(canvas_dir, f"dummy{i}.png"))

    # --- saliency maps: a white blob where the product is, black elsewhere ---
    sal = Image.new("L", (W, H), 0)
    sdraw = ImageDraw.Draw(sal)
    sdraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    sal.save(os.path.join(basnet_dir, f"dummy{i}.png"))
    sal.save(os.path.join(pfpn_dir, f"dummy{i}_pred.png"))

print(f"Wrote {N} dummy canvases + saliency maps under {base}/")
