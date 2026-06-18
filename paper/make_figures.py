"""Generate the paper figures from real data/outputs. -> paper/figures/*.png"""
import os
import glob
import shutil
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
FONT = os.path.join(ROOT, "assets", "fonts", "Pretendard-SemiBold.ttf")
FONTB = os.path.join(ROOT, "assets", "fonts", "Pretendard-Bold.ttf")


def f(sz, bold=False):
    p = FONTB if bold else FONT
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return ImageFont.load_default()


def montage(paths, cols, cell=340, pad=10, bg=(255, 255, 255)):
    ims = [Image.open(p).convert("RGB") for p in paths]
    rows = (len(ims) + cols - 1) // cols
    cw = ch = cell
    W = cols * cw + pad * (cols + 1)
    H = rows * ch + pad * (rows + 1)
    canvas = Image.new("RGB", (W, H), bg)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        t = im.copy(); t.thumbnail((cw, ch))
        x = pad + c * (cw + pad) + (cw - t.width) // 2
        y = pad + r * (ch + pad) + (ch - t.height) // 2
        canvas.paste(t, (x, y))
    return canvas


def fig_dataset():
    paths = sorted(glob.glob(os.path.join(ROOT, "images_new", "*.jpg")))
    pick = paths[::max(1, len(paths) // 6)][:6]
    montage(pick, cols=3, cell=330).save(os.path.join(FIG, "fig_dataset.png"))
    print("fig_dataset")


def fig_ablation():
    rows = {r["name"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results", "ablation.csv")))}
    sel = [("scratch\n(no pretrain)", "e04_n_scratch"),
           ("baseline\n(n, freeze10)", "e01_baseline_n_freeze10"),
           ("aug_card", "e07_aug_card"),
           ("yolov8s", "e13_yolov8s_freeze10"),
           ("long300_card\n(best)", "e15_long300_card")]
    labels = [s[0] for s in sel]
    vals = [float(rows[s[1]]["map"]) for s in sel]
    colors = ["#bbb", "#88a", "#6a9", "#5a8", "#2a7"]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylabel("mAP@50-95"); plt.ylim(0, 0.8)
    plt.title("Detector ablation (109 imgs, YOLOv8, pseudo-labels)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_ablation.png"), dpi=130); plt.close()
    print("fig_ablation")


def box(d, xy, wh, title, sub, fill, fg=(255, 255, 255)):
    x, y = xy; w, h = wh
    d.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=fill)
    d.text((x + w / 2, y + h / 2 - 12), title, font=f(22, True), fill=fg, anchor="mm")
    if sub:
        d.text((x + w / 2, y + h / 2 + 14), sub, font=f(15), fill=fg, anchor="mm")


def arrow(d, p0, p1, color=(90, 90, 90)):
    d.line([p0, p1], fill=color, width=4)
    import math
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (2.6, -2.6):
        d.line([p1, (p1[0] - 16 * math.cos(ang - s/6), p1[1] - 16 * math.sin(ang - s/6))],
               fill=color, width=4)


def fig_pipeline():
    W, H = 1180, 360
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box(d, (20, 140), (170, 80), "1. Collect", "687 cards", (52, 73, 94))
    arrow(d, (190, 180), (235, 180))
    box(d, (235, 140), (190, 80), "2. Detect", "YOLOv8 (title/body)", (39, 119, 90))
    arrow(d, (425, 180), (470, 180))
    box(d, (470, 40), (230, 80), "3a. Template engine", "extract + render (ours)", (34, 110, 60))
    box(d, (470, 240), (230, 80), "3b. DS-GAN", "generate (baseline)", (120, 80, 40))
    arrow(d, (700, 80), (760, 130))
    arrow(d, (700, 280), (760, 230))
    box(d, (760, 140), (180, 80), "4. Render", "Pretendard typo", (39, 119, 90))
    arrow(d, (940, 180), (985, 180))
    box(d, (985, 140), (175, 80), "Card-news", "1080x1350", (52, 73, 94))
    img.save(os.path.join(FIG, "fig_pipeline.png"))
    print("fig_pipeline")


def fig_results():
    items = []
    for p in (glob.glob(os.path.join(ROOT, "gen_output", "demo", "demo_0*.png"))[:2]):
        items.append(p)
    montage(items, cols=2, cell=420).save(os.path.join(FIG, "fig_results_template.png"))
    cps = sorted(glob.glob(os.path.join(ROOT, "gen_output", "copy", "copy_*.png")))
    if cps:
        Image.open(cps[0]).convert("RGB").save(os.path.join(FIG, "fig_results_copy.png"))
    print("fig_results")


def fig_detector():
    src = os.path.join(ROOT, "results", "e15_long300_card", "val_batch0_pred.jpg")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(FIG, "fig_detector_pred.jpg"))
    src2 = os.path.join(ROOT, "analysis", "fig_timeseries.png")
    if os.path.exists(src2):
        shutil.copy2(src2, os.path.join(FIG, "fig_timeseries.png"))
    print("fig_detector + timeseries")


if __name__ == "__main__":
    fig_dataset(); fig_ablation(); fig_pipeline(); fig_results(); fig_detector()
    print("figures ->", FIG)
