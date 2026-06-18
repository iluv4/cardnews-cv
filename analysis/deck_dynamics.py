"""Sequential (time-series-style) analysis of card-news DECKS.

A card-news deck is an ordered sequence of pages (cover -> content -> closing).
We treat the page index as a discrete time axis and measure how low-level visual
/ layout features evolve along it, aggregated over all decks. Pure cv2 (no GPU).

Features per page (proxies for layout/visual structure):
  - edge_density : fraction of Canny edge pixels (text + graphic density)
  - brightness   : mean luma / 255
  - colorfulness : Hasler & Susstrunk (2003) metric, normalized
  - saturation   : mean HSV S / 255

Outputs:
  analysis/deck_dynamics.csv         (per relative-position bin, mean features)
  analysis/fig_timeseries.png        (feature trajectories along the deck)

Run locally:  py -3 analysis/deck_dynamics.py
"""
import os
import csv
from collections import defaultdict

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images_new")
MAP = os.path.join(IMG_DIR, "_mapping_new.csv")
NBINS = 10
MIN_LEN = 4  # only decks with >= this many pages (need a sequence)


def features(path):
    img = cv2.imread(path)
    if img is None:
        # non-ASCII safe read
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (384, 384))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())
    brightness = float(gray.mean()) / 255.0
    b, g, r = [img[:, :, i].astype(np.float32) for i in (0, 1, 2)]
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorful = float(np.sqrt(rg.std()**2 + yb.std()**2)
                     + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)) / 255.0
    sat = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) / 255.0
    return dict(edge_density=edge_density, brightness=brightness,
                colorfulness=colorful, saturation=sat)


def main():
    rows = list(csv.DictReader(open(MAP, encoding="utf-8")))
    decks = defaultdict(list)
    for r in rows:
        decks[(r["source"], r["series_folder"])].append((int(r["page_idx"]), r["new_name"]))

    keys = ["edge_density", "brightness", "colorfulness", "saturation"]
    bins = [defaultdict(list) for _ in range(NBINS)]
    n_decks = 0
    for pages in decks.values():
        pages.sort()
        L = len(pages)
        if L < MIN_LEN:
            continue
        n_decks += 1
        for i, (_pi, name) in enumerate(pages):
            f = features(os.path.join(IMG_DIR, name))
            if not f:
                continue
            rel = i / (L - 1)                       # 0 = first page, 1 = last
            b = min(NBINS - 1, int(rel * NBINS))
            for k in keys:
                bins[b][k].append(f[k])

    centers = [(b + 0.5) / NBINS for b in range(NBINS)]
    means = {k: [float(np.mean(bins[b][k])) if bins[b][k] else float("nan")
                 for b in range(NBINS)] for k in keys}

    with open(os.path.join(ROOT, "analysis", "deck_dynamics.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rel_position"] + keys)
        for b in range(NBINS):
            w.writerow([round(centers[b], 3)] + [round(means[k][b], 4) for k in keys])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(centers, means["edge_density"], "-o", color="#2a7", label="edge density")
    ax[0].plot(centers, means["colorfulness"], "-s", color="#d63", label="colorfulness")
    ax[0].set_title("Structure / color vs. deck position")
    ax[1].plot(centers, means["brightness"], "-o", color="#37b", label="brightness")
    ax[1].plot(centers, means["saturation"], "-s", color="#b3b", label="saturation")
    ax[1].set_title("Tone vs. deck position")
    for a in ax:
        a.set_xlabel("relative page position (0=cover  →  1=last)")
        a.set_ylabel("normalized value"); a.grid(alpha=0.3); a.legend()
    fig.suptitle(f"Card-news deck dynamics  (n={n_decks} decks, ≥{MIN_LEN} pages)")
    fig.tight_layout()
    out = os.path.join(ROOT, "analysis", "fig_timeseries.png")
    fig.savefig(out, dpi=130)
    print(f"decks analyzed: {n_decks}")
    print(f"wrote {out} + deck_dynamics.csv")


if __name__ == "__main__":
    main()
