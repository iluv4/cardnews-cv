"""Deterministic, torch-free visual features for each reference image.

Everything here runs locally with opencv + numpy (already installed). These give
the index real signal for color/brightness/layout-feel search before the heavy
RunPod pass (CLIP embeddings + detector layout tags) adds semantic + structural
tags on top.
"""
import numpy as np
import cv2


def _load_small(path, side=128):
    """Read RGB, downscale so the long side == `side` (fast, stable features)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:  # fall back to PIL for odd formats
        from PIL import Image
        bgr = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    s = side / max(h, w)
    if s < 1.0:
        bgr = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return bgr, (w, h)


def _kmeans_palette(rgb, k=5, iters=8):
    """Tiny numpy k-means over pixels -> (centers[k,3] uint8, weights[k] float)."""
    pix = rgb.reshape(-1, 3).astype(np.float32)
    if len(pix) > 4000:  # subsample for speed
        idx = np.linspace(0, len(pix) - 1, 4000).astype(int)
        pix = pix[idx]
    k = min(k, len(np.unique(pix, axis=0)))
    if k <= 1:
        c = pix.mean(0, keepdims=True)
        return c.astype(np.uint8), np.array([1.0], np.float32)
    # init: evenly spaced by luminance so clusters are reproducible (no RNG)
    lum = pix @ np.array([0.299, 0.587, 0.114], np.float32)
    order = np.argsort(lum)
    cents = pix[order[np.linspace(0, len(pix) - 1, k).astype(int)]].copy()
    for _ in range(iters):
        d = ((pix[:, None, :] - cents[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                cents[j] = pix[m].mean(0)
    counts = np.bincount(lab, minlength=k).astype(np.float32)
    w = counts / counts.sum()
    o = np.argsort(-w)
    return cents[o].astype(np.uint8), w[o]


def _colorfulness(rgb):
    """Hasler-Susstrunk colorfulness metric (grayscale ~0, vivid ~100+)."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def extract(path):
    """All local features for one image -> json-safe dict."""
    bgr, (W, H) = _load_small(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    centers, weights = _kmeans_palette(rgb)
    brightness = float(gray.mean() / 255.0)
    saturation = float(hsv[..., 1].mean() / 255.0)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())

    # dark theme = the dominant (most-weighted) palette color is dark
    dom = centers[0].astype(np.float32)
    dom_lum = float(dom @ np.array([0.299, 0.587, 0.114]) / 255.0)

    return {
        "w": W, "h": H, "aspect": round(W / H, 4),
        "palette": [[int(c[0]), int(c[1]), int(c[2])] for c in centers],
        "palette_w": [round(float(x), 4) for x in weights],
        "brightness": round(brightness, 4),
        "saturation": round(saturation, 4),
        "colorfulness": round(_colorfulness(rgb), 2),
        "edge_density": round(edge_density, 4),
        "dark": bool(dom_lum < 0.45),
        "dom_lum": round(dom_lum, 4),
    }


def color_vector(feat):
    """Compact color descriptor for similarity ranking: top-3 palette colors in a
    perceptually-evened space, weighted, + brightness/sat/colorfulness."""
    pal = np.array(feat["palette"][:3], np.float32) / 255.0
    w = np.array((feat["palette_w"] + [0, 0, 0])[:3], np.float32)
    w = w / (w.sum() + 1e-6)
    v = (pal * w[:, None]).reshape(-1)  # 9 dims
    extra = np.array([feat["brightness"], feat["saturation"],
                      min(feat["colorfulness"] / 100.0, 1.0), feat["edge_density"]], np.float32)
    return np.concatenate([v, extra]).astype(np.float32)
