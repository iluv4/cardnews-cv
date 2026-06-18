"""Fast, classical saliency/busyness analysis for photo backgrounds (no GPU).

Used to (a) choose which template band (top/center/bottom) sits over the calmest
part of a photo, and (b) decide text color from region luminance.
"""
import numpy as np
import cv2


def busyness_profile(pil_rgb, bands=3):
    """Mean gradient magnitude per horizontal band. Lower = calmer = safer text."""
    g = cv2.cvtColor(np.array(pil_rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (256, 256))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    h = mag.shape[0]
    return [float(mag[i * h // bands:(i + 1) * h // bands].mean()) for i in range(bands)]


def calmest_band(pil_rgb):
    """Return 'top' | 'center' | 'bottom' — the calmest third for text."""
    p = busyness_profile(pil_rgb, 3)
    return ["top", "center", "bottom"][int(np.argmin(p))]


def region_luma(pil_rgb, box_px):
    x0, y0, x1, y1 = [int(v) for v in box_px]
    img = pil_rgb.convert("RGB")
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, max(x0 + 1, x1)), min(img.height, max(y0 + 1, y1))
    crop = np.asarray(img.crop((x0, y0, x1, y1)).convert("L"))
    return float(crop.mean()) if crop.size else 128.0
