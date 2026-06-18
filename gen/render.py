"""Stage 5 output: COMPOSE a designed card-news image from a generated layout.

Quality goals (per design feedback):
  - High resolution: 1080x1350 (Instagram 4:5), not the 513x750 model canvas.
  - Real Korean designer font: bundled Pretendard (Bold for titles, Regular body).
  - Readable on ANY background: luminance-aware text color + a soft scrim panel
    behind text (also visually covers any leftover source text).
  - Type hierarchy: large bold title vs. smaller regular body, auto-fit + spacing.

Element classes: 1=text, 2=logo, 3=underlay. Box coords are normalized [0,1].

Modes:
  demo : no torch/GPU. Synthesize a layout and render onto a real card so you can
         SEE the output locally.   ->  py -3 gen/render.py --demo --n 4
  real : load DS-GAN output (boxes/clses .pt) + clean backgrounds.
         python gen/render.py --boxes .../boxes-Epoch300.pt \
                --clses .../clses-Epoch300.pt --canvas Dataset/test/image_canvas
"""
import os
import glob
import random
import argparse

from PIL import Image, ImageDraw, ImageFilter, ImageStat

from common import ROOT, list_images, find_korean_font

RW, RH = 1080, 1350           # render resolution (Instagram portrait 4:5)
OUT_DIR = os.path.join(ROOT, "gen_output")

SAMPLE_TITLE = ["스마트팜으로\n바뀌는 농업", "알아두면\n쓸모있는 정보",
                "2024 청년\n농업 지원 사업", "지금 꼭 알아야 할\n핵심 요약"]
SAMPLE_BODY = [
    "복잡한 내용을 한 장에 정리했습니다.\n핵심만 빠르게 확인하세요.",
    "데이터로 정리한 내용입니다.\n자세한 사항은 본문을 참고하세요.",
    "누구나 이해하기 쉽게 풀었습니다.\n끝까지 확인해 주세요.",
]


# ---------- helpers ----------
def region_luma(img_rgb, box_px):
    x0, y0, x1, y1 = [int(v) for v in box_px]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_rgb.width, max(x0 + 1, x1)), min(img_rgb.height, max(y0 + 1, y1))
    crop = img_rgb.crop((x0, y0, x1, y1)).convert("L")
    return ImageStat.Stat(crop).mean[0]


def wrap(draw, text, font, max_w):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            t = cur + ch
            if draw.textlength(t, font=font) <= max_w or not cur:
                cur = t
            else:
                lines.append(cur); cur = ch
        lines.append(cur)
    return lines


def fit_font(draw, text, box_w, box_h, weight, start_frac, min_px=18, lh=1.18):
    size = max(min_px, int(box_h * start_frac))
    while size > min_px:
        font = find_korean_font(size, weight)
        lines = wrap(draw, text, font, box_w)
        asc, desc = font.getmetrics()
        line_h = (asc + desc) * lh
        if line_h * len(lines) <= box_h and all(draw.textlength(l, font=font) <= box_w for l in lines):
            return font, lines, line_h
        size = int(size * 0.92)
    font = find_korean_font(min_px, weight)
    return font, wrap(draw, text, font, box_w), (sum(font.getmetrics())) * lh


def draw_scrim(img_rgba, box_px, dark, pad=0.10, radius=28, alpha=150):
    x0, y0, x1, y1 = box_px
    pw = (x1 - x0) * pad
    ph = (y1 - y0) * pad
    rect = [x0 - pw, y0 - ph, x1 + pw, y1 + ph]
    layer = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    col = (0, 0, 0, alpha) if dark else (255, 255, 255, int(alpha * 1.25))
    ImageDraw.Draw(layer).rounded_rectangle(rect, radius=radius, fill=col)
    layer = layer.filter(ImageFilter.GaussianBlur(2))
    img_rgba.alpha_composite(layer)


def draw_text_block(img_rgba, box_px, text, weight, start_frac, center, accent=None):
    """Luminance-aware text with scrim. accent = optional RGB for a left bar (title)."""
    draw = ImageDraw.Draw(img_rgba)
    x0, y0, x1, y1 = box_px
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    luma = region_luma(img_rgba.convert("RGB"), box_px)
    bg_dark = luma < 130
    # scrim is the inverse of the chosen text color -> guaranteed contrast + covers
    # any residual source text underneath the placement region.
    draw_scrim(img_rgba, box_px, dark=bg_dark)
    fg = (255, 255, 255) if bg_dark else (26, 26, 26)

    font, lines, line_h = fit_font(draw, text, bw, bh, weight, start_frac)
    total_h = line_h * len(lines)
    y = y0 + (bh - total_h) / 2
    if accent is not None and lines:
        bar_w = max(4, int(bw * 0.012))
        draw.rounded_rectangle([x0 - bar_w * 2, y + 4, x0 - bar_w, y + total_h - 4],
                               radius=bar_w, fill=accent)
    for ln in lines:
        tw = draw.textlength(ln, font=font)
        x = x0 + (bw - tw) / 2 if center else x0
        draw.text((x, y), ln, font=font, fill=fg)
        y += line_h


def draw_logo(img_rgba, box_px):
    x0, y0, x1, y1 = [int(v) for v in box_px]
    d = ImageDraw.Draw(img_rgba)
    d.ellipse([x0, y0, x1, y1], fill=(17, 17, 17, 220))
    f = find_korean_font(max(14, int((y1 - y0) * 0.34)), "Bold")
    t = "LOGO"
    tw = d.textlength(t, font=f)
    asc, desc = f.getmetrics()
    d.text((x0 + ((x1 - x0) - tw) / 2, y0 + ((y1 - y0) - (asc + desc)) / 2),
           t, font=f, fill=(255, 255, 255))


def draw_underlay(img_rgba, box_px):
    x0, y0, x1, y1 = [int(v) for v in box_px]
    layer = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle([x0, y0, x1, y1], radius=32,
                                            fill=(255, 255, 255, 165))
    img_rgba.alpha_composite(layer)


def render_card(bg_rgb, elements, texts=None):
    """elements: list of (cls, x0,y0,x1,y1) normalized [0,1]. Returns RGB 1080x1350."""
    img = bg_rgb.convert("RGB").resize((RW, RH)).convert("RGBA")
    texts = texts or {}
    px = lambda b: (b[0] * RW, b[1] * RH, b[2] * RW, b[3] * RH)
    order = sorted(elements, key=lambda e: {3: 0, 2: 1, 1: 2}.get(e[0], 3))
    ti = 0
    for cls, x0, y0, x1, y1 in order:
        box = px((x0, y0, x1, y1))
        if cls == 3:
            draw_underlay(img, box)
        elif cls == 2:
            draw_logo(img, box)
        elif cls == 1:
            is_title = (y1 - y0) >= 0.06 and y0 < 0.40 and ti == 0
            txt = texts.get("title" if is_title else ti) or texts.get(ti) \
                or ("제목 텍스트" if is_title else "본문 텍스트입니다.")
            draw_text_block(img, box, txt,
                            weight="Bold" if is_title else "Regular",
                            start_frac=0.62 if is_title else 0.34,
                            center=is_title,
                            accent=(255, 90, 60) if is_title else None)
            ti += 1
    return img.convert("RGB")


# ---------- demo (no torch) ----------
def demo_layout(seed):
    r = random.Random(seed)
    j = lambda v, s=0.02: min(0.97, max(0.03, v + r.uniform(-s, s)))
    return [
        (1, 0.10, 0.11, 0.90, 0.27),           # title (its scrim acts as the panel)
        (1, 0.13, j(0.55), 0.87, j(0.71)),     # body
        (2, 0.06, 0.885, 0.17, 0.955),         # logo
    ]


def run_demo(n):
    os.makedirs(os.path.join(OUT_DIR, "demo"), exist_ok=True)
    bgs = list_images("images_new") or list_images("images")
    if not bgs:
        raise SystemExit("no backgrounds in images_new/ or images/")
    random.Random(7).shuffle(bgs)
    for i in range(min(n, len(bgs))):
        bg = Image.open(bgs[i]).convert("RGB").resize((RW, RH))
        # local stand-in for LaMa: blur source so its own text is muted (the real
        # pipeline inpaints the background cleanly at full res).
        bg = bg.filter(ImageFilter.GaussianBlur(6))
        texts = {"title": SAMPLE_TITLE[i % len(SAMPLE_TITLE)],
                 1: SAMPLE_BODY[i % len(SAMPLE_BODY)]}
        out = render_card(bg, demo_layout(i), texts)
        op = os.path.join(OUT_DIR, "demo", f"demo_{i:02d}.png")
        out.save(op)
        print(f"  wrote {op}")
    print(f"demo -> {os.path.join(OUT_DIR, 'demo')}")


# ---------- real (DS-GAN output) ----------
def run_real(boxes_pt, clses_pt, canvas_dir):
    import torch
    boxes = torch.load(boxes_pt, map_location="cpu")
    clses = torch.load(clses_pt, map_location="cpu")
    boxes = boxes.numpy() if hasattr(boxes, "numpy") else boxes
    clses = clses.numpy() if hasattr(clses, "numpy") else clses
    bgs = sorted(glob.glob(os.path.join(canvas_dir, "*.png")) +
                 glob.glob(os.path.join(canvas_dir, "*.jpg")))
    os.makedirs(os.path.join(OUT_DIR, "real"), exist_ok=True)
    n = min(len(bgs), boxes.shape[0])
    print(f"rendering {n} generated layouts")
    for i in range(n):
        bg = Image.open(bgs[i]).convert("RGB")
        elems = []
        for e in range(boxes.shape[1]):
            cls = int(clses[i, e].reshape(-1)[0])
            if cls == 0:
                continue
            x0, y0, x1, y1 = [float(v) for v in boxes[i, e].tolist()]
            elems.append((cls, min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
        out = render_card(bg, elems)
        out.save(os.path.join(OUT_DIR, "real", f"gen_{i:03d}.png"))
    print(f"generated card-news -> {os.path.join(OUT_DIR, 'real')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--boxes", default="PosterLayout-CVPR2023/output/boxes-Epoch300.pt")
    ap.add_argument("--clses", default="PosterLayout-CVPR2023/output/clses-Epoch300.pt")
    ap.add_argument("--canvas", default="Dataset/test/image_canvas")
    args = ap.parse_args()
    ab = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
    if args.demo:
        run_demo(args.n)
    else:
        run_real(ab(args.boxes), ab(args.clses), ab(args.canvas))


if __name__ == "__main__":
    main()
