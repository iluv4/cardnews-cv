"""Stage 5 output: COMPOSE a card-news image from a generated layout.

Two modes:
  real : load DS-GAN output (boxes/clses .pt from infer.py) + clean backgrounds
         from Dataset/test/image_canvas, and render each layout.
  demo : no torch / no GPU needed. Synthesize a plausible layout and render it
         onto a real card background so you can SEE the generation end-to-end
         locally (PIL only).

Element classes: 1=text, 2=logo, 3=underlay.

  # local visual check (uses images_new backgrounds):
  py -3 gen/render.py --demo --n 4
  # after DS-GAN inference on RunPod:
  python gen/render.py --boxes PosterLayout-CVPR2023/output/boxes-Epoch300.pt \
                       --clses PosterLayout-CVPR2023/output/clses-Epoch300.pt \
                       --canvas Dataset/test/image_canvas
"""
import os
import glob
import random
import argparse

from PIL import Image, ImageDraw, ImageFilter

from common import W, H, ROOT, list_images, find_korean_font

OUT_DIR = os.path.join(ROOT, "gen_output")

# sample copy used to fill text boxes in demo mode
SAMPLE_TITLE = ["스마트팜으로\n바뀌는 농업", "알아두면\n쓸모있는 정보", "2024\n카드뉴스", "오늘의\n핵심 요약"]
SAMPLE_BODY = [
    "복잡한 내용을 한 장에 정리했습니다. 핵심만 빠르게 확인하세요.",
    "데이터 기반으로 정리한 내용입니다. 자세한 내용은 본문을 참고하세요.",
    "누구나 이해하기 쉽게 풀어 설명합니다. 끝까지 확인해 주세요.",
]


def _wrap(draw, text, font, max_w):
    """Greedy wrap honoring explicit \n; wraps long lines to max_w px."""
    lines = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            t = cur + ch
            if draw.textlength(t, font=font) <= max_w or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = ch
        lines.append(cur)
    return lines


def draw_text_box(img, box, text, accent=(20, 20, 20), align_center=False):
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    size = max(14, min(int(bh * 0.6), 60))
    font = find_korean_font(size)
    # shrink until it fits the box height
    for _ in range(8):
        lines = _wrap(draw, text, font, bw)
        lh = (font.getbbox("가")[3] - font.getbbox("가")[1]) + 6
        if lh * len(lines) <= bh or size <= 14:
            break
        size = int(size * 0.85)
        font = find_korean_font(size)
    y = y0
    for ln in lines:
        tw = draw.textlength(ln, font=font)
        x = x0 + (bw - tw) / 2 if align_center else x0
        # subtle shadow for readability over photos
        draw.text((x + 1, y + 1), ln, font=font, fill=(255, 255, 255))
        draw.text((x, y), ln, font=font, fill=accent)
        y += (font.getbbox("가")[3] - font.getbbox("가")[1]) + 6


def draw_underlay(img, box, color=(255, 255, 255), alpha=170, radius=18):
    x0, y0, x1, y1 = [int(v) for v in box]
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle([x0, y0, x1, y1], radius=radius,
                                              fill=color + (alpha,))
    img.alpha_composite(overlay)


def draw_logo(img, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    d = ImageDraw.Draw(img)
    d.ellipse([x0, y0, x1, y1], fill=(0, 0, 0, 200))
    f = find_korean_font(max(12, int((y1 - y0) * 0.4)))
    t = "LOGO"
    tw = d.textlength(t, font=f)
    d.text((x0 + ((x1 - x0) - tw) / 2, y0 + (y1 - y0) * 0.28), t, font=f, fill=(255, 255, 255))


def render_layout(bg_rgb, elements, texts=None):
    """elements: list of (cls, x0,y0,x1,y1) in 513x750 px. Returns composited RGB."""
    img = bg_rgb.convert("RGBA").resize((W, H))
    texts = texts or {}
    # underlays first (behind), then logos, then text on top
    order = sorted(elements, key=lambda e: {3: 0, 2: 1, 1: 2}.get(e[0], 3))
    ti = 0
    for cls, x0, y0, x1, y1 in order:
        box = [x0, y0, x1, y1]
        if cls == 3:
            draw_underlay(img, box)
        elif cls == 2:
            draw_logo(img, box)
        elif cls == 1:
            txt = texts.get("title") if (y1 - y0) > H * 0.06 and "title" not in texts.get("_used", []) else None
            if txt is None:
                txt = texts.get(ti, "내용 텍스트")
            draw_text_box(img, box, txt, align_center=(y0 < H * 0.4))
            ti += 1
    return img.convert("RGB")


# ---------------- demo (no torch) ----------------
def demo_layout(seed):
    r = random.Random(seed)
    j = lambda v: v + r.randint(-12, 12)
    underlay = (3, j(40), j(95), j(473), j(235))
    title = (1, 60, 110, 453, 215)
    body = (1, 60, j(270), 453, j(380))
    logo = (2, 40, 672, 150, 722)
    return [underlay, title, body, logo]


def run_demo(n):
    os.makedirs(os.path.join(OUT_DIR, "demo"), exist_ok=True)
    bgs = list_images("images_new") or list_images("images")
    if not bgs:
        raise SystemExit("no backgrounds found in images_new/ or images/")
    random.Random(0).shuffle(bgs)
    for i in range(min(n, len(bgs))):
        bg = Image.open(bgs[i]).convert("RGB").resize((W, H))
        # Stand-in for LaMa: blur + white veil so the source card's own text is
        # muted, approximating the clean background the real pipeline inpaints.
        bg = bg.filter(ImageFilter.GaussianBlur(7))
        bg = Image.blend(bg, Image.new("RGB", (W, H), (255, 255, 255)), 0.45)
        elems = demo_layout(i)
        texts = {"title": SAMPLE_TITLE[i % len(SAMPLE_TITLE)],
                 0: SAMPLE_TITLE[i % len(SAMPLE_TITLE)],
                 1: SAMPLE_BODY[i % len(SAMPLE_BODY)]}
        out = render_layout(bg, elems, texts)
        op = os.path.join(OUT_DIR, "demo", f"demo_{i:02d}.png")
        out.save(op)
        print(f"  wrote {op}")
    print(f"demo images -> {os.path.join(OUT_DIR, 'demo')}")


# ---------------- real (DS-GAN output) ----------------
def run_real(boxes_pt, clses_pt, canvas_dir):
    import torch
    boxes = torch.load(boxes_pt, map_location="cpu")  # [N, max_elem, 4] normalized xyxy
    clses = torch.load(clses_pt, map_location="cpu")  # [N, max_elem, 1] argmax ids
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
            x0, y0, x1, y1 = boxes[i, e].tolist()
            elems.append((cls, x0 * W, y0 * H, x1 * W, y1 * H))
        out = render_layout(bg, elems, {"title": "제목 텍스트", 1: "본문 텍스트"})
        op = os.path.join(OUT_DIR, "real", f"gen_{i:03d}_{os.path.basename(bgs[i])}.png")
        out.save(op)
    print(f"generated card-news -> {os.path.join(OUT_DIR, 'real')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--boxes", default="PosterLayout-CVPR2023/output/boxes-Epoch300.pt")
    ap.add_argument("--clses", default="PosterLayout-CVPR2023/output/clses-Epoch300.pt")
    ap.add_argument("--canvas", default="Dataset/test/image_canvas")
    args = ap.parse_args()
    if args.demo:
        run_demo(args.n)
    else:
        run_real(os.path.join(ROOT, args.boxes) if not os.path.isabs(args.boxes) else args.boxes,
                 os.path.join(ROOT, args.clses) if not os.path.isabs(args.clses) else args.clses,
                 os.path.join(ROOT, args.canvas) if not os.path.isabs(args.canvas) else args.canvas)


if __name__ == "__main__":
    main()
