"""Visual QA for the template library: overlay extracted title/body blocks on
each reference thumbnail and tile them into a contact sheet.

  py -3 gen/preview_templates.py [--n 24]  -> service/library/contact_sheet.png
"""
import os
import json
import argparse

from PIL import Image, ImageDraw

from common import ROOT
from extract_templates import OUT_DIR

TITLE_COL = (244, 70, 90)
BODY_COL = (60, 200, 220)


def label_font(size):
    from cardgen.assets import font as _f  # reuse engine fonts
    return _f(size, "Bold")


def thumb(t, w=300):
    src = os.path.join(ROOT, t["source"])
    img = Image.open(src).convert("RGB")
    iw, ih = img.size
    h = int(w * ih / iw)
    img = img.resize((w, h))
    d = ImageDraw.Draw(img, "RGBA")
    for role, col in (("title", TITLE_COL), ("body", BODY_COL)):
        for b in t["blocks"][role]:
            d.rectangle([b[0] * w, b[1] * h, b[2] * w, b[3] * h],
                        outline=col + (255,), width=3)
    # caption bar
    d.rectangle([0, 0, w, 26], fill=(0, 0, 0, 150))
    try:
        import sys
        sys.path.insert(0, os.path.join(ROOT, "service"))
        f = label_font(18)
    except Exception:
        f = None
    cap = f"{t['id']}  {t['archetype']}  T{t['n_title']}/B{t['n_body']}"
    d.text((6, 4), cap, fill=(255, 255, 255), font=f)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    args = ap.parse_args()

    templates = json.load(open(os.path.join(OUT_DIR, "templates.json"),
                               encoding="utf-8"))
    # sample across archetypes so the sheet is representative
    by_a = {}
    for t in templates:
        by_a.setdefault(t["archetype"], []).append(t)
    picked, i = [], 0
    while len(picked) < min(args.n, len(templates)):
        for a in sorted(by_a):
            if i < len(by_a[a]):
                picked.append(by_a[a][i])
            if len(picked) >= args.n:
                break
        i += 1
        if i > 200:
            break

    thumbs = [thumb(t) for t in picked]
    cw = max(t.width for t in thumbs)
    ch = max(t.height for t in thumbs)
    cols = args.cols
    rows = (len(thumbs) + cols - 1) // cols
    gap = 10
    sheet = Image.new("RGB", (cols * cw + gap * (cols + 1),
                              rows * ch + gap * (rows + 1)), (250, 250, 250))
    for k, im in enumerate(thumbs):
        r, c = divmod(k, cols)
        sheet.paste(im, (gap + c * (cw + gap), gap + r * (ch + gap)))
    out = os.path.join(OUT_DIR, "contact_sheet.png")
    sheet.save(out)
    print(f"contact sheet ({len(thumbs)} refs) -> {out}")


if __name__ == "__main__":
    main()
