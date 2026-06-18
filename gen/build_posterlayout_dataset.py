"""Stage 3+4 + CSV: turn raw card images into a PosterLayout-format Dataset/.

Single pass (each heavy model is loaded once):
  YOLO best.pt   -> element boxes  -> train_csv.csv (cls 1=text) + inpaint masks
  rembg u2net    -> saliencymaps_basnet/{stem}.png      (Stage 3, map A)
  rembg isnet    -> saliencymaps_pfpn/{stem}_pred.png   (Stage 3, map B)
  LaMa inpaint   -> train/inpainted_poster + test/image_canvas  (Stage 4)

Output (matches posterlayout_patches/infer.py + upstream dataloader):
  Dataset/
    train/{inpainted_poster, saliencymaps_basnet, saliencymaps_pfpn}/
    test/{image_canvas, saliencymaps_basnet, saliencymaps_pfpn}/
    train_csv.csv         # poster_path, cls_elem, box_elem (xyxy px on 513x750)

Run on RunPod GPU:
  pip install -r requirements-gen.txt
  python gen/build_posterlayout_dataset.py
"""
import os
import csv
import random
import argparse

from PIL import Image
import numpy as np

from common import W, H, ROOT, list_images, load_yolo, detect_boxes

TEST_FRAC = 0.10
SEED = 0
MASK_DILATE = 9  # px, grow element boxes so LaMa erases the glyph edges too


def saliency_runner(model_name):
    """Return fn(pil_rgb)->PIL 'L' map at WxH using rembg's matte for `model_name`."""
    from rembg import new_session, remove
    sess = new_session(model_name)

    def run(pil_rgb):
        out = remove(pil_rgb, session=sess, only_mask=True)  # 'L' alpha matte
        return out.convert("L").resize((W, H))
    return run


def lama_runner():
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()

    def run(pil_rgb, pil_mask_L):
        return lama(pil_rgb, pil_mask_L).convert("RGB").resize((W, H))
    return run


def boxes_to_mask(boxes):
    from PIL import ImageDraw
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for _cls, x0, y0, x1, y1 in boxes:
        d.rectangle([x0 - MASK_DILATE, y0 - MASK_DILATE,
                     x1 + MASK_DILATE, y1 + MASK_DILATE], fill=255)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", nargs="+", default=["images", "images_new"])
    ap.add_argument("--out", default=os.path.join(ROOT, "Dataset"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap #images (debug)")
    args = ap.parse_args()

    paths = list_images(*args.srcs)
    if args.limit:
        paths = paths[:args.limit]
    random.Random(SEED).shuffle(paths)
    n_test = max(1, int(len(paths) * TEST_FRAC))
    test_set = set(paths[:n_test])
    print(f"images: {len(paths)}  (train {len(paths)-n_test} / test {n_test})")

    d = {
        "tr_inp": os.path.join(args.out, "train", "inpainted_poster"),
        "tr_bas": os.path.join(args.out, "train", "saliencymaps_basnet"),
        "tr_pfpn": os.path.join(args.out, "train", "saliencymaps_pfpn"),
        "te_can": os.path.join(args.out, "test", "image_canvas"),
        "te_bas": os.path.join(args.out, "test", "saliencymaps_basnet"),
        "te_pfpn": os.path.join(args.out, "test", "saliencymaps_pfpn"),
    }
    for p in d.values():
        os.makedirs(p, exist_ok=True)

    print("loading models (YOLO + rembg u2net/isnet + LaMa)...")
    yolo = load_yolo(args.weights)
    sal_basnet = saliency_runner("u2net")
    sal_pfpn = saliency_runner("isnet-general-use")
    lama = lama_runner()

    csv_rows = []
    for i, p in enumerate(paths):
        canvas = Image.open(p).convert("RGB").resize((W, H))
        stem = f"img_{i:04d}"
        is_test = p in test_set

        # Stage 3: two saliency maps (both splits)
        bas = sal_basnet(canvas)
        pfpn = sal_pfpn(canvas)
        if is_test:
            bas.save(os.path.join(d["te_bas"], stem + ".png"))
            pfpn.save(os.path.join(d["te_pfpn"], stem + "_pred.png"))
        else:
            bas.save(os.path.join(d["tr_bas"], stem + ".png"))
            pfpn.save(os.path.join(d["tr_pfpn"], stem + "_pred.png"))

        # detect elements -> mask -> Stage 4 inpaint (clean background)
        boxes = detect_boxes(yolo, canvas)
        mask = boxes_to_mask(boxes)
        clean = lama(canvas, mask)
        if is_test:
            clean.save(os.path.join(d["te_can"], stem + ".png"))
        else:
            poster = stem + ".png"
            clean.save(os.path.join(d["tr_inp"], poster))
            for cls, x0, y0, x1, y1 in boxes:
                csv_rows.append([poster, cls,
                                 f"[{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]"])

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(paths)}")

    csv_path = os.path.join(args.out, "train_csv.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["poster_path", "cls_elem", "box_elem"])
        w.writerows(csv_rows)

    print(f"\nDone. {len(csv_rows)} element rows -> {csv_path}")
    print(f"Dataset/ ready at {args.out}")
    print("NOTE: detector only labels TEXT (1); logo(2)/underlay(3) need manual "
          "labels for richer layouts — text-only still trains/generates.")


if __name__ == "__main__":
    main()
