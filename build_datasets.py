"""Build a LEAK-FREE comparison setup that isolates ONE variable: dataset size.

Pool = images/ (109 originals) + images_new/ (578 new).
We hold out a COMMON TEST set (15% of the NEW images) that BOTH models are
evaluated on, and that NEITHER model trains on. Then:

    base109  trains on: 109 originals
    full687  trains on: 109 originals + (new - common_test)   (~601)
    both     eval  on : common_test  (~86 new images, unseen by both)

Because the two YOLO configs share the SAME val list (common_test), their mAP is
directly comparable; because common_test is excluded from both training sets,
there is no train/test leakage (base109 never sees any new image at all).

Labels: EasyOCR pseudo-labels, identical rules to autolabel.py
    text region -> class 0 (title) if height >= 4.5% of image, else 1 (body).
    logo (2) / underlay (3) not auto-labelled (manual).

Run on RunPod (needs torch + easyocr + GPU):
    pip install easyocr pillow numpy
    python build_datasets.py
Outputs: data_cmp/ (images, labels, *.txt splits) + dataset_base109.yaml +
dataset_full687.yaml + data_cmp/mapping.csv
"""
import os
import csv
import random
import numpy as np
from PIL import Image
import easyocr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data_cmp")
IMG_DIR = os.path.join(OUT, "images")
LBL_DIR = os.path.join(OUT, "labels")

ORIG_DIR = "images"        # 109 originals
NEW_DIR = "images_new"     # 578 new (run prepare_new_images.py first)

CONF = 0.30
TITLE_H_RATIO = 0.045
TEST_FRAC = 0.15           # share of NEW images held out as the common test set
SEED = 0
EXTS = (".png", ".jpg", ".jpeg")

_reader = None


def reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["ko", "en"], gpu=True)
    return _reader


def list_dir(d):
    ad = os.path.join(ROOT, d)
    if not os.path.isdir(ad):
        raise SystemExit(f"missing source dir: {ad}")
    return sorted(e.path for e in os.scandir(ad)
                  if e.is_file() and e.name.lower().endswith(EXTS))


def label_lines(img):
    W, H = img.size
    results = reader().readtext(np.array(img))
    lines, nt, nb = [], 0, 0
    for bbox, _text, conf in results:
        if conf < CONF:
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x0, x1 = max(0, min(xs)), min(W, max(xs))
        y0, y1 = max(0, min(ys)), min(H, max(ys))
        bw, bh = x1 - x0, y1 - y0
        if bw <= 1 or bh <= 1:
            continue
        cls = 0 if (bh / H) >= TITLE_H_RATIO else 1
        cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
        nt += cls == 0
        nb += cls == 1
    return lines, nt, nb


def encode_all(orig, new):
    """Encode every image once into data_cmp/images + labels. Returns dict
    src_path -> (image_abs_path, role) where role in {'orig','new'}."""
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)
    out, rows = {}, []
    tot = {"title": 0, "body": 0, "empty": 0}
    items = [(p, "orig") for p in orig] + [(p, "new") for p in new]
    for i, (p, role) in enumerate(items):
        img = Image.open(p).convert("RGB")
        lines, nt, nb = label_lines(img)
        stem = f"img_{i:04d}"
        ip = os.path.join(IMG_DIR, stem + ".jpg")
        img.save(ip, quality=92)
        with open(os.path.join(LBL_DIR, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines))
        out[p] = (ip, role)
        tot["title"] += nt
        tot["body"] += nb
        tot["empty"] += not lines
        rows.append([stem + ".jpg", os.path.basename(p), role, len(lines)])
        if (i + 1) % 50 == 0:
            print(f"  labelled {i+1}/{len(items)}")
    print(f"boxes: title {tot['title']}  body {tot['body']}  empty imgs {tot['empty']}")
    return out, rows


def write_list(path, img_paths):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(p.replace(os.sep, "/") for p in img_paths))


def write_yaml(name, train_txt, val_txt):
    p = os.path.join(ROOT, name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(
            f"path: {OUT.replace(os.sep, '/')}\n"
            f"train: {os.path.basename(train_txt)}\n"
            f"val: {os.path.basename(val_txt)}\n"
            "names:\n  0: title\n  1: body\n  2: logo\n  3: underlay\n"
        )
    return p


def main():
    orig = list_dir(ORIG_DIR)
    new = list_dir(NEW_DIR)
    print(f"originals: {len(orig)}  new: {len(new)}")

    # deterministic common test set = TEST_FRAC of NEW images
    new_shuf = new[:]
    random.Random(SEED).shuffle(new_shuf)
    n_test = max(1, int(len(new_shuf) * TEST_FRAC))
    test_new = set(new_shuf[:n_test])
    train_new = [p for p in new if p not in test_new]

    enc, rows = encode_all(orig, new)

    test_imgs = [enc[p][0] for p in new if p in test_new]
    base_train = [enc[p][0] for p in orig]                       # 109
    full_train = [enc[p][0] for p in orig] + [enc[p][0] for p in train_new]

    common_test = os.path.join(OUT, "common_test.txt")
    base_txt = os.path.join(OUT, "base109_train.txt")
    full_txt = os.path.join(OUT, "full687_train.txt")
    write_list(common_test, test_imgs)
    write_list(base_txt, base_train)
    write_list(full_txt, full_train)

    yb = write_yaml("dataset_base109.yaml", base_txt, common_test)
    yf = write_yaml("dataset_full687.yaml", full_txt, common_test)

    # traceability: append split role to mapping
    by_name = {r[0]: r for r in rows}
    for p in new:
        nm = os.path.basename(enc[p][0])
        if nm in by_name:
            by_name[nm].append("test" if p in test_new else "train")
    with open(os.path.join(OUT, "mapping.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["yolo_name", "original", "role", "n_boxes", "new_split"])
        for r in rows:
            w.writerow(r + ([] if len(r) > 4 else [""]))

    print(f"\ncommon_test: {len(test_imgs)}   base_train: {len(base_train)}   "
          f"full_train: {len(full_train)}")
    print(f"yamls: {yb}\n       {yf}")
    print("next:  python compare_data_scale.py")


if __name__ == "__main__":
    main()
