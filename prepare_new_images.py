"""Flatten the new Korean card-news folders into repo-local `images_new/`.

Each source is a directory of *series* sub-folders, each holding the pages of one
card-news deck (1.jpg, 2.jpg, ...). We flatten every page into a single folder
with ASCII filenames + RGB JPEG (so the new images travel inside the git repo and
Ultralytics' cv2.imread works on Windows / cloud regardless of the original
Korean filenames).

Mirrors autolabel.py's handling: open with PIL, convert RGB, re-encode JPEG.
We do NOT resize here — the YOLO detector resizes internally (imgsz); the 513x750
PosterLayout canvas is applied later in the generation pipeline.

Run locally (only needs Pillow):
    py -3 prepare_new_images.py
"""
import os
import re
import csv
import unicodedata
from PIL import Image

# --- sources (the two KakaoTalk folders) -------------------------------------
SOURCES = [
    r"C:\Users\SmucUserC316\Documents\카카오톡 받은 파일\newData\newData",
    r"C:\Users\SmucUserC316\Documents\카카오톡 받은 파일\newData(smartFarmKorea)",
]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images_new")
EXTS = (".png", ".jpg", ".jpeg")
MAX_SIDE = 2048  # cap longest side (matches the existing committed images' scale)


def slugify(name: str, fallback: str) -> str:
    """ASCII-only, filesystem-safe slug. Falls back when nothing ASCII remains."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    s = re.sub(r"_+", "_", s)
    return (s[:40] or fallback)


def page_key(path: str):
    """Natural sort by the first number in the filename (1 < 2 < 10)."""
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    return (int(nums[0]) if nums else 10**9, base.lower())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    n_series = 0
    n_pages = 0
    n_skipped = 0

    for src in SOURCES:
        if not os.path.isdir(src):
            print(f"!! source not found: {src}")
            continue
        src_tag = slugify(os.path.basename(src.rstrip("\\/")), "src")
        # os.scandir (not glob): folder/file names contain '[', ']' which glob
        # would mis-read as character classes and silently drop those series.
        series_dirs = sorted(
            e.path for e in os.scandir(src) if e.is_dir()
        )
        for sdir in series_dirs:
            pages = [e.path for e in os.scandir(sdir)
                     if e.is_file() and e.name.lower().endswith(EXTS)]
            if not pages:
                continue
            pages.sort(key=page_key)
            slug = slugify(os.path.basename(sdir), f"series{n_series:03d}")
            for pi, p in enumerate(pages, 1):
                out_name = f"{src_tag}__{n_series:03d}_{slug}__p{pi:02d}.jpg"
                out_path = os.path.join(OUT_DIR, out_name)
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        if max(im.size) > MAX_SIDE:
                            r = MAX_SIDE / max(im.size)
                            im = im.resize((round(im.width * r), round(im.height * r)),
                                           Image.LANCZOS)
                        im.save(out_path, "JPEG", quality=92)
                except Exception as e:  # noqa: BLE001 - report and continue
                    n_skipped += 1
                    print(f"  skip (unreadable): {p}  [{e}]")
                    continue
                rows.append([out_name, src_tag, os.path.basename(sdir), os.path.basename(p), pi])
                n_pages += 1
            n_series += 1

    map_path = os.path.join(OUT_DIR, "_mapping_new.csv")
    with open(map_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["new_name", "source", "series_folder", "original_file", "page_idx"])
        w.writerows(rows)

    print(f"\nseries: {n_series}  pages written: {n_pages}  skipped: {n_skipped}")
    print(f"out: {OUT_DIR}")
    print(f"map: {map_path}")


if __name__ == "__main__":
    main()
