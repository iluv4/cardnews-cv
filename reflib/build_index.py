"""Build the reference-library index (LOCAL tier, no torch).

For every image in the corpus, store filename metadata + visual features in
`reflib/data/index.json`. This is the foundation the RunPod tier augments:
  - tag_layout.py  -> adds `layout` (detector element counts/positions + signature)
  - embed_clip.py  -> writes clip.npy aligned to this index for semantic search

Usage:
  py -3 reflib/build_index.py                 # index images_new + images
  py -3 reflib/build_index.py --dirs images_new
  py -3 reflib/build_index.py --limit 20      # quick smoke test
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import features


def build(dirs, limit=None):
    paths = common.list_images(dirs)
    if limit:
        paths = paths[:limit]
    n = len(paths)
    if not n:
        raise SystemExit(f"no images found under {dirs} (cwd={common.ROOT})")

    print(f"indexing {n} images from {list(dirs)} ...")
    records, t0, fails = [], time.time(), 0
    for i, p in enumerate(paths, 1):
        meta = common.parse_name(p)
        try:
            feat = features.extract(p)
        except Exception as e:  # keep going; record the failure
            fails += 1
            print(f"  ! {os.path.basename(p)}: {e}")
            continue
        rec = {**meta, "path": os.path.relpath(p, common.ROOT).replace("\\", "/"), **feat}
        records.append(rec)
        if i % 50 == 0 or i == n:
            rate = i / (time.time() - t0 + 1e-6)
            print(f"  {i}/{n}  ({rate:.0f} img/s)")

    os.makedirs(common.INDEX_DIR, exist_ok=True)
    decks = sorted({r["deck"] for r in records})
    out = {
        "version": 1,
        "root": common.ROOT.replace("\\", "/"),
        "count": len(records),
        "n_decks": len(decks),
        "dirs": list(dirs),
        "has_clip": os.path.exists(common.EMB_PATH),
        "records": records,
    }
    with open(common.INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"\nwrote {common.INDEX_PATH}")
    print(f"  {len(records)} images, {len(decks)} decks, {fails} failures, "
          f"{time.time() - t0:.1f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=list(common.CORPUS_DIRS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(tuple(args.dirs), args.limit)


if __name__ == "__main__":
    main()
