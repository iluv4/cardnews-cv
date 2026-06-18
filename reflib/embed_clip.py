"""CLIP embeddings for semantic search  (RunPod / any torch machine).

Builds an image embedding per indexed reference, aligned to reflib/data/index.json,
and writes:
  reflib/data/clip.npy       float32 [N, D]   (L2-normalized)
  reflib/data/clip_ids.json  list[str]        (record ids, same row order)

search.py picks these up automatically (semantic similar_to). For semantic TEXT
search, run this module with --text to encode a query and rank against the saved
image embeddings (needs torch where the query runs, e.g. the API host or RunPod).

Setup (RunPod):
  pip install open_clip_torch torch torchvision pillow numpy
Build:
  python reflib/embed_clip.py --build
Query:
  python reflib/embed_clip.py --text "스마트팜 청년 지원 카드뉴스" --k 8
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k")


def _load_model():
    import torch
    import open_clip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(dev).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, dev


def build(batch=32):
    import torch
    from PIL import Image
    index = json.load(open(common.INDEX_PATH, encoding="utf-8"))
    recs = index["records"]
    model, preprocess, _, dev = _load_model()

    embs, ids = [], []
    buf, buf_ids = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(dev)
        with torch.no_grad():
            f = model.encode_image(x).float()
            f /= f.norm(dim=-1, keepdim=True) + 1e-9
        embs.append(f.cpu().numpy())
        ids.extend(buf_ids)
        buf.clear(); buf_ids.clear()

    for i, r in enumerate(recs, 1):
        p = os.path.join(common.ROOT, r["path"])
        try:
            buf.append(preprocess(Image.open(p).convert("RGB")))
            buf_ids.append(r["id"])
        except Exception as e:
            print(f"  ! {r['id']}: {e}")
        if len(buf) >= batch:
            flush()
        if i % 100 == 0:
            print(f"  {i}/{len(recs)}")
    flush()

    E = np.concatenate(embs, 0).astype(np.float32)
    os.makedirs(common.INDEX_DIR, exist_ok=True)
    np.save(common.EMB_PATH, E)
    json.dump(ids, open(common.EMB_IDS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    index["has_clip"] = True
    json.dump(index, open(common.INDEX_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {common.EMB_PATH}  shape={E.shape}")


def encode_text(texts):
    """-> float32 [len(texts), D], L2-normalized."""
    import torch
    model, _, tokenizer, dev = _load_model()
    with torch.no_grad():
        f = model.encode_text(tokenizer(texts).to(dev)).float()
        f /= f.norm(dim=-1, keepdim=True) + 1e-9
    return f.cpu().numpy().astype(np.float32)


def text_search(query, k=8):
    E = np.load(common.EMB_PATH).astype(np.float32)
    ids = json.load(open(common.EMB_IDS_PATH, encoding="utf-8"))
    q = encode_text([query])[0]
    sims = E @ q
    order = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--text")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    if args.build:
        build(args.batch)
    if args.text:
        for rid, s in text_search(args.text, args.k):
            print(f"  {s:.4f}  {rid}")
    if not args.build and not args.text:
        ap.print_help()


if __name__ == "__main__":
    main()
