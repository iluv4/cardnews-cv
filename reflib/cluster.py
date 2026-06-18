"""Cluster the indexed references into visual archetypes and write the cluster id
back into each record (`cluster` field) + a per-cluster summary.

Clusters on the richest signal available:
  - CLIP embeddings (reflib/data/clip.npy) if the RunPod embed pass has run,
  - otherwise the local color/feel vector from features.color_vector.

Usage:
  py -3 reflib/cluster.py --k 8
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import features


def _kmeans(X, k, iters=50):
    """Deterministic k-means (k-means++-ish seeding without RNG)."""
    n = len(X)
    k = min(k, n)
    # seed: farthest-point sampling from the global mean
    cents = [int(np.argmax(((X - X.mean(0)) ** 2).sum(1)))]
    while len(cents) < k:
        d = np.min([((X - X[c]) ** 2).sum(1) for c in cents], axis=0)
        cents.append(int(np.argmax(d)))
    C = X[cents].copy()
    lab = np.zeros(n, int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = X[m].mean(0)
    return lab, C


def load_vectors(index):
    """Return (matrix[N,D], source_str). CLIP if present else local color vector."""
    if os.path.exists(common.EMB_PATH) and os.path.exists(common.EMB_IDS_PATH):
        emb = np.load(common.EMB_PATH).astype(np.float32)
        ids = json.load(open(common.EMB_IDS_PATH, encoding="utf-8"))
        by_id = {i: emb[r] for r, i in enumerate(ids)}
        rows = [by_id[r["id"]] for r in index["records"] if r["id"] in by_id]
        if len(rows) == len(index["records"]):
            X = np.array(rows, np.float32)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            return X, "clip"
    X = np.array([features.color_vector(r) for r in index["records"]], np.float32)
    # standardize so no single feature dominates
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return X, "color"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    index = json.load(open(common.INDEX_PATH, encoding="utf-8"))
    X, src = load_vectors(index)
    lab, _ = _kmeans(X, args.k)
    for r, l in zip(index["records"], lab):
        r["cluster"] = int(l)

    summary = {}
    for r in index["records"]:
        c = r["cluster"]
        s = summary.setdefault(c, {"n": 0, "dark": 0, "decks": set(), "tokens": {}})
        s["n"] += 1
        s["dark"] += int(r["dark"])
        s["decks"].add(r["deck"])
        for t in r.get("topic_tokens", []):
            s["tokens"][t] = s["tokens"].get(t, 0) + 1
    index["clusters"] = {
        str(c): {"n": s["n"], "dark_frac": round(s["dark"] / s["n"], 2),
                 "n_decks": len(s["decks"]),
                 "top_tokens": [t for t, _ in sorted(s["tokens"].items(), key=lambda x: -x[1])[:6]]}
        for c, s in sorted(summary.items())
    }
    index["cluster_basis"] = src
    json.dump(index, open(common.INDEX_PATH, "w", encoding="utf-8"), ensure_ascii=False)

    print(f"clustered {len(index['records'])} refs into {args.k} archetypes (basis={src})")
    for c, s in index["clusters"].items():
        print(f"  #{c}: n={s['n']:>3}  dark={s['dark_frac']}  "
              f"tokens={', '.join(s['top_tokens'][:5])}")


if __name__ == "__main__":
    main()
