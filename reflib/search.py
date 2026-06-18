"""Reference-library search.

ReferenceLibrary wraps the index and answers product queries:

  - text  : lexical match over topic tokens (always available). When CLIP text
            embeddings are precomputed for queries, semantic scores blend in.
  - color : rank by palette/feel similarity to a target color or hex.
  - filters: dark/light, page role (cover/interior), aspect, source, cluster.
  - similar_to(id): nearest neighbours by CLIP embedding if present, else by the
            local color/feel vector. Works fully offline (no text encoder needed).

CLI:
  py -3 reflib/search.py --text "스마트팜 지원사업" --k 8
  py -3 reflib/search.py --color "#15402b" --dark --cover
  py -3 reflib/search.py --similar newdata_smartfarmkorea__056_2_..._p01
"""
import os
import sys
import re
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import features

_WORD = re.compile(r"[\w가-힣]+", re.U)


def _hex_to_rgb(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _tokenize(text):
    return [t.lower() for t in _WORD.findall(text or "") if len(t) > 1]


class ReferenceLibrary:
    def __init__(self, index_path=common.INDEX_PATH):
        if not os.path.exists(index_path):
            raise SystemExit(f"no index at {index_path} — run reflib/build_index.py first")
        self.index = json.load(open(index_path, encoding="utf-8"))
        self.records = self.index["records"]
        self.by_id = {r["id"]: r for r in self.records}
        self._color = np.array([features.color_vector(r) for r in self.records], np.float32)
        self._cstd = self._color.std(0) + 1e-6
        self._emb = self._emb_ids = None
        if os.path.exists(common.EMB_PATH) and os.path.exists(common.EMB_IDS_PATH):
            self._emb = np.load(common.EMB_PATH).astype(np.float32)
            self._emb /= (np.linalg.norm(self._emb, axis=1, keepdims=True) + 1e-9)
            self._emb_ids = {i: r for r, i in enumerate(json.load(open(common.EMB_IDS_PATH, encoding="utf-8")))}

    # ---- scoring components (each returns array aligned to self.records) ----
    def _lexical(self, text):
        q = set(_tokenize(text))
        if not q:
            return np.zeros(len(self.records), np.float32)
        out = np.zeros(len(self.records), np.float32)
        for i, r in enumerate(self.records):
            toks = set(r.get("topic_tokens", []))
            if toks:
                out[i] = len(q & toks) / len(q)
        return out

    def _color_sim(self, rgb):
        target = features.color_vector({
            "palette": [list(rgb)] * 3, "palette_w": [1, 0, 0],
            "brightness": (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0,
            "saturation": 0.5, "colorfulness": 40.0, "edge_density": 0.1})
        d = np.linalg.norm((self._color - target) / self._cstd, axis=1)
        return 1.0 / (1.0 + d)

    def _emb_sim_to(self, idx):
        if self._emb is None:
            v = (self._color - self._color[idx]) / self._cstd
            d = np.linalg.norm(v, axis=1)
            return 1.0 / (1.0 + d)
        rid = self.records[idx]["id"]
        if rid not in self._emb_ids:
            return np.zeros(len(self.records), np.float32)
        q = self._emb[self._emb_ids[rid]]
        sims = np.zeros(len(self.records), np.float32)
        for i, r in enumerate(self.records):
            j = self._emb_ids.get(r["id"])
            if j is not None:
                sims[i] = float(q @ self._emb[j])
        return sims

    # ---- public API ----
    def query(self, text=None, color=None, dark=None, cover=None, source=None,
              cluster=None, aspect=None, k=12):
        n = len(self.records)
        mask = np.ones(n, bool)
        for i, r in enumerate(self.records):
            if dark is not None and r["dark"] != dark:
                mask[i] = False
            if cover is not None and (r.get("page") == 1) != cover:
                mask[i] = False
            if source is not None and r["source"] != source:
                mask[i] = False
            if cluster is not None and r.get("cluster") != cluster:
                mask[i] = False
            if aspect is not None and abs(r["aspect"] - aspect) > 0.12:
                mask[i] = False

        score = np.zeros(n, np.float32)
        if text:
            score += 1.0 * self._lexical(text)
        if color is not None:
            rgb = _hex_to_rgb(color) if isinstance(color, str) else tuple(color)
            score += 0.8 * self._color_sim(rgb)
        if not text and color is None:
            score += 0.0  # pure filter/browse: keep index order

        score[~mask] = -1.0
        order = np.argsort(-score)
        out = []
        for i in order[:k]:
            if score[i] < 0:
                break
            out.append({**self.records[i], "score": round(float(score[i]), 4)})
        return out

    def similar_to(self, ref_id, k=8):
        if ref_id not in self.by_id:
            raise KeyError(ref_id)
        idx = self.records.index(self.by_id[ref_id])
        sims = self._emb_sim_to(idx)
        order = np.argsort(-sims)
        out = []
        for i in order:
            if self.records[i]["id"] == ref_id:
                continue
            out.append({**self.records[i], "score": round(float(sims[i]), 4)})
            if len(out) >= k:
                break
        return out


def _print(results):
    for r in results:
        tag = "dark" if r["dark"] else "light"
        cl = f" c{r['cluster']}" if "cluster" in r else ""
        print(f"  {r.get('score', ''):<7} [{tag}{cl}] {r['id']}  ({r['path']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--color")
    ap.add_argument("--similar")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--cover", action="store_true")
    ap.add_argument("--source")
    ap.add_argument("--cluster", type=int)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    lib = ReferenceLibrary()
    print(f"library: {len(lib.records)} refs  (clip={'yes' if lib._emb is not None else 'no'})\n")
    if args.similar:
        _print(lib.similar_to(args.similar, args.k))
    else:
        dark = True if args.dark else (False if args.light else None)
        _print(lib.query(text=args.text, color=args.color, dark=dark,
                         cover=True if args.cover else None, source=args.source,
                         cluster=args.cluster, k=args.k))


if __name__ == "__main__":
    main()
