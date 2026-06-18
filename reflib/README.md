# reflib — reference library + retrieval

The product/ML core: index a large library of **real** Korean card-news, then
search/select a reference to copy its layout. Two tiers so the useful part runs
now without a GPU, and the heavy ML plugs in on RunPod.

```
reflib/
  common.py        paths, corpus listing, filename -> {source, deck, page, topic}
  features.py      torch-free visual features (palette, brightness, edges, dark…)
  build_index.py   LOCAL  -> reflib/data/index.json   (687 imgs, ~8s)
  cluster.py       LOCAL  -> archetype id per record (CLIP if present, else color)
  search.py        LOCAL  -> ReferenceLibrary: text / color / filters / similar_to
  embed_clip.py    RUNPOD -> clip.npy  (semantic similar_to + semantic text search)
  tag_layout.py    RUNPOD -> per-record layout signature from the detector
```

## Local tier (no torch — runs today)

```bash
py -3 reflib/build_index.py            # all corpus images -> data/index.json
py -3 reflib/cluster.py --k 8          # archetype clusters (color basis until CLIP)

py -3 reflib/search.py --text "smart farm young farmer support" --k 8
py -3 reflib/search.py --color "#15402b" --dark --cover
py -3 reflib/search.py --similar <ref_id> --k 8
```

What works offline: lexical text search (filename topic tokens), color/feel
similarity, filters (dark/light, cover=page 1, source, cluster, aspect), and
`similar_to` (color-vector NN until CLIP embeddings exist).

## RunPod tier (torch — semantic + structural)

```bash
git clone https://github.com/iluv4/cardnews-cv.git && cd cardnews-cv
py -3 reflib/build_index.py            # (or copy the committed index.json)

pip install open_clip_torch torch torchvision pillow numpy
python reflib/embed_clip.py --build    # -> reflib/data/clip.npy (semantic)
python reflib/embed_clip.py --text "스마트팜 청년 지원 카드뉴스" --k 8

pip install ultralytics
python reflib/tag_layout.py --imgsz 1280   # -> per-record layout signature
```

After the CLIP build, `search.py --similar` and `cluster.py` automatically switch
to the semantic embedding (no code change). For semantic **text** search at query
time, the API host needs torch to encode the query (`embed_clip.encode_text`);
otherwise search falls back to lexical.

## Index record shape

```json
{
  "id": "newdata_smartfarmkorea__064_smart_farm_..._p09",
  "source": "newdata_smartfarmkorea", "deck": "064_smart_farm_...", "page": 9,
  "topic_tokens": ["smart","farm","promotion","big","mama"],
  "path": "images_new/....jpg",
  "w": 900, "h": 900, "aspect": 1.0,
  "palette": [[r,g,b]…], "palette_w": [...], "brightness": .59,
  "saturation": .03, "colorfulness": 9.8, "edge_density": .09, "dark": false,
  "cluster": 3,
  "layout": {"counts": {...}, "title_pos": [.5,.18], "text_cov": .22,
             "signature": "title-top|body-mid|logo-tr"}   // after tag_layout.py
}
```

`index.json` is committed so search works on a fresh clone; `clip.npy` is
gitignored and regenerated on RunPod.
