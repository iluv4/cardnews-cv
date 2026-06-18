"""Shared paths / image listing / filename parsing for the reference library.

The reference corpus is the real Korean card-news images under `images_new/`
(+ `images/`), named like:

    newdata_smartfarmkorea__056_2_employment_insurance_for_self_employed__p01.jpg
    <source>            __ <deck-id + topic>                          __ <page>

We parse source / deck / page / topic-tokens straight from the filename so the
index has useful metadata even before any ML runs.
"""
import os
import re
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTS = (".png", ".jpg", ".jpeg")

# Default corpus dirs (relative to ROOT), in priority order. The dataset/ dirs
# are the labeled seed set: their stems (img_NNN) match the template-library ids
# (service/library/templates.json), so indexing them makes template-backed
# references searchable + selectable in the service until RunPod extends
# templates to the full descriptive-named corpus.
CORPUS_DIRS = ("images_new", "images", "dataset/images/train", "dataset/images/val")

INDEX_DIR = os.path.join(ROOT, "reflib", "data")
INDEX_PATH = os.path.join(INDEX_DIR, "index.json")
EMB_PATH = os.path.join(INDEX_DIR, "clip.npy")        # float32 [N, D], rows align to index order
EMB_IDS_PATH = os.path.join(INDEX_DIR, "clip_ids.json")  # list[str] ids, same order as clip.npy

_STOP = {"for", "of", "the", "and", "in", "on", "to", "a", "card", "news", "2",
         "2022", "2023", "2024", "2025", "p", "vol"}
_PAGE_RE = re.compile(r"^p(\d+)$", re.I)


def list_images(dirs=CORPUS_DIRS):
    """Absolute paths of every corpus image, sorted, de-duplicated by filename."""
    seen, out = set(), []
    for d in dirs:
        ad = d if os.path.isabs(d) else os.path.join(ROOT, d)
        if not os.path.isdir(ad):
            continue
        for ext in EXTS:
            for p in glob.glob(os.path.join(ad, "*" + ext)):
                name = os.path.basename(p)
                if name not in seen:
                    seen.add(name)
                    out.append(p)
    return sorted(out)


def parse_name(path):
    """Filename -> {id, source, deck, page, topic_tokens}.

    Robust to names without the `__` convention (deck = stem, page = None).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("__")
    source = parts[0] if len(parts) > 1 else "unknown"
    page = None
    if len(parts) >= 3:
        deck = parts[1]
        m = _PAGE_RE.match(parts[-1])
        page = int(m.group(1)) if m else None
    else:
        deck = stem

    # topic tokens: words from the deck id, minus pure numbers and stopwords
    toks = []
    for t in re.split(r"[_\W]+", deck.lower()):
        if t and not t.isdigit() and t not in _STOP and len(t) > 1:
            toks.append(t)
    return {"id": stem, "source": source, "deck": deck, "page": page,
            "topic_tokens": toks}
