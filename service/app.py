"""FastAPI service: search the reference library -> select -> auto-fill -> render.

Ties the two halves together:
  - reflib.ReferenceLibrary  (search/select over the real corpus)
  - cardgen.generate_card/deck  (deterministic engine v2 rendering)

Run:
  py -3 -m uvicorn service.app:app --reload --port 8000
  open http://127.0.0.1:8000/

Endpoints:
  GET  /api/search    text, color, dark, cover, cluster, source, k  -> [refs]
  GET  /api/similar/{id}?k=                                          -> [refs]
  GET  /api/clusters                                                 -> summary
  GET  /api/reference/{id}            -> the reference image (for thumbnails)
  POST /api/generate  {title, subtitle, checklist[], theme, brand, ...} -> PNG
"""
import os
import sys
import io
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "service"))

from reflib.search import ReferenceLibrary
from cardgen import generate_card, generate_deck
from cardgen.from_template import render_from_template

TEMPLATES_PATH = os.path.join(ROOT, "service", "library", "templates.json")

app = FastAPI(title="cardnews-cv", version="2")
_LIB = None
_TEMPLATES = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = ReferenceLibrary()
    return _LIB


def templates():
    """id -> layout template (from gen/extract_templates.py). {} if none built."""
    global _TEMPLATES
    if _TEMPLATES is None:
        if os.path.exists(TEMPLATES_PATH):
            _TEMPLATES = {t["id"]: t for t in json.load(open(TEMPLATES_PATH, encoding="utf-8"))}
        else:
            _TEMPLATES = {}
    return _TEMPLATES


import store

_FB = None          # cached {ref_id: net}; invalidated on each new rating


def feedback_summary():
    """id -> net rating (sum of +1/-1). Cached until the next write."""
    global _FB
    if _FB is None:
        _FB = store.load_summary()
    return _FB


# ---- map a chosen reference to an engine theme (until live extraction lands) --
def suggest_theme(rec):
    r, g, b = rec["palette"][0]
    warm = r > max(g, b) + 20
    greenish = g >= max(r, b)
    bluish = b >= max(r, g) + 10
    if rec["dark"]:
        if greenish:
            return "forest"
        if warm:
            return "coral"
        return "midnight" if b < 150 else "sky"
    return "mint" if greenish else "mono"


def _slim(rec):
    """Trim a record for the wire (drop bulky palette weights etc.)."""
    return {k: rec[k] for k in ("id", "source", "deck", "page", "path", "aspect",
                                "dark", "cluster", "score", "palette")
            if k in rec} | {"theme": suggest_theme(rec),
                            "has_template": rec["id"] in templates()}


def _apply_feedback_boost(res):
    """Nudge results that users rated positively above similar-scoring ones."""
    import math
    fb = feedback_summary()
    if not fb:
        return res
    for r in res:
        net = fb.get(r["id"], 0)
        if net:
            r["score"] = round(r.get("score", 0.0) + 0.15 * math.tanh(net / 3.0), 4)
    res.sort(key=lambda r: -r.get("score", 0.0))
    return res


@app.get("/api/search")
def api_search(text: str = None, color: str = None, dark: bool = None,
               cover: bool = None, cluster: int = None, source: str = None,
               only_templates: bool = False, k: int = 24):
    # over-fetch a pool, filter/boost, then truncate (so feedback can re-rank)
    pool = lib().query(text=text, color=color, dark=dark, cover=cover,
                       cluster=cluster, source=source, k=k * 6)
    if only_templates:
        tids = templates()
        pool = [r for r in pool if r["id"] in tids]
    pool = _apply_feedback_boost(pool)[:k]
    return {"count": len(pool), "results": [_slim(r) for r in pool]}


class FbReq(BaseModel):
    ref_id: str
    rating: int                       # +1 (good) or -1 (bad)
    query: str | None = None
    mode: str | None = None           # template | theme-fallback
    title: str | None = None
    note: str | None = None


@app.post("/api/feedback")
def api_feedback(req: FbReq):
    """Record a user's evaluation of a generated result. Feeds search re-ranking
    and doubles as a labeled eval dataset (Supabase, or JSONL locally)."""
    global _FB
    rating = 1 if req.rating > 0 else -1
    store.save_feedback({"ref_id": req.ref_id, "rating": rating, "query": req.query,
                         "mode": req.mode, "title": req.title, "note": req.note})
    _FB = None                          # invalidate; reload includes the new row
    fb = feedback_summary()
    return {"ok": True, "ref_id": req.ref_id, "net": fb.get(req.ref_id, 0),
            "store": store.backend()}


@app.get("/api/feedback/summary")
def api_feedback_summary():
    fb = feedback_summary()
    pos = sum(1 for v in fb.values() if v > 0)
    return {"net_by_ref": fb, "rated_refs": len(fb), "net_positive_refs": pos,
            "store": store.backend()}


@app.get("/api/similar/{ref_id}")
def api_similar(ref_id: str, k: int = 12):
    try:
        res = lib().similar_to(ref_id, k)
    except KeyError:
        raise HTTPException(404, f"unknown reference {ref_id}")
    return {"count": len(res), "results": [_slim(r) for r in res]}


@app.get("/api/clusters")
def api_clusters():
    return lib().index.get("clusters", {})


THUMB_DIR = os.path.join(ROOT, "reflib", "data", "thumbs")


@app.get("/api/reference/{ref_id}")
def api_reference(ref_id: str, w: int = 0):
    """Serve a reference image. With ?w=N, serve a cached N-px JPEG thumbnail
    (grid uses this — much faster than streaming full-size jpgs)."""
    rec = lib().by_id.get(ref_id)
    if not rec:
        raise HTTPException(404, f"unknown reference {ref_id}")
    p = os.path.join(ROOT, rec["path"])
    if not os.path.exists(p):
        raise HTTPException(404, "image file missing")
    if w and w > 0:
        from PIL import Image
        os.makedirs(THUMB_DIR, exist_ok=True)
        tp = os.path.join(THUMB_DIR, f"{ref_id}_{w}.jpg")
        if not os.path.exists(tp):
            im = Image.open(p).convert("RGB")
            im.thumbnail((w, w * 2))
            im.save(tp, "JPEG", quality=82)
        return FileResponse(tp)
    return FileResponse(p)


class GenReq(BaseModel):
    title: str
    subtitle: str | None = None
    body: str | None = None
    checklist: list[str] | None = None
    eyebrow: str | None = None
    theme: str = "forest"
    brand: str | None = None
    mascot: str | None = None
    size: int = 1080
    seed: int = 0


@app.post("/api/generate")
def api_generate(req: GenReq):
    kw = dict(subtitle=req.subtitle, body=req.body, checklist=req.checklist,
              eyebrow=req.eyebrow, theme=req.theme, mascot=req.mascot,
              size=(req.size, req.size), seed=req.seed)
    if req.brand:           # else fall through to the engine default
        kw["brand"] = req.brand
    img = generate_card(req.title, **kw)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


class FromRefReq(BaseModel):
    ref_id: str
    title: str
    subtitle: str | None = None
    checklist: list[str] | None = None
    brand: str | None = None
    size: int = 1080
    seed: int | None = None


@app.post("/api/generate_from_ref")
def api_generate_from_ref(req: FromRefReq):
    """The real product step: copy the SELECTED reference's actual layout and
    refill it with the user's text. Falls back to the themed v2 layout when the
    reference has no extracted template yet (e.g. before the RunPod detector pass
    grows templates to the full corpus)."""
    rec = lib().by_id.get(req.ref_id)
    if not rec:
        raise HTTPException(404, f"unknown reference {req.ref_id}")
    t = templates().get(req.ref_id)
    if t is not None:
        img = render_from_template(t, req.title, subtitle=req.subtitle,
                                   checklist=req.checklist, brand=req.brand,
                                   width=req.size, seed=req.seed)
        mode = "template"
    else:
        kw = dict(subtitle=req.subtitle, checklist=req.checklist,
                  theme=suggest_theme(rec), size=(req.size, req.size),
                  seed=req.seed or 0)
        if req.brand:
            kw["brand"] = req.brand
        img = generate_card(req.title, **kw)
        mode = "theme-fallback"
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"X-Render-Mode": mode})


class DeckReq(BaseModel):
    items: list[dict]                 # [{title, subtitle?/body?, checklist?, eyebrow?}, ...]
    theme: str = "forest"
    brand: str | None = None
    mascot: str | None = None
    size: int = 1080
    seed: int = 0


@app.post("/api/deck")
def api_deck(req: DeckReq):
    """Render a whole deck (cover + interiors) -> zip of PNGs."""
    import zipfile
    if not req.items:
        raise HTTPException(400, "items is empty")
    kw = dict(theme=req.theme, mascot=req.mascot, size=(req.size, req.size), seed=req.seed)
    if req.brand:
        kw["brand"] = req.brand
    cards = generate_deck(req.items, **kw)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, card in enumerate(cards):
            b = io.BytesIO(); card.save(b, "PNG")
            z.writestr(f"card_{i:02d}.png", b.getvalue())
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="deck.zip"'})


# ---- deck-centric workflow: render / export a whole deck honoring a reference --
class DeckCard(BaseModel):
    title: str
    subtitle: str | None = None
    checklist: list[str] | None = None


class DeckBuildReq(BaseModel):
    cards: list[DeckCard]
    ref_id: str | None = None         # selected reference; None -> default theme
    brand: str | None = None
    size: int = 1080
    seed: int = 0


def _render_deck(ref_id, cards, brand, size, seed):
    """Render every card of a deck as one designed series.

    Template-backed refs copy the selected reference's REAL layout on every card
    (the product's "copy that layout, refill" promise); a ref without a template
    drives the themed v2 engine with a palette synthesized from it (cover +
    page-numbered interiors); no ref at all falls back to the default theme.
    Returns (list[PIL.Image], mode)."""
    rec = lib().by_id.get(ref_id) if ref_id else None
    if ref_id and not rec:
        raise HTTPException(404, f"unknown reference {ref_id}")
    t = templates().get(ref_id) if ref_id else None
    theme = suggest_theme(rec) if rec else "forest"
    n = len(cards)
    imgs = []
    for i, c in enumerate(cards):
        title = (c.get("title") or "").replace("\\n", "\n")
        subtitle = c.get("subtitle") or None
        checklist = [s for s in (c.get("checklist") or []) if s] or None
        if t is not None:
            img = render_from_template(t, title, subtitle=subtitle,
                                       checklist=checklist, brand=brand or None,
                                       width=size, seed=seed + i)
        else:
            img = generate_card(title, subtitle=subtitle, checklist=checklist,
                                theme=theme, kind=("cover" if i == 0 else "auto"),
                                brand=(brand or None),
                                page=(None if i == 0 else i + 1),
                                total=(None if i == 0 else n),
                                size=(size, size), seed=seed)
        imgs.append(img)
    mode = "template" if t is not None else ("theme" if rec else "theme-default")
    return imgs, mode


@app.post("/api/deck/render")
def api_deck_render(req: DeckBuildReq):
    """Render the whole deck and return each card as a base64 PNG data URL — the
    review step's filmstrip pulls the entire deck in one request."""
    import base64
    if not req.cards:
        raise HTTPException(400, "cards is empty")
    cards = [c.model_dump() for c in req.cards]
    imgs, mode = _render_deck(req.ref_id, cards, req.brand, req.size, req.seed)
    out = []
    for im in imgs:
        b = io.BytesIO(); im.save(b, "PNG")
        out.append("data:image/png;base64," + base64.b64encode(b.getvalue()).decode())
    return {"mode": mode, "count": len(out), "cards": out}


@app.post("/api/deck/export")
def api_deck_export(req: DeckBuildReq):
    """Render the deck and return a zip of PNGs (one per card)."""
    import zipfile
    if not req.cards:
        raise HTTPException(400, "cards is empty")
    cards = [c.model_dump() for c in req.cards]
    imgs, mode = _render_deck(req.ref_id, cards, req.brand, req.size, req.seed)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, im in enumerate(imgs):
            b = io.BytesIO(); im.save(b, "PNG")
            z.writestr(f"card_{i:02d}.png", b.getvalue())
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="deck.zip"',
                             "X-Render-Mode": mode})


@app.get("/", response_class=HTMLResponse)
def index():
    p = os.path.join(ROOT, "service", "static", "index.html")
    return HTMLResponse(open(p, encoding="utf-8").read())
