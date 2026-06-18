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


@app.get("/api/search")
def api_search(text: str = None, color: str = None, dark: bool = None,
               cover: bool = None, cluster: int = None, source: str = None,
               k: int = 24):
    res = lib().query(text=text, color=color, dark=dark, cover=cover,
                      cluster=cluster, source=source, k=k)
    return {"count": len(res), "results": [_slim(r) for r in res]}


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


@app.get("/api/reference/{ref_id}")
def api_reference(ref_id: str):
    rec = lib().by_id.get(ref_id)
    if not rec:
        raise HTTPException(404, f"unknown reference {ref_id}")
    p = os.path.join(ROOT, rec["path"])
    if not os.path.exists(p):
        raise HTTPException(404, "image file missing")
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


@app.get("/", response_class=HTMLResponse)
def index():
    p = os.path.join(ROOT, "service", "static", "index.html")
    return HTMLResponse(open(p, encoding="utf-8").read())
