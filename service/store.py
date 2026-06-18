"""Feedback storage — Supabase (deployed) or JSONL (local), chosen by env.

Set SUPABASE_URL + SUPABASE_ANON_KEY (or SUPABASE_KEY) to use Supabase Postgres
via PostgREST; otherwise feedback lands in reflib/data/feedback.jsonl. The rest
of the app calls only save_feedback() / load_summary(), so swapping the backend
(SQLite, another DB) is a change here alone.

Uses urllib (stdlib) so the slim deploy image needs no extra deps.
"""
import os
import json
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL_PATH = os.path.join(ROOT, "reflib", "data", "feedback.jsonl")

SB_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SB_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY") or ""
USE_SB = bool(SB_URL and SB_KEY)

_FIELDS = ("ref_id", "rating", "query", "mode", "title", "note", "session")


def backend():
    return "supabase" if USE_SB else "jsonl"


def _sb(method, path, body=None, params=""):
    url = f"{SB_URL}/rest/v1/{path}{params}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("apikey", SB_KEY)
    req.add_header("Authorization", f"Bearer {SB_KEY}")
    req.add_header("Content-Type", "application/json")
    if method == "POST":
        req.add_header("Prefer", "return=minimal")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def save_feedback(rec):
    """rec: dict with ref_id, rating(+1/-1), query?, mode?, title?, note?, session?"""
    row = {k: rec.get(k) for k in _FIELDS}
    if USE_SB:
        _sb("POST", "feedback", body=row)
    else:
        os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_summary():
    """-> {ref_id: net_rating}. Aggregated client-side (team scale)."""
    out = {}
    if USE_SB:
        try:
            raw = _sb("GET", "feedback", params="?select=ref_id,rating&limit=10000")
            for d in json.loads(raw):
                rid = d.get("ref_id")
                if rid:
                    out[rid] = out.get(rid, 0) + int(d.get("rating", 0))
        except Exception:
            return {}
    elif os.path.exists(JSONL_PATH):
        for ln in open(JSONL_PATH, encoding="utf-8"):
            try:
                d = json.loads(ln)
            except Exception:
                continue
            rid = d.get("ref_id")
            if rid:
                out[rid] = out.get(rid, 0) + int(d.get("rating", 0))
    return out
