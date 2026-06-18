"""Export the collected feedback as a dataset (CSV).

Pulls every rating row from Supabase (or reads the local JSONL) and writes a flat
CSV you can load in pandas for analysis / re-ranking / eval.

  SUPABASE_URL=... SUPABASE_ANON_KEY=... py -3 scripts/export_feedback.py
  py -3 scripts/export_feedback.py            # local JSONL fallback

Output: data_export/feedback.csv
"""
import os
import csv
import json
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SB_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SB_KEY = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY") or ""
JSONL = os.path.join(ROOT, "reflib", "data", "feedback.jsonl")
COLS = ["created_at", "ref_id", "rating", "query", "mode", "title", "note", "session"]


def from_supabase():
    url = f"{SB_URL}/rest/v1/feedback?select=*&order=created_at.asc&limit=100000"
    req = urllib.request.Request(url)
    req.add_header("apikey", SB_KEY)
    req.add_header("Authorization", f"Bearer {SB_KEY}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def from_jsonl():
    if not os.path.exists(JSONL):
        return []
    return [json.loads(ln) for ln in open(JSONL, encoding="utf-8") if ln.strip()]


def main():
    rows = from_supabase() if (SB_URL and SB_KEY) else from_jsonl()
    out_dir = os.path.join(ROOT, "data_export")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "feedback.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    src = "supabase" if (SB_URL and SB_KEY) else "jsonl"
    print(f"exported {len(rows)} rows from {src} -> {out}")
    # quick net-per-ref summary
    net = {}
    for r in rows:
        net[r["ref_id"]] = net.get(r["ref_id"], 0) + int(r.get("rating", 0))
    pos = sum(1 for v in net.values() if v > 0)
    print(f"  {len(net)} rated refs, {pos} net-positive")


if __name__ == "__main__":
    main()
