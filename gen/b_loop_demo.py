"""B-loop prototype — the hybrid product gate (BUSINESS_CASE.md, direction B).

Wires our proprietary assets as the CONTROL + QA layer around GPT Image 2:

  (1) reflib search        find a good real reference for the query     [no torch]
  (2) layout lookup        read that reference's extracted layout        [no torch]
  (3) layout -> prompt     turn block geometry into a precise GPT brief  [no torch]
  (4) GPT Image 2 render    the pixels (Korean text OK now)              [OPENAI key]
  (5) detector QA          score the output's layout vs intent          [torch/GPU]

Steps 1-3 run anywhere. Step 4 needs OPENAI_API_KEY + network. Step 5 needs the
trained detector (ultralytics/torch) -> that's the RunPod half. See RUNPOD_BLOOP.md.

Run (full loop on a RunPod PyTorch pod):
    export OPENAI_API_KEY=sk-...
    py -3 gen/b_loop_demo.py --text "스마트팜 지원사업 안내" \
        --title "스마트팜으로\n바뀌는 농업" \
        --body "복잡한 지원 절차를 한 장에 정리했습니다." --out gen/out_bloop

Dry run (skip GPT + QA, just show the chosen reference + prompt):
    py -3 gen/b_loop_demo.py --text "스마트팜 지원사업" --dry
"""
import os
import sys
import io
import json
import base64
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                                          # repo root only

from reflib.search import ReferenceLibrary   # noqa: E402  (reflib uses its own common)

TEMPLATES = os.path.join(ROOT, "service", "library", "templates.json")


def load_templates():
    """templates.json (list) -> {id: record}. Holds the pre-extracted layouts."""
    data = json.load(open(TEMPLATES, encoding="utf-8"))
    return {r["id"]: r for r in data}


def pick_reference(query, templates, k=12):
    """(1)+(2): search the library, return the top hit that also has a layout."""
    lib = ReferenceLibrary()
    hits = lib.query(text=query, k=k)
    for h in hits:
        if h["id"] in templates:
            return h, templates[h["id"]]
    # No searched hit has an extracted layout yet -> that ref needs the detector
    # extraction path (gen/extract_templates_detector.py) before it's usable here.
    raise SystemExit(
        f"top-{k} search hits have no template in templates.json "
        f"(run extract_templates_detector.py to cover the full corpus first)")


def _zone(box):
    """Normalized xyxy -> coarse human zone label for the prompt."""
    x0, y0, x1, y1 = box
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    v = "top" if cy < 0.38 else "middle" if cy < 0.66 else "bottom"
    h = "left" if cx < 0.38 else "center" if cx < 0.62 else "right"
    return f"{v}-{h}"


def build_prompt(layout, title, body):
    """(3): translate block geometry + content into a precise GPT Image 2 brief.

    The reference contributes STRUCTURE (where things sit, alignment, mood/bg),
    the user contributes the Korean COPY. GPT renders the pixels."""
    blocks = layout.get("blocks", {})
    title_zones = [_zone(b) for b in blocks.get("title", [])] or ["top-center"]
    body_zones = [_zone(b) for b in blocks.get("body", [])] or ["middle-center"]
    bg = layout.get("bg", [20, 20, 20])
    dark = (sum(bg) / 3) < 110
    align = layout.get("title_align", "center")
    arche = layout.get("archetype", "statement")
    ar = layout.get("aspect", "portrait")

    title_txt = title.replace("\\n", " / ")
    body_txt = body.replace("\\n", " ")
    return (
        f"A Korean Instagram card-news slide, {ar}, {arche} style. "
        f"{'Dark' if dark else 'Light'} background (approx RGB {tuple(bg)}), clean editorial design, "
        f"NOT cluttered, professional designer quality, no watermark.\n"
        f"TITLE text (render exactly, large, {align}-aligned, placed {title_zones[0]}): \"{title_txt}\"\n"
        f"BODY text (render exactly, smaller, placed {body_zones[0]}): \"{body_txt}\"\n"
        f"All Korean text must be crisp and correctly spelled. Keep the salient area uncluttered. "
        f"Match the layout positions described above."
    )


def render_gpt_image(prompt, size, out_png):
    """(4): GPT Image 2 -> PNG. Needs OPENAI_API_KEY."""
    from openai import OpenAI
    from PIL import Image
    client = OpenAI()
    r = client.images.generate(model="gpt-image-2", prompt=prompt, size=size)
    img = Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json)))
    img.save(out_png)
    return img


def qa_layout(out_png, layout, conf=0.25, imgsz=1280):
    """(5): run the trained detector on the GPT output and score it against intent.

    A cheap first metric: did GPT put roughly the right NUMBER of title/body
    blocks in roughly the right places? Returns a dict report. Needs torch."""
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    model = YOLO(os.path.join(ROOT, "results", "best_model", "best.pt"))
    img = Image.open(out_png).convert("RGB")
    W, H = img.size
    res = model.predict(np.array(img), conf=conf, imgsz=imgsz, verbose=False)[0]
    det = {0: 0, 1: 0}
    for b in res.boxes:
        c = int(b.cls[0])
        if c in det:
            det[c] += 1
    want = {0: layout.get("n_title", len(layout["blocks"].get("title", []))),
            1: layout.get("n_body", len(layout["blocks"].get("body", [])))}
    return {
        "detected_title": det[0], "wanted_title": want[0],
        "detected_body": det[1], "wanted_body": want[1],
        "title_ok": det[0] >= max(1, want[0]),
        "body_ok": det[1] >= max(1, want[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="search query (brand/topic)")
    ap.add_argument("--title", default="제목을\\n여기에")
    ap.add_argument("--body", default="본문을 여기에 넣습니다.")
    ap.add_argument("--out", default=os.path.join(ROOT, "gen", "out_bloop"))
    ap.add_argument("--size", default="1024x1536")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dry", action="store_true", help="skip GPT + QA")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    templates = load_templates()
    hit, layout = pick_reference(args.text, templates)
    print(f"(1)(2) reference: {hit['id']}  score={hit.get('score')}  "
          f"archetype={layout.get('archetype')}  n_title={layout.get('n_title')} "
          f"n_body={layout.get('n_body')}")

    prompt = build_prompt(layout, args.title, args.body)
    print("(3) GPT prompt:\n" + prompt + "\n")
    json.dump({"reference": hit["id"], "prompt": prompt, "layout": layout},
              open(os.path.join(args.out, "brief.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    if args.dry:
        print("--dry: stopping before GPT render + QA.")
        return

    out_png = os.path.join(args.out, "render.png")
    render_gpt_image(prompt, args.size, out_png)
    print(f"(4) rendered -> {out_png}")

    report = qa_layout(out_png, layout, conf=args.conf)
    json.dump(report, open(os.path.join(args.out, "qa.json"), "w"), indent=1)
    print("(5) QA:", report)
    print("PASS" if report["title_ok"] and report["body_ok"]
          else "FAIL — GPT output layout did not match intent (the interesting case)")


if __name__ == "__main__":
    main()
