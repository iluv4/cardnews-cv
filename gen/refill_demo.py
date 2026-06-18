"""Proof of the product loop: pick a reference template -> COPY its layout ->
refill with new user text (v2 quality). Saves original | refilled montages.

  py -3 gen/refill_demo.py            -> gen_output/refill/

Uses only the committed labels (gen/extract_templates.py output), so it runs
locally with no torch. On RunPod the same templates come from the live detector
over the full 687-image corpus — identical render path.
"""
import os
import sys
import json

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "service"))

from common import ROOT
from cardgen.from_template import render_from_template

LIB = os.path.join(ROOT, "service", "library", "templates.json")
OUT = os.path.join(ROOT, "gen_output", "refill")

# new content to pour into whatever layout we pick, by archetype
CONTENT = {
    "cover": dict(title="청년 농부\n도전 프로젝트", subtitle="지금 바로 시작하세요!"),
    "statement": dict(title="왜 스마트팜\n인가요?",
                      subtitle="노동력은 줄이고 *생산성은 두 배로*"),
    "list": dict(title="신청 전\n체크리스트",
                 checklist=["만 18~39세 청년 농업인", "영농경력 3년 이하 신규 농업인",
                            "농업경영체 등록을 마친 경우"]),
    "body": dict(title="한 줄 요약", subtitle="복잡한 정책을 *한 장으로* 정리"),
    "mixed": dict(title="핵심 정리", subtitle="꼭 알아야 할 *3가지 포인트*"),
}


def panel(im, h=760, bg=(248, 248, 248)):
    w = int(im.width * h / im.height)
    return im.convert("RGB").resize((w, h))


def montage(a, b, gap=14, bg=(248, 248, 248)):
    pa, pb = panel(a), panel(b)
    h = max(pa.height, pb.height)
    W = pa.width + pb.width + gap * 3
    canvas = Image.new("RGB", (W, h + 2 * gap), bg)
    canvas.paste(pa, (gap, gap))
    canvas.paste(pb, (gap * 2 + pa.width, gap))
    return canvas


def main():
    os.makedirs(OUT, exist_ok=True)
    templates = json.load(open(LIB, encoding="utf-8"))

    # pick a few representative templates: prefer dark, one per archetype
    picks = {}
    for t in sorted(templates, key=lambda x: (not x["is_dark"], x["id"])):
        a = t["archetype"]
        if a not in picks and a in CONTENT:
            picks[a] = t
    print(f"library: {len(templates)} templates; demoing {len(picks)} archetypes")

    for a, t in picks.items():
        content = CONTENT[a]
        refilled = render_from_template(t, brand="AGRI", **content)
        orig = Image.open(os.path.join(ROOT, t["source"])).convert("RGB")
        out = os.path.join(OUT, f"refill_{a}_{t['id']}.png")
        montage(orig, refilled).save(out)
        print(f"  {a:10s} <- {t['id']:8s} ({t['n_title']}T/{t['n_body']}B)  -> {out}")


if __name__ == "__main__":
    main()
