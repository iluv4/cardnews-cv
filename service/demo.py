"""Generate sample cards to eyeball engine quality.  py -3 service/demo.py"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cardgen import generate_card, generate_deck

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "service", "out")
os.makedirs(OUT, exist_ok=True)


def main():
    # 1) design-background single cards across themes/templates
    samples = [
        dict(title="스마트팜으로\n바뀌는 농업", body="복잡한 내용을 한 장에 정리했습니다. 핵심만 빠르게 확인하세요.",
             eyebrow="SMART FARM", theme="forest", template="editorial"),
        dict(title="2024 청년\n농업 지원 사업", body="신청 자격과 혜택을 한눈에. 마감 전에 꼭 확인하세요.",
             eyebrow="NOTICE", theme="midnight", template="centered"),
        dict(title="오늘의 핵심 요약", body="3가지만 기억하세요.", eyebrow="TODAY",
             theme="coral", template="centered"),
        dict(title="에너지 절감\n실천 가이드", body="시설 원예 농가를 위한 실전 팁 모음입니다.",
             eyebrow="GUIDE", theme="mint", template="editorial"),
    ]
    for i, s in enumerate(samples):
        generate_card(**s, logo_text="AGRI").save(os.path.join(OUT, f"card_{i:02d}_{s['theme']}.png"))

    # 2) a multi-page deck (cover + content), consistent theme
    deck = [
        dict(title="스마트팜\n완전정복", eyebrow="2024 가이드"),
        dict(title="스마트팜이란?", body="ICT를 접목해 작물 생육 환경을 원격·자동으로 관리하는 농장입니다."),
        dict(title="3가지 핵심 이점", body="1. 생산성 향상  2. 노동력 절감  3. 품질 균일화"),
        dict(title="지금 시작하세요", body="정부 지원사업으로 초기 비용 부담을 낮출 수 있습니다."),
    ]
    for i, card in enumerate(generate_deck(deck, theme="forest", logo_text="AGRI")):
        card.save(os.path.join(OUT, f"deck_{i:02d}.png"))

    # 3) photo background (saliency-aware placement + scrim), if a photo exists
    photos = glob.glob(os.path.join(ROOT, "images_new", "*.jpg"))
    if photos:
        generate_card("여름철 폭우\n대비 요령", "침수 위험 지역은 미리 점검하세요.",
                      background=photos[20 % len(photos)], theme="midnight",
                      template="bottom", eyebrow="SAFETY", logo_text="CITY"
                      ).save(os.path.join(OUT, "photo_00.png"))

    print("wrote samples ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print(" ", f)


if __name__ == "__main__":
    main()
