"""Generate sample cards to eyeball engine v2 quality.  py -3 service/demo.py

Showcases the reference look: dark themed bg + scattered deco + two-tone glow
title + *keyword* subtitle + white checklist panel + brand.
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cardgen import generate_card, generate_deck

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "service", "out")
os.makedirs(OUT, exist_ok=True)

SQ = (1080, 1080)  # square, like the gov/agri reference decks


def main():
    # 1) a full EPIS-style deck (cover + checklist interiors), square, consistent
    deck = [
        dict(title="역량 맞춤형\n토마토 교육", subtitle="농업 교육, 수준별로 똑똑하게!",
             eyebrow="NEW"),
        dict(title="역량 맞춤형\n토마토 교육?",
             subtitle="농장 규모와 시설, 재배기술 등에 따라 초보 농업인부터 숙련된 전문가까지 *4수준으로 구분한* 토마토 교육",
             checklist=[
                 "반복학습이 가능한 온라인 교육으로 진행",
                 "업계 전문가들이 직접 강의하여 실무적인 지식 습득 가능",
                 "효율적인 학습을 위해 자료실에서 교육자료 제공",
             ]),
        dict(title="이렇게\n신청하세요",
             subtitle="누리집에서 *3단계*면 신청 완료",
             checklist=[
                 "농업교육포털에 회원가입 후 로그인",
                 "수준별 과정 선택 후 수강 신청",
                 "마이페이지에서 학습 현황 확인",
             ]),
        dict(title="지금 바로\n시작하세요", subtitle="*무료 교육*, 마감 전에 신청하세요!"),
    ]
    cards = generate_deck(deck, theme="forest", brand="EPIS", mascot="tomato",
                          size=SQ, seed=0)
    for i, card in enumerate(cards):
        card.save(os.path.join(OUT, f"deck_{i:02d}.png"))

    # 2) one checklist card per theme (verify the palette system holds up)
    themes = ["midnight", "coral", "sky", "mint", "mono"]
    for t in themes:
        generate_card(
            "오늘의 핵심\n3가지 요약", subtitle="놓치면 안 되는 *핵심 포인트*만 정리했어요",
            checklist=["복잡한 내용을 한 장에 정리", "실무에 바로 쓰는 체크리스트",
                       "마감과 신청 조건까지 한눈에"],
            theme=t, brand="AGRI", size=SQ).save(
            os.path.join(OUT, f"card_{t}.png"))

    # 3) photo background still supported (saliency-light placement + scrim)
    photos = glob.glob(os.path.join(ROOT, "images_new", "*.jpg"))
    if photos:
        generate_card("여름철 폭우\n대비 요령", body="침수 위험 지역은 미리 점검하세요.",
                      background=photos[20 % len(photos)], theme="midnight",
                      brand="CITY", size=SQ).save(
            os.path.join(OUT, "photo_00.png"))

    print("wrote samples ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print(" ", f)


if __name__ == "__main__":
    main()
