"""Collect curated Korean card-news images from the carnews-insta repo into the
detector dataset folder. Skips blog-screenshot / non-card folders.
"""
import os
import shutil
import glob

SRC = r"C:\Users\Administrator\Downloads\carnews-insta"
DST = r"C:\Users\Administrator\Downloads\cardnews-detector\images"
os.makedirs(DST, exist_ok=True)

# Folders that actually contain card-news (relative to SRC). Excludes 부암동
# (naver blog screenshots), 영어 표현_캡처 (screen captures), woorI_photo (raw photos).
INCLUDE = [
    r"public\saved-refs",
    r"references\docs\images\영어 카드뉴스_GPT",
    r"references\docs\images\red_tone_card_news",
    r"references\docs\images\seoul_stratup",
    r"references\docs\images\insurance_cardnews",
    r"references\docs\images\단국대 소개팅 부스_카드뉴스",
    r"references\docs\images\2026-ux-trend",
    r"references\docs\images\매거진 카드뉴스",
    r"references\docs\images\marketing_crew",
    r"public\templates",
]

exts = (".png", ".jpg", ".jpeg")
n = 0
for rel in INCLUDE:
    folder = os.path.join(SRC, rel)
    if not os.path.isdir(folder):
        print(f"  (missing) {rel}")
        continue
    tag = rel.replace("\\", "_").replace(" ", "")
    for p in glob.glob(os.path.join(folder, "*")):
        if not p.lower().endswith(exts):
            continue
        base = f"{tag}__{os.path.basename(p)}"
        # normalize extension to .jpg/.png as-is; just copy
        shutil.copy2(p, os.path.join(DST, base))
        n += 1
print(f"Collected {n} images into {DST}")
