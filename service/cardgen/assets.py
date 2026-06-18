"""Fonts + paths for the card-news engine. Font objects are cached so the hot
render path stays fast (no disk I/O per call)."""
import os
from functools import lru_cache
from PIL import ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONT_DIR = os.path.join(REPO, "assets", "fonts")

_FALLBACKS = {
    "Bold": [r"C:\Windows\Fonts\malgunbd.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"],
    "SemiBold": [r"C:\Windows\Fonts\malgunbd.ttf"],
    "Medium": [r"C:\Windows\Fonts\malgun.ttf"],
    "Regular": [r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"],
}


@lru_cache(maxsize=256)
def font(size=40, weight="Bold"):
    paths = [os.path.join(FONT_DIR, f"Pretendard-{weight}.ttf"),
             os.path.join(FONT_DIR, "Pretendard-Regular.ttf")]
    paths += _FALLBACKS.get(weight, [])
    paths += ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    for p in paths:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def has_pretendard():
    return os.path.exists(os.path.join(FONT_DIR, "Pretendard-Bold.ttf"))
