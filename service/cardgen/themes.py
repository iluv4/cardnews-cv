"""Rich design themes distilled from real Korean card-news (e.g. gov/agri decks:
dark themed background, scattered faint deco icons, two-tone glowing title, a
white rounded content panel with an accent checklist, brand logo).

Each theme carries enough colors to build those components consistently.
"""

THEMES = {
    "forest": {           # smart-farm / agriculture (EPIS-style dark green + pink)
        "bg": [(21, 64, 43), (9, 36, 25)], "angle": 90,
        "title": (255, 255, 255), "title_accent": (247, 168, 184),
        "body": (216, 235, 223), "accent": (244, 138, 160),
        "panel": (249, 251, 249), "panel_text": (24, 58, 42), "panel_sub": (110, 130, 116),
        "check": (244, 120, 150), "deco": (255, 255, 255), "brand": (255, 255, 255),
        "dark": True,
    },
    "midnight": {
        "bg": [(26, 34, 64), (10, 13, 27)], "angle": 90,
        "title": (255, 255, 255), "title_accent": (255, 201, 71),
        "body": (210, 218, 238), "accent": (255, 201, 71),
        "panel": (248, 249, 252), "panel_text": (22, 28, 52), "panel_sub": (95, 105, 130),
        "check": (245, 180, 40), "deco": (255, 255, 255), "brand": (255, 255, 255),
        "dark": True,
    },
    "coral": {
        "bg": [(255, 122, 92), (240, 78, 110)], "angle": 60,
        "title": (255, 255, 255), "title_accent": (255, 240, 150),
        "body": (255, 238, 232), "accent": (255, 246, 170),
        "panel": (255, 252, 250), "panel_text": (140, 45, 55), "panel_sub": (175, 100, 100),
        "check": (240, 110, 90), "deco": (255, 255, 255), "brand": (255, 255, 255),
        "dark": True,
    },
    "mono": {              # light, editorial (white + near-black + red accent)
        "bg": [(252, 252, 250), (238, 240, 238)], "angle": 90,
        "title": (24, 24, 26), "title_accent": (230, 57, 70),
        "body": (70, 72, 75), "accent": (230, 57, 70),
        "panel": (255, 255, 255), "panel_text": (28, 28, 30), "panel_sub": (120, 122, 125),
        "check": (230, 57, 70), "deco": (40, 50, 60), "brand": (24, 24, 26),
        "dark": False,
    },
    "mint": {
        "bg": [(214, 245, 235), (176, 230, 217)], "angle": 90,
        "title": (16, 58, 48), "title_accent": (0, 150, 120),
        "body": (33, 84, 72), "accent": (0, 150, 120),
        "panel": (255, 255, 255), "panel_text": (16, 58, 48), "panel_sub": (90, 130, 118),
        "check": (0, 150, 120), "deco": (16, 90, 70), "brand": (16, 58, 48),
        "dark": False,
    },
    "sky": {
        "bg": [(86, 140, 245), (52, 96, 220)], "angle": 75,
        "title": (255, 255, 255), "title_accent": (255, 236, 150),
        "body": (234, 242, 255), "accent": (255, 236, 150),
        "panel": (255, 255, 255), "panel_text": (28, 50, 95), "panel_sub": (95, 120, 165),
        "check": (52, 120, 235), "deco": (255, 255, 255), "brand": (255, 255, 255),
        "dark": True,
    },
}

ORDER = ["forest", "midnight", "coral", "mono", "mint", "sky"]


def get(name):
    return THEMES.get(name, THEMES["forest"])


def pick(seed=0):
    return THEMES[ORDER[seed % len(ORDER)]]
