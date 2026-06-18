"""Layout templates = ordered element slots in normalized [0,1] coords.

Slot: (role, x0, y0, x1, y1, align)  role in {eyebrow,title,body,logo,rule}.
These archetypes are distilled from real Korean card-news layouts (the kind the
detector extracts from the corpus). `auto` picks one from the content shape.
"""

TEMPLATES = {
    # eyebrow + big title top-left, body under, logo bottom-left
    "editorial": [
        ("eyebrow", 0.09, 0.10, 0.70, 0.145, "left"),
        ("title",   0.09, 0.16, 0.93, 0.40,  "left"),
        ("rule",    0.09, 0.43, 0.30, 0.437, "left"),
        ("body",    0.09, 0.48, 0.91, 0.72,  "left"),
        ("logo",    0.09, 0.90, 0.30, 0.95,  "left"),
    ],
    # centered title in the middle, body under — strong, simple
    "centered": [
        ("eyebrow", 0.10, 0.30, 0.90, 0.35,  "center"),
        ("title",   0.08, 0.37, 0.92, 0.57,  "center"),
        ("body",    0.12, 0.62, 0.88, 0.78,  "center"),
        ("logo",    0.40, 0.90, 0.60, 0.95,  "center"),
    ],
    # bottom panel (good over photos): title+body anchored low
    "bottom": [
        ("title",   0.08, 0.56, 0.92, 0.76,  "left"),
        ("rule",    0.08, 0.79, 0.28, 0.797, "left"),
        ("body",    0.08, 0.82, 0.90, 0.93,  "left"),
        ("logo",    0.80, 0.06, 0.94, 0.11,  "right"),
    ],
    # cover: huge title, minimal — first page of a deck
    "cover": [
        ("eyebrow", 0.10, 0.40, 0.90, 0.45,  "center"),
        ("title",   0.06, 0.46, 0.94, 0.66,  "center"),
        ("logo",    0.40, 0.90, 0.60, 0.95,  "center"),
    ],
}

ORDER = ["editorial", "centered", "bottom", "cover"]


def get(name):
    return TEMPLATES.get(name, TEMPLATES["editorial"])


def auto(title, body, has_photo=False):
    """Pick a template from content shape."""
    if has_photo:
        return TEMPLATES["bottom"]          # keep text low, image visible on top
    if not body:
        return TEMPLATES["cover"]           # title-only -> cover
    if len(title) <= 14 and len(body) <= 60:
        return TEMPLATES["centered"]
    return TEMPLATES["editorial"]
