"""reflib — reference library + retrieval (the product/ML core).

Local tier (no torch): build_index.py + features.py + cluster.py + search.py.
RunPod tier (torch):   embed_clip.py (semantic) + tag_layout.py (structural).
"""
from .search import ReferenceLibrary  # noqa: F401
