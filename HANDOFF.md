# cardnews-cv — Handoff / Status (2026-06-18)

Single source of truth to continue in a fresh chat. Repo: `Downloads/cardnews-cv`
(remote: https://github.com/iluv4/cardnews-cv, branch `main`). Python = `py -3`
(3.13, Windows). GPU present: RTX 3050 6GB (CUDA 12.6). torch NOT installed
locally (heavy ML → RunPod). Installed locally: Pillow, numpy, opencv,
matplotlib, reportlab.

## What this project is
Korean card-news (카드뉴스) layout **analysis + generation**, started as a CV
class project (adapting PosterLayout/DS-GAN, CVPR2023) and now steering toward a
**real, startup-grade product**. Goal quality bar: looks like a human designer
made it — NOT "AI-tic".

## Key decisions (do not re-litigate)
- **Template/render engine > DS-GAN for the PRODUCT.** GAN can't render legible
  Korean text and is unstable. DS-GAN stays as a research/report baseline only.
- **The ML value is in analysis + retrieval, not pixel generation:** trained
  detector (layout understanding) + reference search (embeddings) + auto-fill.
  Deterministic high-quality rendering on top.
- **Product direction = search/select over a large REAL reference library, then
  copy that layout and refill with the user's text.** Selection-based UX (avoids
  random "AI-tic" output). Engine must extract layout from ANY reference, not
  just hand-coded templates.
- Detector (Stage 2) is DONE: best `e15_long300_card` mAP@50-95=0.718,
  mAP@50=0.854; 5-fold 0.611±0.049; transfer learning decisive (scratch 0.502).

## What is BUILT and verified (runs locally unless noted)
- `prepare_new_images.py` ✅ ran → `images_new/` (578 imgs, 66 decks, 95MB).
- `build_datasets.py` (RunPod) — leak-free 109-vs-687 datasets, high-recall OCR
  (canvas_size 2560, mag_ratio 2.0, low_text 0.3) + line→block merging.
- `compare_data_scale.py` (RunPod) — 109 vs 687 detector comparison.
- `train_detector.py` (RunPod) — production detector, imgsz 1280, → results/best_model/best.pt.
- `gen/` (RunPod GPU): `build_posterlayout_dataset.py` (saliency U2/ISNet + LaMa
  inpaint + train_csv), `render.py` (DS-GAN output → composited card; has `--demo`),
  `copy_layout.py` (layout-copy demo using committed labels + cv2 inpaint; ran).
- `analysis/deck_dynamics.py` ✅ ran → time-series of deck page-position
  (covers darker/more saturated/less edge-dense than interiors). fig + csv.
- `paper/` — CVPR-style LaTeX (`main.tex`, `refs.bib`, `figures/`). A background
  agent was building `paper/build_pdf.py` → `paper/cardnews_paper.pdf` + `paper/README.md`
  (check if it finished; Overleaf is the submission path).
- `service/cardgen/` — engine v1 (assets/themes/templates/placement/render).
  **Output judged too generic / "AI-tic"** vs references → being redesigned.

## Reference design language (from the corpus — MATCH THIS)
Example: `newData(smartFarmKorea)/Card News_Tomato Education/`. Real gov/agri
card-news use: dark themed bg + faint scattered deco icons (consistent across
deck) + **two-tone glowing title** (white + accent/pink) + keyword-highlighted
subtitle + **white rounded panel with an accent checklist + dotted dividers** +
brand logo top-right + sometimes a mascot/illustration. The engine v1 lacked all
of this (flat gradient + text only). themes.py was just upgraded with rich color
fields (title_accent, panel, check, deco, brand) to support these components;
render.py still needs the component rewrite (scatter, glow two-tone title,
white panel + checklist, keyword highlight).

## NEXT STEPS (priority order)
1. **Reference library + retrieval (the real product/ML core):**
   - Index all images: detector → layout-type tag; color/palette extract; CLIP
     embeddings for semantic search; cluster into archetypes.
   - Search API: text query / filters (topic, color, layout) → ranked refs.
2. **Layout extraction from ANY selected reference** (generalize copy_layout.py
   beyond committed labels → run detector live; needs torch on RunPod or local).
3. **Engine v2 components** to hit reference quality: scatter deco, two-tone glow
   title, white panel + checklist, keyword highlight, brand/mascot slot.
4. **Service**: FastAPI `POST /generate` + search endpoints + simple web UI
   (search → select reference → auto-fill text → edit → export deck).
5. RunPod runbook (RUNPOD_GENERATION.md) for detector/DS-GAN/labeling.

## Customer segments (planning)
① 공공/정책 홍보  ② 농업·스마트팜 기관  ③ 소상공인·마케터  ④ 콘텐츠 크리에이터.
**MVP focus = ①+②** (our data strength). Product = search real designs → select →
auto-fill → export deck.

## How to run (local, no torch)
- New cards demo: `py -3 service/demo.py` → `service/out/`
- Layout-copy demo: `py -3 gen/copy_layout.py --n 4` → `gen_output/copy/`
- Deck time-series: `py -3 analysis/deck_dynamics.py` → `analysis/`
- Paper figures: `py -3 paper/make_figures.py` → `paper/figures/`

## RunPod (heavy ML) — public repo, no token needed to clone
```
git clone https://github.com/iluv4/cardnews-cv.git && cd cardnews-cv
pip install -U ultralytics easyocr pillow numpy pandas matplotlib
python build_datasets.py            # labels -> data_cmp + yamls
EPOCHS=300 SEEDS=0,1,2 BATCH=16 WORKERS=8 AMP=1 python compare_data_scale.py
IMGSZ=1280 python train_detector.py # production detector -> results/best_model/best.pt
# generation: pip install rembg onnxruntime simple-lama-inpainting; python gen/build_posterlayout_dataset.py
```
Local work committed & pushed (commit "Add new 578-image dataset + ...").
Uncommitted as of handoff: build_datasets/compare/train_detector edits, gen/,
analysis/, paper/, service/ — commit before relying on a fresh clone.
