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
- `service/cardgen/` — **engine v2 DONE** (assets/themes/templates/placement/
  render + new `components.py`). Matches the EPIS reference look: dark vignette
  bg + consistent scattered deco + two-tone GLOW title (white line + accent
  line, per-line halo) + *keyword*-highlighted subtitle (word-aware wrap) +
  white rounded checklist panel (light-tint check boxes + accent check mark +
  dotted dividers, balanced 2-line wrapping) + brand mark + vector tomato
  mascot in a speech bubble. `kind=cover|checklist|statement` (auto from
  content). Photo path kept from v1. Verified locally `py -3 service/demo.py`
  → `service/out/` (square 1080² deck + per-theme cards). Old v1 output was
  judged too generic / "AI-tic"; this closes that gap.

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
1. **Reference library + retrieval (the real product/ML core): DONE local tier**
   (2026-06-18). `reflib/` indexes all 687 imgs → `reflib/data/index.json`
   (filename meta + palette/brightness/edge/dark + color/feel vector), clusters
   into archetypes (`cluster.py`), and serves search (`search.py`,
   `ReferenceLibrary`): lexical text, color, filters (dark/cover/source/cluster),
   `similar_to`. Runs today with NO torch (`py -3 reflib/build_index.py` →
   `cluster.py` → `search.py`). **CLIP + layout tags now BUILT LOCALLY**
   (2026-06-18, CPU torch — torch IS runnable locally for inference, no RunPod
   needed): `reflib/embed_clip.py --build` → `reflib/data/clip.npy` [796×512]
   (gitignored; regenerate locally), `reflib/tag_layout.py` → per-record `layout`
   signature in index.json (top: title-top|body-bot 167, title-top|body-mid 128…),
   `reflib/cluster.py --k 8` re-run on CLIP (basis=clip, baked into committed
   index.json). `similar_to` auto-uses CLIP when clip.npy is present; cluster filter
   is semantic. Text search stays lexical on purpose (ViT-B-32 laion2b is English).
   See `reflib/README.md`.
2. **Layout extraction from ANY selected reference: DONE (FULL CORPUS, LOCAL)**
   (2026-06-18). `gen/extract_templates.py` turns the committed labels into a
   refillable template per card; **`gen/extract_templates_detector.py` now ran the
   trained detector over the full 796-img corpus locally (CPU torch, ~2 min) →
   `service/library/templates.json` = 763 templates** (was 95), covering 763/796
   indexed refs (archetypes cover=216, list=271, statement=276; the 33 misses had
   no detections → theme fallback). `service/cardgen/from_template.py`
   (`render_from_template`) copies a template's actual title/body blocks + palette
   and refills with user text. Verified: smart-farm search returns 24/24
   template-backed, deck render mode=`template`. So **the deck workflow's "레이아웃
   복사" is now the norm, theme fallback the exception (4%)**. NOTE: two RunPod-era
   scripts had to be fixed to run locally — `extract_templates_detector.py`
   (gen/common vs reflib/common name collision; now loads reflib/common via
   importlib) and `reflib/tag_layout.py` (added ROOT to sys.path for
   `from gen.common import load_yolo`). Detector weights: `results/best_model/best.pt`.
3. ~~Engine v2 components to hit reference quality~~ **DONE** (2026-06-18) —
   but NOTE THE ROLE: v2 is the **RENDERER (quality layer), one hand-authored
   archetype**, NOT the product. The product needs **one layout template per
   reference in the dataset** (≈687 imgs / 66 deck archetypes), extracted by the
   detector, not hand-coded. Pipeline = detector → extract slots/structure from
   each ref → store as template (library) → user searches/selects → copy that
   template → refill with user text → render with v2 components. So engine
   count == dataset size (via extraction), and steps 1–2 ARE the real engine.
   `copy_layout.py` is the seed of this; live detector extraction needs
   torch/RunPod. v2 polish ideas (lower priority): more deco icons, real mascot
   art, per-segment theme presets, topic→theme auto-pick.
4. **Service: DONE local-verifiable** (2026-06-18). `service/app.py` (FastAPI) +
   `service/static/index.html`: `/api/search`, `/api/similar/{id}`,
   `/api/clusters`, `/api/reference/{id}`, `POST /api/generate` (PNG),
   `POST /api/deck` (zip), `POST /api/generate_from_ref` (copy the selected ref's
   layout). reflib now also indexes the labeled `dataset/` imgs (796 total) so the
   95 template-backed refs are searchable (UI shows a "레이아웃" badge + has_template).
   **Web UI redesigned to a deck-centric 5-step production workflow** (2026-06-18,
   `service/static/index.html`): ① 기획(덱 제목·브랜드·세그먼트·검색 키워드) →
   ② 구성(다중 카드 아웃라인 에디터: 추가/삭제/순서변경, 카드별 제목·부제·체크리스트,
   종류 자동판별 커버/체크리스트/본문) → ③ 스타일(레퍼런스 검색·1개 선택, 덱 전체에
   톤/레이아웃 적용) → ④ 생성·검토(덱 전체 렌더 → 필름스트립, 클릭 확대, '수정'→②로) →
   ⑤ 내보내기(ZIP 다운로드 + 레퍼런스 평가). 상태는 localStorage 저장(새로고침 보존).
   Backend got **two deck endpoints** (`POST /api/deck/render` → list of base64 PNG
   data-URLs for the filmstrip; `POST /api/deck/export` → zip), both via the shared
   `_render_deck(ref_id, cards, brand, size, seed)` helper: template-backed ref →
   every card copies the real layout (mode `template`); ref w/o template → themed v2
   engine with palette synthesized from the ref, cover + page-numbered interiors
   (mode `theme`); no ref → default theme (`theme-default`). The older single-card
   endpoints (`/api/generate`, `/api/generate_from_ref`, `/api/deck`) are kept intact.
   Verified end-to-end (TestClient + in-browser via Claude Preview): search→select→
   render(템플릿/테마 양쪽)→export 모두 동작, 렌더 품질=레퍼런스 디자인 언어 일치.
   `.claude/launch.json` added (preview server config). Run:
   `py -3 -m uvicorn service.app:app --reload --port 8000` (deps:
   `pip install fastapi "uvicorn[standard]"`). The themed fallback shrinks to zero
   once step 2's RunPod pass gives every searchable ref a template.
5. RunPod runbook (RUNPOD_GENERATION.md) for detector/DS-GAN/labeling.

## Customer segments (planning)
① 공공/정책 홍보  ② 농업·스마트팜 기관  ③ 소상공인·마케터  ④ 콘텐츠 크리에이터.
**MVP focus = ①+②** (our data strength). Product = search real designs → select →
auto-fill → export deck.

## How to run (local, no torch)
- Build reference library: `py -3 reflib/build_index.py` then `py -3 reflib/cluster.py --k 8`
- Search refs (CLI): `py -3 reflib/search.py --text "smart farm" --dark --k 8`
- Service + web UI: `py -3 -m uvicorn service.app:app --reload --port 8000` → http://127.0.0.1:8000/
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
