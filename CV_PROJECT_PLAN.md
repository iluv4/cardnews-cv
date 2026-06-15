# Semester CV Project — Content-Aware Layout Generation for Korean Card-News
### (Adapting PosterLayout / DS-GAN, CVPR 2023, to a Korean dataset)

**Course fit:** Computer Vision class — requirement is to *train/fine-tune a deep CV model*.
**Chosen direction:** A — Detection → PosterLayout.
**Timeline:** semester (3+ weeks).
**Compute:** cloud GPU (see §6). Local box (RTX 2060 6 GB) = dev/inference only.

---

## 1. Problem (the CV framing)
Content-aware visual-textual layout generation: given a *clean background image* + a
*saliency map*, predict where to place each design element — **title / body / logo /
underlay** — as (class, bounding-box). Good layouts keep elements off salient regions
(non-occlusion), readable, and aligned. We **reproduce and adapt DS-GAN** (PosterLayout)
to Korean card-news (인스타 카드뉴스).

## 2. Why this is a legitimate CV project
You train/fine-tune **three** deep vision models, not just call an API:
1. **Element detector** (YOLOv8 / DETR) — detect + classify card elements → boxes. *(primary trained model)*
2. **Saliency model** (U²-Net / BASNet) — produce saliency maps (PosterLayout uses BASNet ⊕ PFPN). *(inference, optional fine-tune)*
3. **DS-GAN layout generator** — fine-tune/retrain on the assembled Korean dataset. *(capstone)*
Plus **inpainting** (LaMa) to erase elements → clean training backgrounds (classical/deep image processing).

## 3. The data problem (READ THIS FIRST)
- Current usable images in repo: **~100–150** (41 `public/saved-refs` + themed `references/docs/images/*`, minus blog screenshots).
- Detector fine-tune: OK with ~150 labeled + augmentation.
- DS-GAN from scratch: needs thousands (orig = **9,973**). With ~150 → fine-tune only, weak.
- **Action:** collect **1,000–3,000** Korean card-news images (Instagram hashtags like #카드뉴스, Naver blog/post card decks). Detector then auto-labels them → enough to retrain the layout model. Track source/license for the report.

## 4. PosterLayout data format we must PRODUCE from raw images
```
Dataset/
├─ train/
│  ├─ inpainted_poster/     # background with elements erased (Stage 4)
│  ├─ saliencymaps_basnet/  # saliency (Stage 3)
│  ├─ saliencymaps_pfpn/    # 2nd saliency (Stage 3)
├─ test/
│  ├─ image_canvas/         # clean test backgrounds
│  ├─ saliencymaps_basnet/
│  ├─ saliencymaps_pfpn/
└─ train_csv.csv            # per-element: poster_path, cls_elem(1/2/3), box_elem [x,y,x,y]  (Stage 2)
```
(cls_elem: 1=text, 2=logo, 3=underlay/decoration.)

## 5. Pipeline — each stage = a report section / CV deliverable
**Stage 1 — Collect & curate.** Gather N Korean card images; dedupe; resize to canonical 513×750 (PosterLayout's ratio).

**Stage 2 — Annotation + train the detector.**
- Hand-label ~150–300 images (title/body/logo/underlay boxes) in **Roboflow** or **Label Studio**.
- Bootstrap text boxes with a Korean text detector (CRAFT / DBNet) to cut labeling time.
- Train **YOLOv8** (or DETR) on the labels → run on the *whole* set → auto-annotations → `train_csv.csv`.
- *CV content:* object detection, transfer learning, mAP eval, augmentation.

**Stage 3 — Saliency.** Run **U²-Net / BASNet** to make saliency maps (two variants ≈ basnet + pfpn). Optionally fine-tune on a few hand-masked cards. *CV content:* salient-object detection.

**Stage 4 — Inpainting → clean backgrounds.** Build masks from detected element boxes; erase with **LaMa** (or OpenCV `inpaint` baseline) → `inpainted_poster/` + `image_canvas/`. *CV content:* image inpainting.

**Stage 5 — Layout model (capstone).** Assemble `Dataset/` in PosterLayout format; **fine-tune/retrain DS-GAN** on cloud; evaluate with the paper's metrics (`val, ove, ali, und_l, und_s, uti, occ, rea`); ablate (pretrained vs fine-tuned vs scratch). *CV content:* GAN, content-aware generation, quantitative eval.

**Stage 6 (optional) — Integrate** the trained layout model into `carnews-insta` as a real CV layout tool (replaces / augments the LLM-only placement).

## 6. Cloud GPU — recommendation (you asked for "professional")
| Platform | Why | Best for |
|---|---|---|
| **RunPod** ⭐ primary | On-demand A100/4090, persistent volumes, SSH+Jupyter, PyTorch templates, cheap | All three trainings; multi-hour jobs |
| **Lambda Cloud** | Clean, reliable A100/H100, simple | Premium alternative |
| **GCP Vertex AI / AWS SageMaker** | Managed training jobs, MLOps | If the class wants "enterprise" rigor |
| **Kaggle** (free T4×2, 30h/wk) / **Colab Pro+** | Free / cheap | Detector & saliency experiments |

A **single A100 40/80 GB** handles everything (detector, saliency, DS-GAN). DS-GAN orig used 4 GPUs @ batch 128; on one A100 use batch 16–32.

## 7. Timeline (semester)
- **W1–2:** data collection start; local env (done) + PosterLayout **smoke-test baseline**.
- **W3–4:** annotation tooling + hand-label + **train detector** (mAP).
- **W5–6:** auto-annotate full set + saliency + inpainting → assemble `Dataset/`.
- **W7–9:** **train/fine-tune DS-GAN** on cloud + metrics + ablations.
- **W10+:** (optional) integrate into app; write-up + presentation.

## 8. Status / done so far (local)
- ✅ Python 3.10 + venv, PyTorch 1.12.1+cu113 (CUDA works on RTX 2060), all PosterLayout deps.
- ✅ Pretrained weights downloaded (resnet18/50 backbones + DS-GAN-Epoch300).
- ◐ `infer.py` patched for single-GPU/Windows (finishing the smoke test next).

## 9. Immediate next steps
1. Finish PosterLayout local smoke test → confirmed working baseline to fine-tune from.
2. Pick data-collection source + target count (1–3k); stand up Roboflow/Label Studio.
3. Define the 3–4 element classes precisely (title/body/logo/underlay) + labeling guide.
