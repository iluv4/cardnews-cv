# Card-News Layout Detector (Stage 2 of the CV project)

Korean card-news element detector — the trainable deep CV model that recognizes
layout (title / body / logo / underlay). Stage 2 of the plan in
`../CV_PROJECT_PLAN.md`. Adapts PosterLayout (CVPR 2023) to Korean data.

## Status (2026-06-16)
- ✅ 109 curated Korean card images collected (`images/`)
- ✅ EasyOCR **pseudo-labels** generated → YOLO format (`dataset/`, 4 classes)
      - title 284 / body 641 boxes; logo/underlay TODO (need manual labels)
- ⏭️ **NEXT: train YOLOv8n locally (small-data fine-tuning), then scale on cloud**
- ⏭️ Then: Label Studio for proper 4-class manual labels

## Reproduce on ANY machine (this is how you continue without the original PC)
```bash
# Linux / cloud GPU
bash setup.sh
# Windows
./setup.ps1
```
This recreates the venv, installs pinned deps, and re-downloads the public
pretrained weights. The dataset (small) is committed in `dataset/`.

Or fully containerized for cloud:
```bash
docker build -t cardnews-cv .
docker run --gpus all -it -v ${PWD}:/work cardnews-cv
```

## Train (small-data fine-tuning)
```bash
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 \
     batch=8 patience=30 freeze=10 cos_lr=True
```
Planned ablations: freeze vs none · aug on/off · n vs s · LR · k-fold CV.

## Files
- `images/`            raw curated card images (109)
- `dataset/`           YOLO split (images+labels, train/val) + `mapping.csv`
- `dataset.yaml`       class names + paths
- `collect_images.py`  gather images from carnews-insta
- `autolabel.py`       EasyOCR → pseudo-labels
- `viz_labels.py`      draw labels to eyeball quality
- `download_weights.py` re-fetch public PosterLayout weights
- `posterlayout_patches/`  our modified/new PosterLayout scripts (baseline)

## Reproducibility note
We did NOT use Docker from the start, and that's fine: for ML the standard is
Git (code) + pinned deps + cloud storage for big weights + cloud GPU for
training. Docker here is provided only as an optional clean-room for the cloud.
