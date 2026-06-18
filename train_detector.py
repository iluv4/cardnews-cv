"""Train the PRODUCTION detector used by the generation pipeline.

Improvements over the 640px baseline (to catch small body text / text blocks):
  (1) high input resolution  -> IMGSZ default 1280
  (2) high-recall labels      -> build_datasets.py OCR_PARAMS (run that first)
  (3) text-BLOCK labels       -> build_datasets.py MERGE_BLOCKS=1

Best recipe = e15_long300_card (yolov8n + AUG_CARD, long schedule).
Trains on the full 687-image set and copies the weights to
results/best_model/best.pt, which gen/ loads by default.

Run on RunPod (after build_datasets.py):
  IMGSZ=1280 EPOCHS=300 BATCH=8 WORKERS=8 AMP=1 python train_detector.py
  # tip: imgsz=1280 needs more VRAM — drop BATCH if you OOM, or use an A100.
"""
import os
import sys
import json
import shutil
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
FOLDS = os.path.join(ROOT, "folds")
TRAIN_ONE = os.path.join(ROOT, "train_one.py")
PY = sys.executable
os.makedirs(RUNS, exist_ok=True)
os.makedirs(FOLDS, exist_ok=True)

DATA = os.getenv("DATA", "dataset_full687.yaml")
NAME = os.getenv("NAME", "detector_prod")
IMGSZ = int(os.getenv("IMGSZ", "1280"))
EPOCHS = int(os.getenv("EPOCHS", "300"))
BATCH = int(os.getenv("BATCH", "8"))
WORKERS = int(os.getenv("WORKERS", "8"))
AMP = os.getenv("AMP", "1") == "1"
SEED = int(os.getenv("SEED", "0"))

AUG_CARD = dict(mosaic=0.5, mixup=0.0, fliplr=0.0, hsv_h=0.01, hsv_s=0.3,
                hsv_v=0.3, translate=0.05, scale=0.3, erasing=0.2)


def main():
    data = os.path.join(ROOT, DATA)
    if not os.path.exists(data):
        raise SystemExit(f"missing {DATA} — run build_datasets.py first")
    cfg = dict(model="yolov8n.pt", data=data, name=NAME, imgsz=IMGSZ,
               epochs=EPOCHS, batch=BATCH, workers=WORKERS, amp=AMP, freeze=10,
               optimizer="auto", lr0=0.01, device=0, seed=SEED, plots=True,
               verbose=False, exist_ok=True, project=RUNS, **AUG_CARD)
    cfgpath = os.path.join(FOLDS, f"_cfg_{NAME}.json")
    with open(cfgpath, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)

    print(f"training {NAME}: data={DATA} imgsz={IMGSZ} epochs={EPOCHS} batch={BATCH}")
    env = dict(os.environ); env["PYTHONUTF8"] = "1"
    subprocess.run([PY, TRAIN_ONE, cfgpath], cwd=ROOT, env=env, check=True)

    src = os.path.join(RUNS, NAME, "weights", "best.pt")
    dst = os.path.join(ROOT, "results", "best_model", "best.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"\nproduction detector -> {dst}")
        print("gen/ will now use this automatically.")
    else:
        print(f"!! weights not found at {src}")


if __name__ == "__main__":
    main()
