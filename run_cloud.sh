#!/usr/bin/env bash
# RunPod one-shot: run the full small-data fine-tuning ablation on a cloud GPU.
# Use on a RunPod *PyTorch* pod (torch + CUDA already installed).
#
# From the pod's web terminal:
#   git clone https://<YOUR_GH_TOKEN>@github.com/iluv4/cardnews-cv.git
#   cd cardnews-cv && bash run_cloud.sh
#
set -e
echo "[1/2] installing ultralytics (uses the pod's preinstalled torch)"
pip install -q -U ultralytics pandas matplotlib

# Cloud has plenty of RAM/VRAM -> big batch, multiple workers, AMP on, short budget.
export BATCH=${BATCH:-16}
export WORKERS=${WORKERS:-8}
export AMP=${AMP:-1}
export IMGSZ=${IMGSZ:-640}
export TIME_BUDGET_H=${TIME_BUDGET_H:-6}
# Optional: auto-push results back to GitHub (set your token first):
#   export GH_TOKEN=ghp_xxx
echo "[2/2] launching ablation  (BATCH=$BATCH WORKERS=$WORKERS AMP=$AMP IMGSZ=$IMGSZ budget=${TIME_BUDGET_H}h)"
python run_overnight.py

echo "DONE. Results in results/ (ablation.csv, ablation_chart.png), ABLATION.md, runs/<exp>/weights/best.pt"
echo "Download with runpodctl, or: tar czf results.tgz results ABLATION.md DEV_LOG.md && (download via Jupyter)"
