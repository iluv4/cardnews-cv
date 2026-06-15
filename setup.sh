#!/usr/bin/env bash
# Reproduce the environment on Linux / cloud GPU (RunPod, Colab, Lambda, etc.)
# Usage: bash setup.sh
set -e

PY=${PYTHON:-python3.10}
echo "[1/4] create venv"
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo "[2/4] install PyTorch (CUDA 11.3 build; bump to cu118/cu121 for H100)"
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

echo "[3/4] install the rest (pinned)"
pip install -r requirements-lock.txt

echo "[4/4] download public pretrained weights"
python download_weights.py || true

echo "done. dataset is in dataset/ ; train with:"
echo "  yolo detect train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640"
