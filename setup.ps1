# Reproduce the environment on Windows (PowerShell). Usage:  .\setup.ps1
$ErrorActionPreference = "Stop"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "[1/4] create venv"
& $py -m venv .venv
$vpy = ".\.venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip

Write-Host "[2/4] install PyTorch (CUDA 11.3 build)"
& $vpy -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

Write-Host "[3/4] install the rest (pinned)"
& $vpy -m pip install -r requirements-lock.txt

Write-Host "[4/4] download public pretrained weights"
& $vpy download_weights.py

Write-Host "done. Train with:  .\.venv\Scripts\yolo.exe detect train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640"
