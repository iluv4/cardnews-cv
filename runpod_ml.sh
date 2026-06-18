#!/usr/bin/env bash
# RunPod (또는 GPU 머신) 한 방 실행: 전 코퍼스 템플릿 + CLIP 의미검색 임베딩.
# 끝나면 reflib_artifacts.tar.gz 를 만들어 줍니다 -> 로컬로 내려받아 repo 루트에서 풀면 됨.
#
#   git clone https://github.com/iluv4/cardnews-cv.git && cd cardnews-cv
#   bash runpod_ml.sh
#   # -> reflib_artifacts.tar.gz 다운로드
#
# 로컬에서:
#   tar xzf reflib_artifacts.tar.gz        # repo 루트에서
#   .\run_demo.ps1                          # 의미검색 ON + 전 레퍼런스 레이아웃 복사
set -e

echo "== deps =="
pip install -q ultralytics open_clip_torch torch torchvision pillow numpy pandas

echo "== index (full corpus) =="
python reflib/build_index.py

echo "== layout templates for the FULL corpus (detector) =="
python gen/extract_templates_detector.py --imgsz 1280

echo "== CLIP embeddings (semantic search) =="
python reflib/embed_clip.py --build

echo "== package artifacts =="
tar czf reflib_artifacts.tar.gz \
    service/library/templates.json \
    reflib/data/clip.npy \
    reflib/data/clip_ids.json \
    reflib/data/index.json

echo
echo "DONE -> download reflib_artifacts.tar.gz, then locally:"
echo "        tar xzf reflib_artifacts.tar.gz   (at repo root)"
ls -lh reflib_artifacts.tar.gz
