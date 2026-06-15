"""Train ONE YOLOv8 experiment in an isolated process (so an OOM/crash can't
poison the rest of the overnight suite). Config passed as a JSON file path.
Uses workers=0 (Windows paging-file safe) and amp=False (cuDNN-init safe).
"""
import sys, os, json
os.environ.setdefault("PYTHONUTF8", "1")

cfg = json.load(open(sys.argv[1], encoding="utf-8-sig"))
model = cfg.pop("model")
data = cfg.pop("data")

from ultralytics import YOLO
m = YOLO(model)
m.train(data=data, **cfg)
print("TRAIN_ONE_DONE", flush=True)
