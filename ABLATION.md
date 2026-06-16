# Small-data fine-tuning ablation — Korean card-news detector

_109 images (train 93 / val 16), YOLOv8, amp=False, workers=0, RTX 2060._
_Updated +24 min into the run._

| name | model | map50 | map | precision | recall | best_epoch | n_epochs | minutes | status |
|---|---|---|---|---|---|---|---|---|---|
| e15_long300_card | yolov8n.pt | 0.8537 | 0.7183 | 0.7849 | 0.8423 | 219 | 300 | 1.4000 | ok |
| e13_yolov8s_freeze10 | yolov8s.pt | 0.8522 | 0.7050 | 0.8784 | 0.8124 | 143 | 150 | 1.1000 | ok |
| e07_aug_card_seed2 | yolov8n.pt | 0.8406 | 0.7028 | 0.8629 | 0.7744 | 136 | 150 | 0.8000 | ok |
| e07_aug_card_seed1 | yolov8n.pt | 0.8396 | 0.7019 | 0.8641 | 0.7685 | 136 | 150 | 0.8000 | ok |
| e14_yolov8s_freeze0 | yolov8s.pt | 0.8510 | 0.6964 | 0.8323 | 0.8113 | 104 | 150 | 1.4000 | ok |
| e07_aug_card | yolov8n.pt | 0.8385 | 0.6960 | 0.8770 | 0.7582 | 137 | 150 | 0.8000 | ok |
| e03_n_freeze5 | yolov8n.pt | 0.8360 | 0.6824 | 0.8108 | 0.8528 | 138 | 150 | 0.9000 | ok |
| e01_baseline_n_freeze10_seed1 | yolov8n.pt | 0.8441 | 0.6818 | 0.7615 | 0.8190 | 142 | 150 | 0.8000 | ok |
| kfold_2 | yolov8n.pt | 0.8543 | 0.6814 | 0.7969 | 0.8292 | 97 | 120 | 0.7000 | ok |
| e10_imgsz768 | yolov8n.pt | 0.8413 | 0.6799 | 0.7780 | 0.7553 | 140 | 150 | 1.0000 | ok |
| e09_imgsz512 | yolov8n.pt | 0.8377 | 0.6755 | 0.8135 | 0.7513 | 141 | 150 | 0.7000 | ok |
| e01_baseline_n_freeze10_seed2 | yolov8n.pt | 0.8445 | 0.6729 | 0.7992 | 0.7613 | 119 | 150 | 0.8000 | ok |
| e02_n_freeze0 | yolov8n.pt | 0.8512 | 0.6709 | 0.8212 | 0.8507 | 127 | 150 | 1.0000 | ok |
| e01_baseline_n_freeze10 | yolov8n.pt | 0.8443 | 0.6617 | 0.8120 | 0.7526 | 132 | 150 | 0.8000 | ok |
| e08_fliplr_on_BAD | yolov8n.pt | 0.8443 | 0.6617 | 0.8120 | 0.7526 | 132 | 150 | 0.8000 | ok |
| e12_lr_low | yolov8n.pt | 0.8443 | 0.6617 | 0.8120 | 0.7526 | 132 | 150 | 0.8000 | ok |
| e11_adamw | yolov8n.pt | 0.8120 | 0.6538 | 0.7851 | 0.7387 | 143 | 150 | 0.8000 | ok |
| kfold_0 | yolov8n.pt | 0.7933 | 0.6518 | 0.6932 | 0.7832 | 96 | 120 | 0.7000 | ok |
| e05_aug_none | yolov8n.pt | 0.8180 | 0.6456 | 0.8097 | 0.7797 | 55 | 150 | 0.8000 | ok |
| e06_aug_heavy | yolov8n.pt | 0.8381 | 0.6439 | 0.7588 | 0.8078 | 136 | 150 | 0.9000 | ok |
| kfold_3 | yolov8n.pt | 0.7897 | 0.5996 | 0.8540 | 0.6943 | 117 | 120 | 0.7000 | ok |
| kfold_1 | yolov8n.pt | 0.7369 | 0.5695 | 0.7038 | 0.6874 | 120 | 120 | 0.7000 | ok |
| kfold_4 | yolov8n.pt | 0.7351 | 0.5503 | 0.7701 | 0.7120 | 119 | 120 | 0.7000 | ok |
| e04_n_scratch | yolov8n.yaml | 0.7359 | 0.5017 | 0.7321 | 0.7101 | 111 | 150 | 1.0000 | ok |

**Best so far:** `e15_long300_card` — mAP50-95 = 0.7183, mAP50 = 0.8537