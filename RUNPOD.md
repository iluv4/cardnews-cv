# Run the ablation on RunPod (GPU cloud)

The local 16 GB box is too memory-tight; the cloud removes that limit. The dataset
(109 imgs + pseudo-labels) and all code are in this repo, so a pod needs only:
clone → `bash run_cloud.sh`.

## 1. Create the pod
- RunPod → **Pods → Deploy**
- **GPU:** RTX 4090 (24 GB) — best value (~$0.4–0.7/hr; full run ≈1–1.5 h ≈ $1).
- **Template:** an official **PyTorch** image (torch + CUDA preinstalled).
- **Container disk:** ~20 GB. Deploy, then **Connect → Web Terminal** (or Jupyter).

## 2. Clone (private repo → use a GitHub token in the URL)
```bash
git clone https://<YOUR_GH_TOKEN>@github.com/iluv4/cardnews-cv.git
cd cardnews-cv
```
(Make a token at github.com/settings/tokens with `repo` scope. Revoke it after.)

## 3. Run everything
```bash
bash run_cloud.sh
```
This installs ultralytics and runs the full study with cloud settings
(`BATCH=16 WORKERS=8 AMP=1`, ~6 h budget — it finishes well before that).

Override anything inline, e.g. a quick pass:
```bash
BATCH=32 TIME_BUDGET_H=2 bash run_cloud.sh
```

To auto-push results back to GitHub as it runs:
```bash
export GH_TOKEN=ghp_xxx        # same token
bash run_cloud.sh
```

## 4. Get the results
- `results/ablation.csv` + `ABLATION.md` — the comparison table (your paper's experiments)
- `results/ablation_chart.png` — mAP bar chart
- `results/<exp>/` — training curves + val predictions per experiment
- `runs/<best_exp>/weights/best.pt` — the trained detector

Download via the Jupyter file browser, or:
```bash
tar czf results.tgz results ABLATION.md DEV_LOG.md runs/*/weights/best.pt
```
(or `runpodctl send results.tgz`). If you set `GH_TOKEN`, they're already pushed.

## 5. Stop the pod
RunPod → Pods → **Stop/Terminate** (so you don't keep paying).

## Notes
- Same `run_overnight.py` runs locally (defaults batch=2/workers=0/amp off) and on
  cloud (env overrides) — no separate code path.
- `dataset.yaml` path is rewritten at runtime, so it's portable.
- For a much larger future dataset or DS-GAN retrain (Stage 5), pick an A100 80 GB.
