"""Data-scale comparison: does adding the 578 new images help the detector?

Trains the SAME best config (e15_long300_card: yolov8n + AUG_CARD, long schedule)
on two datasets that differ ONLY in size, and evaluates BOTH on the identical
held-out common test set (see build_datasets.py):

    base109  -> dataset_base109.yaml   (train: 109 originals)
    full687  -> dataset_full687.yaml   (train: 109 + 578 new, minus test)

Each is run over several seeds; we report mean +/- std mAP and the delta.
Reuses train_one.py (isolated subprocess per run) so a crash can't poison others.

Run on RunPod AFTER build_datasets.py:
    EPOCHS=300 SEEDS=0,1,2 BATCH=16 WORKERS=8 AMP=1 python compare_data_scale.py
Local smoke test (tiny):
    EPOCHS=3 SEEDS=0 python compare_data_scale.py
"""
import os
import sys
import json
import time
import shutil
import subprocess
import statistics as st

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
RESULTS = os.path.join(ROOT, "results")
FOLDS = os.path.join(ROOT, "folds")
TRAIN_ONE = os.path.join(ROOT, "train_one.py")
PY = sys.executable
for d in (RUNS, RESULTS, FOLDS):
    os.makedirs(d, exist_ok=True)

EPOCHS = int(os.getenv("EPOCHS", "300"))
SEEDS = [int(s) for s in os.getenv("SEEDS", "0,1,2").split(",") if s.strip() != ""]
RUN_TIMEOUT = int(os.getenv("RUN_TIMEOUT_MIN", "120")) * 60

# same knobs as run_overnight.py so local vs cloud share one code path
_BATCH = int(os.getenv("BATCH", "2"))
_WORKERS = int(os.getenv("WORKERS", "0"))
_AMP = os.getenv("AMP", "0") == "1"
_IMGSZ = int(os.getenv("IMGSZ", "640"))

# AUG_CARD + long schedule = the winning e15_long300_card recipe.
AUG_CARD = dict(mosaic=0.5, mixup=0.0, fliplr=0.0, hsv_h=0.01, hsv_s=0.3,
                hsv_v=0.3, translate=0.05, scale=0.3, erasing=0.2)
BEST = dict(model="yolov8n.pt", imgsz=_IMGSZ, batch=_BATCH, epochs=EPOCHS,
            freeze=10, optimizer="auto", lr0=0.01, amp=_AMP, device=0,
            workers=_WORKERS, plots=True, verbose=False, exist_ok=True,
            project=RUNS, **AUG_CARD)

DATASETS = {
    "base109": "dataset_base109.yaml",
    "full687": "dataset_full687.yaml",
}


def parse_metrics(save_dir):
    df = pd.read_csv(os.path.join(save_dir, "results.csv"))
    df.columns = [c.strip() for c in df.columns]
    key = "metrics/mAP50-95(B)"
    best = df.loc[df[key].idxmax()]
    g = lambda k: float(best.get(k, float("nan")))
    return dict(map50=g("metrics/mAP50(B)"), map=g(key),
                precision=g("metrics/precision(B)"), recall=g("metrics/recall(B)"),
                best_epoch=int(best.get("epoch", -1)))


def train_one(name, data, seed):
    cfg = {**BEST, "name": name, "data": os.path.join(ROOT, data), "seed": seed}
    cfgpath = os.path.join(FOLDS, f"_cmp_{name}.json")
    with open(cfgpath, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    env = dict(os.environ); env["PYTHONUTF8"] = "1"
    subprocess.run([PY, TRAIN_ONE, cfgpath], cwd=ROOT, env=env,
                   check=True, timeout=RUN_TIMEOUT)
    return os.path.join(RUNS, name)


def main():
    rows = []
    for tag, yaml in DATASETS.items():
        if not os.path.exists(os.path.join(ROOT, yaml)):
            raise SystemExit(f"missing {yaml} — run build_datasets.py first")
        for seed in SEEDS:
            name = f"cmp_{tag}_seed{seed}"
            t0 = time.time()
            print(f"=== TRAIN {name}  (epochs={EPOCHS}) ===", flush=True)
            try:
                sd = train_one(name, yaml, seed)
                met = parse_metrics(sd)
                rows.append(dict(dataset=tag, seed=seed, status="ok",
                                 minutes=round((time.time()-t0)/60, 1), **met))
                print(f"  DONE {name}: mAP50-95={met['map']:.4f} mAP50={met['map50']:.4f}")
            except Exception as e:  # noqa: BLE001
                rows.append(dict(dataset=tag, seed=seed, status=f"FAIL:{e}",
                                 minutes=round((time.time()-t0)/60, 1)))
                print(f"  FAIL {name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "data_scale_comparison.csv"), index=False)

    # aggregate
    agg = {}
    for tag in DATASETS:
        maps = [r["map"] for r in rows if r["dataset"] == tag and r.get("status") == "ok"]
        m50 = [r["map50"] for r in rows if r["dataset"] == tag and r.get("status") == "ok"]
        if maps:
            agg[tag] = dict(
                n=len(maps),
                map_mean=st.mean(maps), map_std=(st.pstdev(maps) if len(maps) > 1 else 0.0),
                map50_mean=st.mean(m50), map50_std=(st.pstdev(m50) if len(m50) > 1 else 0.0),
            )

    md = ["# Data-scale comparison — 109 vs 687 (Korean card-news detector)", "",
          f"Best config (e15_long300_card: yolov8n + AUG_CARD, epochs={EPOCHS}), "
          f"seeds={SEEDS}. Both evaluated on the SAME held-out common test set "
          "(15% of the new images, unseen by both).", "",
          "| dataset | train imgs | runs | mAP50-95 (mean±std) | mAP50 (mean±std) |",
          "|---|---|---|---|---|"]
    train_sizes = {"base109": "109", "full687": "109 + new(~601)"}
    for tag in DATASETS:
        a = agg.get(tag)
        if a:
            md.append(f"| {tag} | {train_sizes[tag]} | {a['n']} | "
                      f"{a['map_mean']:.4f} ± {a['map_std']:.4f} | "
                      f"{a['map50_mean']:.4f} ± {a['map50_std']:.4f} |")
        else:
            md.append(f"| {tag} | {train_sizes[tag]} | 0 | (no successful run) | |")

    if "base109" in agg and "full687" in agg:
        d = agg["full687"]["map_mean"] - agg["base109"]["map_mean"]
        d50 = agg["full687"]["map50_mean"] - agg["base109"]["map50_mean"]
        pct = (d / agg["base109"]["map_mean"] * 100) if agg["base109"]["map_mean"] else float("nan")
        md += ["", f"**Δ mAP50-95 (full687 − base109): {d:+.4f} ({pct:+.1f}%)**  ",
               f"**Δ mAP50: {d50:+.4f}**", "",
               ("➡️ Adding the new data **helped**." if d > 0 else
                "➡️ Adding the new data did **not** help on this metric (check label quality / class balance).")]
    with open(os.path.join(ROOT, "COMPARISON.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    # chart
    try:
        tags = [t for t in DATASETS if t in agg]
        if tags:
            means = [agg[t]["map_mean"] for t in tags]
            stds = [agg[t]["map_std"] for t in tags]
            plt.figure(figsize=(5, 4))
            plt.bar(tags, means, yerr=stds, capsize=6,
                    color=["#9aa", "#3b7"])
            plt.ylabel("mAP50-95 (common test)")
            plt.title("Effect of adding 578 new images")
            for i, m in enumerate(means):
                plt.text(i, m, f"{m:.3f}", ha="center", va="bottom")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS, "data_scale_chart.png"), dpi=120)
            plt.close()
    except Exception as e:  # noqa: BLE001
        print(f"chart failed: {e}")

    # keep both best weights for downstream use
    for tag in DATASETS:
        oks = [r for r in rows if r["dataset"] == tag and r.get("status") == "ok"]
        if not oks:
            continue
        best = max(oks, key=lambda r: r["map"])
        src = os.path.join(RUNS, f"cmp_{tag}_seed{best['seed']}", "weights", "best.pt")
        if os.path.exists(src):
            dst = os.path.join(RESULTS, "best_model", f"best_{tag}.pt")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    print("\nwrote COMPARISON.md, results/data_scale_comparison.csv, "
          "results/data_scale_chart.png")


if __name__ == "__main__":
    main()
