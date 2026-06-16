"""Autonomous ~10h small-data fine-tuning OPTIMIZATION study for the Korean
card-news layout detector (YOLOv8). Time-budgeted queue of experiments + seed
repeats + k-fold CV. ROBUST: each experiment runs in its OWN subprocess
(train_one.py) so a crash/OOM can't kill the suite; the orchestrator itself
never imports torch/CUDA (only one CUDA process at a time). Results + status +
chart are written and git-committed after EACH run.

Launch (background):
  $env:PYTHONUTF8="1"; $env:GH_TOKEN="..."(optional); python run_overnight.py
"""
from __future__ import annotations
import os, sys, time, glob, json, shutil, subprocess, traceback, random

os.environ.setdefault("PYTHONUTF8", "1")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
RUNS = os.path.join(ROOT, "runs")
FOLDS = os.path.join(ROOT, "folds")
DATA = os.path.join(ROOT, "dataset.yaml")
PY = sys.executable                       # the venv python running this script
TRAIN_ONE = os.path.join(ROOT, "train_one.py")
for d in (RESULTS, RUNS, FOLDS):
    os.makedirs(d, exist_ok=True)

START = time.time()
TIME_BUDGET = float(os.getenv("TIME_BUDGET_H", "9.5")) * 3600
RUN_TIMEOUT = int(os.getenv("RUN_TIMEOUT_MIN", "90")) * 60   # per-experiment hard cap
LOG = os.path.join(ROOT, "overnight.log")
MASTER_CSV = os.path.join(RESULTS, "ablation.csv")
ALL_ROWS: list[dict] = []

# workers=0 -> no DataLoader subprocess spawning (Windows paging-file safe).
# Knobs are env-configurable so the SAME script runs on the tight local box
# (defaults batch=2/workers=0/amp off — fits ~4GB commit headroom) and on a cloud
# GPU (RunPod: BATCH=16 WORKERS=8 AMP=1 -> fast, no memory issues).
_BATCH = int(os.getenv("BATCH", "2"))
_WORKERS = int(os.getenv("WORKERS", "0"))
_AMP = os.getenv("AMP", "0") == "1"
_IMGSZ = int(os.getenv("IMGSZ", "640"))
DEFAULTS = dict(imgsz=_IMGSZ, batch=_BATCH, epochs=150, freeze=10, optimizer="auto",
                lr0=0.01, amp=_AMP, device=0, workers=_WORKERS, seed=0, plots=True,
                verbose=False, exist_ok=True, project=RUNS)

# Korean text: horizontal flip mirrors Hangul -> fliplr=0 is the correct default.
AUG_NONE = dict(mosaic=0.0, close_mosaic=0, hsv_h=0, hsv_s=0, hsv_v=0, degrees=0,
                translate=0, scale=0, shear=0, fliplr=0.0, flipud=0.0,
                erasing=0.0, mixup=0.0, copy_paste=0.0)
AUG_HEAVY = dict(mosaic=1.0, mixup=0.15, copy_paste=0.1, degrees=5,
                 translate=0.1, scale=0.5, fliplr=0.0)
AUG_CARD = dict(mosaic=0.5, mixup=0.0, fliplr=0.0, hsv_h=0.01, hsv_s=0.3,
                hsv_v=0.3, translate=0.05, scale=0.3, erasing=0.2)

EXPERIMENTS = [
    dict(name="e01_baseline_n_freeze10", model="yolov8n.pt"),
    dict(name="e02_n_freeze0",           model="yolov8n.pt", freeze=0),
    dict(name="e03_n_freeze5",           model="yolov8n.pt", freeze=5),
    dict(name="e04_n_scratch",           model="yolov8n.yaml", freeze=0),
    dict(name="e05_aug_none",            model="yolov8n.pt", **AUG_NONE),
    dict(name="e06_aug_heavy",           model="yolov8n.pt", **AUG_HEAVY),
    dict(name="e07_aug_card",            model="yolov8n.pt", **AUG_CARD),
    dict(name="e08_fliplr_on_BAD",       model="yolov8n.pt", fliplr=0.5),
    dict(name="e09_imgsz512",            model="yolov8n.pt", imgsz=512),
    dict(name="e10_imgsz768",            model="yolov8n.pt", imgsz=768),
    dict(name="e11_adamw",               model="yolov8n.pt", optimizer="AdamW", lr0=0.001),
    dict(name="e12_lr_low",              model="yolov8n.pt", lr0=0.005),
    dict(name="e13_yolov8s_freeze10",    model="yolov8s.pt"),
    dict(name="e14_yolov8s_freeze0",     model="yolov8s.pt", freeze=0),
    dict(name="e15_long300_card",        model="yolov8n.pt", epochs=300, **AUG_CARD),
]
SEED_REPEATS = [
    ("e01_baseline_n_freeze10", dict(model="yolov8n.pt")),
    ("e07_aug_card",            dict(model="yolov8n.pt", **AUG_CARD)),
]


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}][+{int(time.time()-START)}s] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def git(*args):
    try:
        subprocess.run(["git", *args], cwd=ROOT, check=False,
                       capture_output=True, text=True, timeout=120)
    except Exception as e:
        log(f"git {' '.join(args)} failed: {e}")


def commit_push(message: str):
    git("add", "results", "ABLATION.md", "STATUS.md", "DEV_LOG.md", "DONE.md")
    git("-c", "user.email=4mins12@gmail.com", "-c", "user.name=iluv4",
        "commit", "-m", message)
    tok = os.environ.get("GH_TOKEN", "")
    if tok:
        try:
            subprocess.run(["git", "push", f"https://{tok}@github.com/iluv4/cardnews-cv.git",
                            "main:main"], cwd=ROOT, check=False,
                           capture_output=True, text=True, timeout=300)
        except Exception as e:
            log(f"push failed (token revoked?): {e}")


def parse_metrics(save_dir: str) -> dict:
    df = pd.read_csv(os.path.join(save_dir, "results.csv"))
    df.columns = [c.strip() for c in df.columns]
    key = "metrics/mAP50-95(B)"
    best = df.loc[df[key].idxmax()]
    g = lambda k: float(best.get(k, float("nan")))
    return dict(map50=g("metrics/mAP50(B)"), map=g(key),
                precision=g("metrics/precision(B)"), recall=g("metrics/recall(B)"),
                best_epoch=int(best.get("epoch", -1)), n_epochs=len(df))


def save_artifacts(name: str, save_dir: str):
    dst = os.path.join(RESULTS, name)
    os.makedirs(dst, exist_ok=True)
    for f in ("results.png", "results.csv", "confusion_matrix.png",
              "val_batch0_pred.jpg", "val_batch0_labels.jpg", "BoxPR_curve.png"):
        src = os.path.join(save_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, f))


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_outputs():
    if not ALL_ROWS:
        return
    pd.DataFrame(ALL_ROWS).to_csv(MASTER_CSV, index=False)
    cols = ["name", "model", "map50", "map", "precision", "recall",
            "best_epoch", "n_epochs", "minutes", "status"]
    md = ["# Small-data fine-tuning ablation — Korean card-news detector", "",
          "_109 images (train 93 / val 16), YOLOv8, amp=False, workers=0, RTX 2060._",
          f"_Updated +{int((time.time()-START)/60)} min into the run._", "",
          "| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
    for r in sorted(ALL_ROWS, key=lambda x: (x.get("map") if isinstance(x.get("map"), float) else -1), reverse=True):
        md.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    ok = [r for r in ALL_ROWS if isinstance(r.get("map"), float)]
    if ok:
        b = max(ok, key=lambda x: x["map"])
        md += ["", f"**Best so far:** `{b['name']}` — mAP50-95 = {b['map']:.4f}, mAP50 = {b['map50']:.4f}"]
    with open(os.path.join(ROOT, "ABLATION.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    try:
        ok.sort(key=lambda x: x["map"])
        if ok:
            plt.figure(figsize=(9, max(3, 0.4*len(ok))))
            plt.barh([r["name"] for r in ok], [r["map"] for r in ok], color="#3b7")
            plt.xlabel("mAP50-95"); plt.title("Small-data fine-tuning ablation")
            plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "ablation_chart.png"), dpi=120)
            plt.close()
    except Exception as e:
        log(f"chart failed: {e}")


def write_status(phase: str):
    ok = len([r for r in ALL_ROWS if r.get("status") == "ok"])
    fail = len([r for r in ALL_ROWS if str(r.get("status", "")).startswith("FAIL")])
    s = ["# Overnight run status", "",
         f"- elapsed: {(time.time()-START)/3600:.2f} h / budget 9.5 h",
         f"- phase: {phase}",
         f"- runs ok: {ok} | failed: {fail}",
         f"- last update: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    okrows = [r for r in ALL_ROWS if isinstance(r.get("map"), float)]
    if okrows:
        b = max(okrows, key=lambda x: x["map"])
        s.append(f"- best: {b['name']} mAP50-95={b['map']:.4f}")
    with open(os.path.join(ROOT, "STATUS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(s))


def train_subprocess(name: str, model: str, data: str, cfg: dict) -> str:
    """Run one training in an isolated process. Returns save_dir."""
    full = {**cfg, "name": name, "model": model, "data": data}
    cfgpath = os.path.join(FOLDS, f"_cfg_{name}.json")
    with open(cfgpath, "w", encoding="utf-8") as fh:
        json.dump(full, fh)
    env = dict(os.environ); env["PYTHONUTF8"] = "1"
    subprocess.run([PY, TRAIN_ONE, cfgpath], cwd=ROOT, env=env,
                   check=True, timeout=RUN_TIMEOUT)
    return os.path.join(RUNS, name)


def run_one(exp: dict, data: str = DATA):
    name = exp["name"]
    if time.time() - START > TIME_BUDGET:
        log(f"skip {name}: time budget reached"); return
    cfg = {**DEFAULTS, **{k: v for k, v in exp.items() if k not in ("name", "model")}}
    model = exp.get("model", "yolov8n.pt")
    t0 = time.time()
    row = dict(name=name, model=model, status="FAIL",
               minutes=round((time.time()-t0)/60, 1))
    for attempt in (1, 2):   # retry once: failures are usually transient OOM
        log(f"START {name}  model={model} (attempt {attempt})")
        try:
            save_dir = train_subprocess(name, model, data, cfg)
            met = parse_metrics(save_dir)
            save_artifacts(name, save_dir)
            row = dict(name=name, model=model, status="ok",
                       minutes=round((time.time()-t0)/60, 1), **met)
            log(f"DONE  {name}  mAP50-95={met['map']:.4f} mAP50={met['map50']:.4f} ({row['minutes']}m)")
            break
        except subprocess.TimeoutExpired:
            row["status"] = "FAIL_timeout"; log(f"FAIL  {name}: timeout"); break
        except Exception as e:
            row = dict(name=name, model=model, status="FAIL",
                       minutes=round((time.time()-t0)/60, 1))
            log(f"attempt {attempt} failed for {name}: {e}")
            if attempt == 1:
                time.sleep(45)   # let system memory free, then retry
    ALL_ROWS.append(row)
    write_outputs(); write_status(f"ran {name}")
    commit_push(f"overnight: {name} ({row.get('status')})")
    time.sleep(10)   # pause between runs so the prior process fully releases memory


def make_kfold(k=5, seed=0):
    imgs = sorted(glob.glob(os.path.join(ROOT, "dataset", "images", "train", "*.jpg")) +
                  glob.glob(os.path.join(ROOT, "dataset", "images", "val", "*.jpg")))
    random.Random(seed).shuffle(imgs)
    folds = [imgs[i::k] for i in range(k)]
    yamls = []
    for i in range(k):
        val = folds[i]; train = [p for j in range(k) if j != i for p in folds[j]]
        tr = os.path.join(FOLDS, f"fold{i}_train.txt"); va = os.path.join(FOLDS, f"fold{i}_val.txt")
        open(tr, "w", encoding="utf-8").write("\n".join(train))
        open(va, "w", encoding="utf-8").write("\n".join(val))
        yp = os.path.join(FOLDS, f"fold{i}.yaml")
        open(yp, "w", encoding="utf-8").write(
            f"train: {tr}\nval: {va}\nnames:\n  0: title\n  1: body\n  2: logo\n  3: underlay\n")
        yamls.append(yp)
    return yamls


def run_kfold():
    log("=== K-fold CV on baseline config ===")
    maps = []
    for i, yp in enumerate(make_kfold(5)):
        if time.time() - START > TIME_BUDGET:
            break
        name = f"kfold_{i}"; t0 = time.time()
        try:
            save_dir = train_subprocess(name, "yolov8n.pt", yp, {**DEFAULTS, "epochs": 120})
            met = parse_metrics(save_dir)
            maps.append(met["map"])
            ALL_ROWS.append(dict(name=name, model="yolov8n.pt", status="ok",
                                 minutes=round((time.time()-t0)/60, 1), **met))
            log(f"DONE  {name} mAP50-95={met['map']:.4f}")
        except Exception as e:
            ALL_ROWS.append(dict(name=name, model="yolov8n.pt", status="FAIL",
                                 minutes=round((time.time()-t0)/60, 1)))
            log(f"FAIL  {name}: {e}")
        write_outputs(); write_status(f"kfold {i}")
        commit_push(f"overnight: kfold {i}")
    if maps:
        import statistics as st
        mean, sd = st.mean(maps), (st.pstdev(maps) if len(maps) > 1 else 0.0)
        log(f"K-fold mAP50-95 = {mean:.4f} +/- {sd:.4f}")
        with open(os.path.join(RESULTS, "kfold_summary.txt"), "w") as fh:
            fh.write(f"5-fold mAP50-95: mean={mean:.4f} std={sd:.4f}\nfolds={maps}\n")


def write_dev_log():
    ok = [r for r in ALL_ROWS if r.get("status") == "ok"]
    okrows = [r for r in ALL_ROWS if isinstance(r.get("map"), float)]
    best = max(okrows, key=lambda x: x["map"]) if okrows else {}
    txt = f"""# 개발 일지 — Stage 2 Detector (자동 야간 실행)

## 오늘 자동으로 한 것
- YOLOv8 소규모 파인튜닝 **최적화 ablation** {len(ok)}개 설정 학습 (109장, amp=False, workers=0, RTX 2060)
- 비교 축: transfer/freeze, 모델 크기(n/s), augmentation(없음/heavy/카드튜닝), 좌우반전(한글 깨짐 → fliplr=0 검증), 이미지 크기, optimizer/LR, 시드 반복, 5-fold CV
- 결과: `results/ablation.csv`, `ABLATION.md`, 차트 `results/ablation_chart.png`

## 핵심 결과
- **Best: `{best.get('name','?')}` — mAP50-95 = {best.get('map','?')}**
- 전체 표는 ABLATION.md 참고

## 다음 (수동 작업 필요)
1. Label Studio로 4클래스(title/body/logo/underlay) 수동 라벨링 (현재 pseudo-label = title/body)
2. 수동 라벨로 재학습 → 클라우드(RunPod A100) 본 학습
3. detector 출력(box/cls) → PosterLayout annotation으로 연결
"""
    with open(os.path.join(ROOT, "DEV_LOG.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)


def fix_dataset_yaml():
    """Rewrite dataset.yaml with an absolute path for THIS machine so it is
    portable across Windows / Linux / cloud (the committed copy may differ)."""
    p = os.path.join(ROOT, "dataset").replace("\\", "/")
    with open(DATA, "w", encoding="utf-8") as fh:
        fh.write(f"path: {p}\ntrain: images/train\nval: images/val\n"
                 "names:\n  0: title\n  1: body\n  2: logo\n  3: underlay\n")


def main():
    fix_dataset_yaml()
    log(f"=== overnight run start (budget {TIME_BUDGET/3600:.1f}h, isolated subprocesses) ===")
    for exp in EXPERIMENTS:
        run_one(exp)
    for base, kw in SEED_REPEATS:
        for seed in (1, 2):
            if time.time() - START > TIME_BUDGET:
                break
            run_one(dict(name=f"{base}_seed{seed}", seed=seed, **kw))
    if time.time() - START < TIME_BUDGET:
        run_kfold()
    write_dev_log(); write_outputs(); write_status("DONE")
    try:
        okrows = [r for r in ALL_ROWS if isinstance(r.get("map"), float)]
        if okrows:
            best = max(okrows, key=lambda x: x["map"])
            src = os.path.join(RUNS, best["name"], "weights", "best.pt")
            if os.path.exists(src):
                os.makedirs(os.path.join(RESULTS, "best_model"), exist_ok=True)
                shutil.copy2(src, os.path.join(RESULTS, "best_model", "best.pt"))
                open(os.path.join(RESULTS, "best_model", "which.txt"), "w").write(
                    f"{best['name']}  mAP50-95={best['map']}\n")
    except Exception as e:
        log(f"best copy failed: {e}")
    open(os.path.join(ROOT, "DONE.md"), "w").write(
        f"finished {time.strftime('%Y-%m-%d %H:%M:%S')}, "
        f"{len([r for r in ALL_ROWS if r.get('status')=='ok'])} runs ok\n")
    commit_push("overnight: COMPLETE")
    log(f"=== overnight COMPLETE: {len(ALL_ROWS)} runs ===")


if __name__ == "__main__":
    main()
