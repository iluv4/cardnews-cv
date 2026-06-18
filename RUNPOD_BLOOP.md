# RunPod runbook — B-loop prototype (the hybrid product gate)

Runs `gen/b_loop_demo.py`: reflib search → layout lookup → GPT Image 2 render →
detector QA. This is the **go/no-go gate** for the hybrid product in
`BUSINESS_CASE.md`. The detector QA step (5) needs torch → that's why it's on a pod.

What each step needs:
- (1)(2)(3) search + layout + prompt — **no GPU**, runs anywhere (`--dry` stops here).
- (4) GPT Image 2 render — **`OPENAI_API_KEY` + network** (no GPU).
- (5) detector QA — **torch + the trained detector** (`results/best_model/best.pt`).

## 1. Create the pod
- RunPod → Pods → Deploy → **RTX 4090 (24 GB)**, official **PyTorch** template,
  container disk ~20 GB. This is inference-only (one image at a time) so even a
  cheaper GPU works; 4090 is just the known-good default. Connect → Web Terminal.

## 2. Clone + install
```bash
git clone https://<YOUR_GH_TOKEN>@github.com/iluv4/cardnews-cv.git
cd cardnews-cv
git checkout claude/hungry-payne-615cdd      # this branch (until it's merged to main)
pip install -q -U ultralytics openai pillow numpy
```
(`results/best_model/best.pt`, `service/library/templates.json`, and
`reflib/data/index.json` are committed — no extra data download needed.)

## 3. Set the key + run the full loop
```bash
export OPENAI_API_KEY=sk-...
py -3 gen/b_loop_demo.py \
  --text "스마트팜 지원사업 안내" \
  --title "스마트팜으로\n바뀌는 농업" \
  --body "복잡한 지원 절차를 한 장에 정리했습니다." \
  --out gen/out_bloop --size 1024x1536
```
Output in `gen/out_bloop/`:
- `brief.json` — chosen reference + the GPT prompt + the layout it copied
- `render.png` — GPT Image 2 output
- `qa.json` — detector's verdict (did the output's title/body block count match intent?)

The script prints **PASS** or **FAIL** at the end. FAIL is the *interesting* case —
it means GPT didn't follow the layout, which is exactly what the QA/control layer
is for (feed it back, re-prompt, or fall back to the vector overlay).

## 4. What to actually evaluate (this is the gate, not just "does it run")
Look at `render.png` and ask:
1. **Korean text** — crisp and correctly spelled? (Confirms the v1 moat is really gone.)
2. **Layout fidelity** — did GPT respect the reference's structure, or freestyle?
3. **Brand control** — could a brand team accept this as-is, or does it need the
   vector overlay (step 5 in BUSINESS_CASE.md) for exact logo/color/font?

If (1) is great but (2)/(3) are loose → the product is the **control + QA + overlay**
layer, not the pixels. That confirms direction B.

## 5. Get results back + stop
```bash
tar czf bloop.tgz gen/out_bloop && echo "download bloop.tgz via Jupyter file browser"
```
Then RunPod → Pods → **Stop/Terminate** (stop paying).

## Known limitations of the current scaffold (fix as you go)
- **Search relevance is weak** (lexical only; `score=0.0` is common). The full
  corpus needs `gen/extract_templates_detector.py` (detector-labels every image →
  more templates) and CLIP embeddings (`reflib/embed_clip.py`) for real semantic
  search. Until then it falls back to the first template-having index record.
- **Prompt builder (3) is coarse** — coarse zones (top/middle/bottom × l/c/r), one
  title + one body block. Improve once you see where GPT actually drifts.
- **QA (4) is count-only** — checks block *counts*, not positions/overlap. Add
  IoU-vs-intended-boxes once the basic loop proves out.
