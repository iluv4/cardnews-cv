# paper/ — report & manuscript

Two ready-to-read PDFs are generated locally (no LaTeX needed):

| File | What it is | Rebuild |
|------|------------|---------|
| `cardnews_paper.pdf` | English CVPR-style preview of the manuscript (figures + ablation table + references) | `py -3 paper/build_pdf.py` |
| `cardnews_report_ko.pdf` | Korean presentation report | `py -3 paper/build_report_ko.py` |

These are **previews** built with reportlab. The actual conference submission
artifact is the LaTeX source below.

## Submitting to CVPR (or another venue) via Overleaf
1. Create a new Overleaf project and upload: `main.tex`, `refs.bib`, and the
   whole `figures/` folder.
2. Set the compiler to **pdfLaTeX** (Menu → Compiler). The bibliography uses
   `natbib` + `ieeetr` and resolves automatically from `refs.bib`.
3. Compile. `main.tex` is written to be **self-contained** (standard packages
   only), so it also compiles standalone without any extra style file.
4. For an official CVPR submission, drop the body (everything from `\section`
   onward) into the official **CVPR template** (`cvpr.sty`) and keep `refs.bib`
   and `figures/` as-is.

## Figures
`figures/` is produced by `py -3 paper/make_figures.py` from the real
experiment outputs and the new dataset (dataset samples, detector ablation bar,
pipeline diagram, validation predictions, deck time-series, generation results).

## Numbers in the paper are real
Detector results come from `results/ablation.csv` / `results/kfold_summary.txt`
(best mAP@50-95 = 0.718; 5-fold = 0.611 ± 0.049). The 109-vs-687 data-scale
comparison is presented as a leak-free *protocol*; its cloud run is pending and
is labeled as such rather than reported as a finished result.
