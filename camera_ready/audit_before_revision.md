# Camera-ready baseline audit — 2026-09-03

## Repository and synchronization

- Resolved repository: `/Users/moradibx/Library/CloudStorage/OneDrive-JamesMadisonUniversity/JMU/Research/Python/Autocorrelation Mirage/Autocorrelation-Mirage`.
- Baseline branch/commit: `main` / `75ed986`; it matched `origin/main` before the camera-ready branch was created.
- A tracked-work patch and untracked-file manifest were preserved beside the repository as `../Autocorrelation-Mirage-pre-camera-ready-20260903-1942.{patch,untracked.txt}`.
- The configured remote is `origin` (`https://github.com/Behnaz-m/Autocorrelation-Mirage.git`). No remote URL contains `git.overleaf.com`. After `git fetch --all --prune`, the only Overleaf-labelled remote-tracking branch was `origin/overleaf-2026-06-08-1435` at `dc72938`; it is an older anonymous manuscript and is an ancestor of the local revision history, not a safe camera-ready starting point.

## Source/build audit

- Primary LNCS source: `main2_iberamia.tex`; bibliography: `references.bib`; figures: `auc_comparison_bars.png` and `robustness_drift_companion.png`.
- Experiment entry points: `experiments/run_simulation.py`, `experiments/run_strengthening_experiments.py`, and `experiments/run_highsignal_benchmark.py`.
- Environment: `venv`, Python 3.12; dependencies declared in `requirements.txt`.
- Baseline compilation: `latexmk -pdf -halt-on-error -interaction=nonstopmode main2_iberamia.tex` produced `main2_iberamia.pdf`, 16 pages. BibTeX completed and the final LaTeX pass resolved citations/references, but reported several overfull/underfull boxes.
- Baseline was anonymous and visibly contained `Paper ID 73`; its author metadata was not camera-ready. The manuscript exceeded the stated 12-page maximum.

## Result audit

- Main benchmark: `results/protocol_main_30/main_benchmark.csv`, 30 paired seeds. Grouped AUROC 0.552; row-wise AUROC 0.835; mean paired gap 0.283, 95% t interval [0.246, 0.320]. Temporal AUROC 0.917.
- Higher-signal benchmark: `results/protocol_main_30/highsignal_benchmark_latest.csv`, 30 paired seeds. Grouped AUROC 0.723; row-wise 0.922; paired gap 0.199, 95% t interval [0.170, 0.229].
- Robustness grid: `results/strengthening_pooled/robustness_grid_latest.csv`, 810 rows: 81 cells × 10 replicates, no duplicate rows.
- Normalization/drift DGP: `results/strengthening_pooled/drift_experiment_latest.csv`, 270 rows: 9 model/condition cells × 30 replicates, no duplicate rows.

## Blocking findings

The project contains no authoritative final author order, affiliations, countries, email addresses, or corresponding-author designation. The requested exact metadata therefore cannot be inserted without invention. The current 16-page source also cannot be submitted under a 12-page limit. No final submission archive or upload-ready PDF was created while either condition remains unresolved.
