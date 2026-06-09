# The Autocorrelation Mirage

Code and manuscript sources for the paper:

`The Autocorrelation Mirage: Dependence-Aware Evaluation for Panel-Expanded Event Forecasting`

This repository studies three distinct sources of optimistic evaluation in panel-expanded forecasting:

- explicit future-information leakage,
- preprocessing leakage,
- pseudoreplication from row-wise cross-validation across dependent episode rows.

The current codebase also handles administrative censoring explicitly: censored no-event episodes contribute only rows whose full forecast horizon is observed.

## Repository Layout

- `main2_iberamia.tex`: current LNCS/IBERAMIA manuscript source
- `references.bib`: bibliography
- `src/data_generation.py`: synthetic panel generator and eligibility filtering
- `src/evaluation.py`: grouped CV, row-wise CV, temporal-blocked within-episode CV, pooled metrics, episode-weighted metrics, and `Delta_CV` uncertainty helpers
- `src/leakage_injection.py`: explicit-leak and preprocessing-leak transforms
- `src/plotting.py`: figure generation
- `experiments/run_simulation.py`: main benchmark
- `experiments/run_strengthening_experiments.py`: robustness grid and drift-DGP normalization experiment
- `scripts/run_benchmark.py`: convenience wrapper for smoke tests and paper-scale reruns
- `tests/test_censoring_and_grouping.py`: censoring and grouping regression tests
- `results/`: generated CSVs and figures

## Environment

Install dependencies into a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The main benchmark uses `HistGradientBoostingClassifier` consistently across environments. If `xgboost` is unavailable at runtime, robustness-grid requests for the `xgboost` model fall back to the same boosted-tree family.

## Quick Checks

Run the unit tests:

```bash
venv/bin/python -m unittest discover -s tests -v
```

Run a lightweight smoke test for the main benchmark:

```bash
MPLCONFIGDIR=/private/tmp/iberamia_mpl_smoke \
venv/bin/python experiments/run_simulation.py \
  --n_replicates 2 \
  --delta_bootstrap_reps 100 \
  --output_dir results/smoke_main
```

Run a lightweight smoke test for the strengthening experiments:

```bash
venv/bin/python experiments/run_strengthening_experiments.py \
  --n_replicates 1 \
  --output_dir results/smoke_strengthening
```

Or run both smoke tests through the wrapper:

```bash
venv/bin/python scripts/run_benchmark.py --smoke-test
```

## Reproducing Paper Results

Main benchmark used in the paper:

```bash
MPLCONFIGDIR=/private/tmp/iberamia_mpl_main \
venv/bin/python experiments/run_simulation.py \
  --n_replicates 30 \
  --delta_bootstrap_reps 500 \
  --skip_temporal_baseline \
  --output_dir results/protocol_main_30
```

Exploratory robustness grid and drift-DGP preprocessing experiment:

```bash
venv/bin/python experiments/run_strengthening_experiments.py \
  --n_replicates 2 \
  --episode_grid 20,50,100 \
  --ar_grid 0.0,0.6,0.9 \
  --feature_grid 5,20,100 \
  --model_grid logistic,random_forest,boosted_trees \
  --output_dir results/strengthening_pooled
```

Wrapper for both paper-scale runs:

```bash
venv/bin/python scripts/run_benchmark.py --n-reps-main 30 --n-reps-grid 2
```

The main benchmark writes:

- `results/.../main_benchmark.csv`
- `results/.../main_benchmark_latest.csv`
- `results/.../bootstrap_delta_cv.csv`
- `results/.../main_benchmark_summary_latest.csv`
- `results/.../bootstrap_delta_cv_latest.csv`
- `results/.../tables/table3.csv`
- `results/.../figures/auc_comparison_bars.png`

The strengthening run writes:

- `results/.../robustness_grid.csv`
- `results/.../robustness_summary.csv`
- `results/.../drift_experiment.csv`
- `results/.../drift_summary.csv`
- `results/.../robustness_grid_latest.csv`
- `results/.../robustness_summary_latest.csv`
- `results/.../drift_experiment_latest.csv`
- `results/.../drift_summary_latest.csv`

## Manuscript Build

Build the current paper with:

```bash
latexmk -pdf main2_iberamia.tex
```

If `latexmk` is unavailable, use:

```bash
pdflatex main2_iberamia.tex
bibtex main2_iberamia
pdflatex main2_iberamia.tex
pdflatex main2_iberamia.tex
```

## Notes on Metrics

The repository reports both pooled row-level metrics and episode-aware diagnostics:

- pooled AUROC and pooled Brier score,
- episode-weighted AUROC with row weights proportional to `1 / n_e`,
- episode-mean Brier score,
- `Delta_CV = AUC_row - AUC_group`,
- episode-bootstrap uncertainty for `Delta_CV` based on resampling episode IDs rather than rows.

The temporal-blocked within-episode splitter is included for the different deployment target of forecasting later rows for already observed episodes. The paper tables focus on grouped unseen-episode generalization.
