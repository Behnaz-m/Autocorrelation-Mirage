# Reproduction instructions

Run from the repository root with the existing virtual environment:

```sh
venv/bin/python -m unittest discover -s tests -p 'test_*.py'
venv/bin/python experiments/run_highsignal_benchmark.py --n_replicates 30 --start_seed 42 --alpha_std 1.5 --output_dir results/protocol_main_30
venv/bin/python experiments/run_strengthening_experiments.py --n_replicates 10 --drift_replicates 30 --output_dir results/strengthening_pooled
venv/bin/python experiments/run_mechanism_ablation.py --n-replicates 30 --start-seed 42
latexmk -pdf -halt-on-error -interaction=nonstopmode main2_iberamia.tex
```

The strengthening script writes its per-replicate CSVs and skips neither failures nor completed records; retain its dated outputs and inspect status columns before replacing a `*_latest.csv`. To run persistently on macOS, use `caffeinate -dimsu env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv/bin/python -u experiments/run_strengthening_experiments.py --n_replicates 10 --drift_replicates 30 --output_dir results/strengthening_pooled 2>&1 | tee camera_ready/strengthening-$(date +%Y%m%d-%H%M).log`.

Do not package the source until the authors supply authoritative metadata and the source is within the page limit. Then stage only `main2_iberamia.tex`, `references.bib`, `main2_iberamia.bbl`, required figures, and the LNCS class/style dependencies in a clean temporary directory; compile there with the final `latexmk` command.
