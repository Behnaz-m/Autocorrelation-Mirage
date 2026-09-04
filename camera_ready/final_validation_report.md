# Final validation report — 2026-09-03

## Completed checks

- `venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`: **5/5 passed**.
- Smoke test: `venv/bin/python experiments/run_highsignal_benchmark.py --n_replicates 2 --start_seed 9000 --output_dir /private/tmp/camera_ready_highsignal_smoke`: completed. It produced grouped AUROC 0.653, row-wise AUROC 0.870, and a wide two-run interval as expected for a smoke test.
- Result integrity: main/high-signal files contain 30 paired runs; robustness has 810 unique rows (81 cells × 10); drift has 270 unique rows (9 cells × 30).
- `latexmk -pdf -halt-on-error -interaction=nonstopmode main2_iberamia.tex`: **success**, 16 pages. BibTeX completed; no undefined citation or reference warning remained on the final pass.
- `git diff --check`: no whitespace errors.

## Warnings and blockers

- The LaTeX log contains nonfatal overfull/underfull boxes, including a 12.46pt box near the robustness table. These need visual remediation during substantive shortening.
- `pdfinfo`/`pdffonts` are unavailable, so PDF metadata and embedded-font checks were not performed.
- The compiled paper is 16 pages, above the stated 12-page maximum.
- The author block is intentionally blank because authoritative author metadata is absent. A camera-ready PDF and clean source archive were therefore **not** created, and clean-archive compilation is not applicable yet.

## Required final gate

Before upload: provide the final author metadata, reduce or obtain approval for the page count, repair relevant layout warnings, build a clean source archive, compile that archive, and compare its PDF with the intended final PDF.
