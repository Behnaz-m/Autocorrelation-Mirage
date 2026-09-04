# Final validation report — 2026-09-03

- Unit tests: 5/5 pass.
- Main and stronger-signal benchmarks: 30 paired replicates each; robustness: 81 cells × 10; drift: 9 cells × 30; ablation: 4 cells × 30 paired replicates.
- `latexmk -pdf -halt-on-error -interaction=nonstopmode main2_iberamia.tex` succeeds at 12 pages with resolved citations and references.
- The clean source ZIP was extracted to `/private/tmp/paper76-build` and compiled successfully at 12 pages.
- Figure 2 was inspected at 300 dpi in `camera_ready/figure2_print_check.png`; labels, markers, and non-colour line/marker distinctions are legible at embedded width.
- Final PDF SHA-256: `2ecc1366d4513e520cec46cf26d385f018b83918eb4c3604c3f9e208160fde59`.
- Final ZIP SHA-256: `4951ccd73562bf489f4289227cabd85b7fea27cb6fd3bde0f06c89b239a4f1ce`.

`pdfinfo` and `pdffonts` are unavailable in this environment, so those two optional PDF checks were not run. The source uses the existing raster Figure 2; its page-level print inspection passed.
