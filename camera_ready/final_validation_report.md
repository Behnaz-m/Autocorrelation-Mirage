# Final validation report — 2026-09-03

- Unit tests: 5/5 pass.
- Main and stronger-signal benchmarks: 30 paired replicates each; robustness: 81 cells × 10; drift: 9 cells × 30; ablation: 4 cells × 30 paired replicates.
- `latexmk -pdf -halt-on-error -interaction=nonstopmode main2_iberamia.tex` succeeds at 12 pages with resolved citations and references.
- The clean source ZIP was extracted to `/private/tmp/paper76-build` and compiled successfully at 12 pages.
- Figure 2 was inspected at 300 dpi in `camera_ready/figure2_print_check.png`; labels, markers, and non-colour line/marker distinctions are legible at embedded width.
- Corrected mechanism isolation: fingerprint off/event time off -0.001 [-0.008, 0.006]; fingerprint off/event time on 0.038 [0.028, 0.048]; fingerprint on/event time off 0.006 [-0.001, 0.014]; both on 0.283 [0.246, 0.320].
- Figure 2 is `robustness_drift_companion.pdf` (vector PDF; TrueType embedding configured) and was re-rendered at 300 dpi.
- Final PDF SHA-256: `b8d475367cea699f59cb307cdfef25835022a0d2be3d405851e535f6bc5e5389`.
- Final ZIP SHA-256: `77e3a6dce00b8e9089566abf0ebedfff4ed88b77e0b49a8ee482691165a315e9`.

`pdfinfo` and `pdffonts` are unavailable in this environment, so those two optional PDF checks were not run. The source uses the existing raster Figure 2; its page-level print inspection passed.
