# Camera-ready change report — Paper 76

## Provenance

- Working branch: `iberamia2026-paper76-camera-ready`.
- Branch-point commit: `75ed986` (`main` / `origin/main` at start).
- Overleaf-labelled remote-tracking commit inspected: `origin/overleaf-2026-06-08-1435` / `dc72938`. No `git.overleaf.com` remote is configured; this branch is older than the local revision history.
- No new Git commit or push was made. The tree began with substantial tracked and untracked work; it was preserved in a dated patch and manifest before the branch was created.

## Changes recorded

- Updated the obsolete visible Paper 73 reference in the internal response to Paper 76 and removed Paper 73 from the LNCS manuscript.
- Reduced the abstract to approximately 165 words using values reproducible from the saved CSVs.
- Added audit, reviewer-action, reproducibility, internal-response, and validation records in `camera_ready/`.
- Marked the production package as intentionally blocked instead of generating a deceptively labelled archive.

## Experimental evidence

- Main benchmark: 30 paired replicates; grouped AUROC 0.552, row-wise 0.835, gap 0.283 [0.246, 0.320].
- Higher signal: 30 paired replicates; grouped 0.723, row-wise 0.922, gap 0.199 [0.170, 0.229].
- Robustness: 81 cells × 10 replicates.
- Normalization drift: 9 model/condition cells × 30 replicates.

## Optional alternative title

“Episode Memorization in Panel-Expanded Event Forecasting: Aligning Cross-Validation with the Deployment Unit.” It foregrounds the demonstrated mechanism and deployment mismatch while retaining the accepted title in the manuscript.

## Final state

Final author metadata is inserted, the source compiles to 12 pages, and the clean archive was compiled successfully. The mechanism ablation adds 30 paired replicates to each of four cells; its results are in `results/camera_ready/mechanism_ablation.csv`.
