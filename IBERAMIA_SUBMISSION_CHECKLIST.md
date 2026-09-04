# IBERAMIA Submission Checklist

## Files to use

- Upload PDF: `main2_iberamia.pdf`
- Source file for final edits: `main2_iberamia.tex`

## Required camera-ready metadata before submission

- Paper ID is **76**. Do not display it in the manuscript unless Springer explicitly requires it.
- Replace the intentionally blank author block in `main2_iberamia.tex` with the authors' final, authoritative names, order, affiliations, countries, email addresses, and corresponding-author designation. This repository does not contain that authoritative metadata.

## Superseded double-blind checks

- The old anonymous submission removed author names and affiliations. That is not suitable for camera-ready upload.
- The conference version does not mention GitHub, Overleaf, or a public repository.
- A raw string scan of `main2_iberamia.pdf` did not reveal obvious author or institution names.

## Remaining manual double-blind checks

- Confirm the final Springer/IBERAMIA author and PDF-metadata requirements with the production instructions.

## Upload hygiene

- Upload only the conference PDF unless the site explicitly requests source files.
- Do not upload `main2.tex`, the ACM-style draft.
- Do not upload auxiliary LaTeX files such as `.aux`, `.bbl`, `.blg`, `.log`, or `.out`.
- Do not upload `AILET-2026-0020_Proof_hi (2).pdf`.

## Current status

- Paper was accepted with minor revisions (2026-08). Reviewer 1 asked for a related-work section, a separate conclusions section, tighter intro motivation, and a shorter abstract; Reviewer 2 asked for a higher-signal robustness condition, a 10-replicate robustness grid, fully specified second DGP, and several framing fixes. All were implemented; see git log for the revision commit.
- `main2_iberamia.pdf` compiles successfully in LNCS format.
- Current length: 16 pages. This exceeds the stated 12-page IBERAMIA limit and must be substantively shortened or explicitly approved by the organizers before upload.
- The drift-DGP normalization experiment (Section 6) was re-run at 10 replicates, then at 30 (matching the main benchmark) once the 10-replicate run showed the estimate was still unstable for two of three model families. Final numbers: +0.045 logistic (95% CI [-0.010, 0.100]), +0.063 random forest (95% CI [0.029, 0.096], clearly significant), +0.034 boosted trees (95% CI [-0.010, 0.077]) — small, consistent, mostly borderline. This does not match the original 2-replicate estimate reported in the accepted version (logistic regression: was 0.445→0.641), which the properly replicated run shows was driven by sampling noise. The manuscript reports the corrected, 30-replicate numbers and says so explicitly. Worth a final sanity check before submission.

## Strengthening priority

- Highest impact: add one compact robustness grid over episode count and dependence, even if the full reviewer-suggested grid is too large.
- Highest impact: add a second DGP with pre-event drift so normalization leakage has positive empirical evidence rather than only taxonomy-level discussion.
- High impact: add one short subsection on `\DeltaCV` specificity, explicitly noting false positives and false negatives.
- Moderate impact: add a reproducibility appendix, supplement, or pseudocode block that makes the simulation fully reconstructable from the paper.

## Suggested minimal scope if time is short

- Robustness grid:
  use `E in {20, 50, 100}` and `rho in {0, 0.6, 0.9}` first.
- Model comparison:
  at minimum compare logistic regression, random forest, and XGBoost.
- Normalization-leak DGP:
  add a simple pre-event drift term so feature means worsen as the event approaches.
- `\DeltaCV` specificity:
  discuss at least four alternative causes of a positive gap:
  distribution shift, fold imbalance, episode heterogeneity, and deployment-target mismatch.
- Reproducibility:
  report `T_max`, feature count `p`, censoring handling, event-time generation, positive-row prevalence, AUC aggregation, and treatment of folds with no positives.

## Suggested full scope if there is enough time

- Full simulation grid:
  `E in {20, 30, 50, 100}`, `rho in {0, 0.3, 0.6, 0.9}`, `p in {5, 20, 100}`.
- Full model suite:
  logistic regression, random forest, XGBoost, and one simple sequence model.
- Main empirical claim to support:
  the `\DeltaCV` gap increases with within-episode dependence and model capacity.

## Suggested paper edits if new experiments are added

- In the abstract:
  mention robustness across dependence levels and model classes.
- In methods:
  add a simulation-grid table and an exact metric-aggregation paragraph.
- In results:
  add one figure for `\DeltaCV` versus `rho` and one table for model comparisons.
- In discussion:
  add a paragraph on when a large positive `\DeltaCV` may not imply episode memorization.
