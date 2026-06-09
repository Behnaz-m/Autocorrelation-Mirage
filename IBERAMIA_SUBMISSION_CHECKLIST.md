# IBERAMIA Submission Checklist

## Files to use

- Upload PDF: `main2_iberamia.pdf`
- Source file for final edits: `main2_iberamia.tex`

## Required manual confirmation before submission

- Confirm that `Paper ID 73` in `main2_iberamia.tex` matches the actual METEOR tracking number assigned by the submission system.

## Double-blind checks completed here

- Author names and affiliations were removed from the LNCS submission file.
- The conference version does not mention GitHub, Overleaf, or a public repository.
- A raw string scan of `main2_iberamia.pdf` did not reveal obvious author or institution names.

## Remaining manual double-blind checks

- If there is a public preprint, public repo, or project page with the same distinctive title, decide whether that is acceptable under the conference's double-blind policy.
- If the submission site asks for supplementary material, avoid uploading files whose names or metadata identify the authors.
- If you regenerate the PDF elsewhere, recheck that the author field in the exported PDF metadata does not contain author names.

## Upload hygiene

- Upload only the conference PDF unless the site explicitly requests source files.
- Do not upload `main2.tex`, the ACM-style draft.
- Do not upload auxiliary LaTeX files such as `.aux`, `.bbl`, `.blg`, `.log`, or `.out`.
- Do not upload `AILET-2026-0020_Proof_hi (2).pdf`.

## Current status

- `main2_iberamia.pdf` compiles successfully in LNCS format.
- Current length: 12 pages.
- The document is under the 12-page IBERAMIA limit.

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
