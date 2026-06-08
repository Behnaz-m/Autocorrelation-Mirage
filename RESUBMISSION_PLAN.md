# Resubmission Plan for IBERAMIA 2026

## Current status

- The current workspace does not contain the manuscript source file.
- The attached review text contains the editor summary and Reviewer 1 comments, but Reviewer 2's detailed attachment is missing.
- The official IBERAMIA 2026 CFP page states that the paper submission deadline was postponed to June 10, 2026.
- The CFP PDF still shows the older date of May 31, 2026, so use the conference website as the latest source.
- Other current CFP dates are:
  - notification: July 15, 2026
  - camera ready: July 28, 2026
  - conference dates: November 18-20, 2026
- As of June 7, 2026, there are three days left before the extended deadline.

## Main risks identified from the reviews

1. The paper may be perceived as hard to follow.
2. Key terms are not defined early and concretely enough.
3. The paper may overstate the prevalence of the problem without showing real-world evidence.
4. The paper may blur together different failure modes:
   - explicit future-information leakage
   - normalization leakage
   - pseudoreplication / episode overlap across train and test
5. Some language may make the argument sound like a strawman, especially if explicit leakage is framed as an unrealistic baseline.
6. The model specification and the basis for the `ΔCV` threshold need to be stated explicitly in the paper.

## Revision strategy

### 1. Reframe the paper's main claim

Avoid a headline claim like "pseudoreplication dominates label leakage" unless the revised paper can defend it across more than one controlled simulation setting.

Safer framing:

- pseudoreplication can create severe optimism even when features are strictly causal
- random cross-validation on expanded panel data can overestimate predictive performance
- grouped evaluation is necessary for valid assessment in episode-based panel forecasting
- `ΔCV` is a useful empirical warning signal, not a universal theorem

### 2. Define the three failure modes early

Add a subsection near the end of the introduction or at the start of methods:

- `Explicit leakage`: a feature directly or indirectly encodes future event timing.
- `Normalization leakage`: preprocessing uses information from observations that should be unavailable at prediction time.
- `Pseudoreplication`: multiple rows from the same episode appear in both training and test sets, allowing the model to recognize episode identity or episode-specific trajectory structure.

The paper should say clearly that pseudoreplication is not the same thing as feature leakage, even though both can inflate evaluation.

### 3. Add a toy worked example

Include a very small example with 2-3 episodes and 2 folds that shows:

- how expanding an episode into multiple rows creates dependence
- how random `KFold` places the same episode in both train and test
- why grouped CV blocks this shortcut

This is likely the single best way to answer the "difficult to follow" criticism.

### 4. Add one summary figure or table

Recommended content:

- rows: explicit leakage, normalization leakage, pseudoreplication
- columns: what causes it, what information leaks, whether future labels are needed, typical symptom, correct fix

This directly addresses Reviewer 1's request for a contrastive summary.

### 5. State the model and protocol explicitly

The paper should specify in the methods section:

- model: `XGBClassifier`
- core hyperparameters
- number of folds / leave-one-group-out choice
- what normalization is done in each condition
- how AUC and Brier are aggregated
- how many simulation replicates are used

Right now this is present in code but not guaranteed to be visible to readers.

### 6. Soften or justify the `ΔCV > 0.05` rule

At present, the repository states that a gap greater than `0.05` indicates episode memorization, but no derivation is visible in the codebase.

Safer options:

- present `0.05` as a provisional heuristic used in this study
- replace the fixed threshold with a null-distribution calibration from repeated grouped-vs-random comparisons
- report `ΔCV` with bootstrap uncertainty rather than declaring a universal cutoff

If space is tight, the cleanest wording is:

"In this study, large positive `ΔCV` values served as a warning signal of evaluation optimism; we do not claim a universal decision threshold."

### 7. Reduce any strawman cues

Avoid wording that suggests reviewers are expected to accept obviously incorrect pipelines as common practice.

In particular, be careful with claims like:

- "the standard tool is more dangerous than the worst feature-engineering error"
- "explicit leakage is a straw man"
- "every practitioner uses this"

Instead:

- emphasize that random row-wise CV is a default tool that becomes inappropriate after episode expansion
- present explicit leakage only as a reference condition, not as an unrealistic opponent
- separate "common by software default" from "common in competent practice"

### 8. Strengthen external validity

The editor's summary suggests the biggest scientific weakness is not the simulation itself but the lack of evidence that the problem matters in practice.

Best option:

- add one real-data re-evaluation on a public longitudinal or episode-based dataset
- compare random row-wise CV vs grouped-by-episode CV
- show how the headline metric changes

If a real dataset cannot be added in time:

- narrow the claims
- present the paper as a methodological cautionary note supported by simulation
- explicitly state that prevalence in published practice remains an open empirical question

### 9. Reposition for IBERAMIA

A strong IBERAMIA version would read as a methodological paper on trustworthy evaluation in AI, not only as a critique.

Suggested angle:

- trustworthy evaluation for rare-event panel forecasting
- dependence-aware validation in longitudinal AI systems
- practical diagnostics for overoptimistic evaluation under repeated observations

## Comment-to-action mapping

### Associate Editor

Concern: difficult to follow, missing concrete examples, key terms undefined.

Action:

- add definitions subsection
- add toy episode-splitting example
- add one contrastive summary table or figure

Concern: does not show the problem is prevalent in practice.

Action:

- add one real-data demonstration if at all possible
- otherwise narrow claims and acknowledge scope limits

Concern: does not clearly distinguish pseudoreplication from other leakage types.

Action:

- separate definitions, mechanisms, symptoms, and fixes

Concern: strawman argument.

Action:

- remove adversarial wording
- present explicit leakage only as a comparison condition

### Reviewer 1

Concern: real-world datasets would better illustrate the findings.

Action:

- add public dataset case study if feasible

Concern: model used for evaluation is not explicitly stated.

Action:

- specify `XGBClassifier` and hyperparameters in methods

Concern: origin of `ΔCV = 0.05` threshold is unclear.

Action:

- either justify empirically or downgrade to a heuristic

Concern: paper should contrast explicit leakage, normalization leakage, and pseudoreplication.

Action:

- add summary table or figure

## Suggested revised contribution statement

Possible replacement language for the introduction:

"We study evaluation bias in rare-event panel forecasting after episode expansion. Even when features are strictly causal, row-wise cross-validation can split observations from the same episode across training and test folds, producing overly optimistic discrimination estimates. We clarify how this mechanism differs from explicit feature leakage and preprocessing leakage, quantify its effect in controlled simulations, and propose grouped validation together with a simple cross-protocol discrepancy diagnostic."

## Priority order

1. Obtain the manuscript source and Reviewer 2 attachment.
2. Rewrite title, abstract, introduction, and contributions to reduce overclaiming.
3. Add definitions plus a toy example.
4. Add a summary figure or table contrasting the three failure modes.
5. Make model/protocol details explicit.
6. Either justify or soften the `ΔCV` threshold.
7. Add a real-data demonstration if time permits.
8. Convert to Springer LNCS format and enforce the 12-page limit.

## What to do next

Once the manuscript file is available, revise these sections first:

- title
- abstract
- introduction
- methods / experimental design
- discussion / limitations

Then check whether the figures and tables can be compressed to fit the IBERAMIA page limit without losing the comparison table.
