# Reviewer action matrix — Paper 76

| Reviewer request | Action/evidence | Status |
|---|---|---|
| R1: concrete motivation, deployment mismatch, roadmap | Introduction uses hospital, maintenance, and conflict examples; explains unseen-episode target and ends with roadmap. | Complete |
| R1: related work | Section 2 covers leakage, grouped/blocked and subject-wise CV, pseudoreplication, and panel forecasting. | Complete |
| R1: connect formalism, conclusion, shorter abstract | Practical implications occur beside definitions; conclusions added; abstract reduced to 165 words. | Complete |
| R2: nontrivial signal | 30 paired higher-signal runs: grouped 0.723, row-wise 0.922, gap 0.199 [0.170, 0.229]; `results/protocol_main_30/highsignal_benchmark_latest.csv`. | Complete |
| R2: framing/mechanism isolation | Abstract/intro state heterogeneity and shared event-time structure; Section 6 reports nonzero gap at rho=0. A fully factorial on/off ablation is not present. | Partially complete |
| R2: at least 10 robustness replicates | 81 cells × 10 replicates, three model families; `results/strengthening_pooled/robustness_grid_latest.csv`. | Complete |
| R2: demote Theorem 1 | Converted to Remark 1; Proposition retained. | Complete |
| R2: second DGP/normalization | Section 6 specifies drift DGP and reports logistic, RF, and boosted trees, 30 paired replicates; `results/strengthening_pooled/drift_experiment_latest.csv`. | Complete |
| R2: real-data limitation | Abstract and Discussion explicitly state simulation-only evidence. No suitable dataset is in repository. | Complete |
| R2 minor: temporal result | Discussion reports temporal AUROC 0.917 and why it does not target new episodes. | Complete |
| R2 minor: audit target, precision, Figure 2 | Checklist names deployment target; headline values use three decimals; companion figure is regenerated. Printed-size visual inspection remains required. | Partially complete |
| Camera-ready identity and 12-page limit | Author data absent; source is 16 pages. | Blocked |
