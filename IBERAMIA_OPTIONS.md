# IBERAMIA Options

## Backup titles

1. Avoiding Episode Memorization: Dependence-Aware Evaluation for Rare-Event Panel Forecasting
2. Dependence-Aware Evaluation for Rare-Event Panel Forecasting
3. The Autocorrelation Mirage: Grouped Validation for Rare-Event Panel Forecasting

## Backup abstract variant

Rare-event forecasting from panel data often expands episodes into daily observations to overcome small-sample limitations. This expansion also creates a dependence-aware evaluation problem: row-wise cross-validation can place observations from the same episode in both training and test sets, allowing a model to exploit episode-specific autocorrelation rather than learn signal that transfers to unseen episodes. We call this mechanism episode memorization. This paper distinguishes episode memorization from explicit feature leakage and preprocessing leakage, and studies its impact in a controlled simulation with strictly causal features and an XGBoost classifier. Row-wise K-Fold cross-validation inflates mean AUC from 0.56 to 0.86, whereas an explicit time-to-event leak reaches 0.77 under grouped evaluation. We therefore propose $\DeltaCV$, the gap between row-wise and grouped estimates, as a practical warning signal rather than a universal threshold, and we give a dependence-aware protocol based on grouped splitting and fold-local preprocessing. The contribution is methodological: a compact taxonomy, a mechanistic explanation, and a simple protocol for more trustworthy rare-event panel forecasting.
