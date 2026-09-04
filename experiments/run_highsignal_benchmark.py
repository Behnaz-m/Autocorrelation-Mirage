#!/usr/bin/env python
"""
Higher-signal robustness check for the main benchmark.

The main benchmark's grouped-CV baseline is near chance (AUC ~0.55), which
leaves open whether the row-wise gap reflects a genuine evaluation artifact
or simply the fact that grouped CV has little real signal to lose. This
script raises the episode-effect standard deviation (the transferable,
between-episode signal observed through feature X_5) rather than the
AR(1)-to-hazard coefficient (an episode-specific, non-transferable signal),
and reports paired grouped-vs-row-wise AUC under the stronger configuration.

Run with: python experiments/run_highsignal_benchmark.py [--n_replicates N]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import generate_panel_data, prepare_modeling_data, get_feature_columns
from src.evaluation import (
    evaluate_grouped_cv,
    evaluate_random_cv,
    compute_pooled_oof_metrics,
    compute_effective_sample_size,
)


def paired_t_interval(values: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    se = clean.std(ddof=1) / np.sqrt(len(clean))
    t_crit = stats.t.ppf(1 - alpha / 2, df=len(clean) - 1)
    mean = clean.mean()
    return mean - t_crit * se, mean + t_crit * se


def run_highsignal_benchmark(
    n_replicates: int = 30,
    start_seed: int = 42,
    alpha_std: float = 1.5,
    output_dir: str = "results/protocol_main_30",
) -> pd.DataFrame:
    rows = []
    for i in range(n_replicates):
        seed = start_seed + i
        df = generate_panel_data(
            n_episodes=30, T_max=60, ar_coef=0.7, noise_std=0.3,
            hazard_coef=0.15, base_hazard=-3.0, alpha_std=alpha_std,
            horizon=14, seed=seed,
        )
        feature_cols = get_feature_columns(df)
        X, y, groups = prepare_modeling_data(df, feature_cols)
        grouped_res = evaluate_grouped_cv(X, y, groups, n_splits=5)
        row_res = evaluate_random_cv(X, y, n_splits=5, seed=seed, groups=groups)
        grouped_metrics = compute_pooled_oof_metrics(grouped_res)
        row_metrics = compute_pooled_oof_metrics(row_res)
        n, m, rho, n_eff = compute_effective_sample_size(groups, y)
        rows.append({
            "seed": seed,
            "auc_grouped": grouped_metrics["auc"],
            "brier_grouped": grouped_metrics["brier"],
            "auc_row": row_metrics["auc"],
            "brier_row": row_metrics["brier"],
            "event_rate": y.mean(),
            "n_obs": len(y),
            "n_eff": n_eff,
            "rho": rho,
        })

    df_res = pd.DataFrame(rows)
    df_res["delta_cv"] = df_res["auc_row"] - df_res["auc_grouped"]

    os.makedirs(output_dir, exist_ok=True)
    df_res.to_csv(f"{output_dir}/highsignal_benchmark_latest.csv", index=False)
    return df_res


def main() -> None:
    parser = argparse.ArgumentParser(description="Higher-signal robustness check")
    parser.add_argument("--n_replicates", type=int, default=30)
    parser.add_argument("--start_seed", type=int, default=42)
    parser.add_argument("--alpha_std", type=float, default=1.5)
    parser.add_argument("--output_dir", type=str, default="results/protocol_main_30")
    args = parser.parse_args()

    df_res = run_highsignal_benchmark(
        n_replicates=args.n_replicates,
        start_seed=args.start_seed,
        alpha_std=args.alpha_std,
        output_dir=args.output_dir,
    )

    lo, hi = paired_t_interval(df_res["delta_cv"])
    print(f"Grouped AUC: {df_res['auc_grouped'].mean():.3f} +/- {df_res['auc_grouped'].std():.3f}")
    print(f"Row-wise AUC: {df_res['auc_row'].mean():.3f} +/- {df_res['auc_row'].std():.3f}")
    print(f"Delta_CV: {df_res['delta_cv'].mean():.3f}, paired 95% CI: [{lo:.3f}, {hi:.3f}]")
    print(f"Event rate: {df_res['event_rate'].mean():.3f}")


if __name__ == "__main__":
    main()
