#!/usr/bin/env python
"""
Run compact robustness and normalization-leakage experiments for the paper.

This script adds two pieces of evidence:
1. A compact grid over number of episodes, dependence strength, feature count,
   and model family to quantify how DeltaCV changes.
2. A drift-based normalization experiment showing that episode-wise
   normalization can inflate grouped-CV performance when pre-event trends are
   present.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):
        return iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import (
    add_noise_features,
    generate_panel_data,
    get_feature_columns,
    prepare_modeling_data,
)
from src.evaluation import (
    compute_pooled_oof_metrics,
    compute_effective_sample_size,
    create_model,
    evaluate_grouped_cv,
    evaluate_random_cv,
    get_model_display_name,
)
from src.leakage_injection import (
    apply_episodewise_normalization,
    apply_global_normalization,
)


@dataclass(frozen=True)
class GridConfig:
    n_episodes: int
    ar_coef: float
    total_features: int
    model_name: str


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def evaluate_panel(
    df: pd.DataFrame,
    model_name: str,
    grouped_splits: int,
    random_splits: int,
    seed: int,
    feature_cols: list[str] | None = None,
) -> dict:
    """Evaluate one dataset with grouped and random CV."""
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X, y, groups = prepare_modeling_data(df, feature_cols)
    model = create_model(model_name, seed=seed)

    grouped_res = evaluate_grouped_cv(
        X,
        y,
        groups,
        model=model,
        normalize_per_fold=True,
        n_splits=grouped_splits,
    )
    random_res = evaluate_random_cv(
        X,
        y,
        n_splits=random_splits,
        model=model,
        normalize_before=True,
        seed=seed,
    )
    n, m, rho_hat, n_eff = compute_effective_sample_size(groups, y)

    grouped_metrics = compute_pooled_oof_metrics(grouped_res)
    random_metrics = compute_pooled_oof_metrics(random_res)
    grouped_auc = grouped_metrics["auc"]
    random_auc = random_metrics["auc"]

    return {
        "grouped_auc": grouped_auc,
        "grouped_brier": grouped_metrics["brier"],
        "random_auc": random_auc,
        "random_brier": random_metrics["brier"],
        "delta_cv": random_auc - grouped_auc,
        "event_rate": y.mean(),
        "n_obs": len(y),
        "n_eff": n_eff,
        "rho_hat": rho_hat,
        "avg_episode_size": m,
    }


def run_robustness_grid(
    n_replicates: int,
    episode_grid: Iterable[int],
    ar_grid: Iterable[float],
    feature_grid: Iterable[int],
    model_grid: Iterable[str],
    grouped_splits: int,
    random_splits: int,
) -> pd.DataFrame:
    """Run the compact robustness grid."""
    configs = [
        GridConfig(
            n_episodes=n_episodes,
            ar_coef=ar_coef,
            total_features=total_features,
            model_name=model_name,
        )
        for n_episodes in episode_grid
        for ar_coef in ar_grid
        for total_features in feature_grid
        for model_name in model_grid
    ]

    rows: list[dict] = []

    for config_idx, config in enumerate(configs):
        for replicate in tqdm(
            range(n_replicates),
            total=n_replicates,
            leave=False,
            desc=(
                f"Grid E={config.n_episodes}, rho={config.ar_coef}, "
                f"p={config.total_features}, model={config.model_name}"
            ),
        ):
            seed = 1000 * config_idx + replicate
            df = generate_panel_data(
                n_episodes=config.n_episodes,
                T_max=60,
                ar_coef=config.ar_coef,
                noise_std=0.3,
                hazard_coef=0.15,
                base_hazard=-3.0,
                alpha_std=0.5,
                horizon=14,
                seed=seed,
            )
            df = add_noise_features(df, total_features=config.total_features, seed=seed)
            metrics = evaluate_panel(
                df,
                model_name=config.model_name,
                grouped_splits=grouped_splits,
                random_splits=random_splits,
                seed=seed,
            )
            rows.append(
                {
                    "seed": seed,
                    "replicate": replicate,
                    "n_episodes": config.n_episodes,
                    "ar_coef": config.ar_coef,
                    "total_features": config.total_features,
                    "model_name": config.model_name,
                    "model_label": get_model_display_name(config.model_name),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def grouped_auc_only(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    grouped_splits: int,
    seed: int,
    normalize_per_fold: bool,
) -> dict:
    """Evaluate grouped-CV performance for one feature representation."""
    X, y, groups = prepare_modeling_data(df, feature_cols)
    model = create_model(model_name, seed=seed)
    grouped_res = evaluate_grouped_cv(
        X,
        y,
        groups,
        model=model,
        normalize_per_fold=normalize_per_fold,
        n_splits=grouped_splits,
    )
    pooled_metrics = compute_pooled_oof_metrics(grouped_res)
    return {
        "auc": pooled_metrics["auc"],
        "brier": pooled_metrics["brier"],
        "event_rate": y.mean(),
        "n_obs": len(y),
    }


def run_drift_normalization_experiment(
    n_replicates: int,
    model_grid: Iterable[str],
    grouped_splits: int,
    drift_strength: float,
) -> pd.DataFrame:
    """
    Evaluate normalization leakage on a DGP with monotone pre-event drift.
    """
    rows: list[dict] = []

    for model_idx, model_name in enumerate(model_grid):
        for replicate in tqdm(
            range(n_replicates),
            total=n_replicates,
            leave=False,
            desc=f"Drift DGP model={model_name}",
        ):
            seed = 90000 + 1000 * model_idx + replicate
            df = generate_panel_data(
                n_episodes=30,
                T_max=60,
                ar_coef=0.7,
                noise_std=0.3,
                hazard_coef=0.15,
                base_hazard=-3.0,
                alpha_std=0.5,
                horizon=14,
                drift_strength=drift_strength,
                seed=seed,
            )
            base_feature_cols = get_feature_columns(df)

            leak_free = grouped_auc_only(
                df,
                feature_cols=base_feature_cols,
                model_name=model_name,
                grouped_splits=grouped_splits,
                seed=seed,
                normalize_per_fold=True,
            )
            rows.append(
                {
                    "seed": seed,
                    "replicate": replicate,
                    "model_name": model_name,
                    "model_label": get_model_display_name(model_name),
                    "condition": "leak_free",
                    "drift_strength": drift_strength,
                    **leak_free,
                }
            )

            df_global, _ = apply_global_normalization(df, base_feature_cols)
            global_feature_cols = [f"{col}_norm" for col in base_feature_cols]
            global_norm = grouped_auc_only(
                df_global,
                feature_cols=global_feature_cols,
                model_name=model_name,
                grouped_splits=grouped_splits,
                seed=seed,
                normalize_per_fold=False,
            )
            rows.append(
                {
                    "seed": seed,
                    "replicate": replicate,
                    "model_name": model_name,
                    "model_label": get_model_display_name(model_name),
                    "condition": "global_norm",
                    "drift_strength": drift_strength,
                    **global_norm,
                }
            )

            df_episode = apply_episodewise_normalization(df, base_feature_cols)
            episode_feature_cols = [f"{col}_epinorm" for col in base_feature_cols]
            episode_norm = grouped_auc_only(
                df_episode,
                feature_cols=episode_feature_cols,
                model_name=model_name,
                grouped_splits=grouped_splits,
                seed=seed,
                normalize_per_fold=False,
            )
            rows.append(
                {
                    "seed": seed,
                    "replicate": replicate,
                    "model_name": model_name,
                    "model_label": get_model_display_name(model_name),
                    "condition": "episode_norm",
                    "drift_strength": drift_strength,
                    **episode_norm,
                }
            )

    return pd.DataFrame(rows)


def summarize_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate robustness-grid results."""
    summary = (
        df.groupby(["model_label", "n_episodes", "ar_coef", "total_features"])
        .agg(
            grouped_auc_mean=("grouped_auc", "mean"),
            random_auc_mean=("random_auc", "mean"),
            delta_cv_mean=("delta_cv", "mean"),
            delta_cv_std=("delta_cv", "std"),
            n_eff_mean=("n_eff", "mean"),
            event_rate_mean=("event_rate", "mean"),
        )
        .reset_index()
    )
    return summary.round(3)


def summarize_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalization-leakage results on the drift DGP."""
    summary = (
        df.groupby(["model_label", "condition", "drift_strength"])
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            brier_mean=("brier", "mean"),
        )
        .reset_index()
    )
    return summary.round(3)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_replicates", type=int, default=2)
    parser.add_argument("--episode_grid", type=str, default="20,50,100")
    parser.add_argument("--ar_grid", type=str, default="0.0,0.6,0.9")
    parser.add_argument("--feature_grid", type=str, default="5,20,100")
    parser.add_argument("--model_grid", type=str, default="logistic,random_forest,boosted_trees")
    parser.add_argument("--grouped_splits", type=int, default=3)
    parser.add_argument("--random_splits", type=int, default=3)
    parser.add_argument("--drift_strength", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="results/strengthening")
    args = parser.parse_args()

    episode_grid = parse_int_list(args.episode_grid)
    ar_grid = parse_float_list(args.ar_grid)
    feature_grid = parse_int_list(args.feature_grid)
    model_grid = parse_str_list(args.model_grid)

    ensure_dir(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    robustness_df = run_robustness_grid(
        n_replicates=args.n_replicates,
        episode_grid=episode_grid,
        ar_grid=ar_grid,
        feature_grid=feature_grid,
        model_grid=model_grid,
        grouped_splits=args.grouped_splits,
        random_splits=args.random_splits,
    )
    robustness_summary = summarize_robustness(robustness_df)

    drift_df = run_drift_normalization_experiment(
        n_replicates=args.n_replicates,
        model_grid=model_grid,
        grouped_splits=args.grouped_splits,
        drift_strength=args.drift_strength,
    )
    drift_summary = summarize_drift(drift_df)

    robustness_path = f"{args.output_dir}/robustness_grid_{timestamp}.csv"
    robustness_summary_path = f"{args.output_dir}/robustness_summary_{timestamp}.csv"
    drift_path = f"{args.output_dir}/drift_experiment_{timestamp}.csv"
    drift_summary_path = f"{args.output_dir}/drift_summary_{timestamp}.csv"

    robustness_df.to_csv(robustness_path, index=False)
    robustness_summary.to_csv(robustness_summary_path, index=False)
    drift_df.to_csv(drift_path, index=False)
    drift_summary.to_csv(drift_summary_path, index=False)

    robustness_df.to_csv(f"{args.output_dir}/robustness_grid.csv", index=False)
    robustness_summary.to_csv(f"{args.output_dir}/robustness_summary.csv", index=False)
    drift_df.to_csv(f"{args.output_dir}/drift_experiment.csv", index=False)
    drift_summary.to_csv(f"{args.output_dir}/drift_summary.csv", index=False)
    robustness_df.to_csv(f"{args.output_dir}/robustness_grid_latest.csv", index=False)
    robustness_summary.to_csv(f"{args.output_dir}/robustness_summary_latest.csv", index=False)
    drift_df.to_csv(f"{args.output_dir}/drift_experiment_latest.csv", index=False)
    drift_summary.to_csv(f"{args.output_dir}/drift_summary_latest.csv", index=False)

    print(f"Saved robustness results to {robustness_path}")
    print(f"Saved drift results to {drift_path}")
    print("\nRobustness summary (collapsed across replicates):")
    print(robustness_summary.to_string(index=False))
    print("\nDrift normalization summary:")
    print(drift_summary.to_string(index=False))


if __name__ == "__main__":
    main()
