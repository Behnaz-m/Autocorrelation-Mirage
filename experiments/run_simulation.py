#!/usr/bin/env python
"""
Main simulation experiment for the temporal leakage paper.

This script:
1. Generates leak-free panel data
2. Creates 4 experimental conditions
3. Evaluates each condition with grouped and random CV
4. Produces Table 3 for the paper
5. Saves results for figure generation

Run with: python experiments/run_simulation.py [--n_replicates N] [--output_dir DIR]
"""

import numpy as np
import pandas as pd
import sys
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation import (
    filter_eligible_rows,
    generate_panel_data,
    prepare_modeling_data,
    get_feature_columns
)
from src.leakage_injection import (
    add_explicit_leak,
    apply_global_normalization
)
from src.evaluation import (
    bootstrap_delta_cv,
    collect_oof_prediction_frame,
    compute_pooled_oof_metrics,
    compute_prediction_frame_metrics,
    evaluate_grouped_cv,
    evaluate_random_cv,
    evaluate_temporal_blocked_within_episode_cv,
    compute_effective_sample_size
)


def _prediction_metrics_from_results(results: pd.DataFrame) -> dict:
    pred_df = collect_oof_prediction_frame(results)
    metrics = compute_prediction_frame_metrics(pred_df)
    return {"pred_df": pred_df, **metrics}


def run_single_replicate(
    seed: int,
    verbose: bool = False,
    delta_bootstrap_reps: int = 500,
    include_temporal_baseline: bool = True,
) -> dict:
    """
    Run one complete replicate of the experiment.

    Parameters
    ----------
    seed : int
        Random seed for this replicate
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results for all 4 conditions
    """
    # Generate data
    df = generate_panel_data(
        n_episodes=30,
        T_max=60,
        ar_coef=0.7,
        noise_std=0.3,
        hazard_coef=0.15,
        base_hazard=-3.0,
        alpha_std=0.5,
        horizon=14,
        seed=seed
    )

    feature_cols = get_feature_columns(df)
    eligible_df = filter_eligible_rows(df)
    row_ids = eligible_df["row_id"].to_numpy()
    times = eligible_df["t"].to_numpy()

    results = {}

    # ====== CONDITION 1: Leak-Free + Grouped CV (Baseline) ======
    X, y, groups = prepare_modeling_data(df, feature_cols)

    # Grouped CV (correct)
    grouped_res = evaluate_grouped_cv(X, y, groups, n_splits=5, row_ids=row_ids)
    grouped_metrics = _prediction_metrics_from_results(grouped_res)
    auc_grouped = grouped_metrics['auc']
    brier_grouped = grouped_metrics['brier']

    results['leak_free_grouped'] = {
        'auc': auc_grouped,
        'brier': brier_grouped,
        'episode_weighted_auc': grouped_metrics['episode_weighted_auc'],
        'episode_mean_brier': grouped_metrics['episode_mean_brier'],
        'method': 'grouped'
    }

    # ====== CONDITION 2: Leak-Free + Random CV (Pseudoreplication) ======
    random_res = evaluate_random_cv(
        X,
        y,
        normalize_before=True,
        seed=seed,
        groups=groups,
        row_ids=row_ids,
    )
    random_metrics = _prediction_metrics_from_results(random_res)
    auc_random = random_metrics['auc']
    brier_random = random_metrics['brier']

    results['leak_free_random'] = {
        'auc': auc_random,
        'brier': brier_random,
        'episode_weighted_auc': random_metrics['episode_weighted_auc'],
        'episode_mean_brier': random_metrics['episode_mean_brier'],
        'method': 'random'
    }

    delta_cv_boot = bootstrap_delta_cv(
        random_metrics["pred_df"],
        grouped_metrics["pred_df"],
        n_bootstrap=delta_bootstrap_reps,
        seed=seed,
    )
    results["delta_cv"] = delta_cv_boot

    if include_temporal_baseline:
        temporal_res = evaluate_temporal_blocked_within_episode_cv(
            X,
            y,
            groups,
            times=times,
            n_splits=5,
            row_ids=row_ids,
        )
        temporal_metrics = _prediction_metrics_from_results(temporal_res)
        results["leak_free_temporal"] = {
            "auc": temporal_metrics["auc"],
            "brier": temporal_metrics["brier"],
            "episode_weighted_auc": temporal_metrics["episode_weighted_auc"],
            "episode_mean_brier": temporal_metrics["episode_mean_brier"],
            "method": "temporal_within_episode",
        }

    # ====== CONDITION 3: Normalization Leak + Grouped CV ======
    df_norm, _ = apply_global_normalization(df, feature_cols)
    norm_cols = [f'{col}_norm' for col in feature_cols]
    norm_eligible_df = filter_eligible_rows(df_norm)
    X_norm, y_norm, groups_norm = prepare_modeling_data(df_norm, norm_cols)

    # Use grouped CV but with leaked features
    grouped_res_norm = evaluate_grouped_cv(
        X_norm,
        y_norm,
        groups_norm,
        normalize_per_fold=False,
        n_splits=5,
        row_ids=norm_eligible_df["row_id"].to_numpy(),
    )
    norm_metrics = _prediction_metrics_from_results(grouped_res_norm)
    auc_norm = norm_metrics['auc']
    brier_norm = norm_metrics['brier']

    results['norm_leak_grouped'] = {
        'auc': auc_norm,
        'brier': brier_norm,
        'episode_weighted_auc': norm_metrics['episode_weighted_auc'],
        'episode_mean_brier': norm_metrics['episode_mean_brier'],
        'method': 'grouped'
    }

    # ====== CONDITION 4: Explicit Leak + Grouped CV ======
    df_leak = add_explicit_leak(df, seed=seed)
    leak_cols = feature_cols + ['X_leak']
    leak_eligible_df = filter_eligible_rows(df_leak)
    X_leak, y_leak, groups_leak = prepare_modeling_data(df_leak, leak_cols)

    grouped_res_leak = evaluate_grouped_cv(
        X_leak,
        y_leak,
        groups_leak,
        n_splits=5,
        row_ids=leak_eligible_df["row_id"].to_numpy(),
    )
    leak_metrics = _prediction_metrics_from_results(grouped_res_leak)
    auc_leak = leak_metrics['auc']
    brier_leak = leak_metrics['brier']

    results['explicit_leak_grouped'] = {
        'auc': auc_leak,
        'brier': brier_leak,
        'episode_weighted_auc': leak_metrics['episode_weighted_auc'],
        'episode_mean_brier': leak_metrics['episode_mean_brier'],
        'method': 'grouped'
    }

    # Store effective sample size (same for all conditions from same data)
    n, m, rho, n_eff = compute_effective_sample_size(groups, y)
    results['effective_n'] = {
        'n': n,
        'm': m,
        'rho': rho,
        'n_eff': n_eff
    }

    results['seed'] = seed
    results['n_episodes'] = df['episode_id'].nunique()
    results['event_rate'] = df[df['at_risk'] == 1]['Y'].mean()

    return results


def run_full_experiment(
    n_replicates: int = 100,
    start_seed: int = 0,
    output_dir: str = "results",
    delta_bootstrap_reps: int = 500,
    include_temporal_baseline: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete simulation experiment with multiple replicates.

    Parameters
    ----------
    n_replicates : int
        Number of random replicates
    start_seed : int
        Starting seed
    output_dir : str
        Directory to save results

    Returns
    -------
    pd.DataFrame
        All results
    """
    print(f"Running {n_replicates} replicates...")

    all_results = []
    delta_rows = []

    for i in tqdm(range(n_replicates), desc="Replicates"):
        seed = start_seed + i
        try:
            results = run_single_replicate(
                seed,
                verbose=False,
                delta_bootstrap_reps=delta_bootstrap_reps,
                include_temporal_baseline=include_temporal_baseline,
            )
            all_results.append(results)
            delta_rows.append({
                "seed": seed,
                **results["delta_cv"],
            })
        except Exception as e:
            print(f"Error in replicate {i} (seed={seed}): {e}")
            continue

    # Convert to DataFrame
    rows = []
    for r in all_results:
        condition_order = [
            'leak_free_grouped',
            'leak_free_random',
            'norm_leak_grouped',
            'explicit_leak_grouped',
        ]
        if include_temporal_baseline and 'leak_free_temporal' in r:
            condition_order.append('leak_free_temporal')

        for condition in condition_order:
            rows.append({
                'seed': r['seed'],
                'condition': condition,
                'auc': r[condition]['auc'],
                'brier': r[condition]['brier'],
                'episode_weighted_auc': r[condition]['episode_weighted_auc'],
                'episode_mean_brier': r[condition]['episode_mean_brier'],
                'n_episodes': r['n_episodes'],
                'event_rate': r['event_rate'],
                'n_eff': r['effective_n']['n_eff'],
                'rho': r['effective_n']['rho']
            })

    df_results = pd.DataFrame(rows)
    df_delta = pd.DataFrame(delta_rows)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/main_benchmark_{timestamp}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    delta_file = f"{output_dir}/bootstrap_delta_cv_{timestamp}.csv"
    df_delta.to_csv(delta_file, index=False)
    print(f"Bootstrap Delta_CV saved to {delta_file}")

    # Also save latest
    df_results.to_csv(f"{output_dir}/main_benchmark.csv", index=False)
    df_results.to_csv(f"{output_dir}/main_benchmark_latest.csv", index=False)
    df_results.to_csv(f"{output_dir}/simulation_results_latest.csv", index=False)
    df_delta.to_csv(f"{output_dir}/bootstrap_delta_cv.csv", index=False)
    df_delta.to_csv(f"{output_dir}/bootstrap_delta_cv_latest.csv", index=False)

    return df_results, df_delta


def paired_t_interval(values: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """Return a two-sided t interval for a vector of paired differences."""
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return np.nan, np.nan
    if len(clean) == 1:
        return clean[0], clean[0]
    se = clean.std(ddof=1) / np.sqrt(len(clean))
    t_crit = stats.t.ppf(1 - alpha / 2, df=len(clean) - 1)
    mean = clean.mean()
    return mean - t_crit * se, mean + t_crit * se


def generate_main_table(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 3 for the paper: Summary of simulation results.

    Parameters
    ----------
    df_results : pd.DataFrame
        Raw results from run_full_experiment

    Returns
    -------
    pd.DataFrame
        Formatted table for paper
    """
    # Compute statistics by condition
    summary = df_results.groupby('condition').agg({
        'auc': ['mean', 'std'],
        'brier': ['mean', 'std'],
        'episode_weighted_auc': ['mean', 'std'],
        'episode_mean_brier': ['mean', 'std'],
    }).round(3)

    # Flatten column names
    summary.columns = [
        'auc_mean', 'auc_std',
        'brier_mean', 'brier_std',
        'episode_weighted_auc_mean', 'episode_weighted_auc_std',
        'episode_mean_brier_mean', 'episode_mean_brier_std',
    ]
    summary = summary.reset_index()

    baseline_by_seed = (
        df_results[df_results['condition'] == 'leak_free_grouped'][['seed', 'auc']]
        .rename(columns={'auc': 'baseline_auc'})
    )
    deltas = df_results.merge(baseline_by_seed, on='seed', how='left')
    deltas['delta_auc_vs_baseline'] = deltas['auc'] - deltas['baseline_auc']
    ci_by_condition = (
        deltas.groupby('condition')['delta_auc_vs_baseline']
        .apply(lambda s: pd.Series(paired_t_interval(s), index=['ci_low', 'ci_high']))
        .reset_index()
    )
    ci_by_condition = ci_by_condition.pivot(index='condition', columns='level_1', values='delta_auc_vs_baseline').reset_index()
    summary = summary.merge(ci_by_condition, on='condition', how='left')
    summary['delta_auc_mean'] = summary['auc_mean'] - summary.loc[summary['condition'] == 'leak_free_grouped', 'auc_mean'].iloc[0]

    # Rename conditions for paper
    condition_names = {
        'leak_free_grouped': 'Leak-Free + Grouped CV',
        'leak_free_random': 'Leak-Free + Row-wise KFold',
        'norm_leak_grouped': 'Normalization Leak + Grouped CV',
        'explicit_leak_grouped': 'Explicit Leak + Grouped CV',
        'leak_free_temporal': 'Leak-Free + Temporal-Blocked Within-Episode CV',
    }
    summary['Condition'] = summary['condition'].map(condition_names)

    # Format for paper
    summary['Brier'] = summary.apply(lambda x: f"{x['brier_mean']:.3f} +/- {x['brier_std']:.3f}", axis=1)
    summary['AUC'] = summary.apply(lambda x: f"{x['auc_mean']:.3f} +/- {x['auc_std']:.3f}", axis=1)
    summary['Delta AUC vs. baseline'] = summary.apply(
        lambda x: 'baseline' if x['condition'] == 'leak_free_grouped' else f"{x['delta_auc_mean']:+.3f}",
        axis=1
    )
    summary['95% paired CI'] = summary.apply(
        lambda x: '--' if x['condition'] == 'leak_free_grouped' else f"[{x['ci_low']:.3f}, {x['ci_high']:.3f}]",
        axis=1
    )

    # Select columns for paper
    table = summary[['Condition', 'AUC', 'Brier', 'Delta AUC vs. baseline', '95% paired CI']]

    # Reorder rows
    order = [
        'Leak-Free + Grouped CV',
        'Leak-Free + Row-wise KFold',
        'Normalization Leak + Grouped CV',
        'Explicit Leak + Grouped CV',
    ]
    table = table.set_index('Condition').reindex(order).dropna(how='all').reset_index()

    return table


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run temporal leakage simulation experiment')
    parser.add_argument('--n_replicates', type=int, default=100,
                        help='Number of simulation replicates (default: 100)')
    parser.add_argument('--start_seed', type=int, default=42,
                        help='Starting random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results (default: results)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run with 10 replicates')
    parser.add_argument('--delta_bootstrap_reps', type=int, default=500,
                        help='Episode-bootstrap replicates for Delta_CV uncertainty')
    parser.add_argument('--skip_temporal_baseline', action='store_true',
                        help='Skip the temporal-blocked within-episode baseline')

    args = parser.parse_args()

    if args.quick:
        args.n_replicates = 10

    print("=" * 60)
    print("TEMPORAL LEAKAGE SIMULATION EXPERIMENT")
    print("=" * 60)

    # Run experiment
    print(f"\nConfiguration:")
    print(f"  Replicates: {args.n_replicates}")
    print(f"  Start seed: {args.start_seed}")
    print(f"  Output dir: {args.output_dir}")

    df_results, df_delta = run_full_experiment(
        n_replicates=args.n_replicates,
        start_seed=args.start_seed,
        output_dir=args.output_dir,
        delta_bootstrap_reps=args.delta_bootstrap_reps,
        include_temporal_baseline=not args.skip_temporal_baseline,
    )

    # Generate Table 3
    print("\n" + "=" * 60)
    print("TABLE 3: Simulation Results")
    print("=" * 60)
    table3 = generate_main_table(df_results)
    print(table3.to_string(index=False))

    # Save table
    os.makedirs(f"{args.output_dir}/tables", exist_ok=True)
    table3.to_csv(f"{args.output_dir}/tables/table3.csv", index=False)
    print(f"\nTable saved to {args.output_dir}/tables/table3.csv")
    summary = (
        df_results.groupby("condition")
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            episode_weighted_auc_mean=("episode_weighted_auc", "mean"),
            episode_mean_brier_mean=("episode_mean_brier", "mean"),
            brier_mean=("brier", "mean"),
        )
        .reset_index()
        .round(3)
    )
    summary.to_csv(f"{args.output_dir}/main_benchmark_summary_latest.csv", index=False)
    print(f"Summary saved to {args.output_dir}/main_benchmark_summary_latest.csv")

    # Print additional statistics
    print("\n" + "=" * 60)
    print("ADDITIONAL STATISTICS")
    print("=" * 60)

    baseline = df_results[df_results['condition'] == 'leak_free_grouped']
    print(f"Average effective n: {baseline['n_eff'].mean():.1f}")
    print(f"Average ICC (rho): {baseline['rho'].mean():.3f}")
    print(f"Average event rate: {baseline['event_rate'].mean():.1%}")

    # Effect sizes
    leak_free_auc = df_results[df_results['condition'] == 'leak_free_grouped']['auc'].mean()
    explicit_leak_auc = df_results[df_results['condition'] == 'explicit_leak_grouped']['auc'].mean()
    print(f"\nAUC gap (explicit leak vs baseline): {explicit_leak_auc - leak_free_auc:.3f}")
    if len(df_delta):
        print(
            "Mean Delta_CV bootstrap CI: "
            f"[{df_delta['ci_lower'].mean():.3f}, {df_delta['ci_upper'].mean():.3f}]"
        )

    # Generate figures
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    try:
        from src.plotting import generate_all_figures
        figures = generate_all_figures(df_results, output_dir=f"{args.output_dir}/figures")
        print("Figures generated successfully!")
    except Exception as e:
        print(f"Warning: Could not generate figures: {e}")
        print("You can generate figures later by running: python src/plotting.py")

    return df_results, df_delta


if __name__ == "__main__":
    df_results, df_delta = main()
