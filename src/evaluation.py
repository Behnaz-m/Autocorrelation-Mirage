"""
Evaluation protocols for temporal panel data.

This module provides:
1. Episode-grouped cross-validation (correct)
2. Random K-fold cross-validation (wrong - for comparison)
3. Metrics computation (AUC, Brier, calibration)
4. Bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from typing import Tuple, List, Dict, Optional, Callable
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):
        return iterable
import warnings

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None
    HAS_XGBOOST = False


def get_default_model(seed: int = 42) -> BaseEstimator:
    """Return the main-benchmark model used in the manuscript."""
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.1,
        max_iter=100,
        random_state=seed
    )


def get_xgboost_model(seed: int = 42) -> BaseEstimator:
    """Return XGBoost when available, else a boosted-tree fallback."""
    if HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
    return get_default_model(seed=seed)


def get_logistic_regression_model(seed: int = 42) -> LogisticRegression:
    """Return a regularized logistic-regression baseline."""
    return LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=seed
    )


def get_random_forest_model(seed: int = 42) -> RandomForestClassifier:
    """Return a random-forest baseline."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1
    )


def create_model(model_name: str, seed: int = 42) -> BaseEstimator:
    """Create a model by name for robustness experiments."""
    model_name = model_name.lower()
    if model_name == 'boosted_trees':
        return get_default_model(seed=seed)
    if model_name == 'xgboost':
        return get_xgboost_model(seed=seed)
    if model_name == 'logistic':
        return get_logistic_regression_model(seed=seed)
    if model_name == 'random_forest':
        return get_random_forest_model(seed=seed)
    raise ValueError(f"Unknown model_name: {model_name}")


def get_model_display_name(model_name: str) -> str:
    """Return a display label for a model identifier."""
    return {
        'boosted_trees': 'Boosted Trees',
        'xgboost': 'Boosted Trees',
        'logistic': 'Logistic Regression',
        'random_forest': 'Random Forest'
    }.get(model_name, model_name)


def _fit_predict_proba(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray
) -> np.ndarray:
    """Fit a fresh clone and return positive-class probabilities."""
    fold_model = clone(model)
    fold_model.fit(X_train, y_train)
    return fold_model.predict_proba(X_test)[:, 1]


def collect_oof_predictions(results: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate out-of-fold labels and probabilities across folds.

    Parameters
    ----------
    results : pd.DataFrame
        Output from evaluate_grouped_cv or evaluate_random_cv containing y_true
        and y_prob arrays for each fold.

    Returns
    -------
    y_true : np.ndarray
        Concatenated true labels.
    y_prob : np.ndarray
        Concatenated out-of-fold predicted probabilities.
    """
    y_true = np.concatenate(results['y_true'].to_numpy()) if len(results) else np.array([])
    y_prob = np.concatenate(results['y_prob'].to_numpy()) if len(results) else np.array([])
    return y_true, y_prob


def collect_oof_prediction_frame(results: pd.DataFrame) -> pd.DataFrame:
    """
    Expand per-fold arrays into one aligned prediction frame.

    Expected columns in ``results`` include ``y_true`` and ``y_prob`` plus
    optional metadata arrays such as ``row_id`` and ``episode_id``.
    """
    if len(results) == 0:
        return pd.DataFrame(columns=["fold", "row_id", "episode_id", "y_true", "y_prob"])

    frames: list[pd.DataFrame] = []
    for row in results.itertuples(index=False):
        n_obs = len(row.y_true)
        frame_dict = {
            "fold": np.repeat(row.fold, n_obs),
            "y_true": np.asarray(row.y_true),
            "y_prob": np.asarray(row.y_prob),
        }
        if hasattr(row, "row_id"):
            frame_dict["row_id"] = np.asarray(row.row_id)
        if hasattr(row, "episode_id"):
            frame_dict["episode_id"] = np.asarray(row.episode_id)
        frames.append(pd.DataFrame(frame_dict))

    pred_df = pd.concat(frames, ignore_index=True)
    if "row_id" not in pred_df.columns:
        pred_df["row_id"] = np.arange(len(pred_df), dtype=int)
    return pred_df


def compute_prediction_frame_metrics(pred_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute pooled and episode-weighted metrics from aligned predictions.

    The episode-weighted AUC uses row weights proportional to ``1 / n_e`` so
    longer episodes do not dominate. The episode-mean Brier score averages the
    within-episode mean squared error across episodes.
    """
    if len(pred_df) == 0:
        return {
            "auc": np.nan,
            "brier": np.nan,
            "episode_weighted_auc": np.nan,
            "episode_mean_brier": np.nan,
            "n_obs": 0,
        }

    y_true = pred_df["y_true"].to_numpy()
    y_prob = pred_df["y_prob"].to_numpy()
    auc = np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    episode_weighted_auc = np.nan
    episode_mean_brier = np.nan
    if "episode_id" in pred_df.columns:
        episode_counts = pred_df.groupby("episode_id").size()
        sample_weight = pred_df["episode_id"].map(lambda e: 1.0 / episode_counts.loc[e]).to_numpy()
        if len(np.unique(y_true)) >= 2:
            episode_weighted_auc = roc_auc_score(y_true, y_prob, sample_weight=sample_weight)
        brier_by_episode = (
            pred_df.assign(brier=(pred_df["y_true"] - pred_df["y_prob"]) ** 2)
            .groupby("episode_id")["brier"]
            .mean()
        )
        episode_mean_brier = brier_by_episode.mean()

    return {
        "auc": auc,
        "brier": brier,
        "episode_weighted_auc": episode_weighted_auc,
        "episode_mean_brier": episode_mean_brier,
        "n_obs": len(pred_df),
    }


def bootstrap_delta_cv(
    pred_row: pd.DataFrame,
    pred_group: pd.DataFrame,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap the split-sensitivity gap ``Delta_CV`` by resampling episodes.

    ``pred_row`` and ``pred_group`` must refer to the same eligible rows and be
    alignable by ``row_id``.
    """
    required_cols = {"row_id", "episode_id", "y_true", "y_prob"}
    if not required_cols.issubset(pred_row.columns) or not required_cols.issubset(pred_group.columns):
        raise KeyError("bootstrap_delta_cv requires row_id, episode_id, y_true, and y_prob columns.")

    merged = (
        pred_row[["row_id", "episode_id", "y_true", "y_prob"]]
        .rename(columns={"y_prob": "y_prob_row"})
        .merge(
            pred_group[["row_id", "episode_id", "y_true", "y_prob"]].rename(
                columns={"y_prob": "y_prob_group", "episode_id": "episode_id_group", "y_true": "y_true_group"}
            ),
            on="row_id",
            how="inner",
        )
    )

    if len(merged) == 0:
        return {
            "delta_cv": np.nan,
            "auc_row": np.nan,
            "auc_group": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_bootstrap_valid": 0,
        }

    if not (merged["episode_id"] == merged["episode_id_group"]).all():
        raise ValueError("Row-wise and grouped predictions disagree on episode_id alignment.")
    if not (merged["y_true"] == merged["y_true_group"]).all():
        raise ValueError("Row-wise and grouped predictions disagree on y_true alignment.")

    y_true = merged["y_true"].to_numpy()
    auc_row = np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, merged["y_prob_row"])
    auc_group = np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, merged["y_prob_group"])
    delta_cv = auc_row - auc_group if np.isfinite(auc_row) and np.isfinite(auc_group) else np.nan

    unique_episodes = merged["episode_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap_deltas: list[float] = []

    for _ in range(n_bootstrap):
        sampled_episodes = rng.choice(unique_episodes, size=len(unique_episodes), replace=True)
        multiplicities = pd.Series(sampled_episodes).value_counts()
        weights = merged["episode_id"].map(multiplicities).fillna(0).to_numpy(dtype=float)
        active = weights > 0
        if active.sum() == 0 or len(np.unique(y_true[active])) < 2:
            continue
        auc_row_boot = roc_auc_score(y_true[active], merged.loc[active, "y_prob_row"], sample_weight=weights[active])
        auc_group_boot = roc_auc_score(y_true[active], merged.loc[active, "y_prob_group"], sample_weight=weights[active])
        bootstrap_deltas.append(auc_row_boot - auc_group_boot)

    if len(bootstrap_deltas) == 0:
        ci_lower = np.nan
        ci_upper = np.nan
    else:
        ci_lower = float(np.percentile(bootstrap_deltas, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2)))

    return {
        "delta_cv": delta_cv,
        "auc_row": auc_row,
        "auc_group": auc_group,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap_valid": len(bootstrap_deltas),
    }


def compute_pooled_oof_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Compute pooled metrics from out-of-fold predictions.

    This avoids averaging fold-level AUCs and avoids dropping folds with only one
    class as long as the pooled out-of-fold predictions contain both classes.
    """
    y_true, y_prob = collect_oof_predictions(results)

    if len(y_true) == 0:
        return {'auc': np.nan, 'brier': np.nan, 'n_obs': 0}

    auc = np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    return {'auc': auc, 'brier': brier, 'n_obs': len(y_true)}


def evaluate_grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model: Optional[BaseEstimator] = None,
    normalize_per_fold: bool = True,
    n_splits: Optional[int] = None,
    row_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Evaluate using leave-one-episode-out cross-validation (CORRECT).

    This ensures that:
    1. Each episode is held out completely
    2. Preprocessing (normalization) is fit on training data only
    3. No information leaks from test episodes to training

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    groups : np.ndarray
        Episode IDs
    model : BaseEstimator, optional
        Model to use (default: histogram gradient boosting)
    normalize_per_fold : bool
        Whether to normalize within each fold
    n_splits : int, optional
        If provided, use GroupKFold with this many splits; otherwise use
        leave-one-group-out validation.

    Returns
    -------
    pd.DataFrame
        Per-fold results with columns including fold, auc, brier, n_obs, n_pos,
        and n_test_groups.
    """
    if model is None:
        model = get_default_model()
    if row_ids is None:
        row_ids = np.arange(len(y), dtype=int)

    splitter = LeaveOneGroupOut() if n_splits is None else GroupKFold(
        n_splits=min(n_splits, len(np.unique(groups)))
    )
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize on training data only (no leakage)
        if normalize_per_fold:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Handle case where test set has only one class
        if len(np.unique(y_test)) < 2:
            # Can't compute AUC with single class
            auc = np.nan
            y_prob = _fit_predict_proba(model, X_train, y_train, X_test)
        else:
            y_prob = _fit_predict_proba(model, X_train, y_train, X_test)
            auc = roc_auc_score(y_test, y_prob)

        # Brier score can always be computed
        brier = brier_score_loss(y_test, y_prob)

        test_groups = np.unique(groups[test_idx])
        results.append({
            'fold': fold_idx,
            'episode_id': groups[test_idx],
            'auc': auc,
            'brier': brier,
            'n_obs': len(y_test),
            'n_pos': y_test.sum(),
            'n_test_groups': len(test_groups),
            'row_id': row_ids[test_idx],
            'y_prob': y_prob,
            'y_true': y_test
        })

    return pd.DataFrame(results)


def evaluate_random_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    model: Optional[BaseEstimator] = None,
    normalize_before: bool = True,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
    row_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Evaluate using random K-fold cross-validation (WRONG for panel data).

    This demonstrates the pseudoreplication problem:
    - Same episode can appear in both train and test
    - Model learns episode-specific patterns

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    n_splits : int
        Number of CV folds
    model : BaseEstimator, optional
        Model to use
    normalize_before : bool
        If True, normalize on ALL data before CV (adds leakage)
        If False, normalize within each fold
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Per-fold results
    """
    if model is None:
        model = get_default_model()
    if row_ids is None:
        row_ids = np.arange(len(y), dtype=int)
    if groups is None:
        groups = np.full(len(y), -1, dtype=int)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []

    # WRONG: Normalize on all data before splitting
    if normalize_before:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize within fold if not done before
        if not normalize_before:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if len(np.unique(y_train)) < 2:
            y_prob = np.full(len(y_test), y_train[0], dtype=float)
        else:
            y_prob = _fit_predict_proba(model, X_train, y_train, X_test)

        auc = np.nan if len(np.unique(y_test)) < 2 else roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        results.append({
            'fold': fold_idx,
            'episode_id': groups[test_idx],
            'auc': auc,
            'brier': brier,
            'n_obs': len(y_test),
            'n_pos': y_test.sum(),
            'row_id': row_ids[test_idx],
            'y_prob': y_prob,
            'y_true': y_test
        })

    return pd.DataFrame(results)


def evaluate_temporal_blocked_within_episode_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    times: np.ndarray,
    n_splits: int = 5,
    model: Optional[BaseEstimator] = None,
    normalize_per_fold: bool = True,
    min_train_size: int = 5,
    row_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Evaluate a within-episode forecasting target using only past rows for each test block.

    Each fold pools a contiguous future block from every episode and trains only
    on chronologically earlier rows from those same episodes. This estimates a
    different target from grouped CV: forecasting later rows of already observed
    episodes rather than transferring to unseen episodes.
    """
    if model is None:
        model = get_default_model()
    if row_ids is None:
        row_ids = np.arange(len(y), dtype=int)

    episode_order = {
        episode_id: episode_idx[np.argsort(times[episode_idx])]
        for episode_id in np.unique(groups)
        for episode_idx in [np.where(groups == episode_id)[0]]
    }
    episode_blocks = {
        episode_id: [block for block in np.array_split(indices, n_splits) if len(block) > 0]
        for episode_id, indices in episode_order.items()
    }

    results = []
    for fold_idx in range(1, n_splits):
        train_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []

        for episode_id, blocks in episode_blocks.items():
            if fold_idx >= len(blocks):
                continue
            train_idx = np.concatenate(blocks[:fold_idx]) if fold_idx > 0 else np.array([], dtype=int)
            test_idx = blocks[fold_idx]
            if len(train_idx) < min_train_size or len(test_idx) == 0:
                continue
            train_parts.append(train_idx)
            test_parts.append(test_idx)

        if not train_parts or not test_parts:
            continue

        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if normalize_per_fold:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if len(np.unique(y_train)) < 2:
            y_prob = np.full(len(y_test), y_train[0], dtype=float)
        else:
            y_prob = _fit_predict_proba(model, X_train, y_train, X_test)

        auc = np.nan if len(np.unique(y_test)) < 2 else roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        results.append({
            "fold": fold_idx,
            "episode_id": groups[test_idx],
            "auc": auc,
            "brier": brier,
            "n_obs": len(y_test),
            "n_pos": y_test.sum(),
            "row_id": row_ids[test_idx],
            "y_prob": y_prob,
            "y_true": y_test,
        })

    return pd.DataFrame(results)


def episode_bootstrap_ci(
    episode_scores: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval by resampling episodes.

    Parameters
    ----------
    episode_scores : np.ndarray
        Score for each episode
    n_bootstrap : int
        Number of bootstrap replicates
    alpha : float
        Significance level (0.05 for 95% CI)
    seed : int
        Random seed

    Returns
    -------
    mean : float
        Point estimate (mean of episode scores)
    ci_lower : float
        Lower bound of CI
    ci_upper : float
        Upper bound of CI
    """
    rng = np.random.default_rng(seed)

    # Remove NaN values
    scores = episode_scores[~np.isnan(episode_scores)]

    if len(scores) == 0:
        return np.nan, np.nan, np.nan

    # Bootstrap
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)

    mean = np.mean(scores)
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return mean, ci_lower, ci_upper


def compute_effective_sample_size(
    groups: np.ndarray,
    y: np.ndarray
) -> Tuple[int, float, float, float]:
    """
    Compute effective sample size accounting for clustering.

    Parameters
    ----------
    groups : np.ndarray
        Episode IDs
    y : np.ndarray
        Labels (used to estimate ICC)

    Returns
    -------
    n : int
        Total observations
    m : float
        Average cluster size
    rho : float
        Estimated ICC (intraclass correlation)
    n_eff : float
        Effective sample size
    """
    n = len(y)
    unique_groups = np.unique(groups)
    E = len(unique_groups)
    m = n / E  # Average cluster size

    # Estimate ICC using ANOVA approach
    # ICC = (MSB - MSW) / (MSB + (m-1)*MSW)
    group_means = np.array([y[groups == g].mean() for g in unique_groups])
    grand_mean = y.mean()

    # Between-group sum of squares
    SSB = sum([len(y[groups == g]) * (group_means[i] - grand_mean)**2
               for i, g in enumerate(unique_groups)])

    # Within-group sum of squares
    SSW = sum([((y[groups == g] - group_means[i])**2).sum()
               for i, g in enumerate(unique_groups)])

    # Mean squares
    MSB = SSB / (E - 1) if E > 1 else 0
    MSW = SSW / (n - E) if n > E else 1

    # ICC
    if MSB + (m - 1) * MSW > 0:
        rho = (MSB - MSW) / (MSB + (m - 1) * MSW)
        rho = max(0, min(1, rho))  # Clip to [0, 1]
    else:
        rho = 0

    # Effective sample size
    design_effect = 1 + (m - 1) * rho
    n_eff = n / design_effect

    return n, m, rho, n_eff


def aggregate_results(
    grouped_results: pd.DataFrame,
    random_results: pd.DataFrame
) -> Dict:
    """
    Aggregate and compare results from both evaluation methods.

    Returns
    -------
    dict
        Summary statistics for both methods
    """
    # Grouped and random CV results from pooled out-of-fold predictions
    grouped_pooled = compute_pooled_oof_metrics(grouped_results)
    random_pooled = compute_pooled_oof_metrics(random_results)

    grouped_auc = grouped_results['auc'].dropna()
    grouped_mean, grouped_ci_low, grouped_ci_high = episode_bootstrap_ci(grouped_auc.values)
    grouped_brier = grouped_results['brier'].values
    brier_mean, brier_ci_low, brier_ci_high = episode_bootstrap_ci(grouped_brier)

    return {
        'grouped_cv': {
            'auc_mean': grouped_pooled['auc'],
            'auc_fold_mean': grouped_mean,
            'auc_ci': (grouped_ci_low, grouped_ci_high),
            'brier_mean': grouped_pooled['brier'],
            'brier_fold_mean': brier_mean,
            'brier_ci': (brier_ci_low, brier_ci_high),
            'n_episodes': len(grouped_results)
        },
        'random_cv': {
            'auc_mean': random_pooled['auc'],
            'auc_std': random_results['auc'].std(),
            'brier_mean': random_pooled['brier'],
            'n_folds': len(random_results)
        },
        'inflation': {
            'auc_absolute': random_pooled['auc'] - grouped_pooled['auc'],
            'auc_relative': (
                (random_pooled['auc'] - grouped_pooled['auc']) / grouped_pooled['auc'] * 100
                if grouped_pooled['auc'] > 0 else np.nan
            )
        }
    }


def run_full_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    condition_name: str = "unnamed"
) -> Dict:
    """
    Run complete evaluation pipeline for one condition.

    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    groups : np.ndarray
        Episode IDs
    condition_name : str
        Name for logging

    Returns
    -------
    dict
        Complete evaluation results
    """
    print(f"\nEvaluating condition: {condition_name}")
    print(f"  Data shape: X={X.shape}, y={y.shape}")
    print(f"  Episodes: {len(np.unique(groups))}")
    print(f"  Event rate: {y.mean():.1%}")

    # Run grouped CV (correct)
    print("  Running grouped CV...")
    grouped_results = evaluate_grouped_cv(X, y, groups)

    # Run random CV (wrong)
    print("  Running random CV...")
    random_results = evaluate_random_cv(X, y)

    # Compute effective sample size
    n, m, rho, n_eff = compute_effective_sample_size(groups, y)

    # Aggregate
    summary = aggregate_results(grouped_results, random_results)
    summary['effective_n'] = {
        'n': n,
        'm': m,
        'rho': rho,
        'n_eff': n_eff
    }
    summary['condition'] = condition_name
    summary['grouped_results'] = grouped_results
    summary['random_results'] = random_results

    return summary


if __name__ == "__main__":
    # Test evaluation
    from src.data_generation import generate_default_data, prepare_modeling_data, get_feature_columns

    print("Testing evaluation protocols...")
    df = generate_default_data(seed=42)
    feature_cols = get_feature_columns(df)
    X, y, groups = prepare_modeling_data(df, feature_cols)

    results = run_full_evaluation(X, y, groups, "leak_free_test")

    print("\n=== Results Summary ===")
    print(f"Grouped CV AUC: {results['grouped_cv']['auc_mean']:.3f} "
          f"[{results['grouped_cv']['auc_ci'][0]:.3f}, {results['grouped_cv']['auc_ci'][1]:.3f}]")
    print(f"Random CV AUC: {results['random_cv']['auc_mean']:.3f} ± {results['random_cv']['auc_std']:.3f}")
    print(f"AUC Inflation: {results['inflation']['auc_absolute']:.3f} ({results['inflation']['auc_relative']:.1f}%)")
    print(f"Effective n: {results['effective_n']['n_eff']:.0f} (from n={results['effective_n']['n']}, ρ={results['effective_n']['rho']:.2f})")
