import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data_generation import generate_single_episode, prepare_modeling_data
from src.evaluation import bootstrap_delta_cv


class CensoringAndGroupingTests(unittest.TestCase):
    def test_censored_episode_has_no_positive_labels(self) -> None:
        rng = np.random.default_rng(123)
        df = generate_single_episode(
            episode_id=0,
            T_max=10,
            alpha_e=0.0,
            ar_coef=0.2,
            noise_std=0.1,
            hazard_coef=0.0,
            base_hazard=-1_000.0,
            horizon=3,
            drift_strength=0.0,
            rng=rng,
        )

        self.assertEqual(int(df["event_observed"].iloc[0]), 0)
        self.assertTrue(np.isnan(df["T_e"]).all())
        self.assertTrue((df["Y"] == 0).all())
        self.assertTrue((df.loc[df["t"] >= 8, "at_risk"] == 0).all())
        self.assertTrue((df.loc[df["at_risk"] == 1, "t"] + 3 <= df.loc[df["at_risk"] == 1, "C_e"]).all())

    def test_observed_event_episode_keeps_only_pre_event_rows(self) -> None:
        rng = np.random.default_rng(456)
        df = generate_single_episode(
            episode_id=1,
            T_max=10,
            alpha_e=0.0,
            ar_coef=0.2,
            noise_std=0.1,
            hazard_coef=0.0,
            base_hazard=1_000.0,
            horizon=3,
            drift_strength=0.0,
            rng=rng,
        )

        self.assertEqual(int(df["event_observed"].iloc[0]), 1)
        event_time = int(df["T_e"].iloc[0])
        self.assertEqual(len(df), event_time)
        self.assertEqual(int(df["t"].max()), event_time - 1)
        self.assertTrue((df["at_risk"] == 1).all())
        self.assertTrue((df["t"] < df["T_e"]).all())

    def test_prepare_modeling_data_filters_to_eligible_rows(self) -> None:
        rng = np.random.default_rng(789)
        df = generate_single_episode(
            episode_id=2,
            T_max=12,
            alpha_e=0.0,
            ar_coef=0.2,
            noise_std=0.1,
            hazard_coef=0.0,
            base_hazard=-1_000.0,
            horizon=4,
            drift_strength=0.0,
            rng=rng,
        )

        X, y, groups = prepare_modeling_data(df)
        eligible = df[df["at_risk"] == 1]

        self.assertEqual(len(X), len(eligible))
        self.assertEqual(len(y), len(eligible))
        self.assertEqual(len(groups), len(eligible))
        self.assertEqual(int(y.sum()), 0)

    def test_groupkfold_keeps_episode_ids_disjoint(self) -> None:
        X = np.arange(24).reshape(12, 2)
        y = np.array([0, 1] * 6)
        groups = np.repeat(np.arange(4), 3)

        splitter = GroupKFold(n_splits=4)
        for train_idx, test_idx in splitter.split(X, y, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_bootstrap_delta_cv_resamples_by_episode(self) -> None:
        pred_row = pd.DataFrame(
            {
                "row_id": np.arange(8),
                "episode_id": np.repeat([0, 1, 2, 3], 2),
                "y_true": np.array([0, 1, 0, 1, 0, 1, 0, 1]),
                "y_prob": np.array([0.2, 0.9, 0.3, 0.8, 0.4, 0.7, 0.1, 0.95]),
            }
        )
        pred_group = pred_row.copy()
        pred_group["y_prob"] = np.array([0.60, 0.40, 0.55, 0.45, 0.50, 0.35, 0.65, 0.30])

        summary = bootstrap_delta_cv(pred_row, pred_group, n_bootstrap=50, seed=123)

        self.assertGreater(summary["delta_cv"], 0)
        self.assertTrue(np.isfinite(summary["ci_lower"]))
        self.assertTrue(np.isfinite(summary["ci_upper"]))
        self.assertGreater(summary["n_bootstrap_valid"], 0)


if __name__ == "__main__":
    unittest.main()
