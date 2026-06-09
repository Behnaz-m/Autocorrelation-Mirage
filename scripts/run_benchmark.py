#!/usr/bin/env python3
"""
Convenience wrapper for reproducing the paper experiments.

Examples
--------
Smoke test:
    venv/bin/python scripts/run_benchmark.py --smoke-test

Paper-scale runs:
    venv/bin/python scripts/run_benchmark.py --n-reps-main 30 --n-reps-grid 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-strengthening", action="store_true")
    parser.add_argument("--n-reps-main", type=int, default=30)
    parser.add_argument("--n-reps-grid", type=int, default=2)
    parser.add_argument("--delta-bootstrap-reps", type=int, default=500)
    parser.add_argument("--python", type=str, default=str(REPO_ROOT / "venv" / "bin" / "python"))
    args = parser.parse_args()

    python_bin = args.python

    if args.smoke_test:
        n_reps_main = 2
        n_reps_grid = 1
        delta_bootstrap_reps = 100
        main_output = "results/smoke_main"
        strengthening_output = "results/smoke_strengthening"
    else:
        n_reps_main = args.n_reps_main
        n_reps_grid = args.n_reps_grid
        delta_bootstrap_reps = args.delta_bootstrap_reps
        main_output = "results/protocol_main_30"
        strengthening_output = "results/strengthening_pooled"

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/iberamia_mpl")

    if not args.skip_main:
        run_command(
            [
                python_bin,
                "experiments/run_simulation.py",
                "--n_replicates",
                str(n_reps_main),
                "--delta_bootstrap_reps",
                str(delta_bootstrap_reps),
                "--skip_temporal_baseline",
                "--output_dir",
                main_output,
            ],
            env=env,
        )

    if not args.skip_strengthening:
        run_command(
            [
                python_bin,
                "experiments/run_strengthening_experiments.py",
                "--n_replicates",
                str(n_reps_grid),
                "--episode_grid",
                "20,50,100",
                "--ar_grid",
                "0.0,0.6,0.9",
                "--feature_grid",
                "5,20,100",
                "--model_grid",
                "logistic,random_forest,boosted_trees",
                "--output_dir",
                strengthening_output,
            ],
            env=env,
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
