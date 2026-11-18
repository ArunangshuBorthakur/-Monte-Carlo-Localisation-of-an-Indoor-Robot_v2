"""Experiment sweeping particle counts: 100, 500, 1000, 5000."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.experiments import run_trial, save_summary_csv, print_grouped_summary
from src.mcl import ParticleFilter


def run_experiment(seeds: list[int], particle_counts: list[int] = [100, 500, 1000, 5000], num_steps: int = 100) -> pd.DataFrame:
    records = []

    print(f"Running Particle Sweep on {len(seeds)} seeds across counts {particle_counts}...")
    for n in particle_counts:
        for seed in seeds:
            res = run_trial(
                seed=seed,
                filter_factory=lambda w, l, count=n, s=seed: ParticleFilter(
                    w, l, n=count, seed=s, model="beam", resampling="systematic"
                ),
                num_steps=num_steps,
            )
            res["particle_count"] = n
            records.append(res)

    df = pd.DataFrame(records)
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed", "particle_count", "position_rmse", "heading_rmse",
        "final_position_error", "final_heading_error", "convergence_time",
        "success", "runtime"
    ]
    save_summary_csv(tables_dir / "particle_sweep.csv", records, fields)

    grouped = df.groupby("particle_count").agg({
        "position_rmse": ["mean", "std"],
        "runtime": ["mean", "std"],
        "success": "mean"
    }).reset_index()

    counts = grouped["particle_count"]
    rmse_mean = grouped["position_rmse"]["mean"]
    rmse_std = grouped["position_rmse"]["std"].fillna(0)
    rt_mean = grouped["runtime"]["mean"]
    rt_std = grouped["runtime"]["std"].fillna(0)

    # 1. RMSE vs Particle Count
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(counts, rmse_mean, yerr=rmse_std, fmt="-o", color="#1f77b4", capsize=5, lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Particle Count (log scale)")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Position RMSE vs Particle Count", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "particle_sweep_rmse.png", dpi=150)
    plt.close(fig)

    # 2. Runtime vs Particle Count
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(counts, rt_mean, yerr=rt_std, fmt="-s", color="#d62728", capsize=5, lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Particle Count (log scale)")
    ax.set_ylabel("Execution Runtime (s)")
    ax.set_title("Execution Runtime vs Particle Count", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "particle_sweep_runtime.png", dpi=150)
    plt.close(fig)

    print_grouped_summary(df, "particle_count")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run particle sweep experiment.")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test (2 seeds, small counts).")
    parser.add_argument("--steps", type=int, default=100, help="Steps per trial.")
    args = parser.parse_args()

    if args.smoke:
        seeds = [42, 43]
        counts = [100, 500]
    else:
        seeds = list(range(42, 52))
        counts = [100, 500, 1000, 5000]

    run_experiment(seeds, particle_counts=counts, num_steps=args.steps)


if __name__ == "__main__":
    main()
