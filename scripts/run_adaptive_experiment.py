"""Experiment comparing Fixed 500 Particles vs Adaptive KLD-MCL."""

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
from src.adaptive import AdaptiveParticleFilter


def run_experiment(seeds: list[int], num_steps: int = 100) -> pd.DataFrame:
    modes = ["fixed_500", "adaptive_kld"]
    labels = {"fixed_500": "Fixed (500)", "adaptive_kld": "Adaptive KLD"}
    colors = {"fixed_500": "#2b5c8f", "adaptive_kld": "#9467bd"}
    records = []

    print(f"Running Adaptive MCL Experiment on {len(seeds)} seeds across {modes}...")
    for mode in modes:
        for seed in seeds:
            if mode == "fixed_500":
                factory = lambda w, l, s=seed: ParticleFilter(
                    w, l, n=500, seed=s, model="beam", resampling="systematic"
                )
            else:
                factory = lambda w, l, s=seed: AdaptiveParticleFilter(
                    w, l, n_min=100, n_max=1000, seed=s, model="beam", resampling="systematic"
                )

            res = run_trial(
                seed=seed,
                filter_factory=factory,
                num_steps=num_steps,
            )
            res["mode"] = mode
            records.append(res)

    df = pd.DataFrame(records)
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "seed", "mode", "position_rmse", "heading_rmse",
        "final_position_error", "final_heading_error", "convergence_time",
        "success", "runtime"
    ]
    save_summary_csv(tables_dir / "adaptive_comparison.csv", records, fields)

    # 1. RMSE Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(seeds))
    width = 0.35
    for i, m in enumerate(modes):
        sub = df[df["mode"] == m]
        ax.bar(x + (i - 0.5) * width, sub["position_rmse"], width, label=labels[m], color=colors[m], alpha=0.85)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Fixed vs Adaptive KLD: Position RMSE", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.grid(True, ls=":", alpha=0.5, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "adaptive_rmse.png", dpi=150)
    plt.close(fig)

    # 2. Runtime
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(modes):
        sub = df[df["mode"] == m]
        ax.bar(x + (i - 0.5) * width, sub["runtime"], width, label=labels[m], color=colors[m], alpha=0.85)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Fixed vs Adaptive KLD: Runtime", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.grid(True, ls=":", alpha=0.5, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "adaptive_runtime.png", dpi=150)
    plt.close(fig)

    # 3. Success Rate
    fig, ax = plt.subplots(figsize=(5, 4))
    rates = [df[df["mode"] == m]["success"].mean() * 100.0 for m in modes]
    ax.bar([labels[m] for m in modes], rates, color=[colors[m] for m in modes], width=0.4)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Fixed vs Adaptive Success Rate", fontweight="bold")
    for idx, r in enumerate(rates):
        ax.text(idx, r + 2, f"{r:.1f}%", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "adaptive_success_rate.png", dpi=150)
    plt.close(fig)

    print_grouped_summary(df, "mode")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive experiment.")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test (3 seeds).")
    parser.add_argument("--steps", type=int, default=100, help="Steps per trial.")
    args = parser.parse_args()

    seeds = [42, 43, 44] if args.smoke else list(range(42, 52))
    run_experiment(seeds, num_steps=args.steps)


if __name__ == "__main__":
    main()
