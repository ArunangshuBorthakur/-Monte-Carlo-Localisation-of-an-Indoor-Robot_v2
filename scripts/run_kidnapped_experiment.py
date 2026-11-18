"""Experiment comparing Standard MCL vs Augmented MCL during robot kidnapping."""

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
from src.adaptive import AugmentedParticleFilter


def run_experiment(seeds: list[int], num_steps: int = 100, kidnap_step: int = 45) -> pd.DataFrame:
    modes = ["standard_mcl", "augmented_mcl"]
    labels = {"standard_mcl": "Standard MCL", "augmented_mcl": "Augmented MCL"}
    colors = {"standard_mcl": "#d62728", "augmented_mcl": "#2ca02c"}
    records = []

    def kidnap_hook(step: int, robot, pf):
        if step == kidnap_step:
            robot.pose = np.array([8.0, 2.0, 0.0])

    print(f"Running Kidnapped Experiment on {len(seeds)} seeds across {modes}...")
    sample_histories = {}

    for mode in modes:
        for seed in seeds:
            if mode == "standard_mcl":
                factory = lambda w, l, s=seed: ParticleFilter(
                    w, l, n=500, seed=s, model="beam", resampling="systematic"
                )
            else:
                factory = lambda w, l, s=seed: AugmentedParticleFilter(
                    w, l, n=500, seed=s, model="beam", resampling="systematic", inject_ratio=0.08
                )

            res = run_trial(
                seed=seed,
                filter_factory=factory,
                num_steps=num_steps,
                on_step=kidnap_hook,
            )
            res["mode"] = mode
            records.append(res)

            if seed == seeds[0]:
                sample_histories[mode] = res["position_errors"]

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
    save_summary_csv(tables_dir / "kidnapped_comparison.csv", records, fields)

    # 1. Recovery error over time for Seed[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in modes:
        ax.plot(sample_histories[m], label=labels[m], color=colors[m], lw=1.8)
    ax.axvline(kidnap_step, color="black", ls="--", label=f"Kidnapped (Step {kidnap_step})")
    ax.axhline(0.3, color="gray", ls=":", label="Threshold (0.3m)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position Error (m)")
    ax.set_title(f"Kidnapped Robot Recovery: Position Error vs Step (Seed {seeds[0]})", fontweight="bold")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "kidnapped_recovery.png", dpi=150)
    plt.close(fig)

    # 2. Final Error
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(seeds))
    width = 0.35
    for i, m in enumerate(modes):
        sub = df[df["mode"] == m]
        ax.bar(x + (i - 0.5) * width, sub["final_position_error"], width, label=labels[m], color=colors[m], alpha=0.85)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Final Position Error (m)")
    ax.set_title("Kidnapped Robot: Final Position Error", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.grid(True, ls=":", alpha=0.5, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "kidnapped_rmse.png", dpi=150)
    plt.close(fig)

    # 3. Success Rate
    fig, ax = plt.subplots(figsize=(5, 4))
    rates = [df[df["mode"] == m]["success"].mean() * 100.0 for m in modes]
    ax.bar([labels[m] for m in modes], rates, color=[colors[m] for m in modes], width=0.4)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Kidnapped Recovery Success Rate", fontweight="bold")
    for idx, r in enumerate(rates):
        ax.text(idx, r + 2, f"{r:.1f}%", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "kidnapped_success_rate.png", dpi=150)
    plt.close(fig)

    print_grouped_summary(df, "mode")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run kidnapped robot experiment.")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test (3 seeds).")
    parser.add_argument("--steps", type=int, default=100, help="Steps per trial.")
    args = parser.parse_args()

    seeds = [42, 43, 44] if args.smoke else list(range(42, 52))
    run_experiment(seeds, num_steps=args.steps)


if __name__ == "__main__":
    main()
