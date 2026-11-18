"""Core experiment harness and baseline runner for Monte Carlo Localisation.

Provides reusable simulation runners, metric aggregators, and plotting functions
used by all specialized experiment scripts.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.world import World
from src.robot import Robot
from src.lidar import Lidar
from src.mcl import ParticleFilter
from src import metrics


def get_scripted_path(num_steps: int = 100) -> list[tuple[float, float]]:
    """Return standard collision-free scripted trajectory controls."""
    controls = [(0.4, -0.05)] * 40 + [(0.4, 0.0)] * 60
    return controls[:num_steps]


def run_trial(
    seed: int,
    filter_factory: Callable[[World, Lidar], Any],
    robot_factory: Callable[[], Robot] | None = None,
    lidar_factory: Callable[[World], Lidar] | None = None,
    num_steps: int = 100,
    conv_threshold: float = 0.3,
    conv_stable_steps: int = 10,
    on_step: Callable[[int, Robot, Any], None] | None = None,
) -> dict[str, Any]:
    """Execute a single simulation trial and return performance metrics."""
    world = World()
    robot = robot_factory() if robot_factory else Robot(seed=seed)
    lidar = lidar_factory(world) if lidar_factory else Lidar(world, seed=seed)
    pf = filter_factory(world, lidar)

    controls = get_scripted_path(num_steps)
    true_poses = []
    est_poses = []
    pos_errors = []
    head_errors = []

    t0 = time.perf_counter()
    for step_i, control in enumerate(controls):
        if on_step is not None:
            on_step(step_i, robot, pf)

        true_pose, odom = robot.step(control, dt=0.1)
        scan = lidar.scan(true_pose)

        if hasattr(pf, "adaptive_step"):
            est_pose, _ = pf.adaptive_step(odom, scan)
        else:
            est_pose = pf.step(odom, scan)

        pe = metrics.position_error(true_pose, est_pose)
        he = metrics.heading_error(true_pose[2], est_pose[2])

        true_poses.append(true_pose)
        est_poses.append(est_pose)
        pos_errors.append(pe)
        head_errors.append(he)

    runtime = time.perf_counter() - t0
    pos_err_arr = np.asarray(pos_errors, dtype=float)
    head_err_arr = np.asarray(head_errors, dtype=float)

    pos_rmse = metrics.rmse(pos_err_arr)
    head_rmse = metrics.rmse(head_err_arr)
    final_pos_err = float(pos_err_arr[-1])
    final_head_err = float(head_err_arr[-1])
    conv_time = metrics.convergence_time(pos_err_arr, threshold=conv_threshold, stable_steps=conv_stable_steps)
    success = bool(conv_time is not None and final_pos_err <= conv_threshold)

    return {
        "seed": seed,
        "position_rmse": pos_rmse,
        "heading_rmse": head_rmse,
        "final_position_error": final_pos_err,
        "final_heading_error": final_head_err,
        "convergence_time": conv_time,
        "success": success,
        "runtime": runtime,
        "true_poses": np.asarray(true_poses),
        "est_poses": np.asarray(est_poses),
        "position_errors": pos_err_arr,
        "heading_errors": head_err_arr,
    }


def save_summary_csv(filepath: Path | str, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Save scalar experiment records to CSV."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {}
            for k in fieldnames:
                val = r.get(k)
                if isinstance(val, float):
                    row[k] = f"{val:.4f}"
                elif val is None:
                    row[k] = "None"
                else:
                    row[k] = str(val)
            writer.writerow(row)
    print(f"  [✓] Saved CSV: {p}")


def print_grouped_summary(df: pd.DataFrame, group_col: str) -> None:
    """Print grouped means and standard deviations."""
    print("\n" + "=" * 88)
    print(f"Grouped Summary by '{group_col}'")
    print("=" * 88)
    header = (
        f"{group_col:<20} | {'Runs':<4} | {'Success':<8} | "
        f"{'Pos RMSE (m)':<17} | {'Final Err (m)':<17} | {'Mean Runtime':<12}"
    )
    print(header)
    print("-" * 88)

    for val, group in df.groupby(group_col):
        n_runs = len(group)
        succ = (group["success"].astype(bool).mean()) * 100.0
        p_rmse_mean, p_rmse_std = group["position_rmse"].mean(), group["position_rmse"].std(ddof=0)
        f_err_mean, f_err_std = group["final_position_error"].mean(), group["final_position_error"].std(ddof=0)
        rt_mean, rt_std = group["runtime"].mean(), group["runtime"].std(ddof=0)

        print(
            f"{str(val):<20} | "
            f"{n_runs:<4d} | "
            f"{succ:6.1f}%  | "
            f"{p_rmse_mean:5.3f} ± {p_rmse_std:5.3f}    | "
            f"{f_err_mean:5.3f} ± {f_err_std:5.3f}    | "
            f"{rt_mean:5.3f} ± {rt_std:5.3f}s"
        )
    print("=" * 88 + "\n")


def run_baseline_experiment(seeds: list[int] = [42], num_steps: int = 100) -> list[dict[str, Any]]:
    """Run baseline configuration: 500 particles, beam model, systematic resampling."""
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for seed in seeds:
        res = run_trial(
            seed=seed,
            filter_factory=lambda w, l: ParticleFilter(w, l, n=500, seed=seed, model="beam", resampling="systematic"),
            num_steps=num_steps,
        )
        res["sensor_model"] = "beam"
        res["resampling"] = "systematic"
        res["n_particles"] = 500
        records.append(res)

        # Plot baseline trajectory and error for the first seed
        if seed == seeds[0]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(res["position_errors"], color="#1f77b4", lw=1.5, label="Position Error (m)")
            ax.axhline(0.3, color="#d62728", ls="--", label="Threshold (0.3m)")
            ax.set_title(f"Baseline MCL Position Error vs Step (Seed {seed})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Step")
            ax.set_ylabel("Error (m)")
            ax.grid(True, ls=":", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures_dir / "baseline_position_error.png", dpi=150)
            plt.close(fig)

    fields = [
        "seed", "sensor_model", "resampling", "n_particles",
        "position_rmse", "heading_rmse", "final_position_error",
        "final_heading_error", "convergence_time", "success", "runtime"
    ]
    save_summary_csv(tables_dir / "baseline_summary.csv", records, fields)
    return records


if __name__ == "__main__":
    print("Running Baseline Experiment...")
    recs = run_baseline_experiment(seeds=[42])
    print(f"Baseline position RMSE: {recs[0]['position_rmse']:.4f} m, Success: {recs[0]['success']}")
