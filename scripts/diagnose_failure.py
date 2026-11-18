"""Diagnostic script for investigating failure on specific random seeds.

Usage:
    python scripts/diagnose_failure.py --seed 43
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.world import World
from src.robot import Robot
from src.lidar import Lidar
from src.mcl import ParticleFilter
from src import metrics


def _pose_arrow(ax, pose, color: str, length: float = 0.4) -> None:
    x, y, theta = pose
    dx = length * math.cos(theta)
    dy = length * math.sin(theta)
    ax.annotate(
        "",
        xy=(x + dx, y + dy),
        xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
    )


def run_diagnosis(seed: int, num_steps: int = 100) -> None:
    world = World()
    robot = Robot(seed=seed)
    lidar = Lidar(world, seed=seed)
    pf = ParticleFilter(world, lidar, n=500, seed=seed, model="beam", resampling="systematic")

    controls = [(0.4, -0.05)] * 40 + [(0.4, 0.0)] * 60
    controls = controls[:num_steps]

    true_poses = []
    est_poses = []
    pos_errors = []

    t0 = time.perf_counter()
    for v, w in controls:
        true_pose, odom = robot.step((v, w), dt=0.1)
        scan = lidar.scan(true_pose)
        est_pose = pf.step(odom, scan)

        true_poses.append(true_pose.copy())
        est_poses.append(est_pose.copy())
        pos_errors.append(metrics.position_error(true_pose, est_pose))

    runtime = time.perf_counter() - t0
    pos_errors_arr = np.array(pos_errors)
    final_pos_err = float(pos_errors_arr[-1])
    conv_step = metrics.convergence_time(pos_errors_arr, threshold=0.3, stable_steps=10)
    final_heading_err = metrics.heading_error(true_poses[-1][2], est_poses[-1][2])

    true_arr = np.array(true_poses)
    est_arr = np.array(est_poses)
    final_particles = pf.particles.copy()
    true_final = true_poses[-1]
    est_final = est_poses[-1]

    # Figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        world.grid,
        cmap="gray_r",
        origin="lower",
        extent=[0, world.width, 0, world.height],
        alpha=0.7,
    )
    ax.scatter(
        final_particles[:, 0], final_particles[:, 1],
        s=4, c="steelblue", alpha=0.35, label="Particles (final)", zorder=3,
    )
    ax.plot(true_arr[:, 0], true_arr[:, 1], color="limegreen", lw=2, label="True Trajectory", zorder=4)
    ax.plot(est_arr[:, 0], est_arr[:, 1], color="tomato", lw=1.5, ls="--", label="Estimated Trajectory", zorder=4)

    ax.plot(true_final[0], true_final[1], marker="o", ms=10, color="limegreen", zorder=6,
            label=f"True final ({true_final[0]:.2f}, {true_final[1]:.2f})")
    _pose_arrow(ax, true_final, color="limegreen")

    ax.plot(est_final[0], est_final[1], marker="s", ms=10, color="tomato", zorder=6,
            label=f"Est. final ({est_final[0]:.2f}, {est_final[1]:.2f})")
    _pose_arrow(ax, est_final, color="tomato")

    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    conv_label = f"step {conv_step}" if conv_step is not None else "None (FAILURE)"
    ax.set_title(
        f"MCL Failure Diagnosis — Seed {seed}\n"
        f"Final pos. error: {final_pos_err:.3f} m  |  Convergence: {conv_label}  |  Runtime: {runtime:.2f} s",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    out_dir = PROJECT_ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"failure_seed_{seed}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"  Seed            : {seed}")
    print(f"  Final pos. error: {final_pos_err:.3f} m")
    print(f"  Final head. err : {math.degrees(float(final_heading_err)):.1f}°")
    print(f"  Convergence     : {conv_label}")
    print(f"  Runtime         : {runtime:.2f} s")
    print(f"  True final pose : x={true_final[0]:.3f}, y={true_final[1]:.3f}, θ={math.degrees(float(true_final[2])):.1f}°")
    print(f"  Est. final pose : x={est_final[0]:.3f}, y={est_final[1]:.3f}, θ={math.degrees(float(est_final[2])):.1f}°")
    print(f"  Figure saved    : {out_path}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose MCL failure on specific seed.")
    parser.add_argument("--seed", type=int, default=43, help="RNG seed to diagnose.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps.")
    args = parser.parse_args()

    run_diagnosis(args.seed, num_steps=args.steps)


if __name__ == "__main__":
    main()
