"""Interactive/Headless live demo for Monte Carlo Localisation.

Supports:
  - Seed selection
  - Particle count
  - Sensor model ('beam' or 'likelihood_field')
  - Resampling ('systematic' or 'multinomial')
  - Adaptive KLD mode
  - Augmented MCL mode
  - Kidnapped-robot mode
  - Headless/no-show mode
  - Saving figure/animation frames
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.world import World
from src.robot import Robot
from src.lidar import Lidar
from src.mcl import ParticleFilter
from src.adaptive import AdaptiveParticleFilter, AugmentedParticleFilter
from src.metrics import position_error, heading_error


def run_live_demo(
    seed: int = 42,
    n_particles: int = 500,
    model: str = "beam",
    resampling: str = "systematic",
    adaptive: bool = False,
    augmented: bool = False,
    kidnap: bool = False,
    kidnap_step: int = 50,
    kidnap_pose: tuple[float, float, float] = (8.0, 2.0, 0.0),
    num_steps: int = 100,
    show: bool = True,
    output_path: str | None = None,
) -> None:
    if not show:
        matplotlib.use("Agg")

    world = World()
    robot = Robot(seed=seed)
    lidar = Lidar(world, seed=seed)

    if adaptive:
        pf = AdaptiveParticleFilter(
            world=world,
            lidar=lidar,
            n_min=100,
            n_max=n_particles,
            seed=seed,
            model=model,
            resampling=resampling,
        )
    elif augmented:
        pf = AugmentedParticleFilter(
            world=world,
            lidar=lidar,
            n=n_particles,
            seed=seed,
            model=model,
            resampling=resampling,
            inject_ratio=0.08,
        )
    else:
        pf = ParticleFilter(
            world=world,
            lidar=lidar,
            n=n_particles,
            seed=seed,
            model=model,
            resampling=resampling,
        )

    controls: list[tuple[float, float]] = (
        [(0.4, -0.05)] * 40 +
        [(0.4, 0.0)] * 60
    )[:num_steps]

    fig, (ax_map, ax_err) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # ---- Map axes ----
    ax_map.imshow(
        world.grid,
        cmap="gray_r",
        origin="lower",
        extent=[0, world.width, 0, world.height],
        alpha=0.7,
    )
    ax_map.set_xlabel("x (m)")
    ax_map.set_ylabel("y (m)")
    ax_map.set_xlim(0, world.width)
    ax_map.set_ylim(0, world.height)

    particles_sc = None
    true_traj_line, = ax_map.plot([], [], "g-", lw=1.8, label="True Trajectory")
    est_traj_line, = ax_map.plot([], [], "r--", lw=1.8, label="Estimated Trajectory")
    true_pose_marker, = ax_map.plot([], [], "go", ms=8, label="True Pose")
    est_pose_marker, = ax_map.plot([], [], "rs", ms=8, label="Est Pose")
    ax_map.legend(loc="upper right", fontsize=8)

    # ---- Error axes ----
    ax_err.set_xlabel("Step")
    ax_err.set_ylabel("Error")
    pos_err_line, = ax_err.plot([], [], "b-", lw=1.5, label="Pos Err (m)")
    head_err_line, = ax_err.plot([], [], "m-", lw=1.5, label="Head Err (rad)")
    ax_err.axhline(0.3, color="gray", linestyle="--", alpha=0.6, label="Threshold 0.3m")
    ax_err.legend(loc="upper right", fontsize=8)
    ax_err.grid(True, ls=":", alpha=0.5)

    true_xs, true_ys = [], []
    est_xs, est_ys = [], []
    pos_errors, head_errors = [], []
    steps = []

    for step_i, (v, w) in enumerate(controls):
        # Kidnap simulation if enabled
        if kidnap and step_i == kidnap_step:
            print(f"  [!] Robot kidnapped at step {step_i} to {kidnap_pose}!")
            robot.pose = np.array(kidnap_pose, dtype=float)

        true_pose, odom = robot.step((v, w), dt=0.1)
        scan = lidar.scan(true_pose)

        if hasattr(pf, "adaptive_step"):
            est_pose, cur_n = pf.adaptive_step(odom, scan)
        else:
            est_pose = pf.step(odom, scan)
            cur_n = pf.n

        pe = position_error(true_pose, est_pose)
        he = abs(heading_error(true_pose[2], est_pose[2]))

        true_xs.append(float(true_pose[0]))
        true_ys.append(float(true_pose[1]))
        est_xs.append(float(est_pose[0]))
        est_ys.append(float(est_pose[1]))
        pos_errors.append(pe)
        head_errors.append(he)
        steps.append(step_i)

        if particles_sc is not None:
            particles_sc.remove()
        particles_sc = ax_map.scatter(
            pf.particles[:, 0],
            pf.particles[:, 1],
            s=2, c="steelblue", alpha=0.35, zorder=3,
        )

        true_traj_line.set_data(true_xs, true_ys)
        est_traj_line.set_data(est_xs, est_ys)
        true_pose_marker.set_data([true_pose[0]], [true_pose[1]])
        est_pose_marker.set_data([est_pose[0]], [est_pose[1]])

        pos_err_line.set_data(steps, pos_errors)
        head_err_line.set_data(steps, head_errors)
        ax_err.set_xlim(0, max(step_i + 1, 10))
        max_y = max(max(pos_errors, default=1.0), max(head_errors, default=1.0)) * 1.2
        ax_err.set_ylim(0, max(max_y, 0.5))

        title = (
            f"Step {step_i+1}/{num_steps} | Pos Err: {pe:.3f} m | Particles: {cur_n} | "
            f"Mode: {'Adaptive' if adaptive else ('Augmented' if augmented else 'Standard')}"
        )
        ax_map.set_title(title, fontsize=9)

        if show:
            plt.pause(0.01)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=150, bbox_inches="tight")
        print(f"  Saved figure to {out_p}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCL live visualisation demo.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--particles", "--particle-count", type=int, default=500, dest="particles", help="Particle count.")
    parser.add_argument("--model", "--sensor-model", type=str, default="beam", choices=["beam", "likelihood_field"], dest="model", help="Sensor model.")
    parser.add_argument("--resampling", type=str, default="systematic", choices=["systematic", "multinomial"], help="Resampling method.")
    parser.add_argument("--adaptive", action="store_true", help="Enable KLD adaptive sampling.")
    parser.add_argument("--augmented", action="store_true", help="Enable augmented MCL.")
    parser.add_argument("--kidnap", "--kidnapped", action="store_true", dest="kidnap", help="Simulate kidnapped robot at step 50.")
    parser.add_argument("--no-show", "--headless", action="store_true", dest="headless", help="Run in headless mode without GUI.")
    parser.add_argument("--output", "--output-path", type=str, default=None, dest="output", help="Path to save final frame PNG.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps.")
    args = parser.parse_args()

    run_live_demo(
        seed=args.seed,
        n_particles=args.particles,
        model=args.model,
        resampling=args.resampling,
        adaptive=args.adaptive,
        augmented=args.augmented,
        kidnap=args.kidnap,
        num_steps=args.steps,
        show=not args.headless,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
