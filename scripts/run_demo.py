"""Terminal demo running Monte Carlo Localisation for 100 steps.

Runs the baseline configuration:
  - 500 particles
  - Beam sensor model
  - Systematic resampling
Prints progress and final localisation metrics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.world import World
from src.robot import Robot
from src.lidar import Lidar
from src.mcl import ParticleFilter
from src import metrics


def run_demo(
    seed: int = 42,
    num_steps: int = 100,
    n_particles: int = 500,
    model: str = "beam",
    resampling: str = "systematic",
) -> None:
    print("=" * 65)
    print("  Monte Carlo Localisation (MCL) Demo")
    print(f"  Seed: {seed} | Particles: {n_particles} | Model: {model} | Resampling: {resampling}")
    print("=" * 65)

    world = World()
    robot = Robot(seed=seed)
    lidar = Lidar(world, seed=seed)
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

    true_history = []
    est_history = []
    pos_errors = []

    t0 = time.perf_counter()

    for step_i, (v, w) in enumerate(controls):
        true_pose, odom = robot.step((v, w), dt=0.1)
        scan = lidar.scan(true_pose)
        est_pose = pf.step(odom, scan)

        pe = metrics.position_error(true_pose, est_pose)
        true_history.append(true_pose)
        est_history.append(est_pose)
        pos_errors.append(pe)

        if (step_i + 1) % 20 == 0 or step_i == 0:
            print(
                f"  Step {step_i + 1:3d}/{num_steps} | "
                f"True: [{true_pose[0]:.2f}, {true_pose[1]:.2f}, {true_pose[2]:.2f}] | "
                f"Est: [{est_pose[0]:.2f}, {est_pose[1]:.2f}, {est_pose[2]:.2f}] | "
                f"Pos Err: {pe:.3f} m"
            )

    runtime = time.perf_counter() - t0
    pos_errors_arr = np.array(pos_errors)
    conv_step = metrics.convergence_time(pos_errors_arr, threshold=0.3, stable_steps=10)
    pos_rmse = metrics.rmse(pos_errors_arr)

    print("-" * 65)
    print(f"  Simulation completed in {runtime:.3f} seconds.")
    print(f"  Position RMSE       : {pos_rmse:.3f} m")
    print(f"  Final Position Error: {pos_errors_arr[-1]:.3f} m")
    print(f"  Convergence Step    : {conv_step if conv_step is not None else 'None (did not converge)'}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCL terminal demo.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps.")
    parser.add_argument("--particles", type=int, default=500, help="Number of particles.")
    parser.add_argument("--model", type=str, default="beam", choices=["beam", "likelihood_field"], help="Sensor model.")
    parser.add_argument("--resampling", type=str, default="systematic", choices=["systematic", "multinomial"], help="Resampling method.")
    args = parser.parse_args()

    run_demo(
        seed=args.seed,
        num_steps=args.steps,
        n_particles=args.particles,
        model=args.model,
        resampling=args.resampling,
    )


if __name__ == "__main__":
    main()
