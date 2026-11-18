"""Live Matplotlib visualisation for Monte Carlo Localisation.

Displays the occupancy-grid map, particle cloud, true and estimated
robot poses, trajectories, lidar rays, and running error metrics.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

from .metrics import position_error, heading_error


def run_live_visualisation(
    world,
    robot,
    lidar,
    particle_filter,
    controls: list[tuple[float, float]],
    dt: float = 0.1,
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Run a full MCL simulation with animated Matplotlib visualisation.

    Parameters
    ----------
    world:
        ``World`` instance.
    robot:
        ``Robot`` instance.
    lidar:
        ``Lidar`` instance.
    particle_filter:
        ``ParticleFilter`` (or subclass) instance.
    controls:
        List of ``(linear_velocity, angular_velocity)`` commands.
    dt:
        Time step in seconds.
    show:
        If ``True``, call ``plt.show()`` and ``plt.pause()`` for
        interactive display.
    save_path:
        If set, save the final frame as an image.
    """
    if not show:
        matplotlib.use("Agg")

    fig, (ax_map, ax_err) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # ---- Map axes --------------------------------------------------- #
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

    # Dynamic artists.
    (particles_sc,) = (None,)
    true_traj_line, = ax_map.plot([], [], "g-", lw=1.5, label="True")
    est_traj_line, = ax_map.plot([], [], "r--", lw=1.5, label="Estimated")
    ax_map.legend(loc="upper right", fontsize=8)

    # ---- Error axes ------------------------------------------------- #
    ax_err.set_xlabel("Step")
    ax_err.set_ylabel("Error")
    pos_err_line, = ax_err.plot([], [], "b-", label="Pos (m)")
    head_err_line, = ax_err.plot([], [], "m-", label="Head (rad)")
    ax_err.legend(loc="upper right", fontsize=8)
    ax_err.grid(True, ls=":", alpha=0.5)

    # ---- Histories -------------------------------------------------- #
    true_xs: list[float] = []
    true_ys: list[float] = []
    est_xs: list[float] = []
    est_ys: list[float] = []
    pos_errors: list[float] = []
    head_errors: list[float] = []
    steps: list[int] = []

    for step_i, (v, w) in enumerate(controls):
        true_pose, odom = robot.step((v, w), dt=dt)
        scan = lidar.scan(true_pose)

        # Support adaptive step if available.
        if hasattr(particle_filter, "adaptive_step"):
            est_pose, _ = particle_filter.adaptive_step(odom, scan)
        else:
            est_pose = particle_filter.step(odom, scan)

        true_xs.append(float(true_pose[0]))
        true_ys.append(float(true_pose[1]))
        est_xs.append(float(est_pose[0]))
        est_ys.append(float(est_pose[1]))

        pe = position_error(true_pose, est_pose)
        he = heading_error(true_pose[2], est_pose[2])
        pos_errors.append(pe)
        head_errors.append(abs(he))
        steps.append(step_i)

        # -- Update map ------------------------------------------------ #
        # Remove old particle scatter.
        if particles_sc is not None:
            particles_sc.remove()
        particles_sc = ax_map.scatter(
            particle_filter.particles[:, 0],
            particle_filter.particles[:, 1],
            s=1, c="steelblue", alpha=0.3,
        )

        # Trajectories.
        true_traj_line.set_data(true_xs, true_ys)
        est_traj_line.set_data(est_xs, est_ys)

        # -- Update error plot ----------------------------------------- #
        pos_err_line.set_data(steps, pos_errors)
        head_err_line.set_data(steps, head_errors)
        ax_err.set_xlim(0, max(step_i + 1, 10))
        ax_err.set_ylim(0, max(max(pos_errors, default=1), max(head_errors, default=1)) * 1.2)

        ax_map.set_title(
            f"Step {step_i}  |  Pos err: {pe:.3f} m  |  "
            f"Particles: {particle_filter.n}",
            fontsize=10,
        )

        if show:
            plt.pause(0.01)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
