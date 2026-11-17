# Name - GARVIT

"""Simulated planar lidar sensor for the indoor robot.

Name - GARVIT

The lidar casts *num_beams* rays equally spaced over a full 360° sweep
using the world's ray-caster and optionally adds Gaussian range noise.
"""

from __future__ import annotations

import numpy as np

from .world import World


class Lidar:
    """Configurable planar lidar scanner.

    Parameters
    ----------
    world:
        The known occupancy-grid world.
    num_beams:
        Number of evenly spaced beams (default 60).
    max_range:
        Maximum detectable range in metres (default 8.0).
    noise_std:
        Standard deviation of additive Gaussian range noise (default 0.05 m).
    seed:
        RNG seed for reproducible noise.

    Attributes
    ----------
    angles : np.ndarray
        Relative beam angles in radians, shape ``(num_beams,)``.
    """

    def __init__(
        self,
        world: World,
        num_beams: int = 60,
        max_range: float = 8.0,
        noise_std: float = 0.05,
        seed: int = 2,
    ) -> None:
        if num_beams < 1:
            raise ValueError("num_beams must be >= 1")
        if max_range <= 0.0:
            raise ValueError("max_range must be > 0")
        if noise_std < 0.0:
            raise ValueError("noise_std must be >= 0")

        self.world = world
        self.num_beams = num_beams
        self.max_range = max_range
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        # Evenly spaced beam angles over [-π, π).
        self.angles: np.ndarray = np.linspace(
            -np.pi, np.pi, num_beams, endpoint=False,
        )

    def scan(
        self,
        pose: np.ndarray,
        noisy: bool = True,
    ) -> np.ndarray:
        """Return lidar range measurements from the given pose.

        Parameters
        ----------
        pose:
            Robot pose ``(x, y, theta)``, shape ``(3,)``.
        noisy:
            If ``True`` (default), add Gaussian noise and clip to
            ``[0, max_range]``.

        Returns
        -------
        np.ndarray
            Range array of shape ``(num_beams,)``.
        """
        pose_arr = np.asarray(pose, dtype=float)
        if pose_arr.shape != (3,):
            raise ValueError("pose must have shape (3,)")

        ranges: np.ndarray = self.world.ray_cast(
            pose_arr, self.angles, self.max_range,
        )

        if noisy and self.noise_std > 0.0:
            noise = self.rng.normal(
                loc=0.0,
                scale=self.noise_std,
                size=ranges.shape,
            )
            ranges = ranges + noise

        # Clip to valid range.
        ranges = np.clip(ranges, 0.0, self.max_range)
        return ranges
