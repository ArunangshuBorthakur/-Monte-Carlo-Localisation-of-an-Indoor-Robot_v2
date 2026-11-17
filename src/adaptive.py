"""Adaptive and augmented MCL extensions.

Provides two ``ParticleFilter`` subclasses:

1. **AdaptiveParticleFilter** — KLD-sampling that adjusts particle count
   based on the spread of the posterior in a 3-D histogram.

2. **AugmentedParticleFilter** — Injects a fraction of random free-space
   particles every step to allow recovery from kidnapping.
"""

from __future__ import annotations

import numpy as np

from .mcl import ParticleFilter
from .robot import wrap_angle


# ------------------------------------------------------------------ #
# Adaptive KLD-sampling                                               #
# ------------------------------------------------------------------ #

class AdaptiveParticleFilter(ParticleFilter):
    """KLD-adaptive Monte Carlo Localisation.

    After each resampling step the particle count is adjusted so that
    the KL divergence between the true posterior and the sample-based
    approximation stays below *epsilon* with probability *1 − delta*.

    Parameters
    ----------
    world, lidar, seed, model, resampling:
        Forwarded to ``ParticleFilter``.
    n_min:
        Minimum number of particles.
    n_max:
        Maximum number of particles.
    epsilon:
        KLD upper bound.
    delta:
        Confidence (1 − delta) for the bound.
    xy_bin:
        Spatial bin width in metres.
    theta_bin:
        Angular bin width in radians.
    """

    def __init__(
        self,
        world,
        lidar,
        n_min: int = 100,
        n_max: int = 2000,
        seed: int = 3,
        model: str = "beam",
        resampling: str = "systematic",
        epsilon: float = 0.05,
        delta: float = 0.01,
        xy_bin: float = 0.5,
        theta_bin: float = 0.2,
    ) -> None:
        # Initialise with n_max so the first step has enough particles.
        super().__init__(
            world, lidar, n=n_max, seed=seed,
            model=model, resampling=resampling,
        )
        self.n_min = max(1, n_min)
        self.n_max = max(n_min, n_max)
        self.epsilon = epsilon
        self.delta = delta
        self.xy_bin = xy_bin
        self.theta_bin = theta_bin

    # ----- KLD target ------------------------------------------------ #

    def _kld_bound(self, k: int) -> int:
        """Wilson–Hilferty approximation for the KLD bound.

        Parameters
        ----------
        k:
            Number of occupied bins.

        Returns
        -------
        int
            Target particle count *n*.
        """
        if k <= 1:
            return self.n_min

        from scipy.stats import norm

        z = norm.ppf(1.0 - self.delta)
        term = 1.0 - 2.0 / (9.0 * (k - 1)) + z * np.sqrt(
            2.0 / (9.0 * (k - 1))
        )
        n_target = int(np.ceil((k - 1) / (2.0 * self.epsilon) * term ** 3))
        return int(np.clip(n_target, self.n_min, self.n_max))

    def _count_bins(self) -> int:
        """Count distinct 3-D histogram bins occupied by the particles."""
        bx = np.floor(self.particles[:, 0] / self.xy_bin).astype(int)
        by = np.floor(self.particles[:, 1] / self.xy_bin).astype(int)
        bt = np.floor(
            (self.particles[:, 2] + np.pi) / self.theta_bin
        ).astype(int)
        keys = set(zip(bx.tolist(), by.tolist(), bt.tolist()))
        return len(keys)

    # ----- Adaptive resample ----------------------------------------- #

    def _adaptive_resample(self) -> int:
        """Resample and then resize the particle set via KLD.

        Returns the new particle count.
        """
        # Standard resample first.
        self.resample()

        k = self._count_bins()
        n_target = self._kld_bound(k)

        if n_target < self.n:
            # Shrink: keep first n_target.
            self.particles = self.particles[:n_target].copy()
            self.weights = np.full(n_target, 1.0 / n_target)
            self.n = n_target
        elif n_target > self.n:
            # Grow: add random free-space particles.
            n_new = n_target - self.n
            new_particles = self._initialise(n_new)
            self.particles = np.vstack([self.particles, new_particles])
            self.weights = np.full(n_target, 1.0 / n_target)
            self.n = n_target

        return self.n

    # ----- Public step ----------------------------------------------- #

    def adaptive_step(
        self,
        odometry: np.ndarray,
        scan: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """One adaptive MCL cycle.

        Returns
        -------
        est_pose:
            Estimated pose ``(x, y, θ)``, shape ``(3,)``.
        n_particles:
            Particle count after adaptive resizing.
        """
        self.predict(odometry)
        self.update(scan)
        est = self.estimate()
        n = self._adaptive_resample()
        return est, n


# ------------------------------------------------------------------ #
# Augmented MCL (kidnapped-robot recovery)                            #
# ------------------------------------------------------------------ #

class AugmentedParticleFilter(ParticleFilter):
    """MCL with random-particle injection for recovery from kidnapping.

    After each resample, a fraction ``inject_ratio`` of the particle
    set is replaced by random free-space samples.  This allows the
    filter to recover when the robot is suddenly teleported.

    Parameters
    ----------
    world, lidar, n, seed, model, resampling:
        Forwarded to ``ParticleFilter``.
    inject_ratio:
        Fraction of particles to replace each step (default 0.05 = 5 %).
    """

    def __init__(
        self,
        world,
        lidar,
        n: int = 500,
        seed: int = 3,
        model: str = "beam",
        resampling: str = "systematic",
        inject_ratio: float = 0.05,
    ) -> None:
        super().__init__(
            world, lidar, n=n, seed=seed,
            model=model, resampling=resampling,
        )
        if inject_ratio < 0.0 or inject_ratio > 1.0:
            raise ValueError("inject_ratio must be in [0, 1]")
        self.inject_ratio = inject_ratio

    def _inject_random_particles(self) -> None:
        """Replace a fraction of particles with fresh random samples."""
        if self.inject_ratio <= 0.0:
            return

        n_inject = max(1, int(self.inject_ratio * self.n))
        new_particles = self._initialise(n_inject)
        # Replace the worst-weighted particles (they were resampled
        # already so weights are uniform; just replace the tail).
        self.particles[-n_inject:] = new_particles

    def step(
        self,
        odometry: np.ndarray,
        scan: np.ndarray,
    ) -> np.ndarray:
        """One augmented MCL cycle.

        Returns
        -------
        np.ndarray
            Estimated pose ``(x, y, θ)``, shape ``(3,)``.
        """
        self.predict(odometry)
        self.update(scan)
        est = self.estimate()
        self.resample()
        self._inject_random_particles()
        return est
