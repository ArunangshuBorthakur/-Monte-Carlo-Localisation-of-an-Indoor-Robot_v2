"""Monte Carlo Localisation particle filter.

Implements the standard MCL algorithm:
    predict → update → estimate → resample
with support for beam and likelihood-field sensor models,
and systematic or multinomial resampling.
"""

from __future__ import annotations

import numpy as np

from .robot import wrap_angle
from .sensor_model import beam_log_likelihood, likelihood_field_log_likelihood


class ParticleFilter:
    """Standard Monte Carlo Localisation filter.

    Parameters
    ----------
    world:
        Known ``World`` instance.
    lidar:
        ``Lidar`` instance (provides ``angles`` and ``max_range``).
    n:
        Number of particles (default 500).
    seed:
        RNG seed for reproducibility.
    model:
        Sensor model — ``'beam'`` or ``'likelihood_field'``.
    resampling:
        Resampling strategy — ``'systematic'`` or ``'multinomial'``.
    """

    def __init__(
        self,
        world,
        lidar,
        n: int = 500,
        seed: int = 3,
        model: str = "beam",
        resampling: str = "systematic",
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        if model not in ("beam", "likelihood_field"):
            raise ValueError(
                f"Unknown sensor model '{model}'. "
                "Choose 'beam' or 'likelihood_field'."
            )
        if resampling not in ("systematic", "multinomial"):
            raise ValueError(
                f"Unknown resampling method '{resampling}'. "
                "Choose 'systematic' or 'multinomial'."
            )

        self.world = world
        self.lidar = lidar
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.model = model
        self.resampling = resampling

        self.particles: np.ndarray = self._initialise(n)
        self.weights: np.ndarray = np.full(n, 1.0 / n)

    # ---------------------------------------------------------------- #
    # Initialisation                                                    #
    # ---------------------------------------------------------------- #

    def _initialise(self, n: int) -> np.ndarray:
        """Sample *n* particles uniformly from free space."""
        out: list[np.ndarray] = []
        while len(out) < n:
            candidates = self.rng.uniform(
                [0.0, 0.0, -np.pi],
                [self.world.width, self.world.height, np.pi],
                size=(n, 3),
            )
            free = self.world.is_free(
                candidates[:, 0], candidates[:, 1],
            )
            out.extend(candidates[free])
        return np.asarray(out[:n], dtype=float)

    # ---------------------------------------------------------------- #
    # Predict                                                           #
    # ---------------------------------------------------------------- #

    def predict(
        self,
        odom: np.ndarray,
        noise: tuple[float, float, float] = (0.02, 0.02, 0.015),
    ) -> None:
        """Move particles by odometry with added Gaussian noise.

        Parameters
        ----------
        odom:
            Odometry delta ``(Δx, Δy, Δθ)``, shape ``(3,)``.
        noise:
            Standard deviations ``(σ_x, σ_y, σ_θ)`` for prediction noise.
        """
        d = np.asarray(odom, dtype=float)
        cos_t = np.cos(self.particles[:, 2])
        sin_t = np.sin(self.particles[:, 2])

        self.particles[:, 0] += d[0] * cos_t - d[1] * sin_t
        self.particles[:, 1] += d[0] * sin_t + d[1] * cos_t
        self.particles[:, 2] = wrap_angle(self.particles[:, 2] + d[2])

        # Add process noise.
        self.particles += self.rng.normal(
            0.0, noise, self.particles.shape,
        )
        self.particles[:, 2] = wrap_angle(self.particles[:, 2])

    # ---------------------------------------------------------------- #
    # Update                                                            #
    # ---------------------------------------------------------------- #

    def update(self, scan: np.ndarray) -> np.ndarray:
        """Weight particles by sensor likelihood.

        Parameters
        ----------
        scan:
            Observed lidar ranges, shape ``(B,)``.

        Returns
        -------
        np.ndarray
            Normalised weights, shape ``(N,)``.
        """
        if self.model == "beam":
            expected = self.world.ray_cast(
                self.particles, self.lidar.angles, self.lidar.max_range,
            )
            ll = beam_log_likelihood(expected, scan)
        else:
            ll = likelihood_field_log_likelihood(
                self.world,
                self.particles,
                scan,
                self.lidar.angles,
            )

        # Numerical stability: shift by max.
        ll -= np.max(ll)
        w = np.exp(ll)
        self.weights = w / (w.sum() + 1e-300)
        return self.weights

    # ---------------------------------------------------------------- #
    # Resample                                                          #
    # ---------------------------------------------------------------- #

    def resample(self) -> None:
        """Resample particles according to their weights."""
        c = np.cumsum(self.weights)
        c[-1] = 1.0  # guard against floating-point drift

        if self.resampling == "systematic":
            u0 = self.rng.random() / self.n
            positions = u0 + np.arange(self.n) / self.n
        else:  # multinomial
            positions = np.sort(self.rng.random(self.n))

        idx = np.searchsorted(c, positions)
        self.particles = self.particles[idx].copy()
        self.weights.fill(1.0 / self.n)

    # ---------------------------------------------------------------- #
    # Estimate                                                          #
    # ---------------------------------------------------------------- #

    def estimate(self) -> np.ndarray:
        """Return the weighted mean pose.

        Uses circular averaging for heading.

        Returns
        -------
        np.ndarray
            Estimated pose ``(x, y, θ)``, shape ``(3,)``.
        """
        xy = np.average(
            self.particles[:, :2],
            axis=0,
            weights=self.weights,
        )
        s = np.average(
            np.sin(self.particles[:, 2]),
            weights=self.weights,
        )
        c = np.average(
            np.cos(self.particles[:, 2]),
            weights=self.weights,
        )
        return np.array([xy[0], xy[1], np.arctan2(s, c)])

    # ---------------------------------------------------------------- #
    # Full step                                                         #
    # ---------------------------------------------------------------- #

    def step(
        self,
        odometry: np.ndarray,
        scan: np.ndarray,
    ) -> np.ndarray:
        """Execute one full predict-update-estimate-resample cycle.

        Returns
        -------
        np.ndarray
            Estimated pose ``(x, y, θ)``, shape ``(3,)``.
        """
        self.predict(odometry)
        self.update(scan)
        est = self.estimate()
        self.resample()
        return est
