"""Lidar sensor likelihood models for Monte Carlo Localisation.

Provides two scoring functions:

1. **Beam model** — Gaussian on the difference between expected and
   observed ranges.  Fast because it operates entirely on pre-computed
   expected ranges.

2. **Likelihood-field model** — Projects each observed beam endpoint
   into the map and scores it by its distance to the nearest occupied
   cell using a precomputed Euclidean distance transform.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


# ------------------------------------------------------------------ #
# Beam model                                                          #
# ------------------------------------------------------------------ #

def beam_log_likelihood(
    expected: np.ndarray,
    observed: np.ndarray,
    sigma: float = 0.12,
) -> np.ndarray:
    """Gaussian beam-range log-likelihood.

    Parameters
    ----------
    expected:
        Expected ranges, shape ``(N, B)`` where *N* is the number of
        particles and *B* is the number of beams.
    observed:
        Observed (actual) ranges, shape ``(B,)``.
    sigma:
        Standard deviation of the Gaussian noise model (metres).

    Returns
    -------
    np.ndarray
        Log-likelihood per particle, shape ``(N,)``.
    """
    e = np.asarray(expected, dtype=float)
    o = np.asarray(observed, dtype=float)

    # Sum over beams: -0.5 * ((e - o) / σ)^2 − ln(σ √(2π))
    return np.sum(
        -0.5 * ((e - o) / sigma) ** 2
        - np.log(sigma * np.sqrt(2.0 * np.pi)),
        axis=-1,
    )


# ------------------------------------------------------------------ #
# Likelihood-field model                                              #
# ------------------------------------------------------------------ #

# Module-level cache so the distance field is built once per world
# identity (uses `id()` of the grid object as key).
_distance_field_cache: dict[int, np.ndarray] = {}


def _get_distance_field(world) -> np.ndarray:
    """Return the obstacle distance field (in metres), cached per world."""
    key = id(world.grid)
    if key not in _distance_field_cache:
        # distance_transform_edt needs "background" = True, so invert:
        # world.grid is True for occupied cells.
        # Passing ~grid gives distance-from-each-free-cell-to-nearest-occupied.
        dt_cells = distance_transform_edt(~world.grid)
        _distance_field_cache[key] = dt_cells * world.resolution
    return _distance_field_cache[key]


def likelihood_field_log_likelihood(
    world,
    particles: np.ndarray,
    observed: np.ndarray,
    angles: np.ndarray,
    sigma: float = 0.20,
) -> np.ndarray:
    """Likelihood-field log-likelihood using a precomputed distance transform.

    For every particle, each observed beam's endpoint is projected into
    world coordinates.  The distance from that endpoint to the nearest
    occupied cell is looked up in a precomputed Euclidean distance
    field.  A Gaussian ``p_hit ∝ exp(-0.5 (d / σ)²)`` scores the
    endpoint, and a small uniform ``p_rand`` provides numerical stability.

    Parameters
    ----------
    world:
        The ``World`` instance (provides ``.grid``, ``.resolution``,
        ``.width``, ``.height``).
    particles:
        Particle poses, shape ``(N, 3)``.
    observed:
        Observed lidar ranges, shape ``(B,)``.
    angles:
        Relative beam angles, shape ``(B,)``.
    sigma:
        Standard deviation of the Gaussian hit model (metres).

    Returns
    -------
    np.ndarray
        Log-likelihood per particle, shape ``(N,)``.
    """
    p = np.asarray(particles, dtype=float)   # (N, 3)
    obs = np.asarray(observed, dtype=float)  # (B,)
    ang = np.asarray(angles, dtype=float)    # (B,)

    # Pre-compute obstacle distance field.
    dist_field = _get_distance_field(world)  # (rows, cols) in metres

    max_range = float(np.max(obs)) if len(obs) > 0 else 8.0
    # Build a mask for beams at max range — treat them as uninformative.
    max_range_threshold = max_range - 0.01
    informative = obs < max_range_threshold  # (B,) bool

    # Beam endpoint world coordinates.
    # global_angle: (N, B)
    global_angle = p[:, 2, None] + ang[None, :]
    ex = p[:, 0, None] + obs[None, :] * np.cos(global_angle)  # (N, B)
    ey = p[:, 1, None] + obs[None, :] * np.sin(global_angle)  # (N, B)

    # Convert to grid indices.
    ix = np.round(ex / world.resolution).astype(int)
    iy = np.round(ey / world.resolution).astype(int)

    rows, cols = world.grid.shape
    valid = (
        (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    )  # (N, B)

    # Look up distances (default to a large penalty for out-of-map).
    d = np.full_like(ex, fill_value=3.0 * sigma)  # (N, B) penalty
    mask = valid
    d[mask] = dist_field[iy[mask], ix[mask]]

    # Gaussian hit probability (in log space).
    p_hit_log = -0.5 * (d / sigma) ** 2  # (N, B)

    # Small uniform random-measurement component for stability.
    p_rand = 1e-4
    # log( exp(p_hit_log) + p_rand ) ≈ p_hit_log when hit is good.
    # Use log-sum-exp trick for numerical safety.
    beam_log = np.log(np.exp(p_hit_log) + p_rand)  # (N, B)

    # Only accumulate informative beams (not at max range).
    beam_log[:, ~informative] = 0.0

    return np.sum(beam_log, axis=-1)  # (N,)
