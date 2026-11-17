"""Localisation performance metrics.

All functions accept scalars or NumPy arrays and return the same type.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


# ------------------------------------------------------------------ #
# Error metrics                                                       #
# ------------------------------------------------------------------ #

def position_error(
    true_pose: np.ndarray,
    est_pose: np.ndarray,
) -> float | np.ndarray:
    """Euclidean position error in metres.

    Parameters
    ----------
    true_pose, est_pose:
        Poses with at least two elements ``[x, y, ...]``.
    """
    t = np.asarray(true_pose, dtype=float)
    e = np.asarray(est_pose, dtype=float)
    dx = t[..., 0] - e[..., 0]
    dy = t[..., 1] - e[..., 1]
    d = np.sqrt(dx ** 2 + dy ** 2)
    return float(d) if d.ndim == 0 else d


def heading_error(
    true_theta: float | np.ndarray,
    est_theta: float | np.ndarray,
) -> float | np.ndarray:
    """Heading error wrapped to ``[-π, π]``.

    Parameters
    ----------
    true_theta, est_theta:
        Heading angles in radians.
    """
    diff = np.asarray(true_theta) - np.asarray(est_theta)
    wrapped = (diff + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


# ------------------------------------------------------------------ #
# Trajectory metrics                                                  #
# ------------------------------------------------------------------ #

class TrajectoryErrors(NamedTuple):
    """Container for per-step error arrays."""
    position_errors: np.ndarray
    heading_errors: np.ndarray


def trajectory_errors(
    true_history: np.ndarray,
    est_history: np.ndarray,
) -> TrajectoryErrors:
    """Compute per-step position and heading errors.

    Parameters
    ----------
    true_history, est_history:
        Arrays of shape ``(T, 3)``.

    Returns
    -------
    TrajectoryErrors
        Named tuple with ``position_errors`` and ``heading_errors``,
        each of shape ``(T,)``.
    """
    t = np.asarray(true_history, dtype=float)
    e = np.asarray(est_history, dtype=float)
    if t.ndim != 2 or t.shape[1] < 3:
        raise ValueError("true_history must have shape (T, 3)")
    if e.ndim != 2 or e.shape[1] < 3:
        raise ValueError("est_history must have shape (T, 3)")
    if t.shape[0] != e.shape[0]:
        raise ValueError("true and estimated histories must have the same length")

    pe = position_error(t, e)
    he = heading_error(t[:, 2], e[:, 2])
    return TrajectoryErrors(pe, he)


def rmse(errors: np.ndarray) -> float:
    """Root mean square error."""
    arr = np.asarray(errors, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


# ------------------------------------------------------------------ #
# Convergence                                                         #
# ------------------------------------------------------------------ #

def convergence_time(
    position_errors: np.ndarray,
    threshold: float = 0.3,
    stable_steps: int = 10,
) -> int | None:
    """Find the first step where the error stays below *threshold*.

    Parameters
    ----------
    position_errors:
        1-D array of per-step position errors.
    threshold:
        Convergence threshold in metres.
    stable_steps:
        Number of consecutive steps the error must remain below
        the threshold.

    Returns
    -------
    int or None
        Step index of convergence, or ``None`` if never converged.
    """
    arr = np.asarray(position_errors, dtype=float)
    if arr.size == 0:
        return None

    below = arr < threshold
    for i in range(len(below) - stable_steps + 1):
        if np.all(below[i : i + stable_steps]):
            return int(i)
    return None


# ------------------------------------------------------------------ #
# Success rate                                                        #
# ------------------------------------------------------------------ #

def success_rate(flags: list | np.ndarray) -> float:
    """Fraction of ``True`` values.

    Parameters
    ----------
    flags:
        Boolean array or list.

    Returns
    -------
    float
        Success rate in ``[0, 1]``.
    """
    arr = np.asarray(flags, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))
