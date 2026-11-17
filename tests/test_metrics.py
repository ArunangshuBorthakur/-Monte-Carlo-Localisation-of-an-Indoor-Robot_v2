"""Tests for localisation performance metrics (src/metrics.py)."""

import numpy as np
import pytest

from src.metrics import (
    position_error,
    heading_error,
    trajectory_errors,
    rmse,
    convergence_time,
    success_rate,
)


class TestPositionError:
    def test_scalar_position_error(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 1.0])
        err = position_error(p1, p2)
        assert isinstance(err, float)
        assert np.isclose(err, 5.0)

    def test_vectorised_position_error(self):
        p1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        p2 = np.array([[3.0, 4.0], [1.0, 4.0]])
        err = position_error(p1, p2)
        assert err.shape == (2,)
        assert np.isclose(err[0], 5.0)
        assert np.isclose(err[1], 3.0)


class TestHeadingError:
    def test_scalar_heading_error(self):
        err = heading_error(0.1, -0.1)
        assert np.isclose(err, 0.2)

    def test_wrapped_heading_error(self):
        # Difference across branch cut (e.g. +3.0 - (-3.0) = 6.0 -> wraps to 6 - 2pi approx -0.283)
        err = heading_error(3.0, -3.0)
        assert err >= -np.pi and err <= np.pi
        assert np.isclose(err, 3.0 - (-3.0) - 2.0 * np.pi)

    def test_vectorised_heading_error(self):
        t = np.array([0.0, np.pi])
        e = np.array([0.5, -np.pi])
        err = heading_error(t, e)
        assert err.shape == (2,)
        assert np.isclose(err[0], -0.5)


class TestTrajectoryErrors:
    def test_trajectory_errors_success(self):
        th = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]])
        eh = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.1]])
        res = trajectory_errors(th, eh)
        assert res.position_errors.shape == (2,)
        assert res.heading_errors.shape == (2,)
        assert np.isclose(res.position_errors[0], 0.0)
        assert np.isclose(res.position_errors[1], 1.0)

    def test_invalid_shapes(self):
        with pytest.raises(ValueError):
            trajectory_errors(np.zeros((5, 2)), np.zeros((5, 3)))
        with pytest.raises(ValueError):
            trajectory_errors(np.zeros((5, 3)), np.zeros((4, 3)))


class TestRmse:
    def test_rmse_values(self):
        errors = np.array([3.0, 4.0])
        assert np.isclose(rmse(errors), np.sqrt((9.0 + 16.0) / 2.0))

    def test_empty_rmse(self):
        assert rmse(np.array([])) == 0.0


class TestConvergenceTime:
    def test_converges_properly(self):
        errors = np.array([1.0, 0.5, 0.2, 0.1, 0.15, 0.25, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2])
        # threshold=0.3, stable_steps=5
        # Index 2 onwards is all <= 0.25 (< 0.3) for 5 steps!
        step = convergence_time(errors, threshold=0.3, stable_steps=5)
        assert step == 2

    def test_never_converges(self):
        errors = np.array([1.0, 0.8, 0.5, 0.4])
        assert convergence_time(errors, threshold=0.3, stable_steps=2) is None

    def test_empty_convergence(self):
        assert convergence_time(np.array([])) is None


class TestSuccessRate:
    def test_success_rate_calc(self):
        flags = [True, False, True, True]
        assert np.isclose(success_rate(flags), 0.75)

    def test_empty_flags(self):
        assert success_rate([]) == 0.0
