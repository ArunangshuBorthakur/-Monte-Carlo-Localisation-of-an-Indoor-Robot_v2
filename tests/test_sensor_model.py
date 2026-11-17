"""Tests for the sensor model module."""

import numpy as np
import pytest

from src.world import World
from src.lidar import Lidar
from src.sensor_model import (
    beam_log_likelihood,
    likelihood_field_log_likelihood,
)


@pytest.fixture
def world():
    return World()


@pytest.fixture
def lidar(world):
    return Lidar(world, num_beams=60, seed=10)


# ------------------------------------------------------------------ #
# Beam model tests                                                    #
# ------------------------------------------------------------------ #

class TestBeamModel:
    def test_output_shape(self):
        expected = np.random.rand(5, 10)
        observed = np.random.rand(10)
        ll = beam_log_likelihood(expected, observed)
        assert ll.shape == (5,)

    def test_finite_values(self):
        expected = np.ones((3, 4))
        observed = np.ones(4)
        ll = beam_log_likelihood(expected, observed)
        assert np.all(np.isfinite(ll))

    def test_perfect_match_scores_highest(self):
        observed = np.array([1.0, 2.0, 3.0])
        perfect = observed[None, :]          # (1, 3)
        bad = observed[None, :] + 2.0        # (1, 3)
        ll_good = beam_log_likelihood(perfect, observed)
        ll_bad = beam_log_likelihood(bad, observed)
        assert ll_good[0] > ll_bad[0]

    def test_single_particle(self):
        expected = np.array([[1.0, 2.0]])
        observed = np.array([1.0, 2.0])
        ll = beam_log_likelihood(expected, observed)
        assert ll.shape == (1,)


# ------------------------------------------------------------------ #
# Likelihood-field model tests                                        #
# ------------------------------------------------------------------ #

class TestLikelihoodFieldModel:
    def test_output_shape(self, world, lidar):
        pose = np.array([1.5, 1.2, 0.0])
        scan = lidar.scan(pose, noisy=False)
        particles = np.tile(pose, (5, 1))
        ll = likelihood_field_log_likelihood(
            world, particles, scan, lidar.angles,
        )
        assert ll.shape == (5,)

    def test_finite_values(self, world, lidar):
        pose = np.array([1.5, 1.2, 0.0])
        scan = lidar.scan(pose, noisy=False)
        particles = np.tile(pose, (10, 1))
        ll = likelihood_field_log_likelihood(
            world, particles, scan, lidar.angles,
        )
        assert np.all(np.isfinite(ll))

    def test_correct_particle_beats_incorrect(self, world, lidar):
        true_pose = np.array([1.5, 1.2, 0.2])
        scan = lidar.scan(true_pose, noisy=False)

        good_particle = true_pose[None, :]          # (1, 3)
        bad_particle = np.array([[8.0, 8.0, 0.0]])  # far away

        ll_good = likelihood_field_log_likelihood(
            world, good_particle, scan, lidar.angles,
        )
        ll_bad = likelihood_field_log_likelihood(
            world, bad_particle, scan, lidar.angles,
        )
        assert ll_good[0] > ll_bad[0]

    def test_out_of_map_endpoints_finite(self, world, lidar):
        """Endpoints that fall outside the map must not cause NaN/inf."""
        particles = np.array([
            [-5.0, -5.0, 0.0],
            [20.0, 20.0, 0.0],
        ])
        scan = np.ones(lidar.num_beams) * 5.0
        ll = likelihood_field_log_likelihood(
            world, particles, scan, lidar.angles,
        )
        assert ll.shape == (2,)
        assert np.all(np.isfinite(ll))

    def test_max_range_beams_ignored(self, world, lidar):
        """Max-range beams should be treated as uninformative."""
        pose = np.array([6.0, 5.0, 0.0])
        # All beams at max range.
        scan = np.full(lidar.num_beams, lidar.max_range)
        particles = np.tile(pose, (3, 1))
        ll = likelihood_field_log_likelihood(
            world, particles, scan, lidar.angles,
        )
        assert np.all(np.isfinite(ll))
        # All particles should score the same since all beams ignored.
        assert np.allclose(ll, ll[0])
