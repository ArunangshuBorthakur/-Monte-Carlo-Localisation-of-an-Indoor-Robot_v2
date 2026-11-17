"""Tests for standard Monte Carlo Localisation filter (src/mcl.py)."""

import numpy as np
import pytest

from src.world import World
from src.lidar import Lidar
from src.mcl import ParticleFilter


@pytest.fixture
def world():
    return World()


@pytest.fixture
def lidar(world):
    return Lidar(world, num_beams=60, seed=10)


class TestParticleFilterInit:
    def test_default_init(self, world, lidar):
        pf = ParticleFilter(world, lidar, n=100, seed=42)
        assert pf.particles.shape == (100, 3)
        assert pf.weights.shape == (100,)
        assert np.isclose(np.sum(pf.weights), 1.0)
        assert np.allclose(pf.weights, 1.0 / 100)

    def test_particles_in_free_space(self, world, lidar):
        pf = ParticleFilter(world, lidar, n=200, seed=42)
        free = world.is_free(pf.particles[:, 0], pf.particles[:, 1])
        assert np.all(free)

    def test_invalid_n(self, world, lidar):
        with pytest.raises(ValueError):
            ParticleFilter(world, lidar, n=0)

    def test_invalid_model(self, world, lidar):
        with pytest.raises(ValueError):
            ParticleFilter(world, lidar, model="unknown_model")

    def test_invalid_resampling(self, world, lidar):
        with pytest.raises(ValueError):
            ParticleFilter(world, lidar, resampling="random_resampling")

    def test_reproducibility(self, world, lidar):
        pf1 = ParticleFilter(world, lidar, n=50, seed=123)
        pf2 = ParticleFilter(world, lidar, n=50, seed=123)
        np.testing.assert_array_equal(pf1.particles, pf2.particles)


class TestParticleFilterPredict:
    def test_predict_moves_particles(self, world, lidar):
        pf = ParticleFilter(world, lidar, n=50, seed=42)
        init_particles = pf.particles.copy()
        pf.predict(odom=np.array([0.5, 0.0, 0.1]), noise=(0.001, 0.001, 0.001))
        assert not np.allclose(pf.particles, init_particles)
        # Heading wrapped in [-pi, pi]
        assert np.all(pf.particles[:, 2] >= -np.pi)
        assert np.all(pf.particles[:, 2] <= np.pi)


class TestParticleFilterUpdate:
    def test_update_normalises_weights_beam(self, world, lidar):
        pf = ParticleFilter(world, lidar, n=50, seed=42, model="beam")
        scan = lidar.scan(np.array([1.5, 1.2, 0.0]))
        weights = pf.update(scan)
        assert weights.shape == (50,)
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(np.isfinite(weights))
        assert np.all(weights >= 0.0)

    def test_update_normalises_weights_likelihood_field(self, world, lidar):
        pf = ParticleFilter(world, lidar, n=50, seed=42, model="likelihood_field")
        scan = lidar.scan(np.array([1.5, 1.2, 0.0]))
        weights = pf.update(scan)
        assert weights.shape == (50,)
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(np.isfinite(weights))


class TestParticleFilterResample:
    def test_systematic_resampling(self, world, lidar):
        pf = ParticleFilter(world, lidar, n=50, seed=42, resampling="systematic")
        pf.weights = np.zeros(50)
        pf.weights[10] = 1.0  # Only particle 10 has non-zero weight
        pf.resample()
        assert pf.particles.shape == (50, 3)
        # All particles should be copies of particle 10 before resample
        assert np.all(pf.weights == 1.0 / 50)

    def test_multinomial_resampling(self, world, lidar):
        pf = ParticleFilter(world, lidar, n=50, seed=42, resampling="multinomial")
        pf.weights = np.zeros(50)
        pf.weights[5] = 1.0
        pf.resample()
        assert pf.particles.shape == (50, 3)
        assert np.all(pf.weights == 1.0 / 50)


class TestParticleFilterEstimate:
    def test_estimate_shape(self, world, lidar):
        pf = ParticleFilter(world, lidar, n=50, seed=42)
        est = pf.estimate()
        assert est.shape == (3,)
        assert np.all(np.isfinite(est))

    def test_circular_heading_average(self, world, lidar):
        pf = ParticleFilter(world, lidar, n=2, seed=42)
        # Place particles near +pi and -pi (e.g., +3.1 rad and -3.1 rad)
        # Their average heading across the branch cut should be around pi or -pi, not 0!
        pf.particles = np.array([
            [1.0, 1.0, 3.1],
            [1.0, 1.0, -3.1]
        ])
        pf.weights = np.array([0.5, 0.5])
        est = pf.estimate()
        assert np.isclose(abs(est[2]), np.pi, atol=0.1)


class TestParticleFilterStep:
    def test_step_runs_full_cycle(self, world, lidar):
        pf = ParticleFilter(world, lidar, n=100, seed=42)
        scan = lidar.scan(np.array([1.5, 1.2, 0.0]))
        odom = np.array([0.1, 0.0, 0.0])
        est = pf.step(odom, scan)
        assert est.shape == (3,)
        assert np.all(np.isfinite(est))
        assert pf.particles.shape == (100, 3)
