"""Tests for Adaptive KLD and Augmented MCL filters (src/adaptive.py)."""

import numpy as np
import pytest

from src.world import World
from src.lidar import Lidar
from src.adaptive import AdaptiveParticleFilter, AugmentedParticleFilter


@pytest.fixture
def world():
    return World()


@pytest.fixture
def lidar(world):
    return Lidar(world, num_beams=60, seed=10)


class TestAdaptiveParticleFilter:
    def test_init_params(self, world, lidar):
        apf = AdaptiveParticleFilter(
            world, lidar, n_min=50, n_max=500, epsilon=0.05, delta=0.01, seed=42
        )
        assert apf.n == 500  # Starts with n_max
        assert apf.particles.shape == (500, 3)
        assert apf.weights.shape == (500,)

    def test_kld_bound_calculation(self, world, lidar):
        apf = AdaptiveParticleFilter(world, lidar, n_min=50, n_max=1000, seed=42)
        b1 = apf._kld_bound(k=1)
        assert b1 == 50
        b10 = apf._kld_bound(k=10)
        assert b10 > 50
        assert b10 <= 1000

    def test_adaptive_step_returns_pose_and_count(self, world, lidar):
        apf = AdaptiveParticleFilter(
            world, lidar, n_min=50, n_max=300, seed=42
        )
        scan = lidar.scan(np.array([1.5, 1.2, 0.0]))
        odom = np.array([0.1, 0.0, 0.0])
        est_pose, n_count = apf.adaptive_step(odom, scan)

        assert est_pose.shape == (3,)
        assert np.all(np.isfinite(est_pose))
        assert isinstance(n_count, int)
        assert 50 <= n_count <= 300
        assert apf.particles.shape == (n_count, 3)
        assert apf.weights.shape == (n_count,)

    def test_bin_counting(self, world, lidar):
        apf = AdaptiveParticleFilter(world, lidar, n_min=10, n_max=50, seed=42)
        # All particles at same location -> 1 bin
        apf.particles = np.zeros((20, 3))
        assert apf._count_bins() == 1


class TestAugmentedParticleFilter:
    def test_init(self, world, lidar):
        aug = AugmentedParticleFilter(world, lidar, n=100, inject_ratio=0.1, seed=42)
        assert aug.particles.shape == (100, 3)
        assert aug.inject_ratio == 0.1

    def test_invalid_inject_ratio(self, world, lidar):
        with pytest.raises(ValueError):
            AugmentedParticleFilter(world, lidar, inject_ratio=-0.1)
        with pytest.raises(ValueError):
            AugmentedParticleFilter(world, lidar, inject_ratio=1.5)

    def test_no_injection_when_zero(self, world, lidar):
        aug = AugmentedParticleFilter(world, lidar, n=50, inject_ratio=0.0, seed=42)
        # Fix particles
        aug.particles = np.full((50, 3), 1.5)
        aug.weights = np.full(50, 1.0 / 50)
        aug._inject_random_particles()
        assert np.allclose(aug.particles, 1.5)

    def test_injection_replaces_tail(self, world, lidar):
        aug = AugmentedParticleFilter(world, lidar, n=50, inject_ratio=0.1, seed=42)
        aug.particles = np.full((50, 3), 1.5)
        aug.weights = np.full(50, 1.0 / 50)
        aug._inject_random_particles()
        # Head should still be 1.5
        assert np.allclose(aug.particles[:40], 1.5)
        # Tail (n_inject = 5) should have random particles in free space
        tail = aug.particles[45:]
        assert len(tail) == 5
        assert np.all(world.is_free(tail[:, 0], tail[:, 1]))

    def test_step_runs_augmented_mcl(self, world, lidar):
        aug = AugmentedParticleFilter(world, lidar, n=100, inject_ratio=0.05, seed=42)
        scan = lidar.scan(np.array([1.5, 1.2, 0.0]))
        odom = np.array([0.1, 0.0, 0.0])
        est = aug.step(odom, scan)
        assert est.shape == (3,)
        assert np.all(np.isfinite(est))
        assert aug.particles.shape == (100, 3)
