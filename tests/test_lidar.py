# Name - GARVIT

"""Tests for the Lidar sensor module.

Name - GARVIT
"""

import numpy as np
import pytest

from src.world import World
from src.lidar import Lidar


@pytest.fixture
def world():
    return World()


@pytest.fixture
def lidar(world):
    return Lidar(world, num_beams=60, seed=10)


class TestLidarConstruction:
    def test_default_beam_count(self, lidar):
        assert lidar.num_beams == 60
        assert lidar.angles.shape == (60,)

    def test_custom_beam_count(self, world):
        lid = Lidar(world, num_beams=12, seed=10)
        assert lid.num_beams == 12
        assert lid.angles.shape == (12,)

    def test_angles_span_full_circle(self, lidar):
        assert lidar.angles[0] == pytest.approx(-np.pi, abs=0.2)
        assert lidar.angles[-1] < np.pi

    def test_invalid_beam_count(self, world):
        with pytest.raises(ValueError):
            Lidar(world, num_beams=0)

    def test_invalid_max_range(self, world):
        with pytest.raises(ValueError):
            Lidar(world, max_range=-1.0)

    def test_invalid_noise_std(self, world):
        with pytest.raises(ValueError):
            Lidar(world, noise_std=-0.1)


class TestLidarScan:
    def test_scan_shape(self, lidar):
        scan = lidar.scan(np.array([1.5, 1.2, 0.0]))
        assert scan.shape == (60,)

    def test_scan_within_range(self, lidar):
        scan = lidar.scan(np.array([1.5, 1.2, 0.0]))
        assert np.all(scan >= 0.0)
        assert np.all(scan <= lidar.max_range)

    def test_noiseless_scan_is_deterministic(self, world):
        lid = Lidar(world, seed=10)
        pose = np.array([1.5, 1.2, 0.0])
        s1 = lid.scan(pose, noisy=False)
        s2 = lid.scan(pose, noisy=False)
        np.testing.assert_array_equal(s1, s2)

    def test_noisy_scan_varies(self, world):
        lid = Lidar(world, seed=10)
        pose = np.array([1.5, 1.2, 0.0])
        s1 = lid.scan(pose, noisy=True)
        s2 = lid.scan(pose, noisy=True)
        # Same RNG state has advanced, so scans differ.
        assert not np.allclose(s1, s2)

    def test_reproducibility(self, world):
        pose = np.array([1.5, 1.2, 0.0])
        lid1 = Lidar(world, seed=42)
        s1 = lid1.scan(pose)
        lid2 = Lidar(world, seed=42)
        s2 = lid2.scan(pose)
        np.testing.assert_array_equal(s1, s2)

    def test_invalid_pose_shape(self, lidar):
        with pytest.raises(ValueError):
            lidar.scan(np.array([1.0, 2.0]))

    def test_finite_values(self, lidar):
        scan = lidar.scan(np.array([6.0, 5.0, 0.0]))
        assert np.all(np.isfinite(scan))
