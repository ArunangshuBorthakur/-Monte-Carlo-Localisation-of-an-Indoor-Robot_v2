"""Integration tests verifying full system communication through agreed interfaces."""

import numpy as np
import pytest

from src.world import World
from src.robot import Robot
from src.lidar import Lidar
from src.mcl import ParticleFilter
from src.adaptive import AdaptiveParticleFilter, AugmentedParticleFilter
from src import metrics


class TestMCLIntegration:
    def test_full_pipeline_step(self):
        # 1. Constructs World, Robot, Lidar, ParticleFilter
        world = World()
        robot = Robot(seed=42)
        lidar = Lidar(world, num_beams=60, seed=42)
        pf = ParticleFilter(world, lidar, n=100, seed=42, model="beam", resampling="systematic")

        # Initial pose checks
        assert robot.pose.shape == (3,)
        assert np.all(np.isfinite(robot.pose))
        assert pf.particles.shape == (100, 3)
        assert np.all(np.isfinite(pf.particles))

        # 2. Runs at least one complete simulation step
        control = (0.4, 0.05)
        dt = 0.1
        true_pose, odom = robot.step(control, dt=dt)
        scan = lidar.scan(true_pose)
        est_pose = pf.step(odom, scan)

        # 3. Verifies finite poses and estimates
        assert np.all(np.isfinite(true_pose))
        assert np.all(np.isfinite(odom))
        assert np.all(np.isfinite(scan))
        assert np.all(np.isfinite(est_pose))

        # 4. Verifies array shapes
        assert true_pose.shape == (3,)
        assert odom.shape == (3,)
        assert scan.shape == (60,)
        assert est_pose.shape == (3,)
        assert pf.particles.shape == (100, 3)
        assert pf.weights.shape == (100,)

        # 5. Verifies error metrics
        pe = metrics.position_error(true_pose, est_pose)
        he = metrics.heading_error(true_pose[2], est_pose[2])
        assert isinstance(pe, float)
        assert isinstance(he, float)
        assert np.isfinite(pe)
        assert np.isfinite(he)

    def test_likelihood_field_integration(self):
        world = World()
        robot = Robot(seed=43)
        lidar = Lidar(world, num_beams=30, seed=43)
        pf = ParticleFilter(world, lidar, n=50, seed=43, model="likelihood_field", resampling="multinomial")

        true_pose, odom = robot.step((0.3, 0.0), dt=0.1)
        scan = lidar.scan(true_pose)
        est_pose = pf.step(odom, scan)

        assert np.all(np.isfinite(est_pose))
        assert est_pose.shape == (3,)

    def test_adaptive_integration(self):
        world = World()
        robot = Robot(seed=44)
        lidar = Lidar(world, num_beams=30, seed=44)
        apf = AdaptiveParticleFilter(world, lidar, n_min=50, n_max=200, seed=44)

        true_pose, odom = robot.step((0.2, -0.05), dt=0.1)
        scan = lidar.scan(true_pose)
        est_pose, n_count = apf.adaptive_step(odom, scan)

        assert np.all(np.isfinite(est_pose))
        assert est_pose.shape == (3,)
        assert 50 <= n_count <= 200

    def test_augmented_integration(self):
        world = World()
        robot = Robot(seed=45)
        lidar = Lidar(world, num_beams=30, seed=45)
        aug = AugmentedParticleFilter(world, lidar, n=80, inject_ratio=0.05, seed=45)

        true_pose, odom = robot.step((0.2, 0.0), dt=0.1)
        scan = lidar.scan(true_pose)
        est_pose = aug.step(odom, scan)

        assert np.all(np.isfinite(est_pose))
        assert est_pose.shape == (3,)
        assert aug.particles.shape == (80, 3)
