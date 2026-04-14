"""Robot config for the target-hold drone.

Inherits from ``BaseQuadWithCameraImuCfg`` (Isaac Gym path — the only path that
produces RGB frames). Overrides the camera to ensure segmentation output is on
(required by ``bbox_from_segmentation``) and to fix the image dimensions that
must agree with TargetHoldTaskCfg.bbox.
"""

import numpy as np

from aerial_gym.config.robot_config.base_quad_config import BaseQuadWithCameraImuCfg
from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig


class TargetHoldCameraCfg(BaseDepthCameraConfig):
    # Real camera intrinsics (from GOLAN sensor):
    #   fx = fy = 2317.65,  cx = 960,  cy = 540
    #   hfov = 45°,  vfov = 30.6°
    #   distortion: plumb_bob, all zeros (none)
    #   native resolution: 1920 x 1080
    #   latency: ~70 ms (not yet consumed — see timing scaffolding)
    #
    # Isaac Gym only needs hfov + resolution; fx/fy/cx/cy are derived and
    # match the real camera exactly (square pixels, centered principal point,
    # no distortion).
    #
  
    width = 480
    height = 270
    horizontal_fov_deg = 45.0

    max_range = 200.0
    min_range = 0.2

    segmentation_camera = True  # REQUIRED — bbox_from_segmentation reads seg mask
    use_collision_geometry = False

    # Keep extrinsics deterministic for now; the seg-mask bbox is invariant to
    # camera pose randomization so this can be flipped to True later without
    # touching bbox logic.
    randomize_placement = False
    nominal_position = [0.10, 0.0, 0.03]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]


class TargetHoldImuCfg(BaseImuConfig):
    # Plain IMU; user will tune noise/bias later.
    pass


class TargetHoldQuadCfg(BaseQuadWithCameraImuCfg):
    class init_config:
        # Drone spawns at world origin (x=0, y=0), altitude 50-120 m.
        # Env bounds: x[-200,200], y[-200,200], z[-1,150]. Range = [400, 400, 151].
        # x/y ratio 0.5 → center → 0 m.
        # z ratio: (50-(-1))/151 = 0.338  to  (120-(-1))/151 = 0.801.
        min_init_state = [
            0.5, 0.5, 0.338,                                  # position ratios
            -np.pi / 6, -np.pi / 6, -np.pi,                   # roll, pitch, yaw (rad)
            1.0,                                                # shape-preserve
            -4.04, -4.04, -4.04,                               # vx, vy, vz (m/s, |v|≤7)
            -3.0, -3.0, -3.0,                                  # wx, wy, wz (rad/s, FPV)
        ]
        max_init_state = [
            0.5, 0.5, 0.801,
            np.pi / 6, np.pi / 6, np.pi,
            1.0,
            4.04, 4.04, 4.04,
            3.0, 3.0, 3.0,
        ]

    class sensor_config(BaseQuadWithCameraImuCfg.sensor_config):
        enable_camera = True
        camera_config = TargetHoldCameraCfg
        enable_imu = True
        imu_config = TargetHoldImuCfg
