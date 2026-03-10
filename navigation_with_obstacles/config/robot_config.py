"""
Custom robot configuration for navigation with obstacles.
Inherits from BaseQuadWithCameraCfg and overrides spawn position.
"""
import numpy as np
from aerial_gym.config.robot_config.base_quad_config import BaseQuadWithCameraCfg


class NavQuadWithCameraCfg(BaseQuadWithCameraCfg):
    """
    Quadrotor with depth camera for navigation task.
    Spawns near the start of the environment (low X) with random yaw.
    """

    class init_config(BaseQuadWithCameraCfg.init_config):
        # Spawn position as ratio of environment bounds [x, y, z]
        # X: 5-15% (near the start/back of the environment)
        # Y: 30-70% (centered, with some randomization)
        # Z: 30-70% (centered, with some randomization)
        min_state_ratio = [
            0.05,     # X: near start
            0.3,      # Y: 30%
            0.3,      # Z: 30%
            0,        # Roll (rad)
            0,        # Pitch (rad)
            -np.pi,   # Yaw min (facing any direction)
            1.0,      # Scale
            0,        # Linear velocity X
            0,        # Linear velocity Y
            0,        # Linear velocity Z
            0,        # Angular velocity X
            0,        # Angular velocity Y
            0,        # Angular velocity Z
        ]
        max_state_ratio = [
            0.15,     # X: spawn zone width
            0.7,      # Y: 70%
            0.7,      # Z: 70%
            0,        # Roll (rad)
            0,        # Pitch (rad)
            np.pi,    # Yaw max
            1.0,      # Scale
            0,        # Linear velocity X
            0,        # Linear velocity Y
            0,        # Linear velocity Z
            0,        # Angular velocity X
            0,        # Angular velocity Y
            0,        # Angular velocity Z
        ]
