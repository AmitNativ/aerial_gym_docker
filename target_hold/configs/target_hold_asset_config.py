"""Asset config for the target box.

A single box asset per env, drawn from the folder of pre-generated cube URDFs
produced by ``target_hold/resources/generate_target_boxes.py``. With
``file=None``, Aerial Gym's ``randomly_pick_assets_from_folder`` picks a random
URDF per env at build time, giving cross-env shape variety. Per-reset shape
randomization is not supported by Aerial Gym today; per-reset pose randomization
is (via ``min_state_ratio`` / ``max_state_ratio``).
"""

import os

import numpy as np

from aerial_gym.config.asset_config.env_object_config import asset_state_params


TARGET_BOX_SEMANTIC_ID = 100

# Absolute path to the generated URDF folder. Using an absolute path here (vs
# relative) because the Aerial Gym asset loader cd's into asset_folder and must
# find URDFs regardless of the caller's cwd.
TARGET_BOX_ASSET_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "resources", "target_boxes")
)


class target_box_asset_params(asset_state_params):
    num_assets = 1

    asset_folder = TARGET_BOX_ASSET_FOLDER
    file = None  # None -> randomly_pick_assets_from_folder selects per env

    # Keep target stationary, collision-on so it has mass/presence but light
    # enough that the drone bouncing off it doesn't explode training.
    collision_mask = 1
    disable_gravity = True  # target floats at its placed pose
    fix_base_link = True
    collapse_fixed_joints = True
    density = 0.001

    # State ratios: [x, y, z, roll, pitch, yaw, 1.0 (shape-preserve), vx, vy, vz, wx, wy, wz]
    # Position is interpolated within env bounds: pos = lower + ratio * (upper - lower).
    #   Env bounds: x[-200,200], y[-200,200], z[-1,150].
    #   z ratio ≈ 0.007 places the target at z=0 (ground level).
    #   URDF origins are bottom-aligned so the box sits on the ground.
    #   x/y spread across env; actual drone-target distance is not enforced
    #   here — task reset logic should handle distance constraints.
    # Orientation: upright (zero roll/pitch), random yaw.
    # x/y: ratio 0.125 to 0.875 on [-200,200] → [-150, 150] m
    min_state_ratio = [
        0.125, 0.125, 0.007,
        0.0, 0.0, -np.pi,
        1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    max_state_ratio = [
        0.875, 0.875, 0.007,
        0.0, 0.0, np.pi,
        1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    keep_in_env = True
    per_link_semantic = False
    semantic_id = TARGET_BOX_SEMANTIC_ID  # unique id consumed by bbox_from_segmentation
    color = [220, 60, 60]
