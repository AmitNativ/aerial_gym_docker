"""Task config for target-hold.

Timing/latency fields under ``timing`` are SCAFFOLDING — they are stored on the
task object but not yet consumed (Aerial Gym has no built-in action-latency or
sensor-rate delay buffers; those will be added when the real observation is).
"""

import torch  # noqa: F401 — kept for parity with other task configs

from target_hold.configs.target_hold_asset_config import TARGET_BOX_SEMANTIC_ID
from target_hold.configs.target_hold_robot_config import TargetHoldCameraCfg


class task_config:
    # --- Registry wiring ---
    seed = 1
    sim_name = "base_sim"
    env_name = "target_hold_env"
    robot_name = "target_hold_quad"
    controller_name = "lee_rates_control"
    args = {}

    # --- Runtime ---
    num_envs = 1024
    use_warp = False  # Isaac Gym path required for RGB
    headless = True
    device = "cuda:0"

    # --- Episode / spaces ---
    # Observation layout (22 dims packed into a single tensor):
    #   [0:4]   robot_orientation quaternion (qw>0 enforced)
    #   [4:7]   robot_body_angvel (body frame, rad/s)
    #   [7:11]  target bbox [tl_x, tl_y, br_x, br_y] normalized to [-1, 1]
    #   [11:12] vz — vertical linear velocity (world frame, m/s)
    #   [12:16] previous actions (raw [-1,1] policy outputs)
    #   [16:19] robot position (world frame)              ← privileged
    #   [19:22] robot linear velocity vx,vy,vz (world)   ← privileged
    # Actor reads [0:12], GRU receives [12:16], Critic reads [0:22].
    observation_space_dim = 22
    privileged_observation_space_dim = 0  # packed into obs, not separate
    
    # Action layout (4 dims, each in [-1, 1] from the policy):
    #   [0]   thrust      — shifted to [0, 1], then scaled by max_thrust_m_s2
    #   [1]   roll rate   — scaled by max_roll_rate_rad_s
    #   [2]   pitch rate  — scaled by max_pitch_rate_rad_s
    #   [3]   yaw rate    — scaled by max_yaw_rate_rad_s
    action_space_dim = 4
    episode_len_steps = 500
    return_state_before_reset = False

    class action_scaling:
        max_thrust_m_s2 = 20.0         # maps [0, 1] thrust to [0, max] m/s²
        max_roll_rate_rad_s = 3.14     # maps [-1, 1] to [-max, max] rad/s
        max_pitch_rate_rad_s = 3.14
        max_yaw_rate_rad_s = 1.57

    def action_transformation_function(action):
        """Scale policy outputs ([-1, 1]) to controller inputs.

        Controller expects [thrust_m_s2, roll_rate, pitch_rate, yaw_rate].
        """
        s = task_config.action_scaling
        clamped = torch.clamp(action, -1.0, 1.0)
        scaled = torch.zeros_like(clamped)
        # Thrust: [-1,1] -> [0,1] -> [0, max_thrust]
        scaled[:, 0] = (clamped[:, 0] + 1.0) / 2.0 * s.max_thrust_m_s2
        # Angular rates: [-1,1] -> [-max, max]
        scaled[:, 1] = clamped[:, 1] * s.max_roll_rate_rad_s
        scaled[:, 2] = clamped[:, 2] * s.max_pitch_rate_rad_s
        scaled[:, 3] = clamped[:, 3] * s.max_yaw_rate_rad_s
        return scaled

    # --- Target-hold task parameters ---
    class target:
        desired_distance_m = 3.0
        desired_distance_tolerance_m = 0.5

    class bbox:
        image_width = TargetHoldCameraCfg.width
        image_height = TargetHoldCameraCfg.height
        target_semantic_id = TARGET_BOX_SEMANTIC_ID
        return_normalized = False
        empty_bbox_value = [0.0, 0.0, 0.0, 0.0]

    class timing:
        # Scaffolding — the task reads these into per-env tensors at reset but
        # does not yet apply delay/jitter. Physics is driven by base_sim.dt
        # (0.01 s) and env_config.num_physics_steps_per_env_step_mean (=1).
        # TODO(user): apply action and observation jitter separately when consuming these fields.
        physics_dt_s = 0.01
        control_decimation = 1
        action_latency_ms_range = [0.0, 20.0]
        action_jitter_ms_std = 2.0
        camera_rate_hz_range = [20.0, 30.0]
        imu_rate_hz_range = [100.0, 200.0]

    # --- Reward parameters (single source of truth) ---
    reward_parameters = {
        "lambda_area_err": -1.0,
        "lambda_d_area": -0.5,
        "lambda_horiz": -0.5,
        "lambda_vert": -0.5,
        "vert_deadzone_radius": 0.3,
        "lambda_vz": -0.1,
        "lambda_att": -0.1,
        "lambda_smooth": -0.01,
        "lambda_vis_loss": -10.0,
        "min_bbox_area_px": 40.0,
        "lambda_safety": -10.0,
        "safety_pitch_limit_deg": 45.0,
        "safety_roll_limit_deg": 45.0,
        "lambda_crash": -10.0,
        "curriculum_success_threshold": 0.75,
        "curriculum_check_window": 2048,
    }
