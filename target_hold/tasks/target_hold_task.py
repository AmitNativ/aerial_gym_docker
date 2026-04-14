"""TargetHoldTask — keep a target box in the camera FOV at a desired distance.

Observation (22 dims, packed into a single tensor for rl_games):
  [0:4]   quaternion (qw>0 enforced)
  [4:7]   body-frame angular velocity (rad/s)
  [7:11]  target bbox [tl_x, tl_y, br_x, br_y] normalized to [-1, 1]
  [11:12] vz — world-frame vertical velocity (m/s)
  [12:16] previous actions (raw [-1,1] policy outputs)
  [16:19] robot position (world frame)              — privileged
  [19:22] robot linear velocity vx,vy,vz (world)    — privileged
  Actor reads [0:12], GRU receives [12:16], Critic reads [0:22].

Action (4 dims, policy outputs in [-1, 1]):
  [0] thrust   — shifted to [0,1], scaled by max_thrust_m_s2
  [1] roll rate  — scaled by max_roll_rate_rad_s
  [2] pitch rate — scaled by max_pitch_rate_rad_s
  [3] yaw rate   — scaled by max_yaw_rate_rad_s
  Scaling is done by ``task_config.action_transformation_function`` before
  passing to the ``lee_rates_control`` controller.

Reward: dense penalties (area, centering, vz, attitude, smoothness) +
  sparse terminations (visibility loss, safety limits, crash).
  Curriculum: visibility termination disabled until 75% success rate.
"""

import numpy as np
import torch
from gym.spaces import Box, Dict

from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.task.base_task import BaseTask
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import quat_rotate, quat_from_euler_xyz

from target_hold.configs.target_hold_robot_config import TargetHoldCameraCfg
from target_hold.utils.bbox_from_segmentation import bbox_from_segmentation


logger = CustomLogger("target_hold_task")


class TargetHoldTask(BaseTask):
    def __init__(
        self,
        task_config,
        seed=None,
        num_envs=None,
        headless=None,
        device=None,
        use_warp=None,
    ):
        # Override config values from make_task kwargs (mirrors PositionSetpointTask).
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device

        # Convert reward params (possibly lists/floats) into tensors so downstream
        # math is batched and device-correct.
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )

        logger.info(
            "Building target-hold env (sim=%s, env=%s, robot=%s, controller=%s, "
            "num_envs=%d, use_warp=%s, headless=%s)",
            self.task_config.sim_name,
            self.task_config.env_name,
            self.task_config.robot_name,
            self.task_config.controller_name,
            self.task_config.num_envs,
            self.task_config.use_warp,
            self.task_config.headless,
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.num_envs = self.sim_env.num_envs
        self.counter = 0

        # Action buffers — raw policy outputs in [-1, 1].
        self.actions = torch.zeros(
            (self.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)

        # Grab the obs dict once — Aerial Gym exposes tensors by reference so
        # subsequent reads see fresh values in-place.
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.num_envs, device=self.device)

        # Per-env timing / latency scaffolding (sampled once at reset, never
        # consumed by the stub — the user hooks these into their real obs/action
        # delay buffers later).
        timing = self.task_config.timing
        self.action_latency_ms = torch.zeros(self.num_envs, device=self.device)
        self.action_jitter_ms = torch.zeros(self.num_envs, device=self.device)
        self.camera_rate_hz = torch.zeros(self.num_envs, device=self.device)
        self.imu_rate_hz = torch.zeros(self.num_envs, device=self.device)
        self._sample_timing_for(torch.arange(self.num_envs, device=self.device))

        # Per-env bbox buffer, filled each step from the seg mask.
        self.target_bbox = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_visible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Reward state buffers.
        self.area_at_reset = torch.zeros(self.num_envs, device=self.device)
        self.area_prev = torch.zeros(self.num_envs, device=self.device)

        # Curriculum state.
        self.curriculum_terminate_on_vis = False
        self.episode_success_count = 0
        self.episode_total_count = 0

        # Precompute min bbox area threshold in normalized coords.
        W = self.task_config.bbox.image_width
        H = self.task_config.bbox.image_height
        min_px = self.task_config.reward_parameters["min_bbox_area_px"]
        # Map pixel area to normalized [-1,1] area.
        # A pixel has normalized width 2/(W-1) and height 2/(H-1), so
        # pixel_area_normalized = pixel_area * 4 / ((W-1)*(H-1)).
        self.min_bbox_area_norm = min_px * 4.0 / ((W - 1) * (H - 1))

        # Gym spaces.
        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )

        self.task_obs = {
            "observations": torch.zeros(
                (self.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (self.num_envs, self.task_config.privileged_observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

        self.infos = {}

    # ------------------------------------------------------------------
    # BaseTask API
    # ------------------------------------------------------------------

    def close(self):
        self.sim_env.delete_env()

    def render(self):
        return None

    def reset(self):
        self.infos = {}
        self.sim_env.reset()
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_timing_for(all_ids)
        self._place_target_in_frustum(all_ids)
        self._compute_initial_area(all_ids)
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        if env_ids.numel() > 0:
            self._sample_timing_for(env_ids)
            self.actions[env_ids] = 0.0
            self._place_target_in_frustum(env_ids)
            self._compute_initial_area(env_ids)

    def step(self, actions):
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        scaled_actions = self.task_config.action_transformation_function(self.actions)
        self.sim_env.step(actions=scaled_actions)

        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        self.sim_env.post_reward_calculation_step()

        # Track curriculum: count episodes that ended by truncation (success)
        # vs termination (failure).
        ended = (self.terminations > 0) | (self.truncations > 0)
        if ended.any():
            n_ended = ended.sum().item()
            n_success = (self.truncations[ended] > 0).sum().item()
            self.episode_total_count += n_ended
            self.episode_success_count += n_success
            # Check curriculum threshold.
            p = self.task_config.reward_parameters
            if (self.episode_total_count >= p["curriculum_check_window"]
                    and not self.curriculum_terminate_on_vis):
                rate = self.episode_success_count / max(self.episode_total_count, 1)
                if rate >= p["curriculum_success_threshold"]:
                    self.curriculum_terminate_on_vis = True
                    logger.info(
                        "Curriculum: enabling visibility termination "
                        "(success rate %.2f >= %.2f after %d episodes)",
                        rate, p["curriculum_success_threshold"],
                        self.episode_total_count,
                    )
                # Reset counters for next window.
                self.episode_total_count = 0
                self.episode_success_count = 0

        if not self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        return return_tuple

    # ------------------------------------------------------------------
    # Obs / reward
    # ------------------------------------------------------------------

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        """Pack the 22-dim observation vector.

        Layout:
          [0:4]   quaternion (qw > 0 enforced)
          [4:7]   body-frame angular velocity (rad/s)
          [7:11]  target bbox [tl_x, tl_y, br_x, br_y] normalized to [-1, 1]
          [11:12] vz — world-frame vertical velocity (m/s)
          [12:16] previous actions (raw [-1,1] policy outputs)
          [16:19] robot position (world frame)              — privileged
          [19:22] robot linear velocity vx,vy,vz (world)    — privileged
        """
        robot_pos = self.obs_dict["robot_position"]          # (N, 3) world frame
        robot_orn = self.obs_dict["robot_orientation"]       # (N, 4) qx,qy,qz,qw
        robot_ang = self.obs_dict["robot_body_angvel"]       # (N, 3)
        robot_linvel = self.obs_dict["robot_linvel"]         # (N, 3) world frame
        target_pos = self.obs_dict["obstacle_position"]      # (N, 1, 3)
        target_orn = self.obs_dict["obstacle_orientation"]   # (N, 1, 4)

        seg = self.obs_dict.get("segmentation_pixels")

        # --- BBox from segmentation mask ---
        if seg is not None:
            self.target_bbox[:], self.target_visible[:] = bbox_from_segmentation(
                seg,
                target_semantic_id=self.task_config.bbox.target_semantic_id,
                image_width=self.task_config.bbox.image_width,
                image_height=self.task_config.bbox.image_height,
                return_normalized=False,
            )
        else:
            self.target_bbox.zero_()
            self.target_visible.zero_()

        # Normalize bbox pixels to [-1, 1].
        # x: [0, W-1] -> [-1, 1],  y: [0, H-1] -> [-1, 1]
        W = self.task_config.bbox.image_width
        H = self.task_config.bbox.image_height
        bbox_norm = self.target_bbox.clone()
        bbox_norm[:, 0] = 2.0 * self.target_bbox[:, 0] / (W - 1) - 1.0  # tl_x
        bbox_norm[:, 1] = 2.0 * self.target_bbox[:, 1] / (H - 1) - 1.0  # tl_y
        bbox_norm[:, 2] = 2.0 * self.target_bbox[:, 2] / (W - 1) - 1.0  # br_x
        bbox_norm[:, 3] = 2.0 * self.target_bbox[:, 3] / (H - 1) - 1.0  # br_y
        # Zero out bbox for envs where target is not visible.
        bbox_norm[~self.target_visible] = 0.0

        # --- Quaternion with qw > 0 to resolve double-cover ambiguity ---
        quat = robot_orn.clone()
        qw_neg = quat[:, 3] < 0.0
        quat[qw_neg] = -quat[qw_neg]

        # --- Pack observation (22 dims) ---
        obs = self.task_obs["observations"]
        obs[:, 0:4] = quat
        obs[:, 4:7] = robot_ang
        obs[:, 7:11] = bbox_norm
        obs[:, 11:12] = robot_linvel[:, 2:3]   # vz world frame
        obs[:, 12:16] = self.actions            # prev actions (raw policy outputs)
        obs[:, 16:19] = robot_pos               # privileged: world position
        obs[:, 19:22] = robot_linvel            # privileged: world linear velocity

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

        self.infos = {
            "target_bbox": self.target_bbox,
            "target_bbox_normalized": bbox_norm,
            "target_visible": self.target_visible,
            "target_position": target_pos.squeeze(1),
            "target_orientation": target_orn.squeeze(1),
            "robot_position": robot_pos,
            "robot_orientation": robot_orn,
            "robot_body_angvel": robot_ang,
            "robot_linvel": robot_linvel,
        }

    def compute_rewards_and_crashes(self, obs_dict):
        """Compute dense penalty reward + sparse terminations.

        Dense terms (all ≤ 0, closer to 0 = better):
          r_dist    — bbox area deviation from reset + rate of change
          r_horiz   — horizontal centering penalty
          r_vert    — vertical centering penalty (outside deadzone)
          r_vz      — vertical velocity penalty
          r_att     — pitch + roll penalty
          r_smooth  — action jerk penalty

        Sparse (applied on events, then terminate):
          r_vis     — target lost from frame / bbox too small
          r_safety  — pitch or roll exceeds safety limit
          r_crash   — collision
        """
        p = self.task_config.reward_parameters
        crashes = obs_dict["crashes"]

        # --- Bbox area in normalized [-1,1] coords ---
        bbox_n = self.task_obs["observations"][:, 7:11]  # already normalized in process_obs
        area = (bbox_n[:, 2] - bbox_n[:, 0]) * (bbox_n[:, 3] - bbox_n[:, 1])
        area = torch.clamp(area, min=0.0)  # guard against inverted bbox

        # --- 1. Distance maintenance via bbox area ---
        area_err = area - self.area_at_reset
        d_area = area - self.area_prev
        r_dist = p["lambda_area_err"] * area_err ** 2 + p["lambda_d_area"] * d_area ** 2
        self.area_prev[:] = area

        # --- 2. Horizontal centering ---
        cx_bbox = (bbox_n[:, 0] + bbox_n[:, 2]) / 2.0
        # Zero out centering penalty when target not visible (bbox is zeroed).
        cx_bbox = torch.where(self.target_visible, cx_bbox, torch.zeros_like(cx_bbox))
        r_horiz = p["lambda_horiz"] * cx_bbox ** 2

        # --- 3. Vertical centering (with deadzone) ---
        cy_bbox = (bbox_n[:, 1] + bbox_n[:, 3]) / 2.0
        cy_bbox = torch.where(self.target_visible, cy_bbox, torch.zeros_like(cy_bbox))
        deadzone = p["vert_deadzone_radius"]
        vert_err = torch.clamp(cy_bbox.abs() - deadzone, min=0.0)
        r_vert = p["lambda_vert"] * vert_err ** 2

        # --- 4. Vertical velocity ---
        vz = obs_dict["robot_linvel"][:, 2]
        r_vz = p["lambda_vz"] * vz ** 2

        # --- 5. Attitude penalty ---
        euler = obs_dict["robot_euler_angles"]  # (N, 3) roll, pitch, yaw
        roll = euler[:, 0]
        pitch = euler[:, 1]
        r_att = p["lambda_att"] * (pitch ** 2 + roll ** 2)

        # --- 6. Smoothness ---
        action_diff = self.actions - self.prev_actions
        r_smooth = p["lambda_smooth"] * (action_diff ** 2).sum(dim=-1)

        # --- Dense total ---
        reward = r_dist + r_horiz + r_vert + r_vz + r_att + r_smooth

        # --- 7. Visibility loss (sparse) ---
        target_lost = ~self.target_visible
        bbox_too_small = self.target_visible & (area < self.min_bbox_area_norm)
        vis_fail = target_lost | bbox_too_small
        reward = torch.where(vis_fail, reward + p["lambda_vis_loss"], reward)

        # --- 8. Safety — excessive attitude (sparse + terminate) ---
        pitch_limit = np.radians(p["safety_pitch_limit_deg"])
        roll_limit = np.radians(p["safety_roll_limit_deg"])
        unsafe = (pitch.abs() > pitch_limit) | (roll.abs() > roll_limit)
        reward = torch.where(unsafe, reward + p["lambda_safety"], reward)

        # --- 9. Crash (sparse + terminate) ---
        reward = torch.where(crashes > 0.0, reward + p["lambda_crash"], reward)

        # --- Terminations ---
        terminate = crashes.clone()
        terminate = torch.where(unsafe, torch.ones_like(terminate), terminate)
        # Visibility termination is curriculum-gated.
        if self.curriculum_terminate_on_vis:
            terminate = torch.where(vis_fail, torch.ones_like(terminate), terminate)

        return reward, terminate

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_initial_area(self, env_ids):
        """Compute the initial bbox area after target placement.

        Triggers a zero-action physics step so the camera renders the new
        target position, then reads the segmentation mask to get the bbox.
        """
        # Step with zero actions to trigger a camera render.
        zero_actions = torch.zeros(
            (self.num_envs, self.task_config.action_space_dim), device=self.device
        )
        self.sim_env.step(actions=zero_actions)

        seg = self.obs_dict.get("segmentation_pixels")
        if seg is not None:
            bbox_px, visible = bbox_from_segmentation(
                seg,
                target_semantic_id=self.task_config.bbox.target_semantic_id,
                image_width=self.task_config.bbox.image_width,
                image_height=self.task_config.bbox.image_height,
                return_normalized=False,
            )
            # Normalize to [-1, 1].
            W = self.task_config.bbox.image_width
            H = self.task_config.bbox.image_height
            tl_x = 2.0 * bbox_px[:, 0] / (W - 1) - 1.0
            tl_y = 2.0 * bbox_px[:, 1] / (H - 1) - 1.0
            br_x = 2.0 * bbox_px[:, 2] / (W - 1) - 1.0
            br_y = 2.0 * bbox_px[:, 3] / (H - 1) - 1.0
            area = torch.clamp((br_x - tl_x) * (br_y - tl_y), min=0.0)
            self.area_at_reset[env_ids] = area[env_ids]
            self.area_prev[env_ids] = area[env_ids]
        else:
            self.area_at_reset[env_ids] = 0.0
            self.area_prev[env_ids] = 0.0

    def _place_target_in_frustum(self, env_ids, max_attempts=10):
        """Place the target on the ground within the camera's field of view.

        Back-projects a random pixel through the camera to the z=0 ground
        plane. If the ray doesn't hit the ground (drone pitched upward), the
        pixel is resampled up to ``max_attempts`` times; remaining failures
        fall back to directly below the drone.
        """
        n = env_ids.numel()
        dev = self.device

        drone_pos = self.obs_dict["robot_position"][env_ids].clone()   # (n, 3)
        drone_quat = self.obs_dict["robot_orientation"][env_ids].clone()  # (n, 4)

        # Camera intrinsics (from config — single source of truth).
        W = TargetHoldCameraCfg.width
        H = TargetHoldCameraCfg.height
        hfov_rad = np.radians(TargetHoldCameraCfg.horizontal_fov_deg)
        fx = fy = (W / 2.0) / np.tan(hfov_rad / 2.0)
        cx, cy = W / 2.0, H / 2.0

        # Camera offset in body frame.
        cam_offset = torch.tensor(
            TargetHoldCameraCfg.nominal_position, device=dev, dtype=torch.float32
        )
        cam_world_pos = drone_pos + quat_rotate(drone_quat, cam_offset.expand(n, -1))

        # Isaac Gym camera convention: looks along -z in body frame.
        # A pixel (u, v) maps to a ray in the camera optical frame:
        #   ray_optical = [(u-cx)/fx, -(v-cy)/fy, -1]   (y/z flipped: IG has y-up, z-into-scene)
        # Then rotated to world frame by the drone quaternion (camera has
        # identity orientation relative to the body).

        # Start with all envs needing placement.
        needs_placement = torch.ones(n, dtype=torch.bool, device=dev)
        ground_pos = torch.zeros((n, 3), device=dev)

        for _ in range(max_attempts):
            m = needs_placement.sum().item()
            if m == 0:
                break
            idx = needs_placement.nonzero(as_tuple=True)[0]

            # Random pixel in the inner 60% of the image (20% margin from each edge)
            # so the target is fully visible and not immediately lost.
            margin_x = 0.2 * W
            margin_y = 0.2 * H
            u = margin_x + torch.rand(m, device=dev) * (W - 2 * margin_x)
            v = margin_y + torch.rand(m, device=dev) * (H - 2 * margin_y)

            # Ray in camera optical frame (Isaac Gym: x-right, y-down, z-forward is -z body).
            ray_cam = torch.stack([
                (u - cx) / fx,
                -(v - cy) / fy,
                -torch.ones(m, device=dev),
            ], dim=-1)
            ray_cam = ray_cam / ray_cam.norm(dim=-1, keepdim=True)

            # Rotate to world frame.
            ray_world = quat_rotate(drone_quat[idx], ray_cam)

            # Intersect with z=0: cam_z + t * ray_z = 0
            ray_z = ray_world[:, 2]
            cam_z = cam_world_pos[idx, 2]
            # Valid if ray points downward (ray_z < 0) and camera is above ground.
            valid_mask = (ray_z < -1e-6) & (cam_z > 0.0)
            t = torch.where(valid_mask, -cam_z / ray_z, torch.zeros_like(cam_z))

            # Also reject hits beyond max camera range.
            valid_mask = valid_mask & (t < TargetHoldCameraCfg.max_range) & (t > 0.0)

            hit = cam_world_pos[idx] + t.unsqueeze(-1) * ray_world
            hit[:, 2] = 0.0  # snap to ground

            ground_pos[idx[valid_mask]] = hit[valid_mask]
            needs_placement[idx[valid_mask]] = False

        # Fallback for any remaining: place directly below drone on ground.
        if needs_placement.any():
            fallback_idx = needs_placement.nonzero(as_tuple=True)[0]
            ground_pos[fallback_idx, 0] = drone_pos[fallback_idx, 0]
            ground_pos[fallback_idx, 1] = drone_pos[fallback_idx, 1]
            ground_pos[fallback_idx, 2] = 0.0

        # Write target position into the obstacle state tensor and push to sim.
        self.obs_dict["obstacle_position"][env_ids, 0, :] = ground_pos
        self.sim_env.IGE_env.write_to_sim()

    def _sample_timing_for(self, env_ids):
        """Re-roll latency/jitter/rate values for the given envs.

        Scaffolding only — nothing consumes these tensors yet.
        """
        t = self.task_config.timing
        n = env_ids.numel()
        dev = self.device
        self.action_latency_ms[env_ids] = _uniform(n, *t.action_latency_ms_range, device=dev)
        self.action_jitter_ms[env_ids] = torch.randn(n, device=dev) * t.action_jitter_ms_std
        self.camera_rate_hz[env_ids] = _uniform(n, *t.camera_rate_hz_range, device=dev)
        self.imu_rate_hz[env_ids] = _uniform(n, *t.imu_rate_hz_range, device=dev)


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo
