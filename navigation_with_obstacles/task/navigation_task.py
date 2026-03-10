"""
Navigation with Obstacles Task for Aerial Gym Simulator.

A navigation task where a quadrotor must:
1. Navigate to a target waypoint in a box-shaped environment
2. Avoid obstacles using depth camera observations encoded by a custom DepthVAE
3. Use acceleration control (accel_x, accel_y, accel_z, yaw_rate)

Features:
- 30-level curriculum: panels (levels 0-5), cumulative panels + objects (levels 6-30)
- Custom 32D DepthVAE encoding (matching VAE training distribution)
- Randomized environment bounds: L×W×H in [8,12]×[5,8]×[4,6]
- Observation: [log_d_hor, d_z, d_norm(3), vel_w(3), angular_vel_b(3), angular_acc_b(3), vae(32)] = 46D
"""
from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import (
    torch_rand_float_tensor,
    torch_interpolate_ratio,
)
from aerial_gym.utils.logging import CustomLogger

import torch
import numpy as np
import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("navigation_with_obstacles_task")


class NavigationWithObstaclesTask(BaseTask):
    """
    Navigation task with obstacle curriculum and acceleration control.

    Observation (46D):
        [0]     log(horizontal_distance_to_target)
        [1]     vertical distance to target (d_z)
        [2:5]   unit displacement vector to target (world frame)
        [5:8]   linear velocity (world frame)
        [8:11]  angular velocity (body frame)
        [11:14] angular acceleration (body frame, numerical diff)
        [14:46] VAE latent encoding (32D)

    Action (4D):
        [0:3]   acceleration command (world frame, m/s²)
        [3]     yaw rate command (rad/s)
    """

    def __init__(
        self,
        task_config,
        seed=None,
        num_envs=None,
        headless=None,
        device=None,
        use_warp=None,
    ):
        # Override config params if provided
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

        # Convert reward params to tensors
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )

        logger.info("Building Navigation with Obstacles environment")
        logger.info(
            f"Sim: {task_config.sim_name}, Env: {task_config.env_name}, "
            f"Robot: {task_config.robot_name}, Controller: {task_config.controller_name}"
        )

        # Build simulation environment
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

        # Target position for each environment
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # Target sampling ratios
        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device
        ).expand(self.sim_env.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device
        ).expand(self.sim_env.num_envs, -1)

        # Previous distance to target (for progress tracking)
        self.prev_dist = torch.zeros(self.sim_env.num_envs, device=self.device)

        # Previous body angular velocity (for angular acceleration computation)
        self.prev_body_angvel = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # VAE encoder for depth images (custom DepthVAE)
        if self.task_config.vae_config.use_vae:
            from vae_depth.vae_image_encoder import DepthVAEImageEncoder
            self.vae_model = DepthVAEImageEncoder(
                config=self.task_config.vae_config, device=self.device
            )
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
        else:
            self.vae_model = None
            self.image_latents = None

        # Get observation dictionary reference from environment
        self.obs_dict = self.sim_env.get_obs()

        # Environment step dt = physics_dt * num_physics_steps_per_env_step
        physics_dt = self.obs_dict.get("dt", 0.01)
        num_physics_steps = getattr(
            self.sim_env.cfg.env, "num_physics_steps_per_env_step_mean", 10
        )
        self.env_step_dt = physics_dt * num_physics_steps

        # Curriculum setup
        self.curriculum_level = self.task_config.curriculum.min_level
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
        self.curriculum_progress_fraction = 0.0

        # Curriculum tracking aggregates
        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0
        self.exceeds_aggregate = 0

        # Logged metrics for tensorboard (updated each curriculum check)
        self.logged_success_rate = 0.0
        self.logged_crash_rate = 0.0
        self.logged_exceed_rate = 0.0

        # Per-step reward component means (for tensorboard)
        self.logged_r_dist = 0.0
        self.logged_r_speed = 0.0
        self.logged_r_dir = 0.0
        self.logged_r_angvel = 0.0
        self.logged_r_perc = 0.0

        # Termination/truncation tensors
        # IMPORTANT: self.terminations is a SEPARATE tensor, NOT an alias of
        # obs_dict["crashes"]. obs_dict["crashes"] is the simulator's collision
        # buffer (dtype=bool) used by post_reward_calculation_step. We must not
        # overwrite it with exceed/arrive flags.
        self.terminations = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool
        )
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.sim_env.num_envs, device=self.device)

        # Define observation and action spaces for rl_games
        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Action transformation function
        self.action_transformation_function = (
            self.task_config.action_transformation_function
        )

        self.num_envs = self.sim_env.num_envs

        # Task observation tensor
        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
        }

        self.infos = {}

        # Episode counter for keep_same_env_for_num_episodes
        self.episode_counter = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.int32
        )
        self.keep_same_env_episodes = getattr(
            self.sim_env.cfg.env, "keep_same_env_for_num_episodes", 1
        )

        logger.info(
            f"Task initialized with {self.num_envs} environments, "
            f"obs_dim={self.task_config.observation_space_dim}, "
            f"action_dim={self.task_config.action_space_dim}, "
            f"curriculum_level={self.curriculum_level}"
        )

    def close(self):
        """Clean up simulation resources."""
        del self.sim_env
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def reset(self):
        """Reset all environments."""
        self.reset_idx(
            torch.arange(self.sim_env.num_envs, device=self.device),
            force_obstacle_reset=True,
        )
        return self.get_return_tuple()

    def reset_idx(self, env_ids, force_obstacle_reset=False):
        """
        Reset specific environments.

        Args:
            env_ids: Tensor of environment indices to reset
            force_obstacle_reset: If True, reset obstacles regardless of episode counter
        """
        if len(env_ids) == 0:
            return

        # Increment episode counter
        self.episode_counter[env_ids] += 1

        # Determine which environments need full obstacle reset
        if force_obstacle_reset:
            envs_needing_obstacle_reset = env_ids
        else:
            needs_reset_mask = (
                self.episode_counter[env_ids] >= self.keep_same_env_episodes
            )
            envs_needing_obstacle_reset = env_ids[needs_reset_mask]

        # Reset episode counter for envs getting new obstacles
        if len(envs_needing_obstacle_reset) > 0:
            self.episode_counter[envs_needing_obstacle_reset] = 0

        # Full reset including obstacles
        if len(envs_needing_obstacle_reset) > 0:
            self.sim_env.reset_idx(envs_needing_obstacle_reset)

        # Robot-only reset for remaining environments
        envs_robot_only = (
            env_ids[~torch.isin(env_ids, envs_needing_obstacle_reset)]
            if len(envs_needing_obstacle_reset) > 0
            else env_ids
        )
        if len(envs_robot_only) > 0:
            self.sim_env.robot_manager.reset_idx(envs_robot_only)
            self.sim_env.IGE_env.write_to_sim()
            self.sim_env.sim_steps[envs_robot_only] = 0

        # Sample new target positions within bounds
        target_ratio = torch_rand_float_tensor(
            self.target_min_ratio, self.target_max_ratio
        )
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )

        # Reset previous distance for progress tracking
        self.prev_dist[env_ids] = torch.norm(
            self.target_position[env_ids] - self.obs_dict["robot_position"][env_ids],
            dim=1,
        )

        # Reset previous angular velocity for angular acceleration computation
        self.prev_body_angvel[env_ids] = 0.0

        self.infos = {}

    def render(self):
        """Render the environment."""
        return self.sim_env.render()

    def step(self, actions):
        """
        Execute one step of the simulation.

        Args:
            actions: Tensor of actions (num_envs, 4) in range [-1, 1]

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        # Transform network outputs to controller commands
        transformed_action = self.action_transformation_function(actions)

        # Step the simulation
        self.sim_env.step(actions=transformed_action)

        # Compute rewards, terminations, and event masks
        self.rewards[:], self.terminations[:], arrive_mask, exceed_mask = (
            self.compute_rewards(self.obs_dict)
        )

        # Check for episode timeout (truncation), only for non-terminated envs
        timeout_mask = (self.sim_env.sim_steps > self.task_config.episode_len_steps) & (
            self.terminations == 0
        )

        # Write exceed/arrive into truncation buffer so post_reward_calculation_step
        # picks them up for reset. Collisions are already in obs_dict["crashes"].
        self.truncations[:] = (timeout_mask | arrive_mask | exceed_mask)

        # Success = arrived at target (from compute_rewards)
        successes = arrive_mask.float()

        # Exceeds = out-of-bounds terminations
        exceeds = exceed_mask.float()

        # Crashes = collision terminations (not arrivals, not exceeds)
        crashes = ((self.terminations > 0) & (~arrive_mask) & (~exceed_mask)).float()

        # Timeouts = episode ran out of steps (not arrive/exceed/collision)
        timeouts = timeout_mask.float()

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = crashes
        self.infos["exceeds"] = exceeds
        # Scalar metrics for tensorboard logging (IsaacAlgoObserver logs scalars from infos)
        self.infos["curriculum_level"] = float(self.curriculum_level)
        self.infos["success_rate"] = self.logged_success_rate
        self.infos["crash_rate"] = self.logged_crash_rate
        self.infos["exceed_rate"] = self.logged_exceed_rate

        # Reward components (mean across progress envs)
        self.infos["reward/r_dist"] = self.logged_r_dist
        self.infos["reward/r_speed"] = self.logged_r_speed
        self.infos["reward/r_dir"] = self.logged_r_dir
        self.infos["reward/r_angvel"] = self.logged_r_angvel
        self.infos["reward/r_perc"] = self.logged_r_perc

        # Flight metrics (mean across all envs)
        robot_pos = self.obs_dict["robot_position"]
        disp = self.target_position - robot_pos
        self.infos["metrics/dist_to_target"] = float(torch.norm(disp, dim=1).mean())
        self.infos["metrics/v_horizontal"] = float(
            torch.norm(self.obs_dict["robot_linvel"][:, :2], dim=1).mean()
        )
        self.infos["metrics/episode_length"] = float(self.sim_env.sim_steps.float().mean())

        # Update curriculum
        self.check_and_update_curriculum_level(
            successes, crashes, timeouts, exceeds
        )

        # Handle resets for terminated/truncated environments
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        # Process depth image through VAE encoder
        self.process_image_observation()

        return self.get_return_tuple()

    def process_image_observation(self):
        """Encode depth image through custom DepthVAE to get latent representation."""
        if self.task_config.vae_config.use_vae and self.vae_model is not None:
            image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
            self.image_latents[:] = self.vae_model.encode(image_obs)

    def get_return_tuple(self):
        """Build and return the step/reset output tuple."""
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        """
        Build observation vector (46D total).

        Observation structure:
        - [0]     log(horizontal_distance_to_target)
        - [1]     d_z: vertical distance to target
        - [2:5]   d_norm: unit displacement vector to target (world frame)
        - [5:8]   vel_w: linear velocity (world frame)
        - [8:11]  angular_vel_b: angular velocity (body frame)
        - [11:14] angular_acc_b: angular acceleration (body frame, numerical diff)
        - [14:46] vae_latent: DepthVAE encoding (32D)
        """
        robot_pos = self.obs_dict["robot_position"]
        target_pos = self.target_position

        # Displacement from drone to target (world frame)
        disp = target_pos - robot_pos

        # Horizontal distance (XY plane)
        d_hor = torch.norm(disp[:, :2], dim=1)

        # Vertical distance
        d_z = disp[:, 2]

        # Normalized displacement unit vector (world frame)
        d_total = torch.norm(disp, dim=1, keepdim=True)
        d_norm = disp / (d_total + 1e-6)

        # [0] log(horizontal distance)
        self.task_obs["observations"][:, 0] = torch.log(d_hor + 1e-6)

        # [1] vertical distance
        self.task_obs["observations"][:, 1] = d_z

        # [2:5] unit displacement vector (world frame)
        self.task_obs["observations"][:, 2:5] = d_norm

        # [5:8] linear velocity (world frame)
        self.task_obs["observations"][:, 5:8] = self.obs_dict["robot_linvel"]

        # [8:11] angular velocity (body frame)
        body_angvel = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 8:11] = body_angvel

        # [11:14] angular acceleration (body frame, numerical differentiation)
        angular_acc = (body_angvel - self.prev_body_angvel) / self.env_step_dt
        self.task_obs["observations"][:, 11:14] = angular_acc
        self.prev_body_angvel[:] = body_angvel

        # [14:46] VAE latent encoding (32D)
        if self.task_config.vae_config.use_vae and self.image_latents is not None:
            self.task_obs["observations"][:, 14:] = self.image_latents

    def check_and_update_curriculum_level(self, successes, crashes, timeouts, exceeds):
        """
        Update curriculum level based on success rate.
        Same logic as NavigationTask.check_and_update_curriculum_level.
        """
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)
        self.exceeds_aggregate += torch.sum(exceeds)

        instances = (
            self.success_aggregate
            + self.crashes_aggregate
            + self.timeouts_aggregate
            + self.exceeds_aggregate
        )

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances
            exceed_rate = self.exceeds_aggregate / instances

            # Update logged metrics for tensorboard
            self.logged_success_rate = float(success_rate)
            self.logged_crash_rate = float(crash_rate)
            self.logged_exceed_rate = float(exceed_rate)

            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            # Clamp curriculum level
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / max(
                self.task_config.curriculum.max_level
                - self.task_config.curriculum.min_level,
                1,
            )

            logger.warning(
                f"Curriculum Level: {self.curriculum_level}, "
                f"Progress: {self.curriculum_progress_fraction:.2f}"
            )
            logger.warning(
                f"Success Rate: {success_rate:.3f}, "
                f"Crash Rate: {crash_rate:.3f}, "
                f"Exceed Rate: {exceed_rate:.3f}, "
                f"Timeout Rate: {timeout_rate:.3f}"
            )
            logger.warning(
                f"Successes: {self.success_aggregate}, "
                f"Crashes: {self.crashes_aggregate}, "
                f"Exceeds: {self.exceeds_aggregate}, "
                f"Timeouts: {self.timeouts_aggregate}"
            )

            # Reset aggregates
            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0
            self.exceeds_aggregate = 0

    def compute_rewards(self, obs_dict):
        """
        Compute reward from four mutually exclusive components (priority order):
        1. r_exceed:    out-of-bounds penalty (terminates)
        2. r_arrive:    reached target bonus (terminates as success)
        3. r_collision: obstacle collision penalty (terminates)
        4. r_prog:      progress reward for normal steps (STUB)

        Returns:
            Tuple of (rewards, terminations, arrive_mask, exceed_mask) tensors
        """
        robot_pos = obs_dict["robot_position"]
        crashes = obs_dict["crashes"]

        # Distance to target
        disp = self.target_position - robot_pos
        dist = torch.norm(disp, dim=1)

        # Condition masks (mutually exclusive, priority order)
        exceed_mask = (
            (robot_pos < obs_dict["env_bounds_min"]).any(dim=1)
            | (robot_pos > obs_dict["env_bounds_max"]).any(dim=1)
        )
        arrive_mask = (~exceed_mask) & (
            dist < self.task_config.reward_parameters["d_min"]
        )
        collision_mask = (~exceed_mask) & (~arrive_mask) & (crashes > 0)
        progress_mask = (~exceed_mask) & (~arrive_mask) & (~collision_mask)

        # Compute each reward component
        reward = torch.zeros(self.num_envs, device=self.device)
        reward[exceed_mask] = self._reward_exceed()
        reward[arrive_mask] = self._reward_arrive()
        reward[collision_mask] = self._reward_collision()
        reward[progress_mask] = self._reward_progress(progress_mask)

        # All three event types terminate the episode
        terminations = exceed_mask | arrive_mask | collision_mask

        return reward, terminations, arrive_mask, exceed_mask

    def _reward_exceed(self):
        """Penalty for flying out of environment bounds."""
        return self.task_config.reward_parameters["exceed_penalty"]

    def _reward_arrive(self):
        """Bonus for reaching the target waypoint."""
        return self.task_config.reward_parameters["arrive_bonus"]

    def _reward_collision(self):
        """Penalty for colliding with an obstacle."""
        return self.task_config.reward_parameters["collision_penalty"]

    def _reward_progress(self, mask):
        """
        Dense shaping reward for non-terminal steps. Balances goal-reaching,
        flight stability and safety. All lambda coefficients are negative.

        Components:
        1. Distance:   λ_d * (log(d_hor) + |d_z|)
        2. Excess speed: λ_v * v_hor * max(0, v_hor - v_max)
        3. Direction:  λ_dir * ‖d_norm - v_norm‖
        4. Angular vel: λ_input * ‖ω_B‖₁
        5. Perception: λ_perc * (|v_body_y| + max(0, -v_body_x))

        Args:
            mask: Boolean tensor indicating which envs get this reward
        Returns:
            Reward tensor for masked envs
        """
        params = self.task_config.reward_parameters
        robot_pos = self.obs_dict["robot_position"][mask]
        target_pos = self.target_position[mask]
        vel_w = self.obs_dict["robot_linvel"][mask]
        body_angvel = self.obs_dict["robot_body_angvel"][mask]
        body_linvel = self.obs_dict["robot_body_linvel"][mask]

        disp = target_pos - robot_pos

        # 1. Distance penalty: λ_d * (log(d_hor) + |d_z|)
        d_hor = torch.norm(disp[:, :2], dim=1)
        d_z = torch.abs(disp[:, 2])
        r_dist = params["lambda_d"] * (torch.log(d_hor + 1e-6) + d_z)

        # 2. Excess speed penalty: λ_v * v_hor * max(0, v_hor - v_max)
        v_hor = torch.norm(vel_w[:, :2], dim=1)
        r_speed = params["lambda_v"] * v_hor * torch.clamp(v_hor - self.task_config.v_max, min=0.0)

        # 3. Direction alignment: λ_dir * ‖d_norm - v_norm‖
        d_norm = disp / (torch.norm(disp, dim=1, keepdim=True) + 1e-6)
        v_norm = vel_w / (torch.norm(vel_w, dim=1, keepdim=True) + 1e-6)
        r_dir = params["lambda_dir"] * torch.norm(d_norm - v_norm, dim=1)

        # 4. Angular velocity penalty: λ_input * ‖ω_B‖₁
        r_angvel = params["lambda_input"] * torch.sum(torch.abs(body_angvel), dim=1)

        # 5. Perception penalty: λ_perc * (|v_body_y| + max(0, -v_body_x))
        r_perc = params["lambda_perc"] * (
            torch.abs(body_linvel[:, 1]) + torch.clamp(-body_linvel[:, 0], min=0.0)
        )

        # Store component means for tensorboard
        self.logged_r_dist = float(r_dist.mean())
        self.logged_r_speed = float(r_speed.mean())
        self.logged_r_dir = float(r_dir.mean())
        self.logged_r_angvel = float(r_angvel.mean())
        self.logged_r_perc = float(r_perc.mean())

        return r_dist + r_speed + r_dir + r_angvel + r_perc
