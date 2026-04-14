"""Env config for the target-hold task.

A minimal env that spawns only the target box asset. No walls, no clutter.
Bounds are sized so the target sits ahead of the drone's spawn column with
enough room for the drone to maintain a stand-off distance.
"""

from target_hold.configs.target_hold_asset_config import target_box_asset_params


class TargetHoldEnvCfg:
    class env:
        num_envs = 64  # overridden by task_config at runtime
        num_env_actions = 0  # no env-side actions (obstacles are static)
        env_spacing = 5.0

        # Timing knobs that the *env manager* honors. Task-level latency/jitter
        # knobs live in TargetHoldTaskCfg.timing and are scaffolding for now.
        num_physics_steps_per_env_step_mean = 1
        num_physics_steps_per_env_step_std = 0

        render_viewer_every_n_steps = 1
        collision_force_threshold = 0.05
        manual_camera_trigger = False
        reset_on_collision = True
        create_ground_plane = True
        sample_timestep_for_latency = False  # don't perturb physics timestep
        perturb_observations = False
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False

        use_warp = False  # Isaac Gym path is required for RGB camera output

        # World bounds — target asset's min/max_state_ratio is interpolated
        # between these. Drone spawns near x=0.1*bound, target sits at ~0.5-0.8*x.
        lower_bound_min = [-200.0, -200.0, -1.0]
        lower_bound_max = [-200.0, -200.0, -1.0]
        upper_bound_min = [200.0, 200.0, 150.0]
        upper_bound_max = [200.0, 200.0, 150.0]

    class env_config:
        include_asset_type = {
            "target_box": True,
        }

        asset_type_to_dict_map = {
            "target_box": target_box_asset_params,
        }
