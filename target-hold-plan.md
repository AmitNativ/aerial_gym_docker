# Plan: `target_hold` — new stubbed RL task for camera-based target tracking

## Implementation status

> Living checklist — update as items are completed or new work is discovered.

### Scaffold (done)
- [x] Package structure & `__init__.py` registry wiring
- [x] `target_hold_task_config.py` (with stub placeholders)
- [x] `target_hold_env_config.py`
- [x] `target_hold_robot_config.py` (RGB camera + IMU)
- [x] `target_hold_asset_config.py`
- [x] `target_hold_task.py` (BaseTask subclass, lifecycle wired, obs/reward stubbed)
- [x] `bbox_from_segmentation.py` (batched, fully functional)
- [x] `training/runner.py` (rl_games wrapper)
- [x] `training/ppo_target_hold.yaml` (stub PPO config)
- [x] `generate_target_boxes.py` + 32 cube URDFs

### Sensors to add
- [x] **Camera**
- [ ] **URDF** Take real URDF from GOLAN
- [ ] **IMU** - Add as second stage, wait for now

### Stubs to replace with real logic
- [x] **Observation space** — 22 dims: actor obs(12) + prev_actions(4) + privileged pos+linvel(6). Actor reads [0:12], GRU gets [12:16], Critic reads [0:22]
- [x] **Action space** — 4 dims: [thrust, roll_rate, pitch_rate, yaw_rate] in [-1,1]; `action_transformation_function` on task_config scales to controller units (thrust shifted to [0,1] then scaled; rates each scaled by configurable max). Controller: `lee_rates_control`
- [x] **Reward function** — dense penalties (area err, centering, vz, attitude, smoothness) + sparse terminations (vis loss, safety, crash) + curriculum (vis termination gated on 75% success)
- [x] **`reward_parameters`** — all coefficients in YAML `penalty:` section, injected by runner.py into task_config at startup
- [x] **Custom policy network** — asymmetric actor-critic: Actor FC(128)→FC(128)→GRU(64)→Linear(4); Critic FC(128)→FC(128)→Linear(1). Registered as `target_hold_actor_critic`
- [x] **PPO hyperparameters** — horizon=96 (~1s), lr=3e-4 adaptive, entropy=0.005, max_epochs=5000
- [x] **Spawning** - drone is spawned with different attitude and velocity and height, but always at origin. Target is spawned in any position on the ground plane visible in drone's camera, but within 50m to 120m top

### Timing / latency scaffolding (sampled but not consumed)
- [ ] **Action latency delay buffer** — `action_latency_ms` / `action_jitter_ms` tensors exist per-env but are never applied to actions
- [ ] **Camera rate subsampling** — `camera_rate_hz` sampled per-env but camera runs every step; need gate/hold logic
- [ ] **IMU rate subsampling** — `imu_rate_hz` sampled per-env but IMU runs every step; same

### Verification (from plan § Verification plan)
- [ ] Headless smoke test — env construction + 10 steps, check tensor shapes
- [ ] BBox sanity check — confirm seg mask contains `target_semantic_id`, bbox non-empty when visible
- [ ] GUI sanity check — drone + box render, camera attached, no Mesa-EGL crash
- [ ] Training wiring check — PPO spins up and takes a few iterations without error

# TODO:
- [ ] check entering full rotation matrix as rotation input for smooth space
- [ ] tune `seq_length` (currently 16) — if policy lacks memory increase it, if training is slow/unstable try shorter. Must divide `horizon_length` evenly

---

## Context

The user wants a new training environment for a **target-hold** task: a drone must keep a target (a box) inside its camera field of view while maintaining a desired distance. The physical task requires:

- An **RGB camera** and an **IMU** on the drone (real sensors wired, observation consumption stubbed)
- A **single target box** per env, with shape variety **across** envs and pose randomized **per reset**
- A **2D bounding box** of the target in the camera image, extracted from the **segmentation mask** produced by the Isaac Gym camera (target carries a unique `semantic_id`, bbox = min/max of pixels matching that id)
- Physics step, control decimation, action latency/jitter, camera rate, IMU rate **exposed in config** even though Aerial Gym doesn't yet consume latency/jitter
- Observation space, action space, reward function, and network are **stubs** — the user will supply real ones

The new package will sit in the workspace root (`/workspaces/aerial_gym_docker/target_hold/`) and follow the existing pattern (`.vscode/tasks.json`, `train_obstacle_avoidance.sh`): `python -m target_hold.training.runner --file=target_hold/training/ppo_target_hold.yaml ...`. It imports the aerial_gym package, registers its task/env/robot at import time, and delegates sim construction to `SimBuilder`.

## Decisions locked in from the user

| Question | Choice |
|---|---|
| Target box | Folder of pre-generated cube URDFs → use existing `randomly_pick_assets_from_folder` path |
| Sensors | Wire real RGB camera + IMU via `BaseQuadWithCameraImuCfg` base; observation consumption stubbed |
| BBox | **Segmentation-mask based** — Isaac Gym already produces a seg image when `segmentation_camera=True`; target carries a unique `semantic_id`; bbox = per-env min/max of matching pixel coords. Robust to occlusion and to any later camera-pose randomization. |
| Timing / latency | All timing fields in config, not yet consumed. Scaffolding only. |

## Assumptions (flag in PR if any are wrong)

- **Package name:** `target_hold/` at workspace root.
- **Controller:** `lee_position_control` in default config (position-level command is the natural fit for hold-distance). Easily swapped via `task_config.controller_name`.
- **Sim:** `base_sim` (`dt=0.01`, `substeps=1`) as default.
- **num_envs default:** 1024 (matches the existing navigation_with_obstacles task in `.vscode/tasks.json`); user will tune.
- **Robot name:** `target_hold_quad` — a new robot registration so we can give it a dedicated sensor_config without touching upstream. Inherits from `BaseQuadWithCameraImuCfg` (the existing combined camera+IMU quad config at `/app/aerial_gym/aerial_gym_simulator/aerial_gym/config/robot_config/base_quad_config.py`).
- **Camera config:** Start from `BaseDepthCameraConfig` but enable RGB capture. Isaac Gym path only (Warp has no RGB). See `/app/aerial_gym/aerial_gym_simulator/aerial_gym/sensors/isaacgym_camera_sensor.py:114-120,151` — RGB is already captured into `self.rgb_pixels`; the robot_manager exposes it at `global_tensor_dict["rgb_pixels"]` (`/app/aerial_gym/aerial_gym_simulator/aerial_gym/robots/robot_manager.py:149-160`).
- **use_warp:** `False` in config (Warp cannot produce RGB — `/app/aerial_gym/aerial_gym_simulator/aerial_gym/sensors/isaacgym_camera_sensor.py:136` comment confirms this).

## Directory layout (all new, all under `/workspaces/aerial_gym_docker/target_hold/`)

```
target_hold/
├── __init__.py                           # registers task/env/robot with aerial_gym registries at import
├── configs/
│   ├── __init__.py
│   ├── target_hold_task_config.py        # task_config class — stub obs/action/reward, timing fields
│   ├── target_hold_env_config.py         # env config — single target box asset, bounds
│   ├── target_hold_robot_config.py       # robot cfg — inherits BaseQuadWithCameraImuCfg, RGB-enabled camera
│   └── target_hold_asset_config.py       # target_box_asset_params (BaseAssetParams subclass)
├── tasks/
│   ├── __init__.py
│   └── target_hold_task.py               # TargetHoldTask(BaseTask) with analytic bbox projection
├── utils/
│   ├── __init__.py
│   └── bbox_from_segmentation.py         # batched mask → [x_min, y_min, x_max, y_max] + visibility flag
├── training/
│   ├── __init__.py
│   ├── runner.py                         # thin wrapper around aerial_gym's rl_games runner, imports target_hold first
│   └── ppo_target_hold.yaml              # stub PPO config (network name placeholder)
└── resources/
    ├── generate_target_boxes.py          # script to generate N cube URDFs with varied dims
    └── target_boxes/                     # output folder of cube_*.urdf files
```

## Per-file detail

### `target_hold/__init__.py`
Mirror the import-time registration style of `/app/aerial_gym/aerial_gym_simulator/aerial_gym/task/__init__.py:50-77`:

```python
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.env_registry import env_config_registry

from target_hold.configs.target_hold_task_config import task_config as target_hold_task_config
from target_hold.configs.target_hold_env_config import TargetHoldEnvCfg
from target_hold.configs.target_hold_robot_config import TargetHoldQuadCfg
from target_hold.tasks.target_hold_task import TargetHoldTask
from aerial_gym.robots.base_multirotor import BaseMultirotor  # or whatever class the base quad uses

env_config_registry.register("target_hold_env", TargetHoldEnvCfg)
robot_registry.register("target_hold_quad", BaseMultirotor, TargetHoldQuadCfg)
task_registry.register_task("target_hold_task", TargetHoldTask, target_hold_task_config)
```

Signatures verified against the registries:
- `task_registry.register_task(name, class, config)` — `/app/aerial_gym/aerial_gym_simulator/aerial_gym/registry/task_registry.py:6`
- `robot_registry.register(name, class, config)` — `/app/aerial_gym/aerial_gym_simulator/aerial_gym/registry/robot_registry.py:16`
- `env_config_registry.register(name, config)` — `/app/aerial_gym/aerial_gym_simulator/aerial_gym/registry/env_registry.py:10` (config only, no class)

**Verify during implementation:** the exact robot class to register for `BaseQuadWithCameraImuCfg` (likely `BaseMultirotor` — inspect what `base_quadrotor` uses in the upstream robot registration file alongside `aerial_gym/robots/__init__.py`).

### `target_hold/configs/target_hold_task_config.py`
Plain Python class (matching style of `/app/aerial_gym/aerial_gym_simulator/aerial_gym/config/task_config/position_setpoint_task_config.py`):

```python
class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "target_hold_env"
    robot_name = "target_hold_quad"
    controller_name = "lee_position_control"
    args = {}
    num_envs = 1024
    use_warp = False
    headless = True
    device = "cuda:0"

    # --- Stubs the user will replace ---
    observation_space_dim = 1           # placeholder
    privileged_observation_space_dim = 0
    action_space_dim = 4                # placeholder
    episode_len_steps = 500
    return_state_before_reset = False

    class target:
        desired_distance_m = 3.0         # stand-off the drone should hold from target
        desired_distance_tolerance_m = 0.5

    class bbox:
        image_width = 240                # must match robot camera_config.width
        image_height = 135               # must match robot camera_config.height
        target_semantic_id = 100         # MUST match TargetBoxAssetParams.semantic_id
        return_normalized = False        # if True, divide by (W,H)
        empty_bbox_value = [0.0, 0.0, 0.0, 0.0]  # returned when target not visible in frame

    class timing:
        # Physics is still driven by sim_config.dt and env_config.num_physics_steps_per_env_step.
        # These fields are SCAFFOLDING — the task reads them but does not yet apply delay buffers.
        physics_dt_s = 0.01              # must match sim_config (base_sim: 0.01)
        control_decimation = 1           # must match env_config.num_physics_steps_per_env_step_mean
        action_latency_ms_range = [0.0, 20.0]   # sampled per-env at reset
        action_jitter_ms_std = 2.0
        camera_rate_hz_range = [20.0, 30.0]     # per-env at reset
        imu_rate_hz_range = [100.0, 200.0]      # per-env at reset

    # --- Stub reward parameters (user will replace) ---
    reward_parameters = {
        "crash_penalty": -100.0,
        # TODO(user): fill in hold-distance, in-FOV, bbox-area shaping gains
    }
```

### `target_hold/configs/target_hold_env_config.py`
Single-box env, no walls. Style matches `/app/aerial_gym/aerial_gym_simulator/aerial_gym/config/env_config/env_with_obstacles.py` but stripped down. Key fields:

- `class env`: `num_envs` (overridden at runtime), `num_env_actions = 0`, `env_spacing`, `lower_bound_min`, `upper_bound_min`, `upper_bound_max` (large enough for desired-distance + FOV corner cases).
- `class env_config.include_asset_type`: only `{"target_box": True}`; all walls/panels/etc. absent.
- `class env_config.asset_type_to_dict_map`: `{"target_box": TargetBoxAssetParams}` (from `target_hold_asset_config.py`).
- `sample_timestep_for_latency = False` (no baked-in latency; we'll stub our own).

### `target_hold/configs/target_hold_asset_config.py`
Matches `object_asset_params` style in `/app/aerial_gym/aerial_gym_simulator/aerial_gym/config/asset_config/env_object_config.py:157-195`:

```python
class TargetBoxAssetParams(BaseAssetParams):
    num_assets = 1
    asset_folder = "<workspace>/target_hold/resources/target_boxes"   # cube_*.urdf files
    file = None                          # None → randomly_pick_assets_from_folder picks one per env
    # Position ratios — keep target inside env at a plausible distance from robot spawn
    min_state_ratio = [0.4, 0.3, 0.3, -np.pi, -np.pi, -np.pi, 1.0, 0,0,0, 0,0,0]
    max_state_ratio = [0.8, 0.7, 0.7,  np.pi,  np.pi,  np.pi, 1.0, 0,0,0, 0,0,0]
    semantic_id = 100                    # distinct ID so segmentation can isolate target later
    collision_mask = 0                   # pass-through; target is a visual/aim point only
```

The `BaseAssetParams` base lives in `/app/aerial_gym/aerial_gym_simulator/aerial_gym/config/asset_config/` — inspect and import from the same module used by `env_object_config.py`.

### `target_hold/configs/target_hold_robot_config.py`
Inherit from `BaseQuadWithCameraImuCfg` (at `/app/aerial_gym/aerial_gym_simulator/aerial_gym/config/robot_config/base_quad_config.py`). Override the camera config to enable RGB:

```python
class TargetHoldCameraCfg(BaseDepthCameraConfig):
    # Isaac Gym path captures depth + RGB + segmentation in one pass.
    # Config lives in /app/aerial_gym/aerial_gym_simulator/aerial_gym/config/sensor_config/camera_config/base_depth_camera_config.py
    width = 240
    height = 135
    horizontal_fov_deg = 87.0
    segmentation_camera = True           # REQUIRED — bbox util reads segmentation_pixels
    nominal_position = [0.10, 0.0, 0.03]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]
    randomize_placement = False          # fine to flip to True later — seg-based bbox is invariant to camera pose

class TargetHoldQuadCfg(BaseQuadWithCameraImuCfg):
    class sensor_config(BaseQuadWithCameraImuCfg.sensor_config):
        enable_camera = True
        camera_config = TargetHoldCameraCfg
        enable_imu = True
        imu_config = BaseImuConfig       # default IMU; user can override
```

**Note:** RGB is already captured by `IsaacGymCameraSensor` unconditionally. Segmentation output is gated by `segmentation_camera=True`. The task reads:
- `obs_dict["rgb_pixels"]` — shape `(num_envs, num_sensors, H, W, 4)`
- `obs_dict["segmentation_pixels"]` — shape `(num_envs, num_sensors, H, W)` dtype int32 (**verify this key name during implementation** — if not propagated into `global_tensor_dict` by `robot_manager`, we'll need a tiny patch there or to pull it directly from the sensor object)
- `obs_dict["imu_measurement"]` — shape `(num_envs, 6)`

### `target_hold/tasks/target_hold_task.py`
Subclass `BaseTask` (`/app/aerial_gym/aerial_gym_simulator/aerial_gym/task/base_task.py`). Model closely after `/app/aerial_gym/aerial_gym_simulator/aerial_gym/task/position_setpoint_task/position_setpoint_task.py` — same lifecycle, stubbed payload. Required pieces:

1. **`__init__(task_config, seed, num_envs, headless, device, use_warp)`** — mirror lines 21-70 of `position_setpoint_task.py`: call `SimBuilder().build_env(sim_name, env_name, robot_name, controller_name, ...)`, stash `self.sim_env`, allocate buffers.
2. **`reset()` / `reset_idx(env_ids)`** — delegate to `self.sim_env.reset()` / `reset_idx`, then re-sample per-env timing/latency values from `task_config.timing` ranges into tensors (unused by stub — just stored).
3. **`step(action)`** — `self.sim_env.step(action)`, pull `self.obs_dict = self.sim_env.get_obs()`, compute stub observation, compute stub reward (all zeros plus the `crash_penalty` when applicable), increment counter, return `(obs, reward, terminations, truncations, infos)`.
4. **`compute_observations()` stub** — read the following fields from `self.obs_dict` to *prove* wiring, but return a zero tensor of size `observation_space_dim`:
   - `obs_dict["robot_position"]`, `obs_dict["robot_orientation"]`, `obs_dict["robot_linvel"]`, `obs_dict["robot_angvel"]` — proprioceptive
   - `obs_dict["rgb_pixels"]` — camera
   - `obs_dict["segmentation_pixels"]` — seg mask (source for bbox)
   - `obs_dict["imu_measurement"]` — IMU
   - `obs_dict["obstacle_position"]`, `obs_dict["obstacle_orientation"]` — target pose (see `/app/aerial_gym/aerial_gym_simulator/aerial_gym/env_manager/IGE_env_manager.py:378-410`; shape `(num_envs, 1, 3)` / `(num_envs, 1, 4)`).
   - `bbox, visible = bbox_from_segmentation(seg_pixels, target_semantic_id)` — from the util; stored in `self.info["target_bbox"]` / `self.info["target_visible"]` for downstream logging, not yet fed into the observation.
5. **`compute_rewards_and_crashes()` stub** — zero vector plus `crash_penalty` on out-of-bounds, mirroring position_setpoint_task's `compute_reward` structure but returning zeros for the task-specific shaping.
6. **`render` / `close`** — delegate to `self.sim_env` (non-abstract pass-throughs matching position_setpoint_task).

### `target_hold/utils/bbox_from_segmentation.py`
A pure-torch, batched function — no rendering code here, just consumes the seg image the renderer already produced. Takes:

- `seg_pixels` — `(num_envs, num_sensors, H, W)` int32 segmentation image from `obs_dict["segmentation_pixels"]`
- `target_semantic_id` — int (from `task_config.bbox.target_semantic_id`)
- `return_normalized` — bool

Returns:
- `bbox` — `(num_envs, 4)` tensor `[x_min, y_min, x_max, y_max]` (zeros where target not visible)
- `visible` — `(num_envs,)` bool tensor, True iff at least one pixel matches the target id

Sketch:
```python
mask = (seg_pixels.squeeze(1) == target_semantic_id)      # (N, H, W)
visible = mask.any(dim=(-1, -2))                           # (N,)
# Per-env col/row presence vectors
cols_any = mask.any(dim=1)                                 # (N, W)
rows_any = mask.any(dim=2)                                 # (N, H)
x_coords = torch.arange(W, device=device)
y_coords = torch.arange(H, device=device)
x_min = torch.where(cols_any, x_coords, W).min(dim=1).values.float()
x_max = torch.where(cols_any, x_coords, -1).max(dim=1).values.float()
y_min = torch.where(rows_any, y_coords, H).min(dim=1).values.float()
y_max = torch.where(rows_any, y_coords, -1).max(dim=1).values.float()
bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
bbox = torch.where(visible[:, None], bbox, torch.zeros_like(bbox))
if return_normalized:
    bbox = bbox / torch.tensor([W, H, W, H], device=device)
return bbox, visible
```

No transform chain, no URDF parsing, no occlusion math. If the target sensor key differs from `"segmentation_pixels"`, grep `robot_manager.py` around line 149 for the actual `global_tensor_dict` key and swap.

### `target_hold/resources/generate_target_boxes.py`
Takes `--count N`, `--min-dim 0.2`, `--max-dim 1.5`, writes `target_boxes/cube_{i}.urdf` — each a single link with `<box size="lx ly lz"/>` geometry, matching the existing `cube.urdf` found in `/app/aerial_gym/aerial_gym_simulator/resources/models/environment_assets/walls/` (find and inspect it during implementation). Also writes a `box_dims.json` alongside so the task can recover box half-extents from the URDF filename without parsing URDF at runtime.

### `target_hold/training/runner.py`
Thin wrapper:

```python
import target_hold  # triggers registrations
from aerial_gym.rl_training.rl_games import runner as aerial_runner
if __name__ == "__main__":
    aerial_runner.main()  # or copy the __main__ block if the upstream has no main()
```

Confirm during implementation whether `aerial_gym/rl_training/rl_games/runner.py` has a reusable `main()` — if not, copy the arg-parsing block and invoke its logic with `task_name="target_hold_task"` forced.

### `target_hold/training/ppo_target_hold.yaml`
Start by copying `/app/aerial_gym/aerial_gym_simulator/aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml` and:
- Change `env_name` / `name` to `target_hold_task`
- Leave `network.name` as the rl_games default MLP (user will swap to their custom net later)
- Set `num_actors` to match `task_config.num_envs`

## Registration order caveat

`target_hold/__init__.py` must import **after** `aerial_gym` has registered its own task/robot/env names (so we don't overwrite them) and **before** the runner calls `task_registry.make_task("target_hold_task", ...)`. The runner wrapper's first line `import target_hold` achieves both.

## Critical references (read before coding each file)

| Area | File | Lines |
|---|---|---|
| Task lifecycle model | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/task/position_setpoint_task/position_setpoint_task.py` | 21-200 |
| BaseTask contract | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/task/base_task.py` | 10-55 |
| Task registration | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/task/__init__.py` | 50-77 |
| Robot config w/ camera+IMU | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/config/robot_config/base_quad_config.py` | (search `BaseQuadWithCameraImuCfg`) |
| RGB capture proof | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/sensors/isaacgym_camera_sensor.py` | 114-120, 151 |
| `rgb_pixels` tensor in env | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/robots/robot_manager.py` | 149-160 |
| Obstacle state tensors | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/env_manager/IGE_env_manager.py` | 378-410 |
| Asset config pattern | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/config/asset_config/env_object_config.py` | 157-195 |
| Random URDF picking | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/env_manager/asset_loader.py` | 46-57 |
| Env obstacles example | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/config/env_config/env_with_obstacles.py` | (whole file) |
| Segmentation image path | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/sensors/isaacgym_camera_sensor.py` | 107-112, 153-155, 209 |
| rl_games runner entry | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runner.py` | 82-161 |

## Verification plan

All verification runs happen inside the devcontainer.

1. **Generate URDF library:**
   ```bash
   cd /workspaces/aerial_gym_docker
   python -m target_hold.resources.generate_target_boxes --count 32 --min-dim 0.3 --max-dim 1.2
   ls target_hold/resources/target_boxes/ | head
   ```

2. **Headless smoke test — env construction + one step (no RL):** write a tiny sibling script or reuse the pattern in `/app/aerial_gym/aerial_gym_simulator/aerial_gym/examples/rl_env_example.py`:
   ```bash
   python -c "
   import torch, target_hold
   from aerial_gym.registry.task_registry import task_registry
   env = task_registry.make_task('target_hold_task', num_envs=4, headless=True)
   obs, *_ = env.reset()
   action = torch.zeros((4, env.task_config.action_space_dim), device='cuda:0')
   for _ in range(10):
       obs, rew, term, trunc, info = env.step(action)
   print('rgb:', env.obs_dict['rgb_pixels'].shape)
   print('imu:', env.obs_dict['imu_measurement'].shape)
   print('obstacle_pos:', env.obs_dict['obstacle_position'].shape)
   print('bbox (info):', info.get('target_bbox', None).shape)
   "
   ```
   Expected: no exceptions, shapes `(4, 1, 135, 240, 4)`, `(4, 6)`, `(4, 1, 3)`, `(4, 4)`.

3. **BBox sanity check:** spawn a single env headlessly, verify (a) `obs_dict["segmentation_pixels"]` contains pixels equal to `target_semantic_id` when the target is in front of the drone, (b) the bbox util returns a non-empty bbox with `visible=True`, and (c) dragging the drone so the target is out of frame flips `visible=False`.

4. **GUI sanity check** (optional, slow): `./run.sh` then inside the container run the same script with `headless=False, num_envs=4` — confirm the drone and target box render, camera is attached, and no Mesa-EGL crash.

5. **Training wiring check (the stubs will NOT learn — reward is zero — just confirm the pipeline accepts the task):**
   ```bash
   python -m target_hold.training.runner \
       --file=target_hold/training/ppo_target_hold.yaml \
       --train --num_envs=128 --headless=True
   ```
   Expect PPO to spin up, take a few iterations, and exit cleanly when interrupted. User will replace obs/reward/network and run real training afterward.

## Out of scope (user will do later)

- Real observation/action spaces and shapes
- Real reward (FOV-keeping + distance-hold shaping)
- Custom policy network (likely CNN over RGB + MLP over IMU — but user's call)
- Actually *consuming* the latency/jitter config fields (delay buffers, subsampled sensor reads)
- Warp-path RGB support (not available upstream)
- Per-reset shape randomization of a single box (not supported by aerial_gym today; folder-picking gives cross-env variety instead)
- Segmentation collision with other asset ids — currently only the target has `semantic_id=100` in this env, but if later envs add other objects, pick a unique id and document it in `target_hold_task_config.bbox.target_semantic_id`
