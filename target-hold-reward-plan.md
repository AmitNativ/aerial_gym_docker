# Plan: Reward function for target_hold

## Context

The target_hold task needs a reward function that teaches the drone to keep a ground target inside the camera FOV while maintaining stable flight. The reward is fully computed from tensors already available in the task — no new sensor wiring needed.

Key constraint: the actor sees 12 dims (quat, angvel, bbox, vz) while the critic sees all 22 dims (including world position and velocity). The reward function can use any tensor from `obs_dict` since it runs inside the task, not inside the network.

## Reward components

All `lambda_*` coefficients are stored in `reward_parameters` in the YAML config so they can be tuned without code changes.

### Dense rewards (every step)

#### 1. Distance maintenance via bbox area

Penalize change in bbox area relative to episode start (drift away / toward target) and the rate of area change (prevents oscillation).

```
area = (br_x - tl_x) * (br_y - tl_y)          # in normalized [-1,1] coords
area_err = area - area_at_reset                 # deviation from initial area
d_area = area - area_prev_step                  # rate of change

r_dist = lambda_area_err * area_err**2 + lambda_d_area * d_area**2
```

Requires storing `self.area_at_reset` (per-env, set in `reset_idx`) and `self.area_prev` (per-env, updated each step).

#### 2. Target horizontal centering

Penalize the horizontal offset of the bbox center from the image center. Quadratic — always negative, stronger near edges.

```
cx_bbox = (tl_x + br_x) / 2.0                  # in [-1, 1]
r_horiz = lambda_horiz * cx_bbox**2
```

#### 3. Target vertical centering (deadzone)

Penalize only when the bbox center is outside a vertical deadzone radius from the image center. Allows the target to sit slightly off-center vertically without penalty.

```
cy_bbox = (tl_y + br_y) / 2.0                  # in [-1, 1]
vert_err = max(|cy_bbox| - deadzone_radius, 0)
r_vert = lambda_vert * vert_err**2
```

`deadzone_radius` is a parameter (e.g. 0.3 in normalized coords).

#### 4. Altitude stability via vz

Penalize vertical velocity to discourage altitude changes.

```
r_vz = lambda_vz * vz**2
```

#### 5. Attitude penalty

Penalize pitch and roll to keep the drone roughly level.

```
# Extract roll, pitch from quaternion (or use robot_euler_angles)
r_att = lambda_att * (pitch**2 + roll**2)
```

#### 6. Action smoothness

Penalize the change in actions between consecutive steps.

```
r_smooth = lambda_smooth * ||a_t - a_{t-1}||**2
```

`a_t` and `a_{t-1}` are the raw policy outputs [-1, 1] (already stored as `self.actions` / `self.prev_actions`).

### Sparse penalties and termination

#### 7. Visibility loss (target exits FOV)

When bbox area drops to zero (target lost from frame), apply a strong penalty. Termination is curriculum-gated: don't terminate on visibility loss until success rate reaches 75%.

```
target_lost = ~self.target_visible                # from bbox_from_segmentation
r_vis_loss = torch.where(target_lost, lambda_vis_loss, 0.0)

# Terminate only if curriculum allows it
terminate_on_vis = target_lost & self.curriculum_terminate_on_vis
```

Minimum bbox area threshold: set by the smallest person-like target (~0.4×1.5 m) at 50 m — the closest spawn distance. That gives ~81 px² (4.6 × 17.4 px). Use **half** of that as the threshold: **40 px²**. If the bbox is smaller than this, the target is too far or too small to meaningfully track. In normalized [-1,1] coords: `40 / (W * H) * 4 ≈ 0.00124`.

```
min_bbox_area_px = 40.0                              # ~half of smallest person at 50m
min_bbox_area_normalized = min_bbox_area_px / (W * H) * 4.0  # ≈ 0.00124
bbox_too_small = (area > 0) & (area < min_bbox_area_normalized)
# Treat as target_lost for reward purposes
```

#### 8. Safety — excessive attitude

Strong penalty and immediate termination if roll or pitch exceeds safety limits.

```
unsafe = (|pitch| > safety_pitch_limit) | (|roll| > safety_roll_limit)
r_safety = torch.where(unsafe, lambda_safety, 0.0)
terminate_on_safety = unsafe                        # always terminate
```

Default limits: ±45° (π/4).

#### 9. Crash

Strong penalty and immediate termination on collision (already detected by `obs_dict["crashes"]`).

```
r_crash = torch.where(crashes > 0, lambda_crash, 0.0)
terminate_on_crash = crashes > 0                    # always terminate
```

### Total reward

```
reward = (r_dist + r_horiz + r_vert + r_vz + r_att + r_smooth
          + r_vis_loss + r_safety + r_crash)
```

### Curriculum

Track success rate over a rolling window. "Success" = episode ended by truncation (time limit), not termination (crash/visibility/safety).

```
success_rate = truncations / (truncations + terminations)
```

When `success_rate >= 0.75`:
- Enable termination on visibility loss (`self.curriculum_terminate_on_vis = True`)

Start with `curriculum_terminate_on_vis = False` so the agent first learns to fly stably before being punished for losing the target.

## Default parameters (YAML)

```yaml
penalty:
  # Dense (all negative — closer to 0 is better)
  lambda_area_err: -1.0
  lambda_d_area: -0.5
  lambda_horiz: -0.5
  lambda_vert: -0.5
  vert_deadzone_radius: 0.3        # normalized [-1,1] coords (not a penalty)
  lambda_vz: -0.1
  lambda_att: -0.1
  lambda_smooth: -0.01

  # Sparse / termination
  lambda_vis_loss: -10.0
  min_bbox_area_px: 40.0           # half of smallest person at 50m (81 px²)
  lambda_safety: -10.0
  safety_pitch_limit_deg: 45.0
  safety_roll_limit_deg: 45.0
  lambda_crash: -10.0

  # Curriculum
  curriculum_success_threshold: 0.75
  curriculum_check_window: 2048    # episodes
```

## State buffers needed

| Buffer | Shape | Set at | Updated at |
|---|---|---|---|
| `self.area_at_reset` | `(num_envs,)` | `reset_idx` (after frustum placement + first obs) | — |
| `self.area_prev` | `(num_envs,)` | `reset_idx` | every step |
| `self.curriculum_terminate_on_vis` | `bool` | `__init__` (False) | checked after each log window |
| `self.episode_success_count` | `int` | `__init__` (0) | incremented on truncation |
| `self.episode_total_count` | `int` | `__init__` (0) | incremented on any episode end |

## Files to modify

| File | Change |
|---|---|
| `target_hold/training/ppo_target_hold.yaml` | Add all `reward.*` parameters |
| `target_hold/configs/target_hold_task_config.py` | Remove old `reward_parameters` dict, replace with reference to YAML |
| `target_hold/tasks/target_hold_task.py` | Replace `compute_rewards_and_crashes` stub; add buffers in `__init__`; update `reset_idx` to set `area_at_reset`; add curriculum logic |

## Implementation notes

- Roll/pitch extraction: use `obs_dict["robot_euler_angles"]` (always available in `global_tensor_dict`) rather than re-deriving from quaternion.
- All reward terms are batched tensors — no per-env loops.
- The `reward_parameters` dict is loaded from YAML at runtime; the task reads them as `self.task_config.reward_parameters["lambda_area_err"]` etc. and converts to tensors once in `__init__`.
- **`area_at_reset` computation**: must happen after `_place_target_in_frustum` AND after the sim renders a frame so the segmentation mask reflects the new target position. Sequence in `reset_idx`:
  1. `sim_env.reset_idx(env_ids)` — places robot + target
  2. `_place_target_in_frustum(env_ids)` — overrides target position, calls `write_to_sim`
  3. `sim_env.step(zero_actions)` — advances one physics step to trigger camera render
  4. Read `segmentation_pixels`, run `bbox_from_segmentation` to get the initial bbox
  5. Compute `area_at_reset[env_ids] = (br_x - tl_x) * (br_y - tl_y)` in normalized coords
  6. Set `area_prev[env_ids] = area_at_reset[env_ids]`
  
  The zero-action step is necessary because Isaac Gym cameras only render during `step()` / `render_all_camera_sensors()`. Without it, the segmentation image still shows the pre-reset state. This adds one wasted physics step per reset, which is negligible.

## Verification

1. Zero-action test: reset, step with zero actions for 10 steps. Reward should be near zero (no motion, target visible, level attitude).
2. Spin test: apply max yaw_rate. `r_smooth` should spike on first step, `r_horiz` should grow as target drifts.
3. Pitch-up test: apply strong pitch. `r_att` should grow, eventually `r_safety` triggers termination at 45°.
4. Curriculum test: run many episodes. Verify `curriculum_terminate_on_vis` stays False until success rate exceeds 75%.
