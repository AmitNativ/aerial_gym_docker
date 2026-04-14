# Plan: Drone & target spawn logic

## Context

The drone should always spawn at the world origin (x=0, y=0) at 50–120 m altitude, with random orientation, velocity up to 7 m/s, and angular velocity. The target should spawn on the ground (z=0) within the camera's field of view from the drone's spawn pose.

## Drone spawn — override init_config

Override `init_config` in `TargetHoldQuadCfg` (in `target_hold_robot_config.py`).

Env bounds: x[-200,200], y[-200,200], z[-1,150]. Range = [400, 400, 151].

| Field | Min | Max | Notes |
|---|---|---|---|
| x ratio | 0.5 | 0.5 | → x=0 (center) |
| y ratio | 0.5 | 0.5 | → y=0 (center) |
| z ratio | 0.338 | 0.801 | → z ≈ 50–120 m |
| roll | -π/6 | π/6 | ±30° |
| pitch | -π/6 | π/6 | ±30° |
| yaw | -π | π | full heading |
| vx | -4.04 | 4.04 | m/s (7/√3, so |v| ≤ 7) |
| vy | -4.04 | 4.04 | m/s |
| vz | -4.04 | 4.04 | m/s |
| wx | -3.0 | 3.0 | rad/s (FPV-class angular rates) |
| wy | -3.0 | 3.0 | rad/s |
| wz | -3.0 | 3.0 | rad/s |

## Target spawn — in camera frustum on ground

After `sim_env.reset_idx()`, override target position to be within the camera's ground footprint.

**Approach: back-project a random pixel to the ground plane.**

1. After reset, read drone position and orientation
2. Compute camera world pose (drone pose + camera offset [0.10, 0, 0.03])
3. Pick a random pixel (u, v) within the inner 60% of the image (20% margin from each edge, so target is fully visible and not immediately lost)
4. Back-project to a 3D ray in camera frame using intrinsics (fx = fy = W/2 / tan(hfov/2))
5. Rotate ray to world frame using camera world orientation
6. Intersect ray with z=0 ground plane
7. If ray doesn't hit ground (looks upward) or hits beyond max_range, resample up to 10 times; fallback to directly below drone
8. That's the target position

This guarantees the target is visible at spawn.

**Isaac Gym camera convention**: camera looks along -z in body frame. A pixel (u, v) maps to ray:
```
ray_cam = [(u - cx) / fx, -(v - cy) / fy, -1]  (y/z flipped for IG convention)
```
Then rotated to world frame by the drone quaternion.

**Edge case**: if the drone is pitched upward, some rays may not hit the ground. The retry loop (up to 10 attempts) handles this, with a fallback to directly below the drone.

## Files modified

| File | Change |
|---|---|
| `target_hold/configs/target_hold_robot_config.py` | Added `init_config` override to `TargetHoldQuadCfg` |
| `target_hold/tasks/target_hold_task.py` | Added `_place_target_in_frustum(env_ids)`, updated `reset()` and `reset_idx()` |

## Verification

```python
# After reset, verify target is visible in the camera
seg = env.obs_dict["segmentation_pixels"]
bbox, visible = bbox_from_segmentation(seg, target_semantic_id=100, ...)
assert visible.all(), "Target should be visible at spawn"

# Verify target is on ground
target_pos = env.obs_dict["obstacle_position"][:, 0, :]
assert (target_pos[:, 2].abs() < 0.1).all(), "Target should be on ground"

# Verify drone at origin
drone_pos = env.obs_dict["robot_position"]
assert (drone_pos[:, 0].abs() < 1.0).all(), "Drone x should be ~0"
assert (drone_pos[:, 1].abs() < 1.0).all(), "Drone y should be ~0"
assert (drone_pos[:, 2] >= 49.0).all() and (drone_pos[:, 2] <= 121.0).all(), "Drone altitude 50-120m"
```
