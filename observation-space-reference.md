# Aerial Gym Simulator — Complete Observation Space Reference

Everything the simulator can provide to a task through `obs_dict` / `global_tensor_dict`.

`N` = `num_envs` throughout.

---

## Robot state (always available)

| Key | Shape | Contents |
|---|---|---|
| `robot_position` | `(N, 3)` | World-frame position [x, y, z] |
| `robot_orientation` | `(N, 4)` | Quaternion [qx, qy, qz, qw] |
| `robot_euler_angles` | `(N, 3)` | Roll, pitch, yaw |
| `robot_linvel` | `(N, 3)` | World-frame linear velocity |
| `robot_angvel` | `(N, 3)` | World-frame angular velocity |
| `robot_body_linvel` | `(N, 3)` | Body-frame linear velocity |
| `robot_body_angvel` | `(N, 3)` | Body-frame angular velocity |
| `robot_vehicle_orientation` | `(N, 4)` | Vehicle-frame orientation quaternion |
| `robot_vehicle_linvel` | `(N, 3)` | Vehicle-frame linear velocity |
| `robot_state_tensor` | `(N, 13)` | Full root state [pos(3), quat(4), linvel(3), angvel(3)] |
| `robot_mass` | `(N,)` | Robot mass per env |
| `robot_inertia` | `(N, 3, 3)` | Inertia matrix per env |

## Robot actions

| Key | Shape | Contents |
|---|---|---|
| `robot_actions` | `(N, num_actions)` | Current actions being applied |
| `robot_prev_actions` | `(N, num_actions)` | Previous step's actions |

## Robot forces / contacts (always available)

| Key | Shape | Contents |
|---|---|---|
| `robot_contact_force_tensor` | `(N, 3)` | Net contact force on robot root link |
| `robot_force_tensor` | `(N, num_robot_bodies, 3)` | Forces on each robot rigid body |
| `robot_torque_tensor` | `(N, num_robot_bodies, 3)` | Torques on each robot rigid body |
| `crashes` | `(N,)` | Collision flag (contact force > threshold) |
| `truncations` | `(N,)` | Episode-length truncation flag |

## Obstacle / target state (conditional — only if env has assets)

| Key | Shape | Contents |
|---|---|---|
| `obstacle_position` | `(N, num_obs, 3)` | World-frame positions |
| `obstacle_orientation` | `(N, num_obs, 4)` | Quaternions |
| `obstacle_euler_angles` | `(N, num_obs, 3)` | Roll, pitch, yaw |
| `obstacle_linvel` | `(N, num_obs, 3)` | World-frame linear velocity |
| `obstacle_angvel` | `(N, num_obs, 3)` | World-frame angular velocity |
| `obstacle_body_linvel` | `(N, num_obs, 3)` | Body-frame linear velocity |
| `obstacle_body_angvel` | `(N, num_obs, 3)` | Body-frame angular velocity |
| `obstacle_force_tensor` | `(N, num_obs, 3)` | Forces on obstacles |
| `obstacle_torque_tensor` | `(N, num_obs, 3)` | Torques on obstacles |
| `env_asset_state_tensor` | `(N, num_obs, 13)` | Full root state of all env assets |

## Environment bounds / physics constants (always available)

| Key | Shape | Contents |
|---|---|---|
| `env_bounds_min` | `(N, 3)` | Lower bounds of env space |
| `env_bounds_max` | `(N, 3)` | Upper bounds of env space |
| `gravity` | `(N, 3)` | Gravity vector [0, 0, -9.81] |
| `dt` | scalar | Simulation timestep (e.g. 0.01) |

## Camera sensor (conditional — `enable_camera=True`)

| Key | Shape | Contents |
|---|---|---|
| `depth_range_pixels` | `(N, num_sensors, H, W)` | Depth image (meters) |
| `rgb_pixels` | `(N, num_sensors, H, W, 4)` | RGBA image (Isaac Gym path only, not Warp) |
| `segmentation_pixels` | `(N, num_sensors, H, W)` int32 | Per-pixel semantic ID (requires `segmentation_camera=True`) |

## IMU sensor (conditional — `enable_imu=True`)

| Key | Shape | Contents |
|---|---|---|
| `imu_measurement` | `(N, 6)` | [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] in sensor frame |
| `force_sensor_tensor` | `(N, 6)` | Raw force sensor [force(3), torque(3)] from Isaac Gym |

### IMU details

- Accelerometer derived from Isaac Gym force sensor attached to `base_link`, divided by robot mass
- Gyroscope read from `robot_body_angvel`, rotated into sensor frame
- Default config (`BaseImuConfig`): not gravity-compensated (real-IMU behavior, stationary reads ~+9.81 m/s² upward)
- Noise model: white Gaussian noise + Brownian random-walk bias, both re-sampled at reset
- Sensor misalignment: randomizable orientation offset (`sensor_quats`)
- Default is body-frame (`world_frame=False`)

## LiDAR sensor (conditional — `enable_lidar=True`)

| Key | Shape | Contents |
|---|---|---|
| `depth_range_pixels` | `(N, num_sensors, H, W)` | Range data (shares key with camera depth) |
| `segmentation_pixels` | `(N, num_sensors, H, W)` int32 | Per-point semantic ID |

## DOF state (conditional — reconfigurable robots with joints)

| Key | Shape | Contents |
|---|---|---|
| `dof_state_tensor` | `(N, num_dofs, 2)` | Joint [position, velocity] |
| `dof_position_setpoint_tensor` | `(N, num_dofs)` | Position setpoints |
| `dof_velocity_setpoint_tensor` | `(N, num_dofs)` | Velocity setpoints |
| `dof_effort_tensor` | `(N, num_dofs)` | Effort commands |

## Raw Isaac Gym tensors (always available, low-level)

| Key | Shape | Contents |
|---|---|---|
| `vec_root_tensor` | `(N, num_actors, 13)` | All actor root states in one tensor |
| `rigid_body_state_tensor` | `(total_bodies, 13)` | Every rigid body in the sim |
| `global_force_tensor` | `(total_bodies, 3)` | Applied forces |
| `global_torque_tensor` | `(total_bodies, 3)` | Applied torques |
| `global_contact_force_tensor` | `(N, bodies_per_env, 3)` | Contact forces on all bodies |

---

## What existing tasks actually use

**Position setpoint task** — 13 dims from ground-truth state only:
- `robot_position`, `robot_orientation`, `robot_body_linvel`, `robot_body_angvel`

**Navigation task** — 13 + 4 + 64 = 81 dims:
- Same 13 ground-truth dims + prev actions (4) + VAE-encoded depth image (64 latent dims from `depth_range_pixels` fed through a pretrained VAE)

No upstream task uses IMU, RGB, segmentation, obstacle state, contact forces, or inertia in its observation vector.

---

## Source files

| Area | File |
|---|---|
| Env manager (most state tensors) | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/env_manager/IGE_env_manager.py` |
| Robot manager (sensors, mass, inertia) | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/robots/robot_manager.py` |
| Base multirotor (body-frame vels, euler) | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/robots/base_multirotor.py` |
| Camera sensor | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/sensors/isaacgym_camera_sensor.py` |
| IMU sensor | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/sensors/imu_sensor.py` |
| Base sensor (tensor wiring) | `/app/aerial_gym/aerial_gym_simulator/aerial_gym/sensors/base_sensor.py` |
