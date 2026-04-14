# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **Docker / devcontainer harness** for training drone RL policies on top of the [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) and **NVIDIA Isaac Gym Preview 4**. The simulator and Isaac Gym are **not in this repo** — they are baked into the container image. This workspace holds the Dockerfile, setup/run scripts, and (in sibling Python packages) the custom tasks, networks, and training configs that plug into the Aerial Gym registries.

### Where things live inside the container

| Path | Contents |
|------|----------|
| `/opt/isaacgym/` | Isaac Gym Preview 4 install (examples at `python/examples/`) |
| `/app/aerial_gym/aerial_gym_simulator/` | Aerial Gym Simulator (pip-installed `-e`, browsable as reference) |
| `/workspaces/aerial_gym_docker/` | This repo — where custom training code and RL runs live |

When reasoning about simulator internals, **read from `/app/aerial_gym/aerial_gym_simulator/`** — it's the authoritative source for observation/action spaces, robot configs, controllers, sensors, and task base classes. If local code conflicts with upstream docs (https://ntnu-arl.github.io/aerial_gym_simulator/), flag the conflict explicitly.

## Build & run

### Container image
The Dockerfile is **`Dockerfile.base`** (multi-stage: `base → isaacgym → aerial_gym → dev`). There is no `docker-compose.yml` in the repo.

```bash
# First-time setup (extracts IsaacGym_Preview_4_Package.tar.gz, creates dirs)
./setup.sh

# Build the image — tag MUST be aerial-gym:base so the devcontainer can FROM it
docker build -f Dockerfile.base -t aerial-gym:base .

# GUI run (helper script handles xhost + X11/GPU flags)
./run.sh

# Sanity check an Isaac Gym example with GUI
./test_gui.sh
```

### Devcontainer (primary dev workflow)
`.devcontainer/Dockerfile` starts `FROM aerial-gym:base` and adds a non-root user matching host UID/GID. VSCode "Reopen in Container" will rebuild this layer — the `aerial-gym:base` image must exist first. The devcontainer bind-mounts this repo at `/workspaces/aerial_gym_docker` and runs `xhost +local:docker` on start.

### Running training (inside container)

Training is launched from **this repo** as Python modules; custom packages live at the workspace root and are invoked with `python -m <pkg>.training.runner`. Pattern (see `.vscode/tasks.json`, `train_obstacle_avoidance.sh`):

```bash
cd /workspaces/aerial_gym_docker
python -m <pkg>.training.runner \
    --file=<pkg>/training/<config>.yaml \
    --train \
    --num_envs=4096 \
    --headless=True

# Play / eval a checkpoint
python -m <pkg>.training.runner \
    --file=<pkg>/training/<config>.yaml \
    --play \
    --num_envs=64 \
    --headless=False \
    --checkpoint=runs/<run_name>/nn/<name>.pth
```

Packages that `.vscode/tasks.json` already wires up (create them under the workspace root as needed): `simple_hover_snn/`, `navigation_with_obstacles/`, `simple_obstacle_avoidance/`. The SNN design is in [aerial-gym-snn-plan.md](aerial-gym-snn-plan.md).

### Upstream smoke tests

```bash
cd /opt/isaacgym/python/examples && python joint_monkey.py --headless
cd /app/aerial_gym/aerial_gym_simulator && python aerial_gym/examples/position_control_example.py --headless
```

## Architecture — Aerial Gym Simulator

Everything is wired together through **registries** in `/app/aerial_gym/aerial_gym_simulator/aerial_gym/registry/`:

- `task_registry` — maps task name → `(task_class, task_config)`; `make_task(name, num_envs, headless, ...)` instantiates a `VecEnv`-style task.
- `robot_registry`, `controller_registry`, `env_registry`, `sim_registry` — same pattern for robots (multirotor, ROV, Morphy), low-level controllers, environment presets (empty / obstacles / trees / etc.), and sim backends.

A task composes: `sim_config` + `env_config` (world + obstacles) + `robot_config` (+ `controller_config`) + `sensor_config`. Config classes live under `aerial_gym/config/{sim,env,robot,controller,sensor,asset,task}_config/`; concrete tasks under `aerial_gym/task/` (e.g. `position_setpoint_task/`, `navigation_task/`, `custom_task/`). Two env managers exist — Isaac-Gym-native (`IGE_env_manager.py`) and warp-based (`warp_env_manager.py`) — selected by `use_warp`.

**To add a new task:** subclass `BaseTask`, write a config, and register both via `task_registry.register_task(...)` at import time. The runner only needs the task *name*; wiring is done by the registry.

### RL training stack

`aerial_gym/rl_training/` ships three runners: `rl_games/`, `cleanrl/`, `sample_factory/`. The canonical one is **rl_games** (`rl_training/rl_games/runner.py`) — it wraps the task as a `gym.Env`, registers it with `rl_games.common.env_configurations` / `vecenv`, then runs PPO from a YAML config. Baseline configs: `ppo_aerial_quad.yaml` (position control), `ppo_aerial_quad_navigation.yaml` (depth-based nav).

Custom networks (e.g. SNN) are added by implementing a `rl_games` `NetworkBuilder` and registering it *before* the runner builds the agent:

```python
from rl_games.algos_torch import model_builder
model_builder.register_network('snn_actor_critic', SNNNetworkBuilder)
```

Then reference `network: { name: snn_actor_critic, ... }` in the YAML. Do not fork the upstream runner; register at import time and let the name-based lookup pick it up.

### Observation / action shapes worth knowing

- Position-control task: **13-D** observation (pose + linear/angular velocities + goal-relative terms), continuous actions sized to the robot's action space.
- Navigation task: adds depth-image features (often via a VAE encoder) on top of proprioceptive state.

Confirm shapes by reading the relevant `task_config` / `robot_config` — don't guess. Tensor layout follows Isaac Gym's `(num_envs, ...)` convention.

## Critical container-level gotchas

1. **Mesa EGL must stay removed.** `Dockerfile.base` deletes `libEGL_mesa.so.0*` and `50_mesa.json` to stop segfaults during rendering — do not reinstall Mesa EGL.
2. **OpenCV is pinned to `4.5.5.64`** to avoid `cv2.dnn.DictValue` errors — do not upgrade without testing.
3. **Isaac Gym tarball is manual.** `IsaacGym_Preview_4_Package.tar.gz` must be present in the repo root before building (requires an NVIDIA developer account to download).
4. **X11/GPU for GUI:** run `xhost +local:docker` on the host before GUI mode; devcontainer does this via `initializeCommand`.
5. **VRAM scales with `num_envs`.** Rough starting points: 8 GB GPU → 256–1024; 24 GB → 4096; 40 GB A100 → 8192–10240. Minibatch size in the PPO YAML must be reduced in lockstep — see recent commits for tuning history.

## Working conventions in this repo

- New training code goes in the workspace root as a Python package (`<pkg>/training/runner.py`, `<pkg>/training/<config>.yaml`, optionally `<pkg>/tasks/`, `<pkg>/networks/`), following `.vscode/tasks.json`.
- Register custom tasks/networks at import time so name-based lookup resolves them.
- Treat `/app/aerial_gym/aerial_gym_simulator/` and `/opt/isaacgym/` as read-only reference unless the user explicitly asks for a local patch — edits to those paths won't survive an image rebuild.

## Additional references
- Isaac Gym: https://developer.nvidia.com/isaac-gym, https://github.com/isaac-sim/IsaacGymEnvs
- Aerial Gym: https://ntnu-arl.github.io/aerial_gym_simulator/, https://github.com/emNavi/AirGym
- snnTorch: https://snntorch.readthedocs.io/en/latest/
- Norse: https://norse.github.io/notebooks/intro_norse.html

## Isaac Gym API cheatsheet
```python
from isaacgym import gymapi, gymtorch, gymutil
# gymapi  — main simulation API (create_sim, create_env, actors, tensors)
# gymtorch — PyTorch zero-copy tensor interop (wrap_tensor, refresh_*_tensor)
# gymutil — arg parsing, asset helpers
```
