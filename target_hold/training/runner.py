"""Training entry point for target_hold_task.

Mirrors the argument handling of
``/app/aerial_gym/aerial_gym_simulator/aerial_gym/rl_training/rl_games/runner.py``
but registers only the target_hold task. Importing ``target_hold`` is enough to
register the task / env / robot with Aerial Gym's registries; here we
additionally hook the task into ``rl_games.env_configurations``.

Usage (from workspace root):
    python -m target_hold.training.runner \\
        --file=target_hold/training/ppo_target_hold.yaml \\
        --train --num_envs=128 --headless=True
"""

import distutils
import os

import isaacgym  # noqa: F401  — must be imported before torch
import torch  # noqa: F401
import yaml

from rl_games.common import env_configurations, vecenv

# Import the upstream runner module for its shared wrapper classes + task
# registrations (AERIAL-RLGPU vecenv is registered once at module import).
from aerial_gym.rl_training.rl_games import runner as aerial_runner
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.helpers import parse_arguments

# Importing the package triggers task / robot / env registration.
import target_hold  # noqa: F401

# Register the custom asymmetric actor-critic network with rl_games.
from rl_games.algos_torch import model_builder
from target_hold.networks.target_hold_network import TargetHoldNetworkBuilder

model_builder.register_network("target_hold_actor_critic", TargetHoldNetworkBuilder)


# Register the target_hold task with rl_games' env_configurations. The
# AERIAL-RLGPU vecenv type itself is already registered by aerial_runner on
# module import (see aerial_gym/rl_training/rl_games/runner.py).
env_configurations.register(
    "target_hold_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("target_hold_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)


def get_args():
    from isaacgym import gymutil  # noqa: F401 — kept to match aerial_runner's flow

    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "required": False},
        {"name": "--train", "action": "store_true", "required": False},
        {"name": "--play", "action": "store_true", "required": False},
        {"name": "--checkpoint", "type": str, "required": False},
        {"name": "--file", "type": str, "default": "target_hold/training/ppo_target_hold.yaml"},
        {"name": "--num_envs", "type": int, "default": 128},
        {"name": "--sigma", "type": float, "required": False},
        {"name": "--track", "action": "store_true"},
        {"name": "--wandb-project-name", "type": str, "default": "rl_games"},
        {"name": "--wandb-entity", "type": str, "default": None},
        {"name": "--task", "type": str, "default": "target_hold_task"},
        {"name": "--experiment_name", "type": str},
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "True",
        },
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "False",  # target_hold needs Isaac Gym for RGB
        },
    ]

    args = parse_arguments(description="target_hold RL runner", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def main():
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(get_args())

    with open(args["file"], "r") as stream:
        config = yaml.safe_load(stream)

    config = aerial_runner.update_config(config, args)

    from rl_games.torch_runner import Runner

    runner = Runner()
    runner.load(config)

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if args["track"] and rank == 0:
        import wandb

        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )
    runner.run(args)
    if args["track"] and rank == 0:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
