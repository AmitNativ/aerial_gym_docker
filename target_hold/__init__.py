"""target_hold — RL task for keeping a target box in the drone camera FOV.

Importing this package registers:
  * env_config  "target_hold_env"  (env_config_registry)
  * robot       "target_hold_quad" (robot_registry, BaseMultirotor + sensor config)
  * task        "target_hold_task" (task_registry, TargetHoldTask + task_config)

The upstream aerial_gym package's own registrations run first (via its
``__init__`` chain triggered by importing any aerial_gym submodule), so these
names extend — not overwrite — the existing registry.
"""

from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.robots.base_multirotor import BaseMultirotor

from target_hold.configs.target_hold_env_config import TargetHoldEnvCfg
from target_hold.configs.target_hold_robot_config import TargetHoldQuadCfg
from target_hold.configs.target_hold_task_config import task_config as target_hold_task_config
from target_hold.tasks.target_hold_task import TargetHoldTask


env_config_registry.register("target_hold_env", TargetHoldEnvCfg)
robot_registry.register("target_hold_quad", BaseMultirotor, TargetHoldQuadCfg)
task_registry.register_task("target_hold_task", TargetHoldTask, target_hold_task_config)
