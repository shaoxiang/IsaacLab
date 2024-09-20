# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .jetbot_camera_env import JetbotEnv, JetbotEnvCfg

"""
Jetbot balancing environment.
"""

print("===============================")
print("Registering Jetbot Environtment")
print("===============================")

gym.register(
    id="Isaac-Jetbot-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.mobile_robot:JetbotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml"
    },
)
