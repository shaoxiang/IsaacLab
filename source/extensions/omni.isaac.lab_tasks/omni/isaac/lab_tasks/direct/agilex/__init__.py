# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
UAV environment.
"""

import gymnasium as gym

from . import agents
from .scout_mini_env import ScoutMiniEnv, ScoutMiniEnvCfg
from .scout_mini_avoidance_env import ScoutMiniAVEnv, ScoutMiniAVEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-ScoutMini-Direct-v0",
    entry_point=f"{__name__}.scout_mini_env:ScoutMiniEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scout_mini_env:ScoutMiniEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ScoutMiniPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-ScoutMini-AV-v0",
    entry_point=f"{__name__}.scout_mini_avoidance_env:ScoutMiniAVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scout_mini_avoidance_env:ScoutMiniAVEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
    },
)