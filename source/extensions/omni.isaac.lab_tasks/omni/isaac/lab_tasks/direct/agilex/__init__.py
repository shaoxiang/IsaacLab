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

##
# Register Gym environments.
##

gym.register(
    id="Isaac-ScoutMini-Direct-v0",
    entry_point=f"{__name__}.scout_mini_env:ScoutMiniEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.scout_mini_env:ScoutMiniEnvCfg",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.ScoutMiniPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)