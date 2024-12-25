# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
UAV environment.
"""

import gymnasium as gym

from . import agents
from .kaya_env import KayaEnv, KayaEnvCfg
from .kaya_tennis_env import KayaTennisEnv, KayaTennisEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Kaya-Direct-v0",
    entry_point=f"{__name__}.kaya_env:KayaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kaya_env:KayaEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.KayaPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Kaya-Tennis-v0",
    entry_point=f"{__name__}.kaya_tennis_env:KayaTennisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kaya_tennis_env:KayaTennisEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.KayaTennisPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)