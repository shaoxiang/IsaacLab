# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg
from .quadcopter_env_smooth import QuadcopterSmoothEnv, QuadcopterSmoothEnvCfg
from .quadcopter_env_imu import QuadcopterImuEnv, QuadcopterImuEnvCfg
from .quadcopter_env_simple import QuadcopterEnvSimple, QuadcopterEnvSimpleCfg
from .quadcopter_lidar_env import QuadcopterLidarEnv, QuadcopterLidarEnvCfg
from .quadcopter_form_env import QuadcopterFormEnv, QuadcopterFormEnvCfg
from .quadcopter_form_path_env import QuadcopterFormPathEnv, QuadcopterFormPathEnvCfg
# from .quadcopter_camera_env import QuadcopterCameraEnv, QuadcopterRGBCameraEnvCfg
from .quadcopter_cam_env import QuadcopterCameraEnv, QuadcopterRGBCameraEnvCfg
from .quadcopter_env_play import QuadcopterEnvPlay, QuadcopterEnvPlayCfg
from .quadcopter_avoid_obs_env import QuadcopterVisionOAEnv, QuadcopterVisionOAEnvCfg
from .quadcopter_form_play_env import QuadcopterFormPlayEnv, QuadcopterFormPlayEnvCfg
from .quadcopter_vision_depth_env import QuadcopterVisionDepthEnv, QuadcopterVisionDepthEnvCfg
from .quadcopter_vision_depth_env2 import QuadcopterVisionDepthEnv2, QuadcopterVisionDepthEnvCfg2


##
# Register Gym environments.
##

gym.register(
    id="Isaac-Quadcopter-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Smooth-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterSmoothEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterSmoothEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-IMU-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterImuEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterImuEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Simple-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterEnvSimple",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvSimpleCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Direct-Lidar-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterLidarEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Direct-play-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterEnvPlay",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvPlayCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-RGB-Camera-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterRGBCameraEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cam_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Vision-OA-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterVisionOAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterVisionOAEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_ov_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Form-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterFormEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterFormEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_form_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPOFormRunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_form_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_form_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Form-Path-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterFormPathEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterFormPathEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_form_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPOFormRunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_form_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_form_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Form-Play-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterFormPlayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterFormPlayEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_form_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPOFormPlayRunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_form_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_form_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Vision-Depth-v0",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterVisionDepthEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterVisionDepthEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_ov_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Vision-Depth-v1",
    entry_point="omni.isaac.lab_tasks.direct.quadcopter:QuadcopterVisionDepthEnv2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterVisionDepthEnvCfg2,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.QuadcopterPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)