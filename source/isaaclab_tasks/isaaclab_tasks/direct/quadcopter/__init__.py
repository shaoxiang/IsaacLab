# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Quadcopter-Direct-v0",
    entry_point=f"{__name__}.quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Smooth-v0",
    entry_point=f"{__name__}.quadcopter_env_smooth:QuadcopterSmoothEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env_smooth:QuadcopterSmoothEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-IMU-v0",
    entry_point=f"{__name__}.quadcopter_env_imu:QuadcopterImuEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env_imu:QuadcopterImuEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Simple-v0",
    entry_point=f"{__name__}.quadcopter_env_simple:QuadcopterEnvSimple",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env_simple:QuadcopterEnvSimpleCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Direct-Lidar-v0",
    entry_point=f"{__name__}.quadcopter_lidar_env:QuadcopterLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_lidar_env:QuadcopterLidarEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Direct-play-v0",
    entry_point=f"{__name__}.quadcopter_env_play:QuadcopterEnvPlay",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env_play:QuadcopterEnvPlayCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-RGB-Camera-Direct-v0",
    entry_point=f"{__name__}.quadcopter_cam_env:QuadcopterCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_cam_env:QuadcopterRGBCameraEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cam_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Vision-OA-v0",
    entry_point=f"{__name__}.quadcopter_avoid_obs_env:QuadcopterVisionOAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_avoid_obs_env:QuadcopterVisionOAEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_ov_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Form-v0",
    entry_point=f"{__name__}.quadcopter_form_env:QuadcopterFormEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_form_env:QuadcopterFormEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_form_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPOFormRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_form_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_form_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Form-Path-v0",
    entry_point=f"{__name__}.quadcopter_form_path_env:QuadcopterFormPathEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_form_path_env:QuadcopterFormPathEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_form_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPOFormRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_form_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_form_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-Form-Play-v0",
    entry_point=f"{__name__}.quadcopter_form_play_env:QuadcopterFormPlayEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_form_play_env:QuadcopterFormPlayEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_form_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPOFormPlayRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_form_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_form_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-Quadcopter-Vision-Depth-v0",
#     entry_point=f"{__name__}.quadcopter_vision_depth_env:QuadcopterVisionDepthEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.quadcopter_vision_depth_env:QuadcopterVisionDepthEnvCfg",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_ov_cfg.yaml",
#     },
# )

# gym.register(
#     id="Isaac-Quadcopter-Vision-Depth-v1",
#     entry_point=f"{__name__}.quadcopter_vision_depth_env2:QuadcopterVisionDepthEnv2",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.quadcopter_vision_depth_env2:QuadcopterVisionDepthEnvCfg2",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
#         "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
#     },
# )