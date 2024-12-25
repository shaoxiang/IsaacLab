# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the jetbot.
Reference: 
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.assets import DeformableObjectCfg

##
# Configuration
##

TENNIS_BALL_CFG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Tennis_Ball/tennis_ball_low.usd",
        scale=(0.0385, 0.0385, 0.0385),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0, contact_offset=0.001),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
    debug_vis=False,
)

TEDDY_BEAR_CFG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.05), rot=(0.707, 0, 0, 0.707)),
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
        scale=(0.01, 0.01, 0.01),
    ),
)

"""Configuration for deformable objects."""