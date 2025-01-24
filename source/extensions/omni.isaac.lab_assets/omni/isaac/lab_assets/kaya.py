# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the kaya.
Reference: 
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg, DCMotorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

KAYA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Kaya/kaya.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_vel={
            "axle_0_joint": 0.0,
            "axle_1_joint": 0.0,
            "axle_2_joint": 0.0,
        },
    ),
    actuators={
        "driver_joints": ImplicitActuatorCfg(
            joint_names_expr=["axle_.*_joint"],
            effort_limit=10.0,
            velocity_limit=50.0,
            stiffness=0.0,
            damping=100.0,
        ),
    }
)

KAYA_TENNIS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Kaya/kaya_basket.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_vel={
            "axle_0_joint": 0.0,
            "axle_1_joint": 0.0,
            "axle_2_joint": 0.0,
        },
    ),
    actuators={
        "driver_joints": ImplicitActuatorCfg(
            joint_names_expr=["axle_.*_joint"],
            effort_limit=10.0,
            velocity_limit=50.0,
            stiffness=0.0,
            damping=100.0,
        ),
        # "roller_joints": ActuatorBaseCfg(
        #     joint_names_expr=["roller_.*_joint"],
        #     effort_limit=40000.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=0.0,
        # ),
    }
)
"""Configuration for a simple kaya mobile robot."""