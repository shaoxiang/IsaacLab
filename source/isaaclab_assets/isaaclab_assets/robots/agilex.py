# Agilex Robotics
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the kaya.
Reference: 
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, ActuatorBaseCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

SCOUT_MINI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/AgilexRobotics/scout_mini/scout_mini_range_sensor.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_vel={
            "front_right_wheel": 0.0,
            "front_left_wheel": 0.0,
            "rear_right_wheel": 0.0,
            "rear_left_wheel": 0.0,
        },
    ),
    actuators={
        "driver_joints": ImplicitActuatorCfg(
            joint_names_expr=["front_.*_wheel", "rear_.*_wheel"],
            # stiffnet and damping tuned with tuning guide: https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_joint_tuning.html
            effort_limit=25.0,          # 0.25 Nm
            velocity_limit = 104.72,    # rpm -> rad/s ; 1000 rpm -> 104.72 rad/s
            stiffness = 0.0,            # needs to be 0 for velocity control
            damping = 1000.0,
        ),
    }
)

"""Configuration for Agilex Robotics."""