# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

UAV_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UAV/uav_ptz/uav_ptz.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5.0),
        rot=(0.70711, 0.70711, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
            "jointX": 0.0,
            "jointY": 0.0,
            "jointZ": 0.0,
        },
    ),
    actuators={
        "quadcopter": ImplicitActuatorCfg(
            joint_names_expr=["m.*_joint"],
            stiffness=0.0,
            damping=0.0,
        ),
        "ptz": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            stiffness=0.0,
            damping=0.0,
        )
    },
)

"""Configuration for the UAV quadcopter."""
