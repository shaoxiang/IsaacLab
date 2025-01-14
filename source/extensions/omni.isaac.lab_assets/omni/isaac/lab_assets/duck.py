# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`DUCK_CFG`: DUCK robot

Reference: 
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
##
# Configuration
##

DUCK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Duck/duck.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": 0.0,
            ".*_knee_pitch_joint": 0.0,
            ".*_ankle_pitch_joint": 0.0,
            "throat_pitch_1_joint": 0.0,
            "throat_pitch_2_pitch": 0.0,
            "throat_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_pitch_joint", ".*_ankle_pitch_joint"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_pitch_joint": 100.0,
                ".*_ankle_pitch_joint": 100.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_pitch_joint": 5.0,
                ".*_ankle_pitch_joint": 5.0,
            },
        ),
        "throat": ImplicitActuatorCfg(
            joint_names_expr=["throat_pitch_1_joint", "throat_pitch_2_pitch", "throat_yaw_joint"],
            effort_limit=100,
            velocity_limit=20.0,
            stiffness={
                "throat_pitch_1_joint": 40.0,
                "throat_pitch_2_pitch": 40.0,
                "throat_yaw_joint": 40.0,
            },
            damping={
                "throat_pitch_1_joint": 10.0,
                "throat_pitch_2_pitch": 10.0,
                "throat_yaw_joint": 10.0,
            },
        ),
    },
)

DUCK_BDX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Duck/bdx.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*_ankle": 0.0,            # (-40.0, 40.0)
            ".*_knee": 0.0,             # (-80.0, 80.0)
            ".*_hip_pitch": 0.0,        # (-60.0, 60.0)
            ".*_hip_roll": 0.0,         # (-30.0, 30.0)
            ".*_hip_yaw": 0.0,          # (-30.0, 30.0)
            "head_roll": 0.0,           # (-50.0, 50.0)
            "head_yaw": 0.0,
            "head_pitch": 1.0472,       # (0, 150.0)
            "neck_pitch": -0.8727,      # (-180.0, -20.0)
            "left_antenna": -1.5708,    # (-200.0 -20.0)
            "right_antenna": 1.5708,    # (20, 200.0)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle", ".*_knee", ".*_hip_pitch", ".*_hip_roll", ".*_hip_yaw"],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness={
                ".*_ankle": 100.0,
                ".*_knee": 100.0,
                ".*_hip_pitch": 100.0,
                ".*_hip_roll": 100.0,
                ".*_hip_yaw": 100.0,
            },
            damping={
                ".*_ankle": 10.0,
                ".*_knee": 10.0,
                ".*_hip_pitch": 10.0,
                ".*_hip_roll": 10.0,
                ".*_hip_yaw": 10.0,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_roll", "head_yaw", "head_pitch", "neck_pitch", "left_antenna", "right_antenna"],
            effort_limit=100,
            velocity_limit=20.0,
            stiffness={
                "head_roll": 40.0,
                "head_yaw": 40.0,
                "head_pitch": 40.0,
                "neck_pitch": 40.0,
                "left_antenna": 40.0,
                "right_antenna": 40.0,
            },
            damping={
                "head_roll": 5.0,
                "head_yaw": 5.0,
                "head_pitch": 5.0,
                "neck_pitch": 5.0,
                "left_antenna": 5.0,
                "right_antenna": 5.0,
            },
        ),
    },
)

"""Configuration for the DUCK robot."""