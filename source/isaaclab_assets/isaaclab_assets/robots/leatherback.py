# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Need to figure out how to get the REPO_ROOT_PATH
# from isaaclab.utils.assets import REPO_ROOT_PATH

##
# Configuration
##

# Eric is usng a simpler Leatherback car
# usd_path=f"{REPO_ROOT_PATH}/source/assets/robots/leatherback_simple_better.usd",
# usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Leatherback/leatherback.usd",
# usd_path=f"{ISAACLAB_NUCLEUS_DIR}/source/assets/robots/leatherback_simple_better.usd",
# /home/goat/Documents/GitHub/renanmb/IsaacLab/source/assets/robots/

LEATHERBACK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/goat/Documents/GitHub/renanmb/IsaacLab/source/assets/robots/leatherback_simple_better.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Spawn the robot 5 cm above the ground so that it has a small, visible drop when it spawns
        pos=(0.0, 0.0, 0.05), 
        joint_pos={
            "Wheel__Knuckle__Front_Left": 0.0, 
            "Wheel__Knuckle__Front_Right": 0.0,
            "Wheel__Upright__Rear_Right": 0.0,
            "Wheel__Upright__Rear_Left": 0.0,
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
        },
    ),
    #  Need to add the correct actuators
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=100000.0, # very high dampening is required because it is a velocity controller
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"], 
            effort_limit=40000.0, 
            velocity_limit=100.0, 
            stiffness=1000.0, # Very high stiffness is required because it is position cotroller
            damping=0.0
        ),
    },
)
"""Configuration for a simple ackermann robot."""
