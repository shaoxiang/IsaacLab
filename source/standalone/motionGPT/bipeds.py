# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate bipedal robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/bipeds.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate bipedal robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets.robots.cassie import CASSIE_CFG  # isort:skip
from isaaclab_assets import H1_CFG  # isort:skip
from isaaclab_assets import G1_CFG  # isort:skip


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(device="cpu", dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    # Robots
    robot_h1 = Articulation(H1_CFG.replace(prim_path="/World/H1"))

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset dof state
            joint_pos, joint_vel = robot_h1.data.default_joint_pos, robot_h1.data.default_joint_vel

            print("joint_pos:", joint_pos) # 19
            print("joint_vel:", joint_vel) # 19
            
            robot_h1.write_joint_state_to_sim(joint_pos, joint_vel)
            root_state = robot_h1.data.default_root_state.clone()
            root_state[:, :3] += origins[0]
            robot_h1.write_root_state_to_sim(root_state)
            robot_h1.reset()

            # reset command
            print(">>>>>>>> Reset!")

            robot_h1.set_joint_position_target(robot_h1.data.default_joint_pos.clone())
            robot_h1.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot_h1.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
