# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment.

In this example, we use a locomotion policy to control the H1 robot. The robot was trained
using Isaac-Velocity-Rough-H1-v0. The robot is commanded to move forward at a constant velocity.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/standalone/lidar_imu_pub/main.py --checkpoint /path/to/jit/checkpoint.pt

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
import rclpy

"""Rest everything follows."""
import io
import os
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY_ROS

from lidar import add_head_lidar
from ros import SensorPubNode
from isaaclab.devices import Se3Keyboard

lin_vel_x = 0.0
lin_vel_y = 0.0
ang_vel_z = 0.0

def vx_control_adder():
    global lin_vel_x
    lin_vel_x += 0.1
    if lin_vel_x > 1.5:
        lin_vel_x = 1.5

def vx_control_subtracter():
    global lin_vel_x
    lin_vel_x -= 0.1
    if lin_vel_x <= 0.0:
        lin_vel_x = 0.0

def vy_control_adder():
    global lin_vel_y
    lin_vel_y += 0.1
    if lin_vel_y > 1.5:
        lin_vel_y = 1.5

def vy_control_subtracter():
    global lin_vel_y
    lin_vel_y -= 0.1
    if lin_vel_y <= 0.0:
        lin_vel_y = 0.0

def ang_z_control_adder():
    global ang_vel_z
    ang_vel_z += 0.1
    if ang_vel_z > 1.0:
        ang_vel_z = 1.0

def ang_z_control_subtracter():
    global ang_vel_z
    ang_vel_z -= 0.1
    if ang_vel_z < -1.0:
        ang_vel_z = -1.0

def stop():
    global lin_vel_x
    global lin_vel_y
    global ang_vel_z
    lin_vel_x = 0.0
    lin_vel_y = 0.0
    ang_vel_z = 0.0

def main():
    """Main function."""
    global lin_vel_x
    global lin_vel_y
    global ang_vel_z
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file)
    env_cfg = H1RoughEnvCfg_PLAY_ROS()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd",
    )
    env_cfg.sim.device = "cpu"
    env_cfg.sim.use_fabric = False
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, _ = env.reset()

    add_head_lidar()
    # start ros2 nodes
    rclpy.init()
    sensor_node = SensorPubNode()
    timeline = omni.timeline.get_timeline_interface()

    teleop_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
    teleop_interface.add_callback("W", vx_control_adder)
    teleop_interface.add_callback("S", vx_control_subtracter)
    teleop_interface.add_callback("Z", vy_control_adder)
    teleop_interface.add_callback("C", vy_control_subtracter)
    teleop_interface.add_callback("A", ang_z_control_adder)
    teleop_interface.add_callback("D", ang_z_control_subtracter)
    teleop_interface.add_callback("X", stop)
    
    while simulation_app.is_running():

        #  3 + 3 + 3 + 3 + 28 * 3 + 160 = 256
        # base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions, height_scan
        obs["policy"][0][9] = lin_vel_x
        obs["policy"][0][10] = lin_vel_y
        obs["policy"][0][11] = ang_vel_z
        action = policy(obs["policy"])  # run inference
        # print(obs["policy"].shape)        
        obs, _, _, _, _ = env.step(action)
        sim_time_sec = timeline.get_current_time()
        # print("lin_vel_x:", lin_vel_x, "lin_vel_y:", lin_vel_y, "ang_vel_z:", ang_vel_z)
        # sensor_node.publish(obs, sim_time_sec)
        # env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.4)
        # env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.3, 0.7)
        # env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()