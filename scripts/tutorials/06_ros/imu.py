# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the different imu sensors that can be attached to a robot.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/06_ros/imu.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates the different camera sensor implementations.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to spawn.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.imu import Imu, ImuCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(color_scheme="random"),
        visual_material=None,
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Create a imu sensor
    imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=not args_cli.headless,
    )

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    print("import rclpy success!")
except Exception as e:
    print(f"Error import rclpy: {e}")

# def imu_setup(prim_path):
#     imu = IMUSensor(
#         prim_path=prim_path,
#         name='imu',
#         frequency=60,  # Tần số lấy mẫu, có thể điều chỉnh tùy theo yêu cầu
#         translation=np.array([0, 0, 0]),  # Vị trí cảm biến trên robot
#         orientation=np.array([1, 0, 0, 0]),  # Hướng cảm biến trên robot
#         linear_acceleration_filter_size=10,
#         angular_velocity_filter_size=10,
#         orientation_filter_size=10,
#     )
#     imu.initialize()
#     return imu

class RobotBaseNode(Node):
    def __init__(self, num_envs):
        super().__init__('imu_driver_node')
        self.num_envs = num_envs

    # def publish_imu(self, cam_image, robot_num):
    #     np_img = cam_image.detach().cpu().numpy()
    #     ros_image = Image()
    #     ros_image.header.stamp = self.get_clock().now().to_msg()
    #     ros_image.header.frame_id = f"robot{robot_num}/camera_frame"
    #     ros_image.height = np_img.shape[0]
    #     ros_image.width = np_img.shape[1]
    #     ros_image.encoding = "rgb8"
    #     ros_image.step = np_img.shape[1] * 3
    #     ros_image.data = np_img.tobytes()
    #     self.image_pub[robot_num].publish(ros_image)

    def pub_imu_graph(self):
        for i in range(self.num_envs):
            imu_topic_name = f"robot_{i}/imu"
            frame_id = f"robot_{i}/imu_frame"

            keys = og.Controller.Keys
            og.Controller.edit(
                {
                    "graph_path": "/ImuROS2Graph",
                    "evaluator_name": "execution",
                },
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("IsaacSimulationGate", "omni.isaac.core_nodes.IsaacSimulationGate"),
                        ("IsaacReadIMU", "isaacsim.sensors.physics.IsaacReadIMU"),
                        ("ROS2PublishImu", "isaacsim.ros2.bridge.ROS2PublishImu"),
                    ],

                    keys.SET_VALUES: [
                        ("IsaacSimulationGate.inputs:step", 1), 
                        ("IsaacReadIMU.inputs:imuPrim", f"/World/envs/env_{i}/Robot/base"),
                        ("IsaacReadIMU.inputs:readGravity", True),
                        ("IsaacReadIMU.inputs:useLatestData", True),                        
                        # publish imu
                        ("ROS2PublishImu.inputs:context", 0),
                        ("ROS2PublishImu.inputs:topicName", imu_topic_name),
                        ("ROS2PublishImu.inputs:frameId", frame_id),
                        ("ROS2PublishImu.inputs:publishAngularVelocity", True),
                        ("ROS2PublishImu.inputs:publishLinerAcceleration", True),
                        ("ROS2PublishImu.inputs:publishOrientation", True),
                    ],

                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "IsaacSimulationGate.inputs:execIn"),
                        ("IsaacSimulationGate.outputs:execOut", "IsaacReadIMU.inputs:execIn"),
                        ("IsaacReadIMU.outputs:execOut", "ROS2PublishImu.inputs:execIn"),
                        ("IsaacReadIMU.outputs:angVel", "ROS2PublishImu.inputs:angularVelocity"),
                        ("IsaacReadIMU.outputs:linAcc", "ROS2PublishImu.inputs:linearAcceleration"),
                        ("IsaacReadIMU.outputs:orientation", "ROS2PublishImu.inputs:orientation"),
                    ],
                },
            )

    # def publish_imu(imu, freq):

    #     step_size = int(60 / freq)
    #     topic_name = imu.name + "_imu"
    #     queue_size = 10  # Kích thước hàng đợi có thể điều chỉnh tùy theo yêu cầu
    #     node_namespace = ""
    #     frame_id = imu.prim_path.split("/")[-1]

    #     # Khởi tạo writer cho IMU
    #     writer = rep.writers.get("ROS2PublishImu")
    #     writer.initialize(
    #         frameId=frame_id,
    #         nodeNamespace=node_namespace,
    #         queueSize=queue_size,
    #         topicName=topic_name
    #     )
    #     writer.attach([imu._imu_sensor_path])

    #     # Cài đặt bước thời gian cho node Isaac Simulation Gate
    #     gate_path = omni.syntheticdata.SyntheticData._get_node_path(
    #         "IMU" + "IsaacSimulationGate", imu._imu_sensor_path
    #     )
    #     og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    def pub_imu(self, cameras, freq:int):
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            step_size = int(60/freq)
            topic_name = f"robot_{i}/imu"
            frame_id = f"robot_{i}/imu_frame"
            node_namespace = ""         
            queue_size = 10
            writer = rep.writers.get("ROS2PublishImu")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name
            )
            writer.attach([imu._imu_sensor_path])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # ROS2 Bridge
    rclpy.init()
    base_node = RobotBaseNode(args_cli.num_envs)
    base_node.pub_imu_graph()

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
                # print information from the sensors
        print("-------------------------------")
        print(scene.sensors["imu"])
        print("Received shape of imu lin_acc_b : ", scene.sensors["imu"].data.lin_acc_b.shape)
        print("Received shape of imu ang_vel_b : ", scene.sensors["imu"].data.ang_vel_b.shape)
        print("Received shape of imu lin_vel_b : ", scene.sensors["imu"].data.lin_vel_b.shape)
        print("Received shape of imu quat_w : ", scene.sensors["imu"].data.quat_w.shape)        
        # print(scene["camera"].render_product_paths)

    base_node.destroy_node()
    rclpy.shutdown()

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
