# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the different camera sensors that can be attached to a robot.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/06_ros/camera.py --enable_cameras

    # Usage in headless mode
    ./isaaclab.sh -p scripts/tutorials/06_ros/camera.py --headless --enable_cameras

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
from isaaclab.sensors import CameraCfg
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

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "depth", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

import omni
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    print("import rclpy success!")
except Exception as e:
    print(f"Error import rclpy: {e}")

class RobotBaseNode(Node):
    def __init__(self, num_envs):
        super().__init__('camera_driver_node')
        self.num_envs = num_envs

    def publish_tiled_image(self, cam_image, robot_num):
        np_img = cam_image.detach().cpu().numpy()
        ros_image = Image()
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = f"robot{robot_num}/camera_frame"
        ros_image.height = np_img.shape[0]
        ros_image.width = np_img.shape[1]
        ros_image.encoding = "rgb8"
        ros_image.step = np_img.shape[1] * 3
        ros_image.data = np_img.tobytes()
        self.image_pub[robot_num].publish(ros_image)

    def pub_image_graph(self):
        for i in range(self.num_envs):
            color_topic_name = f"robot_{i}/front_cam/color_image"
            depth_topic_name = f"robot_{i}/front_cam/depth_image"
            frame_id = f"robot_{i}/front_cam"

            keys = og.Controller.Keys
            og.Controller.edit(
                {
                    "graph_path": "/CameraROS2Graph",
                    "evaluator_name": "execution",
                },
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("IsaacCreateRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                        ("ROS2CameraHelperColor", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                        ("ROS2CameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                        ("ROS2CameraHelperDepthCloud", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ],

                    keys.SET_VALUES: [
                        ("IsaacCreateRenderProduct.inputs:cameraPrim", f"/World/envs/env_{i}/Robot/base/front_cam"),
                        ("IsaacCreateRenderProduct.inputs:enabled", True),
                        ("IsaacCreateRenderProduct.inputs:height", 480),
                        ("IsaacCreateRenderProduct.inputs:width", 640),
                        
                        # color camera
                        ("ROS2CameraHelperColor.inputs:type", "rgb"),
                        ("ROS2CameraHelperColor.inputs:topicName", color_topic_name),
                        ("ROS2CameraHelperColor.inputs:frameId", frame_id),
                        ("ROS2CameraHelperColor.inputs:useSystemTime", True),

                        # depth camera
                        ("ROS2CameraHelperDepth.inputs:type", "depth"),
                        ("ROS2CameraHelperDepth.inputs:topicName", depth_topic_name),
                        ("ROS2CameraHelperDepth.inputs:frameId", frame_id),
                        ("ROS2CameraHelperDepth.inputs:useSystemTime", True),

                        # depth camera cloud
                        # ("ROS2CameraHelperDepthCloud.inputs:type", "depth_pcl"),
                        # ("ROS2CameraHelperDepthCloud.inputs:topicName", depth_cloud_topic_name),
                        # ("ROS2CameraHelperDepthCloud.inputs:frameId", frame_id),
                        # ("ROS2CameraHelperDepthCloud.inputs:useSystemTime", True),
                    ],

                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "IsaacCreateRenderProduct.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperColor.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperColor.inputs:renderProductPath"),
                        ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperDepth.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperDepth.inputs:renderProductPath"),
                        # ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperDepthCloud.inputs:execIn"),
                        # ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperDepthCloud.inputs:renderProductPath"),
                    ],

                },
            )

    def pub_color_image(self, cameras, freq:int):
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = cameras.render_product_paths[i]
            step_size = int(60/freq)
            topic_name = f"robot_{i}/front_cam/color_image"
            frame_id = f"robot_{i}/front_cam"
            node_namespace = ""         
            queue_size = 10
            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
            writer = rep.writers.get(rv + "ROS2PublishImage")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name,
            )
            writer.attach([render_product])
            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    def pub_depth_image(self, cameras):
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = cameras.render_product_paths[i]
            step_size = 1
            topic_name = f"robot_{i}/front_cam/depth_image"
            frame_id = f"robot_{i}/front_cam"
            node_namespace = ""
            queue_size = 1
            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                    sd.SensorType.DistanceToImagePlane.name
                                )
            writer = rep.writers.get(rv + "ROS2PublishImage")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name
            )
            writer.attach([render_product])

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
    base_node.pub_color_image(scene["camera"])

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
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
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
