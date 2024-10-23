# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visual test script for the imu sensor from the Orbit framework.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/06_ros/imu_sensor_to_ros.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(description="Imu Test Script")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to clone.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default="generator",
    choices=["generator", "usd", "plane"],
    help="Type of terrain to import. Can be 'generator' or 'usd' or 'plane'.",
)
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""

import torch
import traceback

import carb
import omni
from omni.isaac.cloner import GridCloner
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import PhysxSchema

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sensors.imu import Imu, ImuCfg
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.lab.terrains.terrain_importer import TerrainImporter
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.timer import Timer

import queue
import threading
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from std_msgs.msg import Header
    from sensor_msgs.msg import Imu as ROS_IMU
    print("import rclpy success!")
except:
    print("import rclpy failed")

data_queue = queue.Queue()

class IMUNode():
    def __init__(self):
        self.node = rclpy.create_node("imu_driver_node")
        qos_profile = QoSProfile(depth=10)
        self.imu_pub = self.node.create_publisher(ROS_IMU, f'/imu', qos_profile)
        self.timer_period = 0.02

    def run(self):
        while rclpy.ok():  
            imu_data = data_queue.get()  # 阻塞，直到队列中有元素  
            self.pub_imu_data(imu_data)
            data_queue.task_done()  # 表示前一个入队任务已经完成  
            rclpy.spin_once(self.node, timeout_sec=self.timer_period)

    def pub_imu_data(self, imu_data):
        try:
            linear_acc_data = imu_data.lin_acc_b[0]
            angular_velocity_data = imu_data.ang_vel_b[0]
            orientation_data = imu_data.quat_w[0]
            # print("imu_data:", linear_acc_data, angular_velocity_data, orientation_data)
            imu_trans = ROS_IMU()
            imu_trans.header.stamp = self.node.get_clock().now().to_msg()
            imu_trans.header.frame_id = f"imu_link"
            imu_trans.linear_acceleration.x = linear_acc_data[0].item()
            imu_trans.linear_acceleration.y = linear_acc_data[1].item()
            imu_trans.linear_acceleration.z = linear_acc_data[2].item()
            imu_trans.angular_velocity.x = angular_velocity_data[0].item()
            imu_trans.angular_velocity.y = angular_velocity_data[1].item()
            imu_trans.angular_velocity.z = angular_velocity_data[2].item()
            # (w, x, y, z)
            imu_trans.orientation.x = orientation_data[1].item()
            imu_trans.orientation.y = orientation_data[2].item()
            imu_trans.orientation.z = orientation_data[3].item()
            imu_trans.orientation.w = orientation_data[0].item()
            self.imu_pub.publish(imu_trans)
        except Exception as error:
            carb.log_error("pub imu data failed!" + str(error))

def run_ros2_node():  
    try:
        rclpy.init()  
        imu_node = IMUNode()
        ros_thread = threading.Thread(target=imu_node.run)
        ros_thread.start()

    except Exception as error:
        # If anything causes your compute to fail report the error and return False
        carb.log_error("init ros2 node failed!" + str(error))

def design_scene(sim: SimulationContext, num_envs: int = 2048) -> RigidObject:
    """Design the scene."""
    # Handler for terrains importing
    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
        max_init_terrain_level=None,
        num_envs=1,
    )
    _ = TerrainImporter(terrain_importer_cfg)
    # obtain the current stage
    stage = omni.usd.get_context().get_stage()
    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    # create source prim
    stage.DefinePrim(envs_prim_paths[0], "Xform")
    # clone the env xform
    cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True)
    # Define the scene
    # -- Light
    cfg = sim_utils.DistantLightCfg(intensity=2000)
    cfg.func("/World/light", cfg)
    # -- Balls
    cfg = RigidObjectCfg(
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        prim_path="/World/envs/env_.*/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )
    balls = RigidObject(cfg)
    # Clone the scene
    # obtain the current physics scene
    physics_scene_prim_path = None
    for prim in stage.Traverse():
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            physics_scene_prim_path = prim.GetPrimPath()
            carb.log_info(f"Physics scene prim path: {physics_scene_prim_path}")
            break
    # filter collisions within each environment instance
    cloner.filter_collisions(
        physics_scene_prim_path,
        "/World/collisions",
        envs_prim_paths,
    )
    return balls

def main():
    """Main function."""

    # Load kit helper
    sim_params = {
        "use_gpu": True,
        "use_gpu_pipeline": True,
        "use_fabric": True,  # used from Isaac Sim 2023.1 onwards
        "enable_scene_query_support": True,
    }
    sim = SimulationContext(
        physics_dt=1.0 / 100.0, rendering_dt=1.0 / 100.0, sim_params=sim_params, backend="torch", device="cuda:0"
    )
    # Set main camera
    set_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5])

    # Parameters
    num_envs = args_cli.num_envs
    # Design the scene
    balls = design_scene(sim=sim, num_envs=num_envs)

    # Create a ray-caster sensor
    imu_cfg = ImuCfg(
        prim_path="/World/envs/env_.*/ball",
        debug_vis=not args_cli.headless,
    )
    # increase scale of the arrows for better visualization
    imu_cfg.visualizer_cfg.markers["arrow"].scale = (1.0, 0.2, 0.2)
    imu = Imu(cfg=imu_cfg)

    # Play simulator and init the Imu
    sim.reset()

    # start imu node
    run_ros2_node()

    # Print the sensor information
    print(imu)

    # Get the ball initial positions
    sim.step(render=not args_cli.headless)
    balls.update(sim.get_physics_dt())
    ball_initial_positions = balls.data.root_pos_w.clone()
    ball_initial_orientations = balls.data.root_quat_w.clone()

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # Reset the scene
        if step_count % 500 == 0:
            # reset ball positions
            balls.write_root_pose_to_sim(torch.cat([ball_initial_positions, ball_initial_orientations], dim=-1))
            balls.reset()
            # reset the sensor
            imu.reset()
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the imu sensor
        imu.update(dt=sim.get_physics_dt(), force_recompute=True)

        # with Timer(f"Imu sensor update with {num_envs} and physics dt {sim.get_physics_dt()}"):
        #     imu.update(dt=sim.get_physics_dt(), force_recompute=True)
        # print("Imu data:", imu.data)

        data_queue.put(imu.data) 

        # Update counter
        step_count += 1

    rclpy.shutdown()

if __name__ == "__main__":
    try:
        # Run the main function
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
