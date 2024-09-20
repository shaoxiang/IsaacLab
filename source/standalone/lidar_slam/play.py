# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import omni
import carb
import numpy as np
import array
import threading
import queue
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

import omni.isaac.lab.sim as sim_utils

try:
    import rclpy
    from rclpy.qos import QoSProfile
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2, PointField, Imu
    from sensor_msgs_py import point_cloud2
    print("import rclpy success!")
except:
    print("import rclpy failed")

data_queue = queue.Queue()  

class RobotDriver():
    def __init__(self):
        self.node = rclpy.create_node("sensor_driver_node")
        qos_profile = QoSProfile(depth=10)
        self.lidar_pub = self.node.create_publisher(PointCloud2, f'/point_cloud2', qos_profile)
        self.imu_pub = self.node.create_publisher(Imu, f'/imu2', qos_profile)
        self.imu_rate = self.node.create_rate(50)
        self.imu_annotator = None

    def run(self):
        while rclpy.ok():  
            sensor_data = data_queue.get()  # 阻塞，直到队列中有元素  
            self.pub_pointcloud(sensor_data["lidar"])
            # self.publish_imu(sensor_data["imu"])
            data_queue.task_done()  # 表示前一个入队任务已经完成  
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def run_imu(self):
        while rclpy.ok():
            # data = self.imu_annotator
            frame = self.imu_annotator.get_current_frame()
            base_lin = frame["lin_acc"]
            ang_vel = frame["ang_vel"]
            orientation = frame["orientation"]
            self.publish_imu(base_lin, ang_vel, orientation)
            self.imu_rate.sleep()

    def pub_pointcloud(self, data_points):
        try:
            # print("point_cloud:", data_points.shape)
            point_cloud = PointCloud2()
            point_cloud.header = Header(frame_id="odom")
            point_cloud.header.stamp = self.node.get_clock().now().to_msg()
            point_cloud.height = data_points.shape[1]
            point_cloud.width = data_points.shape[0]
            point_cloud.point_step = data_points.dtype.itemsize
            point_cloud.row_step = data_points.dtype.itemsize * data_points.shape[0]
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            point_cloud.fields = fields
            # Convert numpy points to array.array
            ###########这部分不耗时###############
            point_cloud = point_cloud2.create_cloud(point_cloud.header, fields, data_points)
            #####################################
            # windows 下发布消息耗时严重
            self.lidar_pub.publish(point_cloud)
        except Exception as error:
            carb.log_error("pub point cloud failed!" + str(error))

    def publish_imu(self, linear_acc_data, angular_velocity_data, orientation_data):
        # print("imu_data:", linear_acc_data, angular_velocity_data, orientation_data)
        # ImuData(pos_w=tensor([[45.3174, -8.7304,  3.1546]], device='cuda:0'), 
        # quat_w=tensor([[0.7127, 0.2179, 0.1149, 0.6568]], device='cuda:0'), 
        # lin_vel_b=tensor([[-0.5269, -1.5967,  1.4445]], device='cuda:0'), 
        # ang_vel_b=tensor([[-1.0782,  0.4206, -1.2723]], device='cuda:0'), 
        # lin_acc_b=tensor([[-1.1804, -4.8344,  3.8072]], device='cuda:0'), 
        # ang_acc_b=tensor([[-74.4913, -31.6464,  12.2332]], device='cuda:0'))

        print("imu_data:", linear_acc_data, angular_velocity_data, orientation_data)
        imu_trans = Imu()
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

    # def publish_imu(self, imu_data):
    #     # print("imu_data:", linear_acc_data, angular_velocity_data, orientation_data)
    #     # ImuData(pos_w=tensor([[45.3174, -8.7304,  3.1546]], device='cuda:0'), 
    #     # quat_w=tensor([[0.7127, 0.2179, 0.1149, 0.6568]], device='cuda:0'), 
    #     # lin_vel_b=tensor([[-0.5269, -1.5967,  1.4445]], device='cuda:0'), 
    #     # ang_vel_b=tensor([[-1.0782,  0.4206, -1.2723]], device='cuda:0'), 
    #     # lin_acc_b=tensor([[-1.1804, -4.8344,  3.8072]], device='cuda:0'), 
    #     # ang_acc_b=tensor([[-74.4913, -31.6464,  12.2332]], device='cuda:0'))

    #     linear_acc_data = imu_data.lin_acc_b[0]
    #     angular_velocity_data = imu_data.ang_vel_b[0]
    #     orientation_data = imu_data.quat_w[0]
    #     print("imu_data:", linear_acc_data, angular_velocity_data, orientation_data)
    #     imu_trans = Imu()
    #     imu_trans.header.stamp = self.node.get_clock().now().to_msg()
    #     imu_trans.header.frame_id = f"imu_link"
    #     imu_trans.linear_acceleration.x = linear_acc_data[0].item()
    #     imu_trans.linear_acceleration.y = linear_acc_data[1].item()
    #     imu_trans.linear_acceleration.z = linear_acc_data[2].item()
    #     imu_trans.angular_velocity.x = angular_velocity_data[0].item()
    #     imu_trans.angular_velocity.y = angular_velocity_data[1].item()
    #     imu_trans.angular_velocity.z = angular_velocity_data[2].item()
    #     # (w, x, y, z)
    #     imu_trans.orientation.x = orientation_data[1].item()
    #     imu_trans.orientation.y = orientation_data[2].item()
    #     imu_trans.orientation.z = orientation_data[3].item()
    #     imu_trans.orientation.w = orientation_data[0].item()
    #     self.imu_pub.publish(imu_trans)

def run_ros2_node(imu_annotator):  
    try:
        rclpy.init()  
        base_node = RobotDriver()
        base_node.imu_annotator = imu_annotator

        ros_thread = threading.Thread(target=base_node.run)
        ros_thread.start()
        ros_thread = threading.Thread(target=base_node.run_imu)
        ros_thread.start()

    except Exception as error:
        # If anything causes your compute to fail report the error and return False
        carb.log_error("init ros2 node failed!" + str(error))

def add_scene_tunnel():
    try:
        cfg_scene = sim_utils.UsdFileCfg(usd_path="D:/workspace/lifelong_slam/scen_test/tunnel_empty.usd")
        cfg_scene.func("/World/tunnel", cfg_scene, translation=(0.0, 0.0, 1.45))
    except:
        print("Error loading custom environment.")

from omni.isaac.sensor import IMUSensor
def add_imu():
    imu_sensor = IMUSensor(
        prim_path="/World/envs/env_0/Robot/body/Imu_Sensor",
    )
    return imu_sensor

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    add_scene_tunnel()

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    # log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env_gym = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env_gym)

    imu_annotator = add_imu()
    run_ros2_node(imu_annotator)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    # reset environment
    obs, obs_raw = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, extras = env.step(actions)

            frame = imu_annotator.get_current_frame()
            base_lin = frame["lin_acc"]
            ang_vel = frame["ang_vel"]
            orientation = frame["orientation"]
            # print("base_lin:", base_lin, "ang_vel:", ang_vel, "orientation:", orientation)

            # joint_efforts = torch.zeros_like(env_gym.env._actions)
            # joint_efforts[:, 0] = 0.05
            # obs, _, _, extras = env.step(joint_efforts)

            # print("extras observations:", extras["observations"]["lidar"])
            if 'lidar' in extras["observations"]:
                lidar_data = extras["observations"]["lidar"]
                data_queue.put(extras["observations"]) 
 
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
