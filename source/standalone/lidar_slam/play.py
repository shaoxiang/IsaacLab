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
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

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

try:
    import rclpy
    from rclpy.qos import QoSProfile
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointCloud2, PointField
    print("import rclpy success!")
except:
    print("import rclpy failed")

data_queue = queue.Queue()  

class RobotDriver():
    def __init__(self):
        self.node = rclpy.create_node("sensor_driver_node")
        qos_profile = QoSProfile(depth=10)
        self.lidar_pub = self.node.create_publisher(PointCloud2, f'/point_cloud2', qos_profile)

    def run(self):
        while True:  
            lidar_data = data_queue.get()  # 阻塞，直到队列中有元素  
            self.pub_pointcloud(lidar_data)
            data_queue.task_done()  # 表示前一个入队任务已经完成  
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def pub_pointcloud(self, data_points):
        try:
            print("point_cloud:", data_points.shape)
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
            memory_view = memoryview(data_points)
            casted = memory_view.cast('B')
            array_array = array.array('B')
            array_array.frombytes(casted)
            point_cloud.data = array_array
            #####################################
            # windows 下发布消息耗时严重
            # self.lidar_pub.publish(point_cloud)
        except Exception as error:
            carb.log_error("pub point cloud failed!" + str(error))

def run_ros2_node():  
    try:
        rclpy.init()  
        base_node = RobotDriver()
        base_node.run()
    except Exception as error:
        # If anything causes your compute to fail report the error and return False
        carb.log_error("init ros2 node failed!" + str(error))

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env_gym = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env_gym, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env_gym)

    print("env_gym:", dir(env_gym))
    print("env:", dir(env))

    ros_thread = threading.Thread(target=run_ros2_node)
    ros_thread.start()

    # print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # # load previously trained model
    # ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # ppo_runner.load(resume_path)

    # # obtain the trained policy for inference
    # policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

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
            # # agent stepping
            # actions = policy(obs)
            # # env stepping
            # obs, _, _, _ = env.step(actions)

            joint_efforts = torch.zeros_like(env_gym.env._actions)
            joint_efforts[:, 0] = 0.05
            obs, _, _, extras = env.step(joint_efforts)

            # print("extras observations:", extras["observations"]["lidar"])
            lidar_data = extras["observations"]["lidar"]
            for key, value in lidar_data.items():
                # print(f"Lidar data has key: {key}")
                for lidar_data in value:
                    # print(f"Data for {key}: {lidar_data.data}")
                    # print(f"Distance for {key}: {lidar_data.distance}")
                    # print(f"Intensity for {key}: {lidar_data.intensity}")
                    point_cloud = lidar_data.data
                    intensity = lidar_data.intensity.reshape(-1, 1)
                    data_points = np.concatenate((point_cloud, intensity), axis=1)
                    data_queue.put(data_points)
                    break

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
