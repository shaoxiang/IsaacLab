# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

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

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import omni
import carb
import numpy as np
import copy
import array
import threading
import queue
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)

import isaaclab.sim as sim_utils

try:
    import rclpy
    from rclpy.qos import QoSProfile
    from std_msgs.msg import Header
    from sensor_msgs.msg import Image
    print("import rclpy success!")
except:
    print("import rclpy failed")

# image_pub_queue = queue.Queue()  

# class RobotDriver():
#     def __init__(self):
#         self.node = rclpy.create_node("sensor_driver_node")
#         qos_profile = QoSProfile(depth=10)
#         self.image_pub = self.node.create_publisher(Image, f'/uav1/camera/color_image', qos_profile)

#     def run_image_pub_node(self):
#         while rclpy.ok():  
#             sensor_data = image_pub_queue.get()  # 阻塞，直到队列中有元素  
#             self.pub_image_data(sensor_data)
#             image_pub_queue.task_done()  # 表示前一个入队任务已经完成  
#             rclpy.spin_once(self.node, timeout_sec=0.05)

#     def pub_image_data(self, frame):
#         try:
#             # processes image data and converts to ros 2 message
#             msg = Image()
#             msg.header.stamp = self.node.get_clock(self).now().to_msg()
#             msg.header.frame_id = 'UAV'
#             msg.height = np.shape(frame)[0]
#             msg.width = np.shape(frame)[1]
#             msg.encoding = "bgr8"
#             msg.is_bigendian = False
#             msg.step = np.shape(frame)[2] * np.shape(frame)[1]
#             msg.data = np.array(frame).tobytes()
#             # publishes message
#             self.image_pub.publish(msg)
#         except Exception as error:
#             carb.log_error("pub point cloud failed!" + str(error))

# def run_ros2_node():  
#     try:
#         rclpy.init()  
#         base_node = RobotDriver()
#         ros_thread = threading.Thread(target=base_node.run_image_pub_node)
#         ros_thread.start()

#     except Exception as error:
#         # If anything causes your compute to fail report the error and return False
#         carb.log_error("init ros2 node failed!" + str(error))

def add_fly_scene():
    try:
        # cfg_scene = sim_utils.UsdFileCfg(usd_path="/home/ai/omniverse/Downloads/Isaac/4.1/Isaac/Environments/scen_test/tunnel_empty.usd")
        cfg_scene = sim_utils.UsdFileCfg(usd_path="D:/workspace/lifelong_slam/scen_test/tunnel_empty.usd")
        cfg_scene.func("/World/tunnel", cfg_scene, translation=(0.0, 0.0, 1.45))
    except:
        print("Error loading custom environment.")

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    add_fly_scene()

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # run_ros2_node()

    # reset environment
    obs, obs_raw = env.get_observations()
    timestep = 0
    import time
    last_time = time.time()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, extras = env.step(actions)

            # print("extras observations:", extras["observations"]["lidar"])
            if 'image' in extras["observations"]:
                image_data = extras["observations"]["image"]
                print("image_data shape:", image_data.shape)
                # image_pub_queue.put(image_data)

            # now_time = time.time()
            # print("time loop:", now_time - last_time)
            # last_time = now_time

    # close the simulator
    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
