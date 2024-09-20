# Original Code:
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# Modifications:
# Copyright (c) 2024, Irvin Hwang
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


"""Rest everything follows."""
from collections.abc import Sequence
import torch
import gymnasium as gym
import numpy as np
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import ArticulationCfg, Articulation, AssetBaseCfg, AssetBase, RigidObject, RigidObjectCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import TiledCameraCfg, CameraCfg
from omni.isaac.lab_assets.jetbot import JETBOT_CFG
import pdb
import math

@configclass
class JetbotSceneCfg(InteractiveSceneCfg):
    # room_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd"))
    room_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/tennis_court", spawn=sim_utils.UsdFileCfg(usd_path=f"D:/workspace/tennis/tennis_3d/tennis_court_egg.usd"))
        
    jetbot: ArticulationCfg = JETBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # camera
    camera = CameraCfg(
        data_types=["rgb"],
        prim_path="/World/envs/env_.*/Robot/chassis/rgb_camera/jetbot_camera",
        spawn=None,
        height=224,
        width=224,
        update_period=.1
    )

    # /jetbot/chassis/CC

    goal_marker = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/marker", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd"), init_state=RigidObjectCfg.InitialStateCfg(pos=(.3,0,0)))

def tennis_ball_config(num_envs, env_origins):
    # for env_index in range(num_envs):
    for env_origin in env_origins:
        for i in range(10):
            # print(f"/World/Cube{i}", self._terrain.env_origins[index])
            x = np.random.uniform(low=-20.0, high=20.0, size=1)  
            y = np.random.uniform(low=-10.0, high=10.0, size=1)  
            z = np.random.uniform(low=0.1, high=2.0, size=1)  
            coords = np.array([x[0], y[0], z[0]]) + env_origin
            # print("prim_path:", f"/World/envs/env_{env_index}/Balls/tb_{i}")
            prim_utils.create_prim(prim_path = f"/World/Balls/tb_{i}", prim_type = "Xform", translation= coords)
    
    tennis_ball = RigidObjectCfg(
        prim_path="/World/Balls/tb_.*/tb",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"D:/workspace/tennis/tennis_3d/tb1.usdc",
            # activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0567),
            # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    return tennis_ball

@configclass 
class JetbotEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 60.0
    num_actions = 2
    action_scale = 100.0
    num_envs = 2

    # Scene
    scene: InteractiveSceneCfg = JetbotSceneCfg(num_envs=num_envs, env_spacing=60.0)

    num_channels = 3
    num_observations = num_channels * scene.camera.height * scene.camera.width

class JetbotEnv(DirectRLEnv):
    cfg: JetbotEnvCfg

    def __init__(self, cfg: JetbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["jetbot"]
        # self._set_goal_position() 
        self.robot_camera = self.scene["camera"]
        self.action_scale = self.cfg.action_scale
        self.goal_marker = self.scene["goal_marker"]
        tennis_ball_cfg = tennis_ball_config(self.cfg.num_envs)
        tennis_ball = RigidObject(cfg=tennis_ball_cfg)

    def _set_goal_position(self):
        robot_orientation = self.robot.data.root_quat_w
        marker = self.scene["goal_marker"]
        # forward_vector = get_basis_vector_z(robot_orientation)
        positions, orientations = marker.get_world_poses()
        positions[:, 2] += 1.5
        marker.set_world_poses(positions, orientations) 
        forward_distance = 1
        # point_in_front = self.robot.data.root_pow_w + forward_distance * forward_vector
        return

    def _configure_gym_env_spaces(self):
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.scene.camera.height, self.cfg.scene.camera.width, self.cfg.num_channels),
        )
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # RL specifics
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)


    def _get_rewards(self) -> torch.Tensor:
        robot_position = self.robot.data.root_pos_w
        goal_position = self.goal_marker.data.root_pos_w
        squared_diffs = (robot_position - goal_position) ** 2
        distance_to_goal = torch.sqrt(torch.sum(squared_diffs, dim=-1))
        rewards = torch.exp(1/(distance_to_goal))
        rewards -= 30

        if (self.common_step_counter % 10 == 0):
            print(f"Reward at step {self.common_step_counter} is {rewards} for distance {distance_to_goal}")
        return rewards

    def _get_observations(self) -> dict:
        observations =  self.robot_camera.data.output["rgb"].clone()
        # get rid of the alpha channel
        observations = observations[:, :, :, :3]
        return {"policy": observations}

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        print("actions:", self.actions)
        self.robot.set_joint_velocity_target(self.actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = .01
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        robot_position = self.robot.data.root_pos_w
        goal_position = self.goal_marker.data.root_pos_w
        squared_diffs = (robot_position - goal_position) ** 2
        distance_to_goal = torch.sqrt(torch.sum(squared_diffs, dim=-1))
        distance_within_epsilon = distance_to_goal < epsilon
        distance_over_limit = distance_to_goal > .31
        position_termination_condition = torch.logical_or(distance_within_epsilon, distance_over_limit)
        position_termination_condition.fill_(False)
        return (position_termination_condition, time_out)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        default_goal_root_state = self.goal_marker.data.default_root_state[env_ids]
        default_goal_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.goal_marker.write_root_pose_to_sim(default_goal_root_state[:, :7], env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)