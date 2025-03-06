# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.leatherback import LEATHERBACK_CFG
from .waypoint import WAYPOINT_CFG
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4          # Decimation - number of time steps between actions, it was 2
    episode_length_s = 20.0 # Max each episode should last in seconds, 30 s seems a lot
    # action_scale = 100.0    # [N]
    action_space = 2        # Number of actions the neural network shuold return   
    observation_space = 8   # Number of observations fed into neural network
    state_space = 0         # Observations to be used in Actor Critic Training

    # simulation frames Hz
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # TODO
    # robot
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # region Waypoints
    waypoint_cfg: VisualizationMarkersCfg = WAYPOINT_CFG

    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    env_spacing = 32.0 # depends on the ammount of Goals, 32 is a lot

    # scene - 4096 environments
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._throttle_dof_idx, _ = self.leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_dof_name)
        # self.action_scale = self.cfg.action_scale

        # self.joint_pos = self.leatherback.data.joint_pos
        # self.joint_vel = self.leatherback.data.joint_vel
        # self._throttle_action = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        # self._steering_action = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        self._throttle_state =  torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state =  torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)

        self._goal_reached =  torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed =  torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        # region Number of Goals
        self._num_goals = 10 # 10 seems too much
        # end region Number of Goals
        self._target_positions =  torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos =  torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)

        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = 2.5
        self.course_width_coefficient = 2.0

        # reward parameters
        # Tolerance
        # position_tolerance: float = 0.15, started at 0.2
        """Tolerance for the position of the robot. Defaults to 1cm."""
        self.position_tolerance: float = 0.15
        self.goal_reached_bonus: float = 10.0
        self.position_progress_weight: float = 1.0
        self.heading_coefficient = 0.25
        self.heading_progress_weight: float = 0.05

        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)

    # TODO
    # region Setup Scene
    # need to get the visualization markers
    def _setup_scene(self):
        self.leatherback = Articulation(self.cfg.robot_cfg)
        # He created a python file to do the waypoints
        self.Waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        #  It is inspired on the repvious examples
        # self.Cones = VisualizationMarkers(self.cfg.cone_cfg)
        # self.Red_Arrows = VisualizationMarkers(self.cfg.red_arrow_cfg)
        # self.Green_Arrows = VisualizationMarkers(self.cfg.green_arrow_cfg)
        # self.Rew_Markers = VisualizationMarkers(self.cfg.rew_marker_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone, filter and replicate
        self.scene.clone_environments(copy_from_source=False) # Clones child environments from parent environment
        self.scene.filter_collisions(global_prim_paths=[])    # Prevents environments from colliding with each other

        # add articulation to scene
        self.scene.articulations["Leatherback"] = self.leatherback

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # The process_actions is the inspiration for writting the _pre_physics_step
    # def process_actions(self, actions: torch.Tensor):
    #     self._previous_actions = self._actions.clone()
    #     self._actions = actions.clone()
    #     self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * self._robot_cfg.throttle_scale
    #     self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * self._robot_cfg.steering_scale
 
    #     # Log data
    #     self.scalar_logger.log("robot_state", "AVG/throttle_action", self._throttle_action[:, 0])
    #     self.scalar_logger.log("robot_state", "AVG/steering_action", self._steering_action[:, 0])
    
    # steering_scale = math.pi / 4.0
    # """Multiplier for the steering position. The action is in the range [-1, 1]"""
    # throttle_scale = 60.0
    # """Multiplier for the throttle velocity. The action is in the range [-1, 1] and the radius of the wheel is 0.06m"""

    # region _pre_physics_step
    # TODO
    # Need to configure the PRE Physics
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Multiplier for the throttle velocity. The action is in the range [-1, 1] and the radius of the wheel is 0.06m"""
        throttle_scale = 1 # when set to 2 it trains but the cars are flying, 3 you get NaNs
        # throttle_scale = 60.0
        throttle_max = 50.0
        """Multiplier for the steering position. The action is in the range [-1, 1]"""
        steering_scale = 0.1
        # steering_scale = math.pi / 4.0
        steering_max = 0.75

        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * throttle_scale
        self._throttle_action += self._throttle_state # RuntimeError: The size of tensor a (2560) must match the size of tensor b (4096) at non-singleton dimension 0
        self.throttle_action = torch.clamp(self._throttle_action, -throttle_max, throttle_max * 0.1)
        self._throttle_state = self._throttle_action
        # The actions[:, 0] should be getting the values from the column 1
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        # Error with tensor Sizes, self._steering_action(The size of tensor a (8192)) mus match self._steering_state the size of tensor b (4096)
        # tensor_b_repeated = tensor_b.repeat(2, 1)  # Repeat tensor_b twice along the first dimension
        # self._steering_state = self._steering_state.repeat(2,1)
        self._steering_action += self._steering_state # RuntimeError: The size of tensor a (8192) must match the size of tensor b (4096) at non-singleton dimension 0
        
        self._steering_action = torch.clamp(self._steering_action, -steering_max, steering_max)
        self._steering_state = self._steering_action
        # self.actions = self.action_scale * actions.clone()
    # end region _pre_physics_step

    # TODO
    def _apply_action(self) -> None:
        # it is missing the _throttle_action
        self.leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.leatherback.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)

    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute the element-wise quaternion multiplication of two batches of quaternions.

        Args:
            q1 (torch.Tensor): Tensor of shape [N, 4] representing N quaternions.
            q2 (torch.Tensor): Tensor of shape [N, 4] representing N quaternions.

        Returns:
        torch.Tensor: Tensor of shape [N, 4] representing the resulting quaternions.
        """
        w1, x1, y1, z1 = q1.unbind(dim=1)
        w2, x2, y2, z2 = q2.unbind(dim=1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=1)
    
    # region _get_observations
    def _get_observations(self) -> dict:

        # position error
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1) # had placed dim=1

        # heading error
        heading = self.leatherback.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 1] - self.leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 0] - self.leatherback.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1),
                torch.cos(self.target_heading_error).unsqueeze(dim=1),
                torch.sin(self.target_heading_error).unsqueeze(dim=1),
                self.leatherback.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
                self.leatherback.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
                self.leatherback.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        observations = {"policy": obs}
        return observations
    # end of region _get_observations
    # region _get_rewards
    def _get_rewards(self) -> torch.Tensor:

        # position progress
        position_progress_rew = self._previous_position_error - self._position_error

        # Heading Distance - changing the numerator to positive make it drive backwards
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)

        # Checks if the goal is reached
        goal_reached = self._position_error < self.position_tolerance

        # if the goal is reached, the target index is updated
        self._target_index = self._target_index + goal_reached

        self.task_completed = self._target_index > (self._num_goals -1)

        self._target_index = self._target_index % self._num_goals

        composite_reward = (
            position_progress_rew*self.position_progress_weight +
            target_heading_rew*self.heading_progress_weight +
            goal_reached*self.goal_reached_bonus
        )

        # region debugging
        # Update Waypoints
        # this is about the CONES 
        # marker0 to marker9 is RED
        # marker 10 to marker19 is BLUE
        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.Waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward
    # end of region _get_rewards
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        task_failed = self.episode_length_buf > self.max_episode_length

        # task completed is calculated in get_rewards before target_index is wrapped around
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # region Reset
        # reset from config
        default_state = self.leatherback.data.default_root_state[env_ids]   # first there are pos, next 4 quats, next 3 vel,next 3 ang vel, 
        leatherback_pose = default_state[:, :7]                             # proper way of getting default pose from config file
        leatherback_velocities = default_state[:, 7:]                       # proper way of getting default velocities from config file
        joint_positions = self.leatherback.data.default_joint_pos[env_ids]  # proper way to get joint positions from config file
        joint_velocities = self.leatherback.data.default_joint_vel[env_ids] # proper way to get joint velocities from config file

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids] # Adds center of each env position in leatherback position

        # Randomize Steering position at start of track
        leatherback_pose[:, 0] -= self.env_spacing / 2
        leatherback_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        # Randomize Starting Heading
        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)

        # Isaac Sim Quaternions are w first (w, x, y, z) To rotate about the Z axis, we will modify the W and Z values
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)
        # end region reset robot

        # region reset goals
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        spacing = 2 / self._num_goals
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        self._target_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) + self.course_length_coefficient
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0

        # Update the visual markers
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.Waypoints.visualize(translations=visualize_pos)
        # end Region Reset Goals

        # Region make sure the position error and position dist are up to date after the reset
        # reset positions error
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.leatherback.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        # reset heading error
        heading = self.leatherback.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.leatherback.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()
        # end region

# # Do we need this ????
# @torch.jit.script
# def compute_rewards(
#     rew_scale_alive: float,
#     rew_scale_terminated: float,
#     rew_scale_pole_pos: float,
#     rew_scale_cart_vel: float,
#     rew_scale_pole_vel: float,
#     pole_pos: torch.Tensor,
#     pole_vel: torch.Tensor,
#     cart_pos: torch.Tensor,
#     cart_vel: torch.Tensor,
#     reset_terminated: torch.Tensor,
# ):
#     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
#     rew_termination = rew_scale_terminated * reset_terminated.float()
#     rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
#     rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
#     rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
#     total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
#     return total_reward
