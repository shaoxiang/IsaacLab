# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym
import random

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import random_orientation, subtract_frame_transforms, yaw_angle, wrap_to_pi, quat_error_magnitude
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterFormEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterFormEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene 
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=10.0, replicate_physics=True) # 65536 32768 16384 
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True) 
    # 8192
    # robot
    formations = [[0.5, 0.5, 5.0], [-0.5, 0.5, 5.0], [-0.5, -0.5, 5.0], [0.5, -0.5, 5.0]]

    # robots = []
    # for i, formation in enumerate(formations):
    #     crazyfile_cfg = CRAZYFLIE_CFG.copy()
    #     crazyfile_cfg.prim_path = f"/World/envs/env_.*/Origin{i}"
    #     crazyfile_cfg.init_state.pos = formation
    #     robots.append(crazyfile_cfg)

    crazyfile_cfg = CRAZYFLIE_CFG.copy()
    crazyfile_cfg.prim_path = f"/World/envs/env_.*/Origin0"
    crazyfile_cfg.init_state.pos = formations[0]
    robot1: ArticulationCfg = crazyfile_cfg
    crazyfile_cfg = CRAZYFLIE_CFG.copy()
    crazyfile_cfg.prim_path = f"/World/envs/env_.*/Origin1"
    crazyfile_cfg.init_state.pos = formations[1]
    robot2: ArticulationCfg = crazyfile_cfg
    crazyfile_cfg = CRAZYFLIE_CFG.copy()
    crazyfile_cfg.prim_path = f"/World/envs/env_.*/Origin2"
    crazyfile_cfg.init_state.pos = formations[2]
    robot3: ArticulationCfg = crazyfile_cfg
    crazyfile_cfg = CRAZYFLIE_CFG.copy()
    crazyfile_cfg.prim_path = f"/World/envs/env_.*/Origin3"
    crazyfile_cfg.init_state.pos = formations[3]
    robot4: ArticulationCfg = crazyfile_cfg

    # robot1: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Origin0")
    # robot2: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Origin1")
    # robot3: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Origin2")
    # robot4: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Origin3")
    
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # env
    episode_length_s = 5.0 # 10.0
    decimation = 2
    num_actions = 4 * 4
    num_observations = 3 * 4 * 3 + 6 + 3 + 1 + 4 * 3
    num_states = 0
    debug_vis = True

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.1

    # lin_vel_reward_scale = -0.5
    # ang_vel_reward_scale = -1.0
    max_formation_width = 3.0 # 5.0

    angle_reward_scale = 1.0
    distance_to_goal_reward_scale = 30.0 # 30..0
    goal_reached_reward_scale = 10.0
    effort_reward_scale = 1.0
    speed_reward_scale = 0.1
    formation_reward_scale = 3.0

class QuadcopterFormEnv(DirectRLEnv):
    cfg: QuadcopterFormEnvCfg

    def __init__(self, cfg: QuadcopterFormEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust1 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment1 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust2 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment2 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust3 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment3 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust4 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment4 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._is_goal_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._is_formation_failed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._goal_distance = torch.ones(self.num_envs, device=self.device)
        self._form_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._form_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "angle_error",
                "goal_reached",
                "effort",
                "formation"
            ]
        }
        # Get specific body indices
        self._body_id1 = self._robot1.find_bodies("body")[0]
        self._body_id2 = self._robot2.find_bodies("body")[0]
        self._body_id3 = self._robot3.find_bodies("body")[0]
        self._body_id4 = self._robot4.find_bodies("body")[0]
        
        self._robot_mass = self._robot1.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot1 = Articulation(self.cfg.robot1)
        self.scene.articulations["robot1"] = self._robot1
        self._robot2 = Articulation(self.cfg.robot2)
        self.scene.articulations["robot2"] = self._robot2
        self._robot3 = Articulation(self.cfg.robot3)
        self.scene.articulations["robot3"] = self._robot3
        self._robot4 = Articulation(self.cfg.robot4)
        self.scene.articulations["robot4"] = self._robot4

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust1[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 3.0
        self._moment1[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:4] / 4.0
        self._thrust2[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 4] + 1.0) / 3.0
        self._moment2[:, 0, :] = self.cfg.moment_scale * self._actions[:, 5:8] / 4.0
        self._thrust3[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 8] + 1.0) / 3.0
        self._moment3[:, 0, :] = self.cfg.moment_scale * self._actions[:, 9:12] / 4.0
        self._thrust4[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 12] + 1.0) / 3.0
        self._moment4[:, 0, :] = self.cfg.moment_scale * self._actions[:, 13:16] / 4.0

    def _apply_action(self):
        self._robot1.set_external_force_and_torque(self._thrust1, self._moment1, body_ids=self._body_id1)
        self._robot2.set_external_force_and_torque(self._thrust2, self._moment2, body_ids=self._body_id2)
        self._robot3.set_external_force_and_torque(self._thrust3, self._moment3, body_ids=self._body_id3)
        self._robot4.set_external_force_and_torque(self._thrust4, self._moment4, body_ids=self._body_id4)
        
    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,)
        )
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
            self.state_space = gym.vector.utils.batch_space(self.single_observation_space["critic"], self.num_envs)

    def get_formation_state(self):
        form_pos = (self._robot1.data.root_state_w[:, :3] + self._robot2.data.root_state_w[:, :3] + self._robot3.data.root_state_w[:, :3] + self._robot4.data.root_state_w[:, :3]) / 4.0
        form_attitude = (self._robot1.data.root_state_w[:, 3:7] +  self._robot2.data.root_state_w[:, 3:7] + self._robot3.data.root_state_w[:, 3:7] + self._robot4.data.root_state_w[:, 3:7]) / 4.0
        return form_pos, form_attitude
    
    def formation_check(self, data_or_result = True):        
        f1 = torch.square(self._robot1.data.root_state_w[:, :3] - self._robot2.data.root_state_w[:, :3]).sum(dim=1).sqrt()  
        f2 = torch.square(self._robot2.data.root_state_w[:, :3] - self._robot3.data.root_state_w[:, :3]).sum(dim=1).sqrt() 
        f3 = torch.square(self._robot3.data.root_state_w[:, :3] - self._robot4.data.root_state_w[:, :3]).sum(dim=1).sqrt()  
        f4 = torch.square(self._robot4.data.root_state_w[:, :3] - self._robot1.data.root_state_w[:, :3]).sum(dim=1).sqrt()
        f5 = torch.square(self._robot1.data.root_state_w[:, :3] - self._robot3.data.root_state_w[:, :3]).sum(dim=1).sqrt()
        f6 = torch.square(self._robot2.data.root_state_w[:, :3] - self._robot4.data.root_state_w[:, :3]).sum(dim=1).sqrt()

        if data_or_result:
            return f1, f2, f3, f4, f5, f6
        else:
            dones = (f1 > self.cfg.max_formation_width) + (f2 > self.cfg.max_formation_width) + (f3 > self.cfg.max_formation_width) + (f4 > self.cfg.max_formation_width) + (f5 > 1.414 * self.cfg.max_formation_width) + (f6 > 1.414 * self.cfg.max_formation_width)
            rewards = 1.0 - 4.0 * torch.square(f1 - 1.0) - 4.0 * torch.square(f2 - 1.0) - 4.0 * torch.square(f3 - 1.0) - 4.0 * torch.square(f4 - 1.0) - 2.0 * torch.square(f5 - 1.414) - 2.0 * torch.square(f6 - 1.414)
            return rewards, dones

    def _get_observations(self) -> dict:
        self._form_pos_w, self._form_quat_w = self.get_formation_state()
        desired_pos_b, _ = subtract_frame_transforms(
            self._form_pos_w, self._form_quat_w, self._desired_pos_w
        )
        # pos_error, angle_error = compute_pose_error(self._robot1.data.root_pos_w, self._robot1.data.root_quat_w, self._desired_pos_w, self._desired_quat_w)
        # desired_pos_b = self._robot1.data.root_state_w[:, :3] - self._desired_pos_w
        yaw_robo = yaw_angle(self._form_quat_w)
        yaw_desired = yaw_angle(self._desired_quat_w)
        angle_error = wrap_to_pi(yaw_robo - yaw_desired).unsqueeze(1)
        f1, f2, f3, f4, f5, f6 = self.formation_check(True)

        # print("f1:", torch.stack((f1, f2, f3, f4), dim=0).shape, "root_lin_vel_b:", self._robot1.data.root_lin_vel_b.shape, "angle_error:", angle_error.shape)

        # obs = torch.cat(
        #     [
        #         (self._robot1.data.root_lin_vel_b + self._robot2.data.root_lin_vel_b + self._robot3.data.root_lin_vel_b + self._robot4.data.root_lin_vel_b) / 4.0,
        #         (self._robot1.data.root_ang_vel_b + self._robot2.data.root_ang_vel_b + self._robot3.data.root_ang_vel_b + self._robot4.data.root_ang_vel_b) / 4.0,
        #         (self._robot1.data.projected_gravity_b + self._robot2.data.projected_gravity_b + self._robot3.data.projected_gravity_b + self._robot4.data.projected_gravity_b) / 4.0,
        #         torch.stack((f1, f2, f3, f4, f5, f6), dim=1),
        #         desired_pos_b,
        #         # pos_error,
        #         angle_error,
        #     ],
        #     dim=-1,
        # )

        p1 = self._robot1.data.root_state_w[:, :3] - self._desired_pos_w
        p2 = self._robot2.data.root_state_w[:, :3] - self._desired_pos_w
        p3 = self._robot3.data.root_state_w[:, :3] - self._desired_pos_w
        p4 = self._robot4.data.root_state_w[:, :3] - self._desired_pos_w

        obs = torch.cat(
            [
                self._robot1.data.root_lin_vel_b,
                self._robot1.data.root_ang_vel_b,
                self._robot1.data.projected_gravity_b,
                self._robot2.data.root_lin_vel_b,
                self._robot2.data.root_ang_vel_b,
                self._robot2.data.projected_gravity_b,
                self._robot3.data.root_lin_vel_b,
                self._robot3.data.root_ang_vel_b,
                self._robot3.data.projected_gravity_b,
                self._robot4.data.root_lin_vel_b,
                self._robot4.data.root_ang_vel_b,
                self._robot4.data.projected_gravity_b,
                torch.stack((f1, f2, f3, f4, f5, f6), dim=1),
                p1, p2, p3, p4,
                desired_pos_b,
                # pos_error,
                angle_error,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        
        form_lin_vel = torch.sum(torch.square(self._robot1.data.root_lin_vel_b) + torch.square(self._robot2.data.root_lin_vel_b) + torch.square(self._robot3.data.root_lin_vel_b) + torch.square(self._robot4.data.root_lin_vel_b), dim=1)

        # lin_vel = torch.sum(torch.square(self._robot1.data.root_lin_vel_b), dim=1)

        form_ang_vel = (self._robot1.data.root_ang_vel_b + self._robot2.data.root_ang_vel_b + self._robot3.data.root_ang_vel_b + self._robot4.data.root_ang_vel_b) / 4.0

        root_ang_vel_b_clone = form_ang_vel.clone()
        root_ang_vel_b_clone[:, 2] = 0
        ang_vel = torch.sum(torch.square(root_ang_vel_b_clone), dim=1)

        # self._form_pos_w, self._form_quat_w = self.get_formation_state()
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._form_pos_w, dim=1)
        self._is_goal_reached = torch.linalg.norm(self._desired_pos_w - self._form_pos_w, dim=1) < 0.05

        distance_to_goal_exp = torch.exp(-1.5 * distance_to_goal)
        distance_to_goal_mapped = (1.0 / (0.1 + distance_to_goal / self._goal_distance)) - 0.9

        yaw_robo = yaw_angle(self._form_quat_w)
        yaw_desired = yaw_angle(self._desired_quat_w)
        angle_error = wrap_to_pi(yaw_robo - yaw_desired)

        angle_error_mapped = 10.0 * (torch.cos(angle_error) - 0.99)
        reach_goal_mean_speed = self._goal_distance / (self.episode_length_buf * self.step_dt)

        # valid_mean_speed = (self._goal_distance - distance_to_goal) / (self.episode_length_buf * self.step_dt)
        # effort = torch.exp(-self._actions.sum(dim=1))
        effort = torch.exp(-torch.abs(self._actions.sum(dim=1)))
        # keep formation
        formation_reward, self._is_formation_failed = self.formation_check(False)

        rewards = {
            "lin_vel": form_lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "angle_error": distance_to_goal_exp * angle_error_mapped * self.cfg.angle_reward_scale * self.step_dt,
            "goal_reached": self._is_goal_reached * reach_goal_mean_speed * self.cfg.goal_reached_reward_scale,
            "effort": effort * self.cfg.effort_reward_scale * self.step_dt,
            "formation": formation_reward * self.cfg.formation_reward_scale * self.step_dt,
            # "mean_speed": valid_mean_speed * self.cfg.speed_reward_scale,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # print("valid_mean_speed:", valid_mean_speed, "_goal_distance:", self._goal_distance, "reach_goal_mean_speed:", reach_goal_mean_speed, self.max_episode_length, self.episode_length_buf, rewards)
        # print("rewards:", reward, rewards)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = torch.logical_or((self.episode_length_buf >= self.max_episode_length - 1), self._is_goal_reached)
        p1 = torch.logical_or(self._robot1.data.root_pos_w[:, 2] < 0.05, self._robot1.data.root_pos_w[:, 2] > 15.0)
        p2 = torch.logical_or(self._robot2.data.root_pos_w[:, 2] < 0.05, self._robot2.data.root_pos_w[:, 2] > 15.0)
        p3 = torch.logical_or(self._robot3.data.root_pos_w[:, 2] < 0.05, self._robot3.data.root_pos_w[:, 2] > 15.0)
        p4 = torch.logical_or(self._robot4.data.root_pos_w[:, 2] < 0.05, self._robot4.data.root_pos_w[:, 2] > 15.0)
        died = torch.logical_or(p1 + p2 + p3 + p4, self._is_formation_failed)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot1._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._form_pos_w[env_ids], dim=1
        ) 
        final_distance_to_goal_scale = (final_distance_to_goal / self._goal_distance[env_ids]).mean()
        # print("final_distance_to_goal:", final_distance_to_goal, "length:", self._goal_distance[env_ids], "scale:", final_distance_to_goal_scale)

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal_scale"] = final_distance_to_goal_scale.item()
        self.extras["log"].update(extras)

        self._robot1.reset(env_ids)
        self._robot2.reset(env_ids)
        self._robot3.reset(env_ids)
        self._robot4.reset(env_ids)
        
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        # x、y
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-10.0, 10.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # z
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 8.0)
        self._desired_quat_w[env_ids,] = random_orientation(num = 1, device=self.device)

        height_random = random.uniform(-4.0, 6.0)
        x_shift = random.uniform(-1.0, 1.0)
        y_shift = random.uniform(-1.0, 1.0)
        
        # Reset robot state
        joint_pos = self._robot1.data.default_joint_pos[env_ids]
        joint_vel = self._robot1.data.default_joint_vel[env_ids]
        default_root_state = self._robot1.data.default_root_state[env_ids]
        default_root_state[:, 0] += (x_shift + random.uniform(-0.5, 0.5))
        default_root_state[:, 1] += (y_shift + random.uniform(-0.5, 0.5))
        default_root_state[:, 2] += (height_random + random.uniform(-0.5, 0.5)) 
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot1.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot1.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot1.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # print("default_root_state:", default_root_state)

        joint_pos = self._robot2.data.default_joint_pos[env_ids]
        joint_vel = self._robot2.data.default_joint_vel[env_ids]
        default_root_state = self._robot2.data.default_root_state[env_ids]
        default_root_state[:, 0] += (x_shift + random.uniform(-0.5, 0.5))
        default_root_state[:, 1] += (y_shift + random.uniform(-0.5, 0.5))
        default_root_state[:, 2] += (height_random + random.uniform(-0.5, 0.5)) 
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot2.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot2.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot2.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        joint_pos = self._robot3.data.default_joint_pos[env_ids]
        joint_vel = self._robot3.data.default_joint_vel[env_ids]
        default_root_state = self._robot3.data.default_root_state[env_ids]
        default_root_state[:, 0] += (x_shift + random.uniform(-0.5, 0.5))
        default_root_state[:, 1] += (y_shift + random.uniform(-0.5, 0.5))
        default_root_state[:, 2] += (height_random + random.uniform(-0.5, 0.5)) 
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot3.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot3.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot3.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        joint_pos = self._robot4.data.default_joint_pos[env_ids]
        joint_vel = self._robot4.data.default_joint_vel[env_ids]
        default_root_state = self._robot4.data.default_root_state[env_ids]
        default_root_state[:, 0] += (x_shift + random.uniform(-0.5, 0.5))
        default_root_state[:, 1] += (y_shift + random.uniform(-0.5, 0.5))
        default_root_state[:, 2] += (height_random + random.uniform(-0.5, 0.5)) 
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot4.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot4.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot4.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._is_goal_reached[env_ids] &= False
        self._is_formation_failed[env_ids] &= False

        self._form_pos_w, self._form_quat_w = self.get_formation_state()
        self._goal_distance[env_ids] = torch.linalg.norm(self._desired_pos_w[env_ids] - self._form_pos_w[env_ids], dim=1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_pos_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.3)
                self.base_target_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/quadcopter_current"
                marker_cfg.markers["arrow"].scale = (0.04, 0.04, 0.14)
                self.base_quadcopter_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_target_goal_visualizer.set_visibility(True)
            self.base_quadcopter_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.base_target_goal_visualizer.set_visibility(False)
                self.base_quadcopter_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # get marker location
        # -- base state
        quadcopter_pos_w = self._form_pos_w.clone()
        quadcopter_quat_w = self._form_quat_w.clone()
        # quadcopter_pos_w[:, 2] += 0.05
        # -- resolve the scales and quaternions
        # vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        # vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.base_target_goal_visualizer.visualize(self._desired_pos_w, self._desired_quat_w)
        self.base_quadcopter_visualizer.visualize(quadcopter_pos_w, quadcopter_quat_w)

