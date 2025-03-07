# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg

##
# Pre-defined configs
##
from isaaclab_assets import SCOUT_MINI_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG, Sphere_MARKER_CFG  # isort: skip

class ScoutMiniEnvWindow(BaseEnvWindow):
    """Window manager for the ScoutMini environment."""

    def __init__(self, env: ScoutMiniEnv, window_name: str = "IsaacLab"):
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
class ScoutMiniEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 50.0
    decimation = 2
    action_space = 2
    observation_space = 15
    state_space = 0
    debug_vis = True

    ui_window_class_type = ScoutMiniEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        enable_scene_query_support = True,
        use_fabric=True, 
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing = 10.0, replicate_physics=True)
    # robot
    robot: ArticulationCfg = SCOUT_MINI_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor_base_link: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link", update_period=0.01, history_length=3, debug_vis=False
    )

    action_scale = 20.0

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.1
    distance_to_goal_reward_scale = 5.0
    collision_reward_scale = -1.0
    action_rate_reward_scale = -0.01
    goal_reached_reward_scale = 20.0
    delta_goal_reward_scale = 50.0

class ScoutMiniEnv(DirectRLEnv):
    cfg: ScoutMiniEnvCfg

    def __init__(self, cfg: ScoutMiniEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the uav
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._collision = torch.zeros(self.num_envs, device=self.device)
        self._is_goal_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._dist_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._previous_dist_to_goal = torch.zeros(self.num_envs, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "collision",
                "action_rate_l2",
                "goal_reached",
                "delta_goal_dist",
            ]
        }
        wheels_dof_names = ["front_right_wheel", "front_left_wheel", "rear_right_wheel", "rear_left_wheel"]

        # Get specific body indices
        self._wheels_dof_idx, _ = self._robot.find_joints(wheels_dof_names)
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor_base_link = ContactSensor(self.cfg.contact_sensor_base_link)
        self.scene.sensors["contact_sensor_base_link"] = self._contact_sensor_base_link
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
        self._actions = self.cfg.action_scale * actions.clone().clamp(-1.0, 1.0)
        
    def _apply_action(self):
        wheel_actions = torch.cat((self._actions, self._actions), dim=1)
        # print("wheel_actions:", self._actions, wheel_actions)
        self._robot.set_joint_velocity_target(wheel_actions, self._wheels_dof_idx)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                self._dist_to_goal.unsqueeze(1),
                self._actions,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        self._dist_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(0.2 * self._dist_to_goal)
        delta_goal_dist = self._previous_dist_to_goal - self._dist_to_goal
        self._previous_dist_to_goal = self._dist_to_goal.clone()

        net_contact_forces = self._contact_sensor_base_link.data.net_forces_w_history
        max_net_contact_forces, _ = torch.max(net_contact_forces.view(net_contact_forces.size(0), -1), dim=1)
        self._collision = max_net_contact_forces > 3.0

        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        self._is_goal_reached = self._dist_to_goal < 0.5
        reset_goal_ids = self._is_goal_reached.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_goal_ids) > 0:
            self.reset_goal(reset_goal_ids)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "collision": self._collision.float() * self.cfg.collision_reward_scale,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "goal_reached": self._is_goal_reached.float() * self.cfg.goal_reached_reward_scale,
            "delta_goal_dist": delta_goal_dist * self.cfg.delta_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # time_out = torch.logical_or(self.episode_length_buf >= self.max_episode_length - 1, self._is_goal_reached)
        died = self._collision
        # died = self._robot.data.root_pos_w[:, 2] > 1.0
        # print("max_net_contact_forces:", max_net_contact_forces)
        # died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.5, self._robot.data.root_pos_w[:, 2] > 15.0)
        return died, time_out
    
    def reset_goal(self, env_ids: torch.Tensor | None):
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-20.0, 20.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])
        self._previous_dist_to_goal[env_ids] = torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
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
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._collision[env_ids] = 0
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._is_goal_reached[env_ids] &= False
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-20.0, 20.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._previous_dist_to_goal[env_ids] = torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = Sphere_MARKER_CFG.copy()
                marker_cfg.markers["spheroid"].radius = 0.2
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
