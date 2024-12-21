# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from omni.isaac.lab_assets import UAV_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

# import omni
# ext_manager = omni.kit.app.get_app().get_extension_manager()
# ext_manager.set_extension_enabled_immediate("omni.physx.forcefields", True)

class UAVControlEnvWindow(BaseEnvWindow):
    """Window manager for the UAV environment."""

    def __init__(self, env: UAVControlEnv, window_name: str = "IsaacLab"):
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
class UAVControlEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 13
    state_space = 0
    debug_vis = False

    ui_window_class_type = UAVControlEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = UAV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 10.0
    moment_scale = 0.05

    # reward scales
    lin_vel_reward_scale = 10.0
    ang_vel_reward_scale = 1.0
    error_to_goal_reward_scale = 15.0

class UAVControlEnv(DirectRLEnv):
    cfg: UAVControlEnvCfg

    def __init__(self, cfg: UAVControlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the uav
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal cmd
        self._desired_cmd = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                # "error_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.robot.init_state.pos = (0.0, 0.0, 100.0)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # rigidbody
        self._character = RigidObject(self.cfg.character_cfg)
        self.scene.rigid_objects["character"] = self._character
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 1] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._desired_cmd,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        vx = torch.square(self._robot.data.root_lin_vel_b[:, 0] - self._desired_cmd[:, 0])
        vy = torch.square(self._robot.data.root_lin_vel_b[:, 1] - self._desired_cmd[:, 1])
        vz = torch.square(self._robot.data.root_lin_vel_b[:, 2] - self._desired_cmd[:, 2])
        wz = torch.square(self._robot.data.root_ang_vel_b[:, 2] - self._desired_cmd[:, 3])

        # lin_vel_error_to_goal = vx + vy + vz
        lin_vel_error_to_goal_mapped = 3.0 - torch.tanh(vx) - torch.tanh(vy) - torch.tanh(vz)

        # ang_vel_error_to_goal = wz
        ang_vel_error_to_goal_mapped = 1.0 - torch.tanh(wz)

        # print("lin_vel_error_to_goal:", lin_vel_error_to_goal, lin_vel_error_to_goal_mapped, "ang_vel_error_to_goal:", ang_vel_error_to_goal, ang_vel_error_to_goal_mapped)
        
        rewards = {
            "lin_vel": lin_vel_error_to_goal_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_error_to_goal_mapped * self.cfg.ang_vel_reward_scale * self.step_dt,
            # "error_to_goal": error_to_goal_mapped * self.cfg.error_to_goal_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # print("rewards:", rewards, "reward:", reward, "dt:", self.step_dt)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 200.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_error_to_goal = torch.linalg.norm(
            self._desired_cmd[env_ids, 0:3] - self._robot.data.root_lin_vel_b[env_ids], dim=1
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
        extras["Metrics/final_error_to_goal"] = final_error_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands        
        self._desired_cmd[env_ids, 0] = torch.zeros_like(self._desired_cmd[env_ids, 0]).uniform_(-2.0, 2.0)
        self._desired_cmd[env_ids, 1] = torch.zeros_like(self._desired_cmd[env_ids, 1]).uniform_(-2.0, 2.0)
        self._desired_cmd[env_ids, 2] = torch.zeros_like(self._desired_cmd[env_ids, 2]).uniform_(-4.0, 4.0)
        self._desired_cmd[env_ids, 3] = torch.zeros_like(self._desired_cmd[env_ids, 3]).uniform_(-3.0, 3.0)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     # create markers if necessary for the first tome
    #     if debug_vis:
    #         if not hasattr(self, "goal_pos_visualizer"):
    #             marker_cfg = CUBOID_MARKER_CFG.copy()
    #             marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
    #             # -- goal pose
    #             marker_cfg.prim_path = "/Visuals/Command/goal_position"
    #             self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
    #         # set their visibility to true
    #         self.goal_pos_visualizer.set_visibility(True)
    #     else:
    #         if hasattr(self, "goal_pos_visualizer"):
    #             self.goal_pos_visualizer.set_visibility(False)

    # def _debug_vis_callback(self, event):
    #     # update the markers
    #     self.goal_pos_visualizer.visualize(self._desired_pos_w)

class UAVPTZControlEnv(DirectRLEnv):
    cfg: UAVControlEnvCfg

    def __init__(self, cfg: UAVControlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the uav
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal cmd
        self._desired_cmd = torch.zeros(self.num_envs, 4, device=self.device)
        self._ptz_action = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                # "error_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.ptz_joint_x_idx, _ = self._robot.find_joints("jointX")
        self.ptz_joint_y_idx, _ = self._robot.find_joints("jointY")
        self.ptz_joint_z_idx, _ = self._robot.find_joints("jointZ")
        self.ptz_joint_x_scale = 0.9 * 100.0
        self.ptz_joint_y_scale = 0.9 * 50.0
        self.ptz_joint_z_scale = 0.9 * 150.0
        
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.robot.init_state.pos = (0.0, 0.0, 100.0)
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
        self._thrust[:, 0, 1] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        self._robot.set_joint_position_target(self.ptz_joint_x_scale * self._ptz_action[:, :, 0], joint_ids=self.ptz_joint_x_idx)
        self._robot.set_joint_position_target(self.ptz_joint_y_scale * self._ptz_action[:, :, 1], joint_ids=self.ptz_joint_y_idx)
        self._robot.set_joint_position_target(self.ptz_joint_z_scale * self._ptz_action[:, :, 2], joint_ids=self.ptz_joint_z_idx)

    def _get_observations(self) -> dict:

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._desired_cmd,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        # ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        vx = torch.square(self._robot.data.root_lin_vel_b[:, 0] - self._desired_cmd[:, 0])
        vy = torch.square(self._robot.data.root_lin_vel_b[:, 1] - self._desired_cmd[:, 1])
        vz = torch.square(self._robot.data.root_lin_vel_b[:, 2] - self._desired_cmd[:, 2])
        wz = torch.square(self._robot.data.root_ang_vel_b[:, 2] - self._desired_cmd[:, 3])

        # print("vx:",vx,"vy:",vy,"vz:",vz,"wz:",wz)
        # print("vx_:",self._robot.data.root_lin_vel_b[:, 0],"vy_:",self._robot.data.root_lin_vel_b[:, 1],"vz_:",self._robot.data.root_lin_vel_b[:, 2],"wz_:",self._robot.data.root_ang_vel_b[:, 2])
        
        # lin_vel_error_to_goal = vx + vy + vz
        lin_vel_error_to_goal_mapped = 3.0 - torch.tanh(vx) - torch.tanh(vy) - torch.tanh(vz)

        # ang_vel_error_to_goal = wz
        ang_vel_error_to_goal_mapped = 1.0 - torch.tanh(wz)

        # print("lin_vel_error_to_goal:", lin_vel_error_to_goal, lin_vel_error_to_goal_mapped, "ang_vel_error_to_goal:", ang_vel_error_to_goal, ang_vel_error_to_goal_mapped)
        
        # rewards = {
        #     "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
        #     "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
        #     # "error_to_goal": error_to_goal_mapped * self.cfg.error_to_goal_reward_scale * self.step_dt,
        # }

        rewards = {
            "lin_vel": lin_vel_error_to_goal_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_error_to_goal_mapped * self.cfg.ang_vel_reward_scale * self.step_dt,
            # "error_to_goal": error_to_goal_mapped * self.cfg.error_to_goal_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # print("rewards:", rewards, "reward:", reward, "dt:", self.step_dt)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 200.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_error_to_goal = torch.linalg.norm(
            self._desired_cmd[env_ids, 0:3] - self._robot.data.root_lin_vel_b[env_ids], dim=1
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
        extras["Metrics/final_error_to_goal"] = final_error_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        lin_vel = 5.0
        lin_ang = 5.0        
        self._desired_cmd[env_ids, 0] = torch.zeros_like(self._desired_cmd[env_ids, 0]).uniform_(-lin_vel, lin_vel)
        self._desired_cmd[env_ids, 1] = torch.zeros_like(self._desired_cmd[env_ids, 1]).uniform_(-lin_vel, lin_vel)
        self._desired_cmd[env_ids, 2] = torch.zeros_like(self._desired_cmd[env_ids, 2]).uniform_(-lin_vel, lin_vel)
        self._desired_cmd[env_ids, 3] = torch.zeros_like(self._desired_cmd[env_ids, 3]).uniform_(-lin_ang, lin_ang)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._ptz_action[env_ids, :] = torch.zeros_like(self._ptz_action[env_ids, :]).uniform_(-1.0, 1.0) + joint_pos[:,4:7].unsqueeze(1)

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     # create markers if necessary for the first tome
    #     if debug_vis:
    #         if not hasattr(self, "goal_pos_visualizer"):
    #             marker_cfg = CUBOID_MARKER_CFG.copy()
    #             marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
    #             # -- goal pose
    #             marker_cfg.prim_path = "/Visuals/Command/goal_position"
    #             self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
    #         # set their visibility to true
    #         self.goal_pos_visualizer.set_visibility(True)
    #     else:
    #         if hasattr(self, "goal_pos_visualizer"):
    #             self.goal_pos_visualizer.set_visibility(False)

    # def _debug_vis_callback(self, event):
    #     # update the markers
    #     self.goal_pos_visualizer.visualize(self._desired_pos_w)