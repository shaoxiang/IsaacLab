# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, DeformableObject, DeformableObjectCfg, RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from omni.isaac.lab_assets import KAYA_TENNIS_CFG, TENNIS_BALL_CFG   # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

class KayaEnvWindow(BaseEnvWindow):
    """Window manager for the Kaya environment."""

    def __init__(self, env: KayaTennisEnv, window_name: str = "IsaacLab"):
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
class TennisSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = KAYA_TENNIS_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # tennis_ball: DeformableObjectCfg = TENNIS_BALL_CFG.replace(prim_path="/World/envs/env_.*/Tennis_ball")

    # Rigid Object
    tennis_ball_rigid: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tennis_ball_rigid",
        spawn = sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
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

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # contact sensor
    # contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/check_goal", update_period=0.01, history_length=3, debug_vis=False, filter_prim_paths_expr = ["/World/envs/env_.*/Tennis_ball"]
    # )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/check_goal", update_period=0.01, history_length=3, debug_vis=False, filter_prim_paths_expr = ["/World/envs/env_.*/Tennis_ball_rigid"]
    )

@configclass
class KayaTennisEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 3
    observation_space = 18
    state_space = 0
    debug_vis = True
    ui_window_class_type = KayaEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        enable_scene_query_support = True,
        use_fabric=True, 
        # physx=PhysxCfg(
        #     gpu_max_soft_body_contacts = 2**21,
        # ),
    )

    # scene
    scene: TennisSceneCfg = TennisSceneCfg(num_envs=4096, env_spacing=8.0, replicate_physics=False)
    # robot
    action_scale = 40.0
    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.2
    distance_to_goal_reward_scale = 3.0
    lose_goal_reward_scale = -1.0
    reach_goal_reward_scale = 1.0
    action_rate_reward_scale = -0.01
    joint_accel_reward_scale = -2.5e-7

class KayaTennisEnv(DirectRLEnv):
    cfg: KayaTennisEnvCfg

    def __init__(self, cfg: KayaTennisEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the uav
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # Goal position
        # self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._tennis_ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._tennis_ball_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._is_reach_goal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        self._robot = self.scene["robot"]
        # self._tennis_ball = self.scene["tennis_ball"]
        self._tennis_ball_rigid = self.scene["tennis_ball_rigid"]
        self._contact_sensor = self.scene["contact_sensor"]
        self._terrain = self.scene.terrain

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "lose_goal",
                "reach_goal",
                "distance_to_goal",
                "action_rate_l2",
                "joint_accel"
            ]
        }
        wheels_dof_names = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
        # Get specific body indices
        self._wheels_dof_idx, _ = self._robot.find_joints(wheels_dof_names)
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this isfv set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = self.cfg.action_scale * actions.clone().clamp(-1.0, 1.0)
        
    def _apply_action(self):
        # print("actions:", self._actions)
        self._robot.set_joint_velocity_target(self._actions, self._wheels_dof_idx)

    def _get_observations(self) -> dict:
        # desired_pos_b, _ = subtract_frame_transforms(
        #     self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._tennis_ball.data.root_pos_w
        # )

        # desired_pos_b, _ = subtract_frame_transforms(
        #     self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._tennis_ball_rigid.data.root_pos_w
        # )
        self._previous_actions = self._actions.clone()
        
        desired_pos_b = self._robot.data.root_pos_w - self._tennis_ball_rigid.data.root_pos_w

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                self._tennis_ball_rigid.data.root_lin_vel_b,
                # self._tennis_ball_rigid.data.projected_gravity_b,
                self._actions,
            ],
            dim=-1,
        )
        # print("obs:", obs)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        max_net_contact_forces, _ = torch.max(net_contact_forces.view(net_contact_forces.size(0), -1), dim=1)
        self._is_reach_goal = max_net_contact_forces > 0.1
        # self._is_collision_occurred = max_net_contact_forces > 0.05
        # print("max_net_contact_forces:", max_net_contact_forces)

        # reach_goal_num = self._is_reach_goal.sum().item()
        # if reach_goal_num > 0:
        #     print("Success get goal:", reach_goal_num)

        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        distance_to_goal = torch.linalg.norm(self._tennis_ball_rigid.data.root_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(0.6 * distance_to_goal)
        # distance_to_goal_mapped = distance_to_goal < 0.5

        reset_ball = self._tennis_ball_rigid.data.root_pos_w[:, 2] < 0.08
        reset_ball_ids = reset_ball.nonzero(as_tuple=False).squeeze(-1)
        # print("reset_ball_ids:", reset_ball_ids)
        if len(reset_ball_ids) > 0:
            # self.reset_tennis_ball(reset_ball_ids)
            self.reset_tennis_ball_rigid(reset_ball_ids)
        
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            # "lose_goal": reset_ball * self.cfg.lose_goal_reward_scale,
            "reach_goal": self._is_reach_goal * self.cfg.reach_goal_reward_scale,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "joint_accel": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        distance_to_orin = torch.linalg.norm(self._terrain.env_origins - self._robot.data.root_pos_w, dim=1)
        # far_away_orin = distance_to_orin > 4.0
        died = distance_to_orin > 5.0
        # died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.5, self._robot.data.root_pos_w[:, 2] > 15.0)
        time_out = torch.logical_or(time_out, self._is_reach_goal)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._tennis_ball_rigid.data.root_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        final_distance_to_orin = torch.linalg.norm(
            self._terrain.env_origins[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
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
        extras["Metrics/final_distance_to_orin"] = final_distance_to_orin.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._is_reach_goal[env_ids] = False
        # Sample new commands
        # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-4.0, 4.0)
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # self.reset_tennis_ball(env_ids)
        self.reset_tennis_ball_rigid(env_ids)

    def reset_tennis_ball_rigid(self, env_ids: torch.Tensor | None):
        default_tennis_ball_rigid_state = self._tennis_ball_rigid.data.default_root_state[env_ids]
        default_tennis_ball_rigid_state[:, :2] = torch.zeros_like(self._tennis_ball_pos[env_ids, :2]).uniform_(-2.5, 2.5)
        default_tennis_ball_rigid_state[:, 2] = torch.zeros_like(self._tennis_ball_pos[env_ids, 2]).uniform_(2.0, 6.0)
        default_tennis_ball_rigid_state[:, :3] += self._terrain.env_origins[env_ids]
        self._tennis_ball_rigid.write_root_pose_to_sim(default_tennis_ball_rigid_state[:, :7], env_ids)
        self._tennis_ball_rigid.write_root_velocity_to_sim(default_tennis_ball_rigid_state[:, 7:], env_ids)
        self._tennis_ball_rigid.reset(env_ids)

    def reset_tennis_ball(self, env_ids: torch.Tensor | None):
        # reset the nodal state of the object
        nodal_state = self._tennis_ball.data.default_nodal_state_w[env_ids].clone()
        
        self._tennis_ball_pos[env_ids, :2] = torch.zeros_like(self._tennis_ball_pos[env_ids, :2]).uniform_(-0.2, 0.2)
        # self._tennis_ball_pos[env_ids, :2] = torch.zeros_like(self._tennis_ball_pos[env_ids, :2])
        self._tennis_ball_pos[env_ids, 2] = torch.zeros_like(self._tennis_ball_pos[env_ids, 2]).uniform_(2.0, 5.0)
        # self._tennis_ball_pos[env_ids, :3] += self._terrain.env_origins[env_ids]
        nodal_state[..., :3] = self._tennis_ball.transform_nodal_pos(nodal_state[..., :3], self._tennis_ball_pos[env_ids])
        # nodal_state[..., 3:] = torch.zeros_like(nodal_state[..., 3:]).uniform_(-10.0, 10.0)
        # write nodal state to simulation
        self._tennis_ball.write_nodal_state_to_sim(nodal_state, env_ids = env_ids)
        # reset buffers
        self._tennis_ball.reset(env_ids)

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

@configclass
class TennisPlaySceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = KAYA_TENNIS_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    tennis_ball: DeformableObjectCfg = TENNIS_BALL_CFG.replace(prim_path="/World/envs/env_.*/Tennis_ball")

    # Rigid Object

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

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # contact sensor
    # contact_sensor: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/check_goal", update_period=0.01, history_length=3, debug_vis=False, filter_prim_paths_expr = ["/World/envs/env_.*/Tennis_ball"]
    # )

    table = AssetBaseCfg(
        prim_path="/World/Tennis_Court",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Tennis_Court/tennis_court_egg.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/check_goal", update_period=0.01, history_length=3, debug_vis=False, filter_prim_paths_expr = ["/World/envs/env_.*/Tennis_ball_rigid"]
    )

@configclass
class KayaTennisPlayEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 3
    observation_space = 18
    state_space = 0
    debug_vis = True
    ui_window_class_type = KayaEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        enable_scene_query_support = True,
        use_fabric=True, 
        physx=PhysxCfg(
            gpu_max_soft_body_contacts = 2**22,
        ),
    )

    # scene
    scene: TennisPlaySceneCfg = TennisPlaySceneCfg(num_envs=4096, env_spacing=8.0, replicate_physics=False)
    # robot
    action_scale = 20.0
    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 1.0
    lose_goal_reward_scale = -1.0
    reach_goal_reward_scale = 1000.0

class KayaTennisPlayEnv(DirectRLEnv):
    cfg: KayaTennisPlayEnvCfg

    def __init__(self, cfg: KayaTennisPlayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the uav
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # self._last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # Goal position
        # self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._tennis_ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._tennis_ball_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._is_reach_goal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        self._robot = self.scene["robot"]
        self._tennis_ball = self.scene["tennis_ball"]
        # self._tennis_ball_rigid = self.scene["tennis_ball_rigid"]
        self._contact_sensor = self.scene["contact_sensor"]
        self._terrain = self.scene.terrain

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "lose_goal",
                "reach_goal",
                # "distance_to_goal",
            ]
        }
        wheels_dof_names = ["axle_0_joint", "axle_1_joint", "axle_2_joint"]
        # Get specific body indices
        self._wheels_dof_idx, _ = self._robot.find_joints(wheels_dof_names)
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this isfv set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = self.cfg.action_scale * actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        # print("actions:", self._actions)
        self._robot.set_joint_velocity_target(self._actions, self._wheels_dof_idx)

    def _get_observations(self) -> dict:
        # desired_pos_b, _ = subtract_frame_transforms(
        #     self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._tennis_ball.data.root_pos_w
        # )

        # desired_pos_b, _ = subtract_frame_transforms(
        #     self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._tennis_ball_rigid.data.root_pos_w
        # )
        
        desired_pos_b = self._robot.data.root_pos_w - self._tennis_ball.data.root_pos_w

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                self._tennis_ball.data.root_vel_w,
                # self._tennis_ball_rigid.data.projected_gravity_b,
                self._actions,
            ],
            dim=-1,
        )
        # print("obs:", obs)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        max_net_contact_forces, _ = torch.max(net_contact_forces.view(net_contact_forces.size(0), -1), dim=1)
        self._is_reach_goal = max_net_contact_forces > 0.1
        # self._is_collision_occurred = max_net_contact_forces > 0.05
        # print("max_net_contact_forces:", max_net_contact_forces)

        # reach_goal_num = self._is_reach_goal.sum().item()
        # if reach_goal_num > 0:
        #     print("Success get goal:", reach_goal_num)

        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # distance_to_goal = torch.linalg.norm(self._tennis_ball_rigid.data.root_pos_w - self._robot.data.root_pos_w, dim=1)
        # distance_to_goal_mapped = 1 - torch.tanh(0.4 * distance_to_goal)
        # distance_to_goal_mapped = distance_to_goal < 0.5

        # reset_ball = self._tennis_ball.data.root_pos_w[:, 2] < 0.08
        # reset_ball_ids = reset_ball.nonzero(as_tuple=False).squeeze(-1)
        # # print("reset_ball_ids:", reset_ball_ids)
        # if len(reset_ball_ids) > 0:
        #     # self.reset_tennis_ball(reset_ball_ids)
        #     self.reset_tennis_ball_rigid(reset_ball_ids)
        
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            # "lose_goal": reset_ball * self.cfg.lose_goal_reward_scale,
            "reach_goal": self._is_reach_goal * self.cfg.reach_goal_reward_scale,
            # "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        distance_to_orin = torch.linalg.norm(self._terrain.env_origins - self._robot.data.root_pos_w, dim=1)
        # far_away_orin = distance_to_orin > 4.0
        died = distance_to_orin > 5.0
        # died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.5, self._robot.data.root_pos_w[:, 2] > 15.0)
        # time_out = torch.logical_or(time_out, self._is_reach_goal)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._tennis_ball.data.root_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        final_distance_to_orin = torch.linalg.norm(
            self._terrain.env_origins[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
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
        extras["Metrics/final_distance_to_orin"] = final_distance_to_orin.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-4.0, 4.0)
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.reset_tennis_ball(env_ids)
        # self.reset_tennis_ball_rigid(env_ids)

    def reset_tennis_ball(self, env_ids: torch.Tensor | None):
        # reset the nodal state of the object
        nodal_state = self._tennis_ball.data.default_nodal_state_w[env_ids].clone()
        
        self._tennis_ball_pos[env_ids, :2] = torch.zeros_like(self._tennis_ball_pos[env_ids, :2]).uniform_(-0.2, 0.2)
        # self._tennis_ball_pos[env_ids, :2] = torch.zeros_like(self._tennis_ball_pos[env_ids, :2])
        self._tennis_ball_pos[env_ids, 2] = torch.zeros_like(self._tennis_ball_pos[env_ids, 2]).uniform_(5.0, 6.0)
        # self._tennis_ball_pos[env_ids, :3] += self._terrain.env_origins[env_ids]
        nodal_state[..., :3] = self._tennis_ball.transform_nodal_pos(nodal_state[..., :3], self._tennis_ball_pos[env_ids])
        # nodal_state[..., 3:] = torch.zeros_like(nodal_state[..., 3:]).uniform_(-10.0, 10.0)
        # write nodal state to simulation
        self._tennis_ball.write_nodal_state_to_sim(nodal_state, env_ids = env_ids)
        # reset buffers
        self._tennis_ball.reset(env_ids)