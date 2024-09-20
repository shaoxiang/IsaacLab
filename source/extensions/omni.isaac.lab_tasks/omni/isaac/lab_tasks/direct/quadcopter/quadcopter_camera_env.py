# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import gymnasium as gym
import numpy as np
import random

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import random_orientation, subtract_frame_transforms, yaw_angle, wrap_to_pi
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterCameraEnv, window_name: str = "IsaacLabSX"):
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
class QuadcopterRGBCameraEnvCfg(DirectRLEnvCfg):
    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        disable_contact_processing=True,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=True) # 32

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)
        ),
        width=640,
        height=480,
    )
    
    # calc max_episode_length_s
    episode_length_s = 20.0
    # calc max_episode_length = math.ceil(self.max_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
    decimation = 2

    num_actions = 4
    num_channels = 3
    num_observations = num_channels * tiled_camera.height * tiled_camera.width
    num_observations_critic = 15

    num_states = 0
    debug_vis = True

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    angle_reward_scale = 1.0
    distance_to_goal_reward_scale = 15.0

class QuadcopterDepthCameraEnvCfg(QuadcopterRGBCameraEnvCfg):
    # depth camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=640,
        height=480,
    )

    # env
    num_channels = 1
    num_observations = num_channels * tiled_camera.height * tiled_camera.width

class QuadcopterCameraEnv(DirectRLEnv):
    cfg: QuadcopterRGBCameraEnvCfg | QuadcopterDepthCameraEnvCfg

    def __init__(
        self, cfg: QuadcopterRGBCameraEnvCfg | QuadcopterDepthCameraEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "angle_error"
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # debug drawing for lines connecting the frame
        import omni.isaac.debug_draw._debug_draw as omni_debug_draw
        self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "The Cartpole camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}"
            )

    def close(self):
        """Cleanup for the environment."""
        super().close()

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_observations_critic = self.cfg.num_observations_critic
        self.num_states = self.cfg.num_states

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        # self.single_observation_space["policy"] = gym.spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(self.cfg.tiled_camera.height, self.cfg.tiled_camera.width, self.cfg.num_channels),
        # )

        self.single_observation_space["policy"] = gym.spaces.Dict()
        self.single_observation_space["policy"]["observation"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.tiled_camera.height, self.cfg.tiled_camera.width, self.cfg.num_channels),)
        self.single_observation_space["policy"]["reward"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observations_critic,))
        self.single_observation_space["policy"]["last_action"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))
        
        if self.num_states > 0:
            self.single_observation_space["critic"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_observations_critic, ),
            )
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        # self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add articultion and sensors to scene
        self.scene.articulations["robot"] = self._robot
        # self.scene.sensors["tiled_camera"] = self._tiled_camera

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
        # 取最后一列施加力,即在Z轴施加力, actions[:, 0] 为力输出
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # 在X、Y、Z施加力矩, actions[:, 1：] 为力矩输出
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        # print("actions:", actions, "self._actions:", self._actions, self._thrust, self._moment)

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        # data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        # observations = {"policy": self._tiled_camera.data.output[data_type].clone()}
        # root_state_w [pos, quat, lin_vel, ang_vel]

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        yaw_robo = yaw_angle(self._robot.data.root_quat_w)
        yaw_desired = yaw_angle(self._desired_quat_w)
        angle_error = wrap_to_pi(yaw_robo - yaw_desired).unsqueeze(1)

        # pos_error, angle_error = compute_pose_error(self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w, self._desired_quat_w)

        # print("p1:", self._robot.data.root_pos_w, "q1:", self._robot.data.root_quat_w, "p2:", self._desired_pos_w, "q2:", self._desired_quat_w)
        # print("pos_error:", pos_error, "angle_error:", angle_error)

        # print("projected_gravity_b:", self._robot.data.projected_gravity_b)

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                angle_error,
            ],
            dim=-1,
        )
        # observations = {"policy": {"observation":self._tiled_camera.data.output[data_type].clone(), "reward": obs, "last_action": self._actions}, "critic": obs}
        # observations = {"policy": {"observation":self._tiled_camera.data.output[data_type].clone(), "reward": obs, "last_action": self._actions}}
        observations = {"policy": {"observation": torch.zeros((480,640,3), device="cuda:0"), "reward": obs, "last_action": self._actions}}
        
        # print("observations:", observations)
        # print("observations shape:", observations["policy"]["observation"].shape)
        # print("observations reward:", observations["policy"]["reward"].shape)
        # print("observations last_action:", observations["policy"]["last_action"].shape)
        return observations
    
    def vectors_to_axis_angles(self, vectors):  
        # 确保输入的vectors是一个形状为(N, 3)的张量，其中N是向量的数量  
        # 如果vectors不是单位向量，则先对其进行归一化  
        vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)  
        
        # 定义坐标轴的单位向量张量，形状为(1, 3)，可以与vectors进行广播  
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0)  
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).unsqueeze(0)  
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0)  
        
        # 计算vectors与每个坐标轴的点积，结果形状为(N,)  
        dot_x = torch.sum(vectors * x_axis, dim=1)  
        dot_y = torch.sum(vectors * y_axis, dim=1)  
        dot_z = torch.sum(vectors * z_axis, dim=1)  
        
        # 计算夹角（使用acos得到弧度值，然后转换为角度）  
        angle_x = torch.acos(dot_x) * 180 / torch.pi  
        angle_y = torch.acos(dot_y) * 180 / torch.pi  
        angle_z = torch.acos(dot_z) * 180 / torch.pi
        # 将结果打包到一个元组中返回  
        return angle_x, angle_y, angle_z 

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        # distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        distance_to_goal_mapped = torch.exp(-1.5 * distance_to_goal)
        
        # pos_error, angle_error = compute_pose_error(self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w, self._desired_quat_w)
        # print("pos_error:", pos_error, "angle_error:", angle_error, angle_error / 3.1415926 * 180.0)

        yaw_robo = yaw_angle(self._robot.data.root_quat_w)
        yaw_desired = yaw_angle(self._desired_quat_w)
        angle_error = wrap_to_pi(yaw_robo - yaw_desired)
        print("yaw_robo:", yaw_robo, "yaw_desired:", yaw_desired, angle_error / 3.1415926 * 180.0)

        # quadcopter_angle = euler_xyz_from_quat(self._robot.data.root_quat_w)
        # target_angle = euler_xyz_from_quat(self._desired_quat_w)
        
        # print("quadcopter_angle:", quadcopter_angle, "target_angle:", target_angle)

        # pos_error_angle = self.vectors_to_axis_angles(pos_error)
        # print("pos_error_angle:", pos_error_angle)

        # 10.0\left(\cos\left(x\ \right)\ -\ 0.99\right)
        # angle_error = torch.sum(10.0 * (torch.cos(angle_error) - 0.99), dim = 1)
        angle_error_mapped = 10.0 * (torch.cos(angle_error) - 0.99)
        
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "angle_error": distance_to_goal_mapped * angle_error_mapped * self.cfg.angle_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }

        # print("rewards:", rewards)

        # print("distance_to_goal:", distance_to_goal, rewards)
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) < 0.1
        time_out = torch.logical_or((self.episode_length_buf >= self.max_episode_length - 1), distance_to_goal)
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 3.0)
        # if died.any():
        #     print("died:", died, self._robot.data.root_pos_w, self._desired_pos_w, self.episode_length_buf, self.max_episode_length)
        return died, time_out

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
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        # x,y 分布
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # z 轴高度
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # self._desired_quat_w[env_ids,] = torch.randn_like(self._desired_quat_w[env_ids,])
        # norm = torch.norm(self._desired_quat_w[env_ids,], keepdim=True)  
        # # 归一化四元数  
        # self._desired_quat_w[env_ids,] = self._desired_quat_w[env_ids,] / norm 

        self._desired_quat_w[env_ids,] = random_orientation(num = 1, device=self.device)
        # w,x,y,z
        # self._desired_quat_w[env_ids,] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        # self._desired_quat_w[env_ids,] = torch.tensor([[0.707, 0.0, 0.0, 0.707]], device=self.device)
        # self._desired_quat_w[env_ids,] = torch.tensor([[0.866, 0.0, 0.5, 0.0]], device=self.device)
        # self._desired_quat_w[env_ids,] = torch.tensor([[0.612, -0.354, 0.354, -0.612]], device=self.device)
        
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

        
        # lines_colors = [[1.0, 1.0, 0.0, 1.0]] * source_pos.shape[0]
        # line_thicknesses = [5.0] * source_pos.shape[0]
        # self.draw_interface.draw_lines(source_pos.tolist(), target_pos.tolist(), lines_colors, line_thicknesses)

    def _debug_vis_callback(self, event):
        self.draw_interface.clear_lines()
        # get marker location
        # -- base state
        quadcopter_pos_w = self._robot.data.root_pos_w.clone()
        quadcopter_quat_w = self._robot.data.root_quat_w.clone()
        quadcopter_pos_w[:, 2] += 0.05
        # -- resolve the scales and quaternions
        # vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        # vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.base_target_goal_visualizer.visualize(self._desired_pos_w, self._desired_quat_w)
        self.base_quadcopter_visualizer.visualize(quadcopter_pos_w, quadcopter_quat_w)

        # 设置参数范围  
        t = np.linspace(0, 2 * np.pi, 1000)  # 参数t的范围从0到2π  
        # 定义8字形轨迹的参数方程  
        # 这里我们使用两个正弦函数，其中一个带有相位偏移来创建8字形状  
        x = 10.0 * np.sin(t)  
        y = 5.0 * np.sin(2 * t)  # 使用2t来加快y方向上的变化，并可以调整系数来改变形状  
        # z可以保持为常数或者也使用一个正弦函数  
        # z = np.zeros_like(t)  # z轴为0，形成平面图形  
        z = np.sin(2* t) + 1.5

        # colors = [(random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1), 1) for _ in range(N)]

        point_list_1 = np.column_stack((x, y, z))
        self.draw_interface.draw_lines_spline(point_list_1, (1, 0, 0, 1), 5, False)


################################################
# self.seed = 42
# Setting seed: 42
# Started to train
# Exact experiment name requested from command line: 2024-06-07_10-35-55
# seq_length: 4
# current training device: cuda:0
# conv_name: conv2d
# build mlp: 272384
# RunningMeanStd:  (1,)
# RunningMeanStd:  (480, 640, 3)
# Module omni.isaac.lab.utils.warp.kernels load on device 'cuda:0' took 4.50 ms
################################################
