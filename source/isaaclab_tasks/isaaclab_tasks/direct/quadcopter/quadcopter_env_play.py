# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import random_orientation, subtract_frame_transforms, yaw_angle, wrap_to_pi
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG


# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnvPlay, window_name: str = "IsaacLab"):
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
class QuadcopterEnvPlayCfg(DirectRLEnvCfg):
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # env
    episode_length_s = 10000.0
    decimation = 2
    num_actions = 4
    num_observations = 13
    num_states = 0
    debug_vis = True

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.1

    # lin_vel_reward_scale = -0.5
    # ang_vel_reward_scale = -1.0
    
    angle_reward_scale = 1.0
    distance_to_goal_reward_scale = 15.0
    goal_reached_reward_scale = 10.0
    effort_reward_scale = 1.0
    speed_reward_scale = 0.1    

class QuadcopterEnvPlay(DirectRLEnv):
    cfg: QuadcopterEnvPlayCfg

    def __init__(self, cfg: QuadcopterEnvPlayCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_point_index = torch.zeros(self.num_envs, device=self.device, dtype=int)
        self._is_goal_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._goal_distance = torch.ones(self.num_envs, device=self.device)
        self._path_max_count = torch.ones(self.num_envs, device=self.device)
        
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
                # "mean_speed"
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        print("robot_mass:", self._robot_mass, self._gravity_magnitude, self._robot_weight)
        
        # debug drawing for lines connecting the frame
        # import omni.isaac.debug_draw._debug_draw as omni_debug_draw
        # self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self._target_path_point, self._target_path_quat = self.tunnel_flight_path()
        # self._target_path_point, self._target_path_quat = self.gen_path()
        

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        # print("self._robot:", dir(self._robot))
        # print("self.cfg.robot:", dir(self.cfg.robot))
        self.cfg.robot.init_state.pos = (8.0, 8.0, 4.0)
 
        self.scene.articulations["robot"] = self._robot

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
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 3.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:] / 4.0

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def update_desired_pos(self, indices):
        # print("self._target_point_index[indices]:", self._target_point_index[indices])
        for idx in indices:
            if self._target_point_index[idx] < self._path_max_count[idx]:
                self._desired_pos_w[idx,] = self._target_path_point[self._target_point_index[idx]]
                # self._desired_pos_w[idx, :2] += self._terrain.env_origins[idx, :2]
                self._desired_quat_w[idx,] = self._target_path_quat[self._target_point_index[idx]]
                print("desired_pos_w:", self._desired_pos_w[idx,])
            else:
                self.episode_length_buf[idx] = self.max_episode_length

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        # desired_pos_b = self._desired_pos_w - self._robot.data.root_state_w[:, :3]
        # self._is_goal_reached = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) < 0.1

        # 使用.nonzero()获取True值的索引（对于一维tensor，我们只关心第一个维度）  
        indices = self._is_goal_reached.nonzero()[:, 0]
        # print("indices:", indices)
        self._target_point_index[indices] += 1
        self.update_desired_pos(indices)

        yaw_robo = yaw_angle(self._robot.data.root_quat_w)
        yaw_desired = yaw_angle(self._desired_quat_w)
        angle_error = wrap_to_pi(yaw_robo - yaw_desired).unsqueeze(1)

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                # pos_error,
                angle_error,
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        # print("observations:", observations)

        # print("imu lin:", self._robot.data.root_lin_vel_b, "ang_vel:", self._robot.data.root_ang_vel_b, "g:", self._robot.data.projected_gravity_b)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)        
        root_ang_vel_b_clone = self._robot.data.root_ang_vel_b.clone()
        root_ang_vel_b_clone[:, 2] = 0
        ang_vel = torch.sum(torch.square(root_ang_vel_b_clone), dim=1)

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)

        self._is_goal_reached = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) < 0.05
        # distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        distance_to_goal_mapped = torch.exp(-1.5 * distance_to_goal)

        yaw_robo = yaw_angle(self._robot.data.root_quat_w)
        yaw_desired = yaw_angle(self._desired_quat_w)
        angle_error = wrap_to_pi(yaw_robo - yaw_desired)
        

        # pos_error, angle_error = compute_pose_error(self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w, self._desired_quat_w)
        angle_error_mapped = 10.0 * (torch.cos(angle_error) - 0.99)
        reach_goal_mean_speed = self._goal_distance / (self.episode_length_buf * self.step_dt)

        # valid_mean_speed = (self._goal_distance - distance_to_goal) / (self.episode_length_buf * self.step_dt)
        
        # effort = torch.exp(-self._actions.sum(dim=1))
        effort = torch.exp(-torch.abs(self._actions.sum(dim=1)))

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "angle_error": distance_to_goal_mapped * angle_error_mapped * self.cfg.angle_reward_scale * self.step_dt,
            "goal_reached": self._is_goal_reached * reach_goal_mean_speed * self.cfg.goal_reached_reward_scale,
            "effort": effort * self.cfg.effort_reward_scale * self.step_dt,
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
        time_out = self.episode_length_buf < -1
        # died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        died = self._robot.data.root_pos_w[:, 2] < 0.1
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
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._target_point_index[env_ids] = 0
        self.update_desired_pos(env_ids)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        print("----------------------------------------------")
        print("reset!!!")
        print("----------------------------------------------")

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
        quadcopter_pos_w = self._robot.data.root_pos_w.clone()
        quadcopter_quat_w = self._robot.data.root_quat_w.clone()
        quadcopter_pos_w[:, 2] += 0.05
        # -- resolve the scales and quaternions
        # vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        # vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.base_target_goal_visualizer.visualize(self._desired_pos_w, self._desired_quat_w)
        self.base_quadcopter_visualizer.visualize(quadcopter_pos_w, quadcopter_quat_w)

    def vector_to_quaternion(self, direction_vector):
        
        # Step 3: Normalize the Direction Vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        
        # Step 4: Handle zero vector (no rotation)
        if np.allclose(direction_vector, np.zeros(3)):
            return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (no rotation)

        # Step 5: Rotation axis (normalized direction vector)
        rotation_axis = direction_vector

        # Step 6: Calculate the rotation angle (using arctan2 for quadrant handling)
        angle = np.arctan2(direction_vector[1], direction_vector[0])  # Angle around z-axis

        # Step 7: Compute the Quaternion
        quaternion = R.from_euler('z', angle, degrees=False).as_quat()
        
        quaternion = np.roll(quaternion, shift=1)
        
        return quaternion

    def gen_path(self):
        # 设置参数范围  
        t = np.linspace(0, 2 * np.pi, 60)  # 参数t的范围从0到2π  
        # 定义8字形轨迹的参数方程  
        # 这里我们使用两个正弦函数，其中一个带有相位偏移来创建8字形状  
        x = 10.0 * np.sin(t)  
        y = 5.0 * np.sin(2 * t)  # 使用2t来加快y方向上的变化，并可以调整系数来改变形状  
        # z可以保持为常数或者也使用一个正弦函数  
        # z = np.zeros_like(t)  # z轴为0，形成平面图形  
        z = np.sin(2* t) + 3.5
        # colors = [(random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1), 1) for _ in range(N)]
        path_point_list = np.column_stack((x, y, z))
        path_quaternion_list = []

        for idx, point in enumerate(path_point_list):
            vector = point - path_point_list[idx - 1]
            quaternion = self.vector_to_quaternion(vector)
            path_quaternion_list.append(quaternion)

        path_point_tensor = torch.tensor(path_point_list).to(self.device)
        path_quaternion_tensor = torch.tensor(np.array(path_quaternion_list)).to(self.device)

        # print(self._target_path_point[0], self._desired_pos_w[env_ids,], env_ids)
        # env_origins: tensor([1.2500, 0.0000], device='cuda:0')
        # env_origins: tensor([-1.2500,  0.0000], device='cuda:0')

        for idx in range(self.num_envs):
            self._path_max_count[idx] = len(path_point_list)
            # self.draw_interface.draw_lines_spline(path_point_list + self._terrain.env_origins[idx].cpu().numpy(), (1, 0, 0, 1), 1, False)
            print("env_origins:", self._terrain.env_origins[idx])

        return path_point_tensor.float(), path_quaternion_tensor.float()
    
    def interpolate_equidistant(self, point1, point2, interpolation_distance):
        """  
        对线段进行等距离插值 
        """  
        distance_between_point = np.linalg.norm(np.array(point1) - np.array(point2))

        # print("distance_between_point:", distance_between_point, point1, point2)

        if distance_between_point < interpolation_distance:
            return [point1]
        else:
            num_points = int(distance_between_point / interpolation_distance)
            # 初始化插值点列表  
            points = [point1]

            dx = (point2[0] - point1[0]) / num_points
            dy = (point2[1] - point1[1]) / num_points
            dz = (point2[2] - point1[2]) / num_points

            # print("dx:", dx, dy, dz, num_points)
            
            # 计算并添加插值点  
            for i in range(1, num_points):  
                x = point1[0] + i * dx  
                y = point1[1] + i * dy
                z = point1[2] + i * dz
                points.append([x, y, z]) 

            point_last = np.array([point1[0] + num_points * dx, point1[1] + num_points * dy, point1[2] + num_points * dz])
            distance_between_point = np.linalg.norm(point_last - point2)
            if distance_between_point > interpolation_distance:
                points.append(point_last) 
    
        return points
    
    def tunnel_flight_path(self):
        path_points = [(16.857454, 4.733797, 3), (23.785446, 0.7394848, 3), (37.238182, -5.8582616, 3), (50.593536, -10.577381, 3), (61.530663, -8.310559, 3), (66.24306, -6.116083, 3), (89.07483, 16.195843, 3), (86.178894, 15.035391, 3.0), (71.75105, 0.41696435, 3.0), (55.901173, -11.562683, 3.0), (40.07982, -6.569447, 3.0), (7.67947, 7.4372797, 3.0)]
        path_point_interpolate = []
        path_quaternion_list = []

        for i in range(len(path_points) - 1):
            new_points = self.interpolate_equidistant(path_points[i], path_points[i + 1], 2.0)
            path_point_interpolate += new_points
        
        path_point_list = np.array(path_point_interpolate)

        for idx, point in enumerate(path_point_list):
            vector = point - path_point_list[idx - 1]
            quaternion = self.vector_to_quaternion(vector)
            path_quaternion_list.append(quaternion)

        path_point_tensor = torch.tensor(path_point_list).to(self.device)
        path_quaternion_tensor = torch.tensor(np.array(path_quaternion_list)).to(self.device)
        self._path_max_count[0] = len(path_point_list)

        # self.draw_interface.draw_lines_spline(path_point_list, (1, 0, 0, 1), 1, False)
        return path_point_tensor.float(), path_quaternion_tensor.float()