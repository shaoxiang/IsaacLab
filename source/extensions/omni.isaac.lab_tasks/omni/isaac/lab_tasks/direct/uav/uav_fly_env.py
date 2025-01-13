# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from scipy.spatial.transform import Rotation as R

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import UAV_CFG  # isort: skip
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

import numpy as np

# import omni
# ext_manager = omni.kit.app.get_app().get_extension_manager()
# ext_manager.set_extension_enabled_immediate("omni.physx.forcefields", True)

class UAVEnvWindow(BaseEnvWindow):
    """Window manager for the UAV environment."""

    def __init__(self, env: UAVEnv, window_name: str = "IsaacLab"):
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
class UAVFlyEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 500.0
    decimation = 2
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = UAVEnvWindow

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

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.024216989562340085, -0.07150677787090418, 0.00604856209541946), rot=(0.70711, 0.0, 0.70711, 0.0), convention="ros"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 50.0)
        ),
        width=480,
        height=320,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = UAV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 10.0
    moment_scale = 0.05

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0

class UAVFlyEnv(DirectRLEnv):
    cfg: UAVFlyEnvCfg

    def __init__(self, cfg: UAVFlyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the uav
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
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
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        self._target_path_point, self._target_path_quat = self.flight_path()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.cfg.robot.init_state.pos = self.flight_start_point()
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
        # add camera
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 1] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        indices = self._is_goal_reached.nonzero()[:, 0]
        self._target_point_index[indices] += 1
        self.update_desired_pos(indices)

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )

        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        image_data = self._tiled_camera.data.output[data_type].clone()
        # print("image_data:", image_data.shape, image_data.dtype)
        # image_data_permute = image_data.permute(0, 3, 1, 2) / 255.0

        observations = {"policy": obs, "image": image_data}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        self._is_goal_reached = distance_to_goal < 0.1
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.01, self._robot.data.root_pos_w[:, 2] > 5000.0)
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
        # Sample new commands
        self._target_point_index[env_ids] = 0
        self._is_goal_reached[env_ids] = False
        self.update_desired_pos(env_ids)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

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

    def update_desired_pos(self, indices):
        # print("self._target_point_index[indices]:", self._target_point_index[indices])
        for idx in indices:
            if self._target_point_index[idx] < self._path_max_count[idx]:
                self._desired_pos_w[idx,] = self._target_path_point[self._target_point_index[idx]]
                # self._desired_pos_w[idx, :2] += self._terrain.env_origins[idx, :2]
                self._desired_quat_w[idx,] = self._target_path_quat[self._target_point_index[idx]]
                # print("desired_pos_w:", self._desired_pos_w[idx,])
            else:
                self.episode_length_buf[idx] = self.max_episode_length

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

    def flight_start_point(self):
        return (16.0, 5.0, 3.0)

    def flight_path(self):
        path_points = [(16.857454, 4.733797, 3), (23.785446, 0.7394848, 3), (37.238182, -5.8582616, 3), (50.593536, -10.577381, 3), (61.530663, -8.310559, 3), (66.24306, -6.116083, 3), (89.07483, 16.195843, 3), (86.178894, 15.035391, 3.0), (71.75105, 0.41696435, 3.0), (55.901173, -11.562683, 3.0), (40.07982, -6.569447, 3.0), (7.67947, 7.4372797, 3.0)]
        # path_points = [(864.695, -760.905, 210.696), (865.452, -760.83, 210.696), (-575.197, -692.121, 210.696), (-641.875, -738.985, 210.696), (-708.554, -785.85, 210.696), (-695.006, -780.859, 210.696), (-701.77, -855.017, 210.696), (-708.533, -929.175, 210.696), (-499.534, -2017.27, 194.696), (-596.328, -1938, 194.696), (-693.123, -1858.73, 194.696), (-918.9, -1258.65, 253.696), (-1008.93, -1373.29, 253.696), (-1098.95, -1487.93, 253.696), (-1616.53, -2284.36, 195.696), (-1616.53, -2284.36, 195.696), (-1616.53, -2284.36, 195.696), (-1635.16, -2355.92, 189.696), (-1711.27, -2345.5, 210.696), (-1795.7, -2333.94, 233.993), (-3186.2, -2401.72, 210.696), (-3926.79, -2420.55, 210.696), (-4199.21, -2427.47, 210.696), (-4383.96, -2450.87, 210.696), (-4383.96, -2450.87, 210.696), (-4383.96, -2450.87, 210.696), (-4569.43, -2430.26, 247.696), (-4566.76, -2433.27, 235.696), (-4564.08, -2436.29, 223.696), (-4569.06, -2462.48, 186.696), (-4569.06, -2462.48, 186.696), (-4569.06, -2462.48, 186.696), (-4789.52, -2446.14, 217.385), (-4789.52, -2446.14, 217.385), (-4789.52, -2446.14, 217.385), (-4733.74, -2247.81, 246.982), (-4733.74, -2247.81, 246.982), (-4733.74, -2247.81, 246.982), (-4785.11, -2391.24, 247.97), (-4785.11, -2391.24, 247.97), (-4785.11, -2391.24, 247.97), (-4540.7, -2424.07, 210.696), (-4540.7, -2424.07, 210.696), (-4540.7, -2424.07, 210.696), (-1575.95, -2362.06, 210.696), (-1575.95, -2362.06, 210.696)]
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
