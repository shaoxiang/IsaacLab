# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import random
import math

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, random_yaw_orientation
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns, RTXRayCasterCfg, RTXRayCaster
from isaaclab.sensors import RangeSensor, RangeSensorCfg
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.prims import XFormPrim

##
# enable animation graph
##
# import omni
# from isaacsim.core.utils.extensions import enable_extension
# manager = omni.kit.app.get_app().get_extension_manager()
# if not manager.is_extension_enabled("omni.anim.graph.bundle"):
#     enable_extension("omni.anim.graph.bundle")

##
# Pre-defined configs
##
from isaaclab_assets import SCOUT_MINI_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG, Person_MARKER_CFG  # isort: skip
import torch.nn.functional as F

class ScoutMiniAVEnvWindow(BaseEnvWindow):
    """Window manager for the ScoutMini environment."""

    def __init__(self, env: ScoutMiniAVEnv, window_name: str = "IsaacLab"):
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
class ScoutMiniAVEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 2
    action_space = 2
    state_space = 0
    debug_vis = True

    ui_window_class_type = ScoutMiniAVEnvWindow

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

    # camera
    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/base_link/Camera",
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.186, 0.0, 0.0625), rot=(0.5,0.5,-0.5,-0.5), convention="opengl"),
    #     data_types=["rgb", "depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     width=320,
    #     height=240,
    # )

    contact_sensor_base_link: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link", update_period=0.01, history_length=3, debug_vis=False
    )

    # lidar_scanner = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base_link",
    #     mesh_prim_paths=["/World/envs/env_.*/Obs_A"],
    #     # TODO skip rays cfg
    #     pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(-135.0, 135.0), horizontal_res=0.12),
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
    #     attach_yaw_only=True,
    #     debug_vis=True, # TODO flag for when video is recorded
    #     drift_range=(-0.05, 0.05), # TODO check drift range
    #     # TODO implement noise for distance readings
    #     max_distance=10.0
    # )

    # lidar = RTXRayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base_link/lidar",
    #     offset=RTXRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
    #     # spawn=sim_utils.LidarCfg(lidar_type=sim_utils.LidarCfg.LidarType.SLAMTEC_RPLIDAR_S2E)
    #     spawn=sim_utils.LidarCfg(lidar_type=sim_utils.LidarCfg.LidarType.SICK_TIM781),
    #     debug_vis=True,
    # )

    physx_lidar = RangeSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/Lidar",
        # update_period=0.025,  # Update rate of 40Hz
        # data_types=["point_cloud"],  # Assuming the LiDAR generates point cloud data
        horizontal_fov=360.0,  # Horizontal field of view of 270 degrees
        horizontal_resolution=0.4,  # Horizontal resolution of 0.5 degrees
        max_range=30.0,  # Maximum range of 30 meters
        min_range=0.020,  # Minimum range of 0.1 meters
        rotation_rate=0.0,  # Rotation rate of 0.0 radians per second
        offset=RangeSensorCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.01),  # Example position offset from the robot base
            rot=(1.0, 0.0, 0.0, 0.0),  # Example rotation offset; no rotation in this case
            convention="ros"  # Frame convention
        ),
        draw_lines=False,
        draw_points=True,
    )

    # SICK_TIM781
    # info: RTXRayCasterInfo(numChannels=811, numEchos=1, numReturnsPerScan=811, renderProductPath='/Render/OmniverseKit/HydraTextures/Replicator_01', ticksPerScan=1)

    # object collection
    obstacle_cfg: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "obs_A": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_A",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 2.0, 0.5)),
            ),
            "obs_B": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_B",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.7, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.5)),
            ),
            "obs_C": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_C",
                spawn=sim_utils.CylinderCfg(
                    radius=0.35,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, -2.0, 0.5)),
            ),
            "obs_D": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_D",
                spawn=sim_utils.CylinderCfg(
                    radius=0.25,
                    height=0.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, -2.0, 0.5)),
            ),
            "obs_E": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_E",
                spawn=sim_utils.CylinderCfg(
                    radius=0.35,
                    height=0.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, 0.0, 0.5)),
            ),
            "obs_F": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_F",
                spawn=sim_utils.CuboidCfg(
                    size=(0.7, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.0, 0.5)),
            ),
            "obs_G": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_G",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, 2.0, 0.5)),
            ),
            "obs_H": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_H",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1, 1.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.0, 0.5)),
            ),
        }
    )

    action_scale = 20.0

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.1
    distance_to_goal_reward_scale = 15.0
    collision_reward_scale = -35.0
    action_rate_reward_scale = -0.01
    goal_reached_reward_scale = 50.0
    delta_goal_reward_scale = 50.0
    # other
    lidar_points_num = 811
    max_perception_distance = 10.0

    # observation_space = {
    #     "robot-state": 15,
    #     "lidar": lidar_points_num,
    # }

    observation_space = {
        "robot-state": 15,
        "lidar": 900,
    }

    # observation_space = 15

class ScoutMiniAVEnv(DirectRLEnv):
    cfg: ScoutMiniAVEnvCfg

    def __init__(self, cfg: ScoutMiniAVEnvCfg, render_mode: str | None = None, **kwargs):
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
        self._dist_to_obstacles = torch.zeros(self.num_envs, device=self.device)
        # 设置碰撞距离阈值
        self._collision_threshold = 0.4

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
                "safety_static",
                "reverse_penalty",
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
        # self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # self.scene.sensors["tiled_camera"] = self._tiled_camera
        # self._lidar_scanner = RayCaster(self.cfg.lidar_scanner)
        # self.scene.sensors["lidar_scanner"] = self._lidar_scanner
        # add rtx lidar
        # self._lidar = RTXRayCaster(self.cfg.lidar)
        # self.scene.sensors["lidar"] = self._lidar

        # add physX lidar
        self._physx_lidar = RangeSensor(self.cfg.physx_lidar)
        self.scene.sensors["physx_lidar"] = self._physx_lidar

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # obstacles
        self._obstacles = RigidObjectCollection(self.cfg.obstacle_cfg)
        self.scene.rigid_object_collections["obstacles"] = self._obstacles

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
        self._update_dist_to_obstacles()
        self._previous_actions = self._actions.clone()
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        # RTX Lidar
        # distances = []
        # for key, value in self._lidar.data.items():
        #     lidar_data = value[0].distance 
        #     if lidar_data.shape == torch.Size([self.cfg.lidar_points_num]):
        #         # print("distance:", lidar_data.shape)
        #         distances.append(lidar_data)
        #     elif lidar_data.shape == torch.Size([0]):
        #         distances.append(torch.zeros((self.cfg.lidar_points_num)))
        #     else:
        #         distance = self.cfg.max_perception_distance * torch.ones(self.cfg.lidar_points_num)
        #         distance[value[0].index.long()] = lidar_data
        #         # print("distance:", distance, distance.shape)
        #         distances.append(distance)
        #         # print("index:", value[0].index.shape, value[0].index)
       
        # distance_combined_tensor = torch.stack(distances, dim=0)
        # distance_combined_tensor_process = torch.clamp(distance_combined_tensor, max=self.cfg.max_perception_distance)

        # PhysX Lidar
        # print("depth:", self._physx_lidar.data.output["depth"], self._physx_lidar.data.output["depth"].shape)
        # print("linear_depth:", self._physx_lidar.data.output["linear_depth"], self._physx_lidar.data.output["linear_depth"].shape)
        # print("azimuth:", self._physx_lidar.data.output["azimuth"], self._physx_lidar.data.output["azimuth"].shape)
        
        robot_state = torch.cat(
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

        # print("robot_state:", robot_state.shape, distance_combined_tensor_process.shape)

        # observations = {
        #     "policy": {
        #         "robot-state": robot_state,
        #         "lidar": distance_combined_tensor_process.to(self.device),
        #     }
        # }
        
        # observations = {"policy": robot_state}

        observations = {
            "policy": {
                "robot-state": robot_state,
                "lidar": self._physx_lidar.data.output["linear_depth"],
            }
        }

        return observations
    def _update_dist_to_obstacles(self):
        lidar_data = self._physx_lidar.data.output["linear_depth"]
        min_dist, _ = torch.min(lidar_data, dim=1)
        self._dist_to_obstacles = min_dist

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        self._dist_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(0.3* self._dist_to_goal)
        delta_goal_dist = self._previous_dist_to_goal - self._dist_to_goal
        self._previous_dist_to_goal = self._dist_to_goal.clone()

        net_contact_forces = self._contact_sensor_base_link.data.net_forces_w_history
        max_net_contact_forces, _ = torch.max(net_contact_forces.view(net_contact_forces.size(0), -1), dim=1)
        #self._collision = max_net_contact_forces > 1.5
        #print("max_net_contact_forces:", max_net_contact_forces)
        self._collision = self._dist_to_obstacles < self._collision_threshold
        #print(self._dist_to_obstacles)

        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        self._is_goal_reached = self._dist_to_goal < 0.7
        reset_goal_ids = self._is_goal_reached.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_goal_ids) > 0:
            self.reset_goal(reset_goal_ids)
        
        goal_position_reward = torch.zeros_like(self._is_goal_reached, dtype=torch.float)
        # 获取小车当前的xy坐标
        current_x = self._robot.data.root_pos_w[:, 0]
        current_y = self._robot.data.root_pos_w[:, 1]
        # 判断是否到达目标点且xy坐标绝对值均大于2.0
        #valid_goal_position = self._is_goal_reached & (torch.abs(current_x) > 1.75) & (torch.abs(current_y) > 2.0）
        valid_goal_position = self._is_goal_reached & ~((torch.abs(current_x) < 2.00) & (torch.abs(current_y) < 2.0))
        goal_position_reward[valid_goal_position] = 30
        lidar_data = self._physx_lidar.data.output["linear_depth"]
        min_dist, _ = torch.min(lidar_data, dim=1)
        # 计算剩余空间（从障碍物到LiDAR最大探测距离）
        remaining_space = 1.65 - min_dist
        # 对剩余空间取自然对数，并计算平均值作为奖励
        reward_safety_static = torch.pow(torch.tensor(10.0), remaining_space.clamp(min=1e-6, max=1.65))-1
        #print(reward_safety_static)
        #rewards["safety_static"] = reward_safety_static * 10
        lin_vel_b = self._robot.data.root_lin_vel_b
        reverse_penalty = (lin_vel_b[:, 0] < 0).float() * (-3.0)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "collision": self._collision.float() * self.cfg.collision_reward_scale,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "goal_reached": self._is_goal_reached.float() * self.cfg.goal_reached_reward_scale + goal_position_reward,
            "delta_goal_dist": delta_goal_dist * self.cfg.delta_goal_reward_scale * self.step_dt,
            "safety_static" : reward_safety_static * self.step_dt,
            "reverse_penalty" : reverse_penalty * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        #print(self._collision.float() * self.cfg.collision_reward_scale)
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
        
        min_square_side = 1.5
        max_square_side = 2.5
        # 生成目标点并检查是否在禁止区域内
        for _ in range(100):  # 尝试最多100次
            phi = random.uniform(0, math.pi)
            theta = random.uniform(0, 2 * math.pi)
            radius = random.uniform(3.0, 6.0)
            ball_x = radius * math.sin(phi) * math.cos(theta)
            ball_y = radius * math.sin(phi) * math.sin(theta)
            desired_pos = torch.tensor([ball_x, ball_y], device=self.device)

            # 检查是否在禁止区域内
            # 禁止区域是以（0，0，0）为中心，平行于x、y轴，边长在1.5到2.5之间的正方形
            x_abs = torch.abs(torch.tensor(ball_x, device=self.device))
            y_abs = torch.abs(torch.tensor(ball_y, device=self.device))
          

            in_big_square = (x_abs <= max_square_side) & (y_abs <= max_square_side)
            out_small_square = ~((x_abs < min_square_side) & (y_abs < min_square_side))
            in_forbidden_zone = in_big_square & out_small_square

            # 如果所有点都不在禁止区域内
            if not torch.any(in_forbidden_zone):
                # 如果不在禁止区域内，设置为目标点
                self._desired_pos_w[env_ids, 0] = ball_x
                self._desired_pos_w[env_ids, 1] = ball_y
                self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
                self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])
                #self._previous_dist_to_goal[env_ids] = torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1)
                #print(self._desired_pos_w)
                break
        else:
            # 如果经过100次尝试仍未找到有效点，可能需要重新考虑生成逻辑
            print("Warning: Failed to find a valid goal position after 100 attempts.")
        """phi = random.uniform(0, math.pi)  
        theta = random.uniform(0, 2*torch.pi)
        radius = random.uniform(3.0, 6.0)
        ball_x = radius * math.sin(phi) * math.cos(theta)  
        ball_y = radius * math.sin(phi) * math.sin(theta)  
        self._desired_pos_w[env_ids, 0] = ball_x
        self._desired_pos_w[env_ids, 1] = ball_y
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])
        self._previous_dist_to_goal[env_ids] = torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1)"""

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
        
        #self._obstacles.reset(env_ids)

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
        """min_square_side = 1.5
        max_square_side = 2.5
        # 生成目标点并检查是否在禁止区域内
        for _ in range(100):  # 尝试最多100次
            phi = random.uniform(0, math.pi)
            theta = random.uniform(0, 2 * math.pi)
            radius = random.uniform(3.0, 6.0)
            ball_x = radius * math.sin(phi) * math.cos(theta)
            ball_y = radius * math.sin(phi) * math.sin(theta)
            desired_pos = torch.tensor([ball_x, ball_y], device=self.device)

            # 检查是否在禁止区域内
            # 禁止区域是以（0，0，0）为中心，平行于x、y轴，边长在1.5到2.5之间的正方形
          

            # 检查目标点是否在禁止区域内
            if not (min_square_side <= abs(desired_pos[0]) <= max_square_side and min_square_side <= abs(desired_pos[1]) <= max_square_side):
                # 如果不在禁止区域内，设置为目标点
                self._desired_pos_w[env_ids, 0] = ball_x
                self._desired_pos_w[env_ids, 1] = ball_y
                self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
                self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])
                #self._previous_dist_to_goal[env_ids] = torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1)
                #print(self._desired_pos_w)
                break
        else:
            # 如果经过100次尝试仍未找到有效点，可能需要重新考虑生成逻辑
            print("Warning: Failed to find a valid goal position after 100 attempts.")"""
        """phi = random.uniform(0, math.pi)  
        theta = random.uniform(0, 2*torch.pi)
        radius = random.uniform(3.0, 6.0)
        ball_x = radius * math.sin(phi) * math.cos(theta)  
        ball_y = radius * math.sin(phi) * math.sin(theta)  
        self._desired_pos_w[env_ids, 0] = ball_x
        self._desired_pos_w[env_ids, 1] = ball_y
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])"""
        # Reset robot state
        self.reset_goal(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        #self._previous_dist_to_goal[env_ids] = torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1)
        
        

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
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

@configclass
class ScoutMiniAVYoloEnvCfg(ScoutMiniAVEnvCfg):
    # env
    episode_length_s = 30.0
    action_space = 2
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing = 20.0, replicate_physics=True)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.186, 0.0, 0.0625), rot=(0.5,0.5,-0.5,-0.5), convention="opengl"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=256,
        height=512,
    )

    physx_lidar = RangeSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/Lidar",
        # update_period=0.025,  # Update rate of 40Hz
        # data_types=["point_cloud"],  # Assuming the LiDAR generates point cloud data
        horizontal_fov=360.0,  # Horizontal field of view of 270 degrees
        horizontal_resolution=1.0,  # Horizontal resolution of 0.5 degrees
        max_range=5.0,  # Maximum range of 30 meters
        min_range=0.02,  # Minimum range of 0.1 meters
        rotation_rate = 0.0,  # Rotation rate of 0.0 radians per second
        offset=RangeSensorCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1),  # Example position offset from the robot base
            rot=(1.0, 0.0, 0.0, 0.0),  # Example rotation offset; no rotation in this case
            convention="ros"  # Frame convention
        ),
        draw_lines=False,
        draw_points=False,
    )

    # SICK_TIM781
    # info: RTXRayCasterInfo(numChannels=811, numEchos=1, numReturnsPerScan=811, renderProductPath='/Render/OmniverseKit/HydraTextures/Replicator_01', ticksPerScan=1)

    # object collection
    obstacle_cfg: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "obs_A": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_A",
                spawn=sim_utils.SphereCfg(
                    radius=0.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 5.0, 0.5)),
            ),
            "obs_B": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_B",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 0.5)),
            ),
            "obs_C": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_C",
                spawn=sim_utils.ConeCfg(
                    radius=0.3,
                    height=0.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, -5.0, 0.5)),
            ),
            "obs_D": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_D",
                spawn=sim_utils.CylinderCfg(
                    radius=0.35,
                    height=0.8,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.0, -5.0, 0.5)),
            ),
            "obs_E": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_E",
                spawn=sim_utils.CapsuleCfg(
                    radius=0.35,
                    height=0.8,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.0, 0.0, 0.5)),
            ),
            "obs_F": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Obs_F",
                spawn=sim_utils.SphereCfg(
                    radius=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -5.0, 0.5)),
            ),
        }
    )

    # character_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/person",
    #     init_state=RigidObjectCfg.InitialStateCfg(),
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/People/Characters/biped_demo/biped_walk.usd",
    #         scale=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #         ),
    #     ),
    # )

    # character_cfg: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/person",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/People/Characters/biped_demo/biped_walk.usd",
    #         scale=(1.0, 1.0, 1.0),
    #     ),
    # )

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.1
    distance_to_goal_reward_scale = 10.0
    collision_reward_scale = -1.0
    action_rate_reward_scale = -0.01
    # goal_reached_reward_scale = 20.0
    delta_goal_reward_scale = 5.0
    near_obstacle_reward_scale = 10.0
    yolo_reward_scale = 50.0
    # other
    max_perception_distance = 10.0
    min_safety_distance = 0.5
    max_person_num = 2

    observation_space = {
        "robot-state": 5 * max_person_num + action_space, #15,
        "lidar": 360, # 900
        "privileged-state": 9,
    }

class ScoutMiniAVYoloEnv(DirectRLEnv):
    cfg: ScoutMiniAVYoloEnvCfg

    def __init__(self, cfg: ScoutMiniAVYoloEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the uav
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_rot_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._collision = torch.zeros(self.num_envs, device=self.device)
        self._is_goal_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._previous_dist_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._yolo_rewards = torch.zeros(self.num_envs, device=self.device)

        # Load yolo model
        from ultralytics import YOLO
        self.yolo_model = YOLO("./source/third_part/YOLO/yolo11n.pt")

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
                "obstacle",
                "yolo_rewards",
                "reverse_penalty",
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
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # self._lidar_scanner = RayCaster(self.cfg.lidar_scanner)
        # self.scene.sensors["lidar_scanner"] = self._lidar_scanner
        # add rtx lidar
        # self._lidar = RTXRayCaster(self.cfg.lidar)
        # self.scene.sensors["lidar"] = self._lidar
        # add physX lidar
        self._physx_lidar = RangeSensor(self.cfg.physx_lidar)
        self.scene.sensors["physx_lidar"] = self._physx_lidar

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # obstacles
        self._obstacles = RigidObjectCollection(self.cfg.obstacle_cfg)
        self.scene.rigid_object_collections["obstacles"] = self._obstacles
        # add Character rigidbody
        # self._character = RigidObject(self.cfg.character_cfg)
        # self.scene.rigid_objects["character"] = self._character

        # self.cfg.character_cfg.spawn.func(
        #     self.cfg.character_cfg.prim_path,
        #     self.cfg.character_cfg.spawn,
        #     translation=self.cfg.character_cfg.init_state.pos,
        #     orientation=self.cfg.character_cfg.init_state.rot,
        # )
        # self._character = XFormPrim(self.cfg.character_cfg.prim_path, reset_xform_properties=False)
        # self.scene.extras["character"] = self._character

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = self.cfg.action_scale * actions.clone().clamp(-1.0, 1.0)
        
    def _apply_action(self):
        wheel_actions = torch.cat((self._actions, self._actions), dim=1)
        # print("wheel_actions:", self._actions, wheel_actions)
        self._robot.set_joint_velocity_target(wheel_actions, self._wheels_dof_idx)

    def yolo_results_filter(self, yolo_results, max_person = 3, choose_device = 'cuda'):
        new_results = torch.zeros(self.num_envs, max_person, 5, device = choose_device)
        self._yolo_rewards = torch.zeros(self.num_envs, device = choose_device)
                
        for i, yolo_result in enumerate(yolo_results):
            person_num = 0
            tmp_num = 0
            tmp_results = torch.zeros(max_person, 5, device = choose_device)
            box_size = torch.tensor(0.0)
            # 将带有bounding box 和 类别组合成状态量，优先放入类别为 "person" 的数据，不够再用别的类别凑，凑满 max_person 个数据
            for index, box in enumerate(yolo_result.boxes.xyxyn.cuda()):
                # print("box:", box, yolo_result.boxes.cls[index])
                one_result = torch.cat((box, (yolo_result.boxes.cls[index] + 1).unsqueeze(0)), dim=0).to(choose_device)
                # yolo中类别'person'序号为 0 
                if yolo_result.boxes.cls[index] == 0.:
                    x1, y1 , x2, y2 = box.to("cpu")
                    box_size += (x2 - x1) * (y2 - y1) * 10.0
                    new_results[i][person_num] = one_result
                    person_num += 1
                else:
                    if tmp_num < max_person:
                        tmp_results[tmp_num] = one_result
                        tmp_num += 1

                if person_num >= max_person:
                    break

            if person_num < max_person:
                for tmp_index, tmp_result in enumerate(tmp_results):
                    if person_num + tmp_index >= max_person:
                        break
                    new_results[i][person_num + tmp_index] = tmp_result

            if person_num > 0:
                self._yolo_rewards[i] = box_size.to(device = choose_device)

        # print(new_results)
        return new_results

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        image_data = self._tiled_camera.data.output[data_type].clone()
        # print("image_data:", image_data.shape, image_data.dtype)
        image_data_permute = image_data.permute(0, 3, 1, 2) / 255.0
        # print("image_data_permute:", image_data_permute.shape, image_data_permute.dtype)
        results = self.yolo_model(image_data_permute, stream=True, verbose = False)
        # print("results:", results)
        # print("results boxes:", results[0].boxes, results[0].boxes.cls)
        # print("results boxes len:", len(results[0].boxes), len(results[0].boxes.cls))
        yolo_obs = self.yolo_results_filter(results, max_person = self.cfg.max_person_num)
        yolo_obs_view = yolo_obs.view(self.num_envs, -1)
        person_obs = torch.cat([yolo_obs_view, self._actions.clone()], dim = -1)
        # print("yolo_obs_view:", yolo_obs_view.shape, "self._actions:", self._actions.shape, "person_obs:", person_obs.shape, "self._yolo_rewards:", self._yolo_rewards.shape)

        privileged_state = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
            ],
            dim=-1,
        )

        observations = {
            "policy": {
                "robot-state": person_obs,
                "lidar": self._physx_lidar.data.output["linear_depth"].clone(),
                "privileged-state": privileged_state,
            }
        }
        # print("observations:", observations["policy"]["lidar"].shape)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        net_contact_forces = self._contact_sensor_base_link.data.net_forces_w_history
        max_net_contact_forces, _ = torch.max(net_contact_forces.view(net_contact_forces.size(0), -1), dim=1)
        self._collision = max_net_contact_forces > 3.0
        # print("max_net_contact_forces:", max_net_contact_forces)

        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        reset_goal_ids = self._is_goal_reached.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_goal_ids) > 0:
            self.reset_goal(reset_goal_ids)

        # print("yolo_rewards:", self._yolo_rewards)
        reverse_penalty = (self._robot.data.root_lin_vel_b[:, 0] < 0).float() * (-1.0)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "collision": self._collision.float() * self.cfg.collision_reward_scale,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "obstacle": -F.relu(1.0 - self.min_dis_to_obstacle()) * self.cfg.near_obstacle_reward_scale * self.step_dt,
            "yolo_rewards": self._yolo_rewards * self.cfg.yolo_reward_scale * self.step_dt,
            "reverse_penalty": reverse_penalty * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # time_out = torch.logical_or(self.episode_length_buf >= self.max_episode_length - 1, self._is_goal_reached)
        safe_check = self.min_dis_to_obstacle() < self.cfg.min_safety_distance
        died = torch.logical_or(self._collision, safe_check)
        # died = self._robot.data.root_pos_w[:, 2] > 1.0
        # print("max_net_contact_forces:", max_net_contact_forces)
        # died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.5, self._robot.data.root_pos_w[:, 2] > 15.0)
        return died, time_out
    
    def min_dis_to_obstacle(self):
        lidar_data = self._physx_lidar.data.output["linear_depth"]
        min_dist, _ = torch.min(lidar_data, dim = 1)
        # print("min_dist:", min_dist)
        return min_dist

    def generate_points_in_annulus(self, r1, r2, num_points):
        """
            生成圆环（内径r1，外径r2）内的均匀随机点
            参数：
                r1 (float): 内半径
                r2 (float): 外半径
                num_points (int): 生成点数
            返回：
                torch.Tensor: 形状为(num_points, 2)的二维坐标张量
        """
        theta = torch.rand(num_points) * 2 * math.pi
        u = torch.rand(num_points)
        r = torch.sqrt(r1**2 + (r2**2 - r1**2) * u)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack((x, y), dim=1)

    def reset_goal(self, env_ids: torch.Tensor | None):
        gen_points = self.generate_points_in_annulus(7.0, 15.0, env_ids.size(0))
        # print("gen_points:", gen_points)
        self._desired_pos_w[env_ids, :2] = gen_points.to(self.device)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2])
        self._desired_rot_w[env_ids] = random_yaw_orientation(len(env_ids), self.device)
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
        self._yolo_rewards[env_ids] = 0
        self._is_goal_reached[env_ids] &= False
        # Sample new commands
        self.reset_goal(env_ids)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = Person_MARKER_CFG.copy()
                # marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.markers["people"].size = (1.0, 1.0, 1.0)
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
        self.goal_pos_visualizer.visualize(self._desired_pos_w, self._desired_rot_w)