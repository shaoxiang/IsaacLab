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
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from omni.isaac.lab.utils.math import random_orientation, subtract_frame_transforms, yaw_angle, wrap_to_pi
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip
from omni.physx import get_physx_interface, get_physx_simulation_interface
import omni.isaac.core.utils.stage as stage_utils
from pxr import PhysxSchema, Usd, UsdPhysics

from source.third_part.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from torchvision.models import resnet18

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterVisionDepthEnv2, window_name: str = "IsaacLab"):
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
class QuadcopterVisionDepthEnvCfg2(DirectRLEnvCfg):
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

    # add cube
    # cube_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/Cube.*/Cube",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.2, 0.2, 0.2),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    # )

    # # spawn a green cone with colliders and rigid body
    # cone_cfg: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/Cone.*/Cone",
    #     spawn=sim_utils.ConeCfg(
    #         radius=0.1,
    #         height=0.2,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    # )

    # cube_spawn_cfg = sim_utils.CuboidCfg(
    #         size=(0.2, 0.2, 0.2),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
    #     )
    
    # # spawn a green cone with colliders and rigid body
    # cone_spawn_cfg = sim_utils.ConeCfg(
    #         radius=0.1,
    #         height=0.2,
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #     )

    # scene 
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=5.0, replicate_physics=True) # 65536
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True) 
    # 8192
    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)
        ),
        width=224,
        height=224,
    )

    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body", update_period=0.01, history_length=3, debug_vis=False
    )
    
    # env
    episode_length_s = 5.0
    decimation = 2
    num_actions = 4
    num_observations = 256 + 13 
    num_states = 0
    debug_vis = True

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    angle_reward_scale = 1.0
    distance_to_goal_reward_scale = 15.0
    goal_reached_reward_scale = 10.0
    collision_occurred_reward_scale = -2.0
    effort_reward_scale = 0.1
    speed_reward_scale = 0.1    

class QuadcopterVisionDepthEnv2(DirectRLEnv):
    cfg: QuadcopterVisionDepthEnvCfg2

    def __init__(
        self, cfg: QuadcopterVisionDepthEnvCfg2, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._is_goal_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._is_collision_occurred = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._goal_distance = torch.ones(self.num_envs, device=self.device)
        
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
                # "mean_speed",
                "collision_occurred",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # init model
        perception_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vits' # or 'vitl', 'vitb', 'vitg'
        self.perception_model = DepthAnythingV2(**perception_model_configs[encoder])
        self.perception_model.load_state_dict(torch.load(f'source/third_part/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.perception_model = self.perception_model.to('cuda').eval()
        # self.encode_model = resnet18(pretrained=True).to('cuda')
        # self.encode_model.fc = torch.nn.Identity()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def add_obstacles(self):
        cube_idxs = np.random.randint(0, self.num_envs, size=(self._cube_num))
        cone_idxs = np.random.randint(0, self.num_envs, size=(self._cone_num))
        
        for i, index in enumerate(cube_idxs):
            # print(f"/World/Cube{i}", self._terrain.env_origins[index])
            prim_utils.create_prim(f"/World/Cube{i}", "Xform", translation=self._terrain.env_origins[index].cpu().tolist() + np.random.uniform(low=-1.0, high=1.0, size=(3)))
            
        for i, index in enumerate(cone_idxs):
            # print(f"/World/Cone{i}", self._terrain.env_origins[index])
            prim_utils.create_prim(f"/World/Cone{i}", "Xform", translation=self._terrain.env_origins[index].cpu().tolist() + np.random.uniform(low=-1.0, high=1.0, size=(3)))

        # self.cube_object = RigidObject(cfg=self.cfg.cube_cfg)
        # self.cone_object = RigidObject(cfg=self.cfg.cone_cfg)

        self.cfg.cube_spawn_cfg.func(
            "/World/Cube.*/Cube", self.cfg.cube_spawn_cfg, translation=(0.0, 0.0, 1.0), orientation=(0.5, 0.0, 0.5, 0.0))
        
        self.cfg.cone_spawn_cfg.func(
            "/World/Cone.*/Cone", self.cfg.cone_spawn_cfg, translation=(0.0, 0.0, 1.0), orientation=(0.5, 0.0, 0.5, 0.0))

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add camera
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self._cube_num = 5 * self.scene.cfg.num_envs
        self._cone_num = 5 * self.scene.cfg.num_envs
        # add contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        # add obstacles
        # self.add_obstacles()

        # apply contact report
        # spherePath = "/World/envs/env_0/Robot/body/collision"
        # stage = stage_utils.get_current_stage()
        # spherePrim = stage.GetPrimAtPath(spherePath)
        # print("spherePrim:", spherePrim)
        # contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(spherePrim)
        # contactReportAPI.CreateThresholdAttr().Set(200000)
        # self.contact_check()
        
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        
    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

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

    def depth_to_rgb(self, depth):  
        min_depth, _ = depth.min(dim=-1).min(dim=-1)
        max_depth, _ = depth.max(dim=-1).max(dim=-1)
        print("min_depth:", min_depth, "max_depth:", max_depth)
        # 归一化深度值到[0, 1]  
        normalized_depth = (depth - min_depth) / (max_depth - min_depth)  
        # 将归一化的深度值映射到[0, 255]的整数范围，用于RGB  
        scaled_depth = normalized_depth * 255 
        rgb_image = torch.stack([scaled_depth, scaled_depth, scaled_depth], dim=-1).to(torch.uint8)  
        return rgb_image 
    
    def depth_to_vector(self, depth):
        # 遍历每个颜色通道 
        slice_size = 16
        block_size = depth.size(-1) // slice_size
        output_tensor = torch.zeros(depth.size(0), slice_size * slice_size).to('cuda') 
        # 计算每个小块在原始图像中的起始坐标  
        for i in range(slice_size):  
            for j in range(slice_size):  
                y_start = i * block_size  
                y_end = y_start + block_size  
                x_start = j * block_size  
                x_end = x_start + block_size
                # 从原始tensor中切出当前小块  
                block = depth[:, y_start:y_end, x_start:x_end]
                # 计算新tensor中的索引位置  
                # 注意：由于我们要将16x16个小块平铺成一个维度，所以需要计算当前小块的索引  
                index = i * 16 + j
                # 将切出的小块最小值放入新tensor的对应位置  
                block_min, _  = block.min(dim=-1)
                block_min, _  = block_min.min(dim=-1)
                # print("block_min:", block_min.shape)
                output_tensor[:, index] = block_min

        return output_tensor
  
    def image_depth_encode(self):
        rgb_img = self._tiled_camera.data.output["rgb"].clone()
        # print("rgb_img shape:", rgb_img.shape) # ([N, 224, 224, 3])
        rgb_img = rgb_img.permute(0, 3, 1, 2)
        # print("rgb_img shape:", rgb_img.shape) # ([N, 3, 224, 224])
        depth_img = self.perception_model.forward(rgb_img)
        # print("depth_img:", depth_img.shape) # ([N, 224, 224])
        # depth_out = self.encode_model.forward(depth_img)
        depth_encode = self.depth_to_vector(depth_img)
        # print("depth_encode:", depth_encode.shape)
        return depth_encode

    def _get_observations(self) -> dict:
        
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        yaw_robo = yaw_angle(self._robot.data.root_quat_w)
        yaw_desired = yaw_angle(self._desired_quat_w)
        angle_error = wrap_to_pi(yaw_robo - yaw_desired).unsqueeze(1)

        depth_encode = self.image_depth_encode()
        obs = torch.cat(
            [
                depth_encode,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
                angle_error,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        # print("observations:", observations)
        return observations
        
    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        self._is_goal_reached = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) < 0.05
        
        distance_to_goal_exp = torch.exp(-1.5 * distance_to_goal)
        distance_to_goal_mapped = (1.0 / (0.1 + distance_to_goal / self._goal_distance)) - 0.9

        yaw_robo = yaw_angle(self._robot.data.root_quat_w)
        yaw_desired = yaw_angle(self._desired_quat_w)
        angle_error = wrap_to_pi(yaw_robo - yaw_desired)
        
        # pos_error, angle_error = compute_pose_error(self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w, self._desired_quat_w)
        angle_error_mapped = 10.0 * (torch.cos(angle_error) - 0.99)
        reach_goal_mean_speed = self._goal_distance / (self.episode_length_buf * self.step_dt)
        # valid_mean_speed = (self._goal_distance - distance_to_goal) / (self.episode_length_buf * self.step_dt)
        effort = torch.exp(-torch.abs(self._actions.sum(dim=1)))

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        max_net_contact_forces, _ = torch.max(net_contact_forces.view(net_contact_forces.size(0), -1), dim=1)
        self._is_collision_occurred = max_net_contact_forces > 0.05

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "angle_error": distance_to_goal_exp * angle_error_mapped * self.cfg.angle_reward_scale * self.step_dt,
            "goal_reached": self._is_goal_reached * reach_goal_mean_speed * self.cfg.goal_reached_reward_scale,
            "effort": effort * self.cfg.effort_reward_scale * self.step_dt,
            "collision_occurred": self._is_collision_occurred * max_net_contact_forces * self.cfg.collision_occurred_reward_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # print("valid_mean_speed:", valid_mean_speed, "_goal_distance:", self._goal_distance, "reach_goal_mean_speed:", reach_goal_mean_speed, self.max_episode_length, self.episode_length_buf, rewards)
        # print("rewards:", reward, rewards)

        # print("net_contact_forces: ", net_contact_forces, self._contact_sensor.data.net_forces_w)
        # print("Received max contact force of: ", max_net_contact_forces, self._is_collision_occurred)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = torch.logical_or((self.episode_length_buf >= self.max_episode_length - 1), self._is_goal_reached)
        died = torch.logical_or(torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 3.0), self._is_collision_occurred)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        final_distance_to_goal_scale = (final_distance_to_goal / self._goal_distance[env_ids]).mean()

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
        extras["Metrics/final_distance_to_goal_scale"] = final_distance_to_goal_scale.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        self._desired_quat_w[env_ids,] = random_orientation(num = 1, device=self.device)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._is_goal_reached[env_ids] &= False 
        self._is_collision_occurred[env_ids] &= False
        self._goal_distance[env_ids] = torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids])

        # # reset root state
        # root_state = self.cone_object.data.default_root_state.clone()
        # # sample a random position on a cylinder around the origins
        # root_state[:, :3] += origins
        # root_state[:, :3] += math_utils.sample_cylinder(
        #     radius=0.1, h_range=(0.25, 0.5), size=cone_object.num_instances, device=cone_object.device
        # )
        # # write root state to simulation
        # self.cone_object.write_root_state_to_sim(root_state)

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

