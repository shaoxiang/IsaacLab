# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import TiledCamera, TiledCameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_assets import UAV_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

# from ultralytics import YOLO
import omni
import torch
import io

import Semantics

class PTZControlEnvWindow(BaseEnvWindow):
    """Window manager for the UAV environment."""

    def __init__(self, env: PTZControlEnv, window_name: str = "IsaacLab"):
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

class PTZSceneCfg(InteractiveSceneCfg):
    # person
    # character_cfg: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="/World/Character",
    #     spawn=sim_utils.UsdFileCfg(usd_path=f"D:/Share folders/demo/Collected_enemy/enemy.usd"),
    # )

    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    # )

    character_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/person",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, 12.0), rot=(0.70711, 0.70711, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/People/Characters/biped_demo/biped_rigidbody.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
        ),
    )

@configclass
class PTZControlEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 5.0
    decimation = 2
    action_space = 4 + 3
    observation_space = 15 + 7
    state_space = 0
    debug_vis = False
    max_person_num = 3

    ui_window_class_type = PTZControlEnvWindow

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
    
    # PTZ/Camera_Link/Camera
    # (-0.024216989562340085, -0.07150677787090418, 0.006048562095419465)
    # (0.70711, 0.0, 0.70711, 0.0)

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

    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body", update_period=0.01, history_length=3, debug_vis=False, filter_prim_paths_expr = ["/World/envs/env_.*/person"]
    )

    # add character
    character_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/person",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0, 12.0), rot=(0.70711, 0.70711, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/People/Characters/biped_demo/biped_rigidbody.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = UAV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 5.0
    moment_scale = 0.05

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    error_to_goal_reward_scale = 15.0

    # reward scales
    joint_torque_reward_scale = -2.5e-6
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    contact_forces_scale = 0.1
    ptz_control_scale = 1.0
    yolo_reward_scale = 1.0

    num_channels = 3
    # num_observations_img = num_channels * tiled_camera.height * tiled_camera.width

class PTZControlEnv(DirectRLEnv):
    cfg: PTZControlEnvCfg

    def __init__(self, cfg: PTZControlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the uav
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal cmd
        self._desired_cmd = torch.zeros(self.num_envs, 4, device=self.device)
        self._ptz_action = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._last_ptz_action = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._yolo_rewards = torch.zeros(self.num_envs, device=self.device)
        self._is_collision_occurred = torch.zeros(self.num_envs, device=self.device)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "error_to_goal",
                "contact_forces",
                "yolo_rewards",
                "ang_vel_xyz_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2"
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

        # character
        self._character_body_id = self._character.find_bodies("person")[0]
        
        # Load yolo model
        # self.yolo_model = YOLO("./source/third_part/YOLO/yolo11n.pt")

        # Load low policy model    
        file_content = omni.client.read_file("./source/policy/UAV/policy.pt")[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        self.policy = torch.jit.load(file, map_location=self.device)
        
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.robot.init_state.pos = (0.0, 0.0, 10.0)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # add Character rigidbody
        self._character = RigidObject(self.cfg.character_cfg)
        self.scene.rigid_objects["character"] = self._character
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add camera
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # stage = omni.usd.get_context().get_stage()
        # # add semantics for in-hand cube
        # prim = stage.GetPrimAtPath("/World/envs/env_0/person")
        # sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        # sem.CreateSemanticTypeAttr()
        # sem.CreateSemanticDataAttr()
        # sem.GetSemanticTypeAttr().Set("class")
        # sem.GetSemanticDataAttr().Set("enemy")

    def low_policy_action(self):
        low_policy_obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._desired_cmd,
            ],
            dim=-1,
        )

        low_policy_actions = self.policy(low_policy_obs)
        low_policy_actions = low_policy_actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 1] = self.cfg.thrust_to_weight * self._robot_weight * (low_policy_actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * low_policy_actions[:, 1:]
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        character_thrust = torch.zeros_like(self._thrust).uniform_(-100.0, 100.0)
        character_moment = torch.zeros_like(self._moment).uniform_(-1.0, 1.0)
        self._character.set_external_force_and_torque(character_thrust, character_moment, body_ids=self._character_body_id)
        
    def _pre_physics_step(self, actions: torch.Tensor):
        # _desired_cmd 和 ptz control
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._desired_cmd = 5.0 * self._actions[:, 0:4]
        # print(self._ptz_action[:, :, 0].shape, self._actions[:, 4].shape)
        self._ptz_action[:, 0, 0] = self._actions[:, 4]
        self._ptz_action[:, 0, 1] = self._actions[:, 5]
        self._ptz_action[:, 0, 2] = self._actions[:, 6]
        
    def _apply_action(self):
        self._robot.set_joint_position_target(self.ptz_joint_x_scale * self._ptz_action[:, :, 0], joint_ids=self.ptz_joint_x_idx)
        self._robot.set_joint_position_target(self.ptz_joint_y_scale * self._ptz_action[:, :, 1], joint_ids=self.ptz_joint_y_idx)
        self._robot.set_joint_position_target(self.ptz_joint_z_scale * self._ptz_action[:, :, 2], joint_ids=self.ptz_joint_z_idx)
        self.low_policy_action()

    def yolo_results_filter(self, yolo_results, max_person = 3, choose_device = 'cuda'):
        new_results = torch.zeros(self.num_envs, max_person, 5, device = choose_device)
        self._yolo_rewards = torch.zeros(self.num_envs, device = choose_device)
                
        for i, yolo_result in enumerate(yolo_results):
            person_num = 0
            tmp_num = 0
            tmp_results = torch.zeros(max_person, 5, device = choose_device)
            for index, box in enumerate(yolo_result.boxes.xyxyn.cuda()):
                # print("box:", box, yolo_result.boxes.cls[index])
                one_result = torch.cat((box, (yolo_result.boxes.cls[index] + 1).unsqueeze(0)), dim=0).to(choose_device)
                if yolo_result.boxes.cls[index] == 0.:
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
                self._yolo_rewards[i] = 1

        # print(new_results)
        return new_results

    # def _get_observations(self) -> dict:
    #     self._previous_actions = self._actions.clone()
    #     data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
    #     image_data = self._tiled_camera.data.output[data_type].clone()
    #     # print("image_data:", image_data.shape, image_data.dtype)
    #     image_data_permute = image_data.permute(0, 3, 1, 2) / 255.0
    #     # print("image_data_permute:", image_data_permute.shape, image_data_permute.dtype)
    #     results = self.yolo_model(image_data_permute, stream=True, verbose = False)
    #     # print("results:", results)
    #     # print("results boxes:", results[0].boxes, results[0].boxes.cls)
    #     # print("results boxes len:", len(results[0].boxes), len(results[0].boxes.cls))
    #     yolo_obs = self.yolo_results_filter(results, max_person = self.cfg.max_person_num)
        
    #     yolo_obs_view = yolo_obs.view(self.num_envs, -1)
    #     obs = torch.cat((yolo_obs_view, self._actions), dim = -1)
    #     observations = {"policy": obs}
    #     return observations
    
    def _get_observations(self) -> dict:

        # desired_pos_b, _ = subtract_frame_transforms(
        #     self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        # )

        desired_pos_b = self._robot.data.root_pos_w - self._character.data.root_pos_w

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )

        observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        # vx = torch.square(self._robot.data.root_lin_vel_b[:, 0] - self._desired_cmd[:, 0])
        # vy = torch.square(self._robot.data.root_lin_vel_b[:, 1] - self._desired_cmd[:, 1])
        # vz = torch.square(self._robot.data.root_lin_vel_b[:, 2] - self._desired_cmd[:, 2])
        # wz = torch.square(self._robot.data.root_ang_vel_b[:, 2] - self._desired_cmd[:, 3])

        # print("vx:",vx,"vy:",vy,"vz:",vz,"wz:",wz)
        # print("vx_:",self._robot.data.root_lin_vel_b[:, 0],"vy_:",self._robot.data.root_lin_vel_b[:, 1],"vz_:",self._robot.data.root_lin_vel_b[:, 2],"wz_:",self._robot.data.root_ang_vel_b[:, 2])
        
        # lin_vel_error_to_goal = vx + vy + vz
        # lin_vel_error_to_goal_mapped = 3.0 - torch.tanh(vx) - torch.tanh(vy) - torch.tanh(vz)

        # ang_vel_error_to_goal = wz
        # ang_vel_error_to_goal_mapped = 1.0 - torch.tanh(wz)

        # print("lin_vel_error_to_goal:", lin_vel_error_to_goal, lin_vel_error_to_goal_mapped, "ang_vel_error_to_goal:", ang_vel_error_to_goal, ang_vel_error_to_goal_mapped)
        
        distance_to_goal = torch.linalg.norm(self._robot.data.root_pos_w - self._character.data.root_pos_w, dim=1)        
        # distance_to_goal_exp = torch.exp(-1.5 * distance_to_goal)
        distance_to_goal_mapped = 1.0 - torch.tanh(distance_to_goal / 2.0)
        
        net_contact_forces = self._contact_sensor.data.force_matrix_w
        max_net_contact_forces, _ = torch.max(net_contact_forces.view(net_contact_forces.size(0), -1), dim=1)
        self._is_collision_occurred = max_net_contact_forces > 0.01
        # print("max_net_contact_forces:", max_net_contact_forces)

        # ptz_control_error = 0.1 - torch.sum(torch.square(self._last_ptz_action[:,0,:] - self._ptz_action[:,0,:]), dim=1)
        # ptz_control_error = torch.clamp(ptz_control_error, max=0.0)
        # self._last_ptz_action = self._ptz_action.clone()

        # angular velocity x/y/z
        # ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :3]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "error_to_goal": distance_to_goal_mapped * self.cfg.error_to_goal_reward_scale * self.step_dt,
            # "ptz_control_error": ptz_control_error * self.cfg.ptz_control_scale * self.step_dt, 
            "contact_forces": max_net_contact_forces * self.cfg.contact_forces_scale,
            # "yolo_rewards": self._yolo_rewards * self.cfg.yolo_reward_scale * self.step_dt,
            # "ang_vel_xyz_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            # "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            # "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            # "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # print("rewards:", rewards, "reward:", reward, "dt:", self.step_dt)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = torch.logical_or((self.episode_length_buf >= self.max_episode_length - 1), self._is_collision_occurred)
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < -0.1, self._robot.data.root_pos_w[:, 2] > 200.0)
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
        self._character.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._yolo_rewards[env_ids] = 0
        self._is_collision_occurred[env_ids] = 0

        # Sample new commands        
        
        # Reset robot state
        terrain_origins = self._terrain.env_origins[env_ids]
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += terrain_origins
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._ptz_action[env_ids, :] = torch.zeros_like(self._ptz_action[env_ids, :]).uniform_(-1.0, 1.0) + joint_pos[:,4:7].unsqueeze(1)

        character_state = self._character.data.default_root_state[env_ids].clone()
        character_state[:, :3] += (terrain_origins + torch.zeros_like(terrain_origins).uniform_(-5.0, 5.0))
        self._character.write_root_state_to_sim(character_state, env_ids)

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
