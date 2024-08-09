# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional

import carb
import numpy as np
import omni.isaac.IsaacSensorSchema as IsaacSensorSchema
import omni.kit.commands
from omni.isaac.core.prims.base_sensor import BaseSensor
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.stage import traverse_stage
from omni.isaac.core_nodes.bindings import _omni_isaac_core_nodes
from omni.isaac.sensor import _sensor
from pxr import PhysxSchema


class IMUSensor(BaseSensor):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "imu_sensor",
        frequency: Optional[int] = None,
        dt: Optional[float] = None,
        translation: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        linear_acceleration_filter_size: Optional[int] = 1,
        angular_velocity_filter_size: Optional[int] = 1,
        orientation_filter_size: Optional[int] = 1,
    ) -> None:
        if frequency is not None and dt is not None:
            raise Exception("Sensor Frequency and Sensor dt can't be both specified")

        if frequency is not None:
            dt = float(1 / frequency)

        self._body_prim_path = "/".join(prim_path.split("/")[:-1])
        self._sensor_name = prim_path.split("/")[-1]
        self._imu_sensor_interface = _sensor.acquire_imu_sensor_interface()
        self._core_nodes = _omni_isaac_core_nodes.acquire_interface()

        linear_acceleration_filter_size = max(linear_acceleration_filter_size, 1)
        angular_velocity_filter_size = max(angular_velocity_filter_size, 1)
        orientation_filter_size = max(orientation_filter_size, 1)

        if is_prim_path_valid(prim_path):
            self._isaac_sensor_prim = IsaacSensorSchema.IsaacImuSensor(get_prim_at_path(prim_path))
            BaseSensor.__init__(
                self,
                prim_path=prim_path,
                name=name,
                translation=translation,
                position=position,
                orientation=orientation,
            )
            if dt is not None:
                self.set_dt(dt)
        else:
            if dt is None:
                for prim in traverse_stage():
                    if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                        current_physics_prim = prim
                physx_scene_api = PhysxSchema.PhysxSceneAPI(current_physics_prim)
                current_physics_frequency = physx_scene_api.GetTimeStepsPerSecondAttr().Get()
                dt = 1.0 / current_physics_frequency
            carb.log_warn("Creating a new IMU prim at path {}".format(prim_path))
            success, self._isaac_sensor_prim = omni.kit.commands.execute(
                "IsaacSensorCreateImuSensor",
                path="/" + self._sensor_name,
                parent=self._body_prim_path,
                sensor_period=dt,
                linear_acceleration_filter_size=linear_acceleration_filter_size,
                angular_velocity_filter_size=angular_velocity_filter_size,
                orientation_filter_size=orientation_filter_size,
            )
            if not success:
                raise Exception("Not successful")
            BaseSensor.__init__(
                self,
                prim_path=prim_path,
                name=name,
                translation=translation,
                position=position,
                orientation=orientation,
            )
        self._pause = False
        self._current_time = 0
        self._current_frame = dict()
        self._current_frame["time"] = 0
        self._current_frame["physics_step"] = 0
        self._current_frame["lin_acc"] = self._backend_utils.create_zeros_tensor(
            shape=[3], dtype="float32", device=self._device
        )
        self._current_frame["ang_vel"] = self._backend_utils.create_zeros_tensor(
            shape=[3], dtype="float32", device=self._device
        )
        self._current_frame["orientation"] = self._backend_utils.create_zeros_tensor(
            shape=[4], dtype="float32", device=self._device
        )
        self._current_frame["orientation"][0] = 1
        return

    def initialize(self, physics_sim_view=None) -> None:
        BaseSensor.initialize(self, physics_sim_view=physics_sim_view)

    def get_current_frame(self, read_gravity=True) -> dict:
        imu_sensor_reading = self._imu_sensor_interface.get_sensor_reading(self.prim_path, read_gravity=read_gravity)
        if imu_sensor_reading.is_valid:
            self._current_frame["lin_acc"] = self._backend_utils.create_tensor_from_list(
                [
                    imu_sensor_reading.lin_acc_x,
                    imu_sensor_reading.lin_acc_y,
                    imu_sensor_reading.lin_acc_z,
                ],
                dtype="float32",
                device=self._device,
            )
            self._current_frame["ang_vel"] = self._backend_utils.create_tensor_from_list(
                [
                    imu_sensor_reading.ang_vel_x,
                    imu_sensor_reading.ang_vel_y,
                    imu_sensor_reading.ang_vel_z,
                ],
                dtype="float32",
                device=self._device,
            )
            self._current_frame["orientation"] = self._backend_utils.create_tensor_from_list(
                [
                    imu_sensor_reading.orientation[3],
                    imu_sensor_reading.orientation[0],
                    imu_sensor_reading.orientation[1],
                    imu_sensor_reading.orientation[2],
                ],
                dtype="float32",
                device=self._device,
            )
            self._current_frame["time"] = float(imu_sensor_reading.time)
            self._current_frame["physics_step"] = float(self._core_nodes.get_physics_num_steps())
        return self._current_frame

    def resume(self) -> None:
        self._isaac_sensor_prim.GetEnabledAttr().Set(True)
        return

    def pause(self) -> None:
        self._isaac_sensor_prim.GetEnabledAttr().Set(False)
        return

    def is_paused(self) -> bool:
        if not self._isaac_sensor_prim.GetEnabledAttr().Get():
            return True
        return False

    def set_frequency(self, value: int) -> None:
        self._isaac_sensor_prim.GetSensorPeriodAttr().Set(1.0 / value)
        return

    def get_frequency(self) -> int:
        return int(1.0 / self._isaac_sensor_prim.GetSensorPeriodAttr().Get())

    def get_dt(self) -> float:
        return self._isaac_sensor_prim.GetSensorPeriodAttr().Get()

    def set_dt(self, value: float) -> None:
        self._isaac_sensor_prim.GetSensorPeriodAttr().Set(value)
        return
