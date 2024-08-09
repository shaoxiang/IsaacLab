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
from pxr import Gf, PhysxSchema, UsdPhysics


class ContactSensor(BaseSensor):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "contact_sensor",
        frequency: Optional[int] = None,
        dt: Optional[float] = None,
        translation: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> None:
        if frequency is not None and dt is not None:
            raise Exception("Sensor Frequency and Sensor dt can't be both specified")

        if frequency is not None:
            dt = int(1 / frequency)

        self._body_prim_path = "/".join(prim_path.split("/")[:-1])
        if is_prim_path_valid(prim_path) and not get_prim_at_path(self._body_prim_path).HasAPI(UsdPhysics.CollisionAPI):
            raise Exception("Contact Sensor needs to be created under another prim that has collision api enabled on.")
        self._sensor_name = prim_path.split("/")[-1]
        self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
        if is_prim_path_valid(prim_path):
            self._isaac_sensor_prim = IsaacSensorSchema.IsaacContactSensor(get_prim_at_path(prim_path))
            BaseSensor.__init__(self, prim_path=prim_path, name=name, translation=translation, position=position)
            if dt is not None:
                self.set_dt(dt)
            if min_threshold is not None:
                self.set_min_threshold(min_threshold)
            if max_threshold is not None:
                self.set_max_threshold(max_threshold)
            if radius is not None:
                self.set_radius(radius)
        else:
            if min_threshold is None:
                min_threshold = 0
            if max_threshold is None:
                max_threshold = 100000
            if radius is None:
                radius = -1
            if dt is None:
                for prim in traverse_stage():
                    if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                        current_physics_prim = prim
                physx_scene_api = PhysxSchema.PhysxSceneAPI(current_physics_prim)
                current_physics_frequency = physx_scene_api.GetTimeStepsPerSecondAttr().Get()
                dt = 1.0 / current_physics_frequency
            color_rgba = np.array([1.0, 1.0, 1.0, 1.0])
            carb.log_info("Creating a new contact sensor prim at path {}".format(prim_path))
            success, self._isaac_sensor_prim = omni.kit.commands.execute(
                "IsaacSensorCreateContactSensor",
                path="/" + self._sensor_name,
                parent=self._body_prim_path,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                color=Gf.Vec4f(*color_rgba.tolist()),
                radius=radius,
                sensor_period=dt,
            )
            if not success:
                raise Exception("Not successful")
            BaseSensor.__init__(self, prim_path=prim_path, name=name, translation=translation, position=position)
        self._pause = False
        self._current_time = 0
        self._number_of_physics_steps = 0
        self._current_frame = dict()
        self._current_frame["time"] = 0
        self._current_frame["physics_step"] = 0
        self._core_nodes = _omni_isaac_core_nodes.acquire_interface()
        return

    def initialize(self, physics_sim_view=None) -> None:
        BaseSensor.initialize(self, physics_sim_view=physics_sim_view)
        return

    def get_current_frame(self) -> None:
        cs_sensor_reading = self._contact_sensor_interface.get_sensor_reading(self.prim_path)
        cs_raw_data = self._contact_sensor_interface.get_contact_sensor_raw_data(self.prim_path)
        if cs_sensor_reading.is_valid:
            self._current_frame["in_contact"] = bool(cs_sensor_reading.in_contact)
            self._current_frame["force"] = float(cs_sensor_reading.value)
            self._current_frame["time"] = float(cs_sensor_reading.time)
            # self._current_frame["physics_step"] = float(self._number_of_physics_steps)
            self._current_frame["number_of_contacts"] = len(cs_raw_data)
            if "contacts" in self._current_frame:
                self._current_frame["contacts"] = []
                for i in range(len(cs_raw_data)):
                    contact_frame = dict()
                    contact_frame["body0"] = self._contact_sensor_interface.decode_body_name(
                        int(cs_raw_data["body0"][i])
                    )
                    contact_frame["body1"] = self._contact_sensor_interface.decode_body_name(
                        int(cs_raw_data["body1"][i])
                    )
                    contact_frame["position"] = self._backend_utils.create_tensor_from_list(
                        [
                            cs_raw_data["position"][i][0],
                            cs_raw_data["position"][i][1],
                            cs_raw_data["position"][i][2],
                        ],
                        dtype="float32",
                        device=self._device,
                    )
                    contact_frame["normal"] = self._backend_utils.create_tensor_from_list(
                        [cs_raw_data["normal"][i][0], cs_raw_data["normal"][i][1], cs_raw_data["normal"][i][2]],
                        dtype="float32",
                        device=self._device,
                    )
                    contact_frame["impulse"] = self._backend_utils.create_tensor_from_list(
                        [
                            cs_raw_data["impulse"][i][0],
                            cs_raw_data["impulse"][i][1],
                            cs_raw_data["impulse"][i][2],
                        ],
                        dtype="float32",
                        device=self._device,
                    )
                    self._current_frame["contacts"].append(contact_frame)
            self._current_frame["physics_step"] = float(self._core_nodes.get_physics_num_steps())
        return self._current_frame

    def add_raw_contact_data_to_frame(self) -> None:
        self._current_frame["contacts"] = []
        return

    def remove_raw_contact_data_from_frame(self) -> None:
        del self._current_frame["contacts"]
        return

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

    def get_radius(self) -> float:
        return self.prim.GetAttribute("radius").Get()

    def set_radius(self, value: float) -> None:
        if self.get_radius() is None:
            self._isaac_sensor_prim.CreateRadiusAttr().Set(value)
        else:
            self.prim.GetAttribute("radius").Set(value)
        return

    def get_min_threshold(self) -> float:
        threshold = self.prim.GetAttribute("threshold").Get()
        if threshold is not None:
            return threshold[0]
        else:
            return None

    def set_min_threshold(self, value: float) -> None:
        if self.get_min_threshold() is None:
            self._isaac_sensor_prim.CreateThresholdAttr().Set((value, 10000))
        else:
            self.prim.GetAttribute("threshold").Set((value, self.get_max_threshold()))
        return

    def get_max_threshold(self) -> float:
        threshold = self.prim.GetAttribute("threshold").Get()
        if threshold is not None:
            return threshold[1]
        else:
            return None

    def set_max_threshold(self, value: float) -> None:
        if self.get_max_threshold() is None:
            self._isaac_sensor_prim.CreateThresholdAttr().Set((0, value))
        else:
            self.prim.GetAttribute("threshold").Set((self.get_min_threshold(), value))
        return
