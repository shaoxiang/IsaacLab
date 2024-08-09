# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional, Tuple

import carb
import numpy as np
import omni
import omni.isaac.RangeSensorSchema as RangeSensorSchema
from omni.isaac.core.prims.base_sensor import BaseSensor
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.range_sensor import _range_sensor
from pxr import Sdf


class RotatingLidarPhysX(BaseSensor):
    def __init__(
        self,
        prim_path: str,
        name: str = "rotating_lidar_physX",
        rotation_frequency: Optional[float] = None,
        rotation_dt: Optional[float] = None,
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        fov: Optional[Tuple[float, float]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        valid_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        if rotation_frequency is not None and rotation_dt is not None:
            raise Exception("Rotation Frequency and Rotation dt can't be both specified")

        if rotation_dt is not None:
            rotation_frequency = int(1 / rotation_dt)

        self._lidar_sensor_interface = _range_sensor.acquire_lidar_sensor_interface()
        if is_prim_path_valid(prim_path):
            self._lidar_prim = RangeSensorSchema.Lidar(get_prim_at_path(prim_path))
        else:
            carb.log_info("Creating a new Lidar prim at path {}".format(prim_path))
            self._lidar_prim = RangeSensorSchema.Lidar.Define(get_current_stage(), Sdf.Path(prim_path))
            if rotation_frequency is None:
                rotation_frequency = 20.0
            if fov is None:
                fov = (360.0, 10.0)
            if resolution is None:
                resolution = (1.0, 1.0)
            if valid_range is None:
                valid_range = (0.4, 2000.0)
        BaseSensor.__init__(
            self, prim_path=prim_path, name=name, translation=translation, position=position, orientation=orientation
        )
        if rotation_frequency is not None:
            self.set_rotation_frequency(rotation_frequency)
        if fov is not None:
            self.set_fov(fov)
        if resolution is not None:
            self.set_resolution(resolution)
        if valid_range is not None:
            self.set_valid_range(valid_range)
        self._pause = False
        self._current_time = 0
        self._number_of_physics_steps = 0
        self._current_frame = dict()
        self._current_frame["time"] = 0
        self._current_frame["physics_step"] = 0
        return

    def initialize(self, physics_sim_view=None) -> None:
        BaseSensor.initialize(self, physics_sim_view=physics_sim_view)
        self._acquisition_callback = omni.physx.acquire_physx_interface().subscribe_physics_step_events(
            self._data_acquisition_callback
        )
        self._stage_open_callback = (
            omni.usd.get_context()
            .get_stage_event_stream()
            .create_subscription_to_pop_by_type(int(omni.usd.StageEventType.OPENED), self._stage_open_callback_fn)
        )
        timeline = omni.timeline.get_timeline_interface()
        self._timer_reset_callback = timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._timeline_timer_callback_fn
        )
        return

    def _stage_open_callback_fn(self, event):
        self._acquisition_callback = None
        self._timer_reset_callback = None
        self._stage_open_callback = None
        return

    def _timeline_timer_callback_fn(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._current_time = 0
            self._number_of_physics_steps = 0
        return

    def post_reset(self) -> None:
        BaseSensor.post_reset(self)
        self._current_time = 0
        self._number_of_physics_steps = 0
        return

    def add_depth_data_to_frame(self) -> None:
        self._current_frame["depth"] = None
        return

    def remove_depth_data_from_frame(self) -> None:
        del self._current_frame["depth"]
        return

    def add_linear_depth_data_to_frame(self) -> None:
        self._current_frame["linear_depth"] = None
        return

    def remove_linear_depth_data_from_frame(self) -> None:
        del self._current_frame["linear_depth"]
        return

    def add_intensity_data_to_frame(self) -> None:
        self._current_frame["intensity"] = None
        return

    def remove_intensity_data_from_frame(self) -> None:
        del self._current_frame["intensity"]
        return

    def add_zenith_data_to_frame(self) -> None:
        self._current_frame["zenith"] = None
        return

    def remove_zenith_data_from_frame(self) -> None:
        del self._current_frame["zenith"]
        return

    def add_azimuth_data_to_frame(self) -> None:
        self._current_frame["azimuth"] = None
        return

    def remove_azimuth_data_from_frame(self) -> None:
        del self._current_frame["azimuth"]
        return

    def add_point_cloud_data_to_frame(self) -> None:
        self._current_frame["point_cloud"] = None
        return

    def remove_point_cloud_data_from_frame(self) -> None:
        del self._current_frame["point_cloud"]
        return

    def add_semantics_data_to_frame(self) -> None:
        if not self.is_semantics_enabled():
            self.enable_semantics()
        self._current_frame["semantics"] = None
        return

    def remove_semantics_data_from_frame(self) -> None:
        self.disable_semantics()
        del self._current_frame["semantics"]
        return

    def _data_acquisition_callback(self, step_size: float) -> None:
        self._current_time += step_size
        self._number_of_physics_steps += 1
        if not self._pause:
            for key in self._current_frame:
                if key not in ["physics_step", "time"]:
                    if key == "semantics":
                        self._current_frame[key] = self._backend_utils.create_tensor_from_list(
                            self._lidar_sensor_interface.get_semantic_data(self.prim_path),
                            dtype="float32",
                            device=self._device,
                        )
                    else:
                        self._current_frame[key] = self._backend_utils.create_tensor_from_list(
                            getattr(self._lidar_sensor_interface, "get_{}_data".format(key))(self.prim_path),
                            dtype="float32",
                            device=self._device,
                        )

            self._current_frame["physics_step"] = self._number_of_physics_steps
            self._current_frame["time"] = self._current_time
        return

    def get_num_rows(self) -> int:
        return self._lidar_sensor_interface.get_num_rows(self.prim_path)

    def get_num_cols(self) -> int:
        return self._lidar_sensor_interface.get_num_cols(self.prim_path)

    def get_num_cols_in_last_step(self) -> int:
        return self._lidar_sensor_interface.get_num_cols_ticked(self.prim_path)

    def get_current_frame(self) -> dict:
        return self._current_frame

    def resume(self) -> None:
        self._pause = False
        return

    def pause(self) -> None:
        self._pause = True
        return

    def is_paused(self) -> bool:
        return self._pause

    def set_fov(self, value: Tuple[float, float]) -> None:
        if self.prim.GetAttribute("horizontalFov").Get() is None:
            self._lidar_prim.CreateHorizontalFovAttr().Set(value[0])
        else:
            self.prim.GetAttribute("horizontalFov").Set(value[0])
        if self.prim.GetAttribute("verticalFov").Get() is None:
            self._lidar_prim.CreateVerticalFovAttr().Set(value[1])
        else:
            self.prim.GetAttribute("verticalFov").Set(value[1])

    def get_fov(self) -> Tuple[float, float]:
        return self.prim.GetAttribute("horizontalFov").Get(), self.prim.GetAttribute("verticalFov").Get()

    def set_resolution(self, value: float) -> None:
        if self.prim.GetAttribute("horizontalResolution").Get() is None:
            self._lidar_prim.CreateHorizontalResolutionAttr().Set(value[0])
        else:
            self.prim.GetAttribute("horizontalResolution").Set(value[0])
        if self.prim.GetAttribute("verticalResolution").Get() is None:
            self._lidar_prim.CreateVerticalResolutionAttr().Set(value[1])
        else:
            self.prim.GetAttribute("verticalResolution").Set(value[1])

    def get_resolution(self) -> float:
        return self.prim.GetAttribute("horizontalResolution").Get(), self.prim.GetAttribute("verticalResolution").Get()

    def set_rotation_frequency(self, value: int) -> None:
        if self.get_rotation_frequency() is None:
            self._lidar_prim.CreateRotationRateAttr().Set(value)
        else:
            self.prim.GetAttribute("rotationRate").Set(value)

    def get_rotation_frequency(self) -> int:
        return self.prim.GetAttribute("rotationRate").Get()

    def set_valid_range(self, value: Tuple[float, float]) -> None:
        if self.prim.GetAttribute("minRange").Get() is None:
            self._lidar_prim.CreateMinRangeAttr().Set(value[0])
        else:
            self.prim.GetAttribute("minRange").Set(value[0])
        if self.prim.GetAttribute("maxRange").Get() is None:
            self._lidar_prim.CreateMaxRangeAttr().Set(value[1])
        else:
            self.prim.GetAttribute("maxRange").Set(value[1])
        return

    def get_valid_range(self) -> Tuple[float, float]:
        return self.prim.GetAttribute("minRange").Get(), self.prim.GetAttribute("maxRange").Get()

    def enable_semantics(self) -> None:
        if self.prim.GetAttribute("enableSemantics").Get() is None:
            self._lidar_prim.CreateEnableSemanticsAttr().Set(True)
        else:
            self.prim.GetAttribute("enableSemantics").Set(True)

    def disable_semantics(self) -> None:
        if self.prim.GetAttribute("enableSemantics").Get() is None:
            self._lidar_prim.CreateEnableSemanticsAttr().Set(True)
        else:
            self.prim.GetAttribute("enableSemantics").Set(True)

    def is_semantics_enabled(self) -> bool:
        return self.prim.GetAttribute("enableSemantics").Get()

    def enable_visualization(self, high_lod: bool = False, draw_points: bool = True, draw_lines: bool = True) -> None:
        if self.prim.GetAttribute("highLod").Get() is None:
            self._lidar_prim.CreateHighLodAttr().Set(high_lod)
        else:
            self.prim.GetAttribute("highLod").Set(high_lod)

        if self.prim.GetAttribute("drawPoints").Get() is None:
            RangeSensorSchema.RangeSensor(self._lidar_prim).CreateDrawPointsAttr().Set(draw_points)
        else:
            self.prim.GetAttribute("drawPoints").Set(draw_points)

        if self.prim.GetAttribute("drawLines").Get() is None:
            RangeSensorSchema.RangeSensor(self._lidar_prim).CreateDrawLinesAttr().Set(draw_lines)
        else:
            self.prim.GetAttribute("drawLines").Set(draw_lines)
        return

    def disable_visualization(self) -> None:
        self.enable_visualization(high_lod=False, draw_points=False, draw_lines=False)
        return
