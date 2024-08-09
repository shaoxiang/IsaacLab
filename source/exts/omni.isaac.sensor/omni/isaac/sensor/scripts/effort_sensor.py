# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import copy
from typing import Optional

import carb
import numpy as np
import omni.kit.utils
import omni.usd
from omni.isaac.core.articulations.articulation import Articulation
from omni.isaac.core_nodes.bindings import _omni_isaac_core_nodes


class EsSensorReading:
    def __init__(self, is_valid: bool = False, time: float = 0, value: float = 0) -> None:
        self.is_valid = is_valid
        self.time = time
        self.value = value


class EffortSensor(Articulation):
    def __init__(
        self,
        prim_path: str,
        sensor_period: float = -1,
        use_latest_data: bool = False,
        enabled: bool = True,
    ) -> None:
        self.current_time = 0
        self.sensor_time = 0
        self.sensor_period = sensor_period
        self.enabled = enabled
        self.use_latest_data = use_latest_data
        self.step_size = 0
        self.data_buffer_size = 10
        self.sensor_reading_buffer = [EsSensorReading() for i in range(self.data_buffer_size)]
        self.interpolation_buffer = copy.deepcopy(self.sensor_reading_buffer)
        self.core_nodes = _omni_isaac_core_nodes.acquire_interface()
        self.physics_num_steps = 0
        self.is_initialized = False
        self.dof = None

        self.body_prim_path = "/".join(prim_path.split("/")[:-1])
        self.dof_name = prim_path.split("/")[-1]

        super().__init__(prim_path=self.body_prim_path, name=prim_path)
        self.initialize_callbacks()
        return

    def initialize_callbacks(self) -> None:
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

    def lerp(self, start: float, end: float, time: float) -> float:
        return start + ((end - start) * time)

    def _stage_open_callback_fn(self, event=None) -> None:
        self._acquisition_callback = None
        self._timer_reset_callback = None
        self._stage_open_callback = None
        return

    def _timeline_timer_callback_fn(self, event) -> None:
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self.current_time = 0
            self.sensor_time = 0
            self.sensor_reading_buffer = [EsSensorReading() for i in range(self.data_buffer_size)]
            self.interpolation_buffer = copy.deepcopy(self.sensor_reading_buffer)
            self.physics_num_steps = 0
        elif event.type == int(omni.timeline.TimelineEventType.PLAY):
            self.is_initialized = False
        return

    def _data_acquisition_callback(self, step_size: float) -> None:
        # update sensor reading pair
        self.step_size = step_size
        self.current_time = float(self.core_nodes.get_sim_time())
        self.physics_num_steps = float(self.core_nodes.get_physics_num_steps())
        if self.physics_num_steps <= 2:
            return

        elif not self.is_initialized:
            super().initialize()
            self.dof = self.get_dof_index(self.dof_name)
            self.is_initialized = True

        if self.enabled:
            efforts = self.get_measured_joint_efforts(self.dof)
            self.sensor_reading_buffer.insert(0, EsSensorReading())
            self.sensor_reading_buffer.pop()

            self.sensor_reading_buffer[0].time = self.current_time

            if efforts is None or len(efforts) != 1:
                self.sensor_reading_buffer[0].value = 0
                self.sensor_reading_buffer[0].is_valid = False
                carb.log_warn(
                    f"Effort sensor error, none or multiple efforts found for path:  {self.prim_path} with joint {self.dof_name}"
                )

            else:
                self.sensor_reading_buffer[0].value = efforts[0]
                self.sensor_reading_buffer[0].is_valid = True

            if self.sensor_period <= self.step_size:
                self.sensor_time = self.current_time
            elif self.sensor_time + self.sensor_period <= self.current_time:
                self.interpolation_buffer = copy.deepcopy(self.sensor_reading_buffer)
                self.sensor_time += self.sensor_period

    def get_sensor_reading(self, interpolation_function=None, use_latest_data=False) -> EsSensorReading():
        sensor_reading = EsSensorReading()
        if self.enabled:
            # case 1: get latest reading when sensor freq is higher
            # use latest data step enabled
            # sensor time + period is slower than last step (out of sync)
            if (
                self.sensor_period <= self.step_size
                or self.use_latest_data
                or use_latest_data
                or self.sensor_time + self.sensor_period < self.sensor_reading_buffer[1].time
            ):
                sensor_reading = self.sensor_reading_buffer[0]

                # if the data is invalid, output default
                if not self.sensor_reading_buffer[0].is_valid:
                    sensor_reading = EsSensorReading()

                elif (
                    self.sensor_period > self.step_size
                    and not self.use_latest_data
                    and not use_latest_data
                    and self.sensor_time + self.sensor_period < self.sensor_reading_buffer[1].time
                ):
                    carb.log_warn("sensor time out of sync, using latest data")

            # case 2: use interpolated data
            else:
                sensor_reading.time = self.sensor_time
                # if both pairs are valid, interpolate
                if self.sensor_reading_buffer[0].is_valid and self.sensor_reading_buffer[1].is_valid:
                    if interpolation_function is None:
                        if self.interpolation_buffer[1].time == self.interpolation_buffer[0].time:
                            sensor_reading.value = self.lerp(
                                self.interpolation_buffer[1].value,
                                self.interpolation_buffer[0].value,
                                float((sensor_reading.time - self.interpolation_buffer[1].time) / (self.step_size)),
                            )
                        else:
                            sensor_reading.value = self.lerp(
                                self.interpolation_buffer[1].value,
                                self.interpolation_buffer[0].value,
                                float(
                                    (sensor_reading.time - self.interpolation_buffer[1].time)
                                    / (self.interpolation_buffer[0].time - self.interpolation_buffer[1].time)
                                ),
                            )
                        sensor_reading.is_valid = True
                    else:
                        sensor_reading = interpolation_function(self.interpolation_buffer, self.sensor_time)

                else:
                    # if the most recent one is valid, but old data is not, use the most recent
                    if self.sensor_reading_buffer[1].is_valid:
                        sensor_reading = self.sensor_reading_buffer[0]
                        sensor_reading.is_valid = True

                    # no valid data, reset it
                    else:
                        sensor_reading = EsSensorReading()
        return sensor_reading

    def update_dof_name(self, dof_name: str) -> None:
        if self.physics_num_steps <= 2:
            carb.log_warn("unable to update path, please call again after 3 physics steps")
            return
        self.dof_name = dof_name
        try:
            self.dof = self.get_dof_index(self.dof_name)
        except:
            carb.log_warn("unable to find joint corresponding to the dof name, disabling sensor")
            self.dof = None

    def change_buffer_size(self, new_buffer_size: int) -> None:
        self.sensor_reading_buffer = np.resize(np.array(self.sensor_reading_buffer), new_buffer_size).tolist()
        self.interpolation_buffer = np.resize(np.array(self.interpolation_buffer), new_buffer_size).tolist()
        self.data_buffer_size = new_buffer_size
