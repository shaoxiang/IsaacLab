# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import json
import math
import os
import os.path
from typing import Optional, Sequence, Tuple

import carb
import numpy as np
import omni
import omni.graph.core as og
import omni.replicator.core as rep
from omni.isaac.core.prims.base_sensor import BaseSensor
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_type_name, is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core_nodes.bindings import _omni_isaac_core_nodes
from omni.isaac.IsaacSensorSchema import IsaacRtxLidarSensorAPI
from omni.syntheticdata import sensors


class LidarRtx(BaseSensor):
    def __init__(
        self,
        prim_path: str,
        name: str = "lidar_rtx",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        config_file_name: Optional[str] = None,
        firing_frequency: Optional[int] = None,
        firing_dt: Optional[float] = None,
        rotation_frequency: Optional[int] = None,
        rotation_dt: Optional[float] = None,
        valid_range: Optional[Tuple[float, float]] = None,
        scan_type: Optional[str] = None,
        elevation_range: Optional[Tuple[float, float]] = None,
        range_resolution: Optional[float] = None,
        range_accuracy: Optional[float] = None,
        avg_power: float = None,
        wave_length: float = None,
        pulse_time: float = None,
    ) -> None:
        if rotation_dt:
            rotation_frequency = int(1.0 / rotation_dt)

        if firing_dt:
            firing_frequency = int(1.0 / firing_dt)

        if elevation_range:
            carb.log_warn(
                "elevation_range is deprecated; Lidar config no longer supports upElevationDeg, downElevationDeg."
            )

        self._temp_data_file_path = None
        if is_prim_path_valid(prim_path):
            if get_prim_type_name(prim_path) != "Camera" or not get_prim_at_path(prim_path).HasAPI(
                IsaacRtxLidarSensorAPI
            ):
                raise Exception("prim path does not correspond to a Isaac Lidar prim.")
            carb.log_warn("Using existing RTX Lidar prim at path {}".format(prim_path))
        else:
            if config_file_name is None:
                file_index = 0
                file_path = os.path.abspath(
                    os.path.join(
                        get_extension_path_from_name("omni.isaac.sensor"),
                        "data/lidar_configs/Temp_Config_" + str(file_index) + ".json",
                    )
                )
                while os.path.isfile(file_path):
                    file_index += 1
                    file_path = os.path.abspath(
                        os.path.join(
                            get_extension_path_from_name("omni.isaac.sensor"),
                            "data/lidar_configs/Temp_Config_" + str(file_index) + ".json",
                        )
                    )
                config_file_name = "Temp_Config_" + str(file_index)
                self._temp_data_file_path = file_path
                self._create_rtx_lidar_json_file(
                    file_path=file_path,
                    firing_frequency=firing_frequency,
                    rotation_frequency=rotation_frequency,
                    valid_range=valid_range,
                    scan_type=scan_type,
                    range_resolution=range_resolution,
                    range_accuracy=range_accuracy,
                    avg_power=avg_power,
                    wave_length=wave_length,
                    pulse_time=pulse_time,
                )
            if orientation is None:
                orientation = [1, 0, 0, 0]
            if translation is None:
                translation = [0, 0, 1]
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar", path=prim_path, parent=None, config=config_file_name
            )
        self._render_product = rep.create.render_product(prim_path, resolution=(1, 1))
        self._render_product_path = self._render_product.path
        self._point_cloud_node_path = None
        self._flat_scan_node_path = None
        self._create_point_cloud_graph_node()
        self._create_flat_scan_graph_node()
        self._debug_draw_node_path = None
        self._core_nodes_interface = _omni_isaac_core_nodes.acquire_interface()
        BaseSensor.__init__(
            self, prim_path=prim_path, name=name, translation=translation, position=position, orientation=orientation
        )
        if position is not None and orientation is not None:
            self.set_world_pose(position=position, orientation=orientation)
        elif translation is not None and orientation is not None:
            self.set_local_pose(translation=translation, orientation=orientation)
        elif orientation is not None:
            self.set_local_pose(orientation=orientation)
        self._current_frame = dict()
        self._current_frame["rendering_time"] = 0
        self._current_frame["rendering_frame"] = 0
        self._writer = None

        self._attribute_map = {
            "point_cloud_data": "data",
            "range": "range",
            "azimuth": "azimuth",
            "elevation": "elevation",
            "linear_depth_data": "linearDepthData",
            "intensities_data": "intensitiesData",
        }
        return

    def __del__(self):
        if self._render_product:
            self._render_product.destroy()

    def get_render_product_path(self) -> str:
        """
        Returns:
            string: gets the path to the render product used by the lidar
        """
        return self._render_product_path

    def get_current_frame(self) -> dict:
        return self._current_frame

    def _create_point_cloud_graph_node(self):
        self._point_cloud_annotator = rep.AnnotatorRegistry.get_annotator(
            "RtxSensorCpu" + "IsaacComputeRTXLidarPointCloud"
        )
        self._point_cloud_annotator.attach([self._render_product_path])
        self._point_cloud_node_path = self._point_cloud_annotator.get_node().get_prim_path()
        return

    def _create_flat_scan_graph_node(self):
        self._flat_scan_annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpu" + "IsaacComputeRTXLidarFlatScan")
        self._flat_scan_annotator.attach([self._render_product_path])
        self._flat_scan_node_path = self._flat_scan_annotator.get_node().get_prim_path()
        return

    def initialize(self, physics_sim_view=None) -> None:
        BaseSensor.initialize(self, physics_sim_view=physics_sim_view)
        self._acquisition_callback = (
            omni.kit.app.get_app_interface()
            .get_update_event_stream()
            .create_subscription_to_pop(self._data_acquisition_callback)
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
        self._stage_open_callback = None
        self._timer_reset_callback = None
        return

    def _timeline_timer_callback_fn(self, event):
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self.pause()
        elif event.type == int(omni.timeline.TimelineEventType.PLAY):
            self.resume()
        return

    def post_reset(self) -> None:
        BaseSensor.post_reset(self)
        return

    def resume(self) -> None:
        self._acquisition_callback = (
            omni.kit.app.get_app_interface()
            .get_update_event_stream()
            .create_subscription_to_pop(self._data_acquisition_callback)
        )
        return

    def pause(self) -> None:
        self._acquisition_callback = None
        return

    def is_paused(self) -> bool:
        return self._acquisition_callback is None

    # TODO105 : ASYNCRENDERING VALIDATION
    def _data_acquisition_callback(self, event: carb.events.IEvent):
        self._current_frame["rendering_frame"] = (
            og.Controller()
            .node("/Render/PostProcess/SDGPipeline/PostProcessDispatcher")
            .get_attribute("outputs:referenceTimeNumerator")
            .get()
        )

        self._current_frame["rendering_time"] = self._core_nodes_interface.get_sim_time_at_swh_frame(
            self._current_frame["rendering_frame"]
        )
        for key in self._current_frame:
            attribute_name = "".join([word[0].upper() + word[1:] for word in key.split("_")])
            attribute_name = attribute_name[0].lower() + attribute_name[1:]
            if key not in ["rendering_time", "rendering_frame"]:
                if key in ["point_cloud_data", "range", "azimuth", "elevation"]:
                    data = self._point_cloud_annotator.get_data()
                    if key == "point_cloud_data":
                        self._current_frame[key] = data[self._attribute_map[key]]
                    else:
                        self._current_frame[key] = data["info"][self._attribute_map[key]]
                elif key in ["linear_depth_data", "intensities_data"]:
                    data = self._flat_scan_annotator.get_data()
                    self._current_frame[key] = data[self._attribute_map[key]]
        return

    def add_point_cloud_data_to_frame(self):
        self._current_frame["point_cloud_data"] = []
        return

    def remove_point_cloud_data_from_frame(self):
        del self._current_frame["point_cloud_data"]
        return

    def add_linear_depth_data_to_frame(self):
        self._current_frame["linear_depth_data"] = []
        return

    def remove_linear_depth_data_from_frame(self):
        del self._current_frame["linear_depth_data"]
        return

    def add_intensities_data_to_frame(self):
        self._current_frame["intensities_data"] = []
        return

    def remove_intensities_data_from_frame(self):
        del self._current_frame["intensities_data"]
        return

    def add_range_data_to_frame(self):
        self._current_frame["range"] = []
        return

    def remove_range_data_from_frame(self):
        del self._current_frame["range"]
        return

    def add_azimuth_data_to_frame(self):
        self._current_frame["azimuth"] = []
        return

    def remove_azimuth_data_from_frame(self):
        del self._current_frame["azimuth"]
        return

    def add_elevation_data_to_frame(self):
        self._current_frame["elevation"] = []
        return

    def remove_elevation_data_from_frame(self):
        del self._current_frame["elevation"]
        return

    def get_horizontal_resolution(self) -> float:
        return og.Controller().node(self._flat_scan_node_path).get_attribute("outputs:horizontalResolution").get()

    def get_horizontal_fov(self) -> float:
        return og.Controller().node(self._flat_scan_node_path).get_attribute("outputs:horizontalFov").get()

    def get_num_rows(self) -> int:
        return og.Controller().node(self._flat_scan_node_path).get_attribute("outputs:numRows").get()

    def get_num_cols(self) -> int:
        return og.Controller().node(self._flat_scan_node_path).get_attribute("outputs:numCols").get()

    def get_rotation_frequency(self) -> float:
        return og.Controller().node(self._flat_scan_node_path).get_attribute("outputs:rotationRate").get()

    def get_depth_range(self) -> Tuple[float, float]:
        result = og.Controller().node(self._flat_scan_node_path).get_attribute("outputs:depthRange").get()
        return result[0], result[1]

    def get_azimuth_range(self) -> Tuple[float, float]:
        result = og.Controller().node(self._flat_scan_node_path).get_attribute("outputs:azimuthRange").get()
        return result[0], result[1]

    def enable_visualization(self):
        self._writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud")
        self._writer.initialize()
        self._writer.attach([self._render_product_path])

        return

    def disable_visualization(self):
        if self._writer:
            self._writer.detach()
        self._writer = None
        return

    def _create_rtx_lidar_json_file(
        self,
        file_path: str,
        firing_frequency: Optional[int] = None,
        rotation_frequency: Optional[int] = None,
        valid_range: Optional[Tuple[float, float]] = None,
        scan_type: Optional[str] = None,
        range_resolution: Optional[float] = None,
        range_accuracy: Optional[float] = None,
        avg_power: float = None,
        wave_length: float = None,
        pulse_time: float = None,
    ):
        file_name = file_path.split("/")[-1]
        carb.log_info(f"writing to {file_path}")
        config = dict()
        config["class"] = "sensor"
        config["type"] = "lidar"
        config["name"] = str(file_name.split(".json")[0].replace("_", " "))
        config["driveWorksId"] = "GENERIC"
        config["profile"] = dict()
        if scan_type:
            config["profile"]["scanType"] = scan_type
        else:
            config["profile"]["scanType"] = "rotary"

        config["profile"]["intensityProcessing"] = "normalization"
        config["profile"]["rayType"] = "IDEALIZED"

        if valid_range:
            nearRangeM = valid_range[0] / get_stage_units()
            config["profile"]["nearRangeM"] = nearRangeM
            if nearRangeM < 0.4:
                config["profile"]["minDistBetweenEchos"] = nearRangeM
            config["profile"]["farRangeM"] = valid_range[1] / get_stage_units()
        else:
            config["profile"]["nearRangeM"] = 1.0 / get_stage_units()
            config["profile"]["farRangeM"] = 200.0 / get_stage_units()
        if range_resolution:
            config["profile"]["rangeResolutionM"] = range_resolution / get_stage_units()
        else:
            config["profile"]["rangeResolutionM"] = 0.004 / get_stage_units()
        if range_accuracy:
            config["profile"]["rangeAccuracyM"] = range_accuracy / get_stage_units()
        else:
            config["profile"]["rangeAccuracyM"] = 0.02 / get_stage_units()
        if avg_power:
            config["profile"]["avgPowerW"] = avg_power
        else:
            config["profile"]["avgPowerW"] = 0.002
        if wave_length:
            config["profile"]["wavelengthNm"] = wave_length / get_stage_units()
        else:
            config["profile"]["wavelengthNm"] = (9.03e-7 / get_stage_units()) * 1e9
        if pulse_time:
            config["profile"]["pulseTimeNs"] = pulse_time * 1e9
        else:
            config["profile"]["pulseTimeNs"] = 6e-9 * 1e9
        if firing_frequency:
            config["profile"]["reportRateBaseHz"] = firing_frequency
        else:
            config["profile"]["reportRateBaseHz"] = 36000
        if rotation_frequency:
            config["profile"]["scanRateBaseHz"] = rotation_frequency
        else:
            config["profile"]["scanRateBaseHz"] = 10
        config["profile"]["minReflectance"] = 0.1
        config["profile"]["minReflectanceRange"] = 120.0
        config["profile"]["azimuthErrorMean"] = 0.0
        config["profile"]["azimuthErrorStd"] = 0.015
        config["profile"]["elevationErrorMean"] = 0.0
        config["profile"]["elevationErrorStd"] = 0.0000
        config["profile"]["maxReturns"] = 2
        config["profile"]["numberOfEmitters"] = 128
        config["profile"]["intensityMappingType"] = "LINEAR"
        config["profile"]["emitters"] = dict()
        config["profile"]["emitters"]["azimuthDeg"] = [
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
        ]
        config["profile"]["emitters"]["elevationDeg"] = [
            -15.0,
            -14.19,
            -13.39,
            -12.58,
            -11.77,
            -10.97,
            -10.16,
            -9.35,
            -8.55,
            -7.74,
            -6.94,
            -6.13,
            -5.32,
            -4.52,
            -3.71,
            -2.9,
            -2.1,
            -1.29,
            -0.48,
            0.32,
            1.13,
            1.94,
            2.74,
            3.55,
            4.35,
            5.16,
            5.97,
            6.77,
            7.58,
            8.39,
            9.19,
            10.0,
            -15.0,
            -14.19,
            -13.39,
            -12.58,
            -11.77,
            -10.97,
            -10.16,
            -9.35,
            -8.55,
            -7.74,
            -6.94,
            -6.13,
            -5.32,
            -4.52,
            -3.71,
            -2.9,
            -2.1,
            -1.29,
            -0.48,
            0.32,
            1.13,
            1.94,
            2.74,
            3.55,
            4.35,
            5.16,
            5.97,
            6.77,
            7.58,
            8.39,
            9.19,
            10.0,
            -15.0,
            -14.19,
            -13.39,
            -12.58,
            -11.77,
            -10.97,
            -10.16,
            -9.35,
            -8.55,
            -7.74,
            -6.94,
            -6.13,
            -5.32,
            -4.52,
            -3.71,
            -2.9,
            -2.1,
            -1.29,
            -0.48,
            0.32,
            1.13,
            1.94,
            2.74,
            3.55,
            4.35,
            5.16,
            5.97,
            6.77,
            7.58,
            8.39,
            9.19,
            10.0,
            -15.0,
            -14.19,
            -13.39,
            -12.58,
            -11.77,
            -10.97,
            -10.16,
            -9.35,
            -8.55,
            -7.74,
            -6.94,
            -6.13,
            -5.32,
            -4.52,
            -3.71,
            -2.9,
            -2.1,
            -1.29,
            -0.48,
            0.32,
            1.13,
            1.94,
            2.74,
            3.55,
            4.35,
            5.16,
            5.97,
            6.77,
            7.58,
            8.39,
            9.19,
            10.0,
        ]
        config["profile"]["emitters"]["fireTimeNs"] = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            3500,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            7000,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            10500,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            14000,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            17500,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            21000,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
            24500,
        ]
        with open(file_path, "w") as outfile:
            json.dump(config, outfile)
        return

    def __del__(self):
        if self._temp_data_file_path:
            os.remove(self._temp_data_file_path)
        return
