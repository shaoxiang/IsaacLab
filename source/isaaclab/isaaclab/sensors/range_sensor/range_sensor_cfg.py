# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg
from .range_sensor import RangeSensor


@configclass 
class RangeSensorCfg(SensorBaseCfg):
    """Configuration for a range sensor sensor."""

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

        convention: Literal["opengl", "ros", "world"] = "ros"
        """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y`` - Offset is applied in the OpenGL (Usd.Camera) convention.
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y`` - Offset is applied in the ROS convention.
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z`` - Offset is applied in the World Frame convention.

        """

    class_type: type = RangeSensor

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity.

    Note:
        The parent frame is the frame the sensor attaches to. For example, the parent frame of a
        camera at path ``/World/envs/env_0/Robot/Camera`` is ``/World/envs/env_0/Robot``.
    """

    horizontal_fov: float = 360.0
    """The horizontal field of view of the range sensor in degrees. Defaults to 360.0."""
    
    horizontal_resolution: float = 0.4
    """The horizontal resolution of the range sensor. Defaults to None."""
    
    max_range: None | float = None
    """The maximum range of the range sensor in meters. Defaults to None."""
    
    min_range: None | float = None
    """The minimum range of the range sensor in meters. Defaults to None."""
    
    rotation_rate: None | float = None
    """The rotation rate of the range sensor in radians per second. Defaults to None."""
    
    vertical_fov: None | float = None
    """The vertical field of view of the range sensor in degrees. Defaults to None."""
    
    vertical_resolution: None | float = None
    """The vertical resolution of the range sensor. Defaults to None."""

    draw_lines: None | bool = False
    """Whether to draw lines to visualize the range sensor rays. Defaults to False."""
    
    draw_points: None | bool = False
    """Whether to draw points to visualize the range sensor rays. Defaults to False."""
    
    high_lod : None | bool = False
    """Don't know. Defaults to False."""
    
    yaw_offset: None | float = None
    """The yaw offset of the range sensor in radians. Defaults to None."""
    
    enable_semantics: None | bool = False
    """Whether to enable semantic segmentation. Defaults to False. Right now semantics are not supported."""