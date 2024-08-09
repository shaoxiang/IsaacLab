# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


from .test_camera_sensor import *
from .test_camera_view_sensor import *
from .test_contact_sensor import *
from .test_contact_sensor_wrapper import *
from .test_effort_sensor import *
from .test_imu_sensor import *
from .test_imu_sensor_wrapper import *
from .test_lidar_rtx import *
from .test_rotating_lidar_physX import *
from .test_rtx_flat_scan import *
from .test_rtx_rotary_lidar import *

# TODO, solid state test causes other tests to fail
# from .test_rtx_solid_state_lidar import *

scan_for_test_modules = True
