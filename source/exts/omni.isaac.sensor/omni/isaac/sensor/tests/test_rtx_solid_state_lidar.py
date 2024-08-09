# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html

import asyncio

import carb
import carb.tokens
import numpy as np
import omni
import omni.hydratexture
import omni.kit
import omni.kit.commands
import omni.kit.test
import omni.replicator.core as rep
import omni.usd
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils.prims import delete_prim
from omni.isaac.core.utils.stage import create_new_stage_async, update_stage_async


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestRTXSolildStateLidar(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        self._settings = carb.settings.acquire_settings_interface()
        await create_new_stage_async()

        await update_stage_async()
        # This needs to be set so that kit updates match physics updates
        self._physics_rate = 60
        self._sensor_rate = 120
        self._settings.set_bool("/app/runLoops/main/rateLimitEnabled", True)
        self._settings.set_int("/app/runLoops/main/rateLimitFrequency", int(self._physics_rate))
        self._settings.set_int("/persistent/simulation/minFrameRate", int(self._physics_rate))

        pass

    async def tearDown(self):
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            # print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()

        self._settings = None

    async def test_rtx_solid_state_lidar_point_cloud(self):
        VisualCuboid(prim_path="/World/cube1", position=np.array([5, 0, 0]), scale=np.array([1, 20, 1]))
        VisualCuboid(prim_path="/World/cube2", position=np.array([-5, 0, 0]), scale=np.array([1, 20, 1]))
        VisualCuboid(prim_path="/World/cube3", position=np.array([0, 5, 0]), scale=np.array([20, 1, 1]))
        VisualCuboid(prim_path="/World/cube4", position=np.array([0, -5, 0]), scale=np.array([20, 1, 1]))
        await update_stage_async()

        config = "Example_Solid_State"
        _, sensor = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", path="/sensor", parent=None, config=config)
        texture = rep.create.render_product(sensor.GetPath().pathString, resolution=[1, 1])
        render_product_path = texture.path
        rv = "RtxLidar"
        writer = rep.writers.get(rv + "DebugDrawPointCloud")
        writer.attach([render_product_path])
        await update_stage_async()
        await update_stage_async()
        omni.timeline.get_timeline_interface().play()
        await omni.syntheticdata.sensors.next_render_simulation_async(render_product_path, 60)
        # cleanup and shutdown
        omni.timeline.get_timeline_interface().stop()
        writer.detach()
        await update_stage_async()
        delete_prim(sensor.GetPath())
        await update_stage_async()
        texture.destroy()

    pass
