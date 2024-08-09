# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import math

import numpy as np
import omni.kit.test
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage_async, update_stage_async
from omni.isaac.nucleus import get_assets_root_path_async
from omni.isaac.sensor import IMUSensor


class TestIMU(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        await create_new_stage_async()
        self.my_world = World(stage_units_in_meters=1.0)
        await self.my_world.initialize_simulation_context_async()
        await update_stage_async()
        self.my_world.scene.add_default_ground_plane()
        assets_root_path = await get_assets_root_path_async()
        asset_path = assets_root_path + "/Isaac/Robots/Carter/nova_carter_sensors.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Carter")
        my_carter = self.my_world.scene.add(
            Articulation(prim_path="/World/Carter", name="my_carter", position=np.array([0, 0.0, 0.5]))
        )

        self._imu = self.my_world.scene.add(IMUSensor(prim_path="/World/Carter/chassis_link/Imu_Sensor", name="imu"))

        cube_1 = self.my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/cube", name="cube_1", position=np.array([2, 2, 2.5]), scale=np.array([20, 0.2, 5])
            )
        )

        cube_2 = self.my_world.scene.add(
            DynamicCuboid(
                prim_path="/World/cube_2", name="cube_2", position=np.array([2, -2, 2.5]), scale=np.array([20, 0.2, 5])
            )
        )
        await self.my_world.reset_async()
        return

    # After running each test
    async def tearDown(self):
        self.my_world.clear_instance()
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            # print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()
        return

    async def test_data_acquisition(self):
        await update_stage_async()
        await update_stage_async()
        data = self._imu.get_current_frame()
        for key in ["time", "physics_step", "lin_acc", "ang_vel", "orientation"]:
            self.assertTrue(key in data.keys())
        data = self._imu.get_current_frame(read_gravity=False)
        for key in ["time", "physics_step", "lin_acc", "ang_vel", "orientation"]:
            self.assertTrue(key in data.keys())
        return

    async def test_pause_resume(self):
        await update_stage_async()
        await update_stage_async()
        data = self._imu.get_current_frame()
        current_time = data["time"]
        current_step = data["physics_step"]
        self._imu.pause()
        await update_stage_async()
        await update_stage_async()
        await update_stage_async()
        await update_stage_async()
        data = self._imu.get_current_frame()
        self.assertTrue(data["time"] == current_time)
        self.assertTrue(data["physics_step"] == current_step)
        self.assertTrue(self._imu.is_paused())
        current_time = data["time"]
        current_step = data["physics_step"]
        self._imu.resume()
        await update_stage_async()
        data = self._imu.get_current_frame()
        self.assertTrue(data["time"] != current_time)
        self.assertTrue(data["physics_step"] != current_step)
        await self.my_world.reset_async()
        data = self._imu.get_current_frame()
        self.assertTrue(math.isclose(data["time"], 0.016666, abs_tol=0.0001))

        # print(data["physics_step"])
        self.assertTrue(data["physics_step"] == 2)  # data["physics_step"] is 2
        return

    async def test_properties(self):
        self._imu.set_frequency(20)
        # print(self._imu.get_frequency())
        # print(self._imu.get_dt())
        self.assertTrue(math.isclose(20, self._imu.get_frequency(), abs_tol=2))
        self._imu.set_dt(0.2)
        self.assertTrue(math.isclose(0.2, self._imu.get_dt(), abs_tol=0.01))
        return
