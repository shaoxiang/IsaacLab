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
from omni.isaac.sensor import RotatingLidarPhysX


class TestRotatingLidarPhysX(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        await create_new_stage_async()
        self.my_world = World(stage_units_in_meters=1.0)
        await self.my_world.initialize_simulation_context_async()
        await update_stage_async()
        self.my_world.scene.add_default_ground_plane()
        assets_root_path = await get_assets_root_path_async()
        asset_path = assets_root_path + "/Isaac/Robots/Carter/carter_v1.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Carter")
        my_carter = self.my_world.scene.add(
            Articulation(prim_path="/World/Carter", name="my_carter", position=np.array([0, 0.0, 0.5]))
        )

        self._my_lidar = self.my_world.scene.add(
            RotatingLidarPhysX(
                prim_path="/World/Carter/chassis_link/lidar", name="lidar", translation=np.array([-0.06, 0, 0.38])
            )
        )

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
        for annotator in ["depth", "linear_depth", "intensity", "zenith", "azimuth", "point_cloud"]:
            getattr(self._my_lidar, "add_{}_data_to_frame".format(annotator))()
            await update_stage_async()
            data = self._my_lidar.get_current_frame()
            self.assertTrue(annotator in data.keys())
            self.assertTrue(data[annotator].shape[0] > 0)
            getattr(self._my_lidar, "remove_{}_data_from_frame".format(annotator))()
            await update_stage_async()
            data = self._my_lidar.get_current_frame()
            self.assertTrue(annotator not in data.keys())
        self._my_lidar.add_semantics_data_to_frame()
        await update_stage_async()
        data = self._my_lidar.get_current_frame()
        self.assertTrue("semantics" in data.keys())
        self._my_lidar.remove_semantics_data_from_frame()
        data = self._my_lidar.get_current_frame()
        self.assertTrue("semantics" not in data.keys())
        return

    async def test_get_set_properties(self):
        self._my_lidar.set_fov((2, 3))
        value = self._my_lidar.get_fov()
        self.assertTrue(math.isclose(value[0], 2, abs_tol=0.005))
        self.assertTrue(math.isclose(value[1], 3, abs_tol=0.005))
        self._my_lidar.set_resolution((2, 3))
        value = self._my_lidar.get_resolution()
        self.assertTrue(math.isclose(value[0], 2, abs_tol=0.005))
        self.assertTrue(math.isclose(value[1], 3, abs_tol=0.005))
        self._my_lidar.set_valid_range((0.1, 10000))
        value = self._my_lidar.get_valid_range()
        self.assertTrue(math.isclose(value[0], 0.1, abs_tol=0.005))
        self.assertTrue(math.isclose(value[1], 10000, abs_tol=0.005))
        self._my_lidar.set_rotation_frequency(20)
        value = self._my_lidar.get_rotation_frequency()
        self.assertTrue(math.isclose(value, 20, abs_tol=0.005))
        self.assertTrue(self._my_lidar.get_num_rows() > 0)
        self.assertTrue(self._my_lidar.get_num_cols() > 0)
        await update_stage_async()
        # TODO: check why is this returning 0?
        self._my_lidar.get_num_cols_in_last_step()
        self._my_lidar.enable_visualization()
        return

    async def test_visualization(self):
        self._my_lidar.enable_visualization()
        for _ in range(100):
            await update_stage_async()
        self._my_lidar.disable_visualization()
        for _ in range(100):
            await update_stage_async()
        return

    async def test_pause_resume(self):
        await update_stage_async()
        await update_stage_async()
        data = self._my_lidar.get_current_frame()
        current_time = data["time"]
        current_step = data["physics_step"]
        self._my_lidar.pause()
        await update_stage_async()
        await update_stage_async()
        await update_stage_async()
        await update_stage_async()
        data = self._my_lidar.get_current_frame()
        self.assertTrue(data["time"] == current_time)
        self.assertTrue(data["physics_step"] == current_step)
        self.assertTrue(self._my_lidar.is_paused())
        current_time = data["time"]
        current_step = data["physics_step"]
        self._my_lidar.resume()
        await update_stage_async()
        data = self._my_lidar.get_current_frame()
        self.assertTrue(data["time"] != current_time)
        self.assertTrue(data["physics_step"] != current_step)
        await self.my_world.reset_async()
        data = self._my_lidar.get_current_frame()
        self.assertTrue(math.isclose(data["time"], 0.033333335, abs_tol=0.0001))
        self.assertTrue(data["physics_step"] == 2)
        return
