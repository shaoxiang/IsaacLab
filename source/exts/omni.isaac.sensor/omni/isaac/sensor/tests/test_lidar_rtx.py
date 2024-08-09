# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio

import numpy as np
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.kit.test
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage_async, update_stage_async
from omni.isaac.nucleus import get_assets_root_path_async
from omni.isaac.sensor import LidarRtx


class TestRotatingLidarRtx(omni.kit.test.AsyncTestCase):
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
        self.xform = self.my_world.scene.add(
            XFormPrim(prim_path="/World/Carter/chassis_link/front_hawk/right/lidar_rig", name="rig")
        )
        self._my_lidar = self.my_world.scene.add(
            LidarRtx(prim_path="/World/Carter/chassis_link/front_hawk/right/lidar_rig/lidar", name="lidar")
        )

        cube_1 = self.my_world.scene.add(
            VisualCuboid(
                prim_path="/World/cube", name="cube_1", position=np.array([2, 2, 2.5]), scale=np.array([20, 0.2, 5])
            )
        )

        cube_2 = self.my_world.scene.add(
            VisualCuboid(
                prim_path="/World/cube_2", name="cube_2", position=np.array([2, -2, 2.5]), scale=np.array([20, 0.2, 5])
            )
        )
        await self.my_world.reset_async()
        self._timeline = omni.timeline.get_timeline_interface()
        return

    # After running each test
    async def tearDown(self):
        await self.my_world.stop_async()
        self.my_world.clear_instance()
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            # print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()
        return

    async def test_poses(self):
        self.xform.set_world_pose(
            position=np.array([5.0, 0.0, 5.0]),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, -90, 0]), degrees=True),
        )
        self._my_lidar.set_world_pose(
            position=np.array([0.0, 0.0, 25.0]),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
        )
        position, orientation = self._my_lidar.get_world_pose()
        self.assertTrue(np.isclose(position, [0, 0, 25], atol=1e-05).all())
        self.assertTrue(
            np.isclose(
                orientation, rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True), atol=1e-05
            ).all()
        )
        translation, orientation = self._my_lidar.get_local_pose()
        self.assertTrue(np.isclose(translation, [20, 0, 5], atol=1e-05).all())
        self.assertTrue(
            np.isclose(
                orientation, rot_utils.euler_angles_to_quats(np.array([0, 180, 0]), degrees=True), atol=1e-05
            ).all()
        )
        self._my_lidar.set_local_pose(
            translation=np.array([0.0, 0.0, 25.0]),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
        )
        return

    async def test_read_data_annotator(self):
        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpu" + "IsaacReadRTXLidarData")
        annotator.attach([self._my_lidar.get_render_product_path()])

        self._timeline.play()
        await omni.syntheticdata.sensors.next_render_simulation_async(self._my_lidar.get_render_product_path(), 1)
        data = annotator.get_data()
        # TODO: Improve Test
        self.assertEqual(len(data["intensities"]), len(data["azimuths"]))
        annotator.detach()

    async def test_read_pcl_annotator(self):

        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpu" + "IsaacComputeRTXLidarPointCloud")
        annotator.attach([self._my_lidar.get_render_product_path()])

        self._timeline.play()
        await omni.syntheticdata.sensors.next_render_simulation_async(self._my_lidar.get_render_product_path(), 1)
        data = annotator.get_data()
        # TODO: Improve Test
        self.assertTrue(np.all(np.linalg.norm(data["data"], axis=1) > 0))
        annotator.detach()

    async def test_read_buffer_annotator(self):

        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpu" + "IsaacCreateRTXLidarScanBuffer")
        annotator.attach([self._my_lidar.get_render_product_path()])

        self._timeline.play()
        await omni.syntheticdata.sensors.next_render_simulation_async(self._my_lidar.get_render_product_path(), 2)
        data = annotator.get_data()
        # TODO: Improve Test
        self.assertGreater(len(data["data"]), 144000)  # make sure that the number of points is reasonable
        self.assertTrue(np.all(np.linalg.norm(data["data"], axis=1) > 0))
        annotator.detach()

    async def test_data_acquisition(self):
        await omni.syntheticdata.sensors.next_render_simulation_async(self._my_lidar.get_render_product_path(), 1)
        for annotator in ["linear_depth_data", "point_cloud_data", "intensities_data"]:
            getattr(self._my_lidar, "add_{}_to_frame".format(annotator))()
            await omni.syntheticdata.sensors.next_render_simulation_async(self._my_lidar.get_render_product_path(), 10)
            data = self._my_lidar.get_current_frame()
            self.assertTrue(annotator in data.keys())
            self.assertTrue(data[annotator].shape[0] > 0)
            getattr(self._my_lidar, "remove_{}_from_frame".format(annotator))()
            await omni.syntheticdata.sensors.next_render_simulation_async(self._my_lidar.get_render_product_path(), 1)
            data = self._my_lidar.get_current_frame()
            self.assertTrue(annotator not in data.keys())
        for annotator in ["elevation", "azimuth", "range"]:
            getattr(self._my_lidar, "add_{}_data_to_frame".format(annotator))()
            await omni.syntheticdata.sensors.next_render_simulation_async(self._my_lidar.get_render_product_path(), 10)
            data = self._my_lidar.get_current_frame()
            self.assertTrue(annotator in data.keys())
            self.assertTrue(data[annotator].shape[0] > 0)
            getattr(self._my_lidar, "remove_{}_data_from_frame".format(annotator))()
            await omni.syntheticdata.sensors.next_render_simulation_async(self._my_lidar.get_render_product_path(), 1)
            data = self._my_lidar.get_current_frame()
            self.assertTrue(annotator not in data.keys())

        self.assertTrue(self._my_lidar.get_horizontal_resolution() > 0)
        self.assertTrue(self._my_lidar.get_horizontal_fov() > 0)
        self.assertTrue(self._my_lidar.get_num_rows() > 0)
        self.assertTrue(self._my_lidar.get_num_cols() > 0)
        self.assertTrue(self._my_lidar.get_rotation_frequency() > 0)
        low, high = self._my_lidar.get_depth_range()
        self.assertTrue(low == 1.0)
        self.assertTrue(high > 0)
        low, high = self._my_lidar.get_azimuth_range()
        self.assertTrue(low < 0)
        self.assertTrue(high > 0)
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
        current_time = data["rendering_time"]
        current_step = data["rendering_frame"]
        self._my_lidar.pause()
        await omni.syntheticdata.sensors.next_render_simulation_async(self._my_lidar.get_render_product_path(), 2)
        data = self._my_lidar.get_current_frame()
        self.assertTrue(data["rendering_time"] == current_time)
        self.assertTrue(data["rendering_frame"] == current_step)
        self.assertTrue(self._my_lidar.is_paused())
        current_time = data["rendering_time"]
        current_step = data["rendering_frame"]
        self._my_lidar.resume()
        await update_stage_async()
        data = self._my_lidar.get_current_frame()
        self.assertTrue(data["rendering_time"] != current_time)
        self.assertTrue(data["rendering_frame"] != current_step)
        await self.my_world.reset_async()
        return

    # async def test_get_properties(self):
    #     await omni.syntheticdata.sensors.next_render_simulation_async(self._my_lidar.get_render_product_path(), 10)
    #     self.assertTrue(self._my_lidar.get_horizontal_resolution() > 0)
    #     self.assertTrue(self._my_lidar.get_horizontal_fov() > 0)
    #     self.assertTrue(self._my_lidar.get_num_rows() > 0)
    #     self.assertTrue(self._my_lidar.get_num_cols() > 0)
    #     self.assertTrue(self._my_lidar.get_rotation_frequency() > 0)
    #     low, high = self._my_lidar.get_depth_range()
    #     self.assertTrue(low == 0)
    #     self.assertTrue(high > 0)
    #     low, high = self._my_lidar.get_azimuth_range()
    #     self.assertTrue(low > 0)
    #     self.assertTrue(high > 0)
    #     return
