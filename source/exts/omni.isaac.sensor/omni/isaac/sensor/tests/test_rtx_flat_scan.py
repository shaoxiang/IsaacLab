# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio

import omni.kit.test
import omni.replicator.core as rep
from omni.isaac.core.utils import stage
from pxr import Gf, UsdGeom


def add_cube(stage, path, scale, offset, physics=False):
    cubeGeom = UsdGeom.Cube.Define(stage, path)
    cubePrim = stage.GetPrimAtPath(path)
    cubeGeom.CreateSizeAttr(1.0)
    cubeGeom.AddTranslateOp().Set(offset)
    cubeGeom.AddScaleOp().Set(scale)

    return cubePrim


class TestRtxFlatScan(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        await stage.create_new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        add_cube(stage.get_current_stage(), "/World/cxube_x1", (4, 4, 4), (0, 0, 0), physics=False)
        await omni.kit.app.get_app().next_update_async()
        omni.timeline.get_timeline_interface().play()
        await omni.kit.app.get_app().next_update_async()
        return

    # After running each test
    async def tearDown(self):
        return

    async def test_solid_state_flat_scan(self):
        _, sensor_sstate = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/sensor_sstate",
            parent=None,
            config="Simple_Example_Solid_State",
            translation=(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),  # Gf.Quatd is w,i,j,k
        )
        await omni.kit.app.get_app().next_update_async()

        hydra_texture_sstate = rep.create.render_product(sensor_sstate.GetPath(), [1, 1], name="Isaac")
        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpu" + "IsaacComputeRTXLidarFlatScan")
        annotator.attach([hydra_texture_sstate])

        for i in range(10):
            await omni.kit.app.get_app().next_update_async()
        data = annotator.get_data()
        # The output of this test should match the parameters in Simple_Example_Solid_State with all the possible
        # rays shot from the 0 elevation row being hits on the sphere.
        self.assertAlmostEqual(data["horizontalFov"], 3)
        self.assertAlmostEqual(data["horizontalResolution"], 1)
        self.assertAlmostEqual(data["depthRange"][0], 1.0)
        self.assertAlmostEqual(data["depthRange"][1], 200.0)
        self.assertAlmostEqual(data["rotationRate"], 30)
        self.assertEqual(len(data["linearDepthData"]), 4)
        # should be at center of 4 m cube
        for depth in data["linearDepthData"]:
            if depth < 0.0:
                continue  # invalid scan
            self.assertAlmostEqual(depth, 2.0, places=2)
        self.assertEqual(len(data["intensitiesData"]), 4)
        self.assertEqual(data["numRows"], 1)
        self.assertEqual(data["numCols"], 4)
        self.assertAlmostEqual(data["horizontalResolution"], 1)
        self.assertAlmostEqual(data["azimuthRange"][0], -1.5)
        self.assertAlmostEqual(data["azimuthRange"][1], 1.5)
        annotator.detach()

    async def test_rotary_flat_scan(self):

        _, sensor_rotary = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path="/sensor_rotary",
            parent=None,
            config="Debug_Rotary",
            translation=(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),  # Gf.Quatd is w,i,j,k
        )
        await omni.kit.app.get_app().next_update_async()

        hydra_texture_rotary = rep.create.render_product(sensor_rotary.GetPath(), [1, 1], name="Isaac")
        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpu" + "IsaacComputeRTXLidarFlatScan")
        annotator.attach([hydra_texture_rotary])

        for i in range(20):
            await omni.kit.app.get_app().next_update_async()
        data = annotator.get_data()
        # The output of this test should match the parameters in Debug_Rotary with all the possible
        # rays shot from the 0 elevation row being hits on the sphere.
        self.assertAlmostEqual(data["horizontalFov"], 360)
        self.assertAlmostEqual(data["horizontalResolution"], 10)
        self.assertAlmostEqual(data["depthRange"][0], 0.05)
        self.assertAlmostEqual(data["depthRange"][1], 200.0)
        self.assertAlmostEqual(data["rotationRate"], 6)
        self.assertEqual(len(data["linearDepthData"]), 36)
        # should be at center of 4 m cube
        for depth in data["linearDepthData"]:
            if depth < 0.0:
                continue  # invalid scan
            self.assertLessEqual(depth, 2.83)
            self.assertGreaterEqual(depth, 1.99)
        self.assertEqual(len(data["intensitiesData"]), 36)
        self.assertEqual(data["numRows"], 1)
        self.assertEqual(data["numCols"], 36)
        self.assertAlmostEqual(data["azimuthRange"][0], -180.0)
        self.assertAlmostEqual(data["azimuthRange"][1], 180.0)
        annotator.detach()
