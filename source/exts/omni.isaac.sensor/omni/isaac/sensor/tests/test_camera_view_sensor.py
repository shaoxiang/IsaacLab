# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import os

import carb
import numpy as np
import omni.kit.test
import omni.replicator.core as rep
import torch
import warp as wp
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils.stage import create_new_stage_async, update_stage_async
from omni.isaac.sensor import CameraView
from PIL import Image


class TestCameraViewSensor(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        await create_new_stage_async()
        self.my_world = World(stage_units_in_meters=1.0)
        await self.my_world.initialize_simulation_context_async()

        # Add a red and blue cube
        self.cube_1 = self.my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_1",
                name="cube_1",
                position=np.array([0.25, 0.25, 0.25]),
                scale=np.array([0.5, 0.5, 0.5]),
                size=1.0,
                color=np.array([255, 0, 0]),
            )
        )
        self.cube_2 = self.my_world.scene.add(
            VisualCuboid(
                prim_path="/new_cube_2",
                name="cube_2",
                position=np.array([-0.25, -0.25, 0.0]),
                scale=np.array([0.5, 0.5, 0.5]),
                size=1.0,
                color=np.array([0, 0, 255]),
            )
        )
        rep.create.plane(scale=(10, 10, 1))

        # All cameras will be looking down the -z axis
        camera_positions = [(0.5, 0, 2), (0, 0.5, 2), (-0.5, 0, 2), (0, -0.5, 2)]
        for pos in camera_positions:
            rep.create.camera(position=pos, look_at=(pos[0], pos[1], 0))
        await omni.kit.app.get_app().next_update_async()
        self.num_cameras = len(camera_positions)

        self.resolution = (256, 256)
        self.camera_view = CameraView(
            prim_paths_expr="/Replicator/Camera_Xform*/Camera",
            name="camera_prim_view",
            camera_resolution=self.resolution,
            output_annotators=["rgb", "depth"],
        )

        await self.my_world.reset_async()
        # Warmup
        for _ in range(5):
            await update_stage_async()

        self.golden_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "golden")
        # self.out_dir = carb.tokens.get_tokens_interface().resolve("${temp}/test_camera_view_sensor")
        # self.out_dir =  self.golden_dir
        # os.makedirs(self.out_dir, exist_ok=True)

        return

    # After running each test
    async def tearDown(self):
        self.camera_view = None
        self.my_world.clear_instance()
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(0.25)
        await omni.kit.app.get_app().next_update_async()
        return

    async def test_get_data(self):
        # res_x * res_y * num_cameras * num_channels (rgb + depth)
        data_length = self.resolution[0] * self.resolution[1] * self.num_cameras * 4
        data_np = self.camera_view.get_data(device="cpu")
        self.assertEqual(len(data_np), data_length)
        self.assertEqual(data_np.dtype, np.float32)

        data_warp = self.camera_view.get_data(device="cuda")
        self.assertEqual(len(data_warp), data_length)
        self.assertEqual(data_warp.dtype, wp.float32)

    async def test_tiled_rgb_data(self):
        # cpu / numpy
        rgb_np_tiled_out = np.zeros((*self.camera_view.tiled_resolution, 3), dtype=np.float32)
        self.camera_view.get_rgb_tiled(out=rgb_np_tiled_out, device="cpu")
        rgb_tiled_np = self.camera_view.get_rgb_tiled(device="cpu")

        self.assertEqual(rgb_np_tiled_out.dtype, rgb_tiled_np.dtype)
        self.assertEqual(rgb_np_tiled_out.shape, rgb_tiled_np.shape)
        self.assertTrue(np.allclose(rgb_np_tiled_out, rgb_tiled_np, atol=1e-5))

        # cuda / torch
        rgb_tiled_torch_out = torch.zeros((*self.camera_view.tiled_resolution, 3), device="cuda", dtype=torch.float32)
        self.camera_view.get_rgb_tiled(out=rgb_tiled_torch_out, device="cuda")
        rgb_tiled_torch = self.camera_view.get_rgb_tiled(device="cuda")

        self.assertEqual(rgb_tiled_torch_out.dtype, rgb_tiled_torch.dtype)
        self.assertEqual(rgb_tiled_torch_out.shape, rgb_tiled_torch.shape)
        self.assertTrue(torch.allclose(rgb_tiled_torch_out, rgb_tiled_torch, atol=1e-5))

        # Compare numpy and torch outputs as normalized uin8 arrays (images)
        A = (rgb_tiled_np * 255).astype(np.uint8)
        B = (rgb_np_tiled_out * 255).astype(np.uint8)
        C = (rgb_tiled_torch.cpu().numpy() * 255).astype(np.uint8)
        D = (rgb_tiled_torch_out.cpu().numpy() * 255).astype(np.uint8)
        self.assertTrue(np.all([np.allclose(A, B, atol=1), np.allclose(A, C, atol=1), np.allclose(A, D, atol=1)]))

    async def test_tiled_depth_data(self):
        # cpu / numpy
        depth_np_tiled_out = np.zeros(self.camera_view.tiled_resolution, dtype=np.float32)
        self.camera_view.get_depth_tiled(out=depth_np_tiled_out, device="cpu")
        depth_tiled_np = self.camera_view.get_depth_tiled(device="cpu")

        self.assertEqual(depth_np_tiled_out.dtype, depth_tiled_np.dtype)
        self.assertEqual(depth_np_tiled_out.shape, depth_tiled_np.shape)
        self.assertTrue(np.allclose(depth_np_tiled_out, depth_tiled_np, atol=1e-5))

        # cuda / torch
        depth_tiled_torch_out = torch.zeros(self.camera_view.tiled_resolution, device="cuda", dtype=torch.float32)
        self.camera_view.get_depth_tiled(out=depth_tiled_torch_out, device="cuda")
        depth_tiled_torch = self.camera_view.get_depth_tiled(device="cuda")

        self.assertEqual(depth_tiled_torch_out.dtype, depth_tiled_torch.dtype)
        self.assertEqual(depth_tiled_torch_out.shape, depth_tiled_torch.shape)
        self.assertTrue(torch.allclose(depth_tiled_torch_out, depth_tiled_torch, atol=1e-5))

        # Compare numpy and torch outputs as normalized uin8 arrays (images)
        A = (depth_tiled_np * 255).astype(np.uint8)
        B = (depth_np_tiled_out * 255).astype(np.uint8)
        C = (depth_tiled_torch.cpu().numpy() * 255).astype(np.uint8)
        D = (depth_tiled_torch_out.cpu().numpy() * 255).astype(np.uint8)
        self.assertTrue(np.all([np.allclose(A, B, atol=1), np.allclose(A, C, atol=1), np.allclose(A, D, atol=1)]))

    async def test_tiled_rgb_image(self):
        rgb_tiled_np = self.camera_view.get_rgb_tiled(device="cpu")
        rgb_tiled_np_uint8 = (rgb_tiled_np * 255).astype(np.uint8)
        rgb_tiled_img = Image.fromarray(rgb_tiled_np_uint8)
        # img_path = os.path.join(self.out_dir, "camera_view_rgb_tiled.png")
        # rgb_tiled_img.save(img_path)
        golden_img_path = os.path.join(self.golden_dir, "camera_view_rgb_tiled.png")
        golden_img = Image.open(golden_img_path)
        self.assertTrue(np.allclose(np.array(rgb_tiled_img), np.array(golden_img), atol=1))

    async def test_tiled_depth_image(self):
        depth_tiled_np = self.camera_view.get_depth_tiled(device="cpu")
        depth_tiled_np_uint8 = (depth_tiled_np * 255).astype(np.uint8)
        depth_tiled_img = Image.fromarray(depth_tiled_np_uint8, mode="L")
        # img_path = os.path.join(self.out_dir, "camera_view_depth_tiled.png")
        # depth_tiled_img.save(img_path)
        golden_img_path = os.path.join(self.golden_dir, "camera_view_depth_tiled.png")
        golden_img = Image.open(golden_img_path)
        self.assertTrue(np.allclose(np.array(depth_tiled_img), np.array(golden_img), atol=1))

    async def test_batched_rgb_data(self):
        rgb_batched_shape = (self.num_cameras, *self.resolution, 3)
        rgb_batched_out = torch.zeros(rgb_batched_shape, device="cuda", dtype=torch.float32)
        self.camera_view.get_rgb(out=rgb_batched_out)
        rgb_batched = self.camera_view.get_rgb()
        self.assertEqual(rgb_batched.dtype, rgb_batched_out.dtype)
        self.assertEqual(rgb_batched.shape, rgb_batched_out.shape)
        self.assertTrue(torch.allclose(rgb_batched, rgb_batched_out, atol=1e-5))

    async def test_batched_depth_data(self):
        depth_batched_shape = (self.num_cameras, *self.resolution, 1)
        depth_batched_out = torch.zeros(depth_batched_shape, device="cuda", dtype=torch.float32)
        self.camera_view.get_depth(out=depth_batched_out)
        depth_batched = self.camera_view.get_depth()
        self.assertEqual(depth_batched.dtype, depth_batched_out.dtype)
        self.assertEqual(depth_batched.shape, depth_batched_out.shape)
        self.assertTrue(torch.allclose(depth_batched, depth_batched_out, atol=1e-5))

    async def test_batched_rgb_images(self):
        rgb_batched = self.camera_view.get_rgb()
        for camera_id in range(rgb_batched.shape[0]):
            rgb_batched_uint8 = (rgb_batched[camera_id] * 255).to(dtype=torch.uint8)
            rgb_batched_img = Image.fromarray(rgb_batched_uint8.cpu().numpy())
            # img_path = os.path.join(self.out_dir, f"camera_view_rgb_batched_{camera_id}.png")
            # rgb_batched_img.save(img_path)
            golden_img_path = os.path.join(self.golden_dir, f"camera_view_rgb_batched_{camera_id}.png")
            golden_img = Image.open(golden_img_path)
            self.assertTrue(np.allclose(np.array(rgb_batched_img), np.array(golden_img), atol=1))

    async def test_batched_depth_images(self):
        depth_batched = self.camera_view.get_depth()
        for camera_id in range(depth_batched.shape[0]):
            depth_batched_uint8 = (depth_batched[camera_id] * 255).squeeze().to(dtype=torch.uint8)
            depth_batched_img = Image.fromarray(depth_batched_uint8.cpu().numpy(), mode="L")
            # img_path = os.path.join(self.out_dir, f"camera_view_depth_batched_{camera_id}.png")
            # depth_batched_img.save(img_path)
            golden_img_path = os.path.join(self.golden_dir, f"camera_view_depth_batched_{camera_id}.png")
            golden_img = Image.open(golden_img_path)
            self.assertTrue(np.allclose(np.array(depth_batched_img), np.array(golden_img), atol=1))

    async def test_properties(self):
        self.assertTrue(self.num_cameras == len(self.camera_view.prims))
        self.camera_view.set_focal_lengths([5.0] * 4)
        self.assertTrue(np.isclose(self.camera_view.get_focal_lengths(), [5.0] * 4, atol=1e-05).all())
        self.camera_view.set_focus_distances([0.01] * 4)
        self.assertTrue(np.isclose(self.camera_view.get_focus_distances(), [0.01] * 4, atol=1e-05).all())
        self.camera_view.set_lens_apertures([0.01] * 4)
        self.assertTrue(np.isclose(self.camera_view.get_lens_apertures(), [0.01] * 4, atol=1e-05).all())
        self.camera_view.set_horizontal_apertures([1.2] * 4)
        self.assertTrue(np.isclose(self.camera_view.get_horizontal_apertures(), [1.2] * 4, atol=1e-05).all())
        self.camera_view.set_vertical_apertures([1.2] * 4)
        self.assertTrue(np.isclose(self.camera_view.get_vertical_apertures(), [1.2] * 4, atol=1e-05).all())
        self.camera_view.set_projection_types(["fisheyeOrthographic"] * 4)
        self.assertTrue(self.camera_view.get_projection_types() == ["fisheyeOrthographic"] * 4)
        return
