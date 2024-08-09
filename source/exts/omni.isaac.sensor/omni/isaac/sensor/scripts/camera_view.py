# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Callable, List, Optional, Sequence, Tuple, Union

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
import torch
import warp as wp
from omni.isaac.core.prims.xform_prim_view import XFormPrimView
from omni.isaac.core.utils.carb import get_carb_setting
from pxr import Usd, Vt

# from ROS camera convention to USD camera convention
U_R_TRANSFORM = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# from USD camera convention to ROS camera convention
R_U_TRANSFORM = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# from USD camera convention to World camera convention
W_U_TRANSFORM = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

# from World camera convention to USD camera convention
U_W_TRANSFORM = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])


@wp.kernel
def reshape_tiled_image(
    tiled_image_buffer: wp.array(dtype=float),
    batched_image: wp.array(dtype=float, ndim=4),
    image_height: int,
    image_width: int,
    num_channels: int,
    num_tiles_x: int,
    offset: int,
):
    """Reshape a tiled image (height*width*num_channels*num_cameras,) to a batch of images (num_cameras, height, width, num_channels).

    Args:
        tiled_image_buffer: The input image buffer. Shape is ((height*width*num_channels*num_cameras,).
        batched_image: The output image. Shape is (num_cameras, height, width, num_channels).
        image_width: The width of the image.
        image_height: The height of the image.
        num_channels: The number of channels in the image.
        num_tiles_x: The number of tiles in x direction.
        offset: The offset in the image buffer. This is used when multiple image types are concatenated in the buffer.
    """
    camera_id, height_id, width_id = wp.tid()
    tile_x_id = camera_id % num_tiles_x
    tile_y_id = camera_id // num_tiles_x
    pixel_start = (
        offset
        + num_channels * num_tiles_x * image_width * (image_height * tile_y_id + height_id)
        + num_channels * tile_x_id * image_width
        + num_channels * width_id
    )
    for i in range(num_channels):
        batched_image[camera_id, height_id, width_id, i] = tiled_image_buffer[pixel_start + i]


class CameraView(XFormPrimView):
    def __init__(
        self,
        prim_paths_expr: str = None,
        name: str = "camera_prim_view",
        camera_resolution: Tuple[int, int] = (256, 256),
        output_annotators: Optional[List[str]] = ["rgb"],
        positions: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        translations: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        scales: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        visibilities: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        reset_xform_properties: bool = True,
    ):
        XFormPrimView.__init__(
            self,
            prim_paths_expr=prim_paths_expr,
            name=name,
            positions=positions,
            translations=translations,
            orientations=orientations,
            scales=scales,
            visibilities=visibilities,
            reset_xform_properties=reset_xform_properties,
        )

        for annotator in output_annotators:
            if annotator not in ["rgb", "depth"]:
                raise ValueError(
                    f"Unsupported annotator: {annotator}. Currently only support 'rgb' and 'depth' annotators for tiled sensor."
                )

        self.output_annotators = output_annotators

        self.camera_resolution = camera_resolution

        self._tiled_render_product = None
        self._tiled_annotator = None
        self._tiled_sensor = None

        self._setup_tiled_sensor()

    def __del__(self):
        self._clean_up_tiled_sensor()

    def _clean_up_tiled_sensor(self):
        """Clean up the sensor by detaching annotators and destroying render products, and removing related prims."""
        if self._tiled_render_product is not None:
            # detach annotators from render product \
            self._tiled_annotator.detach([self._tiled_render_product.path])

            # delete tiled render products
            self._tiled_render_product.destroy()

    def _get_tiled_resolution(self, num_cameras, resolution) -> Tuple[int, int]:
        """Calculate the resolution for the tiled sensor based on the number of cameras and individual camera resolution.

        Args:
            num_cameras (int): Total number of cameras.
            resolution (Tuple[int, int]): Resolution of each individual camera.

        Returns:
            Tuple[int, int]: The total resolution for the tiled sensor layout.
        """
        num_rows = round(num_cameras**0.5)
        num_columns = (num_cameras + num_rows - 1) // num_rows

        return (num_columns * resolution[0], num_rows * resolution[1])

    def _setup_tiled_sensor(self):
        """Set up the tiled sensor, compute resolutions, attach annotators, and initiate the render process."""

        self._clean_up_tiled_sensor()

        self.tiled_resolution = self._get_tiled_resolution(len(self.prims), self.camera_resolution)

        self._tiled_sensor = rep.create.tiled_sensor(
            cameras=self.prim_paths,
            camera_resolution=self.camera_resolution,
            tiled_resolution=self.tiled_resolution,
            output_types=self.output_annotators,
            name=f"{self.name}_tiled_sensor",
        )

        self._tiled_render_product = rep.create.render_product(
            camera=self._tiled_sensor, resolution=self.tiled_resolution
        )

        self._tiled_annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorGpu")
        self._tiled_annotator.attach(self._tiled_render_product)

        return

    def get_render_product_path(self) -> str:
        """Retrieve the file path of the render product associated with the tiled sensor.

        Returns:
            str: The path to the render product, or None if not available.
        """
        if self._tiled_render_product:
            return self._tiled_render_product.path
        else:
            return None

    def set_frequency(self, value: int) -> None:
        """
        Args:
            value (int): sets the frequency to acquire new data frames
        """
        current_rendering_frequency = get_carb_setting(
            carb.settings.get_settings(), "/app/runLoops/main/rateLimitFrequency"
        )
        if current_rendering_frequency is None:
            # Target rendering frequency is not known, processing all frames
            self._frequency = -1
        else:
            if current_rendering_frequency % value != 0:
                raise Exception("frequency of the camera sensor needs to be a divisible by the rendering frequency.")
            self._frequency = value
        return

    def get_frequency(self) -> float:
        """
        Returns:
            float: gets the frequency to acquire new data frames
        """
        return self._frequency

    def set_dt(self, value: float) -> None:
        """
        Args:
            value (float):  sets the dt to acquire new data frames

        """
        current_rendering_frequency = get_carb_setting(
            carb.settings.get_settings(), "/app/runLoops/main/rateLimitFrequency"
        )
        if current_rendering_frequency is None:
            # Target rendering frequency is not known, processing all frames
            self._frequency = -1
        else:
            if value % (1.0 / current_rendering_frequency) != 0:
                raise Exception("dt of the contact sensor needs to be a multiple of the physics frequency.")
            self._frequency = 1.0 / value
        return

    def get_dt(self) -> float:
        """
        Returns:
            float:  gets the dt to acquire new data frames
        """
        return 1.0 / self._frequency

    # TODO: add functionality to pause and resume tiled sensor, similar to camera.py

    def set_resolutions(self, resolution: Tuple[int, int]) -> None:
        """Set the resolutions for all cameras, updating the tiled sensor configuration if changed.

        Args:
            resolution (Tuple[int, int]): The new resolution to apply to all cameras.
        """

        if not resolution == self.camera_resolution:
            self.camera_resolution = resolution
            # update tiled sensor after changing resolution
            self._setup_tiled_sensor()

        return

    def get_resolutions(self) -> Tuple[int, int]:
        """Retrieve the current resolution setting for all cameras.

        Returns:
            Tuple[int, int]: The resolution of the cameras.
        """
        return self.camera_resolution

    def get_aspect_ratios(self) -> float:
        """Calculate the aspect ratio of the cameras based on current resolution settings.

        Returns:
            float: The aspect ratio, defined as width divided by height.
        """
        width, height = self.get_resolutions()
        return float(width) / float(height)

    def _convert_camera_axes(self, orientations, transform_matrix):
        """Convert orientations using the specified transformation matrix.

        Args:
            orientations (Union[np.ndarray, torch.Tensor, wp.array]): Input orientations.
            transform_matrix (np.ndarray): The transformation matrix to apply.

        Returns:
            The converted orientations.
        """
        if isinstance(orientations, np.ndarray):
            orientation_matrices = self._backend_utils.quats_to_rot_matrices(orientations)
            converted_matrices = np.einsum("ij,kjl->kil", transform_matrix[:3, :3], orientation_matrices)
            return self._backend_utils.rot_matrices_to_quats(converted_matrices)
        elif isinstance(orientations, torch.Tensor):
            orientation_matrices = self._backend_utils.quats_to_rot_matrices(orientations)
            converted_matrices = torch.matmul(
                torch.tensor(transform_matrix[:3, :3], dtype=torch.float32), orientation_matrices
            )
            return self._backend_utils.rot_matrices_to_quats(converted_matrices)
        elif isinstance(orientations, wp.array):
            # Assuming similar operations are possible with wp.array
            orientation_matrices = self._backend_utils.quats_to_rot_matrices(orientations)
            converted_matrices = wp.matmul(wp.array(transform_matrix[:3, :3]), orientation_matrices)
            return self._backend_utils.rot_matrices_to_quats(converted_matrices)
        else:
            raise TypeError("Unsupported type for orientations")

    def get_world_poses(
        self,
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
        camera_axes: str = "world",
        usd: bool = True,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[wp.indexedarray, wp.indexedarray]
    ]:
        """Get the poses of the prims in the view with respect to the world's frame

        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).
            usd (bool, optional): True to query from usd. Otherwise False to query from Fabric data. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[wp.indexedarray, wp.indexedarray]]: first index is positions in the world frame of the prims. shape is (M, 3).
                                                                                     second index is quaternion orientations in the world frame of the prims.
                                                                                     quaternion is scalar-first (w, x, y, z). shape is (M, 4).

        Example:

        """
        if camera_axes not in ["world", "ros", "usd"]:
            raise Exception(
                "camera axes passed {} is not supported: accepted values are ["
                "world"
                ", "
                "ros"
                ", "
                "usd"
                "] only".format(camera_axes)
            )

        translations, orientations = XFormPrimView.get_world_poses(self, indices, usd=usd)

        if orientations is not None:
            if camera_axes == "world":
                orientations = self._convert_camera_axes(orientations, U_W_TRANSFORM)
            elif camera_axes == "ros":
                orientations = self._convert_camera_axes(orientations, U_R_TRANSFORM)

        return translations, orientations

    def set_world_poses(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
        camera_axes: str = "world",
        usd: bool = True,
    ) -> None:
        """Set the world poses for all cameras, adjusting their positions and orientations based on specified indices and coordinate system.

        Args:
            positions (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): New positions for the cameras.
            orientations (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): New orientations for the cameras.
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]]): Specific cameras to update.
            camera_axes (str): The coordinate system to use ('world', 'ros', 'usd').
            usd (bool, optional): True to query from usd. Otherwise False to query from Fabric data. Defaults to True.

        Raises:
            Exception: If the provided camera_axes is not supported.
        """
        if camera_axes not in ["world", "ros", "usd"]:
            raise Exception(
                "camera axes passed {} is not supported: accepted values are ["
                "world"
                ", "
                "ros"
                ", "
                "usd"
                "] only".format(camera_axes)
            )

        if orientations is not None:
            if camera_axes == "world":
                orientations = self._convert_camera_axes(orientations, W_U_TRANSFORM)
            elif camera_axes == "ros":
                orientations = self._convert_camera_axes(orientations, R_U_TRANSFORM)

        return XFormPrimView.set_world_poses(self, positions, orientations, indices, usd=usd)

    def get_local_poses(
        self,
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
        camera_axes: str = "world",
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[wp.indexedarray, wp.indexedarray]
    ]:
        """Get prim poses in the view with respect to the local frame (the prim's parent frame)


        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                 to query. Shape (M,).
                                                                                 Where M <= size of the encapsulated prims in the view.
                                                                                 Defaults to None (i.e: all prims in the view).

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor], Tuple[wp.indexedarray, wp.indexedarray]]: first index is positions in the local frame of the prims. shape is (M, 3).
                                                                                     second index is quaternion orientations in the local frame of the prims.
                                                                                     quaternion is scalar-first (w, x, y, z). shape is (M, 4).

        Example:

        """
        if camera_axes not in ["world", "ros", "usd"]:
            raise Exception(
                "camera axes passed {} is not supported: accepted values are ["
                "world"
                ", "
                "ros"
                ", "
                "usd"
                "] only".format(camera_axes)
            )

        translations, orientations = XFormPrimView.get_local_poses(self, indices)

        if orientations is not None:
            if camera_axes == "world":
                orientations = self._convert_camera_axes(orientations, U_W_TRANSFORM)
            elif camera_axes == "ros":
                orientations = self._convert_camera_axes(orientations, U_R_TRANSFORM)

        return translations, orientations

    def set_local_poses(
        self,
        positions: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        orientations: Optional[Union[np.ndarray, torch.Tensor, wp.array]] = None,
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
        camera_axes: str = "world",
    ) -> None:
        """Set the local poses for all cameras, adjusting their positions and orientations based on specified indices and coordinate system.

        Args:
            positions (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): New positions for the cameras.
            orientations (Optional[Union[np.ndarray, torch.Tensor, wp.array]]): New orientations for the cameras.
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]]): Specific cameras to update.
            camera_axes (str): The coordinate system to use ('world', 'ros', 'usd').

        Raises:
            Exception: If the provided camera_axes is not supported.
        """

        if camera_axes not in ["world", "ros", "usd"]:
            raise Exception(
                "camera axes passed {} is not supported: accepted values are ["
                "world"
                ", "
                "ros"
                ", "
                "usd"
                "] only".format(camera_axes)
            )

        if orientations is not None:
            if camera_axes == "world":
                orientations = self._convert_camera_axes(orientations, W_U_TRANSFORM)
            elif camera_axes == "ros":
                orientations = self._convert_camera_axes(orientations, R_U_TRANSFORM)

        return XFormPrimView.set_local_poses(self, positions, orientations, indices)

    def get_rgb_tiled(self, out=None, device="cpu") -> np.ndarray | torch.Tensor:
        """Fetch the RGB data for all cameras as a single tiled image.

        Args:
            device (str, optional): The device to return the data on ("cpu" or "cuda"). Defaults to "cpu".
            out (np.ndarray | torch.Tensor, optional): Pre-allocated array or tensor to fill with the RGBA data.

        Returns:
            np.ndarray | torch.Tensor: containing the RGBA data for each camera. Depth channel is excluded if present.
        """
        if "rgb" not in self.output_annotators:
            if out is None:
                if device == "cuda":
                    out = torch.zeros((*self.tiled_resolution, 3), device=device, dtype=torch.float32)
                else:  # device == "cpu"
                    out = np.zeros((*self.tiled_resolution, 3), dtype=np.float32)
            print("Warning: RGB data is not available. Please enable RGB annotator when creating CameraView object.")
        else:
            linear_sensor_data = self._tiled_annotator.get_data(device=device)
            if isinstance(linear_sensor_data, wp.types.array):
                linear_sensor_data = wp.to_torch(linear_sensor_data).to(device)

            # If depth data is present, get the index where RGB data ends, otherwise use all the data as RGB data
            if "depth" in self.output_annotators:
                rgb_idx = len(linear_sensor_data) * 3 // 4
                rgb_data = linear_sensor_data[:rgb_idx].reshape(*self.tiled_resolution, 3)
            else:
                rgb_data = linear_sensor_data.reshape(*self.tiled_resolution, 3)

            # Check if the data should be copied to the pre-allocated memory
            if out is not None:
                if isinstance(out, np.ndarray):
                    if isinstance(rgb_data, torch.Tensor):
                        out[:] = rgb_data.cpu().numpy()
                    else:
                        out[:] = rgb_data
                elif isinstance(out, torch.Tensor):
                    if isinstance(rgb_data, np.ndarray):
                        rgb_data = torch.from_numpy(rgb_data).to(device)
                    out[:] = rgb_data.to(out.device)
            else:
                out = rgb_data

        return out

    def get_depth_tiled(self, out=None, device="cpu") -> np.ndarray | torch.Tensor:
        """Fetch the depth data for all cameras as a single tiled image.

        Args:
            device (str, optional): The device to return the data on ("cpu" or "cuda"). Defaults to "cpu".
            out (np.ndarray | torch.Tensor, optional): Pre-allocated array or tensor to fill with the depth data.

        Returns:
            np.ndarray | torch.Tensor: containing the depth data for each camera.
        """
        if "depth" not in self.output_annotators:
            if out is None:
                if device == "cuda":
                    out = torch.zeros(self.tiled_resolution, device=device, dtype=torch.float32)
                else:  # device = "cpu"
                    out = np.zeros(self.tiled_resolution, dtype=np.float32)
            print(
                "Warning: Depth data is not available. Please enable depth annotator when creating CameraView object."
            )
        else:
            linear_sensor_data = self._tiled_annotator.get_data(device=device)
            if isinstance(linear_sensor_data, wp.types.array):
                linear_sensor_data = wp.to_torch(linear_sensor_data).to(device)

            # If RGB data is present, depth data starts after the RGB index, otherwise the data only includes depth
            if "rgb" in self.output_annotators:
                rgb_idx = len(linear_sensor_data) * 3 // 4
                depth_data = linear_sensor_data[rgb_idx:].reshape(*self.tiled_resolution)
            else:
                depth_data = linear_sensor_data.reshape(*self.tiled_resolution)

            # Check if the data should be copied to the pre-allocated memory
            if out is not None:
                if isinstance(out, np.ndarray):
                    if isinstance(depth_data, torch.Tensor):
                        out[:] = depth_data.cpu().numpy()
                    else:
                        out[:] = depth_data
                elif isinstance(out, torch.Tensor):
                    if isinstance(depth_data, np.ndarray):
                        depth_data = torch.from_numpy(depth_data).to(device)
                    out[:] = depth_data.to(out.device)
            else:
                out = depth_data

        return out

    def get_data(self, device="cpu") -> np.ndarray | wp.types.array:
        """Get the tiled annotator data directly.

        Args:
            device (str, optional): The device to return the data on ("cpu" or "cuda"). Defaults to "cpu".

        Returns:
            np.ndarray | wp.types.array: The data from the tiled annotator.
        """
        return self._tiled_annotator.get_data(device=device)

    def get_rgb(self, out=None) -> torch.Tensor:
        """Get the RGB data for all cameras as a batch of images (num_cameras, height, width, num_channels=3).

        Returns:
            torch.Tensor: containing the RGB data for each camera.
            Shape is (num_cameras, height, width, num_channels=3) with type torch.float32.

        """
        # Check if the data should be copied to the pre-allocated memory
        if out is None:
            out_shape = (len(self.prims), *self.camera_resolution, 3)
            out = torch.zeros(out_shape, device="cuda", dtype=torch.float32).contiguous()

        # Get the linear sensor data from the tiled annotator and (if needed) slice it to get only the RGB data
        linear_sensor_data = self._tiled_annotator.get_data(device="cuda")
        if "depth" in self.output_annotators:
            rgb_offset = len(linear_sensor_data) * 3 // 4
            linear_sensor_data = linear_sensor_data[:rgb_offset]

        # Use a warp kernel to convert the linear sensor data to a batch of images
        img_height, img_width = self.camera_resolution
        num_channels = 3
        num_tiles_x = self.tiled_resolution[0] // img_width
        offset = 0  # always 0 since we sliced the data above (if depth data is present)
        wp.launch(
            kernel=reshape_tiled_image,
            dim=(len(self.prims), *self.camera_resolution),
            inputs=[
                linear_sensor_data,
                wp.from_torch(out),
                img_height,
                img_width,
                num_channels,
                num_tiles_x,
                offset,
            ],
            device="cuda",
        )
        return out

    def get_depth(self, out=None) -> torch.Tensor:
        """Get the depth data for all cameras as a batch of images (num_cameras, height, width, num_channels=1).

        Returns:
            torch.Tensor: containing the depth data for each camera.
            Shape is (num_cameras, height, width, num_channels=1) with type torch.float32.

        """
        # Check if the data should be copied to the pre-allocated memory
        if out is None:
            out_shape = (len(self.prims), *self.camera_resolution, 1)
            out = torch.zeros(out_shape, device="cuda", dtype=torch.float32).contiguous()

        # Get the linear sensor data from the tiled annotator and (if needed) slice it to get only the depth data
        linear_sensor_data = self._tiled_annotator.get_data(device="cuda")
        if "rgb" in self.output_annotators:
            depth_offset = len(linear_sensor_data) * 3 // 4
            linear_sensor_data = linear_sensor_data[depth_offset:]

        # Use a warp kernel to convert the linear sensor data to a batch of images
        img_height, img_width = self.camera_resolution
        num_channels = 1
        num_tiles_x = self.tiled_resolution[0] // img_width
        offset = 0  # always 0 since we sliced the data above (if RGB data is present)
        wp.launch(
            kernel=reshape_tiled_image,
            dim=(len(self.prims), *self.camera_resolution),
            inputs=[
                linear_sensor_data,
                wp.from_torch(out),
                img_height,
                img_width,
                num_channels,
                num_tiles_x,
                offset,
            ],
            device="cuda",
        )
        return out

    def get_focal_lengths(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> List[float]:
        """Get the focal length for all cameras
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).

        Returns:
            list[float]: list containing the focal lengths of the cameras.
        """
        focal_lengths = []
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        for i in indices:
            focal_lengths.append(self.prims[i].GetAttribute("focalLength").Get() / 10.0)

        return focal_lengths

    def set_focal_lengths(
        self,
        values: List[float],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the focal length for cameras specific in indices. If indices is None, set for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).
            values (List[float]): list containing the focal lengths to set for the cameras.
                Length of values must match length of indices.
        """

        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        # Ensure length of values matches indices
        if len(indices) != len(values):
            raise ValueError("Length of values must match length of indices")

        for i in indices:
            self.prims[i].GetAttribute("focalLength").Set(values[i] * 10.0)

        return

    def get_focus_distances(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> List[float]:
        """Get the focus distances for cameras specific in indices. If indices is None, get for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).

        Returns:
            list[float]: list containing the focal distances of the cameras.
        """
        focus_distances = []
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        for i in indices:
            focus_distances.append(self.prims[i].GetAttribute("focusDistance").Get())

        return focus_distances

    def set_focus_distances(
        self,
        values: List[float],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the focus distance for cameras specific in indices. If indices is None, set for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to set. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).
            values (List[float]): list containing the focus distances to set for the cameras.
                Length of values must match length of indices.
        """

        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        # Ensure length of values matches indices
        if len(indices) != len(values):
            raise ValueError("Length of values must match length of indices")

        for i in indices:
            self.prims[i].GetAttribute("focusDistance").Set(values[i])

        return

    def get_lens_apertures(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> List[float]:
        """Get the lens apertures for cameras specific in indices. If indices is None, get for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).

        Returns:
            list[float]: list containing the focal distances of the cameras.
        """
        lens_apertures = []
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        for i in indices:
            lens_apertures.append(self.prims[i].GetAttribute("fStop").Get())

        return lens_apertures

    def set_lens_apertures(
        self,
        values: List[float],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the lens apertures for cameras specific in indices. If indices is None, set for all cameras.
        Controls Distance Blurring. Lower Numbers decrease focus range, larger
            numbers increase it.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to set. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).
            values (List[float]): list containing the lens apertures to set for the cameras.
                Length of values must match length of indices.
        """

        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        # Ensure length of values matches indices
        if len(indices) != len(values):
            raise ValueError("Length of values must match length of indices")

        for i in indices:
            self.prims[i].GetAttribute("fStop").Set(values[i])

        return

    def get_horizontal_apertures(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> List[float]:
        """Get the horizontal apertures for cameras specific in indices. If indices is None, get for all cameras.
        Emulates sensor/film width on a camera.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).

        Returns:
            list[float]: list containing the focal distances of the cameras.
        """
        horizontal_apertures = []
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        for i in indices:
            horizontal_apertures.append(self.prims[i].GetAttribute("horizontalAperture").Get())

        return horizontal_apertures

    def set_horizontal_apertures(
        self,
        values: List[float],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the horizontal apertures for cameras specific in indices. If indices is None, set for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to set. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).
            values (List[float]): list containing the horizontal apertures to set for the cameras.
                Length of values must match length of indices.
        """

        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        # Ensure length of values matches indices
        if len(indices) != len(values):
            raise ValueError("Length of values must match length of indices")

        for i in indices:
            self.prims[i].GetAttribute("horizontalAperture").Set(values[i])

        return

    def get_vertical_apertures(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> List[float]:
        """Get the vertical apertures for cameras specific in indices. If indices is None, get for all cameras.
        Emulates sensor/film height on a camera.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).

        Returns:
            list[float]: list containing the focal distances of the cameras.
        """
        vertical_apertures = []
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        for i in indices:
            vertical_apertures.append(self.prims[i].GetAttribute("verticalAperture").Get())

        return vertical_apertures

    def set_vertical_apertures(
        self,
        values: List[float],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the vertical apertures for cameras specific in indices. If indices is None, set for all cameras.
        Emulates sensor/film height on a camera.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to set. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).
            values (List[float]): list containing the vertical apertures to set for the cameras.
                Length of values must match length of indices.
        """

        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        # Ensure length of values matches indices
        if len(indices) != len(values):
            raise ValueError("Length of values must match length of indices")

        for i in indices:
            self.prims[i].GetAttribute("verticalAperture").Set(values[i])

        return

    def get_projection_types(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> List[str]:
        """Get the projection types for cameras specific in indices. If indices is None, get for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).

        Returns:
            list[str]: list of projection types (pinhole, fisheyeOrthographic, fisheyeEquidistant, fisheyeEquisolid, fisheyePolynomial or fisheyeSpherical)
        """
        projection_types = []
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        for i in indices:
            projection_type = self.prims[i].GetAttribute("cameraProjectionType").Get()
            if projection_type is None:
                projection_type = "pinhole"
            projection_types.append(projection_type)

        return projection_types

    def set_projection_types(
        self,
        values: List[str],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the projection types for cameras specific in indices. If indices is None, set for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to set. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).
            values (List[str]): list of projection types (pinhole, fisheyeOrthographic, fisheyeEquidistant, fisheyeEquisolid, fisheyePolynomial or fisheyeSpherical)
                Length of values must match length of indices.
        """

        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        # Ensure length of values matches indices
        if len(indices) != len(values):
            raise ValueError("Length of values must match length of indices")

        for i in indices:
            self.prims[i].GetAttribute("cameraProjectionType").Set(Vt.Token(values[i]))

        return

    def get_projection_modes(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> List[str]:
        """Get the projection modes for cameras specific in indices. If indices is None, get for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).

        Returns:
            list[str]: list of projection modes (perspective, orthographic)
        """
        projection_modes = []
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        for i in indices:
            projection_modes.append(self.prims[i].GetAttribute("projection").Get())

        return projection_modes

    def set_projection_modes(
        self,
        values: List[str],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the projection modes for cameras specific in indices. If indices is None, set for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to set. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).
            values (List[str]): list of projection modes (perspective, orthographic)
                Length of values must match length of indices.
        """

        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        # Ensure length of values matches indices
        if len(indices) != len(values):
            raise ValueError("Length of values must match length of indices")

        for i in indices:
            self.prims[i].GetAttribute("projection").Set(values[i])

        return

    def get_stereo_roles(self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None) -> List[str]:
        """Get the stereo roles for cameras specific in indices. If indices is None, get for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).

        Returns:
            list[str]: list of stereo roles (mono, left, right)
        """
        stereo_roles = []
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        for i in indices:
            stereo_roles.append(self.prims[i].GetAttribute("stereoRole").Get())

        return stereo_roles

    def set_stereo_roles(
        self,
        values: List[str],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the stereo roles for cameras specific in indices. If indices is None, set for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to set. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).
            values (List[str]): list of stereo roles (mono, left, right)
                Length of values must match length of indices.
        """

        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        # Ensure length of values matches indices
        if len(indices) != len(values):
            raise ValueError("Length of values must match length of indices")

        for i in indices:
            self.prims[i].GetAttribute("stereoRole").Set(values[i])

        return

    def get_shutter_properties(
        self, indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None
    ) -> List[Tuple[float, float]]:
        """Get the (delay_open, delay_close) of shutter for cameras specific in indices. If indices is None, get for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to query. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).

        Returns:
            list[Tuple[float, float]]: list of tuple (delay_open, delay_close)
        """
        shutter_properties = []
        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        for i in indices:
            shutter_properties.append(
                (self.prims[i].GetAttribute("shutter:open").Get(), self.prims[i].GetAttribute("shutter:close").Get())
            )

        return shutter_properties

    def set_shutter_properties(
        self,
        values: List[Tuple[float, float]],
        indices: Optional[Union[np.ndarray, list, torch.Tensor, wp.array]] = None,
    ) -> None:
        """Set the (delay_open, delay_close) of shutter for cameras specific in indices. If indices is None, set for all cameras.
        Args:
            indices (Optional[Union[np.ndarray, list, torch.Tensor, wp.array]], optional): indices to specify which prims
                                                                                    to set. Shape (M,).
                                                                                    Where M <= size of the encapsulated prims in the view.
                                                                                    Defaults to None (i.e: all prims in the view).
            values (List[Tuple[float, float]]): list of tuple (delay_open, delay_close)
                Length of values must match length of indices.
        """

        indices = self._backend_utils.resolve_indices(indices, self.count, self._device)
        indices = self._backend_utils.to_list(indices)

        # Ensure length of values matches indices
        if len(indices) != len(values):
            raise ValueError("Length of values must match length of indices")

        for i in indices:
            delay_open, delay_close = values[i]

            if delay_open:
                self.prims[i].GetAttribute("shutter:open").Set(delay_open)
            if delay_close:
                self.prims[i].GetAttribute("shutter:close").Set(delay_close)

        return
