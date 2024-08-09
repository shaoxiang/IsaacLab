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
import math
from typing import List

import carb.tokens
import numpy as np
import omni.kit.commands
import omni.kit.test
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.physics import simulate_async
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.utils.stage import clear_stage
from omni.isaac.core.utils.transformations import get_relative_transform
from omni.isaac.nucleus import get_assets_root_path_async

# Import extension python module we are testing with absolute import path, as if we are external user (other extension)
from omni.isaac.sensor import _sensor
from pxr import Gf, UsdGeom


# Having a test class dervived from omni.kit.test.AsyncTestCase declared on the root of module will make it auto-discoverable by omni.kit.test
class TestIMUSensor(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        self._sensor_rate = 60
        self._is = _sensor.acquire_imu_sensor_interface()
        self._assets_root_path = await get_assets_root_path_async()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        self.my_world = None
        pass

    async def createAnt(self, physics_rate=60):
        self.leg_paths = ["/Ant/Arm_{:02d}/Lower_Arm".format(i + 1) for i in range(4)]
        self.sphere_path = "/Ant/Sphere"
        self.sensor_offsets = [
            Gf.Vec3d(0, 0, 0),
            Gf.Vec3d(0, 0, 0),
            Gf.Vec3d(0, 0, 0),
            Gf.Vec3d(0, 0, 0),
            Gf.Vec3d(0, 0, 0),
        ]

        self.sensor_quatd = [
            Gf.Quatd(1, 0, 0, 0),
            Gf.Quatd(1, 0, 0, 0),
            Gf.Quatd(1, 0, 0, 0),
            Gf.Quatd(1, 0, 0, 0),
            Gf.Quatd(1, 0, 0, 0),
        ]

        self.shoulder_joints = ["/Ant/Arm_{:02d}/Upper_Arm/shoulder_joint".format(i + 1) for i in range(4)]

        self.lower_joints = ["{}/lower_arm_joint".format(i) for i in self.leg_paths]

        await omni.usd.get_context().open_stage_async(self._assets_root_path + "/Isaac/Robots/Simple/ant.usd")
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()

        self.my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / physics_rate, rendering_dt=1.0 / physics_rate)
        await self.my_world.initialize_simulation_context_async()
        self.ant = XFormPrim("/Ant")
        pass

    async def createSimpleArticulation(self, physics_rate=60):

        self.pivot_path = "/Articulation/CenterPivot"
        self.slider_path = "/Articulation/Slider"
        self.arm_path = "/Articulation/Arm"

        # load nucleus asset
        await omni.usd.get_context().open_stage_async(
            self._assets_root_path + "/Isaac/Robots/Simple/simple_articulation.usd"
        )

        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()
        self.my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / physics_rate, rendering_dt=1.0 / physics_rate)
        await self.my_world.initialize_simulation_context_async()

        pass

    # After running each test
    async def tearDown(self):
        if self.my_world:
            self.my_world.stop()
            self.my_world.clear_instance()
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            # print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()
        pass

    async def test_add_sensor_prim(self):
        await self.createAnt()
        self.sensorGeoms = []
        for i in range(4):
            await omni.kit.app.get_app().next_update_async()
            result, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateImuSensor",
                path="/sensor",
                parent=self.leg_paths[i],
                sensor_period=1 / self._sensor_rate,
                translation=self.sensor_offsets[i],
                orientation=self.sensor_quatd[i],
            )
            self.sensorGeoms.append(sensor)
            self.assertTrue(result)
            self.assertIsNotNone(sensor)
            # Add sensor on body sphere
            await omni.kit.app.get_app().next_update_async()
            result, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateImuSensor",
                path="/sensor",
                parent=self.sphere_path,
                sensor_period=1 / self._sensor_rate,
                translation=self.sensor_offsets[4],
                orientation=self.sensor_quatd[4],
            )
            self.assertTrue(result)
            self.sensorGeoms.append(sensor)
            self.assertIsNotNone(sensor)
        pass

    async def test_orientation_imu(self):
        await self.createSimpleArticulation()

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/arm_imu",
            parent=self.arm_path,
            sensor_period=1 / self._sensor_rate,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        self.my_world.play()

        await omni.kit.app.get_app().next_update_async()

        art = Articulation("/Articulation")
        art.initialize()
        await omni.kit.app.get_app().next_update_async()
        num_dofs = art.num_dof
        art._articulation_view.set_gains(kps=np.ones(num_dofs) * 1e8, kds=np.ones(num_dofs) * 1e8)

        ang = 0
        for i in range(70):
            art.set_joint_positions(np.array([math.radians(ang), 0.5]))

            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()

            orientation = quat_to_euler_angles(
                np.array(self._is.get_sensor_reading(self.arm_path + "/arm_imu").orientation), True
            )[0]

            angtest = ang % 360
            if ang >= 180:
                angtest = ang - 360

            self.assertAlmostEqual(orientation, angtest, delta=1e-1)
            ang += 5

        pass

    async def test_ang_vel_imu(self):
        await self.createSimpleArticulation()

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/slider_imu",
            parent=self.slider_path,
            sensor_period=1 / self._sensor_rate,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        self.my_world.play()

        await omni.kit.app.get_app().next_update_async()

        art = Articulation("/Articulation")
        art.initialize()
        await omni.kit.app.get_app().next_update_async()
        num_dofs = art.num_dof

        ang_vel_l = [x * 30 for x in range(0, 20)]

        for x in ang_vel_l:
            art.set_joint_velocities(np.array([math.radians(x), 0]))

            await omni.kit.app.get_app().next_update_async()
            ang_vel_z = self._is.get_sensor_reading(self.slider_path + "/slider_imu").ang_vel_z
            # with sensor frequency = physics rate, all should be the same
            self.assertAlmostEqual(ang_vel_z, math.radians(x), delta=2.0e-1)

            art.set_joint_positions(np.array([0, 0]))
            art.set_joint_velocities(np.array([0, 0]))
            art.set_joint_efforts(np.array([0, 0]))
        pass

    async def test_lin_acc_imu(self):
        await self.createSimpleArticulation()

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/slider_imu",
            parent=self.slider_path,
            sensor_period=1 / self._sensor_rate,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
            linear_acceleration_filter_size=10,
            angular_velocity_filter_size=10,
            orientation_filter_size=10,
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        # await self.test_add_arm_imu()
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/arm_imu",
            parent=self.arm_path,
            sensor_period=1 / 60,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
            linear_acceleration_filter_size=10,
            angular_velocity_filter_size=10,
            orientation_filter_size=10,
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        self.my_world.play()

        await omni.kit.app.get_app().next_update_async()
        art = Articulation("/Articulation")
        art.initialize()
        await omni.kit.app.get_app().next_update_async()
        num_dofs = art.num_dof
        art._articulation_view.set_gains(kps=np.zeros(num_dofs), kds=np.ones(num_dofs))

        x = 0
        for i in range(60):

            art.set_joint_efforts(np.array([math.radians(x), 0]))
            await omni.kit.app.get_app().next_update_async()
            slider_mag = np.linalg.norm(
                [
                    self._is.get_sensor_reading(self.slider_path + "/slider_imu").lin_acc_x,
                    self._is.get_sensor_reading(self.slider_path + "/slider_imu").lin_acc_y,
                ]
            )
            arm_mag = np.linalg.norm(
                [
                    self._is.get_sensor_reading(self.arm_path + "/arm_imu").lin_acc_x,
                    self._is.get_sensor_reading(self.arm_path + "/arm_imu").lin_acc_y,
                ]
            )
            self.assertGreaterEqual(slider_mag, arm_mag)

            x += 1000

        pass

    async def test_gravity_m(self):
        await self.test_add_sensor_prim()
        self.ant.set_world_pose([0, 0, 1])
        UsdGeom.SetStageMetersPerUnit(self._stage, 1.0)

        await omni.kit.app.get_app().next_update_async()

        self.my_world.play()
        for i in range(20):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/sensor")
            sensor_reading_no_gravity = self._is.get_sensor_reading(
                self.sphere_path + "/sensor", use_latest_data=True, read_gravity=False
            )
            # # print(sensor_reading.lin_acc_x, "\t", sensor_reading.lin_acc_y, "\t", sensor_reading.lin_acc_z)
        self.assertAlmostEqual(sensor_reading.lin_acc_z, 0, delta=0.1)
        self.assertAlmostEqual(sensor_reading_no_gravity.lin_acc_z, -9.81, delta=0.1)

        for i in range(100):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/sensor")
            sensor_reading_no_gravity = self._is.get_sensor_reading(
                self.sphere_path + "/sensor", use_latest_data=True, read_gravity=False
            )
            # # print(sensor_reading.lin_acc_x, "\t", sensor_reading.lin_acc_y, "\t", sensor_reading.lin_acc_z)
        self.assertAlmostEqual(sensor_reading.lin_acc_z, 9.81, delta=0.1)
        self.assertAlmostEqual(sensor_reading_no_gravity.lin_acc_z, 0, delta=0.1)

        pass

    async def test_gravity_moon_m(self):
        await self.test_add_sensor_prim()
        self.ant.set_world_pose([0, 0, 1])
        self.my_world.get_physics_context().set_gravity(-1.62)
        UsdGeom.SetStageMetersPerUnit(self._stage, 1.0)

        await omni.kit.app.get_app().next_update_async()

        self.my_world.play()
        for i in range(20):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/sensor", use_latest_data=True)
            # # print(sensor_reading.lin_acc_x, "\t", sensor_reading.lin_acc_y, "\t", sensor_reading.lin_acc_z)
        self.assertAlmostEqual(sensor_reading.lin_acc_z, 0, delta=0.1)
        for i in range(200):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/sensor", use_latest_data=True)
            # # print(sensor_reading.lin_acc_x, "\t", sensor_reading.lin_acc_y, "\t", sensor_reading.lin_acc_z)
        self.assertAlmostEqual(sensor_reading.lin_acc_z, 1.62, delta=0.1)
        pass

    async def test_gravity_cm(self):
        await self.createAnt()
        await self.test_add_sensor_prim()

        UsdGeom.SetStageMetersPerUnit(self._stage, 0.01)

        await omni.kit.app.get_app().next_update_async()

        await omni.kit.app.get_app().next_update_async()
        self.my_world.play()
        for i in range(100):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/sensor", use_latest_data=True)
            # # print(sensor_reading.lin_acc_x, "\t", sensor_reading.lin_acc_y, "\t", sensor_reading.lin_acc_z)
        self.assertAlmostEqual(sensor_reading.lin_acc_z, 981, delta=0.1)

    pass

    async def test_stop_start(self):
        await self.test_add_sensor_prim()

        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        self.my_world.play()

        first = True
        for i in range(200):
            await omni.kit.app.get_app().next_update_async()

            sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/sensor", use_latest_data=True)
            if first:
                init_reading = sensor_reading
                first = False

            rigid_body = RigidPrim(self.sphere_path)
            rigid_body.initialize()
            rigid_body._rigid_prim_view.apply_forces_and_torques_at_pos(
                forces=np.array([10, 10, 10]), positions=np.array([10, 10, 10]), is_global=True
            )

        self.my_world.stop()
        await omni.kit.app.get_app().next_update_async()

        self.my_world.play()
        await omni.kit.app.get_app().next_update_async()
        sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/sensor", use_latest_data=True)

        self.assertEqual(sensor_reading.lin_acc_x, init_reading.lin_acc_x)
        self.assertEqual(sensor_reading.lin_acc_y, init_reading.lin_acc_y)
        self.assertEqual(sensor_reading.lin_acc_z, init_reading.lin_acc_z)

    pass

    async def test_no_physics_scene(self):
        await omni.usd.get_context().open_stage_async(
            self._assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
        )
        await omni.kit.app.get_app().next_update_async()
        self._stage = omni.usd.get_context().get_stage()
        await omni.kit.app.get_app().next_update_async()
        cube_path = "/new_cube"
        DynamicCuboid(prim_path=cube_path, name="cube_1", position=np.array([0, 0, 2]), color=np.array([255, 0, 0]))

        await omni.kit.app.get_app().next_update_async()
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/sensor",
            parent=cube_path,
        )

        await omni.kit.app.get_app().next_update_async()

        omni.timeline.get_timeline_interface().play()
        for i in range(20):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._is.get_sensor_reading(cube_path + "/sensor", use_latest_data=True)
        self.assertAlmostEqual(sensor_reading.lin_acc_z, 0, delta=0.1)
        for i in range(100):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._is.get_sensor_reading(cube_path + "/sensor", use_latest_data=True)
        self.assertAlmostEqual(sensor_reading.lin_acc_z, 9.81, delta=0.1)
        omni.timeline.get_timeline_interface().stop()
        pass

    async def test_rolling_average_attributes(self):
        await self.createAnt(physics_rate=400)
        await omni.kit.app.get_app().next_update_async()

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/sphere_imu_1",
            parent=self.sphere_path,
            sensor_period=1 / self._sensor_rate,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
            linear_acceleration_filter_size=1,
            angular_velocity_filter_size=1,
            orientation_filter_size=1,
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        low_rolling_avg_size_reading = np.zeros(
            6,
        )
        high_rolling_avg_size_reading = np.zeros(
            6,
        )

        self.my_world.play()
        # wait for the ant to settle down
        for i in range(200):
            await omni.kit.app.get_app().next_update_async()

        # test 1, when the rolling average is 1, should expect larger fluctuations
        for i in range(50):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/sphere_imu_1", use_latest_data=True)

            readings = np.array(
                [
                    sensor_reading.lin_acc_x,
                    sensor_reading.lin_acc_y,
                    sensor_reading.lin_acc_z,
                    sensor_reading.ang_vel_x,
                    sensor_reading.ang_vel_y,
                    sensor_reading.ang_vel_z,
                ]
            )

            low_rolling_avg_size_reading = np.vstack((low_rolling_avg_size_reading, readings))

        low_rolling_avg_size_1th_percentile = np.percentile(low_rolling_avg_size_reading, 1, axis=0)
        low_rolling_avg_size_99th_percentile = np.percentile(low_rolling_avg_size_reading, 99, axis=0)

        low_rolling_avg_size_diff = np.subtract(
            low_rolling_avg_size_99th_percentile, low_rolling_avg_size_1th_percentile
        )

        # test 2, when the rolling average is 20, should expect lower fluctuations
        sensor.CreateLinearAccelerationFilterWidthAttr().Set(20)
        sensor.CreateAngularVelocityFilterWidthAttr().Set(20)
        sensor.CreateOrientationFilterWidthAttr().Set(20)

        await omni.kit.app.get_app().next_update_async()

        for i in range(50):
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/sphere_imu_1", use_latest_data=True)

            readings = np.array(
                [
                    sensor_reading.lin_acc_x,
                    sensor_reading.lin_acc_y,
                    sensor_reading.lin_acc_z,
                    sensor_reading.ang_vel_x,
                    sensor_reading.ang_vel_y,
                    sensor_reading.ang_vel_z,
                ]
            )

            high_rolling_avg_size_reading = np.vstack((high_rolling_avg_size_reading, readings))

        high_rolling_avg_size_1th_percentile = np.percentile(high_rolling_avg_size_reading, 1, axis=0)
        high_rolling_avg_size_99th_percentile = np.percentile(high_rolling_avg_size_reading, 99, axis=0)

        high_rolling_avg_size_diff = np.subtract(
            high_rolling_avg_size_99th_percentile, high_rolling_avg_size_1th_percentile
        )

        # low rolling average size is expected to have larger variation than with high rolling average size
        for i in range(len(high_rolling_avg_size_diff)):
            self.assertGreater(low_rolling_avg_size_diff[i], high_rolling_avg_size_diff[i])

    async def test_sensor_period(self):
        await self.test_add_sensor_prim()
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/custom_sensor",
            parent=self.sphere_path,
            sensor_period=1 / (self._sensor_rate / 2),  # 30hz, half of physics rate
            translation=self.sensor_offsets[4],
            orientation=self.sensor_quatd[4],
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        await omni.kit.app.get_app().next_update_async()
        self.my_world.play()
        await omni.kit.app.get_app().next_update_async()
        readings = []

        for i in range(60):  # Simulate for one second
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/custom_sensor", None, False)

            # the sensor is running at 30hz, while the sim is 60hz, so expecting 1/3 readings to be new,
            # the other reading should be identical and have the same timestamp
            if not readings or readings[-1].time != sensor_reading.time:
                readings.append(sensor_reading)
                # print(sensor_reading.time)

        # tolerance +-1 reading (29,30,31) will be accepted)
        # print(len(readings))
        self.assertTrue(abs(len(readings) - 30) <= 1)
        pass

    # use a custom function that sets the values to -100. Verify that the readings are -100 as a result of the function
    # verify that the measurement time are the same as not using the custom function
    # verify that the measuremnt using and not using the custom function are not the same
    async def test_custom_interpolation_function(self):
        def custom_function(sensorReadings: List[_sensor.IsSensorReading], time: float) -> _sensor.IsSensorReading:
            override_sensor_reading = _sensor.IsSensorReading()
            override_sensor_reading.lin_acc_x = -100
            override_sensor_reading.ang_vel_x = -100
            override_sensor_reading.time = time
            return override_sensor_reading

        await self.test_add_sensor_prim()
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/custom_sensor",
            parent=self.sphere_path,
            sensor_period=1 / (self._sensor_rate / 2),  # 30hz, half of physics rate
            translation=self.sensor_offsets[4],
            orientation=self.sensor_quatd[4],
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        await omni.kit.app.get_app().next_update_async()
        self.my_world.play()
        await omni.kit.app.get_app().next_update_async()

        for i in range(10):  # Simulate 10 steps
            await omni.kit.app.get_app().next_update_async()

            # The effect of gravity is applied after the interpolation function
            custom_reading = self._is.get_sensor_reading(
                self.sphere_path + "/custom_sensor", custom_function, read_gravity=False
            )
            sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/custom_sensor", read_gravity=False)

            self.assertEqual(custom_reading.time, sensor_reading.time)
            self.assertEqual(custom_reading.lin_acc_x, -100)
            self.assertEqual(custom_reading.ang_vel_x, -100)
            self.assertNotEqual(custom_reading.lin_acc_x, sensor_reading.lin_acc_x)

    async def test_sensor_latest_data(self):
        await self.test_add_sensor_prim()
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/custom_sensor",
            parent=self.sphere_path,
            sensor_period=1 / (self._sensor_rate / 2),  # 30hz, half of physics rate
            translation=self.sensor_offsets[4],
            orientation=self.sensor_quatd[4],
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        await omni.kit.app.get_app().next_update_async()
        self.my_world.play()
        await omni.kit.app.get_app().next_update_async()

        old_time = -1
        for i in range(10):  # Simulate 10 steps
            await omni.kit.app.get_app().next_update_async()
            latest_sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/custom_sensor", None, True)
            self.assertTrue(latest_sensor_reading.time > old_time)
            old_time = latest_sensor_reading.time

    async def test_wrong_sensor_path(self):
        await self.test_add_sensor_prim()
        await omni.kit.app.get_app().next_update_async()
        self.my_world.play()
        # give it some time to reach the ground first
        await omni.kit.app.get_app().next_update_async()

        for i in range(10):  # Simulate for 10 steps
            await omni.kit.app.get_app().next_update_async()
            latest_sensor_reading = self._is.get_sensor_reading(self.sphere_path + "/wrong_sensor", None, True)

            self.assertFalse(latest_sensor_reading.is_valid)
            self.assertEqual(latest_sensor_reading.time, 0)
        pass

    async def test_change_buffer_size(self):
        self.filter_wdith = 20
        self.actual_buffer_length = 20

        def custom_function(sensorReadings: List[_sensor.IsSensorReading], time: float) -> _sensor.IsSensorReading:
            override_sensor_reading = _sensor.IsSensorReading()
            self.actual_buffer_length = len(sensorReadings)
            return override_sensor_reading

        await self.test_add_sensor_prim()
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/custom_sensor",
            parent=self.sphere_path,
            sensor_period=1 / (self._sensor_rate / 2),  # 30hz, half of physics rate
            translation=self.sensor_offsets[4],
            orientation=self.sensor_quatd[4],
        )
        self.assertTrue(result)
        self.assertIsNotNone(sensor)

        await omni.kit.app.get_app().next_update_async()
        self.my_world.play()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        imu_sensor = get_prim_at_path(self.sphere_path + "/custom_sensor")
        for i in range(10):  # Simulate 10 steps
            self.filter_wdith += 1
            imu_sensor.GetAttribute("linearAccelerationFilterWidth").Set(self.filter_wdith)
            await omni.kit.app.get_app().next_update_async()
            await omni.kit.app.get_app().next_update_async()

            custom_reading = self._is.get_sensor_reading(self.sphere_path + "/custom_sensor", custom_function)

            # the sensor readings length is 2 times the size of the highest filter width, unless if the filter widths are under 10, then it is 20
            self.assertEqual(self.actual_buffer_length, 2 * self.filter_wdith)

        # set the rolling averages to something small, check the rolling average size doesn't go below 20
        imu_sensor.GetAttribute("linearAccelerationFilterWidth").Set(2)
        imu_sensor.GetAttribute("angularVelocityFilterWidth").Set(2)
        imu_sensor.GetAttribute("orientationFilterWidth").Set(2)

        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        custom_reading = self._is.get_sensor_reading(self.sphere_path + "/custom_sensor", custom_function)
        self.assertEqual(self.actual_buffer_length, 20)

    async def test_imu_rigidbody_grandparent(self):
        await self.createAnt()
        cube = self.my_world.scene.add(DynamicCuboid(prim_path="/World/Cube", position=np.array([10, 0, 0])))

        xform = self.my_world.scene.add(
            XFormPrim(prim_path="/World/Cube/xform", name="xform", translation=np.array([10, 0, 0]))
        )

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/custom_sensor",
            parent="/World/Cube/xform",
            sensor_period=0,
            translation=self.sensor_offsets[4],
            orientation=self.sensor_quatd[4],
        )

        await omni.kit.app.get_app().next_update_async()
        self.my_world.play()
        await simulate_async(0.5)
        custom_reading = self._is.get_sensor_reading("/World/Cube/xform/custom_sensor")
        # print(custom_reading.lin_acc_x)
        # print(custom_reading.lin_acc_y)
        # print(custom_reading.lin_acc_z)

        self.assertAlmostEqual(custom_reading.lin_acc_z, 9.81, delta=0.1)

        self.my_world.stop()
        await omni.kit.app.get_app().next_update_async()

        # Rotate the parent cube about y by -90 degree
        # The x axis points upward
        cube.set_local_pose(orientation=np.array([0.70711, 0.0, -0.70711, 0.0]))
        self.my_world.play()
        await omni.kit.app.get_app().next_update_async()

        await simulate_async(0.5)
        custom_reading = self._is.get_sensor_reading("/World/Cube/xform/custom_sensor")
        # print(custom_reading.lin_acc_x)
        # print(custom_reading.lin_acc_y)
        # print(custom_reading.lin_acc_z)
        self.assertAlmostEqual(custom_reading.lin_acc_x, 9.81, delta=0.1)

        # rotated -90 degress abouty, check if this is correct
        # note: (-0.70711, 0 0.70711, 0) and (0.70711, 0, -0.70711, 0) represent the same angle
        self.assertAlmostEquals(abs(custom_reading.orientation.w), 0.70711, delta=1e-4)
        self.assertAlmostEquals(custom_reading.orientation.x, 0.0, delta=1e-4)
        self.assertAlmostEquals(abs(custom_reading.orientation.y), 0.70711, delta=1e-4)
        self.assertAlmostEquals(custom_reading.orientation.z, 0.0, delta=1e-4)
        self.assertAlmostEquals(custom_reading.orientation.w, -custom_reading.orientation.y, delta=1e-4)

    async def test_invalid_imu(self):
        # goal is to make sure an invalid imu doesn't crash the sim
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/sensor",
            parent="/World",
            sensor_period=1 / self._sensor_rate,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
        )
        self.my_world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60, rendering_dt=1.0 / 60)
        await self.my_world.initialize_simulation_context_async()
        self.my_world.play()
        await simulate_async(0.1)
        self.my_world.stop()
        clear_stage()
