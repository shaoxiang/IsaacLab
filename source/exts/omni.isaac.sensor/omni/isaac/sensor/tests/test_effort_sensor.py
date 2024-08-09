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
from typing import List

import carb.tokens
import numpy as np
import omni.kit.test
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.nucleus import get_assets_root_path_async
from omni.isaac.sensor.scripts.effort_sensor import EffortSensor, EsSensorReading
from pxr import UsdPhysics


class TestEffortSensor(omni.kit.test.AsyncTestCase):
    # Before running each test
    async def setUp(self):
        self._assets_root_path = await get_assets_root_path_async()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        self.my_world = None

    async def createSimpleArticulation(
        self, physics_rate=60, include_cube=False, cube_path="/new_cube", cube_position=np.array([1, 0, 0.1])
    ):
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

        prim = get_prim_at_path("/Articulation/Arm/RevoluteJoint")
        self.assertTrue(prim.IsValid())
        joint = UsdPhysics.RevoluteJoint(prim)
        joint.CreateAxisAttr("Y")

        if include_cube:
            DynamicCuboid(
                prim_path=cube_path,
                name="cube_1",
                position=cube_position,
                color=np.array([255, 0, 0]),
                size=0.1,
                mass=1,
            )

        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        pass

    # After running each test
    async def tearDown(self):
        if self.my_world:
            self.my_world.stop()
            self.my_world.clear_instance()
        if self.effort_sensor is not None:
            self.effort_sensor._stage_open_callback_fn()
            self.effort_sensor = None
        await omni.kit.app.get_app().next_update_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            # print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()
        pass

    async def test_sensor_reading(self):
        await self.createSimpleArticulation()

        self.effort_sensor = EffortSensor("/Articulation/Arm/RevoluteJoint")
        self.my_world.play()
        # let physics warm up
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        reading = self.effort_sensor.get_sensor_reading()
        self.assertTrue(reading.time != 0)

        # arm only, 2kg with C of G 1m away from the joint
        expected_effort = float(-2 * 9.81)
        self.assertAlmostEqual(reading.value, expected_effort, 1)
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        # spawn a 1kg cube 1.5m away from the joint
        DynamicCuboid(
            prim_path="/new_cube",
            name="cube_1",
            position=np.array([1.5, 0, 0.1]),
            color=np.array([255, 0, 0]),
            size=0.1,
            mass=1,
        )

        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        expected_effort = float(-3.5 * 9.81)
        reading = self.effort_sensor.get_sensor_reading()
        self.assertAlmostEqual(reading.value, expected_effort, 1)

        self.effort_sensor.enabled = False

        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        reading = self.effort_sensor.get_sensor_reading()
        self.assertFalse(reading.is_valid)

    async def test_sensor_period(self):
        await self.createSimpleArticulation()

        self.effort_sensor = EffortSensor("/Articulation/Arm/RevoluteJoint", 1 / 10)  # 10 hz
        self.my_world.play()
        # # print(self.effort_sensor.sensor_period)

        # let physics warm up
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        readings = []

        for i in range(60):  # Simulate for one second
            await omni.kit.app.get_app().next_update_async()
            sensor_reading = self.effort_sensor.get_sensor_reading()
            # # print(sensor_reading.time)
            # the sensor is running at 10hz, while the sim is 60hz, so expecting 1/6 readings to be new,
            # old reading should be identical, and have the same timestamp
            if not readings or readings[-1] != sensor_reading.time:
                # print(sensor_reading.time)
                readings.append(sensor_reading.time)

        # tolerance +-1 reading (9,10,11) will be accepted)
        # print(len(readings))
        self.assertTrue(abs(len(readings) - 10) <= 1)
        pass

    async def test_custom_interpolation_function(self):
        def custom_function(sensorReadings, time: float) -> EsSensorReading():
            override_sensor_reading = EsSensorReading()
            override_sensor_reading.value = 1000
            override_sensor_reading.time = time
            return override_sensor_reading

        await self.createSimpleArticulation()

        self.effort_sensor = EffortSensor("/Articulation/Arm/RevoluteJoint", 1 / 30)  # running at 30 Hz
        self.my_world.play()
        # # print(self.effort_sensor.sensor_period)

        # let physics warm up
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()

        for i in range(10):  # Simulate 10 steps
            await omni.kit.app.get_app().next_update_async()
            custom_reading = self.effort_sensor.get_sensor_reading(custom_function)
            sensor_reading = self.effort_sensor.get_sensor_reading()

            self.assertEqual(custom_reading.time, sensor_reading.time)
            self.assertEqual(custom_reading.value, 1000)
            self.assertNotEqual(custom_reading.value, sensor_reading.value)
        pass

    # Remove this test later
    async def test_change_to_wrong_dof_name_in_play(self):
        await self.createSimpleArticulation()

        self.effort_sensor = EffortSensor("/Articulation/Arm/RevoluteJoint")
        self.my_world.play()
        # # print(self.effort_sensor.sensor_period)

        # let physics warm up
        for i in range(10):
            await omni.kit.app.get_app().next_update_async()

        # sensor enabled with the correct indices, expect non zero output
        # print(f"dof index is: {self.effort_sensor.dof}")
        reading = self.effort_sensor.get_sensor_reading()
        # print(f"reading time: {reading.time}  reading value: {reading.value}")
        self.assertNotEqual(reading.time, 0)
        self.assertNotEqual(reading.value, 0)
        self.assertEqual(reading.is_valid, True)

        # set the sensor with incorrect joint
        try:
            self.effort_sensor.update_dof_name("RevoluteJoint_doesnt_exist")
        except:
            print(f"can't update dof, dof index is: {self.effort_sensor.dof}")
        for i in range(10):
            await omni.kit.app.get_app().next_update_async()

        # incorrect joint, expecting zero output (and error log message)
        reading = self.effort_sensor.get_sensor_reading()
        # print(f"reading time: {reading.time}  reading value: {reading.value}")
        self.assertEqual(reading.time, 0)
        self.assertEqual(reading.value, 0)
        self.assertEqual(reading.is_valid, False)

        # update it with the correct joint again, expecting non zero output
        self.effort_sensor.update_dof_name("RevoluteJoint")
        # print(f"dof index is: {self.effort_sensor.dof}")
        for i in range(10):
            await omni.kit.app.get_app().next_update_async()
        reading = self.effort_sensor.get_sensor_reading()
        # print(f"reading time: {reading.time}  reading value: {reading.value}")
        self.assertNotEqual(reading.time, 0)
        self.assertNotEqual(reading.value, 0)
        self.assertEqual(reading.is_valid, True)

    async def test_change_buffer_size(self):
        await self.createSimpleArticulation()

        self.effort_sensor = EffortSensor("/Articulation/Arm/RevoluteJoint", sensor_period=1)
        self.my_world.play()
        # # print(self.effort_sensor.sensor_period)

        # let physics warm up
        for i in range(10):
            await omni.kit.app.get_app().next_update_async()

        self.effort_sensor.change_buffer_size(20)

        self.assertEqual(self.effort_sensor.data_buffer_size, 20)
        self.assertEqual(len(self.effort_sensor.interpolation_buffer), 20)
        self.assertEqual(len(self.effort_sensor.sensor_reading_buffer), 20)

        self.effort_sensor.change_buffer_size(5)

        self.assertEqual(self.effort_sensor.data_buffer_size, 5)
        self.assertEqual(len(self.effort_sensor.interpolation_buffer), 5)
        self.assertEqual(len(self.effort_sensor.sensor_reading_buffer), 5)
