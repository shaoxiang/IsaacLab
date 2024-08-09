# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


import asyncio
import weakref

import carb
import omni
import omni.kit.commands
import omni.physx as _physx
import omni.ui as ui
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.sensor import _sensor
from omni.isaac.ui.menu import make_menu_item_description
from omni.isaac.ui.ui_utils import LABEL_WIDTH, get_style, setup_ui_headers
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from pxr import Gf, UsdGeom

EXTENSION_NAME = "IMU Sensor Example"


class Imu_sensor_demo(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        self._ext_id = ext_id
        self._extension_path = ext_manager.get_extension_path(ext_id)

        self._menu_items = [
            MenuItemDescription(
                name="Sensors",
                sub_menu=[make_menu_item_description(ext_id, "IMU", lambda a=weakref.proxy(self): a.build_ui())],
            )
        ]
        add_menu_items(self._menu_items, "Isaac Examples")
        self.meters_per_unit = 1.00
        self._window = None

    def _on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.CLOSED):
            self.on_closed()

    def build_ui(self):
        if self._window is None:
            self._is = _sensor.acquire_imu_sensor_interface()

            self._timeline = omni.timeline.get_timeline_interface()
            self.sub = _physx.get_physx_interface().subscribe_physics_step_events(self._on_update)

            self.body_path = "/Ant/Sphere"

            self.sliders = None
            # self._window = ui.Window(
            #     title="IMU Sensor Sample", width=300, height=200, dockPreference=ui.DockPreference.LEFT_BOTTOM
            # )
            self.sliders = []
            self.colors = [
                0xFFBBBBFF,
                0xFFBBFFBB,
                0xBBFFBBBB,
                0xBBAAEEFF,
                0xAABBFFEE,
                0xFFEEAABB,
                0xFFC8D5D0,
                0xFFC89BD0,
                0xFFAF9BA7,
                0xFFA4B99A,
            ]
            self.acc_names = ["Acc x", "Acc y", "Acc z"]
            self.gyro_names = ["Gyro x", "Gyro y", "Gyro z"]
            self.orient_names = ["Orient x", "Orient y", "Orient z", "Orient w"]

            style = {"background_color": 0xFF888888, "color": 0xFF333333, "secondary_color": self.colors[0]}

            self.plots = []
            self.plot_vals = []
            self._window = ui.Window(
                title=EXTENSION_NAME, width=600, height=0, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM
            )
            with self._window.frame:
                with ui.VStack(spacing=5, height=0):

                    title = "IMU Sensor Example"
                    doc_link = "https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_physics_based_imu.html"

                    overview = "This Example shows the output of the IMU sensor. "
                    overview += "The IMU sensor reads motion of the body of the robot and output simulated accelerometer and gyroscope readings"
                    overview += (
                        "\nPress PLAY to start the simulation, hold 'shift' and left click the model to drag it around"
                    )
                    overview += "\n\nPress the 'Open in IDE' button to view the source code."

                    setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)

                    frame = ui.CollapsableFrame(
                        title="Sensor Readings",
                        height=0,
                        collapsed=False,
                        style=get_style(),
                        style_type_name_override="CollapsableFrame",
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
                    )
                    with frame:
                        with ui.VStack(style=get_style(), spacing=3):
                            for i in range(3):
                                with ui.HStack():
                                    ui.Label(self.acc_names[i], width=LABEL_WIDTH, tooltip="acceleration in m/s^2")
                                    # ui.Spacer(height=0, width=10)
                                    style["secondary_color"] = self.colors[i]
                                    self.sliders.append(ui.FloatDrag(min=-15.0, max=15.0, step=0.001, style=style))
                                    self.sliders[-1].enabled = False
                                    ui.Spacer(width=20)
                            for j in range(3):
                                with ui.HStack():
                                    ui.Label(self.gyro_names[j], width=LABEL_WIDTH, tooltip="angular velocity in rad/s")
                                    # ui.Spacer(height=0, width=10)
                                    style["secondary_color"] = self.colors[3 + j]
                                    self.sliders.append(ui.FloatDrag(min=-15.0, max=15.0, step=0.001, style=style))
                                    self.sliders[-1].enabled = False
                                    ui.Spacer(width=20)
                            for k in range(4):
                                with ui.HStack():
                                    ui.Label(self.orient_names[k], width=LABEL_WIDTH, tooltip="orientation quaternion")
                                    # ui.Spacer(height=0, width=10)
                                    style["secondary_color"] = self.colors[6 + k]
                                    self.sliders.append(ui.FloatDrag(min=-1.0, max=1.0, step=0.001, style=style))
                                    self.sliders[-1].enabled = False
                                    ui.Spacer(width=20)

            asyncio.ensure_future(self.create_scenario())

        self._window.visible = True

    def on_shutdown(self):
        self.on_closed()
        remove_menu_items(self._menu_items, "Isaac Examples")

    def on_closed(self):
        if self._window:
            self.sub = None
            self._timeline = None
            self._stage_event_subscription = None

        self._window = None

    def _on_update(self, dt):
        if self._timeline.is_playing() and self.sliders:
            reading = self._is.get_sensor_reading(self.body_path + "/sensor")
            if reading.is_valid:
                self.sliders[0].model.set_value(float(reading.lin_acc_x) * self.meters_per_unit)  # readings
                self.sliders[1].model.set_value(float(reading.lin_acc_y) * self.meters_per_unit)  # readings
                self.sliders[2].model.set_value(float(reading.lin_acc_z) * self.meters_per_unit)  # readings
                self.sliders[3].model.set_value(float(reading.ang_vel_x))  # readings
                self.sliders[4].model.set_value(float(reading.ang_vel_y))  # readings
                self.sliders[5].model.set_value(float(reading.ang_vel_z))  # readings
                self.sliders[6].model.set_value(float(reading.orientation[0]))  # readings
                self.sliders[7].model.set_value(float(reading.orientation[1]))  # readings
                self.sliders[8].model.set_value(float(reading.orientation[2]))  # readings
                self.sliders[9].model.set_value(float(reading.orientation[3]))  # readings
        else:
            self.sliders[0].model.set_value(0)
            self.sliders[1].model.set_value(0)
            self.sliders[2].model.set_value(0)
            self.sliders[3].model.set_value(0)
            self.sliders[4].model.set_value(0)
            self.sliders[5].model.set_value(0)
            self.sliders[6].model.set_value(0)
            self.sliders[7].model.set_value(0)
            self.sliders[8].model.set_value(0)
            self.sliders[9].model.set_value(1)

    async def create_scenario(self):
        self._assets_root_path = get_assets_root_path()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # Add IMU Sensor
        await omni.usd.get_context().open_stage_async(self._assets_root_path + "/Isaac/Robots/Simple/ant.usd")
        await omni.kit.app.get_app().next_update_async()

        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(omni.usd.get_context().get_stage())

        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/sensor",
            parent=self.body_path,
            sensor_period=-1.0,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
        )

        self._events = omni.usd.get_context().get_stage_event_stream()
        self._stage_event_subscription = self._events.create_subscription_to_pop(
            self._on_stage_event, name="IMU Sensor Sample stage Watch"
        )
