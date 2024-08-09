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

EXTENSION_NAME = "Contact Sensor Example"


class Contact_sensor_demo(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        self._ext_id = ext_id
        self._extension_path = ext_manager.get_extension_path(ext_id)

        self._menu_items = [
            MenuItemDescription(
                name="Sensors",
                sub_menu=[make_menu_item_description(ext_id, "Contact", lambda a=weakref.proxy(self): a.build_ui())],
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
            self._cs = _sensor.acquire_contact_sensor_interface()

            self._timeline = omni.timeline.get_timeline_interface()
            self.sub = _physx.get_physx_interface().subscribe_physics_step_events(self._on_update)

            self.leg_paths = ["/Ant/Arm_{:02d}/Lower_Arm".format(i + 1) for i in range(4)]

            self.shoulder_joints = ["/Ant/Arm_{:02d}/Upper_Arm/shoulder_joint".format(i + 1) for i in range(4)]

            self.lower_joints = ["{}/lower_arm_joint".format(i) for i in self.leg_paths]
            self._sensor_handles = [0 for i in range(4)]
            self.sliders = None
            # self._window = ui.Window(
            #     title="Contact Sensor Sample", width=300, height=200, dockPreference=ui.DockPreference.LEFT_BOTTOM
            # )
            self.sliders = []
            self.colors = [0xFFBBBBFF, 0xFFBBFFBB, 0xBBFFBBBB, 0xBBBBFFFF]
            style = {"background_color": 0xFF888888, "color": 0xFF333333, "secondary_color": self.colors[0]}

            self.plots = []
            self.plot_vals = []
            self._window = ui.Window(
                title=EXTENSION_NAME, width=600, height=0, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM
            )
            with self._window.frame:
                with ui.VStack(spacing=5, height=0):

                    title = "Contact Sensor Example"
                    doc_link = "https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_physics_based_contact.html"

                    overview = "This Example shows how to Surface load sensors applied to a body. "
                    overview += "It works by summing all forces applied on a given trigger shperical region intersected with the given body surface."
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
                        with ui.VStack(style=get_style(), spacing=5):
                            for i in range(4):
                                with ui.HStack():
                                    ui.Label("Arm {}".format(i + 1), width=LABEL_WIDTH, tooltip="Force in Newtons")
                                    # ui.Spacer(height=0, width=10)
                                    style["secondary_color"] = self.colors[i]
                                    self.sliders.append(ui.FloatDrag(min=0.0, max=15.0, step=0.001, style=style))
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
            for i in range(4):
                reading = self._cs.get_sensor_reading(self.leg_paths[i] + "/sensor")
                if reading.is_valid:
                    self.sliders[i].model.set_value(
                        float(reading.value) * self.meters_per_unit
                    )  # readings are in kg⋅m⋅s−2, converting to Newtons
                else:
                    self.sliders[i].model.set_value(0)
            # contacts_raw = self._cs.get_body_contact_raw_data(self.leg_paths[0])
            # if len(contacts_raw):
            #     c = contacts_raw[0]
            #     # print(c)

    async def create_scenario(self):
        self._assets_root_path = get_assets_root_path()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # Add Contact Sensor
        await omni.usd.get_context().open_stage_async(self._assets_root_path + "/Isaac/Robots/Simple/ant.usd")
        await omni.kit.app.get_app().next_update_async()

        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(omni.usd.get_context().get_stage())

        self.sensor_offsets = [Gf.Vec3d(40, 0, 0), Gf.Vec3d(40, 0, 0), Gf.Vec3d(40, 0, 0), Gf.Vec3d(40, 0, 0)]
        self.color = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
        self.sensorGeoms = []

        for i in range(4):
            result, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateContactSensor",
                path="/sensor",
                parent=self.leg_paths[i],
                min_threshold=0,
                max_threshold=10000000,
                color=self.color[i],
                radius=0.12,
                sensor_period=-1,
                translation=self.sensor_offsets[i],
            )

        self._events = omni.usd.get_context().get_stage_event_stream()
        self._stage_event_subscription = self._events.create_subscription_to_pop(
            self._on_stage_event, name="Contact Sensor Sample stage Watch"
        )
