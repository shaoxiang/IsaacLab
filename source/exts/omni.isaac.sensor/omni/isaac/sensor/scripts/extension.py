# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import gc

import carb
import carb.settings
import numpy as np
import omni.ext
import omni.kit.commands
import omni.replicator.core as rep
from omni.isaac.core.utils.stage import traverse_stage
from omni.replicator.core import AnnotatorRegistry
from omni.syntheticdata import sensors

from .. import _sensor
from .menu import IsaacSensorMenu

EXTENSION_NAME = "Isaac Sensor"


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._cs = _sensor.acquire_contact_sensor_interface()
        self._is = _sensor.acquire_imu_sensor_interface()
        self._menu = IsaacSensorMenu(ext_id)

        self.registered_template = []
        self.registered_annotators = []
        try:
            self.register_nodes()
        except Exception as e:
            carb.log_warn(f"Could not register node templates {e}")

        def _on_pre_load_file(event):
            # Rename all nodes from omni.isaac.isaac_sensor to omni.isaac.sensor
            for prim in traverse_stage():
                if prim.HasAttribute("node:type"):
                    type_attr = prim.GetAttribute("node:type")
                    value = type_attr.Get()
                    if "omni.isaac.isaac_sensor" in value:
                        carb.log_warn(
                            f"Updating node type from omni.isaac.isaac_sensor to omni.isaac.sensor for {str(prim.GetPath())}. Please save and reload the asset, The omni.isaac.isaac_sensor extension was renamed to omni.isaac.sensor "
                        )
                        value = value.replace("omni.isaac.isaac_sensor", "omni.isaac.sensor")
                        type_attr.Set(value)

        self._on_stage_load_sub = (
            omni.usd.get_context()
            .get_stage_event_stream()
            .create_subscription_to_pop_by_type(int(omni.usd.StageEventType.OPENED), _on_pre_load_file)
        )

    def on_shutdown(self):
        _sensor.release_contact_sensor_interface(self._cs)
        _sensor.release_imu_sensor_interface(self._is)

        try:
            self.unregister_nodes()
        except Exception as e:
            carb.log_warn(f"Could not unregister node templates {e}")

        self._menu.shutdown()
        self._menu = None
        self._on_stage_load_sub = None
        gc.collect()

    def register_nodes(self):

        ### Add render var to omni.syntheticdata
        settings = carb.settings.get_settings()

        ### Add sync gate
        template_name = "RtxSensorCpu" + "IsaacSimulationGate"
        if template_name not in sensors.get_synthetic_data()._ogn_templates_registry:
            template = sensors.get_synthetic_data().register_node_template(
                omni.syntheticdata.SyntheticData.NodeTemplate(
                    omni.syntheticdata.SyntheticDataStage.ON_DEMAND,
                    "omni.isaac.core_nodes.IsaacSimulationGate",
                    [omni.syntheticdata.SyntheticData.NodeConnectionTemplate("RtxSensorCpu" + "Ptr")],
                ),
                template_name=template_name,
            )
            self.registered_template.append(template)

        template_name = "RtxSensorGpu" + "IsaacSimulationGate"
        if template_name not in sensors.get_synthetic_data()._ogn_templates_registry:
            template = sensors.get_synthetic_data().register_node_template(
                omni.syntheticdata.SyntheticData.NodeTemplate(
                    omni.syntheticdata.SyntheticDataStage.ON_DEMAND,
                    "omni.isaac.core_nodes.IsaacSimulationGate",
                    [omni.syntheticdata.SyntheticData.NodeConnectionTemplate("RtxSensorGpu" + "Ptr")],
                ),
                template_name=template_name,
            )
            self.registered_template.append(template)

        ### Read RtxLidar Data
        annotator_name = "RtxSensorCpu" + "IsaacReadRTXLidarData"
        AnnotatorRegistry.register_annotator_from_node(
            name=annotator_name,
            input_rendervars=["RtxSensorCpu" + "Ptr"],
            node_type_id="omni.isaac.sensor.IsaacReadRTXLidarData",
        )
        self.registered_annotators.append(annotator_name)
        # Register annotator for Replicator telemetry tracking
        AnnotatorRegistry._default_annotators.append(
            annotator_name
        ) if annotator_name not in AnnotatorRegistry._default_annotators else None

        annotator_name = "RtxSensorGpu" + "IsaacReadRTXLidarData"
        AnnotatorRegistry.register_annotator_from_node(
            name=annotator_name,
            input_rendervars=[
                "RtxSensorGpu" + "Ptr",
            ],
            node_type_id="omni.isaac.sensor.IsaacReadRTXLidarData",
        )
        self.registered_annotators.append(annotator_name)
        # Register annotator for Replicator telemetry tracking
        AnnotatorRegistry._default_annotators.append(
            annotator_name
        ) if annotator_name not in AnnotatorRegistry._default_annotators else None

        # NodeConnectionTemplate(
        #    "SemanticBoundingBox2DExtentTightSDhostPtr",
        #    attributes_mapping={"outputs:dataPtr": "inputs:dataPtr", "outputs:bufferSize": "inputs:bufferSize"},
        # ),

        ### RtxLidar Point Cloud
        annotator_name = "RtxSensorCpu" + "IsaacComputeRTXLidarPointCloud"
        AnnotatorRegistry.register_annotator_from_node(
            name=annotator_name,
            input_rendervars=["RtxSensorCpu" + "Ptr"],
            node_type_id="omni.isaac.sensor.IsaacComputeRTXLidarPointCloud",
            output_data_type=np.float32,
            output_channels=3,
        )
        self.registered_annotators.append(annotator_name)
        # Register annotator for Replicator telemetry tracking
        AnnotatorRegistry._default_annotators.append(
            annotator_name
        ) if annotator_name not in AnnotatorRegistry._default_annotators else None

        annotator_name = "RtxSensorCpu" + "IsaacCreateRTXLidarScanBuffer"
        AnnotatorRegistry.register_annotator_from_node(
            name=annotator_name,
            input_rendervars=["RtxSensorCpu" + "Ptr"],
            node_type_id="omni.isaac.sensor.IsaacCreateRTXLidarScanBuffer",
            output_data_type=np.float32,
            output_channels=3,
        )
        self.registered_annotators.append(annotator_name)
        # Register annotator for Replicator telemetry tracking
        AnnotatorRegistry._default_annotators.append(
            annotator_name
        ) if annotator_name not in AnnotatorRegistry._default_annotators else None

        annotator_name = "RtxSensorCpu" + "IsaacRTXLidarOutput"
        AnnotatorRegistry.register_annotator_from_node(
            name=annotator_name,
            input_rendervars=["RtxSensorCpu" + "Ptr"],
            node_type_id="omni.isaac.sensor.IsaacRTXLidarOutput",
            output_data_type=np.float32,
            output_channels=3,
        )
        self.registered_annotators.append(annotator_name)
        # Register annotator for Replicator telemetry tracking
        AnnotatorRegistry._default_annotators.append(
            annotator_name
        ) if annotator_name not in AnnotatorRegistry._default_annotators else None

        ### RtxLidar Point Cloud Print Info Writer
        rep.writers.register_node_writer(
            name="Writer" + "IsaacPrintRTXLidarInfo",
            node_type_id="omni.isaac.sensor.IsaacPrintRTXLidarInfo",
            annotators=[omni.syntheticdata.SyntheticData.NodeConnectionTemplate("RtxSensorCpu" + "Ptr")],
            category="omni.isaac.sensor",
        )
        # Register writer for Replicator telemetry tracking
        rep.WriterRegistry._default_writers.append(
            "Writer" + "IsaacPrintRTXLidarInfo"
        ) if "Writer" + "IsaacPrintRTXLidarInfo" not in rep.WriterRegistry._default_writers else None

        ### RtxRadar Point Cloud Print Info Writer
        rep.writers.register_node_writer(
            name="Writer" + "IsaacPrintRTXRadarInfo",
            node_type_id="omni.isaac.sensor.IsaacPrintRTXRadarInfo",
            annotators=[omni.syntheticdata.SyntheticData.NodeConnectionTemplate("RtxSensorCpu" + "Ptr")],
            category="omni.isaac.sensor",
        )
        # Register writer for Replicator telemetry tracking
        rep.WriterRegistry._default_writers.append(
            "Writer" + "IsaacPrintRTXRadarInfo"
        ) if "Writer" + "IsaacPrintRTXRadarInfo" not in rep.WriterRegistry._default_writers else None

        ### RtxLidar Flat Scan
        annotator_name = "RtxSensorCpu" + "IsaacComputeRTXLidarFlatScan"
        AnnotatorRegistry.register_annotator_from_node(
            name=annotator_name,
            input_rendervars=["RtxSensorCpu" + "Ptr"],
            node_type_id="omni.isaac.sensor.IsaacComputeRTXLidarFlatScan",
            output_data_type=np.float32,
            output_channels=3,
        )
        self.registered_annotators.append(annotator_name)
        # Register annotator for Replicator telemetry tracking
        AnnotatorRegistry._default_annotators.append(
            annotator_name
        ) if annotator_name not in AnnotatorRegistry._default_annotators else None

        # RTX lidar Debug Draw Writer
        rep.writers.register_node_writer(
            name="RtxLidar" + "DebugDrawPointCloud",
            node_type_id="omni.isaac.debug_draw.DebugDrawPointCloud",
            annotators=[
                omni.syntheticdata.SyntheticData.NodeConnectionTemplate(
                    "RtxSensorCpu" + "IsaacComputeRTXLidarPointCloud"
                )
            ],
            category="omni.isaac.sensor",
        )
        # Register writer for Replicator telemetry tracking
        rep.WriterRegistry._default_writers.append(
            "RtxLidar" + "DebugDrawPointCloud"
        ) if "RtxLidar" + "DebugDrawPointCloud" not in rep.WriterRegistry._default_writers else None

        # RTX lidar Debug Draw Writer
        rep.writers.register_node_writer(
            name="RtxLidar" + "DebugDrawPointCloud" + "Buffer",
            node_type_id="omni.isaac.debug_draw.DebugDrawPointCloud",
            annotators=[
                omni.syntheticdata.SyntheticData.NodeConnectionTemplate(
                    "RtxSensorCpu" + "IsaacCreateRTXLidarScanBuffer"
                )
            ],
            doTransform=True,
            category="omni.isaac.sensor",
        )
        # Register writer for Replicator telemetry tracking
        rep.WriterRegistry._default_writers.append(
            "RtxLidar" + "DebugDrawPointCloud" + "Buffer"
        ) if "RtxLidar" + "DebugDrawPointCloud" + "Buffer" not in rep.WriterRegistry._default_writers else None

        # RTX lidar Debug Draw Writer
        rep.writers.register_node_writer(
            name="RtxLidar" + "DebugDrawPointCloud" + "Buffer2",
            node_type_id="omni.isaac.debug_draw.DebugDrawPointCloud",
            annotators=[
                omni.syntheticdata.SyntheticData.NodeConnectionTemplate("RtxSensorCpu" + "IsaacRTXLidarOutput")
            ],
            doTransform=True,
            category="omni.isaac.sensor",
        )

        ### RtxRadar Point Cloud
        annotator_name = "RtxSensorCpu" + "IsaacComputeRTXRadarPointCloud"
        AnnotatorRegistry.register_annotator_from_node(
            name=annotator_name,
            input_rendervars=["RtxSensorCpu" + "Ptr"],
            node_type_id="omni.isaac.sensor.IsaacComputeRTXRadarPointCloud",
            output_data_type=np.float32,
            output_channels=3,
        )
        self.registered_annotators.append(annotator_name)
        # Register annotator for Replicator telemetry tracking
        AnnotatorRegistry._default_annotators.append(
            annotator_name
        ) if annotator_name not in AnnotatorRegistry._default_annotators else None

        # RTX radar Debug Draw Writer
        rep.writers.register_node_writer(
            name="RtxRadar" + "DebugDrawPointCloud",
            node_type_id=f"omni.isaac.debug_draw.DebugDrawPointCloud",
            annotators=["RtxSensorCpu" + "IsaacComputeRTXRadarPointCloud"],
            # hard to see radar points... so make them more visible.
            size=0.2,
            color=[1, 0.2, 0.3, 1],
            category="omni.isaac.sensor",
        )
        # Register writer for Replicator telemetry tracking
        rep.WriterRegistry._default_writers.append(
            "RtxRadar" + "DebugDrawPointCloud"
        ) if "RtxRadar" + "DebugDrawPointCloud" not in rep.WriterRegistry._default_writers else None

    def unregister_nodes(self):
        for template in self.registered_template:
            sensors.get_synthetic_data().unregister_node_template(template)
        for annotator in self.registered_annotators:
            AnnotatorRegistry.unregister_annotator(annotator)
        for writer in rep.WriterRegistry.get_writers(category="omni.isaac.sensor"):
            rep.writers.unregister_writer(writer)
