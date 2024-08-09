.. _omni_isaac_sensor_IsaacCreateRTXLidarScanBuffer_1:

.. _omni_isaac_sensor_IsaacCreateRTXLidarScanBuffer:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Create RTX Lidar Scan Buffer
    :keywords: lang-en omnigraph node isaacSensor sensor isaac-create-r-t-x-lidar-scan-buffer


Isaac Create RTX Lidar Scan Buffer
==================================

.. <description>

This node creates a full scan buffer for RTX Lidar sensor.

.. </description>


Installation
------------

To use this node enable :ref:`omni.isaac.sensor<ext_omni_isaac_sensor>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Error Azimuth (*inputs:accuracyErrorAzimuthDeg*)", "``float``", "Accuracy error of azimuth in degrees applied to all points equally", "0.0"
    "Error Elevation (*inputs:accuracyErrorElevationDeg*)", "``float``", "Accuracy error of elevation in degrees applied to all points equally", "0.0"
    "Error Position (*inputs:accuracyErrorPosition*)", "``float[3]``", "Position offset applied to all points equally", "[0.0, 0.0, 0.0]"
    "Cuda Device Index (*inputs:cudaDeviceIndex*)", "``int``", "Index of the device where the data lives (-1 for host data)", "-1"
    "Data Pointer (*inputs:dataPtr*)", "``uint64``", "Pointer to LiDAR render result.", "0"
    "Exec (*inputs:exec*)", "``execution``", "The input execution port", "None"
    "Keep Only Positive Distance (*inputs:keepOnlyPositiveDistance*)", "``bool``", "Keep points only if the return distance is > 0", "True"
    "Output The Azimuth (*inputs:outputAzimuth*)", "``bool``", "Create an output array for the Azimuth.", "False"
    "Output The BeamId (*inputs:outputBeamId*)", "``bool``", "Create an output array for the BeamId.", "False"
    "Output The Distance (*inputs:outputDistance*)", "``bool``", "Create an output array for the Distance.", "True"
    "Output The Elevation (*inputs:outputElevation*)", "``bool``", "Create an output array for the Elevation.", "False"
    "Output The EmitterId (*inputs:outputEmitterId*)", "``bool``", "Create an output array for the EmitterId.", "False"
    "Output The Intensity (*inputs:outputIntensity*)", "``bool``", "Create an output array for the Intensity.", "True"
    "Output The MaterialId (*inputs:outputMaterialId*)", "``bool``", "Create an output array for the MaterialId.", "False"
    "Output The Normals (*inputs:outputNormal*)", "``bool``", "Create an output array for the Normals.", "False"
    "Output The ObjectId (*inputs:outputObjectId*)", "``bool``", "Create an output array for the ObjectId.", "False"
    "Output The Timestamp (*inputs:outputTimestamp*)", "``bool``", "Create an output array for the Timestamp.", "False"
    "Output The Velocity (*inputs:outputVelocity*)", "``bool``", "Create an output array for the Velocity.", "False"
    "Render Product Path (*inputs:renderProductPath*)", "``token``", "Config is gotten from this", ""
    "Output in World Coordinates (*inputs:transformPoints*)", "``bool``", "Transform point cloud to world coordinates", "False"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Azimuth Buffer Size (*outputs:azimuthBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Azimuth Data Type (*outputs:azimuthDataType*)", "``float``", "type", "4"
    "", "Metadata", "*hidden* = true", ""
    "azimuth (*outputs:azimuthPtr*)", "``uint64``", "azimuth in rad [-pi,pi]", "None"
    "Beam Id Buffer Size (*outputs:beamIdBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Beam Id Data Type (*outputs:beamIdDataType*)", "``uint``", "type", "4"
    "", "Metadata", "*hidden* = true", ""
    "beamId (*outputs:beamIdPtr*)", "``uint64``", "beamId", "None"
    "Buffer Size (*outputs:bufferSize*)", "``uint64``", "Size (in bytes) of the buffer (0 if the input is a texture)", "None"
    "Cuda Device Index (*outputs:cudaDeviceIndex*)", "``int``", "Index of the device where the data lives (-1 for host data)", "-1"
    "Point Cloud Data (*outputs:dataPtr*)", "``uint64``", "Pointer to LiDAR render result.", "None"
    "Distance Buffer Size (*outputs:distanceBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Distance Data Type (*outputs:distanceDataType*)", "``float``", "type", "4"
    "", "Metadata", "*hidden* = true", ""
    "distance (*outputs:distancePtr*)", "``uint64``", "range in m", "None"
    "Elevation Buffer Size (*outputs:elevationBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Elevation Data Type (*outputs:elevationDataType*)", "``float``", "type", "4"
    "", "Metadata", "*hidden* = true", ""
    "elevation (*outputs:elevationPtr*)", "``uint64``", "elevation in rad [-pi/2, pi/2]", "None"
    "Emitter Id Buffer Size (*outputs:emitterIdBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Emitter Id Data Type (*outputs:emitterIdDataType*)", "``uint``", "type", "4"
    "", "Metadata", "*hidden* = true", ""
    "emitterId (*outputs:emitterIdPtr*)", "``uint64``", "emitterId", "None"
    "Exec (*outputs:exec*)", "``execution``", "Output execution triggers when lidar sensor has data", "None"
    "Height (*outputs:height*)", "``uint``", "Height of point cloud buffer, will always return 1", "1"
    "Index Buffer Size (*outputs:indexBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Index Data Type (*outputs:indexDataType*)", "``uint``", "type", "4"
    "", "Metadata", "*hidden* = true", ""
    "index (*outputs:indexPtr*)", "``uint64``", "Index into the full array if keepOnlyPositiveDistance ((startTick+tick)*numChannels*numEchos + channel*numEchos + echo)", "None"
    "Intensity Buffer Size (*outputs:intensityBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Intensity Data Type (*outputs:intensityDataType*)", "``float``", "type", "4"
    "", "Metadata", "*hidden* = true", ""
    "intensity (*outputs:intensityPtr*)", "``uint64``", "intensity [0,1]", "None"
    "Material Id Buffer Size (*outputs:materialIdBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Material Id Data Type (*outputs:materialIdDataType*)", "``uint``", "type", "4"
    "", "Metadata", "*hidden* = true", ""
    "materialId (*outputs:materialIdPtr*)", "``uint64``", "materialId at hit location", "None"
    "Normal Buffer Size (*outputs:normalBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Normal Data Type (*outputs:normalDataType*)", "``float[3]``", "type", "[4, 0, 0]"
    "", "Metadata", "*hidden* = true", ""
    "normal (*outputs:normalPtr*)", "``uint64``", "Normal at the hit location", "None"
    "Num Channels (*outputs:numChannels*)", "``uint``", "Number of channels of the lidar", "None"
    "Num Echos (*outputs:numEchos*)", "``uint``", "Number of echos of the lidar", "None"
    "Num Returns Per Scan (*outputs:numReturnsPerScan*)", "``uint``", "Number of returns in the full scan", "None"
    "Object Id Buffer Size (*outputs:objectIdBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Object Id Data Type (*outputs:objectIdDataType*)", "``uint``", "type", "4"
    "", "Metadata", "*hidden* = true", ""
    "objectId (*outputs:objectIdPtr*)", "``uint64``", "ObjectId for getting usd prim information", "None"
    "Render Product Path (*outputs:renderProductPath*)", "``token``", "Config is gotten from this", "None"
    "Ticks Per Scan (*outputs:ticksPerScan*)", "``uint``", "Number of ticks in a full scan", "None"
    "Timestamp Buffer Size (*outputs:timestampBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Timestamp Data Type (*outputs:timestampDataType*)", "``uint64``", "type", "8"
    "", "Metadata", "*hidden* = true", ""
    "timestamp (*outputs:timestampPtr*)", "``uint64``", "timestamp in ns", "None"
    "Transform (*outputs:transform*)", "``matrixd[4]``", "The transform matrix from lidar to world coordinates", "None"
    "Velocity Buffer Size (*outputs:velocityBufferSize*)", "``uint64``", "size", "None"
    "", "Metadata", "*hidden* = true", ""
    "Velocity Data Type (*outputs:velocityDataType*)", "``float[3]``", "type", "[4, 0, 0]"
    "", "Metadata", "*hidden* = true", ""
    "velocity (*outputs:velocityPtr*)", "``uint64``", "elevation in rad [-pi/2, pi/2]", "None"
    "Width (*outputs:width*)", "``uint``", "3 x Width or number of points in point cloud buffer", "0"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.isaac.sensor.IsaacCreateRTXLidarScanBuffer"
    "Version", "1"
    "Extension", "omni.isaac.sensor"
    "Icon", "ogn/icons/omni.isaac.sensor.IsaacCreateRTXLidarScanBuffer.svg"
    "Has State?", "True"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Create RTX Lidar Scan Buffer"
    "Categories", "isaacSensor"
    "Generated Class Name", "OgnIsaacCreateRTXLidarScanBufferDatabase"
    "Python Module", "omni.isaac.sensor"

