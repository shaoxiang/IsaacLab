.. _omni_isaac_sensor_IsaacReadRTXLidarData_1:

.. _omni_isaac_sensor_IsaacReadRTXLidarData:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Read RTX Lidar Point Data
    :keywords: lang-en omnigraph node isaacSensor sensor isaac-read-r-t-x-lidar-data


Isaac Read RTX Lidar Point Data
===============================

.. <description>

This node reads the data straight from the an RTX Lidar sensor.

.. </description>


Installation
------------

To use this node enable :ref:`omni.isaac.sensor<ext_omni_isaac_sensor>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Buffer Size (*inputs:bufferSize*)", "``uint64``", "number of bytes in dataPtr", "0"
    "Cuda Device Index (*inputs:cudaDeviceIndex*)", "``int``", "Index of the device where the data lives (-1 for host data)", "-1"
    "Cuda Stream (*inputs:cudaStream*)", "``uint64``", "Cuda Stream dataPtr is on if cudaDeviceIndex > -1", "0"
    "Data Pointer (*inputs:dataPtr*)", "``uint64``", "Pointer to LiDAR render result.", "0"
    "Exec (*inputs:exec*)", "``execution``", "The input execution port", "None"
    "Keep Only Positive Distance (*inputs:keepOnlyPositiveDistance*)", "``bool``", "Keep points only if the return distance is > 0", "False"
    "Render Product Path (*inputs:renderProductPath*)", "``token``", "Config is gotten from this", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Azimuths (*outputs:azimuths*)", "``float[]``", "azimuth in deg [0, 360)", "[]"
    "Channels (*outputs:channels*)", "``uint[]``", "channel of point", "[]"
    "Delta Times (*outputs:deltaTimes*)", "``uint[]``", "delta time in ns from the head (relative to tick timestamp)", "[]"
    "Depth Range (*outputs:depthRange*)", "``float[2]``", "The min and max range for sensor to detect a hit [min, max]", "[0, 0]"
    "Distances (*outputs:distances*)", "``float[]``", "distance in m", "[]"
    "Echos (*outputs:echos*)", "``uchar[]``", "echo id in ascending order", "[]"
    "Elevations (*outputs:elevations*)", "``float[]``", "elevation in deg [-90, 90]", "[]"
    "Emitter Ids (*outputs:emitterIds*)", "``uint[]``", "beam/laser detector id", "[]"
    "Exec (*outputs:exec*)", "``execution``", "Output execution triggers when lidar sensor has data", "None"
    "Flags (*outputs:flags*)", "``uchar[]``", "flags", "[]"
    "Frame Id (*outputs:frameId*)", "``uint64``", "The frameId of the current render", "0"
    "Hit Point Normals (*outputs:hitPointNormals*)", "``pointf[3][]``", "hit point Normal", "[]"
    "Intensities (*outputs:intensities*)", "``float[]``", "intensity [0,1]", "[]"
    "Material Ids (*outputs:materialIds*)", "``uint[]``", "hit point material id", "[]"
    "Num Beams (*outputs:numBeams*)", "``uint64``", "The number of lidar beams being output", "0"
    "Object Ids (*outputs:objectIds*)", "``uint[]``", "hit point object id", "[]"
    "Tick States (*outputs:tickStates*)", "``uchar[]``", "emitter state the tick belongs to", "[]"
    "Ticks (*outputs:ticks*)", "``uint[]``", "tick of point", "[]"
    "Timestamp Ns (*outputs:timestampNs*)", "``uint64``", "The time in nanoseconds of the start of frame", "0"
    "Transform (*outputs:transform*)", "``matrixd[4]``", "The transform matrix from lidar to world coordinates at the end of the frame", "None"
    "Transform Start (*outputs:transformStart*)", "``matrixd[4]``", "The transform matrix from lidar to world coordinates at the start of the frame", "None"
    "Velocities (*outputs:velocities*)", "``pointf[3][]``", "velocity at hit point in sensor coordinates [m/s]", "[]"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.isaac.sensor.IsaacReadRTXLidarData"
    "Version", "1"
    "Extension", "omni.isaac.sensor"
    "Icon", "ogn/icons/omni.isaac.sensor.IsaacReadRTXLidarData.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Read RTX Lidar Point Data"
    "Categories", "isaacSensor"
    "Generated Class Name", "OgnIsaacReadRTXLidarDataDatabase"
    "Python Module", "omni.isaac.sensor"

