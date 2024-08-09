.. _omni_isaac_sensor_IsaacComputeRTXLidarFlatScan_2:

.. _omni_isaac_sensor_IsaacComputeRTXLidarFlatScan:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Compute RTX Lidar Flat Scan Node
    :keywords: lang-en omnigraph node isaacSensor sensor isaac-compute-r-t-x-lidar-flat-scan


Isaac Compute RTX Lidar Flat Scan Node
======================================

.. <description>

Accumulates full scan from the lowest elevation emitter on an RTX Lidar

.. </description>


Installation
------------

To use this node enable :ref:`omni.isaac.sensor<ext_omni_isaac_sensor>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Buffer Size (*inputs:bufferSize*)", "``uint64``", "Size (in bytes) of the buffer (0 if the input is a texture)", "0"
    "Data Pointer (*inputs:dataPtr*)", "``uint64``", "Pointer to LiDAR render result.", "0"
    "Exec (*inputs:exec*)", "``execution``", "The input execution port", "None"
    "Render Product Path (*inputs:renderProductPath*)", "``token``", "Used to retrieve lidar configuration.", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Azimuth Range (*outputs:azimuthRange*)", "``float[2]``", "The azimuth range [min, max] (deg). Always [-180, 180] for rotary lidars.", "[0.0, 0.0]"
    "Depth Range (*outputs:depthRange*)", "``float[2]``", "Range for sensor to detect a hit [min, max] (m)", "[0, 0]"
    "Exec (*outputs:exec*)", "``execution``", "Output execution triggers when lidar sensor has accumulated a full scan.", "None"
    "Horizontal Fov (*outputs:horizontalFov*)", "``float``", "Horizontal Field of View (deg)", "0"
    "Horizontal Resolution (*outputs:horizontalResolution*)", "``float``", "Increment between horizontal rays (deg)", "0"
    "Intensities Data (*outputs:intensitiesData*)", "``uchar[]``", "Intensity measurements from full scan, ordered by increasing azimuth", "[]"
    "Linear Depth Data (*outputs:linearDepthData*)", "``float[]``", "Linear depth measurements from full scan, ordered by increasing azimuth (m)", "[]"
    "Num Cols (*outputs:numCols*)", "``int``", "Number of columns in buffers", "0"
    "Num Rows (*outputs:numRows*)", "``int``", "Number of rows in buffers", "0"
    "Rotation Rate (*outputs:rotationRate*)", "``float``", "Rotation rate of sensor in Hz", "0"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.isaac.sensor.IsaacComputeRTXLidarFlatScan"
    "Version", "2"
    "Extension", "omni.isaac.sensor"
    "Icon", "ogn/icons/omni.isaac.sensor.IsaacComputeRTXLidarFlatScan.svg"
    "Has State?", "True"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Compute RTX Lidar Flat Scan Node"
    "Categories", "isaacSensor"
    "Generated Class Name", "OgnIsaacComputeRTXLidarFlatScanDatabase"
    "Python Module", "omni.isaac.sensor"

