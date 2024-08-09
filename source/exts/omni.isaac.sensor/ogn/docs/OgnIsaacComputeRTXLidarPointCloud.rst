.. _omni_isaac_sensor_IsaacComputeRTXLidarPointCloud_1:

.. _omni_isaac_sensor_IsaacComputeRTXLidarPointCloud:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Compute RTX Lidar Point Cloud Node
    :keywords: lang-en omnigraph node isaacSensor sensor isaac-compute-r-t-x-lidar-point-cloud


Isaac Compute RTX Lidar Point Cloud Node
========================================

.. <description>

This node reads from the an RTX Lidar sensor and holds point cloud data buffers

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
    "LiDAR render result (*inputs:dataPtr*)", "``uint64``", "Pointer to LiDAR render result", "0"
    "Exec (*inputs:exec*)", "``execution``", "The input execution port", "None"
    "Keep Only Positive Distance (*inputs:keepOnlyPositiveDistance*)", "``bool``", "Keep points only if the return distance is > 0", "True"
    "Render Product Path (*inputs:renderProductPath*)", "``token``", "Path of the renderProduct to wait for being rendered", ""


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Azimuth (*outputs:azimuth*)", "``float[]``", "azimuth in rad [-pi,pi]", "[]"
    "Buffer Size (*outputs:bufferSize*)", "``uint64``", "Size (in bytes) of the buffer (0 if the input is a texture)", "None"
    "Cuda Device Index (*outputs:cudaDeviceIndex*)", "``int``", "Index of the device where the data lives (-1 for host data)", "-1"
    "Point Cloud Data (*outputs:dataPtr*)", "``uint64``", "Buffer of points containing point cloud data in Lidar coordinates", "None"
    "Elevation (*outputs:elevation*)", "``float[]``", "elevation in rad [-pi/2, pi/2]", "[]"
    "Exec (*outputs:exec*)", "``execution``", "Output execution triggers when lidar sensor has data", "None"
    "Height (*outputs:height*)", "``uint``", "Height of point cloud buffer, will always return 1", "1"
    "Intensity (*outputs:intensity*)", "``float[]``", "intensity [0,1]", "[]"
    "Range (*outputs:range*)", "``float[]``", "range in m", "[]"
    "Transform (*outputs:transform*)", "``matrixd[4]``", "The transform matrix from lidar to world coordinates", "None"
    "Width (*outputs:width*)", "``uint``", "3 x Width or number of points in point cloud buffer", "0"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.isaac.sensor.IsaacComputeRTXLidarPointCloud"
    "Version", "1"
    "Extension", "omni.isaac.sensor"
    "Icon", "ogn/icons/omni.isaac.sensor.IsaacComputeRTXLidarPointCloud.svg"
    "Has State?", "True"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Compute RTX Lidar Point Cloud Node"
    "Categories", "isaacSensor"
    "Generated Class Name", "OgnIsaacComputeRTXLidarPointCloudDatabase"
    "Python Module", "omni.isaac.sensor"

