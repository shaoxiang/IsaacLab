.. _omni_isaac_sensor_IsaacReadIMU_1:

.. _omni_isaac_sensor_IsaacReadIMU:

.. ================================================================================
.. THIS PAGE IS AUTO-GENERATED. DO NOT MANUALLY EDIT.
.. ================================================================================

:orphan:

.. meta::
    :title: Isaac Read IMU Node
    :keywords: lang-en omnigraph node isaacSensor sensor isaac-read-i-m-u


Isaac Read IMU Node
===================

.. <description>

Node that reads out IMU linear acceleration, angular velocity and orientation data

.. </description>


Installation
------------

To use this node enable :ref:`omni.isaac.sensor<ext_omni_isaac_sensor>` in the Extension Manager.


Inputs
------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Exec In (*inputs:execIn*)", "``execution``", "The input execution port", "None"
    "IMU Prim (*inputs:imuPrim*)", "``target``", "Usd prim reference to the IMU prim", "None"
    "Read Gravity (*inputs:readGravity*)", "``bool``", "True to read gravitational acceleration in the measurement, False to ignore gravitational acceleration", "True"
    "Use Latest Data (*inputs:useLatestData*)", "``bool``", "True to use the latest data from the physics step, False to use the data measured by the sensor", "False"


Outputs
-------
.. csv-table::
    :header: "Name", "Type", "Descripton", "Default"
    :widths: 20, 20, 50, 10

    "Angular Velocity Vector (*outputs:angVel*)", "``vectord[3]``", "Angular velocity IMU reading", "[0.0, 0.0, 0.0]"
    "Exec Out (*outputs:execOut*)", "``execution``", "Output execution triggers when sensor has data", "None"
    "Linear Acceleration Vector (*outputs:linAcc*)", "``vectord[3]``", "Linear acceleration IMU reading", "[0.0, 0.0, 0.0]"
    "Sensor Orientation Quaternion (*outputs:orientation*)", "``quatd[4]``", "Sensor orientation as quaternion", "[0.0, 0.0, 0.0, 1.0]"
    "Sensor Time (*outputs:sensorTime*)", "``float``", "Timestamp of the sensor reading", "0"


Metadata
--------
.. csv-table::
    :header: "Name", "Value"
    :widths: 30,70

    "Unique ID", "omni.isaac.sensor.IsaacReadIMU"
    "Version", "1"
    "Extension", "omni.isaac.sensor"
    "Icon", "ogn/icons/omni.isaac.sensor.IsaacReadIMU.svg"
    "Has State?", "False"
    "Implementation Language", "C++"
    "Default Memory Type", "cpu"
    "Generated Code Exclusions", "None"
    "uiName", "Isaac Read IMU Node"
    "Categories", "isaacSensor"
    "Generated Class Name", "OgnIsaacReadIMUDatabase"
    "Python Module", "omni.isaac.sensor"

