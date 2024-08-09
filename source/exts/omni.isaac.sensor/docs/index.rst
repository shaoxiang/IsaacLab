Isaac Sensor Extension [omni.isaac.sensor]
######################################################


The Isaac Sensor Extension provides a set of simulated physics based sensors like contact sensor, inertial measurement unit (IMU) sensor, effort sensor, RTX lidar, and interfaces to access them in the simulator


Contact Sensor
==============

.. automodule:: omni.isaac.sensor.scripts.contact_sensor
    :inherited-members:
    :members:
    :undoc-members:
    :exclude-members:

IMU sensor
============

.. automodule:: omni.isaac.sensor.scripts.imu_sensor
    :inherited-members:
    :members:
    :undoc-members:
    :exclude-members:

Effort sensor
=============

.. automodule:: omni.isaac.sensor.scripts.effort_sensor
    :inherited-members:
    :members:
    :undoc-members:
    :exclude-members:

Lidar RTX sensor
================

.. automodule:: omni.isaac.sensor.scripts.lidar_rtx
    :inherited-members:
    :members:
    :undoc-members:
    :exclude-members:

Rotating Lidar PhysX sensor
============================

.. automodule:: omni.isaac.sensor.scripts.rotating_lidar_physX
    :inherited-members:
    :members:
    :undoc-members:
    :exclude-members:

Camera sensor
===============

.. automodule:: omni.isaac.sensor.scripts.camera
    :inherited-members:
    :members:
    :undoc-members:
    :exclude-members:

Contact Sensor Interface
========================

This submodule  provides an interface to a simulated contact sensor. A simplified command is provided to create a contact sensor in the stage:

.. automethod:: omni.isaac.sensor.scripts.commands.IsaacSensorCreateContactSensor        

Once the contact sensor is created, you must first acquire this interface and then you can use this interface to access the contact sensor

Also, the offset of the contact sensor is also affect by the parent's transformations.

.. code-block:: python
    :linenos:

    from omni.isaac.sensor import _sensor
    _cs = _sensor.acquire_contact_sensor_interface()

.. note::
    if the contact sensor is not initially created under a valid rigid body parent, the contact sensor will not output any valid data even if the contact sensor is later attached to a valid rigid body parent.

Acquiring Extension Interface
-------------------------------

.. automethod:: omni.isaac.sensor._sensor.acquire_contact_sensor_interface
.. automethod:: omni.isaac.sensor._sensor.release_contact_sensor_interface

To collect the most recent reading, call the interface `get_sensor_reading(/path/to/sensor, use_latest_data=True)`. The result will be most recent sensor reading.

.. code-block:: python

    reading = _cs.get_sensor_reading("/World/Cube/Contact_Sensor", use_latest_data)

To collect the reading at the last sensor measurement time based on the sensor period, call the interface `get_sensor_reading(/path/to/sensor)`. This will give you the physics step data closest to the sensor measurement time.

.. code-block:: python

    reading = _cs.get_sensor_reading("/World/Cube/Contact_Sensor")

To collect raw reading, call the interface `get_contact_sensor_raw_data(/path/to/sensor)`. The result will return a list of raw contact data for that body.

.. code-block:: python

    raw_Contact = _cs.get_contact_sensor_raw_data("/World/Cube/Contact_Sensor")


Output Types
-------------
.. autoclass:: omni.isaac.sensor._sensor.CsSensorReading
    :members:
    :undoc-members:
    :exclude-members: 



.. autoclass:: omni.isaac.sensor._sensor.CsRawData
    :members:
    :undoc-members:
    :exclude-members: 


Interface Methods
-------------------

.. autoclass:: omni.isaac.sensor._sensor.ContactSensorInterface
    :members:
    :undoc-members:


IMU sensor Interface
====================

This submodule provides an interface to a simulate IMU sensor, which provides ground truth linear acceleration, angular velocity, orientation data.

A simplified command is provided to create an IMU sensor:

.. automethod:: omni.isaac.sensor.scripts.commands.IsaacSensorCreateImuSensor        

Similiarly, once an IMU sensor is created, you can use this interface to interact with the simulated IMU sensor. 
You must first call the acquire_imu_sensor_interface. 

.. code-block:: python
    :linenos:

    from omni.isaac.sensor import _sensor
    _is = _sensor.acquire_imu_sensor_interface()

.. note::
    if the IMU sensor is not initially created under a valid rigid body parent, the IMU sensor will not output any valid data even if the IMU sensor is later attached to a valid rigid body parent. Also, the offset and orientation of the IMU sensor is also affect by the parent's transformations.


Acquiring Extension Interface
-------------------------------

.. automethod:: omni.isaac.sensor._sensor.acquire_imu_sensor_interface
.. automethod:: omni.isaac.sensor._sensor.release_imu_sensor_interface

To collect the most recent reading, call the interface `get_sensor_reading(/path/to/sensor, use_latest_data = True)`. The result will be most recent sensor reading.

.. code-block:: python

    reading = _is.get_sensor_reading("/World/Cube/Imu_Sensor", use_latest_data = True)

To collect the reading at the last sensor measurement time based on the sensor period, call the interface `get_sensor_reading(/path/to/sensor)`.

.. code-block:: python

    reading = _is.get_sensor_reading("/World/Cube/Imu_Sensor")

Since the sensor reading time is usually between two physics steps, linear interpolation method is used by default to get the reading at sensor time between the physics steps. However the `get_sensor_reading` can also accept a custom function in the event that a different interpolation strategy is prefered..

.. code-block:: python

    from typing import List

    # Input Param: List of past IsSensorReadings, time of the expected sensor reading 
    def interpolation_function(data:List[_sensor.IsSensorReading], time:float) -> _sensor.IsSensorReading:
        interpolated_reading = _sensor.IsSensorReading()
        # do interpolation
        return interpolated_reading
    
    reading = _is.get_sensor_reading("/World/Cube/Imu_Sensor", interpolation_function = interpolation_function)

.. note::
    The interpolation function will only be used if the sensor frequency is lower than the physics frequency and use_latest_data flag is .

Output Types
--------------
.. autoclass:: omni.isaac.sensor._sensor.IsSensorReading
    :members:
    :undoc-members:
    :exclude-members: 

Interface Methods
------------------

.. autoclass:: omni.isaac.sensor._sensor.ImuSensorInterface
    :members:
    :undoc-members:

.. .. automodule:: omni.isaac.sensor._contact_sensor
..    :platform: Windows-x86_64, Linux-x86_64
..    :members:
..    :undoc-members:
..    :show-inheritance:
..    :imported-members:
..    :exclude-members:

.. .. automodule:: omni.isaac.sensor._imu_sensor
..    :platform: Windows-x86_64, Linux-x86_64
..    :members:
..    :undoc-members:
..    :show-inheritance:
..    :imported-members:
..    :exclude-members:

Effort Sensor
===============

Effort sensor is a python class for reading gronud truth joint effort measurements. The Effort sensor can be created directly in Python using the path of the joint of interest.

.. code-block:: python
    :linenos:

    from omni.isaac.sensor.scripts.effort_sensor import EffortSensor
    import numpy as np

    sensor = EffortSensor(prim_path="/World/Robot/revolute_joint")

.. note::
    If the sensor was created with the incorrect prim path, simply delete the sensor and recreate it. If the measured joint needs to be changed and the new joint has the same parent, ``update_dof_name(dof_name:str)`` function maybe used. 

Acquiring Sensor data
----------------------

To collect the most recent reading, call the interface `get_sensor_reading(use_latest_data = True)`. The result will be most recent sensor reading.

.. code-block:: python

    reading = sensor.get_sensor_reading(use_latest_data = True)

To collect the reading at the last sensor measurement time based on the sensor period, call the interface `get_sensor_reading()`.

.. code-block:: python

    reading = sensor.get_sensor_reading()

Since the sensor reading time is usually between two physics steps, linear interpolation method is used by default to get the reading at sensor time between the physics steps. However the `get_sensor_reading` can also accept a custom function in the event that a different interpolation strategy is prefered..

.. code-block:: python
    :linenos:

    from omni.isaac.sensor.scripts.effort_sensor import EsSensorReading    
    
    # Input Param: List of past EsSensorReading, time of the expected sensor reading 
    def interpolation_function(data, time):
        interpolated_reading = EsSensorReading()
        # do interpolation
        return interpolated_reading

    reading = sensor.get_sensor_reading(interpolation_function)

.. note::
    The interpolation function will only be used if the sensor frequency is lower than the physics frequency and use_latest_data flag is not enabled.

Output Types
---------------
**EsSensorReading**

- time (*float*): The time of the sensor reading.

- value (*float*): The measured effort on the joint.

- is_valid (*boolean*): The validitty of the sensor measurement.

Omnigraph Nodes
=======================

.. include::  ogn.rst