// Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include <carb/Defines.h>
#include <carb/Types.h>

#include <usdrt/gf/matrix.h>
// #include <PxActor.h>
#include <limits>

namespace omni
{
namespace isaac
{
namespace sensor
{
/**
 * Properties of a Contact Sensor
 */
struct CsProperties
{
    float radius{ 0.0f }; //<! Radius from the sensor position. negative values indicate it's a full body sensor.
    float minThreshold{ 0.0f }; //<! Minimum force that the sensor can read. Forces below this value will not trigger a
                                // reading.
    float maxThreshold{ 0.0f }; //<! Maximum force that the sensor can register. Forces above this value will be
                                // clamped.
    float sensorPeriod{ 0.0f }; //<! Sensor reading speed, in seconds. Zero means sync with simulation timestep.
};
/**
 * Contact Sensor Reading
 */
struct CsReading
{
    float time{ 0.0f }; //<! Simulation Timestamp for contact sensor reading
    float value{ 0.0f }; //<! Reading value, in N
    bool inContact{ false }; //<! Flag that checks if the sensor is in contact with something or not
    bool is_valid{ false }; //<! Validity of the data. False for when the sensor is disabled, true for enabled
};

/**
 * Simulation Contact Raw Data
 */
struct CsRawData
{
    float time{ 0.0f }; //<! Simulation timestamp
    float dt{ 0.0f }; //<! Simulation time step for the impulse.
    uint64_t body0; //<! First body on contact
    uint64_t body1; //<! Second body on contact
    carb::Float3 position{ 0.0f, 0.0f, 0.0f }; //<! Contact Position, in world coordinates
    carb::Float3 normal{ 0.0f, 0.0f, 0.0f }; //<! Contact Normal, in world coordinates
    carb::Float3 impulse{ 0.0f, 0.0f, 0.0f }; //<! Contact Impulse, in world coordinates
};


/**
 * Properties of a IMU Sensor
 */
struct IsProperties
{
    usdrt::GfMatrix3d orientation; //<! orientation matrix relative to the parent body where the sensor is placed.
    float sensorPeriod{ 0.0f }; //<! Sensor reading speed, in seconds. Zero means sync with simulation timestep.
};
/**
 * IMU Sensor Reading
 */
struct IsReading
{
    float time{ 0.0f }; //<! Simulation Timestamp for IMU sensor reading
    float lin_acc_x{ 0.0f }; //<! Accelerometer reading value x axis, in m/s^2
    float lin_acc_y{ 0.0f }; //<! Accelerometer reading value y axis, in m/s^2
    float lin_acc_z{ 0.0f }; //<! Accelerometer reading value z axis, in m/s^2
    float ang_vel_x{ 0.0f }; //<! Gyroscope reading value x axis, in rad/s
    float ang_vel_y{ 0.0f }; //<! Gyroscope reading value y axis, in rad/s
    float ang_vel_z{ 0.0f }; //<! Gyroscope reading value z axis, in rad/s
    carb::Float4 orientation{ 0.0f, 0.0f, 0.0f, 0.0f }; //<! quaternion orientation of parent body
    bool is_valid{ false }; //<! Validity of the data. False for when the sensor is disabled, true for enabled
};

/**
 * Simulation IMU Raw Data
 */
struct IsRawData
{
    float time{ 0.0f }; //<! Simulation timestamp
    float dt{ 0.0f }; //<! Simulation time step for the impulse.
    float lin_vel_x{ 0.0f }; //<! linear velocity x raw reading value, in m/s
    float lin_vel_y{ 0.0f }; //<! linear velocity y raw reading value, in m/s
    float lin_vel_z{ 0.0f }; //<! linear velocity z raw reading value, in m/s
    float ang_vel_x{ 0.0f }; //<! angular velocity x raw reading value, in rad/s
    float ang_vel_y{ 0.0f }; //<! angular velocity y raw reading value, in rad/s
    float ang_vel_z{ 0.0f }; //<! angular velocity z raw reading value, in rad/s
    carb::Float4 orientation{ 0.0f, 0.0f, 0.0f, 0.0f }; //<! quaternion orientation of parent body
};

}
}
}
