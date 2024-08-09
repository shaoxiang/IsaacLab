// Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include "IsaacSensorTypes.h"


namespace omni
{
namespace isaac
{
namespace sensor
{

struct ContactSensorInterface
{
    CARB_PLUGIN_INTERFACE("omni::isaac::sensor::ContactSensorInterface", 0, 1);


    //! Gets contact raw data from physics engine
    /*! Gets Contact raw data, for validation purposes and ground truth
     * \param primPath path of the sensor prim
     * \param numContacts size of contacts
     * \return Raw Data
     */
    CsRawData*(CARB_ABI* getSensorRawData)(const char* primPath, size_t& numContacts);

    //! Gets Sensor latest simulation
    /*! Check is the prim path contact sensor
     * \param primPath path of the sensor prim
     * \param getLatestValue boolean flag for getting the latest sim value or the last sensor measured value
     * \return time-stamped sensor values
     */
    CsReading(CARB_ABI* getSensorReading)(const char* primPath, const bool& getLatestValue);

    bool(CARB_ABI* isContactSensor)(const char* primPath);

    const char*(CARB_ABI* decodeBodyName)(uint64_t body);

    //! Gets contact raw data of a rigid body with contact report API from physics engine
    /*! Gets Contact raw data, for validation purposes and ground truth
     * \param primPath path of the rigid body prim
     * \param numContacts size of contacts
     * \return Raw Data
     */
    CsRawData*(CARB_ABI* getBodyRawData)(const char* primPath, size_t& numContacts);
};


struct ImuSensorInterface
{
    CARB_PLUGIN_INTERFACE("omni::isaac::sensor::ImuSensorInterface", 0, 1);

    //! Gets Sensor last reading
    /*! Gets the sensor last reading on its latest sensor period

     * \param usdPath sensor prim path
     * \param interpolationFunction interpolation functional pointer
     * \param getLatestValue flag for getting the latest sim value or the last sensor measured value

     * \return time-stamped sensor values.
     */
    IsReading(CARB_ABI* getSensorReading)(
        const char* usdPath,
        const std::function<omni::isaac::sensor::IsReading(std::vector<omni::isaac::sensor::IsReading>, float)>&
            interpolateFunction,
        const bool& getLatestValue,
        const bool& readGravity);

    //! Check is Prim an ImuSensorSchema
    /*! Return True for is, False for is not an ImuSensorSchema
     * \param usdPath sensor prim path
     * \return true for is, false for is not an ImuSensorSchema
     */
    bool(CARB_ABI* isImuSensor)(const char* usdPath);
};

}
}
}
