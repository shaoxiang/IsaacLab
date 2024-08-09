# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import ctypes


class float3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class float4(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("w", ctypes.c_float),
    ]


class FrameAtTime(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestampNs", ctypes.c_uint64),
        ("orientation", float4),
        ("posM", float3),
        ("padding", ctypes.c_uint8 * 4),
    ]


class LidarAuxiliaryData(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("scanComplete", ctypes.c_uint32),  # Whether the scan is complete.
        ("azimuthOffset", ctypes.c_float),  # The offset to +x in radians for specific sensors.
        ("filledAuxMembers", ctypes.c_uint32),  # Which auxiliary data is filled.
        ("emitterId", ctypes.POINTER(ctypes.c_uint32)),  # The emitter ID.
        ("channelId", ctypes.POINTER(ctypes.c_uint32)),  # The channel ID.
        ("echoId", ctypes.POINTER(ctypes.c_uint8)),  # The echo ID.
        ("matId", ctypes.POINTER(ctypes.c_uint32)),  # The material ID.
        ("objId", ctypes.POINTER(ctypes.c_uint32)),  # The object ID.
        ("tickId", ctypes.POINTER(ctypes.c_uint32)),  # The tick ID.
        ("tickStates", ctypes.POINTER(ctypes.c_uint8)),  # The tick states.
        ("hitNormals", ctypes.POINTER(ctypes.c_float)),  # The hit normals.
        ("velocities", ctypes.POINTER(ctypes.c_float)),  # The velocities.
    ]

    def fill(self, ptr, numElements):
        self.scanComplete = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_uint32)
        self.azimuthOffset = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_float)
        self.filledAuxMembers = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_uint32)
        if self.filledAuxMembers & (1 << 0) == (1 << 0):
            self.emitterId = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))
            ptr = ptr + ctypes.sizeof(ctypes.c_uint32) * numElements
        if self.filledAuxMembers & (1 << 1) == (1 << 1):
            self.channelId = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))
            ptr = ptr + ctypes.sizeof(ctypes.c_uint32) * numElements
        if self.filledAuxMembers & (1 << 2) == (1 << 2):
            self.echoId = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8))
            ptr = ptr + ctypes.sizeof(ctypes.c_uint8) * numElements
        if self.filledAuxMembers & (1 << 3) == (1 << 3):
            self.matId = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))
            ptr = ptr + ctypes.sizeof(ctypes.c_uint32) * numElements
        if self.filledAuxMembers & (1 << 4) == (1 << 4):
            self.objId = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))
            ptr = ptr + ctypes.sizeof(ctypes.c_uint32) * numElements
        if self.filledAuxMembers & (1 << 5) == (1 << 5):
            self.tickId = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))
            ptr = ptr + ctypes.sizeof(ctypes.c_uint32) * numElements
        if self.filledAuxMembers & (1 << 6) == (1 << 6):
            self.tickStates = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8))
            ptr = ptr + ctypes.sizeof(ctypes.c_uint8)
        if self.filledAuxMembers & (1 << 7) == (1 << 7):
            self.hitNormals = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
            ptr = ptr + ctypes.sizeof(ctypes.c_float) * 3 * numElements
        if self.filledAuxMembers & (1 << 8) == (1 << 8):
            self.velocities = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
            ptr = ptr + ctypes.sizeof(ctypes.c_float) * 3 * numElements
        return ptr


class RadarAuxiliaryData(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("sensorID", ctypes.c_uint8),  # The ID of the sensor that generated the scan.
        ("scanIdx", ctypes.c_uint8),  # The scan index for sensors with multi-scan support.
        ("timeStampNS", ctypes.c_uint64),  # The scan timestamp in nanoseconds.
        ("cycleCnt", ctypes.c_uint64),  # The scan cycle count (unique per scan index).
        ("maxRangeM", ctypes.c_float),  # The maximum unambiguous range for the scan.
        ("minVelMps", ctypes.c_float),  # The minimum unambiguous velocity for the scan.
        ("maxVelMps", ctypes.c_float),  # The maximum unambiguous velocity for the scan.
        ("minAzRad", ctypes.c_float),  # The minimum unambiguous azimuth for the scan.
        ("maxAzRad", ctypes.c_float),  # The maximum unambiguous azimuth for the scan.
        ("minElRad", ctypes.c_float),  # The minimum unambiguous elevation for the scan.
        ("maxElRad", ctypes.c_float),  # The maximum unambiguous elevation for the scan.
        ("numDetections", ctypes.c_uint32),  # The number of detections.
        ("filledAuxMembers", ctypes.c_uint32),  # Which auxiliary data is filled.
        ("rv_ms", ctypes.POINTER(ctypes.c_float)),  # The radial velocity (m/s), always filled.
        ("semId", ctypes.POINTER(ctypes.c_uint32)),  # The SEM ID, optional.
        ("matId", ctypes.POINTER(ctypes.c_uint32)),  # The material ID, optional.
        ("objId", ctypes.POINTER(ctypes.c_uint32)),  # The object ID, optional.
    ]

    def fill(self, ptr, numElements):
        self.sensorID = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_uint8)
        self.scanIdx = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_uint8)
        self.timeStampNs = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint64)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_uint64)
        self.cycleCnt = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint64)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_uint64)
        self.maxRangeM = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_float)
        self.minVelMps = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_float)
        self.maxVelMps = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_float)
        self.minAzRad = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_float)
        self.maxAzRad = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_float)
        self.minElRad = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_float)
        self.maxElRad = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_float)
        self.numDetections = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_uint32)
        self.filledAuxMembers = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32)).contents
        ptr = ptr + ctypes.sizeof(ctypes.c_uint32)
        self.rv_ms = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        ptr = ptr + ctypes.sizeof(ctypes.c_float) * numElements
        if self.filledAuxMembers & (1 << 0) == (1 << 0):
            self.semId = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))
            ptr = ptr + ctypes.sizeof(ctypes.c_uint32) * numElements
        if self.filledAuxMembers & (1 << 1) == (1 << 1):
            self.matId = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))
            ptr = ptr + ctypes.sizeof(ctypes.c_uint32) * numElements
        if self.filledAuxMembers & (1 << 2) == (1 << 2):
            self.objId = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32))
            ptr = ptr + ctypes.sizeof(ctypes.c_uint8) * numElements
        return ptr


class BasicElements(ctypes.Structure):
    _fields_ = [
        ("timeOffsetNs", ctypes.POINTER(ctypes.c_int32)),  # Time offset from the start of the point cloud
        ("x", ctypes.POINTER(ctypes.c_float)),  # azimuth in degree [-180,180] or cartesian x in m
        ("y", ctypes.POINTER(ctypes.c_float)),  # elevation in degree or cartesian y in m
        ("z", ctypes.POINTER(ctypes.c_float)),  # distance in m or cartesian z in m
        ("scalar", ctypes.POINTER(ctypes.c_float)),  # sensor specific scalar
        ("flags", ctypes.POINTER(ctypes.c_uint8)),  # sensor specific flags
    ]

    def fill(self, ptr, numElements):
        self.timeOffsetNs = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int32))
        ptr = ptr + ctypes.sizeof(ctypes.c_int32) * numElements
        self.x = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        ptr = ptr + ctypes.sizeof(ctypes.c_float) * numElements
        self.y = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        ptr = ptr + ctypes.sizeof(ctypes.c_float) * numElements
        self.z = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        ptr = ptr + ctypes.sizeof(ctypes.c_float) * numElements
        self.scalar = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        ptr = ptr + ctypes.sizeof(ctypes.c_float) * numElements
        self.flags = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8))
        ptr = ptr + ctypes.sizeof(ctypes.c_uint8) * numElements
        return ptr


class GenericModelOutput(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        (
            "magicNumber",
            ctypes.c_uint32,
        ),  # A unique identifier for the output. Should reflect MAGIC_NUMBER_GMO, which is the ASCII for "NGMO".
        ("majorVersion", ctypes.c_uint32),  # The major version number of the model output.
        ("minorVersion", ctypes.c_uint32),  # The minor version number of the model output.
        ("patchVersion", ctypes.c_uint32),  # The number of elements in the array members of the model output.
        ("numElements", ctypes.c_uint32),  # The patch version number of the model output.
        (
            "frameOfReference",
            ctypes.c_uint32,
        ),  # The frame of reference for the model output. The default value is ``FrameOfReference::SENSOR``.
        ("frameId", ctypes.c_uint64),  # The model (simulation) frame ID of the model output.
        ("timestampNs", ctypes.c_uint64),  # The timestamp of the model output in nanoseconds.
        (
            "coordsType",
            ctypes.c_uint32,
        ),  # The type of coordinates used in the model output. The default value is ``CoordsType::SPHERICAL``.
        ("outputType", ctypes.c_uint32),  # The type of output. The default value is ``OutputType::POINTCLOUD``.
        (
            "modelToAppTransform",
            ctypes.c_float * 16,
        ),  # A transformation matrix that transforms from the model's coordinate system to the application's coordinate system.
        (
            "frameStart",
            FrameAtTime,
        ),  # The start frame of the model output. It transforms from the model's coordinate system to the global coordinate system at frame start time.
        (
            "frameEnd",
            FrameAtTime,
        ),  # The end frame of the model output. It transforms from the model's coordinate system to the global coordinate system at frame end time. See below for more information.
        (
            "auxType",
            ctypes.c_uint32,
        ),  # The modality specific type of auxiliary data. The default value is ``AuxType::NONE``. See below for more information.
        ("padding", ctypes.c_uint8 * 4),  # Padding to align the structure to a multiple of 8 bytes.
        ("elements", BasicElements),  # The basic elements of the model output. See below for more information.
        (
            "auxiliaryData",
            ctypes.c_void_p,
        ),  # A pointer to the auxiliary data. This may not be filled. See below for more information.
    ]
