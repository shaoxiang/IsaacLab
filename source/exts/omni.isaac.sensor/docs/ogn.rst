


IsaacComputeRTXLidarFlatScan
----------------------------
    ['Accumulates full scan from the lowest elevation emitter on an RTX Lidar']


**Inputs**
    - **exec** (*execution*): The input execution port.
    - **dataPtr** (*uint64*): Pointer to LiDAR render result.
    - **bufferSize** (*uint64*): Size (in bytes) of the buffer (0 if the input is a texture).
    - **renderProductPath** (*token*): Used to retrieve lidar configuration.

**Outputs**
    - **exec** (*execution*): Output execution triggers when lidar sensor has accumulated a full scan.
    - **horizontalFov** (*float*): Horizontal Field of View (deg).
    - **horizontalResolution** (*float*): Increment between horizontal rays (deg).
    - **depthRange** (*float[2]*): Range for sensor to detect a hit [min, max] (m).
    - **rotationRate** (*float*): Rotation rate of sensor in Hz.
    - **linearDepthData** (*float[]*): Linear depth measurements from full scan, ordered by increasing azimuth (m).
    - **intensitiesData** (*uchar[]*): Intensity measurements from full scan, ordered by increasing azimuth.
    - **numRows** (*int*): Number of rows in buffers.
    - **numCols** (*int*): Number of columns in buffers.
    - **azimuthRange** (*float[2]*): The azimuth range [min, max] (deg). Always [-180, 180] for rotary lidars.


IsaacComputeRTXLidarPointCloud
------------------------------
    ['This node reads from the an RTX Lidar sensor and holds point cloud data buffers']


**Inputs**
    - **exec** (*execution*): The input execution port.
    - **dataPtr** (*uint64*): Pointer to LiDAR render result.
    - **keepOnlyPositiveDistance** (*bool*): Keep points only if the return distance is > 0. Default to True.
    - **accuracyErrorAzimuthDeg** (*float*): Accuracy error of azimuth in degrees applied to all points equally.
    - **accuracyErrorElevationDeg** (*float*): Accuracy error of elevation in degrees applied to all points equally.
    - **accuracyErrorPosition** (*float[3]*): Position offset applied to all points equally.
    - **renderProductPath** (*token*): Path of the renderProduct to wait for being rendered.

**Outputs**
    - **exec** (*execution*): Output execution triggers when lidar sensor has data.
    - **transform** (*matrixd[4]*): The transform matrix from lidar to world coordinates.
    - **dataPtr** (*uint64*): Buffer of points containing point cloud data in Lidar coordinates.
    - **cudaDeviceIndex** (*int*): Index of the device where the data lives (-1 for host data).
    - **bufferSize** (*uint64*): Size (in bytes) of the buffer (0 if the input is a texture).
    - **intensity** (*float[]*): intensity [0,1].
    - **range** (*float[]*): range in m.
    - **azimuth** (*float[]*): azimuth in rad [-pi,pi].
    - **elevation** (*float[]*): elevation in rad [-pi/2, pi/2].
    - **height** (*uint*): Height of point cloud buffer, will always return 1.
    - **width** (*uint*): 3 x Width or number of points in point cloud buffer.


IsaacComputeRTXRadarPointCloud
------------------------------
    ['This node reads from the an RTX Radar sensor and holds point cloud data buffers']


**Inputs**
    - **exec** (*execution*): The input execution port.
    - **dataPtr** (*uint64*): Pointer to Radar render result.
    - **renderProductPath** (*token*): Path of the renderProduct to wait for being rendered.

**Outputs**
    - **exec** (*execution*): Output execution triggers when Radar sensor has data.
    - **transform** (*matrixd[4]*): The input matrix transformed from Radar to World.
    - **sensorID** (*uchar*): Sensor Id for sensor that generated the scan.
    - **scanIdx** (*uchar*): Scan index for sensors with multi scan support.
    - **timeStampNS** (*uint64*): Scan timestamp in nanoseconds.
    - **cycleCnt** (*uint64*): Scan cycle count.
    - **maxRangeM** (*float*): The max unambiguous range for the scan.
    - **minVelMps** (*float*): The min unambiguous velocity for the scan.
    - **maxVelMps** (*float*): The max unambiguous velocity for the scan.
    - **minAzRad** (*float*): The min unambiguous azimuth for the scan.
    - **maxAzRad** (*float*): The max unambiguous azimuth for the scan.
    - **minElRad** (*float*): The min unambiguous elevation for the scan.
    - **maxElRad** (*float*): The max unambiguous elevation for the scan.
    - **numDetections** (*uint*): The number of valid detections in the array.
    - **dataPtr** (*uint64*): Buffer of 3d points containing point cloud data in Radar coordinates.
    - **cudaDeviceIndex** (*int*): Index of the device where the data lives (-1 for host data).
    - **bufferSize** (*uint64*): Size (in bytes) of the buffer (0 if the input is a texture).
    - **height** (*uint*): Height of point cloud buffer, will always return 1.
    - **width** (*uint*): 3 x Width or number of points in point cloud buffer.
    - **radialDistance** (*float[]*): Radial distance (m).
    - **radialVelocity** (*float[]*): Radial velocity (m/s).
    - **azimuth** (*float[]*): Azimuth angle (radians).
    - **elevation** (*float[]*): Angle of elevation (radians).
    - **rcs** (*float[]*): Radar cross section in decibels referenced to a square meter (dBsm).
    - **semanticId** (*uint[]*): semantic ID.
    - **materialId** (*uint[]*): material ID.
    - **objectId** (*uint[]*): object ID.


IsaacCreateRTXLidarScanBuffer
-----------------------------
    ['This node creates a full scan buffer for RTX Lidar sensor.']


**Inputs**
    - **exec** (*execution*): The input execution port.
    - **dataPtr** (*uint64*): Pointer to LiDAR render result.
    - **cudaDeviceIndex** (*int*): Index of the device where the data lives (-1 for host data). Default to -1.
    - **keepOnlyPositiveDistance** (*bool*): Keep points only if the return distance is > 0. Default to True.
    - **transformPoints** (*bool*): Transform point cloud to world coordinates. Default to False.
    - **accuracyErrorAzimuthDeg** (*float*): Accuracy error of azimuth in degrees applied to all points equally.
    - **accuracyErrorElevationDeg** (*float*): Accuracy error of elevation in degrees applied to all points equally.
    - **accuracyErrorPosition** (*float[3]*): Position offset applied to all points equally.
    - **renderProductPath** (*token*): Config is gotten from this.
    - **outputIntensity** (*bool*): Create an output array for the Intensity. Default to True.
    - **outputDistance** (*bool*): Create an output array for the Distance. Default to True.
    - **outputObjectId** (*bool*): Create an output array for the ObjectId. Default to False.
    - **outputVelocity** (*bool*): Create an output array for the Velocity. Default to False.
    - **outputAzimuth** (*bool*): Create an output array for the Azimuth. Default to False.
    - **outputElevation** (*bool*): Create an output array for the Elevation. Default to False.
    - **outputNormal** (*bool*): Create an output array for the Normals. Default to False.
    - **outputTimestamp** (*bool*): Create an output array for the Timestamp. Default to False.
    - **outputEmitterId** (*bool*): Create an output array for the EmitterId. Default to False.
    - **outputBeamId** (*bool*): Create an output array for the BeamId. Default to False.
    - **outputMaterialId** (*bool*): Create an output array for the MaterialId. Default to False.

**Outputs**
    - **exec** (*execution*): Output execution triggers when lidar sensor has data.
    - **dataPtr** (*uint64*): Pointer to LiDAR render result.
    - **cudaDeviceIndex** (*int*): Index of the device where the data lives (-1 for host data).
    - **bufferSize** (*uint64*): Size (in bytes) of the buffer (0 if the input is a texture).
    - **transform** (*matrixd[4]*): The transform matrix from lidar to world coordinates.
    - **intensityPtr** (*uint64*): intensity [0,1].
    - **intensityDataType** (*float*): type.
    - **intensityBufferSize** (*uint64*): size.
    - **distancePtr** (*uint64*): range in m.
    - **distanceDataType** (*float*): type.
    - **distanceBufferSize** (*uint64*): size.
    - **azimuthPtr** (*uint64*): azimuth in rad [-pi,pi].
    - **azimuthDataType** (*float*): type.
    - **azimuthBufferSize** (*uint64*): size.
    - **elevationPtr** (*uint64*): elevation in rad [-pi/2, pi/2].
    - **elevationDataType** (*float*): type.
    - **elevationBufferSize** (*uint64*): size.
    - **velocityPtr** (*uint64*): elevation in rad [-pi/2, pi/2].
    - **velocityDataType** (*float[3]*): type.
    - **velocityBufferSize** (*uint64*): size.
    - **objectIdPtr** (*uint64*): ObjectId for getting usd prim information.
    - **objectIdDataType** (*uint*): type.
    - **objectIdBufferSize** (*uint64*): size.
    - **normalPtr** (*uint64*): Normal at the hit location.
    - **normalDataType** (*float[3]*): type.
    - **normalBufferSize** (*uint64*): size.
    - **timestampPtr** (*uint64*): timestamp in ns.
    - **timestampDataType** (*uint64*): type.
    - **timestampBufferSize** (*uint64*): size.
    - **emitterIdPtr** (*uint64*): emitterId.
    - **emitterIdDataType** (*uint*): type.
    - **emitterIdBufferSize** (*uint64*): size.
    - **beamIdPtr** (*uint64*): beamId.
    - **beamIdDataType** (*uint*): type.
    - **beamIdBufferSize** (*uint64*): size.
    - **materialIdPtr** (*uint64*): materialId at hit location.
    - **materialIdDataType** (*uint*): type.
    - **materialIdBufferSize** (*uint64*): size.
    - **indexPtr** (*uint64*): Index into the full array if keepOnlyPositiveDistance ((startTick+tick)*numChannels*numEchos + channel*numEchos + echo).
    - **indexDataType** (*uint*): type.
    - **indexBufferSize** (*uint64*): size.
    - **numReturnsPerScan** (*uint*): Number of returns in the full scan.
    - **ticksPerScan** (*uint*): Number of ticks in a full scan.
    - **numChannels** (*uint*): Number of channels of the lidar.
    - **numEchos** (*uint*): Number of echos of the lidar.
    - **renderProductPath** (*token*): Config is gotten from this.
    - **height** (*uint*): Height of point cloud buffer, will always return 1.
    - **width** (*uint*): 3 x Width or number of points in point cloud buffer.


IsaacReadContactSensor
----------------------
    Node that reads out contact sensor data


**Inputs**
    - **execIn** (*execution*): The input execution port.
    - **csPrim** (*target*): USD prim reference to contact sensor prim.
    - **useLatestData** (*bool*): True to use the latest data from the physics step, False to use the data measured by the sensor. Default to False.

**Outputs**
    - **execOut** (*execution*): Output execution triggers when sensor has data.
    - **sensorTime** (*float*): Sensor reading timestamp.
    - **inContact** (*bool*): Bool that registers current sensor contact.
    - **value** (*float*): Contact force value reading (N).


IsaacReadIMU
------------
    Node that reads out IMU linear acceleration, angular velocity and orientation data


**Inputs**
    - **execIn** (*execution*): The input execution port.
    - **imuPrim** (*target*): Usd prim reference to the IMU prim.
    - **useLatestData** (*bool*): True to use the latest data from the physics step, False to use the data measured by the sensor. Default to False.
    - **readGravity** (*bool*): True to read gravitational acceleration in the measurement, False to ignore gravitational acceleration. Default to True.

**Outputs**
    - **execOut** (*execution*): Output execution triggers when sensor has data.
    - **sensorTime** (*float*): Timestamp of the sensor reading.
    - **linAcc** (*vectord[3]*): Linear acceleration IMU reading.
    - **angVel** (*vectord[3]*): Angular velocity IMU reading.
    - **orientation** (*quatd[4]*): Sensor orientation as quaternion.


IsaacReadRTXLidarData
---------------------
    ['This node reads the data straight from the an RTX Lidar sensor.']


**Inputs**
    - **exec** (*execution*): The input execution port.
    - **bufferSize** (*uint64*): number of bytes in dataPtr. Default to 0.
    - **dataPtr** (*uint64*): Pointer to LiDAR render result.
    - **cudaDeviceIndex** (*int*): Index of the device where the data lives (-1 for host data). Default to -1.
    - **cudaStream** (*uint64*): Cuda Stream dataPtr is on if cudaDeviceIndex > -1. Default to 0.
    - **keepOnlyPositiveDistance** (*bool*): Keep points only if the return distance is > 0. Default to False.
    - **renderProductPath** (*token*): Config is gotten from this.

**Outputs**
    - **exec** (*execution*): Output execution triggers when lidar sensor has data.
    - **numBeams** (*uint64*): The number of lidar beams being output.
    - **frameId** (*uint64*): The frameId of the current render.
    - **timestampNs** (*uint64*): The time in nanoseconds of the start of frame.
    - **transformStart** (*matrixd[4]*): The transform matrix from lidar to world coordinates at the start of the frame.
    - **transform** (*matrixd[4]*): The transform matrix from lidar to world coordinates at the end of the frame.
    - **depthRange** (*float[2]*): The min and max range for sensor to detect a hit [min, max].
    - **azimuths** (*float[]*): azimuth in deg [0, 360).
    - **elevations** (*float[]*): elevation in deg [-90, 90].
    - **distances** (*float[]*): distance in m.
    - **intensities** (*float[]*): intensity [0,1].
    - **velocities** (*pointf[3][]*): velocity at hit point in sensor coordinates [m/s].
    - **flags** (*uchar[]*): flags.
    - **hitPointNormals** (*pointf[3][]*): hit point Normal.
    - **deltaTimes** (*uint[]*): delta time in ns from the head (relative to tick timestamp).
    - **emitterIds** (*uint[]*): beam/laser detector id.
    - **materialIds** (*uint[]*): hit point material id.
    - **objectIds** (*uint[]*): hit point object id.
    - **ticks** (*uint[]*): tick of point.
    - **tickStates** (*uchar[]*): emitter state the tick belongs to.
    - **channels** (*uint[]*): channel of point.
    - **echos** (*uchar[]*): echo id in ascending order.


IsaacPrintRTXLidarInfo
----------------------
    ['process and print the raw RTX lidar data']


**Inputs**
    - **exec** (*execution*): The input execution port.
    - **dataPtr** (*uint64*): Pointer to LiDAR render result.
    - **testMode** (*bool*): Print less for benchmark tests.


IsaacPrintRTXRadarInfo
----------------------
    ['process and print the raw RTX Radar data']


**Inputs**
    - **exec** (*execution*): The input execution port.
    - **dataPtr** (*uint64*): Pointer to Radar render result.
    - **testMode** (*bool*): Print less for benchmark tests.


IsaacReadEffortSensor
---------------------
    Node that reads out joint effort values


**Inputs**
    - **execIn** (*execution*): The input execution port.
    - **prim** (*target*): Path to the joint getting measured.
    - **useLatestData** (*bool*): True to use the latest data from the physics step, False to use the data measured by the sensor. Default to False.
    - **enabled** (*bool*): True to enable sensor, False to disable the sensor. Default to True.
    - **sensorPeriod** (*float*): Downtime between sensor readings. Default to 0.

**Outputs**
    - **execOut** (*execution*): Output execution triggers when sensor has data.
    - **sensorTime** (*float*): Timestamp of the sensor reading.
    - **value** (*float*): Effort value reading.