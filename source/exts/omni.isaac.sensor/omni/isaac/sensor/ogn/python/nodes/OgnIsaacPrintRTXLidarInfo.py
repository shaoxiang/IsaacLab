# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import ctypes

import carb
from omni.isaac.sensor import BasicElements, GenericModelOutput, LidarAuxiliaryData
from omni.syntheticdata._syntheticdata import acquire_syntheticdata_interface


def object_id_to_prim_path(object_id):
    """Given an ObjectId get a Prim Path

    Args:
        object_id (int): object id, like from a RTX Lidar return

    Returns:
        prim path string
    """
    return acquire_syntheticdata_interface().get_uri_from_instance_segmentation_id(int(object_id))


class OgnIsaacPrintRTXLidarInfo:
    """
    Example to read raw rtx data in python.
    """

    frameCount = 0

    @staticmethod
    def compute(db) -> bool:
        """read a pointer and print data from it assuming it is Rtx"""
        if not db.inputs.dataPtr:
            carb.log_warn("invalid data input to OgnIsaacPrintRTXLidarInfo")
            return True

        gmo = ctypes.cast(db.inputs.dataPtr, ctypes.POINTER(GenericModelOutput)).contents
        if gmo.magicNumber != int("4E474D4F", 16):
            # print a unique id for the node to see how many are running, and the number of returns for each
            print(f"Print Node ID_{id(db.inputs)} has invalid input")
            return True

        numElements = gmo.numElements
        if db.inputs.testMode:
            # print a unique id for the node to see how many are running, and the number of returns for each
            print(f"Print Node ID_{id(db.inputs)} has {numElements} returns")
            return True

        ptr = db.inputs.dataPtr + ctypes.sizeof(GenericModelOutput)
        elements = BasicElements()
        ptr = elements.fill(ptr, numElements)
        auxData = LidarAuxiliaryData()
        ptr = auxData.fill(ptr, numElements)

        print("-------------------- NEW FRAME ------------------------------------------")
        print("-------------------- gmo:")
        print(f"frameId:     {gmo.frameId}")
        print(f"timestampNs: {gmo.timestampNs}")
        print(f"numElements: {gmo.numElements}")
        print(f"auxType: {gmo.auxType}")
        print(f"auxdataFilled: {auxData.filledAuxMembers}")
        print(f"Return 0:")
        print(f"    timeOffsetNs: {elements.timeOffsetNs[0]}")
        print(f"    azimuth:      {elements.x[0]}")
        print(f"    elevation:    {elements.y[0]}")
        print(f"    range:        {elements.z[0]}")
        print(f"    intensity:    {elements.scalar[0]}")
        print(f"Return {gmo.numElements - 1}:")
        print(f"    timeOffsetNs: {elements.timeOffsetNs[gmo.numElements - 1]}")
        print(f"    azimuth:      {elements.x[gmo.numElements - 1]}")
        print(f"    elevation:    {elements.y[gmo.numElements - 1]}")
        print(f"    range:        {elements.z[gmo.numElements - 1]}")
        print(f"    intensity:    {elements.scalar[gmo.numElements - 1]}")
        print(f"Prim <-> Material mapping:")
        material_mapping = {}
        for i in range(numElements):
            objId = auxData.objId[i]
            matId = auxData.matId[i]
            if elements.z[i] > 0.0 and elements.scalar[i] > 0.0:
                if objId not in material_mapping:
                    material_mapping[objId] = matId

        for obj in material_mapping:
            prim_path = object_id_to_prim_path(obj)
            print(f"objectId {obj} with prim path {prim_path} has material ID {material_mapping[obj]}.")
        return True
