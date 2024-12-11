import math
import os
import random

from pxr import Gf, Sdf, Usd
from pxr import UsdGeom, UsdUtils, UsdPhysics, UsdLux
from pxr import PhysxSchema, PhysicsSchemaTools, ForceFieldSchema

import omni.physx.scripts.physicsUtils as physicsUtils
import omni.kit

import omni.physxdemos as demo


class WindForceFieldDemo(demo.Base):
    title = "Wind"
    category = demo.Categories.FORCE_FIELDS
    short_description = "Create a wind to blow objects"
    description = "Set up a wind force field that has wind speed and direction variability."

    def create(self, stage):
        numberOfBoxes = 20
        boxSpacing = 2
        boxPathName = "/box"
        groundPathName = "/ground"
        scenePathName = "/scene"
        windPathName = "/wind"

        # Physics scene
        up = Gf.Vec3f(0.0)
        up[1] = 1.0
        gravityDirection = -up
        gravityMagnitude = 1000.0
        forceRange = Gf.Vec2f(-1.0)
        center = Gf.Vec3f(0.0)
        center[1] = 400.0

        scene = UsdPhysics.Scene.Define(stage, scenePathName)
        scene.CreateGravityDirectionAttr(gravityDirection)
        scene.CreateGravityMagnitudeAttr(gravityMagnitude)

        # light
        sphereLight = UsdLux.SphereLight.Define(stage, "/World/SphereLight")
        sphereLight.CreateRadiusAttr().Set(1.5)
        sphereLight.CreateIntensityAttr().Set(60000000)
        sphereLight.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(6.5, 750.0, 0))
        
        # Plane
        physicsUtils.add_ground_plane(stage, groundPathName, "Y", 750.0, Gf.Vec3f(0.0), Gf.Vec3f(0.5))

        # Create the wind force field
        windPrim = stage.DefinePrim(windPathName)

        windPrim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Float3, False).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        windPrim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatf, False).Set(Gf.Quatf(0.707, 0.0, 0.707, 0.0))
        windPrim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Float3, False).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        #windPrim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.String, False).Set(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

        windPrimApi = ForceFieldSchema.PhysxForceFieldWindAPI.Apply(windPrim, "Wind")
        windPrimApi.CreateDragAttr(1000.0)
        windPrimApi.CreateAverageSpeedAttr(300.0)
        windPrimApi.CreateSpeedVariationAttr(300.0)
        windPrimApi.CreateSpeedVariationFrequencyAttr(0.05)
        windPrimApi.CreateAverageDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        windPrimApi.CreateDirectionVariationAttr(Gf.Vec3f(0.707, 0.707, 0.0))
        windPrimApi.CreateDirectionVariationFrequencyAttr(Gf.Vec3f(0.1, 0.1, 0.0))

        windBaseApi = ForceFieldSchema.PhysxForceFieldAPI(windPrim, "Wind")
        windBaseApi.CreateEnabledAttr(True)
        windBaseApi.CreatePositionAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        windBaseApi.CreateRangeAttr(Gf.Vec2f(-1.0, -1.0))

        # Add the collection 
        collectionAPI = Usd.CollectionAPI.Apply(windPrim, ForceFieldSchema.Tokens.forceFieldBodies)
        collectionAPI.CreateIncludesRel().AddTarget(stage.GetDefaultPrim().GetPath())

        # Boxes
        boxSize = Gf.Vec3f(100.0)
        boxPosition = Gf.Vec3f(0.0)
        m = (int)(math.sqrt(numberOfBoxes))

        for i in range(m):
            for j in range(m):
                boxPath = boxPathName + str(i) + str(j)
                boxPosition[0] = (i + 0.5 - (0.5 * m)) * boxSpacing * boxSize[0]
                boxPosition[1] = 0.5 * boxSize[1]
                boxPosition[2] = (j + 0.5 - (0.5 * m)) * boxSpacing * boxSize[2]
                boxPrim = physicsUtils.add_rigid_box(stage, boxPath, position=boxPosition, size=boxSize)
