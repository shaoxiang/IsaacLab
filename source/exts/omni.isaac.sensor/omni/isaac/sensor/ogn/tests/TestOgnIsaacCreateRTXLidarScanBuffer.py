import os
import omni.kit.test
import omni.graph.core as og
import omni.graph.core.tests as ogts
from omni.graph.core.tests.omnigraph_test_utils import _TestGraphAndNode
from omni.graph.core.tests.omnigraph_test_utils import _test_clear_scene
from omni.graph.core.tests.omnigraph_test_utils import _test_setup_scene
from omni.graph.core.tests.omnigraph_test_utils import _test_verify_scene


class TestOgn(ogts.OmniGraphTestCase):

    async def test_data_access(self):
        test_file_name = "OgnIsaacCreateRTXLidarScanBufferTemplate.usda"
        usd_path = os.path.join(os.path.dirname(__file__), "usd", test_file_name)
        if not os.path.exists(usd_path):  # pragma: no cover
            self.assertTrue(False, f"{usd_path} not found for loading test")
        (result, error) = await ogts.load_test_file(usd_path)
        self.assertTrue(result, f'{error} on {usd_path}')
        test_node = og.Controller.node("/TestGraph/Template_omni_isaac_sensor_IsaacCreateRTXLidarScanBuffer")
        self.assertTrue(test_node.is_valid())
        node_type_name = test_node.get_type_name()
        self.assertEqual(og.GraphRegistry().get_node_type_version(node_type_name), 1)

        def _attr_error(attribute: og.Attribute, usd_test: bool) -> str:  # pragma no cover
            test_type = "USD Load" if usd_test else "Database Access"
            return f"{node_type_name} {test_type} Test - {attribute.get_name()} value error"


        self.assertTrue(test_node.get_attribute_exists("inputs:accuracyErrorAzimuthDeg"))
        attribute = test_node.get_attribute("inputs:accuracyErrorAzimuthDeg")
        self.assertTrue(attribute.is_valid())
        expected_value = 0.0
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:accuracyErrorElevationDeg"))
        attribute = test_node.get_attribute("inputs:accuracyErrorElevationDeg")
        self.assertTrue(attribute.is_valid())
        expected_value = 0.0
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:accuracyErrorPosition"))
        attribute = test_node.get_attribute("inputs:accuracyErrorPosition")
        self.assertTrue(attribute.is_valid())
        expected_value = [0.0, 0.0, 0.0]
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:cudaDeviceIndex"))
        attribute = test_node.get_attribute("inputs:cudaDeviceIndex")
        self.assertTrue(attribute.is_valid())
        expected_value = -1
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:dataPtr"))
        attribute = test_node.get_attribute("inputs:dataPtr")
        self.assertTrue(attribute.is_valid())
        expected_value = 0
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:exec"))
        attribute = test_node.get_attribute("inputs:exec")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("inputs:keepOnlyPositiveDistance"))
        attribute = test_node.get_attribute("inputs:keepOnlyPositiveDistance")
        self.assertTrue(attribute.is_valid())
        expected_value = True
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputAzimuth"))
        attribute = test_node.get_attribute("inputs:outputAzimuth")
        self.assertTrue(attribute.is_valid())
        expected_value = False
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputBeamId"))
        attribute = test_node.get_attribute("inputs:outputBeamId")
        self.assertTrue(attribute.is_valid())
        expected_value = False
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputDistance"))
        attribute = test_node.get_attribute("inputs:outputDistance")
        self.assertTrue(attribute.is_valid())
        expected_value = True
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputElevation"))
        attribute = test_node.get_attribute("inputs:outputElevation")
        self.assertTrue(attribute.is_valid())
        expected_value = False
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputEmitterId"))
        attribute = test_node.get_attribute("inputs:outputEmitterId")
        self.assertTrue(attribute.is_valid())
        expected_value = False
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputIntensity"))
        attribute = test_node.get_attribute("inputs:outputIntensity")
        self.assertTrue(attribute.is_valid())
        expected_value = True
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputMaterialId"))
        attribute = test_node.get_attribute("inputs:outputMaterialId")
        self.assertTrue(attribute.is_valid())
        expected_value = False
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputNormal"))
        attribute = test_node.get_attribute("inputs:outputNormal")
        self.assertTrue(attribute.is_valid())
        expected_value = False
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputObjectId"))
        attribute = test_node.get_attribute("inputs:outputObjectId")
        self.assertTrue(attribute.is_valid())
        expected_value = False
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputTimestamp"))
        attribute = test_node.get_attribute("inputs:outputTimestamp")
        self.assertTrue(attribute.is_valid())
        expected_value = False
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:outputVelocity"))
        attribute = test_node.get_attribute("inputs:outputVelocity")
        self.assertTrue(attribute.is_valid())
        expected_value = False
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:renderProductPath"))
        attribute = test_node.get_attribute("inputs:renderProductPath")
        self.assertTrue(attribute.is_valid())
        expected_value = ""
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:transformPoints"))
        attribute = test_node.get_attribute("inputs:transformPoints")
        self.assertTrue(attribute.is_valid())
        expected_value = False
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("outputs:azimuthBufferSize"))
        attribute = test_node.get_attribute("outputs:azimuthBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:azimuthDataType"))
        attribute = test_node.get_attribute("outputs:azimuthDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:azimuthPtr"))
        attribute = test_node.get_attribute("outputs:azimuthPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:beamIdBufferSize"))
        attribute = test_node.get_attribute("outputs:beamIdBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:beamIdDataType"))
        attribute = test_node.get_attribute("outputs:beamIdDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:beamIdPtr"))
        attribute = test_node.get_attribute("outputs:beamIdPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:bufferSize"))
        attribute = test_node.get_attribute("outputs:bufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:cudaDeviceIndex"))
        attribute = test_node.get_attribute("outputs:cudaDeviceIndex")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:dataPtr"))
        attribute = test_node.get_attribute("outputs:dataPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:distanceBufferSize"))
        attribute = test_node.get_attribute("outputs:distanceBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:distanceDataType"))
        attribute = test_node.get_attribute("outputs:distanceDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:distancePtr"))
        attribute = test_node.get_attribute("outputs:distancePtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:elevationBufferSize"))
        attribute = test_node.get_attribute("outputs:elevationBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:elevationDataType"))
        attribute = test_node.get_attribute("outputs:elevationDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:elevationPtr"))
        attribute = test_node.get_attribute("outputs:elevationPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:emitterIdBufferSize"))
        attribute = test_node.get_attribute("outputs:emitterIdBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:emitterIdDataType"))
        attribute = test_node.get_attribute("outputs:emitterIdDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:emitterIdPtr"))
        attribute = test_node.get_attribute("outputs:emitterIdPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:exec"))
        attribute = test_node.get_attribute("outputs:exec")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:height"))
        attribute = test_node.get_attribute("outputs:height")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:indexBufferSize"))
        attribute = test_node.get_attribute("outputs:indexBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:indexDataType"))
        attribute = test_node.get_attribute("outputs:indexDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:indexPtr"))
        attribute = test_node.get_attribute("outputs:indexPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:intensityBufferSize"))
        attribute = test_node.get_attribute("outputs:intensityBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:intensityDataType"))
        attribute = test_node.get_attribute("outputs:intensityDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:intensityPtr"))
        attribute = test_node.get_attribute("outputs:intensityPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:materialIdBufferSize"))
        attribute = test_node.get_attribute("outputs:materialIdBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:materialIdDataType"))
        attribute = test_node.get_attribute("outputs:materialIdDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:materialIdPtr"))
        attribute = test_node.get_attribute("outputs:materialIdPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:normalBufferSize"))
        attribute = test_node.get_attribute("outputs:normalBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:normalDataType"))
        attribute = test_node.get_attribute("outputs:normalDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:normalPtr"))
        attribute = test_node.get_attribute("outputs:normalPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:numChannels"))
        attribute = test_node.get_attribute("outputs:numChannels")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:numEchos"))
        attribute = test_node.get_attribute("outputs:numEchos")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:numReturnsPerScan"))
        attribute = test_node.get_attribute("outputs:numReturnsPerScan")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:objectIdBufferSize"))
        attribute = test_node.get_attribute("outputs:objectIdBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:objectIdDataType"))
        attribute = test_node.get_attribute("outputs:objectIdDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:objectIdPtr"))
        attribute = test_node.get_attribute("outputs:objectIdPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:renderProductPath"))
        attribute = test_node.get_attribute("outputs:renderProductPath")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:ticksPerScan"))
        attribute = test_node.get_attribute("outputs:ticksPerScan")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:timestampBufferSize"))
        attribute = test_node.get_attribute("outputs:timestampBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:timestampDataType"))
        attribute = test_node.get_attribute("outputs:timestampDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:timestampPtr"))
        attribute = test_node.get_attribute("outputs:timestampPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:transform"))
        attribute = test_node.get_attribute("outputs:transform")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:velocityBufferSize"))
        attribute = test_node.get_attribute("outputs:velocityBufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:velocityDataType"))
        attribute = test_node.get_attribute("outputs:velocityDataType")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:velocityPtr"))
        attribute = test_node.get_attribute("outputs:velocityPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:width"))
        attribute = test_node.get_attribute("outputs:width")
        self.assertTrue(attribute.is_valid())
