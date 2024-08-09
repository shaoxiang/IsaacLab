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
        test_file_name = "OgnIsaacComputeRTXRadarPointCloudTemplate.usda"
        usd_path = os.path.join(os.path.dirname(__file__), "usd", test_file_name)
        if not os.path.exists(usd_path):  # pragma: no cover
            self.assertTrue(False, f"{usd_path} not found for loading test")
        (result, error) = await ogts.load_test_file(usd_path)
        self.assertTrue(result, f'{error} on {usd_path}')
        test_node = og.Controller.node("/TestGraph/Template_omni_isaac_sensor_IsaacComputeRTXRadarPointCloud")
        self.assertTrue(test_node.is_valid())
        node_type_name = test_node.get_type_name()
        self.assertEqual(og.GraphRegistry().get_node_type_version(node_type_name), 1)

        def _attr_error(attribute: og.Attribute, usd_test: bool) -> str:  # pragma no cover
            test_type = "USD Load" if usd_test else "Database Access"
            return f"{node_type_name} {test_type} Test - {attribute.get_name()} value error"


        self.assertTrue(test_node.get_attribute_exists("inputs:dataPtr"))
        attribute = test_node.get_attribute("inputs:dataPtr")
        self.assertTrue(attribute.is_valid())
        expected_value = 0
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:exec"))
        attribute = test_node.get_attribute("inputs:exec")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("inputs:renderProductPath"))
        attribute = test_node.get_attribute("inputs:renderProductPath")
        self.assertTrue(attribute.is_valid())
        expected_value = ""
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("outputs:azimuth"))
        attribute = test_node.get_attribute("outputs:azimuth")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:bufferSize"))
        attribute = test_node.get_attribute("outputs:bufferSize")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:cudaDeviceIndex"))
        attribute = test_node.get_attribute("outputs:cudaDeviceIndex")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:cycleCnt"))
        attribute = test_node.get_attribute("outputs:cycleCnt")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:dataPtr"))
        attribute = test_node.get_attribute("outputs:dataPtr")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:elevation"))
        attribute = test_node.get_attribute("outputs:elevation")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:exec"))
        attribute = test_node.get_attribute("outputs:exec")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:height"))
        attribute = test_node.get_attribute("outputs:height")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:materialId"))
        attribute = test_node.get_attribute("outputs:materialId")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:maxAzRad"))
        attribute = test_node.get_attribute("outputs:maxAzRad")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:maxElRad"))
        attribute = test_node.get_attribute("outputs:maxElRad")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:maxRangeM"))
        attribute = test_node.get_attribute("outputs:maxRangeM")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:maxVelMps"))
        attribute = test_node.get_attribute("outputs:maxVelMps")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:minAzRad"))
        attribute = test_node.get_attribute("outputs:minAzRad")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:minElRad"))
        attribute = test_node.get_attribute("outputs:minElRad")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:minVelMps"))
        attribute = test_node.get_attribute("outputs:minVelMps")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:numDetections"))
        attribute = test_node.get_attribute("outputs:numDetections")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:objectId"))
        attribute = test_node.get_attribute("outputs:objectId")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:radialDistance"))
        attribute = test_node.get_attribute("outputs:radialDistance")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:radialVelocity"))
        attribute = test_node.get_attribute("outputs:radialVelocity")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:rcs"))
        attribute = test_node.get_attribute("outputs:rcs")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:scanIdx"))
        attribute = test_node.get_attribute("outputs:scanIdx")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:semanticId"))
        attribute = test_node.get_attribute("outputs:semanticId")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:sensorID"))
        attribute = test_node.get_attribute("outputs:sensorID")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:timeStampNS"))
        attribute = test_node.get_attribute("outputs:timeStampNS")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:transform"))
        attribute = test_node.get_attribute("outputs:transform")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:width"))
        attribute = test_node.get_attribute("outputs:width")
        self.assertTrue(attribute.is_valid())
