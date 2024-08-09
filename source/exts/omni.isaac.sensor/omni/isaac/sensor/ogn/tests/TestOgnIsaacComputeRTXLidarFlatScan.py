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
        test_file_name = "OgnIsaacComputeRTXLidarFlatScanTemplate.usda"
        usd_path = os.path.join(os.path.dirname(__file__), "usd", test_file_name)
        if not os.path.exists(usd_path):  # pragma: no cover
            self.assertTrue(False, f"{usd_path} not found for loading test")
        (result, error) = await ogts.load_test_file(usd_path)
        self.assertTrue(result, f'{error} on {usd_path}')
        test_node = og.Controller.node("/TestGraph/Template_omni_isaac_sensor_IsaacComputeRTXLidarFlatScan")
        self.assertTrue(test_node.is_valid())
        node_type_name = test_node.get_type_name()
        self.assertEqual(og.GraphRegistry().get_node_type_version(node_type_name), 2)

        def _attr_error(attribute: og.Attribute, usd_test: bool) -> str:  # pragma no cover
            test_type = "USD Load" if usd_test else "Database Access"
            return f"{node_type_name} {test_type} Test - {attribute.get_name()} value error"


        self.assertTrue(test_node.get_attribute_exists("inputs:bufferSize"))
        attribute = test_node.get_attribute("inputs:bufferSize")
        self.assertTrue(attribute.is_valid())
        expected_value = 0
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

        self.assertTrue(test_node.get_attribute_exists("inputs:renderProductPath"))
        attribute = test_node.get_attribute("inputs:renderProductPath")
        self.assertTrue(attribute.is_valid())
        expected_value = ""
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("outputs:azimuthRange"))
        attribute = test_node.get_attribute("outputs:azimuthRange")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:depthRange"))
        attribute = test_node.get_attribute("outputs:depthRange")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:exec"))
        attribute = test_node.get_attribute("outputs:exec")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:horizontalFov"))
        attribute = test_node.get_attribute("outputs:horizontalFov")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:horizontalResolution"))
        attribute = test_node.get_attribute("outputs:horizontalResolution")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:intensitiesData"))
        attribute = test_node.get_attribute("outputs:intensitiesData")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:linearDepthData"))
        attribute = test_node.get_attribute("outputs:linearDepthData")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:numCols"))
        attribute = test_node.get_attribute("outputs:numCols")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:numRows"))
        attribute = test_node.get_attribute("outputs:numRows")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:rotationRate"))
        attribute = test_node.get_attribute("outputs:rotationRate")
        self.assertTrue(attribute.is_valid())
