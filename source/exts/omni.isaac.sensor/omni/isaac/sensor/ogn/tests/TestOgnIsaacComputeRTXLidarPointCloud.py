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
        test_file_name = "OgnIsaacComputeRTXLidarPointCloudTemplate.usda"
        usd_path = os.path.join(os.path.dirname(__file__), "usd", test_file_name)
        if not os.path.exists(usd_path):  # pragma: no cover
            self.assertTrue(False, f"{usd_path} not found for loading test")
        (result, error) = await ogts.load_test_file(usd_path)
        self.assertTrue(result, f'{error} on {usd_path}')
        test_node = og.Controller.node("/TestGraph/Template_omni_isaac_sensor_IsaacComputeRTXLidarPointCloud")
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

        self.assertTrue(test_node.get_attribute_exists("outputs:intensity"))
        attribute = test_node.get_attribute("outputs:intensity")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:range"))
        attribute = test_node.get_attribute("outputs:range")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:transform"))
        attribute = test_node.get_attribute("outputs:transform")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:width"))
        attribute = test_node.get_attribute("outputs:width")
        self.assertTrue(attribute.is_valid())
