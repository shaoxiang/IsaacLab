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
        test_file_name = "OgnIsaacReadRTXLidarDataTemplate.usda"
        usd_path = os.path.join(os.path.dirname(__file__), "usd", test_file_name)
        if not os.path.exists(usd_path):  # pragma: no cover
            self.assertTrue(False, f"{usd_path} not found for loading test")
        (result, error) = await ogts.load_test_file(usd_path)
        self.assertTrue(result, f'{error} on {usd_path}')
        test_node = og.Controller.node("/TestGraph/Template_omni_isaac_sensor_IsaacReadRTXLidarData")
        self.assertTrue(test_node.is_valid())
        node_type_name = test_node.get_type_name()
        self.assertEqual(og.GraphRegistry().get_node_type_version(node_type_name), 1)

        def _attr_error(attribute: og.Attribute, usd_test: bool) -> str:  # pragma no cover
            test_type = "USD Load" if usd_test else "Database Access"
            return f"{node_type_name} {test_type} Test - {attribute.get_name()} value error"


        self.assertTrue(test_node.get_attribute_exists("inputs:bufferSize"))
        attribute = test_node.get_attribute("inputs:bufferSize")
        self.assertTrue(attribute.is_valid())
        expected_value = 0
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:cudaDeviceIndex"))
        attribute = test_node.get_attribute("inputs:cudaDeviceIndex")
        self.assertTrue(attribute.is_valid())
        expected_value = -1
        actual_value = og.Controller.get(attribute)
        ogts.verify_values(expected_value, actual_value, _attr_error(attribute, True))

        self.assertTrue(test_node.get_attribute_exists("inputs:cudaStream"))
        attribute = test_node.get_attribute("inputs:cudaStream")
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

        self.assertTrue(test_node.get_attribute_exists("inputs:keepOnlyPositiveDistance"))
        attribute = test_node.get_attribute("inputs:keepOnlyPositiveDistance")
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

        self.assertTrue(test_node.get_attribute_exists("outputs:azimuths"))
        attribute = test_node.get_attribute("outputs:azimuths")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:channels"))
        attribute = test_node.get_attribute("outputs:channels")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:deltaTimes"))
        attribute = test_node.get_attribute("outputs:deltaTimes")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:depthRange"))
        attribute = test_node.get_attribute("outputs:depthRange")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:distances"))
        attribute = test_node.get_attribute("outputs:distances")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:echos"))
        attribute = test_node.get_attribute("outputs:echos")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:elevations"))
        attribute = test_node.get_attribute("outputs:elevations")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:emitterIds"))
        attribute = test_node.get_attribute("outputs:emitterIds")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:exec"))
        attribute = test_node.get_attribute("outputs:exec")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:flags"))
        attribute = test_node.get_attribute("outputs:flags")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:frameId"))
        attribute = test_node.get_attribute("outputs:frameId")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:hitPointNormals"))
        attribute = test_node.get_attribute("outputs:hitPointNormals")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:intensities"))
        attribute = test_node.get_attribute("outputs:intensities")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:materialIds"))
        attribute = test_node.get_attribute("outputs:materialIds")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:numBeams"))
        attribute = test_node.get_attribute("outputs:numBeams")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:objectIds"))
        attribute = test_node.get_attribute("outputs:objectIds")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:tickStates"))
        attribute = test_node.get_attribute("outputs:tickStates")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:ticks"))
        attribute = test_node.get_attribute("outputs:ticks")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:timestampNs"))
        attribute = test_node.get_attribute("outputs:timestampNs")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:transform"))
        attribute = test_node.get_attribute("outputs:transform")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:transformStart"))
        attribute = test_node.get_attribute("outputs:transformStart")
        self.assertTrue(attribute.is_valid())

        self.assertTrue(test_node.get_attribute_exists("outputs:velocities"))
        attribute = test_node.get_attribute("outputs:velocities")
        self.assertTrue(attribute.is_valid())
