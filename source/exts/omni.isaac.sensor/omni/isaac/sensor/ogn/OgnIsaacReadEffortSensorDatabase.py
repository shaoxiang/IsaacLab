"""Support for simplified access to data on nodes of type omni.isaac.sensor.IsaacReadEffortSensor

 __   ___ .  .  ___  __       ___  ___  __      __   __   __   ___
/ _` |__  |\ | |__  |__)  /\   |  |__  |  \    /  ` /  \ |  \ |__
\__| |___ | \| |___ |  \ /--\  |  |___ |__/    \__, \__/ |__/ |___

 __   __     .  .  __  ___     .  .  __   __     ___
|  \ /  \    |\ | /  \  |      |\/| /  \ |  \ | |__  \ /
|__/ \__/    | \| \__/  |      |  | \__/ |__/ | |     |

Node that reads out joint effort values
"""

import carb
import sys
import traceback
import usdrt

import omni.graph.core as og
import omni.graph.core._omni_graph_core as _og
import omni.graph.tools.ogn as ogn



class OgnIsaacReadEffortSensorDatabase(og.Database):
    """Helper class providing simplified access to data on nodes of type omni.isaac.sensor.IsaacReadEffortSensor

    Class Members:
        node: Node being evaluated

    Attribute Value Properties:
        Inputs:
            inputs.enabled
            inputs.execIn
            inputs.prim
            inputs.sensorPeriod
            inputs.useLatestData
        Outputs:
            outputs.execOut
            outputs.sensorTime
            outputs.value
    """

    # Imprint the generator and target ABI versions in the file for JIT generation
    GENERATOR_VERSION = (1, 77, 0)
    TARGET_VERSION = (2, 170, 0)

    # This is an internal object that provides per-class storage of a per-node data dictionary
    PER_NODE_DATA = {}

    # This is an internal object that describes unchanging attributes in a generic way
    # The values in this list are in no particular order, as a per-attribute tuple
    #     Name, Type, ExtendedTypeIndex, UiName, Description, Metadata,
    #     Is_Required, DefaultValue, Is_Deprecated, DeprecationMsg
    # You should not need to access any of this data directly, use the defined database interfaces
    INTERFACE = og.Database._get_interface([
        ('inputs:enabled', 'bool', 0, None, 'True to enable sensor, False to disable the sensor', {ogn.MetadataKeys.DEFAULT: 'true'}, True, True, False, ''),
        ('inputs:execIn', 'execution', 0, None, 'The input execution port', {}, True, None, False, ''),
        ('inputs:prim', 'target', 0, 'Prim Path', 'Path to the joint getting measured', {}, True, None, False, ''),
        ('inputs:sensorPeriod', 'float', 0, None, 'Downtime between sensor readings', {ogn.MetadataKeys.DEFAULT: '0'}, True, 0, False, ''),
        ('inputs:useLatestData', 'bool', 0, None, 'True to use the latest data from the physics step, False to use the data measured by the sensor', {ogn.MetadataKeys.DEFAULT: 'false'}, True, False, False, ''),
        ('outputs:execOut', 'execution', 0, None, 'Output execution triggers when sensor has data', {}, True, None, False, ''),
        ('outputs:sensorTime', 'float', 0, None, 'Timestamp of the sensor reading', {ogn.MetadataKeys.DEFAULT: '0'}, True, 0, False, ''),
        ('outputs:value', 'float', 0, 'Effort Value', 'Effort value reading', {ogn.MetadataKeys.DEFAULT: '0.0'}, True, 0.0, False, ''),
    ])

    @classmethod
    def _populate_role_data(cls):
        """Populate a role structure with the non-default roles on this node type"""
        role_data = super()._populate_role_data()
        role_data.inputs.execIn = og.AttributeRole.EXECUTION
        role_data.inputs.prim = og.AttributeRole.TARGET
        role_data.outputs.execOut = og.AttributeRole.EXECUTION
        return role_data

    class ValuesForInputs(og.DynamicAttributeAccess):
        LOCAL_PROPERTY_NAMES = {"enabled", "execIn", "sensorPeriod", "useLatestData", "_setting_locked", "_batchedReadAttributes", "_batchedReadValues"}
        """Helper class that creates natural hierarchical access to input attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)
            self._batchedReadAttributes = [self._attributes.enabled, self._attributes.execIn, self._attributes.sensorPeriod, self._attributes.useLatestData]
            self._batchedReadValues = [True, None, 0, False]

        @property
        def prim(self):
            data_view = og.AttributeValueHelper(self._attributes.prim)
            return data_view.get()

        @prim.setter
        def prim(self, value):
            if self._setting_locked:
                raise og.ReadOnlyError(self._attributes.prim)
            data_view = og.AttributeValueHelper(self._attributes.prim)
            data_view.set(value)
            self.prim_size = data_view.get_array_size()

        @property
        def enabled(self):
            return self._batchedReadValues[0]

        @enabled.setter
        def enabled(self, value):
            self._batchedReadValues[0] = value

        @property
        def execIn(self):
            return self._batchedReadValues[1]

        @execIn.setter
        def execIn(self, value):
            self._batchedReadValues[1] = value

        @property
        def sensorPeriod(self):
            return self._batchedReadValues[2]

        @sensorPeriod.setter
        def sensorPeriod(self, value):
            self._batchedReadValues[2] = value

        @property
        def useLatestData(self):
            return self._batchedReadValues[3]

        @useLatestData.setter
        def useLatestData(self, value):
            self._batchedReadValues[3] = value

        def __getattr__(self, item: str):
            if item in self.LOCAL_PROPERTY_NAMES:
                return object.__getattribute__(self, item)
            else:
                return super().__getattr__(item)

        def __setattr__(self, item: str, new_value):
            if item in self.LOCAL_PROPERTY_NAMES:
                object.__setattr__(self, item, new_value)
            else:
                super().__setattr__(item, new_value)

        def _prefetch(self):
            readAttributes = self._batchedReadAttributes
            newValues = _og._prefetch_input_attributes_data(readAttributes)
            if len(readAttributes) == len(newValues):
                self._batchedReadValues = newValues

    class ValuesForOutputs(og.DynamicAttributeAccess):
        LOCAL_PROPERTY_NAMES = {"execOut", "sensorTime", "value", "_batchedWriteValues"}
        """Helper class that creates natural hierarchical access to output attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)
            self._batchedWriteValues = { }

        @property
        def execOut(self):
            value = self._batchedWriteValues.get(self._attributes.execOut)
            if value:
                return value
            else:
                data_view = og.AttributeValueHelper(self._attributes.execOut)
                return data_view.get()

        @execOut.setter
        def execOut(self, value):
            self._batchedWriteValues[self._attributes.execOut] = value

        @property
        def sensorTime(self):
            value = self._batchedWriteValues.get(self._attributes.sensorTime)
            if value:
                return value
            else:
                data_view = og.AttributeValueHelper(self._attributes.sensorTime)
                return data_view.get()

        @sensorTime.setter
        def sensorTime(self, value):
            self._batchedWriteValues[self._attributes.sensorTime] = value

        @property
        def value(self):
            value = self._batchedWriteValues.get(self._attributes.value)
            if value:
                return value
            else:
                data_view = og.AttributeValueHelper(self._attributes.value)
                return data_view.get()

        @value.setter
        def value(self, value):
            self._batchedWriteValues[self._attributes.value] = value

        def __getattr__(self, item: str):
            if item in self.LOCAL_PROPERTY_NAMES:
                return object.__getattribute__(self, item)
            else:
                return super().__getattr__(item)

        def __setattr__(self, item: str, new_value):
            if item in self.LOCAL_PROPERTY_NAMES:
                object.__setattr__(self, item, new_value)
            else:
                super().__setattr__(item, new_value)

        def _commit(self):
            _og._commit_output_attributes_data(self._batchedWriteValues)
            self._batchedWriteValues = { }

    class ValuesForState(og.DynamicAttributeAccess):
        """Helper class that creates natural hierarchical access to state attributes"""
        def __init__(self, node: og.Node, attributes, dynamic_attributes: og.DynamicAttributeInterface):
            """Initialize simplified access for the attribute data"""
            context = node.get_graph().get_default_graph_context()
            super().__init__(context, node, attributes, dynamic_attributes)

    def __init__(self, node):
        super().__init__(node)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT)
        self.inputs = OgnIsaacReadEffortSensorDatabase.ValuesForInputs(node, self.attributes.inputs, dynamic_attributes)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_OUTPUT)
        self.outputs = OgnIsaacReadEffortSensorDatabase.ValuesForOutputs(node, self.attributes.outputs, dynamic_attributes)
        dynamic_attributes = self.dynamic_attribute_data(node, og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE)
        self.state = OgnIsaacReadEffortSensorDatabase.ValuesForState(node, self.attributes.state, dynamic_attributes)

    class abi:
        """Class defining the ABI interface for the node type"""

        @staticmethod
        def get_node_type():
            get_node_type_function = getattr(OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS, 'get_node_type', None)
            if callable(get_node_type_function):  # pragma: no cover
                return get_node_type_function()
            return 'omni.isaac.sensor.IsaacReadEffortSensor'

        @staticmethod
        def compute(context, node):
            def database_valid():
                return True
            try:
                per_node_data = OgnIsaacReadEffortSensorDatabase.PER_NODE_DATA[node.node_id()]
                db = per_node_data.get('_db')
                if db is None:
                    db = OgnIsaacReadEffortSensorDatabase(node)
                    per_node_data['_db'] = db
                if not database_valid():
                    per_node_data['_db'] = None
                    return False
            except:
                db = OgnIsaacReadEffortSensorDatabase(node)

            try:
                compute_function = getattr(OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS, 'compute', None)
                if callable(compute_function) and compute_function.__code__.co_argcount > 1:  # pragma: no cover
                    return compute_function(context, node)

                db.inputs._prefetch()
                db.inputs._setting_locked = True
                with og.in_compute():
                    return OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS.compute(db)
            except Exception as error:  # pragma: no cover
                stack_trace = "".join(traceback.format_tb(sys.exc_info()[2].tb_next))
                db.log_error(f'Assertion raised in compute - {error}\n{stack_trace}', add_context=False)
            finally:
                db.inputs._setting_locked = False
                db.outputs._commit()
            return False

        @staticmethod
        def initialize(context, node):
            OgnIsaacReadEffortSensorDatabase._initialize_per_node_data(node)
            initialize_function = getattr(OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS, 'initialize', None)
            if callable(initialize_function):  # pragma: no cover
                initialize_function(context, node)

            per_node_data = OgnIsaacReadEffortSensorDatabase.PER_NODE_DATA[node.node_id()]

            def on_connection_or_disconnection(*args):
                per_node_data['_db'] = None

            node.register_on_connected_callback(on_connection_or_disconnection)
            node.register_on_disconnected_callback(on_connection_or_disconnection)

        @staticmethod
        def release(node):
            release_function = getattr(OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS, 'release', None)
            if callable(release_function):  # pragma: no cover
                release_function(node)
            OgnIsaacReadEffortSensorDatabase._release_per_node_data(node)

        @staticmethod
        def init_instance(node, graph_instance_id):
            init_instance_function = getattr(OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS, 'init_instance', None)
            if callable(init_instance_function):  # pragma: no cover
                init_instance_function(node, graph_instance_id)

        @staticmethod
        def release_instance(node, graph_instance_id):
            release_instance_function = getattr(OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS, 'release_instance', None)
            if callable(release_instance_function):  # pragma: no cover
                release_instance_function(node, graph_instance_id)
            OgnIsaacReadEffortSensorDatabase._release_per_node_instance_data(node, graph_instance_id)

        @staticmethod
        def update_node_version(context, node, old_version, new_version):
            update_node_version_function = getattr(OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS, 'update_node_version', None)
            if callable(update_node_version_function):  # pragma: no cover
                return update_node_version_function(context, node, old_version, new_version)
            return False

        @staticmethod
        def initialize_type(node_type):
            initialize_type_function = getattr(OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS, 'initialize_type', None)
            needs_initializing = True
            if callable(initialize_type_function):  # pragma: no cover
                needs_initializing = initialize_type_function(node_type)
            if needs_initializing:
                node_type.set_metadata(ogn.MetadataKeys.EXTENSION, "omni.isaac.sensor")
                node_type.set_metadata(ogn.MetadataKeys.UI_NAME, "Isaac Read Effort Node")
                node_type.set_metadata(ogn.MetadataKeys.CATEGORIES, "isaacSensor")
                node_type.set_metadata(ogn.MetadataKeys.DESCRIPTION, "Node that reads out joint effort values")
                node_type.set_metadata(ogn.MetadataKeys.LANGUAGE, "Python")
                icon_path = carb.tokens.get_tokens_interface().resolve("${omni.isaac.sensor}")
                icon_path = icon_path + '/' + "ogn/icons/omni.isaac.sensor.IsaacReadEffortSensor.svg"
                node_type.set_metadata(ogn.MetadataKeys.ICON_PATH, icon_path)
                OgnIsaacReadEffortSensorDatabase.INTERFACE.add_to_node_type(node_type)

        @staticmethod
        def on_connection_type_resolve(node):
            on_connection_type_resolve_function = getattr(OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS, 'on_connection_type_resolve', None)
            if callable(on_connection_type_resolve_function):  # pragma: no cover
                on_connection_type_resolve_function(node)

    NODE_TYPE_CLASS = None

    @staticmethod
    def register(node_type_class):
        OgnIsaacReadEffortSensorDatabase.NODE_TYPE_CLASS = node_type_class
        og.register_node_type(OgnIsaacReadEffortSensorDatabase.abi, 1)

    @staticmethod
    def deregister():
        og.deregister_node_type("omni.isaac.sensor.IsaacReadEffortSensor")
