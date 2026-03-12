"""Tests for core node data structures.

Tests ExecutionNode, TensorInfo, and OperationInfo classes.
"""

import pytest
from vode.core.nodes import ExecutionNode, TensorInfo, OperationInfo


class TestTensorInfo:
    """Test TensorInfo data structure."""

    def test_creation(self):
        """Test basic TensorInfo creation."""
        tensor_info = TensorInfo(
            name="input_0",
            shape=(1, 3, 224, 224),
            dtype="torch.float32",
            device="cpu",
        )

        assert tensor_info.name == "input_0"
        assert tensor_info.shape == (1, 3, 224, 224)
        assert tensor_info.dtype == "torch.float32"
        assert tensor_info.device == "cpu"

    def test_optional_fields(self):
        """Test TensorInfo with optional fields."""
        tensor_info = TensorInfo(name="tensor")

        assert tensor_info.name == "tensor"
        assert tensor_info.shape is None
        assert tensor_info.dtype is None
        assert tensor_info.device is None

    def test_to_dict(self):
        """Test TensorInfo serialization to dict."""
        tensor_info = TensorInfo(
            name="output",
            shape=(1, 10),
            dtype="torch.float32",
            device="cuda:0",
        )

        result = tensor_info.to_dict()

        assert result["name"] == "output"
        assert result["shape"] == (1, 10)
        assert result["dtype"] == "torch.float32"
        assert result["device"] == "cuda:0"


class TestOperationInfo:
    """Test OperationInfo data structure."""

    def test_creation(self):
        """Test basic OperationInfo creation."""
        op_info = OperationInfo(
            op_type="Linear",
            op_name="fc1",
            params_count=7850,
            is_composite=False,
        )

        assert op_info.op_type == "Linear"
        assert op_info.op_name == "fc1"
        assert op_info.params_count == 7850
        assert op_info.is_composite is False

    def test_default_values(self):
        """Test OperationInfo with default values."""
        op_info = OperationInfo(op_type="ReLU", op_name="relu")

        assert op_info.op_type == "ReLU"
        assert op_info.op_name == "relu"
        assert op_info.params_count == 0
        assert op_info.is_composite is False

    def test_composite_operation(self):
        """Test composite operation flag."""
        op_info = OperationInfo(
            op_type="Sequential",
            op_name="block1",
            params_count=15000,
            is_composite=True,
        )

        assert op_info.is_composite is True

    def test_to_dict(self):
        """Test OperationInfo serialization to dict."""
        op_info = OperationInfo(
            op_type="Conv2d",
            op_name="conv1",
            params_count=9408,
            is_composite=False,
        )

        result = op_info.to_dict()

        assert result["op_type"] == "Conv2d"
        assert result["op_name"] == "conv1"
        assert result["params_count"] == 9408
        assert result["is_composite"] is False


class TestExecutionNode:
    """Test ExecutionNode data structure."""

    def test_creation(self):
        """Test basic ExecutionNode creation."""
        inputs = [TensorInfo(name="input", shape=(1, 10))]
        operation = OperationInfo(op_type="Linear", op_name="fc", params_count=110)
        outputs = [TensorInfo(name="output", shape=(1, 10))]

        node = ExecutionNode(
            node_id="node_0",
            name="fc",
            depth=0,
            inputs=inputs,
            operation=operation,
            outputs=outputs,
        )

        assert node.node_id == "node_0"
        assert node.name == "fc"
        assert node.depth == 0
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.operation.op_type == "Linear"
        assert node.is_expandable is False
        assert node.is_expanded is False
        assert node.parent is None
        assert len(node.children) == 0

    def test_add_child(self):
        """Test adding child nodes."""
        parent = ExecutionNode(
            node_id="parent",
            name="Sequential",
            depth=0,
            inputs=[],
            operation=OperationInfo(
                op_type="Sequential", op_name="seq", is_composite=True
            ),
            outputs=[],
        )

        child = ExecutionNode(
            node_id="child",
            name="Linear",
            depth=1,
            inputs=[],
            operation=OperationInfo(op_type="Linear", op_name="fc"),
            outputs=[],
        )

        parent.add_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent
        assert parent.is_expandable is True

    def test_add_duplicate_child(self):
        """Test that duplicate children are not added."""
        parent = ExecutionNode(
            node_id="parent",
            name="Sequential",
            depth=0,
            inputs=[],
            operation=OperationInfo(op_type="Sequential", op_name="seq"),
            outputs=[],
        )

        child = ExecutionNode(
            node_id="child",
            name="Linear",
            depth=1,
            inputs=[],
            operation=OperationInfo(op_type="Linear", op_name="fc"),
            outputs=[],
        )

        parent.add_child(child)
        parent.add_child(child)

        assert len(parent.children) == 1

    def test_can_expand(self):
        """Test can_expand method."""
        # Node without children
        node1 = ExecutionNode(
            node_id="node1",
            name="Linear",
            depth=0,
            inputs=[],
            operation=OperationInfo(op_type="Linear", op_name="fc"),
            outputs=[],
        )
        assert node1.can_expand() is False

        # Node with children but not expandable
        node2 = ExecutionNode(
            node_id="node2",
            name="Test",
            depth=0,
            inputs=[],
            operation=OperationInfo(op_type="Test", op_name="test"),
            outputs=[],
            is_expandable=False,
        )
        child = ExecutionNode(
            node_id="child",
            name="Child",
            depth=1,
            inputs=[],
            operation=OperationInfo(op_type="Child", op_name="child"),
            outputs=[],
        )
        node2.children.append(child)
        assert node2.can_expand() is False

        # Node with children and expandable
        node3 = ExecutionNode(
            node_id="node3",
            name="Sequential",
            depth=0,
            inputs=[],
            operation=OperationInfo(
                op_type="Sequential", op_name="seq", is_composite=True
            ),
            outputs=[],
            is_expandable=True,
        )
        node3.add_child(child)
        assert node3.can_expand() is True

    def test_expand_collapse(self):
        """Test expand and collapse methods."""
        parent = ExecutionNode(
            node_id="parent",
            name="Sequential",
            depth=0,
            inputs=[],
            operation=OperationInfo(
                op_type="Sequential", op_name="seq", is_composite=True
            ),
            outputs=[],
            is_expandable=True,
        )

        child = ExecutionNode(
            node_id="child",
            name="Linear",
            depth=1,
            inputs=[],
            operation=OperationInfo(op_type="Linear", op_name="fc"),
            outputs=[],
        )

        parent.add_child(child)

        # Initially not expanded
        assert parent.is_expanded is False

        # Expand
        parent.expand()
        assert parent.is_expanded is True

        # Collapse
        parent.collapse()
        assert parent.is_expanded is False

    def test_get_depth(self):
        """Test get_depth method."""
        node = ExecutionNode(
            node_id="node",
            name="Test",
            depth=3,
            inputs=[],
            operation=OperationInfo(op_type="Test", op_name="test"),
            outputs=[],
        )

        assert node.get_depth() == 3

    def test_to_dict(self):
        """Test ExecutionNode serialization to dict."""
        inputs = [TensorInfo(name="input", shape=(1, 10))]
        operation = OperationInfo(op_type="Linear", op_name="fc", params_count=110)
        outputs = [TensorInfo(name="output", shape=(1, 10))]

        parent = ExecutionNode(
            node_id="parent",
            name="Sequential",
            depth=0,
            inputs=inputs,
            operation=OperationInfo(
                op_type="Sequential", op_name="seq", is_composite=True
            ),
            outputs=outputs,
            is_expandable=True,
        )

        child = ExecutionNode(
            node_id="child",
            name="Linear",
            depth=1,
            inputs=inputs,
            operation=operation,
            outputs=outputs,
        )

        parent.add_child(child)

        result = parent.to_dict()

        assert result["node_id"] == "parent"
        assert result["name"] == "Sequential"
        assert result["depth"] == 0
        assert len(result["inputs"]) == 1
        assert len(result["outputs"]) == 1
        assert result["operation"]["op_type"] == "Sequential"
        assert len(result["children"]) == 1
        assert result["is_expandable"] is True
        assert result["is_expanded"] is False
        assert result["parent_id"] is None

        # Check child
        assert result["children"][0]["node_id"] == "child"
        assert result["children"][0]["parent_id"] == "parent"

    def test_hierarchical_structure(self):
        """Test multi-level hierarchical structure."""
        # Create a three-level hierarchy
        root = ExecutionNode(
            node_id="root",
            name="Model",
            depth=0,
            inputs=[],
            operation=OperationInfo(
                op_type="Model", op_name="model", is_composite=True
            ),
            outputs=[],
            is_expandable=True,
        )

        level1 = ExecutionNode(
            node_id="level1",
            name="Sequential",
            depth=1,
            inputs=[],
            operation=OperationInfo(
                op_type="Sequential", op_name="seq", is_composite=True
            ),
            outputs=[],
            is_expandable=True,
        )

        level2 = ExecutionNode(
            node_id="level2",
            name="Linear",
            depth=2,
            inputs=[],
            operation=OperationInfo(op_type="Linear", op_name="fc"),
            outputs=[],
        )

        root.add_child(level1)
        level1.add_child(level2)

        assert len(root.children) == 1
        assert root.children[0] == level1
        assert level1.parent == root
        assert len(level1.children) == 1
        assert level1.children[0] == level2
        assert level2.parent == level1
        assert level2.get_depth() == 2
