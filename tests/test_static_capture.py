"""Tests for static capture functionality.

Tests capture_static_execution_graph() and module hierarchy building.
"""

import pytest
import torch
import torch.nn as nn
from vode.capture.static_capture import (
    capture_static_execution_graph,
    _count_parameters,
    _module_to_operation_info,
    _build_execution_node_recursive,
)
from vode.core.nodes import ExecutionNode, OperationInfo


class TestParameterCounting:
    """Test parameter counting utilities."""

    def test_count_parameters_linear(self):
        """Test parameter counting for Linear layer."""
        module = nn.Linear(10, 20)
        count = _count_parameters(module)
        # 10*20 weights + 20 biases = 220
        assert count == 220

    def test_count_parameters_conv2d(self):
        """Test parameter counting for Conv2d layer."""
        module = nn.Conv2d(3, 16, kernel_size=3)
        count = _count_parameters(module)
        # 3*16*3*3 weights + 16 biases = 448
        assert count == 448

    def test_count_parameters_parameterless(self):
        """Test parameter counting for parameterless modules."""
        module = nn.ReLU()
        count = _count_parameters(module)
        assert count == 0

    def test_count_parameters_sequential(self):
        """Test parameter counting for Sequential container."""
        module = nn.Sequential(
            nn.Linear(10, 20),  # 220 params
            nn.ReLU(),  # 0 params
            nn.Linear(20, 10),  # 210 params
        )
        count = _count_parameters(module)
        assert count == 430


class TestModuleToOperationInfo:
    """Test module to OperationInfo conversion."""

    def test_linear_module(self):
        """Test conversion of Linear module."""
        module = nn.Linear(10, 20)
        op_info = _module_to_operation_info(module, "fc1")

        assert op_info.op_type == "Linear"
        assert op_info.op_name == "fc1"
        assert op_info.params_count == 220
        assert op_info.is_composite is False

    def test_sequential_module(self):
        """Test conversion of Sequential module."""
        module = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
        )
        op_info = _module_to_operation_info(module, "block")

        assert op_info.op_type == "Sequential"
        assert op_info.op_name == "block"
        assert op_info.params_count == 220
        assert op_info.is_composite is True

    def test_parameterless_module(self):
        """Test conversion of parameterless module."""
        module = nn.ReLU()
        op_info = _module_to_operation_info(module, "relu")

        assert op_info.op_type == "ReLU"
        assert op_info.op_name == "relu"
        assert op_info.params_count == 0
        assert op_info.is_composite is False

    def test_module_without_name(self):
        """Test conversion without explicit name."""
        module = nn.Linear(10, 20)
        op_info = _module_to_operation_info(module)

        assert op_info.op_type == "Linear"
        assert op_info.op_name == "Linear"


class TestBuildExecutionNodeRecursive:
    """Test recursive ExecutionNode building."""

    def test_simple_linear(self):
        """Test building node for simple Linear layer."""
        module = nn.Linear(10, 20)
        node = _build_execution_node_recursive(
            module, "fc", depth=0, node_id_prefix="root"
        )

        assert node.node_id == "root_fc"
        assert node.name == "fc"
        assert node.depth == 0
        assert node.operation.op_type == "Linear"
        assert node.operation.params_count == 220
        assert node.is_expandable is False
        assert len(node.children) == 0

    def test_sequential_module(self):
        """Test building node for Sequential module."""
        module = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )
        node = _build_execution_node_recursive(
            module, "seq", depth=0, node_id_prefix="root"
        )

        assert node.node_id == "root_seq"
        assert node.name == "seq"
        assert node.depth == 0
        assert node.operation.op_type == "Sequential"
        assert node.is_expandable is True
        assert len(node.children) == 3

        # Check children
        assert node.children[0].operation.op_type == "Linear"
        assert node.children[0].depth == 1
        assert node.children[1].operation.op_type == "ReLU"
        assert node.children[1].depth == 1
        assert node.children[2].operation.op_type == "Linear"
        assert node.children[2].depth == 1

    def test_nested_sequential(self):
        """Test building node for nested Sequential modules."""
        module = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
            ),
            nn.Linear(20, 10),
        )
        node = _build_execution_node_recursive(
            module, "model", depth=0, node_id_prefix="root"
        )

        assert node.depth == 0
        assert len(node.children) == 2

        # First child is Sequential
        child0 = node.children[0]
        assert child0.operation.op_type == "Sequential"
        assert child0.depth == 1
        assert child0.is_expandable is True
        assert len(child0.children) == 2

        # Grandchildren
        assert child0.children[0].operation.op_type == "Linear"
        assert child0.children[0].depth == 2
        assert child0.children[1].operation.op_type == "ReLU"
        assert child0.children[1].depth == 2

    def test_module_list(self):
        """Test building node for ModuleList."""
        module = nn.ModuleList(
            [
                nn.Linear(10, 20),
                nn.Linear(20, 30),
                nn.Linear(30, 10),
            ]
        )
        node = _build_execution_node_recursive(
            module, "layers", depth=0, node_id_prefix="root"
        )

        assert node.operation.op_type == "ModuleList"
        assert node.is_expandable is True
        assert len(node.children) == 3

        for i, child in enumerate(node.children):
            assert child.operation.op_type == "Linear"
            assert child.depth == 1


class TestCaptureStaticExecutionGraph:
    """Test main static capture API."""

    def test_simple_sequential(self):
        """Test capturing simple Sequential model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        root = capture_static_execution_graph(model)

        assert root.node_id == "root"
        assert root.depth == 0
        assert root.operation.op_type == "Sequential"
        assert root.is_expandable is True
        assert len(root.children) == 3

    def test_custom_module(self):
        """Test capturing custom module."""

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(20, 10)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        model = CustomModel()
        root = capture_static_execution_graph(model)

        assert root.operation.op_type == "CustomModel"
        assert root.is_expandable is True
        assert len(root.children) == 3

        # Check children names
        child_types = [child.operation.op_type for child in root.children]
        assert "Linear" in child_types
        assert "ReLU" in child_types

    def test_nested_structure(self):
        """Test capturing nested module structure."""

        class Block(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.fc = nn.Linear(in_features, out_features)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.fc(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = Block(10, 20)
                self.block2 = Block(20, 10)

            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                return x

        model = Model()
        root = capture_static_execution_graph(model)

        assert root.operation.op_type == "Model"
        assert len(root.children) == 2

        # Check blocks
        for block in root.children:
            assert block.operation.op_type == "Block"
            assert block.is_expandable is True
            assert len(block.children) == 2

    def test_parameter_counting(self):
        """Test that parameter counts are correct."""
        model = nn.Sequential(
            nn.Linear(10, 20),  # 220 params
            nn.Linear(20, 10),  # 210 params
        )

        root = capture_static_execution_graph(model)

        # Root should have total params
        assert root.operation.params_count == 430

        # Children should have individual params
        assert root.children[0].operation.params_count == 220
        assert root.children[1].operation.params_count == 210

    def test_empty_sequential(self):
        """Test capturing empty Sequential."""
        model = nn.Sequential()
        root = capture_static_execution_graph(model)

        assert root.operation.op_type == "Sequential"
        assert len(root.children) == 0
        assert root.is_expandable is False

    def test_single_layer(self):
        """Test capturing single layer."""
        model = nn.Linear(10, 20)
        root = capture_static_execution_graph(model)

        assert root.operation.op_type == "Linear"
        assert root.operation.params_count == 220
        assert root.is_expandable is False
        assert len(root.children) == 0

    def test_invalid_input(self):
        """Test that non-Module input raises TypeError."""
        with pytest.raises(TypeError):
            capture_static_execution_graph("not a module")

        with pytest.raises(TypeError):
            capture_static_execution_graph(None)

    def test_parent_child_relationships(self):
        """Test that parent-child relationships are correctly set."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
        )

        root = capture_static_execution_graph(model)

        # Check parent references
        for child in root.children:
            assert child.parent == root

    def test_depth_tracking(self):
        """Test that depth is correctly tracked."""

        class DeepModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.level1 = nn.Sequential(
                    nn.Sequential(
                        nn.Linear(10, 20),
                    )
                )

            def forward(self, x):
                return self.level1(x)

        model = DeepModel()
        root = capture_static_execution_graph(model)

        assert root.depth == 0
        assert root.children[0].depth == 1
        assert root.children[0].children[0].depth == 2
        assert root.children[0].children[0].children[0].depth == 3
