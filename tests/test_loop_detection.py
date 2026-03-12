"""Tests for loop detection functionality in VODE.

This module tests the new loop detection features that identify:
1. Sequential loops (nn.Sequential containers)
2. ModuleList loops (nn.ModuleList containers)
3. Module reuse loops (same module called multiple times)
"""

import pytest
import torch
import torch.nn as nn

from vode.capture.static_capture import capture_static_execution_graph
from vode.capture.dynamic_capture import capture_dynamic_execution_graph
from vode.core.nodes import OperationInfo


class SimpleSequential(nn.Module):
    """Model with Sequential container."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        return self.seq(x)


class ModelWithModuleList(nn.Module):
    """Model with ModuleList container."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(10, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 10),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModelWithReusedModule(nn.Module):
    """Model that reuses the same module multiple times."""

    def __init__(self):
        super().__init__()
        self.shared_layer = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Use the same layer 3 times
        x = self.relu(self.shared_layer(x))
        x = self.relu(self.shared_layer(x))
        x = self.relu(self.shared_layer(x))
        return x


class ComplexModel(nn.Module):
    """Model with multiple loop types."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            [
                nn.Linear(20, 20),
                nn.Linear(20, 20),
            ]
        )
        self.shared = nn.Linear(20, 20)  # Fixed: 20->20 instead of 20->10

    def forward(self, x):
        x = self.seq(x)
        for layer in self.layers:
            x = layer(x)
        x = self.shared(x)
        x = self.shared(x)  # Reuse
        return x


# ============================================================================
# Static Capture Tests
# ============================================================================


def test_static_sequential_detection():
    """Test that Sequential containers are detected as loops in static capture."""
    model = SimpleSequential()
    root = capture_static_execution_graph(model)

    # Find the Sequential node
    seq_node = None
    for child in root.children:
        if child.operation.op_type == "Sequential":
            seq_node = child
            break

    assert seq_node is not None, "Sequential node not found"
    assert seq_node.operation.is_loop is True
    assert seq_node.operation.loop_type == "sequential"
    assert seq_node.operation.iteration_count == 3  # 3 layers in Sequential


def test_static_modulelist_detection():
    """Test that ModuleList containers are detected as loops in static capture."""
    model = ModelWithModuleList()
    root = capture_static_execution_graph(model)

    # Find the ModuleList node
    modulelist_node = None
    for child in root.children:
        if child.operation.op_type == "ModuleList":
            modulelist_node = child
            break

    assert modulelist_node is not None, "ModuleList node not found"
    assert modulelist_node.operation.is_loop is True
    assert modulelist_node.operation.loop_type == "modulelist"
    assert modulelist_node.operation.iteration_count == 3  # 3 layers in ModuleList


def test_static_no_reuse_detection():
    """Test that static capture does NOT detect module reuse (requires dynamic)."""
    model = ModelWithReusedModule()
    root = capture_static_execution_graph(model)

    # Find the shared_layer node
    shared_node = None
    for child in root.children:
        if "shared_layer" in child.name:
            shared_node = child
            break

    assert shared_node is not None, "Shared layer node not found"
    # Static capture should NOT detect reuse
    assert shared_node.operation.is_loop is False


def test_static_complex_model():
    """Test loop detection in a complex model with multiple loop types."""
    model = ComplexModel()
    root = capture_static_execution_graph(model)

    # Check Sequential
    seq_found = False
    modulelist_found = False

    for child in root.children:
        if child.operation.op_type == "Sequential":
            assert child.operation.is_loop is True
            assert child.operation.loop_type == "sequential"
            seq_found = True
        elif child.operation.op_type == "ModuleList":
            assert child.operation.is_loop is True
            assert child.operation.loop_type == "modulelist"
            modulelist_found = True

    assert seq_found, "Sequential not found"
    assert modulelist_found, "ModuleList not found"


# ============================================================================
# Dynamic Capture Tests
# ============================================================================


def test_dynamic_sequential_detection():
    """Test that Sequential containers are detected as loops in dynamic capture."""
    model = SimpleSequential()
    x = torch.randn(2, 10)
    root = capture_dynamic_execution_graph(model, x)

    # Find the Sequential node
    seq_node = None
    for child in root.children:
        if child.operation.op_type == "Sequential":
            seq_node = child
            break

    assert seq_node is not None, "Sequential node not found"
    assert seq_node.operation.is_loop is True
    assert seq_node.operation.loop_type == "sequential"
    assert seq_node.operation.iteration_count == 3


def test_dynamic_modulelist_detection():
    """Test that ModuleList containers are detected as loops in dynamic capture."""
    model = ModelWithModuleList()
    x = torch.randn(2, 10)
    root = capture_dynamic_execution_graph(model, x)

    # Find the ModuleList node
    modulelist_node = None
    for child in root.children:
        if child.operation.op_type == "ModuleList":
            modulelist_node = child
            break

    assert modulelist_node is not None, "ModuleList node not found"
    assert modulelist_node.operation.is_loop is True
    assert modulelist_node.operation.loop_type == "modulelist"
    assert modulelist_node.operation.iteration_count == 3


def test_dynamic_reuse_detection():
    """Test that dynamic capture detects module reuse as loops."""
    model = ModelWithReusedModule()
    x = torch.randn(2, 10)
    root = capture_dynamic_execution_graph(model, x)

    # Find the shared_layer node
    shared_node = None
    for child in root.children:
        if "shared_layer" in child.name:
            shared_node = child
            break

    assert shared_node is not None, "Shared layer node not found"
    # Dynamic capture SHOULD detect reuse
    assert shared_node.operation.is_loop is True
    assert shared_node.operation.loop_type == "reuse"
    assert shared_node.operation.iteration_count == 3  # Called 3 times


def test_dynamic_complex_model():
    """Test loop detection in a complex model with all loop types."""
    model = ComplexModel()
    x = torch.randn(2, 10)
    root = capture_dynamic_execution_graph(model, x)

    # Check Sequential
    seq_found = False
    modulelist_found = False
    reuse_found = False

    def check_node(node):
        nonlocal seq_found, modulelist_found, reuse_found
        if node.operation.op_type == "Sequential":
            assert node.operation.is_loop is True
            assert node.operation.loop_type == "sequential"
            seq_found = True
        elif node.operation.op_type == "ModuleList":
            assert node.operation.is_loop is True
            assert node.operation.loop_type == "modulelist"
            modulelist_found = True
        elif "shared" in node.name and node.operation.is_loop:
            assert node.operation.loop_type == "reuse"
            assert node.operation.iteration_count == 2  # Called 2 times
            reuse_found = True

        for child in node.children:
            check_node(child)

    check_node(root)

    assert seq_found, "Sequential not found"
    assert modulelist_found, "ModuleList not found"
    assert reuse_found, "Reuse not found"


# ============================================================================
# OperationInfo Tests
# ============================================================================


def test_operation_info_loop_fields():
    """Test that OperationInfo has loop detection fields."""
    op = OperationInfo(
        op_type="Sequential",
        op_name="seq",
        params_count=100,
        is_composite=True,
        is_loop=True,
        loop_type="sequential",
        iteration_count=5,
    )

    assert op.is_loop is True
    assert op.loop_type == "sequential"
    assert op.iteration_count == 5


def test_operation_info_to_dict_with_loops():
    """Test that OperationInfo.to_dict() includes loop fields."""
    op = OperationInfo(
        op_type="ModuleList",
        op_name="layers",
        is_loop=True,
        loop_type="modulelist",
        iteration_count=3,
    )

    d = op.to_dict()
    assert d["is_loop"] is True
    assert d["loop_type"] == "modulelist"
    assert d["iteration_count"] == 3


def test_operation_info_no_loop():
    """Test OperationInfo with no loop detection."""
    op = OperationInfo(
        op_type="Linear",
        op_name="fc1",
        params_count=200,
    )

    assert op.is_loop is False
    assert op.loop_type is None
    assert op.iteration_count is None


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_sequential():
    """Test empty Sequential container."""

    class EmptySeq(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential()

        def forward(self, x):
            return self.seq(x)

    model = EmptySeq()
    root = capture_static_execution_graph(model)

    seq_node = None
    for child in root.children:
        if child.operation.op_type == "Sequential":
            seq_node = child
            break

    assert seq_node is not None
    assert seq_node.operation.is_loop is True
    assert seq_node.operation.iteration_count == 0


def test_empty_modulelist():
    """Test empty ModuleList container."""

    class EmptyList(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([])

        def forward(self, x):
            return x

    model = EmptyList()
    root = capture_static_execution_graph(model)

    modulelist_node = None
    for child in root.children:
        if child.operation.op_type == "ModuleList":
            modulelist_node = child
            break

    assert modulelist_node is not None
    assert modulelist_node.operation.is_loop is True
    assert modulelist_node.operation.iteration_count == 0


def test_nested_sequential():
    """Test nested Sequential containers."""

    class NestedSeq(nn.Module):
        def __init__(self):
            super().__init__()
            self.outer = nn.Sequential(
                nn.Linear(10, 20),
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(20, 20),
                ),
                nn.Linear(20, 10),
            )

        def forward(self, x):
            return self.outer(x)

    model = NestedSeq()
    root = capture_static_execution_graph(model)

    # Both outer and inner Sequential should be detected
    seq_count = 0

    def count_sequential(node):
        nonlocal seq_count
        if node.operation.op_type == "Sequential" and node.operation.is_loop:
            seq_count += 1
        for child in node.children:
            count_sequential(child)

    count_sequential(root)
    assert seq_count == 2, f"Expected 2 Sequential loops, found {seq_count}"


def test_single_use_module():
    """Test that modules used only once are NOT marked as reuse loops."""

    class SingleUse(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 10)

        def forward(self, x):
            return self.layer(x)

    model = SingleUse()
    x = torch.randn(2, 10)
    root = capture_dynamic_execution_graph(model, x)

    layer_node = None
    for child in root.children:
        if "layer" in child.name:
            layer_node = child
            break

    assert layer_node is not None
    # Should NOT be marked as reuse loop (only called once)
    assert layer_node.operation.is_loop is False


# ============================================================================
# Integration Tests
# ============================================================================


def test_loop_detection_preserves_other_info():
    """Test that loop detection doesn't break other node information."""
    model = SimpleSequential()
    x = torch.randn(2, 10)
    root = capture_dynamic_execution_graph(model, x)

    # Check that all nodes still have proper structure
    assert root.node_id is not None
    assert root.name is not None
    assert root.operation is not None
    assert isinstance(root.children, list)

    # Check Sequential node specifically
    seq_node = None
    for child in root.children:
        if child.operation.op_type == "Sequential":
            seq_node = child
            break

    assert seq_node is not None
    assert seq_node.operation.params_count > 0  # Has parameters
    assert seq_node.operation.is_composite is True  # Is composite
    assert seq_node.is_expandable is True  # Is expandable


def test_loop_info_in_leaf_nodes():
    """Test that leaf nodes (Linear, ReLU) are not marked as loops."""
    model = SimpleSequential()
    x = torch.randn(2, 10)
    root = capture_dynamic_execution_graph(model, x)

    # Find Sequential node and check its children
    seq_node = None
    for child in root.children:
        if child.operation.op_type == "Sequential":
            seq_node = child
            break

    assert seq_node is not None
    assert len(seq_node.children) > 0

    # Check that leaf nodes are NOT marked as loops
    for child in seq_node.children:
        if child.operation.op_type in ["Linear", "ReLU"]:
            assert child.operation.is_loop is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
