"""Tests for Stage 4 ExecutionNode-based capture mechanisms."""

import torch
import torch.nn as nn
import pytest

from vode.capture.static_capture import capture_static_execution_graph
from vode.capture.dynamic_capture import capture_dynamic_execution_graph

# Check if graphviz is available
HAS_GRAPHVIZ = False
try:
    import graphviz

    HAS_GRAPHVIZ = True
    from vode.visualize.graphviz_renderer import render_execution_graph
except ImportError:
    pass


class SimpleModel(nn.Module):
    """Simple test model with clear hierarchy."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class NestedModel(nn.Module):
    """Model with nested Sequential structure."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def test_static_capture_simple_model():
    """Test static capture builds correct ExecutionNode hierarchy."""
    model = SimpleModel()
    root = capture_static_execution_graph(model)

    # Check root node
    assert root is not None
    assert root.name == "SimpleModel"
    assert root.depth == 0
    assert root.is_expandable is True
    assert len(root.children) == 3  # layer1, relu, layer2

    # Check children
    child_names = [child.name for child in root.children]
    assert "layer1" in child_names
    assert "relu" in child_names
    assert "layer2" in child_names

    # Check operation info
    assert root.operation.op_type == "SimpleModel"
    assert root.operation.is_composite is True

    # Check that Linear layers have parameters
    layer1 = next(c for c in root.children if c.name == "layer1")
    assert layer1.operation.params_count > 0
    assert layer1.operation.op_type == "Linear"

    # Check that ReLU has no parameters
    relu = next(c for c in root.children if c.name == "relu")
    assert relu.operation.params_count == 0
    assert relu.operation.op_type == "ReLU"


def test_static_capture_nested_model():
    """Test static capture handles nested Sequential structures."""
    model = NestedModel()
    root = capture_static_execution_graph(model)

    # Check root
    assert root.name == "NestedModel"
    assert len(root.children) == 2  # encoder, decoder

    # Check encoder
    encoder = next(c for c in root.children if c.name == "encoder")
    assert encoder.operation.op_type == "Sequential"
    assert encoder.is_expandable is True
    assert len(encoder.children) == 3  # Linear, ReLU, Linear

    # Check decoder
    decoder = next(c for c in root.children if c.name == "decoder")
    assert decoder.operation.op_type == "Sequential"
    assert decoder.is_expandable is True
    assert len(decoder.children) == 3  # Linear, ReLU, Linear


def test_static_capture_parent_child_relationships():
    """Test parent-child relationships are properly established."""
    model = NestedModel()
    root = capture_static_execution_graph(model)

    # Check root has no parent
    assert root.parent is None

    # Check children have correct parent
    for child in root.children:
        assert child.parent is root
        assert child.depth == 1

    # Check grandchildren
    encoder = next(c for c in root.children if c.name == "encoder")
    for grandchild in encoder.children:
        assert grandchild.parent is encoder
        assert grandchild.depth == 2


def test_dynamic_capture_simple_model():
    """Test dynamic capture populates tensor information correctly."""
    model = SimpleModel()
    x = torch.randn(5, 10)

    root = capture_dynamic_execution_graph(model, x)

    # Check root node
    assert root is not None
    assert root.name == "SimpleModel"
    assert len(root.inputs) == 1
    assert len(root.outputs) == 1

    # Check input tensor info
    assert root.inputs[0].shape == (5, 10)
    assert "torch.float32" in root.inputs[0].dtype
    assert root.inputs[0].device is not None

    # Check output tensor info
    assert root.outputs[0].shape == (5, 10)
    assert "torch.float32" in root.outputs[0].dtype

    # Check children have tensor info
    for child in root.children:
        if child.name == "layer1":
            assert len(child.inputs) == 1
            assert child.inputs[0].shape == (5, 10)
            assert len(child.outputs) == 1
            assert child.outputs[0].shape == (5, 20)
        elif child.name == "relu":
            assert len(child.inputs) == 1
            assert child.inputs[0].shape == (5, 20)
            assert len(child.outputs) == 1
            assert child.outputs[0].shape == (5, 20)
        elif child.name == "layer2":
            assert len(child.inputs) == 1
            assert child.inputs[0].shape == (5, 20)
            assert len(child.outputs) == 1
            assert child.outputs[0].shape == (5, 10)


def test_dynamic_capture_nested_model():
    """Test dynamic capture handles nested structures with tensor info."""
    model = NestedModel()
    x = torch.randn(3, 10)

    root = capture_dynamic_execution_graph(model, x)

    # Check root
    assert len(root.inputs) == 1
    assert root.inputs[0].shape == (3, 10)
    assert len(root.outputs) == 1
    assert root.outputs[0].shape == (3, 10)

    # Check encoder
    encoder = next(c for c in root.children if c.name == "encoder")
    assert len(encoder.inputs) == 1
    assert encoder.inputs[0].shape == (3, 10)
    assert len(encoder.outputs) == 1
    assert encoder.outputs[0].shape == (3, 30)

    # Check decoder
    decoder = next(c for c in root.children if c.name == "decoder")
    assert len(decoder.inputs) == 1
    assert decoder.inputs[0].shape == (3, 30)
    assert len(decoder.outputs) == 1
    assert decoder.outputs[0].shape == (3, 10)


def test_render_static_execution_graph():
    """Test static capture can be rendered with new renderer."""
    if not HAS_GRAPHVIZ:
        print("⊘ Skipped (graphviz not installed)")
        return

    model = SimpleModel()
    root = capture_static_execution_graph(model)

    # Test rendering at different depths
    dot_depth0 = render_execution_graph(root, max_depth=0)
    assert dot_depth0 is not None
    assert "SimpleModel" in dot_depth0.source

    dot_depth1 = render_execution_graph(root, max_depth=1)
    assert dot_depth1 is not None
    assert "layer1" in dot_depth1.source or "Linear" in dot_depth1.source


def test_render_dynamic_execution_graph():
    """Test dynamic capture can be rendered with new renderer."""
    if not HAS_GRAPHVIZ:
        print("⊘ Skipped (graphviz not installed)")
        return

    model = SimpleModel()
    x = torch.randn(5, 10)
    root = capture_dynamic_execution_graph(model, x)

    # Test rendering at different depths
    dot_depth0 = render_execution_graph(root, max_depth=0)
    assert dot_depth0 is not None
    assert "SimpleModel" in dot_depth0.source

    dot_depth1 = render_execution_graph(root, max_depth=1)
    assert dot_depth1 is not None
    # Should show children
    assert "layer1" in dot_depth1.source or "Linear" in dot_depth1.source


def test_expandable_flags():
    """Test is_expandable flags are set correctly."""
    model = NestedModel()
    root = capture_static_execution_graph(model)

    # Root should be expandable (has children)
    assert root.is_expandable is True

    # Sequential modules should be expandable
    encoder = next(c for c in root.children if c.name == "encoder")
    assert encoder.is_expandable is True

    # Leaf modules should not be expandable
    linear = encoder.children[0]  # First Linear in encoder
    assert linear.is_expandable is False


def test_empty_model():
    """Test capture handles empty model gracefully."""

    class EmptyModel(nn.Module):
        def forward(self, x):
            return x

    model = EmptyModel()
    root = capture_static_execution_graph(model)

    assert root is not None
    assert root.name == "EmptyModel"
    assert len(root.children) == 0
    assert root.is_expandable is False


def test_dynamic_capture_different_input_types():
    """Test dynamic capture handles different input types."""
    model = SimpleModel()

    # Test with single tensor
    x = torch.randn(5, 10)
    root1 = capture_dynamic_execution_graph(model, x)
    assert root1 is not None
    assert len(root1.inputs) == 1

    # Test with tuple
    root2 = capture_dynamic_execution_graph(model, (x,))
    assert root2 is not None
    assert len(root2.inputs) == 1


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Testing static capture with simple model...")
    test_static_capture_simple_model()
    print("✓ Passed")

    print("Testing static capture with nested model...")
    test_static_capture_nested_model()
    print("✓ Passed")

    print("Testing parent-child relationships...")
    test_static_capture_parent_child_relationships()
    print("✓ Passed")

    print("Testing dynamic capture with simple model...")
    test_dynamic_capture_simple_model()
    print("✓ Passed")

    print("Testing dynamic capture with nested model...")
    test_dynamic_capture_nested_model()
    print("✓ Passed")

    print("Testing render static execution graph...")
    test_render_static_execution_graph()
    print("✓ Passed")

    print("Testing render dynamic execution graph...")
    test_render_dynamic_execution_graph()
    print("✓ Passed")

    print("Testing expandable flags...")
    test_expandable_flags()
    print("✓ Passed")

    print("Testing empty model...")
    test_empty_model()
    print("✓ Passed")

    print("Testing different input types...")
    test_dynamic_capture_different_input_types()
    print("✓ Passed")

    print("\n✅ All tests passed!")
