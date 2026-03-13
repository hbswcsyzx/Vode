"""Tests for GraphvizRenderer and rendering functions.

Tests expand_to_depth(), flatten_to_sequence(), and render_execution_graph().
"""

import pytest
import torch
import torch.nn as nn
from vode.core.nodes import ExecutionNode, TensorInfo, OperationInfo
from vode.visualize.graphviz_renderer import (
    expand_to_depth,
    flatten_to_sequence,
    GraphvizRenderer,
)
from vode.capture.computation_tracer import capture_static_execution_graph
from vode.capture.computation_tracer import capture_dynamic_execution_graph


class TestExpandToDepth:
    """Test expand_to_depth function."""

    def test_depth_zero(self):
        """Test expansion at depth 0 (show only root)."""
        # Create a simple hierarchy
        root = ExecutionNode(
            node_id="root",
            name="Sequential",
            depth=0,
            inputs=[],
            operation=OperationInfo(
                op_type="Sequential", op_name="seq", is_composite=True
            ),
            outputs=[],
            is_expandable=True,
        )

        child1 = ExecutionNode(
            node_id="child1",
            name="Linear",
            depth=1,
            inputs=[],
            operation=OperationInfo(op_type="Linear", op_name="fc1"),
            outputs=[],
        )

        root.add_child(child1)

        # Expand to depth 0 should return only root
        result = expand_to_depth(root, max_depth=0)

        assert len(result) == 1
        assert result[0] == root

    def test_depth_one(self):
        """Test expansion at depth 1 (show immediate children)."""
        root = ExecutionNode(
            node_id="root",
            name="Sequential",
            depth=0,
            inputs=[],
            operation=OperationInfo(
                op_type="Sequential", op_name="seq", is_composite=True
            ),
            outputs=[],
            is_expandable=True,
        )

        child1 = ExecutionNode(
            node_id="child1",
            name="Linear",
            depth=1,
            inputs=[],
            operation=OperationInfo(op_type="Linear", op_name="fc1"),
            outputs=[],
        )

        child2 = ExecutionNode(
            node_id="child2",
            name="ReLU",
            depth=1,
            inputs=[],
            operation=OperationInfo(op_type="ReLU", op_name="relu"),
            outputs=[],
        )

        root.add_child(child1)
        root.add_child(child2)

        # Expand to depth 1 should return children
        result = expand_to_depth(root, max_depth=1)

        assert len(result) == 2
        assert child1 in result
        assert child2 in result
        assert root not in result

    def test_depth_two(self):
        """Test expansion at depth 2 (two levels deep)."""
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

        # Expand to depth 2 should return level2
        result = expand_to_depth(root, max_depth=2)

        assert len(result) == 1
        assert result[0] == level2

    def test_non_expandable_node(self):
        """Test that non-expandable nodes are returned as-is."""
        node = ExecutionNode(
            node_id="node",
            name="Linear",
            depth=0,
            inputs=[],
            operation=OperationInfo(op_type="Linear", op_name="fc"),
            outputs=[],
            is_expandable=False,
        )

        result = expand_to_depth(node, max_depth=5)

        assert len(result) == 1
        assert result[0] == node

    def test_mixed_expandable_nodes(self):
        """Test expansion with mix of expandable and non-expandable nodes."""
        root = ExecutionNode(
            node_id="root",
            name="Sequential",
            depth=0,
            inputs=[],
            operation=OperationInfo(
                op_type="Sequential", op_name="seq", is_composite=True
            ),
            outputs=[],
            is_expandable=True,
        )

        # Expandable child
        expandable_child = ExecutionNode(
            node_id="expandable",
            name="Block",
            depth=1,
            inputs=[],
            operation=OperationInfo(
                op_type="Block", op_name="block", is_composite=True
            ),
            outputs=[],
            is_expandable=True,
        )

        # Non-expandable child
        leaf_child = ExecutionNode(
            node_id="leaf",
            name="ReLU",
            depth=1,
            inputs=[],
            operation=OperationInfo(op_type="ReLU", op_name="relu"),
            outputs=[],
            is_expandable=False,
        )

        # Grandchild
        grandchild = ExecutionNode(
            node_id="grandchild",
            name="Linear",
            depth=2,
            inputs=[],
            operation=OperationInfo(op_type="Linear", op_name="fc"),
            outputs=[],
        )

        root.add_child(expandable_child)
        root.add_child(leaf_child)
        expandable_child.add_child(grandchild)

        # Expand to depth 1
        result = expand_to_depth(root, max_depth=1)

        # Should get expandable_child and leaf_child
        assert len(result) == 2
        assert expandable_child in result
        assert leaf_child in result

    def test_real_model_expansion(self):
        """Test expansion on a real PyTorch model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        root = capture_static_execution_graph(model)

        # Depth 0: just root
        result0 = expand_to_depth(root, max_depth=0)
        assert len(result0) == 1

        # Depth 1: children
        result1 = expand_to_depth(root, max_depth=1)
        assert len(result1) == 3


class TestFlattenToSequence:
    """Test flatten_to_sequence function."""

    def test_already_flat(self):
        """Test that flat list is returned as-is."""
        nodes = [
            ExecutionNode(
                node_id=f"node{i}",
                name=f"Node{i}",
                depth=0,
                inputs=[],
                operation=OperationInfo(op_type="Linear", op_name=f"fc{i}"),
                outputs=[],
            )
            for i in range(3)
        ]

        result = flatten_to_sequence(nodes)

        assert len(result) == 3
        assert result == nodes

    def test_empty_list(self):
        """Test with empty list."""
        result = flatten_to_sequence([])
        assert len(result) == 0


class TestGraphvizRendererExecutionGraph:
    """Test GraphvizRenderer.render_execution_graph method."""

    def test_render_simple_model(self):
        """Test rendering a simple model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
        )

        root = capture_static_execution_graph(model)

        # Create renderer
        renderer = GraphvizRenderer()

        # Render at depth 1
        dot_string = renderer.render_execution_graph(root, max_depth=1)

        # Check basic structure
        assert "digraph ExecutionGraph" in dot_string
        assert "op0" in dot_string
        assert "op1" in dot_string
        assert "op0 -> op1" in dot_string

    def test_render_depth_zero(self):
        """Test rendering at depth 0."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
        )

        root = capture_static_execution_graph(model)

        renderer = GraphvizRenderer()

        # Render at depth 0 (only root)
        dot_string = renderer.render_execution_graph(root, max_depth=0)

        assert "digraph ExecutionGraph" in dot_string
        assert "op0" in dot_string
        # Should only have one node
        assert "op1" not in dot_string

    def test_render_with_tensor_info(self):
        """Test rendering with actual tensor information."""
        model = nn.Linear(10, 20)
        x = torch.randn(1, 10)

        root = capture_dynamic_execution_graph(model, x)

        renderer = GraphvizRenderer()

        dot_string = renderer.render_execution_graph(root, max_depth=0)

        # Check that shapes are included
        assert "(1, 10)" in dot_string
        assert "(1, 20)" in dot_string

    def test_render_nested_model(self):
        """Test rendering nested model structure."""

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 20)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.fc(x))

        model = nn.Sequential(
            Block(),
            nn.Linear(20, 10),
        )

        root = capture_static_execution_graph(model)

        renderer = GraphvizRenderer()

        # Render at depth 1
        dot_string = renderer.render_execution_graph(root, max_depth=1)

        assert "digraph ExecutionGraph" in dot_string
        assert "op0" in dot_string
        assert "op1" in dot_string

    def test_render_rankdir(self):
        """Test different rankdir options."""
        model = nn.Linear(10, 20)
        root = capture_static_execution_graph(model)

        renderer = GraphvizRenderer()

        # Test LR (left-right)
        dot_lr = renderer.render_execution_graph(root, max_depth=0, rankdir="LR")
        assert "rankdir=LR" in dot_lr

        # Test TB (top-bottom)
        dot_tb = renderer.render_execution_graph(root, max_depth=0, rankdir="TB")
        assert "rankdir=TB" in dot_tb


class TestGraphvizRendererHelpers:
    """Test GraphvizRenderer helper methods."""

    def test_format_shape(self):
        """Test _format_shape method."""
        renderer = GraphvizRenderer()

        # Test normal shape
        assert renderer._format_shape((1, 10, 20)) == "(1, 10, 20)"

        # Test None
        assert renderer._format_shape(None) == "unknown"

        # Test empty tuple
        assert renderer._format_shape(()) == "()"

    def test_format_number(self):
        """Test _format_number method."""
        renderer = GraphvizRenderer()

        # Test small numbers
        assert renderer._format_number(100) == "100"
        assert renderer._format_number(999) == "999"

        # Test thousands
        assert renderer._format_number(1000) == "1.0K"
        assert renderer._format_number(5500) == "5.5K"

        # Test millions
        assert renderer._format_number(1_000_000) == "1.0M"
        assert renderer._format_number(2_500_000) == "2.5M"

        # Test billions
        assert renderer._format_number(1_000_000_000) == "1.0B"
        assert renderer._format_number(3_500_000_000) == "3.5B"

    def test_format_tensors_for_column(self):
        """Test _format_tensors_for_column method."""
        renderer = GraphvizRenderer()

        # Test empty list
        assert renderer._format_tensors_for_column([]) == "-"

        # Test single tensor
        tensors = [TensorInfo(name="input", shape=(1, 10))]
        result = renderer._format_tensors_for_column(tensors)
        assert "input" in result
        assert "(1, 10)" in result

        # Test multiple tensors
        tensors = [
            TensorInfo(name="input1", shape=(1, 10)),
            TensorInfo(name="input2", shape=(1, 20)),
        ]
        result = renderer._format_tensors_for_column(tensors)
        assert "input1" in result
        assert "input2" in result

    def test_format_operation_for_column(self):
        """Test _format_operation_for_column method."""
        renderer = GraphvizRenderer()

        # Test simple operation
        op = OperationInfo(op_type="Linear", op_name="fc1", params_count=220)
        result = renderer._format_operation_for_column(op)
        assert "Linear" in result
        assert "fc1" in result
        assert "220" in result

        # Test composite operation
        op = OperationInfo(
            op_type="Sequential",
            op_name="block",
            params_count=1000,
            is_composite=True,
        )
        result = renderer._format_operation_for_column(op)
        assert "Sequential" in result
        assert "composite" in result

        # Test parameterless operation
        op = OperationInfo(op_type="ReLU", op_name="relu", params_count=0)
        result = renderer._format_operation_for_column(op)
        assert "ReLU" in result
        assert "params" not in result


class TestRenderExecutionNodeHTML:
    """Test _render_execution_node_html method."""

    def test_render_with_all_info(self):
        """Test rendering node with complete information."""
        inputs = [TensorInfo(name="input", shape=(1, 10))]
        operation = OperationInfo(op_type="Linear", op_name="fc1", params_count=220)
        outputs = [TensorInfo(name="output", shape=(1, 20))]

        node = ExecutionNode(
            node_id="node",
            name="fc1",
            depth=0,
            inputs=inputs,
            operation=operation,
            outputs=outputs,
        )

        renderer = GraphvizRenderer()

        html = renderer._render_execution_node_html(node)

        # Check structure
        assert "<TABLE" in html
        assert "input" in html
        assert "(1, 10)" in html
        assert "Linear" in html
        assert "fc1" in html
        assert "output" in html
        assert "(1, 20)" in html

    def test_render_without_tensors(self):
        """Test rendering node without tensor information."""
        operation = OperationInfo(op_type="ReLU", op_name="relu")

        node = ExecutionNode(
            node_id="node",
            name="relu",
            depth=0,
            inputs=[],
            operation=operation,
            outputs=[],
        )

        renderer = GraphvizRenderer()

        html = renderer._render_execution_node_html(node)

        assert "<TABLE" in html
        assert "ReLU" in html
        # Should show "-" for empty inputs/outputs
        assert "-" in html


class TestIntegration:
    """Integration tests for the full rendering pipeline."""

    def test_static_to_render(self):
        """Test full pipeline from static capture to rendering."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        )

        # Capture
        root = capture_static_execution_graph(model)

        # Render
        renderer = GraphvizRenderer()
        dot_string = renderer.render_execution_graph(root, max_depth=1)

        # Verify output
        assert "digraph ExecutionGraph" in dot_string
        assert "Linear" in dot_string
        assert "ReLU" in dot_string

    def test_dynamic_to_render(self):
        """Test full pipeline from dynamic capture to rendering."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
        )
        x = torch.randn(1, 10)

        # Capture
        root = capture_dynamic_execution_graph(model, x)

        # Render
        renderer = GraphvizRenderer()
        dot_string = renderer.render_execution_graph(root, max_depth=1)

        # Verify output includes shapes
        assert "(1, 10)" in dot_string
        assert "(1, 20)" in dot_string
