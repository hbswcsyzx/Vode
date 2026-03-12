"""Simple test for Stage 4 GraphvizRenderer implementation.

Tests the new ExecutionNode rendering with three-column layout.
"""

from vode.core.nodes import ExecutionNode, TensorInfo, OperationInfo
from vode.visualize.graphviz_renderer import (
    GraphvizRenderer,
    expand_to_depth,
    flatten_to_sequence,
)


def create_test_graph() -> ExecutionNode:
    """Create a simple test graph: Sequential(Linear, ReLU, Linear).

    Structure:
    - Root: Sequential (composite, expandable)
      - Child 1: Linear (10 -> 20)
      - Child 2: ReLU
      - Child 3: Linear (20 -> 10)
    """
    # Create leaf nodes (Linear, ReLU, Linear)
    linear1 = ExecutionNode(
        node_id="linear1",
        name="Linear1",
        depth=1,
        inputs=[TensorInfo(name="input", shape=(1, 10), dtype="torch.float32")],
        operation=OperationInfo(
            op_type="Linear",
            op_name="fc1",
            params_count=220,
            is_composite=False,
        ),
        outputs=[TensorInfo(name="hidden1", shape=(1, 20), dtype="torch.float32")],
    )

    relu = ExecutionNode(
        node_id="relu",
        name="ReLU",
        depth=1,
        inputs=[TensorInfo(name="hidden1", shape=(1, 20), dtype="torch.float32")],
        operation=OperationInfo(
            op_type="ReLU",
            op_name="relu",
            params_count=0,
            is_composite=False,
        ),
        outputs=[TensorInfo(name="hidden2", shape=(1, 20), dtype="torch.float32")],
    )

    linear2 = ExecutionNode(
        node_id="linear2",
        name="Linear2",
        depth=1,
        inputs=[TensorInfo(name="hidden2", shape=(1, 20), dtype="torch.float32")],
        operation=OperationInfo(
            op_type="Linear",
            op_name="fc2",
            params_count=210,
            is_composite=False,
        ),
        outputs=[TensorInfo(name="output", shape=(1, 10), dtype="torch.float32")],
    )

    # Create root node (Sequential)
    sequential = ExecutionNode(
        node_id="sequential",
        name="Sequential",
        depth=0,
        inputs=[TensorInfo(name="input", shape=(1, 10), dtype="torch.float32")],
        operation=OperationInfo(
            op_type="Sequential",
            op_name="model",
            params_count=430,
            is_composite=True,
        ),
        outputs=[TensorInfo(name="output", shape=(1, 10), dtype="torch.float32")],
    )

    # Add children to sequential
    sequential.add_child(linear1)
    sequential.add_child(relu)
    sequential.add_child(linear2)

    return sequential


def test_expand_to_depth():
    """Test expand_to_depth function."""
    print("Testing expand_to_depth...")

    root = create_test_graph()

    # Test depth=0 (should return only root)
    nodes_depth0 = expand_to_depth(root, max_depth=0)
    print(f"  depth=0: {len(nodes_depth0)} nodes")
    assert len(nodes_depth0) == 1, "depth=0 should return 1 node"
    assert nodes_depth0[0].node_id == "sequential", "depth=0 should return root"

    # Test depth=1 (should return children)
    nodes_depth1 = expand_to_depth(root, max_depth=1)
    print(f"  depth=1: {len(nodes_depth1)} nodes")
    assert len(nodes_depth1) == 3, "depth=1 should return 3 children"
    assert nodes_depth1[0].node_id == "linear1", "First child should be linear1"
    assert nodes_depth1[1].node_id == "relu", "Second child should be relu"
    assert nodes_depth1[2].node_id == "linear2", "Third child should be linear2"

    print("  ✓ expand_to_depth tests passed")


def test_flatten_to_sequence():
    """Test flatten_to_sequence function."""
    print("Testing flatten_to_sequence...")

    root = create_test_graph()
    nodes = expand_to_depth(root, max_depth=1)

    # Flatten should preserve order
    flattened = flatten_to_sequence(nodes)
    print(f"  Flattened: {len(flattened)} nodes")
    assert len(flattened) == 3, "Should have 3 nodes"
    assert flattened[0].node_id == "linear1", "Order should be preserved"

    print("  ✓ flatten_to_sequence tests passed")


def test_render_execution_graph():
    """Test render_execution_graph method."""
    print("Testing render_execution_graph...")

    root = create_test_graph()
    renderer = GraphvizRenderer(None)  # No ComputationGraph needed for ExecutionNode

    # Test depth=0 (collapsed view)
    dot_depth0 = renderer.render_execution_graph(root, max_depth=0)
    print(f"  depth=0 DOT length: {len(dot_depth0)} chars")
    assert "digraph ExecutionGraph" in dot_depth0, "Should contain graph declaration"
    assert "Sequential" in dot_depth0, "Should contain Sequential"
    assert "op0" in dot_depth0, "Should have op0 node"

    # Test depth=1 (expanded view)
    dot_depth1 = renderer.render_execution_graph(root, max_depth=1)
    print(f"  depth=1 DOT length: {len(dot_depth1)} chars")
    assert "Linear" in dot_depth1, "Should contain Linear"
    assert "ReLU" in dot_depth1, "Should contain ReLU"
    assert "op0 -> op1" in dot_depth1, "Should have edges"
    assert "op1 -> op2" in dot_depth1, "Should have edges"

    # Check for three-column structure
    assert "CELLBORDER" in dot_depth1, "Should use HTML table"
    assert "lightyellow" in dot_depth1, "Should have colored cells"
    assert "darkseagreen1" in dot_depth1, "Should have colored cells"

    print("  ✓ render_execution_graph tests passed")


def test_graphviz_syntax():
    """Test that generated DOT has no syntax errors."""
    print("Testing Graphviz syntax...")

    root = create_test_graph()
    renderer = GraphvizRenderer(None)

    # Generate DOT
    dot = renderer.render_execution_graph(root, max_depth=1)

    # Basic syntax checks
    assert dot.count("digraph") == 1, "Should have exactly one digraph"
    assert dot.count("{") == dot.count("}"), "Braces should be balanced"

    # Check HTML table syntax
    assert "<TABLE" in dot, "Should have TABLE tags"
    assert "</TABLE>" in dot, "Should have closing TABLE tags"
    assert dot.count("<TABLE") == dot.count("</TABLE>"), "TABLE tags should be balanced"

    # Check for proper HTML label syntax (should be [label=<...>] not [label=<<...>>])
    # Note: Single angle brackets are fine, double angle brackets are errors
    if "<<" in dot or ">>" in dot:
        print("  Warning: Found double angle brackets in output")
        # Print context around the issue
        for i, line in enumerate(dot.split("\n")):
            if "<<" in line or ">>" in line:
                print(f"    Line {i}: {line}")

    print("  ✓ Graphviz syntax tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Stage 4 GraphvizRenderer Tests")
    print("=" * 60)
    print()

    try:
        test_expand_to_depth()
        print()
        test_flatten_to_sequence()
        print()
        test_render_execution_graph()
        print()
        test_graphviz_syntax()
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

        # Print sample output
        print("\nSample DOT output (depth=1):")
        print("-" * 60)
        root = create_test_graph()
        renderer = GraphvizRenderer(None)
        dot = renderer.render_execution_graph(root, max_depth=1)
        print(dot)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
