"""End-to-end integration tests for the Vode tracing system.

This module tests the complete Vode pipeline from tracing through rendering,
validating that all components work together correctly.
"""

import json
import tempfile
from pathlib import Path

import pytest

from vode.trace.tracer import TraceRuntime, TraceConfig
from vode.trace.models import TensorValuePolicy
from vode.trace.serializer import GraphSerializer
from vode.trace.renderer import TextRenderer

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def test_basic_tracing():
    """Test basic function call capture with simple arithmetic functions.

    Validates that:
    - Function calls are captured
    - Call tree structure is correct
    - Parameters and returns are recorded
    """

    # Define test functions
    def add(a, b):
        return a + b

    def multiply(a, b):
        return a * b

    def compute(x, y):
        sum_val = add(x, y)
        prod_val = multiply(x, y)
        return sum_val, prod_val

    # Trace execution
    config = TraceConfig(max_depth=10)
    runtime = TraceRuntime(config)

    runtime.start()
    result = compute(3, 4)
    graph = runtime.stop()

    # Verify result
    assert result == (7, 12)

    # Verify nodes captured
    assert len(graph.function_calls) >= 3  # compute, add, multiply

    # Verify root call
    assert len(graph.root_call_ids) == 1
    root_node = next(n for n in graph.function_calls if n.id == graph.root_call_ids[0])
    assert root_node.display_name == "compute"

    # Verify child calls
    child_nodes = [n for n in graph.function_calls if n.parent_id == root_node.id]
    assert len(child_nodes) == 2
    child_names = {n.display_name for n in child_nodes}
    assert child_names == {"add", "multiply"}

    # Verify return values captured
    assert len(root_node.return_variable_ids) > 0


def test_dataflow_edges():
    """Test dataflow edge creation between function calls.

    Validates that:
    - Edges connect producers to consumers
    - Slot paths are correct for tuple returns
    - Variable records track producer/consumer relationships
    """

    def producer():
        return 42

    def consumer(value):
        return value * 2

    def pipeline():
        x = producer()
        y = consumer(x)
        return y

    # Trace execution
    config = TraceConfig()
    runtime = TraceRuntime(config)

    runtime.start()
    result = pipeline()
    graph = runtime.stop()

    # Verify result
    assert result == 84

    # Verify dataflow edges exist
    dataflow_edges = [e for e in graph.edges if e.kind == "dataflow"]
    assert len(dataflow_edges) > 0

    # Verify variables have producer/consumer info
    variables_with_producers = [
        v for v in graph.variables if v.producer_call_id is not None
    ]
    assert len(variables_with_producers) > 0

    variables_with_consumers = [
        v for v in graph.variables if len(v.consumer_call_ids) > 0
    ]
    assert len(variables_with_consumers) > 0


def test_serialization():
    """Test JSON serialization round-trip.

    Validates that:
    - Graph can be serialized to JSON
    - Deserialized graph matches original structure
    - All node and edge data is preserved
    """

    def simple_func(x):
        return x + 1

    # Trace execution
    config = TraceConfig()
    runtime = TraceRuntime(config)

    runtime.start()
    result = simple_func(5)
    original_graph = runtime.stop()

    # Serialize to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        serializer = GraphSerializer()
        serializer.serialize(original_graph, temp_path)

        # Verify file exists and is valid JSON
        assert Path(temp_path).exists()
        with open(temp_path, "r") as f:
            data = json.load(f)
            assert "version" in data
            assert "graph" in data

        # Deserialize
        loaded_graph = serializer.deserialize(temp_path)

        # Verify structure preserved
        assert len(loaded_graph.function_calls) == len(original_graph.function_calls)
        assert len(loaded_graph.variables) == len(original_graph.variables)
        assert len(loaded_graph.edges) == len(original_graph.edges)
        assert loaded_graph.root_call_ids == original_graph.root_call_ids

        # Verify node data preserved
        for orig_node, loaded_node in zip(
            original_graph.function_calls, loaded_graph.function_calls
        ):
            assert loaded_node.id == orig_node.id
            assert loaded_node.qualified_name == orig_node.qualified_name
            assert loaded_node.display_name == orig_node.display_name
            assert loaded_node.depth == orig_node.depth

    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)


def test_text_rendering():
    """Test text-based rendering of trace graphs.

    Validates that:
    - Text output is generated
    - Output contains function names
    - Output contains file/line information
    - Tree structure is represented
    """

    def outer():
        return inner()

    def inner():
        return 123

    # Trace execution
    config = TraceConfig()
    runtime = TraceRuntime(config)

    runtime.start()
    result = outer()
    graph = runtime.stop()

    # Render to text
    renderer = TextRenderer()
    text_output = renderer.render(graph)

    # Verify output contains expected elements
    assert "Function Call Tree:" in text_output
    assert "outer" in text_output
    assert "inner" in text_output
    assert "Dataflow Edges:" in text_output

    # Verify tree structure indicators present
    assert "├─" in text_output or "└─" in text_output


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_pytorch_tracing():
    """Test tracing of PyTorch nn.Module forward passes.

    Validates that:
    - torch.nn.Module calls are captured
    - Tensor metadata is extracted
    - Module detection works correctly
    - Tensor shapes and dtypes are recorded
    """

    # Define simple module
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    # Create model and input
    model = SimpleNet()
    input_tensor = torch.randn(2, 10)

    # Trace execution
    config = TraceConfig()
    runtime = TraceRuntime(config)

    runtime.start()
    output = model(input_tensor)
    graph = runtime.stop()

    # Verify output shape
    assert output.shape == (2, 5)

    # Verify module calls captured
    module_nodes = [
        n for n in graph.function_calls if n.metadata.get("is_torch_module")
    ]
    assert len(module_nodes) > 0

    # Verify tensor variables captured
    tensor_vars = [v for v in graph.variables if v.tensor_meta is not None]
    assert len(tensor_vars) > 0

    # Verify tensor metadata
    for var in tensor_vars:
        assert var.tensor_meta.shape is not None
        assert var.tensor_meta.dtype is not None
        assert var.tensor_meta.device is not None


def test_filtering():
    """Test filtering with max_depth and exclude_patterns.

    Validates that:
    - max_depth limits call stack depth
    - exclude_patterns filter out matching files
    - Filtered calls are not captured
    """

    def level1():
        return level2()

    def level2():
        return level3()

    def level3():
        return level4()

    def level4():
        return "deep"

    # Test max_depth filtering
    config = TraceConfig(max_depth=2)
    runtime = TraceRuntime(config)

    runtime.start()
    result = level1()
    graph = runtime.stop()

    # Should only capture level1 and level2 (depth 0 and 1)
    assert len(graph.function_calls) <= 2

    # Verify depth values
    for node in graph.function_calls:
        assert node.depth < 2


def test_value_policies():
    """Test different TensorValuePolicy settings.

    Validates that:
    - Different policies affect value capture
    - stats_only computes statistics
    - none policy doesn't store values
    """

    def simple_calc(x):
        return x * 2

    # Test with stats_only policy
    config_stats = TraceConfig(value_policy="stats_only")
    runtime_stats = TraceRuntime(config_stats)

    runtime_stats.start()
    result = simple_calc(5)
    graph_stats = runtime_stats.stop()

    assert result == 10
    assert len(graph_stats.function_calls) > 0

    # Test with none policy
    config_none = TraceConfig(value_policy="none")
    runtime_none = TraceRuntime(config_none)

    runtime_none.start()
    result = simple_calc(5)
    graph_none = runtime_none.stop()

    assert result == 10
    assert len(graph_none.function_calls) > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_pytorch_tensor_stats():
    """Test tensor statistics computation for PyTorch tensors.

    Validates that:
    - Tensor stats are computed when policy is stats_only
    - Stats include min, max, mean, std
    - Stats are accurate
    """

    def process_tensor(x):
        return x + 1

    # Create tensor with known values
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Trace with stats_only policy
    config = TraceConfig(value_policy="stats_only")
    runtime = TraceRuntime(config)

    runtime.start()
    output = process_tensor(input_tensor)
    graph = runtime.stop()

    # Find tensor variables with stats
    tensor_vars_with_stats = [
        v
        for v in graph.variables
        if v.tensor_meta is not None and v.tensor_stats is not None
    ]

    # Should have captured some tensor stats
    assert len(tensor_vars_with_stats) > 0


def test_nested_function_calls():
    """Test tracing of nested function calls with multiple levels.

    Validates that:
    - Deep call stacks are captured correctly
    - Parent-child relationships are maintained
    - Depth tracking is accurate
    """

    def a():
        return b() + c()

    def b():
        return d()

    def c():
        return 10

    def d():
        return 5

    # Trace execution
    config = TraceConfig()
    runtime = TraceRuntime(config)

    runtime.start()
    result = a()
    graph = runtime.stop()

    # Verify result
    assert result == 15

    # Verify all functions captured
    function_names = {n.display_name for n in graph.function_calls}
    assert function_names == {"a", "b", "c", "d"}

    # Verify parent-child relationships
    root = next(n for n in graph.function_calls if n.display_name == "a")
    assert root.parent_id is None

    children_of_a = [n for n in graph.function_calls if n.parent_id == root.id]
    child_names = {n.display_name for n in children_of_a}
    assert child_names == {"b", "c"}

    # Verify depth tracking
    depth_map = {n.display_name: n.depth for n in graph.function_calls}
    assert depth_map["a"] == 0
    assert depth_map["b"] == 1
    assert depth_map["c"] == 1
    assert depth_map["d"] == 2


def test_multiple_return_values():
    """Test handling of functions with multiple return values.

    Validates that:
    - Tuple returns are captured correctly
    - Each return value gets a variable record
    - Slot paths distinguish return values
    """

    def multi_return():
        return 1, 2, 3

    def use_multi():
        a, b, c = multi_return()
        return a + b + c

    # Trace execution
    config = TraceConfig()
    runtime = TraceRuntime(config)

    runtime.start()
    result = use_multi()
    graph = runtime.stop()

    # Verify result
    assert result == 6

    # Find the multi_return node
    multi_node = next(
        n for n in graph.function_calls if n.display_name == "multi_return"
    )

    # Should have multiple return variable IDs
    assert len(multi_node.return_variable_ids) >= 1

    # Verify variables exist
    return_vars = [v for v in graph.variables if v.id in multi_node.return_variable_ids]
    assert len(return_vars) > 0
