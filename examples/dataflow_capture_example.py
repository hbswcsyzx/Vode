"""Example demonstrating DataflowCapture usage.

This example shows how to use DataflowCapture to track tensor operations
during model forward pass.
"""

import torch
import torch.nn as nn

from vode.nn.capture import DataflowCapture, RecorderTensor
from vode.nn.graph.nodes import TensorNode


def main():
    """Run dataflow capture example."""
    print("=== VODE DataflowCapture Example ===\n")

    # Create a simple model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    # Create input data
    input_data = torch.randn(1, 10)

    # Wrap input as RecorderTensor with initial TensorNode
    input_node = TensorNode(
        node_id="input_0",
        name="input",
        depth=0,
        tensor_id=str(id(input_data)),
        shape=tuple(input_data.shape),
        dtype=str(input_data.dtype),
        device=str(input_data.device),
    )
    rec_input = input_data.as_subclass(RecorderTensor)
    rec_input.tensor_nodes = [input_node]

    # Capture dataflow during forward pass
    with DataflowCapture(model) as capture:
        output = model(rec_input)
        graph = capture.get_graph()

    # Display results
    print(f"Input shape: {tuple(input_data.shape)}")
    print(f"Output shape: {tuple(torch.Tensor.size(output))}")
    print(f"\nCaptured {len(graph.get_nodes())} nodes")
    print(f"Captured {len(graph.get_edges())} edges")

    # Show node types
    from vode.nn.graph.nodes import ModuleNode

    module_nodes = [n for n in graph.get_nodes() if isinstance(n, ModuleNode)]
    tensor_nodes = [n for n in graph.get_nodes() if isinstance(n, TensorNode)]

    print(f"\nModuleNodes: {len(module_nodes)}")
    for node in module_nodes:
        print(f"  - {node.name} (depth={node.depth})")

    print(f"\nTensorNodes: {len(tensor_nodes)}")
    for node in tensor_nodes:
        print(f"  - {node.name} shape={node.shape} (depth={node.depth})")

    print("\nDataflow edges:")
    for edge in graph.get_edges():
        src = graph.get_node(edge.src_id)
        dst = graph.get_node(edge.dst_id)
        print(f"  {src.name} -> {dst.name} ({edge.label})")


if __name__ == "__main__":
    main()
