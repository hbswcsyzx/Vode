"""Debug script to trace edge creation."""

import torch
import torch.nn as nn
from vode.nn.capture.dataflow_capture import DataflowCapture
from vode.nn.capture.recorder_tensor import RecorderTensor
from vode.nn.graph.nodes import TensorNode, ModuleNode
from vode.nn.graph.builder import Edge

model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
input_data = torch.randn(2, 10).as_subclass(RecorderTensor)
input_data.tensor_nodes = []

with DataflowCapture(model) as capture:
    output = model(input_data)
    graph = capture.get_graph()

    print(f"Initial edges count: {len(graph.edges)}")
    for edge in graph.edges:
        print(f"  {edge.src_id} -> {edge.dst_id}")

    # Filter by depth
    all_nodes = graph.get_nodes()
    module_nodes = [n for n in all_nodes if isinstance(n, ModuleNode)]
    target_depth = max(node.depth for node in module_nodes) if module_nodes else 0

    filtered_nodes = {}
    for node in all_nodes:
        if isinstance(node, ModuleNode) and node.depth == target_depth:
            node.parents = []
            node.children = []
            filtered_nodes[node.node_id] = node

    filtered_edges = []
    for edge in graph.get_edges():
        if edge.src_id in filtered_nodes and edge.dst_id in filtered_nodes:
            filtered_edges.append(edge)

    print(f"\nFiltered edges count: {len(filtered_edges)}")
    for edge in filtered_edges:
        print(f"  {edge.src_id} -> {edge.dst_id}")

    graph.nodes = filtered_nodes
    graph.edges = filtered_edges

    print(f"\nAfter assignment, edges count: {len(graph.edges)}")

    # Add Input/Output nodes
    input_node = TensorNode(
        node_id="input_tensor",
        name="Input",
        depth=-1,
        tensor_id="input",
        shape=(2, 10),
        dtype="float32",
        device="cpu",
    )
    output_node = TensorNode(
        node_id="output_tensor",
        name="Output",
        depth=-1,
        tensor_id="output",
        shape=(2, 5),
        dtype="float32",
        device="cpu",
    )

    graph.nodes["input_tensor"] = input_node
    graph.nodes["output_tensor"] = output_node

    print(f"\nAfter adding I/O nodes, edges count: {len(graph.edges)}")
    for edge in graph.edges:
        print(f"  {edge.src_id} -> {edge.dst_id}")

    # Create new edges
    module_ids = list(filtered_nodes.keys())
    print(f"\nmodule_ids: {module_ids}")

    nodes_with_incoming = {edge.dst_id for edge in filtered_edges}
    print(f"nodes_with_incoming: {nodes_with_incoming}")

    first_nodes = [nid for nid in module_ids if nid not in nodes_with_incoming]
    print(f"first_nodes: {first_nodes}")

    nodes_with_outgoing = {edge.src_id for edge in filtered_edges}
    print(f"nodes_with_outgoing: {nodes_with_outgoing}")

    last_nodes = [nid for nid in module_ids if nid not in nodes_with_outgoing]
    print(f"last_nodes: {last_nodes}")

    new_edges = list(filtered_edges)
    print(f"\nStarting with {len(new_edges)} edges from filtered_edges")

    for first_node_id in first_nodes:
        print(f"Adding edge: input_tensor -> {first_node_id}")
        new_edges.append(Edge(src_id="input_tensor", dst_id=first_node_id))

    for last_node_id in last_nodes:
        print(f"Adding edge: {last_node_id} -> output_tensor")
        new_edges.append(Edge(src_id=last_node_id, dst_id="output_tensor"))

    print(f"\nNew edges count before assignment: {len(new_edges)}")
    for edge in new_edges:
        print(f"  {edge.src_id} -> {edge.dst_id}")

    graph.edges = new_edges

    print(f"\nFinal edges count: {len(graph.edges)}")
    for edge in graph.edges:
        print(f"  {edge.src_id} -> {edge.dst_id}")
