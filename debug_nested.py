"""Debug nested model to understand edge structure."""

import torch
import torch.nn as nn
from vode.nn.capture.dataflow_capture import DataflowCapture
from vode.nn.capture.recorder_tensor import RecorderTensor
from vode.nn.graph.nodes import ModuleNode


class CustomBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = CustomBlock(10, 20)
        self.block2 = CustomBlock(20, 15)
        self.block3 = CustomBlock(15, 5)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


model = NestedModel()
input_data = torch.randn(2, 10).as_subclass(RecorderTensor)
input_data.tensor_nodes = []

with DataflowCapture(model) as capture:
    output = model(input_data)
    graph = capture.get_graph()

    print("=== All nodes by depth ===")
    for depth in range(3):
        nodes_at_depth = [
            n
            for n in graph.get_nodes()
            if isinstance(n, ModuleNode) and n.depth == depth
        ]
        print(f"\nDepth {depth}: {len(nodes_at_depth)} nodes")
        for node in nodes_at_depth:
            print(f"  {node.node_id}: {node.module_type}")

    print("\n=== All edges ===")
    for edge in graph.edges:
        src_node = graph.get_node(edge.src_id)
        dst_node = graph.get_node(edge.dst_id)
        src_depth = src_node.depth if src_node else "?"
        dst_depth = dst_node.depth if dst_node else "?"
        src_type = type(src_node).__name__ if src_node else "?"
        dst_type = type(dst_node).__name__ if dst_node else "?"
        print(
            f"  [{src_depth}] {edge.src_id} ({src_type}) -> [{dst_depth}] {edge.dst_id} ({dst_type})"
        )
