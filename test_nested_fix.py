"""Test nested model with current fix."""

import torch
import torch.nn as nn
from vode.nn import visualize_model


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
input_data = torch.randn(2, 10)

paths = visualize_model(
    model,
    input_data,
    save_path="vode/output/test_nested_fixed",
    format="gv",
    graph_type="dataflow",
    debug=True,
)

print(f"\n=== Generated: {paths['dataflow']} ===")

# Check the edges
with open(paths["dataflow"], "r") as f:
    content = f.read()
    lines = [l.strip() for l in content.split("\n") if "->" in l]
    print("\nEdges in generated graph:")
    for line in lines:
        print(f"  {line}")
