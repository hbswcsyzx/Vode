"""Test Input/Output nodes."""

import torch
import torch.nn as nn
from vode.nn import visualize_model

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
input_data = torch.randn(2, 10)

paths = visualize_model(
    model,
    input_data,
    save_path="vode/output/test_io",
    format="gv",
    graph_type="dataflow",
)

with open(paths["dataflow"], "r") as f:
    content = f.read()
    print(content)
    if "Input" in content and "Output" in content:
        print("\n✓ Input and Output nodes present!")
    else:
        print("\n✗ Missing Input or Output nodes!")
