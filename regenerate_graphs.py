"""Regenerate dataflow graphs to verify Input/Output fix."""
import torch
import torch.nn as nn
from vode.nn import visualize_model

# Simple Sequential model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
)

input_data = torch.randn(2, 10)

# Regenerate the graph
paths = visualize_model(
    model,
    input_data,
    save_path="vode/output/complete_format_svg_dataflow",
    format="gv",
    graph_type="dataflow",
)

print(f"Regenerated: {paths['dataflow']}")

# Check the output
with open(paths["dataflow"], "r") as f:
    content = f.read()
    
# Count problematic edges
self_loops = content.count("input_tensor -> input_tensor") + content.count("output_tensor -> output_tensor")
direct_io = content.count("input_tensor -> output_tensor")

print(f"\nSelf-loops found: {self_loops}")
print(f"Direct Input->Output: {direct_io}")

if self_loops == 0 and direct_io == 0:
    print("✓ SUCCESS: No self-loops or direct Input->Output connections!")
else:
    print("✗ FAILED: Still has problematic edges")
    print("\nEdges section:")
    lines = content.split('\n')
    for line in lines:
        if '->' in line:
            print(line)
