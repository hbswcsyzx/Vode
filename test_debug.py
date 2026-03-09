"""Test debug functionality."""
import torch
import torch.nn as nn
from vode.nn import visualize_model

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
input_data = torch.randn(2, 10)

print("=== Testing with debug=True ===")
paths = visualize_model(
    model, input_data, 
    save_path="vode/output/test_debug", 
    format="gv", 
    graph_type="dataflow",
    debug=True
)

print(f"\n=== Generated: {paths['dataflow']} ===")
