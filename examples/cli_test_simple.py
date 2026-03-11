"""Simple test script for CLI - creates a basic model.

This script can be used to test the vode CLI:
    vode examples/cli_test_simple.py
    vode --format png --output test.png examples/cli_test_simple.py
"""

import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """Simple feedforward network."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Create the model
model = SimpleNet()
print(f"Created model: {model.__class__.__name__}")
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
