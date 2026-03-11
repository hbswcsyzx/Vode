"""Complex test script for CLI - creates a nested model with arguments.

This script can be used to test the vode CLI with arguments:
    vode examples/cli_test_complex.py --layers 3 --hidden 64
    vode --depth 2 --format pdf examples/cli_test_complex.py --layers 5
"""

import argparse
import torch
import torch.nn as nn


class ComplexNet(nn.Module):
    """Multi-layer network with configurable depth."""

    def __init__(self, input_size=10, hidden_size=20, num_layers=3, output_size=10):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a complex model")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--hidden", type=int, default=20, help="Hidden size")
    args = parser.parse_args()

    # Create the model with arguments
    model = ComplexNet(num_layers=args.layers, hidden_size=args.hidden)
    print(f"Created model: {model.__class__.__name__}")
    print(f"  Layers: {args.layers}")
    print(f"  Hidden size: {args.hidden}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
