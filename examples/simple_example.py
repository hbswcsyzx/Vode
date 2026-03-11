"""Simple VODE usage example.

Demonstrates basic static and dynamic capture with a small model.

NOTE: Run from workspace root with vode in PYTHONPATH:
    cd /path/to/workspace
    python vode/examples/simple_example.py
"""

import sys
from pathlib import Path

# Add vode to path (if not installed)
vode_path = Path(__file__).parent.parent.parent
if vode_path.exists():
    sys.path.insert(0, str(vode_path))

import torch
import torch.nn as nn
from vode.visualize import vode


# Define a simple model
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


def main():
    """Run simple examples."""
    print("VODE Simple Example")
    print("=" * 50)

    # Create model
    model = SimpleNet()
    print(f"\nModel: {model.__class__.__name__}")

    # Example 1: Static capture (no input needed)
    print("\n1. Static capture (structure only)...")
    output_path = vode(model, mode="static", output="simple_static.svg")
    print(f"   Generated: {output_path}")

    # Example 2: Dynamic capture (with input)
    print("\n2. Dynamic capture (with tensor shapes)...")
    x = torch.randn(1, 10)
    output_path = vode(
        model, x, mode="dynamic", output="simple_dynamic.svg", compute_stats=True
    )
    print(f"   Generated: {output_path}")

    # Example 3: Different formats
    print("\n3. Different output formats...")
    vode(model, mode="static", output="simple.png", format="png")
    print(f"   Generated: simple.png")

    vode(model, mode="static", output="simple.gv", format="gv")
    print(f"   Generated: simple.gv (Graphviz source)")

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("\nGenerated files:")
    print("  - simple_static.svg")
    print("  - simple_dynamic.svg")
    print("  - simple.png")
    print("  - simple.gv")


if __name__ == "__main__":
    main()
