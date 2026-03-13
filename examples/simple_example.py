"""Simple VODE Example: Basic Sequential Model

This example demonstrates basic VODE usage with a simple feedforward neural network.
The model consists of three linear layers with ReLU activations.

Usage:
    # Run directly with Python:
    python simple_example.py

    # Or visualize with VODE CLI:
    vode --depth 1 simple_example.py
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Get script directory and output directory
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "simple"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only import VODE if not being analyzed by VODE CLI
# This prevents recursive analysis when using vode command
if __name__ == "__main__":
    from vode.capture.computation_tracer import (
        capture_static_execution_graph,
        capture_dynamic_execution_graph,
    )
    from vode.visualize.graphviz_renderer import render_execution_graph


class SimpleModel(nn.Module):
    """Simple feedforward neural network.

    Architecture:
        Input (10) -> Linear (20) -> ReLU -> Linear (20) -> ReLU -> Linear (10)
    """

    def __init__(self, input_size=10, hidden_size=20, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def main():
    """Main function demonstrating VODE usage."""
    print("=" * 60)
    print("VODE Simple Example: Basic Sequential Model")
    print("=" * 60)

    # Create model
    print("\n1. Creating model...")
    model = SimpleModel(input_size=10, hidden_size=20, output_size=10)
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Static capture (no forward pass needed)
    print("\n2. Static capture (structure only)...")
    static_root = capture_static_execution_graph(model)
    print(f"   Captured root node: {static_root.name}")
    print(f"   Number of children: {len(static_root.children)}")
    print(f"   Is expandable: {static_root.is_expandable}")

    # Dynamic capture (with sample input)
    print("\n3. Dynamic capture (with runtime data)...")
    sample_input = torch.randn(1, 10)
    dynamic_root = capture_dynamic_execution_graph(model, sample_input)
    print(
        f"   Input shape: {dynamic_root.inputs[0].shape if dynamic_root.inputs else 'N/A'}"
    )
    print(
        f"   Output shape: {dynamic_root.outputs[0].shape if dynamic_root.outputs else 'N/A'}"
    )

    # Render at different depths
    print("\n4. Rendering visualizations...")

    # Depth 0: Show only root
    print("   - Depth 0 (root only)...")
    dot_depth0 = render_execution_graph(static_root, max_depth=0)
    output_file = OUTPUT_DIR / "simple_depth0.gv"
    with open(output_file, "w") as f:
        f.write(dot_depth0.source)
    print(f"     Saved to: {output_file}")

    # Depth 1: Show immediate children
    print("   - Depth 1 (one level)...")
    dot_depth1 = render_execution_graph(static_root, max_depth=1)
    output_file = OUTPUT_DIR / "simple_depth1.gv"
    with open(output_file, "w") as f:
        f.write(dot_depth1.source)
    print(f"     Saved to: {output_file}")

    # Dynamic visualization with shapes
    print("   - Dynamic (with shapes)...")
    dot_dynamic = render_execution_graph(dynamic_root, max_depth=1)
    output_file = OUTPUT_DIR / "simple_dynamic.gv"
    with open(output_file, "w") as f:
        f.write(dot_dynamic.source)
    print(f"     Saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nTo visualize the .gv files:")
    print(f"  cd {OUTPUT_DIR}")
    print("  dot -Tpng simple_depth1.gv -o simple_depth1.png")
    print("  dot -Tsvg simple_dynamic.gv -o simple_dynamic.svg")


if __name__ == "__main__":
    main()
