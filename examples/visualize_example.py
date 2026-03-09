"""Example demonstrating the high-level visualize_model API.

This example shows how to use the simple visualize_model() function to generate
both structure and dataflow graphs for a PyTorch model.
"""

import torch
import torch.nn as nn
from vode.nn import visualize_model


def main():
    """Demonstrate visualize_model API with a simple neural network."""

    # Create a simple model
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    # Create sample input
    input_data = torch.randn(1, 10)

    print("=" * 60)
    print("VODE High-Level API Example")
    print("=" * 60)

    # Example 1: Generate both graphs as SVG
    print("\n1. Generating both structure and dataflow graphs as SVG...")
    paths = visualize_model(
        model,
        input_data,
        save_path="output/example_model",
        format="svg",
        graph_type="both",
    )
    print(f"   Generated files:")
    for graph_type, path in paths.items():
        print(f"   - {graph_type}: {path}")

    # Example 2: Generate only structure graph as PNG
    print("\n2. Generating structure graph only as PNG...")
    paths = visualize_model(
        model,
        input_data,
        save_path="output/example_structure",
        format="png",
        graph_type="structure",
    )
    print(f"   Generated files:")
    for graph_type, path in paths.items():
        print(f"   - {graph_type}: {path}")

    # Example 3: Generate only dataflow graph as PDF
    print("\n3. Generating dataflow graph only as PDF...")
    paths = visualize_model(
        model,
        input_data,
        save_path="output/example_dataflow",
        format="pdf",
        graph_type="dataflow",
    )
    print(f"   Generated files:")
    for graph_type, path in paths.items():
        print(f"   - {graph_type}: {path}")

    # Example 4: Generate Graphviz source files only (no rendering)
    print("\n4. Generating Graphviz source files only (.gv)...")
    paths = visualize_model(
        model,
        input_data,
        save_path="output/example_source",
        format="gv",
        graph_type="both",
    )
    print(f"   Generated files:")
    for graph_type, path in paths.items():
        print(f"   - {graph_type}: {path}")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
