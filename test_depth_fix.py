"""Test script to verify depth filtering fixes."""

import torch
import torch.nn as nn
from vode.nn import visualize_model


class NestedModel(nn.Module):
    """Model with multiple depth levels."""

    def __init__(self):
        super().__init__()
        # Depth 0: NestedModel
        # Depth 1: Sequential
        # Depth 2: Linear, ReLU, Linear
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    """Test depth filtering."""
    model = NestedModel()
    input_data = torch.randn(2, 10)

    # Test 1: Default (show deepest level)
    print("Test 1: Default depth (should show deepest level)")
    paths = visualize_model(
        model,
        input_data,
        save_path="vode/output/test_depth_default",
        format="gv",
        graph_type="dataflow",
        depth_limit=None,
    )
    print(f"Generated: {paths}")

    # Read and print the .gv file to check depths
    with open(paths["dataflow"], "r") as f:
        content = f.read()
        print("\nGenerated graph content:")
        print(content)
        print("\n" + "=" * 80 + "\n")

    # Test 2: Specific depth level
    print("Test 2: Depth limit = 1")
    paths = visualize_model(
        model,
        input_data,
        save_path="vode/output/test_depth_1",
        format="gv",
        graph_type="dataflow",
        depth_limit=1,
    )
    print(f"Generated: {paths}")

    with open(paths["dataflow"], "r") as f:
        content = f.read()
        print("\nGenerated graph content:")
        print(content)


if __name__ == "__main__":
    main()
