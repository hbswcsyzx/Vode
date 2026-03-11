"""Test visualization functionality."""

import torch
import torch.nn as nn
from vode import vode, capture_static, visualize


# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_basic_visualization():
    """Test basic visualization with simple model."""
    print("Testing basic visualization (static mode)...")

    model = SimpleModel()

    # Test convenience wrapper with static mode (no input needed)
    try:
        output_path = vode(model, mode="static", output="test_simple.gv", format="gv")
        print(f"✓ Generated visualization: {output_path}")

        # Read and print first few lines
        with open(output_path, "r") as f:
            lines = f.readlines()[:20]
            print("\nFirst 20 lines of generated DOT file:")
            print("".join(lines))
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


def test_dynamic_visualization():
    """Test dynamic visualization."""
    print("\n\nTesting dynamic visualization...")

    model = SimpleModel()
    x = torch.randn(2, 10)

    try:
        output_path = vode(
            model, x, mode="dynamic", output="test_dynamic.gv", format="gv"
        )
        print(f"✓ Generated dynamic visualization: {output_path}")

        # Read and print first few lines
        with open(output_path, "r") as f:
            lines = f.readlines()[:20]
            print("\nFirst 20 lines of generated DOT file:")
            print("".join(lines))
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


def test_depth_control():
    """Test depth control."""
    print("\n\nTesting depth control...")

    model = SimpleModel()

    try:
        # Capture graph
        graph = capture_static(model)
        print(f"Graph stats: {graph.get_stats()}")

        # Visualize with depth limit
        output_path = visualize(
            graph, output_path="test_depth.gv", max_depth=2, format="gv"
        )
        print(f"✓ Generated depth-limited visualization: {output_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


def test_nested_model():
    """Test with nested model."""
    print("\n\nTesting nested model...")

    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 10),
            )
            self.decoder = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5),
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    model = NestedModel()

    try:
        output_path = vode(model, mode="static", output="test_nested.gv", format="gv")
        print(f"✓ Generated nested model visualization: {output_path}")

        # Show graph stats
        graph = capture_static(model)
        print(f"Graph stats: {graph.get_stats()}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_basic_visualization()
    test_dynamic_visualization()
    test_depth_control()
    test_nested_model()
    print("\n\nAll tests completed!")
