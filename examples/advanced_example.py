"""Advanced VODE usage example.

Demonstrates depth control, loop detection, and complex model visualization.

NOTE: Run from workspace root with vode in PYTHONPATH:
    cd /path/to/workspace
    python vode/examples/advanced_example.py
"""

import sys
from pathlib import Path

# Add vode to path (if not installed)
vode_path = Path(__file__).parent.parent.parent
if vode_path.exists():
    sys.path.insert(0, str(vode_path))

import torch
import torch.nn as nn
from vode.capture import capture_static, capture_dynamic
from vode.visualize import visualize


# Define a complex model with nested modules
class ConvBlock(nn.Module):
    """Convolutional block with batch norm and activation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity  # Skip connection
        return out


class ComplexNet(nn.Module):
    """Complex network with multiple levels of nesting."""

    def __init__(self):
        super().__init__()
        self.input_conv = ConvBlock(3, 64)

        # Use Sequential (detected as loop)
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def demonstrate_depth_control():
    """Show how depth control helps visualize large models."""
    print("\nDEPTH CONTROL DEMONSTRATION")
    print("=" * 50)

    model = ComplexNet()

    # Full depth
    print("\n1. Full depth (all modules)...")
    graph = capture_static(model)
    print(f"   Captured {len(graph.nodes)} nodes")
    visualize(graph, output_path="complex_full.svg")
    print(f"   Generated: complex_full.svg")

    # Depth 3 (high-level overview)
    print("\n2. Depth 3 (high-level overview)...")
    visualize(graph, output_path="complex_d3.svg", max_depth=3)
    print(f"   Generated: complex_d3.svg")

    # Depth 5 (medium detail)
    print("\n3. Depth 5 (medium detail)...")
    visualize(graph, output_path="complex_d5.svg", max_depth=5)
    print(f"   Generated: complex_d5.svg")


def demonstrate_loop_detection():
    """Show loop detection for Sequential and ModuleList."""
    print("\n\nLOOP DETECTION DEMONSTRATION")
    print("=" * 50)

    # Model with Sequential (detected as loop)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )

    print("\n1. Sequential model (detected as loop)...")
    graph = capture_static(model)
    print(f"   Captured {len(graph.nodes)} nodes")
    print(f"   Detected {len(graph.detected_loops)} loops")

    # Collapsed view
    print("\n2. Collapsed loop view...")
    visualize(graph, output_path="loop_collapsed.svg", collapse_loops=True)
    print(f"   Generated: loop_collapsed.svg")

    # Expanded view
    print("\n3. Expanded loop view...")
    visualize(graph, output_path="loop_expanded.svg", collapse_loops=False)
    print(f"   Generated: loop_expanded.svg")


def demonstrate_dynamic_capture():
    """Show dynamic capture with tensor metadata."""
    print("\n\nDYNAMIC CAPTURE DEMONSTRATION")
    print("=" * 50)

    model = ComplexNet()
    x = torch.randn(2, 3, 32, 32)

    print("\n1. Dynamic capture without statistics...")
    graph = capture_dynamic(model, x, compute_stats=False)
    print(f"   Captured {len(graph.nodes)} nodes")
    visualize(graph, output_path="dynamic_no_stats.svg")
    print(f"   Generated: dynamic_no_stats.svg")

    print("\n2. Dynamic capture with statistics...")
    graph = capture_dynamic(model, x, compute_stats=True)
    print(f"   Captured {len(graph.nodes)} nodes")
    visualize(graph, output_path="dynamic_with_stats.svg")
    print(f"   Generated: dynamic_with_stats.svg")

    # Show some tensor metadata
    print("\n3. Sample tensor metadata:")
    for node_id, node in list(graph.nodes.items())[:3]:
        if hasattr(node, "shape"):
            print(f"   - {node.name}: shape={node.shape}, dtype={node.dtype}")


def demonstrate_separate_api():
    """Show separate capture and visualization API."""
    print("\n\nSEPARATE API DEMONSTRATION")
    print("=" * 50)

    model = ComplexNet()

    print("\n1. Capture once...")
    graph = capture_static(model)
    print(f"   Captured {len(graph.nodes)} nodes")

    print("\n2. Visualize multiple times with different settings...")

    # Different depths
    visualize(graph, output_path="separate_d3.svg", max_depth=3)
    print(f"   Generated: separate_d3.svg (depth 3)")

    visualize(graph, output_path="separate_d5.svg", max_depth=5)
    print(f"   Generated: separate_d5.svg (depth 5)")

    # Different formats
    visualize(graph, output_path="separate.png", format="png")
    print(f"   Generated: separate.png")

    # Different orientations
    visualize(graph, output_path="separate_tb.svg", rankdir="TB")
    print(f"   Generated: separate_tb.svg (top-bottom layout)")


def main():
    """Run all advanced examples."""
    print("\n" + "=" * 70)
    print("VODE ADVANCED EXAMPLES")
    print("=" * 70)

    try:
        demonstrate_depth_control()
        demonstrate_loop_detection()
        demonstrate_dynamic_capture()
        demonstrate_separate_api()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nKey takeaways:")
        print("  - Use max_depth to control visualization complexity")
        print("  - Sequential/ModuleList are detected as loops")
        print("  - Dynamic capture provides tensor shapes and statistics")
        print("  - Capture once, visualize many times with different settings")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
