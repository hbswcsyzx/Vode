"""Advanced VODE Example: ResNet-style Architecture

This example demonstrates VODE with a more complex architecture featuring:
- Custom residual blocks
- Skip connections
- Nested Sequential modules
- Multiple module types

Usage:
    # Run directly with Python:
    python advanced_example.py

    # Or visualize with VODE CLI:
    vode --stage4 --depth 2 advanced_example.py
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Get script directory and output directory
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "advanced"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only import VODE if not being analyzed by VODE CLI
if __name__ == "__main__":
    from vode.capture.static_capture import capture_static_execution_graph
    from vode.capture.dynamic_capture import capture_dynamic_execution_graph
    from vode.visualize.graphviz_renderer import render_execution_graph, expand_to_depth


class ResidualBlock(nn.Module):
    """Residual block with skip connection.

    Implements: output = F(x) + x
    where F(x) is a two-layer network with ReLU activation.
    """

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x):
        """Forward pass with residual connection."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip connection
        out = self.relu(out)

        return out


class DownsampleBlock(nn.Module):
    """Downsampling block that reduces spatial dimensions.

    Uses strided convolution to downsample while increasing channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass with downsampling."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNetStage(nn.Module):
    """A stage in ResNet consisting of multiple residual blocks.

    Each stage maintains the same spatial dimensions and channel count.
    """

    def __init__(self, features, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(features) for _ in range(num_blocks)]
        )

    def forward(self, x):
        """Forward pass through all blocks in the stage."""
        for block in self.blocks:
            x = block(x)
        return x


class AdvancedModel(nn.Module):
    """Advanced ResNet-style model with multiple stages.

    Architecture:
        - Initial convolution (3 -> 64 channels)
        - Stage 1: 2 residual blocks (64 channels, 32x32)
        - Downsample (64 -> 128 channels, 16x16)
        - Stage 2: 2 residual blocks (128 channels, 16x16)
        - Downsample (128 -> 256 channels, 8x8)
        - Stage 3: 2 residual blocks (256 channels, 8x8)
        - Global average pooling
        - Fully connected layer (256 -> num_classes)
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64 channels
        self.stage1 = ResNetStage(features=64, num_blocks=2)

        # Downsample to 128 channels
        self.downsample1 = DownsampleBlock(64, 128)

        # Stage 2: 128 channels
        self.stage2 = ResNetStage(features=128, num_blocks=2)

        # Downsample to 256 channels
        self.downsample2 = DownsampleBlock(128, 256)

        # Stage 3: 256 channels
        self.stage3 = ResNetStage(features=256, num_blocks=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.initial(x)

        x = self.stage1(x)
        x = self.downsample1(x)

        x = self.stage2(x)
        x = self.downsample2(x)

        x = self.stage3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def print_model_summary(model):
    """Print a summary of the model architecture."""
    print("\nModel Summary:")
    print("-" * 60)

    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

            if params > 0:
                print(f"{name:40s} {params:>10,} params")
                total_params += params
                trainable_params += trainable

    print("-" * 60)
    print(f"{'Total Parameters':40s} {total_params:>10,}")
    print(f"{'Trainable Parameters':40s} {trainable_params:>10,}")
    print("-" * 60)


def demonstrate_depth_expansion(root):
    """Demonstrate expansion at different depths."""
    print("\nDepth Expansion Demonstration:")
    print("-" * 60)

    for depth in range(4):
        expanded = expand_to_depth(root, max_depth=depth)
        print(f"Depth {depth}: {len(expanded)} nodes")

        if depth <= 2:
            for i, node in enumerate(expanded[:5]):  # Show first 5
                print(f"  [{i}] {node.operation.op_type:15s} - {node.name}")
            if len(expanded) > 5:
                print(f"  ... and {len(expanded) - 5} more nodes")


def main():
    """Main function demonstrating advanced VODE usage."""
    print("=" * 60)
    print("VODE Advanced Example: ResNet-style Architecture")
    print("=" * 60)

    # Create model
    print("\n1. Creating advanced model...")
    model = AdvancedModel(num_classes=10)
    print(f"   Model: {model.__class__.__name__}")

    # Print model summary
    print_model_summary(model)

    # Static capture
    print("\n2. Static capture (structure only)...")
    static_root = capture_static_execution_graph(model)
    print(f"   Root node: {static_root.name}")
    print(f"   Number of direct children: {len(static_root.children)}")
    print(f"   Is expandable: {static_root.is_expandable}")

    # Show hierarchy
    print("\n   Model hierarchy:")
    for i, child in enumerate(static_root.children):
        print(f"   [{i}] {child.operation.op_type:15s} - {child.name}")
        if child.is_expandable:
            print(f"       └─ {len(child.children)} sub-modules")

    # Demonstrate depth expansion
    demonstrate_depth_expansion(static_root)

    # Dynamic capture
    print("\n3. Dynamic capture (with runtime data)...")
    sample_input = torch.randn(1, 3, 32, 32)
    print(f"   Sample input shape: {tuple(sample_input.shape)}")

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
    output_file = OUTPUT_DIR / "advanced_depth0.gv"
    with open(output_file, "w") as f:
        f.write(dot_depth0.source)
    print(f"     Saved to: {output_file}")

    # Depth 1: Show main stages
    print("   - Depth 1 (main stages)...")
    dot_depth1 = render_execution_graph(static_root, max_depth=1)
    output_file = OUTPUT_DIR / "advanced_depth1.gv"
    with open(output_file, "w") as f:
        f.write(dot_depth1.source)
    print(f"     Saved to: {output_file}")

    # Depth 2: Show stage internals
    print("   - Depth 2 (stage internals)...")
    dot_depth2 = render_execution_graph(static_root, max_depth=2)
    output_file = OUTPUT_DIR / "advanced_depth2.gv"
    with open(output_file, "w") as f:
        f.write(dot_depth2.source)
    print(f"     Saved to: {output_file}")

    # Dynamic visualization
    print("   - Dynamic (with shapes)...")
    dot_dynamic = render_execution_graph(dynamic_root, max_depth=1)
    output_file = OUTPUT_DIR / "advanced_dynamic.gv"
    with open(output_file, "w") as f:
        f.write(dot_dynamic.source)
    print(f"     Saved to: {output_file}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Custom residual blocks with skip connections")
    print("  ✓ Nested Sequential modules")
    print("  ✓ ModuleList for repeated blocks")
    print("  ✓ Multi-stage architecture")
    print("  ✓ Depth-based expansion control")
    print("\nTo visualize the .gv files:")
    print(f"  cd {OUTPUT_DIR}")
    print("  dot -Tpng advanced_depth1.gv -o advanced_depth1.png")
    print("  dot -Tsvg advanced_depth2.gv -o advanced_depth2.svg")


if __name__ == "__main__":
    main()
