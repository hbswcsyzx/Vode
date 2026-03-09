"""Comprehensive example demonstrating all VODE Stage 3 features.

This example showcases:
- Multiple model architectures (Sequential, Nested, Skip Connections)
- All graph types (structure, dataflow, both)
- All output formats (svg, png, pdf, gv)
- Complete end-to-end workflow
"""

import torch
import torch.nn as nn
from vode.nn import visualize_model


# ============================================================================
# Model Definitions
# ============================================================================


class SimpleSequential(nn.Module):
    """Simple sequential model for basic demonstration."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

    def forward(self, x):
        return self.layers(x)


class CustomBlock(nn.Module):
    """Custom module block for nested model demonstration."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class NestedModel(nn.Module):
    """Model with nested custom modules."""

    def __init__(self):
        super().__init__()
        self.block1 = CustomBlock(10, 20)
        self.block2 = CustomBlock(20, 15)
        self.block3 = CustomBlock(15, 5)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Linear(features, features)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Linear(features, features)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + identity  # Skip connection
        out = self.relu2(out)
        return out


class SkipConnectionModel(nn.Module):
    """Model with skip connections (residual-like architecture)."""

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(10, 16)
        self.res_block1 = ResidualBlock(16)
        self.res_block2 = ResidualBlock(16)
        self.output_layer = nn.Linear(16, 5)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.output_layer(x)
        return x


# ============================================================================
# Visualization Functions
# ============================================================================


def visualize_simple_sequential():
    """Demonstrate visualization of a simple sequential model."""
    print("\n" + "=" * 70)
    print("1. SIMPLE SEQUENTIAL MODEL")
    print("=" * 70)

    model = SimpleSequential()
    input_data = torch.randn(2, 10)

    # Generate all graph types as SVG
    print("\n  a) Generating both structure and dataflow graphs (SVG)...")
    paths = visualize_model(
        model,
        input_data,
        save_path="output/complete_simple",
        format="svg",
        graph_type="both",
    )
    for graph_type, path in paths.items():
        print(f"     - {graph_type}: {path}")

    # Generate structure only as PNG
    print("\n  b) Generating structure graph only (PNG)...")
    paths = visualize_model(
        model,
        input_data,
        save_path="output/complete_simple_struct",
        format="png",
        graph_type="structure",
    )
    for graph_type, path in paths.items():
        print(f"     - {graph_type}: {path}")


def visualize_nested_model():
    """Demonstrate visualization of a nested model with custom modules."""
    print("\n" + "=" * 70)
    print("2. NESTED MODEL WITH CUSTOM MODULES")
    print("=" * 70)

    model = NestedModel()
    input_data = torch.randn(2, 10)

    # Generate both graphs as PDF
    print("\n  a) Generating both structure and dataflow graphs (PDF)...")
    paths = visualize_model(
        model,
        input_data,
        save_path="output/complete_nested",
        format="pdf",
        graph_type="both",
    )
    for graph_type, path in paths.items():
        print(f"     - {graph_type}: {path}")

    # Generate dataflow only as SVG
    print("\n  b) Generating dataflow graph only (SVG)...")
    paths = visualize_model(
        model,
        input_data,
        save_path="output/complete_nested_dataflow",
        format="svg",
        graph_type="dataflow",
    )
    for graph_type, path in paths.items():
        print(f"     - {graph_type}: {path}")


def visualize_skip_connection_model():
    """Demonstrate visualization of a model with skip connections."""
    print("\n" + "=" * 70)
    print("3. MODEL WITH SKIP CONNECTIONS (RESIDUAL-LIKE)")
    print("=" * 70)

    model = SkipConnectionModel()
    input_data = torch.randn(2, 10)

    # Generate both graphs as PNG
    print("\n  a) Generating both structure and dataflow graphs (PNG)...")
    paths = visualize_model(
        model,
        input_data,
        save_path="output/complete_residual",
        format="png",
        graph_type="both",
    )
    for graph_type, path in paths.items():
        print(f"     - {graph_type}: {path}")


def demonstrate_all_formats():
    """Demonstrate all output formats with a single model."""
    print("\n" + "=" * 70)
    print("4. ALL OUTPUT FORMATS DEMONSTRATION")
    print("=" * 70)

    model = SimpleSequential()
    input_data = torch.randn(2, 10)

    formats = ["svg", "png", "pdf", "gv"]

    for fmt in formats:
        print(f"\n  Generating graphs in {fmt.upper()} format...")
        paths = visualize_model(
            model,
            input_data,
            save_path=f"output/complete_format_{fmt}",
            format=fmt,
            graph_type="both",
        )
        for graph_type, path in paths.items():
            print(f"     - {graph_type}: {path}")


def print_summary():
    """Print summary of all generated files."""
    print("\n" + "=" * 70)
    print("SUMMARY OF GENERATED FILES")
    print("=" * 70)

    expected_files = [
        # Simple Sequential
        "output/complete_simple_structure.svg",
        "output/complete_simple_dataflow.svg",
        "output/complete_simple_struct_structure.png",
        # Nested Model
        "output/complete_nested_structure.pdf",
        "output/complete_nested_dataflow.pdf",
        "output/complete_nested_dataflow_dataflow.svg",
        # Skip Connection Model
        "output/complete_residual_structure.png",
        "output/complete_residual_dataflow.png",
        # All Formats
        "output/complete_format_svg_structure.svg",
        "output/complete_format_svg_dataflow.svg",
        "output/complete_format_png_structure.png",
        "output/complete_format_png_dataflow.png",
        "output/complete_format_pdf_structure.pdf",
        "output/complete_format_pdf_dataflow.pdf",
        "output/complete_format_gv_structure.gv",
        "output/complete_format_gv_dataflow.gv",
    ]

    print("\nExpected output files:")
    for i, file_path in enumerate(expected_files, 1):
        print(f"  {i:2d}. {file_path}")

    print(f"\nTotal files generated: {len(expected_files)}")
    print("\nNote: Actual file generation depends on Graphviz availability.")
    print("      .gv files are always generated; rendered formats require Graphviz.")


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("VODE STAGE 3 - COMPREHENSIVE EXAMPLE")
    print("=" * 70)
    print("\nThis example demonstrates:")
    print("  - Multiple model architectures")
    print("  - All graph types (structure, dataflow, both)")
    print("  - All output formats (svg, png, pdf, gv)")
    print("  - Complete end-to-end workflow")

    try:
        # Run all demonstrations
        visualize_simple_sequential()
        visualize_nested_model()
        visualize_skip_connection_model()
        demonstrate_all_formats()

        # Print summary
        print_summary()

        print("\n" + "=" * 70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
