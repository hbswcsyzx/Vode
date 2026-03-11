#!/usr/bin/env python3
"""
Test vode tool on unifolm-world-model-action inference model.
Validates captured structure against model.log.
"""

import sys
import os
from pathlib import Path

# Add vode module path
vode_path = Path("/home/zyf/XXX")
if vode_path.exists():
    sys.path.insert(0, str(vode_path))

# Add unifolm directory to Python path
unifolm_path = Path("/home/zyf/XXX/unifolm-world-model-action/src")
if unifolm_path.exists():
    sys.path.insert(0, str(unifolm_path))

import torch
from omegaconf import OmegaConf

# Import vode
from vode import capture_static, visualize


def load_model_from_config(config_path, ckpt_path=None):
    """Load model from config file."""
    print(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)

    # Import the instantiate function
    from unifolm_wma.utils.utils import instantiate_from_config

    # Instantiate model
    print("Instantiating model from config...")
    model = instantiate_from_config(config.model)

    # Load checkpoint if provided
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model


def read_model_log(log_path):
    """Read model.log and extract module names."""
    with open(log_path, "r") as f:
        lines = f.readlines()

    modules = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            # Extract module names (lines that contain ':' followed by module type)
            if ":" in stripped and "(" in stripped:
                # Extract the module path before the colon
                module_name = stripped.split(":")[0].strip()
                # Remove leading parentheses and numbers
                module_name = (
                    module_name.lstrip("(").lstrip("0123456789").lstrip("-").strip()
                )
                if module_name and not module_name.startswith("("):
                    modules.append(module_name)

    return modules


def compare_structures(captured_modules, log_modules):
    """Compare captured modules with model.log modules."""
    captured_set = set(captured_modules)
    log_set = set(log_modules)

    # Find matches and differences
    matches = captured_set & log_set
    missing = log_set - captured_set
    extra = captured_set - log_set

    return {
        "matches": sorted(matches),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "match_rate": len(matches) / len(log_set) if log_set else 0,
    }


def test_static_capture():
    """Test static capture on unifolm model."""
    print("\n" + "=" * 80)
    print("STATIC CAPTURE TEST")
    print("=" * 80 + "\n")

    config_path = "/home/zyf/XXX/unifolm-world-model-action/configs/inference/world_model_interaction.yaml"
    ckpt_path = "/home/zyf/XXX/unifolm-world-model-action/ckpts/unifolm_wma_dual.ckpt"

    try:
        # Load model
        model = load_model_from_config(
            config_path, ckpt_path if os.path.exists(ckpt_path) else None
        )
        print(f"Model loaded successfully: {type(model).__name__}")

        # Capture static structure
        print("\nCapturing static structure...")
        graph = capture_static(model)

        # Extract module names from captured structure
        captured_modules = []

        def extract_module_names_from_graph(graph):
            """Extract module names from graph nodes."""
            for node_id, node in graph.nodes.items():
                if hasattr(node, "metadata") and "module_path" in node.metadata:
                    captured_modules.append(node.metadata["module_path"])
                elif hasattr(node, "name"):
                    captured_modules.append(node.name)

        extract_module_names_from_graph(graph)

        print(f"Captured {len(captured_modules)} modules")
        print(f"Root nodes: {graph.root_node_ids}")
        print(f"Total nodes: {len(graph.nodes)}")

        # Generate visualizations at different depths
        print("\nGenerating visualizations...")

        # Full depth
        visualize(
            graph, output_path="unifolm_static_full.gv", max_depth=None, format="gv"
        )
        print("  - unifolm_static_full.gv (full depth)")

        # Depth 3
        visualize(graph, output_path="unifolm_static_d3.gv", max_depth=3, format="gv")
        print("  - unifolm_static_d3.gv (depth 3)")

        # Depth 5
        visualize(graph, output_path="unifolm_static_d5.gv", max_depth=5, format="gv")
        print("  - unifolm_static_d5.gv (depth 5)")

        return graph, captured_modules

    except Exception as e:
        print(f"Error during static capture: {e}")
        import traceback

        traceback.print_exc()
        return None, []


def test_dynamic_capture():
    """Test dynamic capture on unifolm model (if feasible)."""
    print("\n" + "=" * 80)
    print("DYNAMIC CAPTURE TEST")
    print("=" * 80 + "\n")

    print("Dynamic capture requires:")
    print("  - conda activate new environment")
    print("  - Sample input data")
    print("  - Running inference")
    print("\nSkipping dynamic capture for now (can be added later)")
    print("Focus on static capture validation first")

    return None


def validate_against_log():
    """Validate captured structure against model.log."""
    print("\n" + "=" * 80)
    print("VALIDATION AGAINST model.log")
    print("=" * 80 + "\n")

    log_path = "model.log"

    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found")
        return

    # Read model.log
    print(f"Reading {log_path}...")
    log_modules = read_model_log(log_path)
    print(f"Found {len(log_modules)} modules in model.log")

    # Show first few modules from log
    print("\nFirst 10 modules from model.log:")
    for i, mod in enumerate(log_modules[:10]):
        print(f"  {i+1}. {mod}")

    return log_modules


def main():
    """Main test function."""
    print("=" * 80)
    print("VODE TOOL TEST ON UNIFOLM-WORLD-MODEL-ACTION")
    print("=" * 80)

    # Test 1: Validate against model.log
    log_modules = validate_against_log()

    # Test 2: Static capture
    graph, captured_modules = test_static_capture()

    # Test 3: Dynamic capture (optional)
    # test_dynamic_capture()

    # Test 4: Compare results
    if graph and log_modules:
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80 + "\n")

        comparison = compare_structures(captured_modules, log_modules)

        print(f"Match rate: {comparison['match_rate']*100:.1f}%")
        print(f"Matches: {len(comparison['matches'])}")
        print(f"Missing from capture: {len(comparison['missing'])}")
        print(f"Extra in capture: {len(comparison['extra'])}")

        if comparison["missing"]:
            print("\nFirst 10 missing modules:")
            for mod in comparison["missing"][:10]:
                print(f"  - {mod}")

        if comparison["extra"]:
            print("\nFirst 10 extra modules:")
            for mod in comparison["extra"][:10]:
                print(f"  + {mod}")

        # Save detailed results
        with open("test_results.txt", "w") as f:
            f.write("VODE TOOL TEST RESULTS\n")
            f.write("=" * 80 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model: LatentVisualDiffusion (WMAModel)\n")
            f.write(f"Captured modules: {len(captured_modules)}\n")
            f.write(f"Expected modules (from model.log): {len(log_modules)}\n")
            f.write(f"Match rate: {comparison['match_rate']*100:.1f}%\n")
            f.write(f"Matches: {len(comparison['matches'])}\n")
            f.write(f"Missing: {len(comparison['missing'])}\n")
            f.write(f"Extra: {len(comparison['extra'])}\n\n")

            f.write("VISUALIZATIONS GENERATED\n")
            f.write("-" * 80 + "\n")
            f.write("  - unifolm_static_full.svg (full depth)\n")
            f.write("  - unifolm_static_d3.svg (depth 3)\n")
            f.write("  - unifolm_static_d5.svg (depth 5)\n\n")

            if comparison["missing"]:
                f.write("MISSING MODULES (from model.log but not captured)\n")
                f.write("-" * 80 + "\n")
                for mod in comparison["missing"]:
                    f.write(f"  - {mod}\n")
                f.write("\n")

            if comparison["extra"]:
                f.write("EXTRA MODULES (captured but not in model.log)\n")
                f.write("-" * 80 + "\n")
                for mod in comparison["extra"]:
                    f.write(f"  + {mod}\n")
                f.write("\n")

            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            if comparison["match_rate"] >= 0.9:
                f.write("✓ Excellent match rate (>90%)\n")
                f.write("  - Vode successfully captures the model structure\n")
            elif comparison["match_rate"] >= 0.7:
                f.write("✓ Good match rate (70-90%)\n")
                f.write("  - Minor discrepancies may be due to naming conventions\n")
            else:
                f.write("⚠ Low match rate (<70%)\n")
                f.write("  - Review missing modules\n")
                f.write("  - Check if model.log format differs from captured format\n")

            if comparison["missing"]:
                f.write("  - Investigate why some modules are missing\n")
            if comparison["extra"]:
                f.write("  - Extra modules may be internal/helper modules\n")

        print("\nDetailed results saved to: test_results.txt")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
