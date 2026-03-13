"""Command-line interface for VODE.

Provides commands for:
- trace: Capture model execution (function flow or computation flow)
- export: Export static visualization
- view: Interactive visualization (future)
- editor: Visual programming interface (future)

Usage:
    vode trace [options] script.py [script_args...]
    vode export [options] trace_file output_file
    vode view [options] trace_file
    vode script.py  # Direct visualization (shortcut)
"""

import argparse
import sys
import runpy
from pathlib import Path
from typing import Any, List

# Import from reorganized modules
from vode.capture import (
    capture_static,
    capture_dynamic,
    capture_static_execution_graph,
    capture_dynamic_execution_graph,
)
from vode.core import save_graph, load_graph
from vode.visualize import visualize, start_server
from vode.visualize.graphviz_renderer import (
    render_execution_graph,
    expand_to_depth,
    flatten_to_sequence,
)

# Global registry for tracking created models
_model_registry: List[Any] = []


def _track_model_creation():
    """Hook into torch.nn.Module.__init__ to track model creation."""
    try:
        import torch.nn as nn

        original_init = nn.Module.__init__

        def tracked_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _model_registry.append(self)

        nn.Module.__init__ = tracked_init
    except ImportError:
        pass  # PyTorch not available


def _restore_model_tracking():
    """Restore original torch.nn.Module.__init__."""
    _model_registry.clear()


def cmd_trace(args):
    """Trace command: capture model structure and save to file."""
    from vode import capture_static, capture_dynamic
    from vode.core.graph import ComputationGraph
    import json

    # Track model creation
    _track_model_creation()

    # Execute the script
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}", file=sys.stderr)
        return 1

    # Prepare sys.argv for the script
    old_argv = sys.argv
    sys.argv = [str(script_path)] + args.script_args

    try:
        # Run the script
        runpy.run_path(str(script_path), run_name="__main__")

        # Get the model
        if not _model_registry:
            print("Warning: No PyTorch models detected in script", file=sys.stderr)
            return 1

        # Use the last created model (usually the main model)
        model = _model_registry[-1]
        print(f"Detected model: {model.__class__.__name__}")

        # Capture based on mode
        if args.mode == "static":
            print("Capturing static structure...")
            graph = capture_static(model)
        else:
            print(
                "Error: Dynamic mode requires sample input (use Python API)",
                file=sys.stderr,
            )
            return 1

        # Save to file
        output_path = args.output or f"{script_path.stem}_trace.json"
        print(f"Saving trace to: {output_path}")

        # Convert graph to JSON
        graph_data = {
            "nodes": [node.to_dict() for node in graph.all_nodes.values()],
            "root_id": graph.root_node.node_id if graph.root_node else None,
            "mode": args.mode,
        }

        with open(output_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        print(f"✓ Trace saved successfully")
        return 0

    except Exception as e:
        print(f"Error during tracing: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        sys.argv = old_argv
        _restore_model_tracking()


def cmd_view(args):
    """View command: visualize a saved trace or directly visualize a model."""
    from vode import visualize
    from vode.core.graph import ComputationGraph
    import json

    if args.graph_file:
        # Load from file
        print(f"Loading trace from: {args.graph_file}")
        with open(args.graph_file, "r") as f:
            graph_data = json.load(f)

        # Reconstruct graph (simplified)
        print("Reconstructing graph...")
        # TODO: Implement proper graph reconstruction
        print("Error: Graph reconstruction not yet implemented", file=sys.stderr)
        return 1
    else:
        print("Error: No graph file specified", file=sys.stderr)
        return 1


def cmd_visualize(args):
    """Visualize command: directly visualize a model from script."""
    from vode import capture_static, visualize

    # Track model creation
    _track_model_creation()

    # Execute the script
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}", file=sys.stderr)
        return 1

    # Prepare sys.argv for the script
    old_argv = sys.argv
    sys.argv = [str(script_path)] + args.script_args

    try:
        # Run the script
        print(f"Executing script: {script_path}")
        runpy.run_path(str(script_path), run_name="__main__")

        # Get the model
        if not _model_registry:
            print("Warning: No PyTorch models detected in script", file=sys.stderr)
            return 1

        # Use the last created model or find by name
        if args.model_name:
            # TODO: Implement model name matching
            model = _model_registry[-1]
        else:
            model = _model_registry[-1]

        print(f"Visualizing model: {model.__class__.__name__}")

        # Capture model structure
        print(f"Capturing structure (depth={args.depth or 1})...")

        if args.mode == "static":
            exec_node = capture_static_execution_graph(model)
        else:
            print(
                "Error: Dynamic mode requires sample input (use Python API)",
                file=sys.stderr,
            )
            return 1

        # Render visualization
        dot_graph = render_execution_graph(exec_node, max_depth=args.depth or 1)

        # Determine output path and format
        if args.output:
            output_path = args.output
            # Infer format from file extension if output is specified
            if "." in output_path:
                inferred_format = output_path.rsplit(".", 1)[1]
                if inferred_format in ["gv", "png", "svg", "pdf"]:
                    format_to_use = inferred_format
                else:
                    format_to_use = args.format
            else:
                format_to_use = args.format
        else:
            format_to_use = args.format
            # Save to output directory if it exists, otherwise current directory
            output_dir = script_path.parent / "output" / script_path.stem
            if output_dir.exists():
                output_path = str(output_dir / f"{script_path.stem}.{format_to_use}")
            else:
                output_path = f"{script_path.stem}.{format_to_use}"

        # Save output
        if format_to_use == "gv":
            # Save DOT source directly
            with open(output_path, "w") as f:
                f.write(dot_graph.source)
            print(f"✓ DOT source saved to: {output_path}")
        else:
            # Render to image format
            dot_graph.render(
                output_path.rsplit(".", 1)[0], format=format_to_use, cleanup=True
            )
            print(f"✓ Visualization saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"Error during visualization: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        sys.argv = old_argv
        _restore_model_tracking()


def main():
    """Main CLI entry point."""
    # Check if first argument is a subcommand
    if len(sys.argv) > 1 and sys.argv[1] in ["trace", "view"]:
        # Use subcommand mode
        parser = argparse.ArgumentParser(
            prog="vode",
            description="VODE - Visualization of Deep Execution",
        )
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        # Trace command
        trace_parser = subparsers.add_parser(
            "trace", help="Capture model structure and save to file"
        )
        trace_parser.add_argument("script", help="Python script to trace")
        trace_parser.add_argument(
            "script_args", nargs="*", help="Arguments to pass to script"
        )
        trace_parser.add_argument(
            "--mode",
            "-m",
            choices=["static", "dynamic"],
            default="static",
            help="Capture mode (default: static)",
        )
        trace_parser.add_argument("--output", "-o", help="Output file path")
        trace_parser.add_argument()
        trace_parser.add_argument(
            "--depth",
            type=int,
            default=1,
        )
        trace_parser.add_argument(
            "--format",
            "-f",
            choices=["gv", "png", "svg", "pdf"],
            default="gv",
            help="Output format (default: gv)",
        )

        # View command
        view_parser = subparsers.add_parser("view", help="Visualize a saved trace")
        view_parser.add_argument("graph_file", help="Trace file to visualize")
        view_parser.add_argument(
            "--format",
            "-f",
            choices=["svg", "png", "pdf", "gv"],
            default="svg",
            help="Output format (default: svg)",
        )
        view_parser.add_argument("--output", "-o", help="Output file path")
        view_parser.add_argument(
            "--depth",
            "-d",
            type=int,
            default=1,
            help="Maximum depth to visualize (default: 1)",
        )
        view_parser.add_argument(
            "--collapse-loops",
            action="store_true",
            default=True,
            help="Collapse loop patterns (default: True)",
        )
        view_parser.add_argument(
            "--no-collapse-loops",
            dest="collapse_loops",
            action="store_false",
            help="Do not collapse loop patterns",
        )
        view_parser.add_argument()

        args = parser.parse_args()

        # Execute command
        if args.command == "trace":
            return cmd_trace(args)
        elif args.command == "view":
            return cmd_view(args)
        else:
            parser.print_help()
            return 0
    else:
        # Default mode: direct visualization
        parser = argparse.ArgumentParser(
            prog="vode",
            description="VODE - Visualization of Deep Execution",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Visualize a model (default: static mode, SVG output)
  vode script.py
  
  # Trace and save to file
  vode trace --output model_trace.json script.py
  
  # View a saved trace
  vode view model_trace.json
  
  # Pass arguments to your script
  vode script.py --layers 5 --hidden 128
            """,
        )
        parser.add_argument("script", help="Python script to visualize")
        parser.add_argument(
            "script_args", nargs="*", help="Arguments to pass to script"
        )
        parser.add_argument(
            "--mode",
            "-m",
            choices=["static", "dynamic"],
            default="static",
            help="Capture mode (default: static)",
        )
        parser.add_argument(
            "--format",
            "-f",
            choices=["svg", "png", "pdf", "gv"],
            default="svg",
            help="Output format (default: svg)",
        )
        parser.add_argument("--output", "-o", help="Output file path")
        parser.add_argument(
            "--depth",
            "-d",
            type=int,
            default=1,
            help="Maximum depth to visualize (default: 1)",
        )
        parser.add_argument("--model-name", help="Variable name of model to visualize")
        parser.add_argument(
            "--collapse-loops",
            action="store_true",
            default=True,
            help="Collapse loop patterns (default: True)",
        )
        parser.add_argument(
            "--no-collapse-loops",
            dest="collapse_loops",
            action="store_false",
            help="Do not collapse loop patterns",
        )
        parser.add_argument()

        args = parser.parse_args()
        return cmd_visualize(args)


if __name__ == "__main__":
    sys.exit(main())
