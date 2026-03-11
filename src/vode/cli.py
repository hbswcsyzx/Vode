"""Command-line interface for Vode tracing system.

This module provides the main CLI entry point for tracing Python scripts
and viewing saved traces.
"""

import argparse
import sys
import runpy
from pathlib import Path
from typing import Any

from vode.trace.tracer import TraceRuntime, TraceConfig
from vode.trace.serializer import GraphSerializer
from vode.trace.renderer import TextRenderer
from vode.trace.dataflow_resolver import DataflowResolver


# Global registry for tracking created models
_model_registry: list[Any] = []


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
    try:
        import torch.nn as nn

        # This is a simplified restore - in practice we'd save the original
        # For now, just clear the registry
        _model_registry.clear()
    except ImportError:
        pass


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="vode",
        description="Vode: Visualization and execution tracer for Python/PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize models in a script (static mode)
  vode script.py
  
  # Visualize with options
  vode --mode static --format pdf --output model.pdf script.py
  
  # Trace execution
  vode trace script.py --output trace.json
  
  # View saved trace
  vode view trace.json --web
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Visualize command (default, also accessible without subcommand)
    viz_parser = subparsers.add_parser(
        "viz",
        help="Visualize PyTorch models in a script",
        add_help=False,  # We'll handle help at the parent level
    )
    viz_parser.add_argument(
        "script",
        help="Python script to visualize",
    )
    viz_parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the script",
    )
    viz_parser.add_argument(
        "--mode",
        "-m",
        choices=["static", "dynamic"],
        default="static",
        help="Visualization mode: 'static' (structure only) or 'dynamic' (with tensor shapes)",
    )
    viz_parser.add_argument(
        "--format",
        "-f",
        choices=["svg", "png", "pdf", "gv"],
        default="svg",
        help="Output format (default: svg)",
    )
    viz_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: auto-generated from script name)",
    )
    viz_parser.add_argument(
        "--depth",
        "-d",
        type=int,
        help="Maximum depth for visualization (default: None = full depth)",
    )
    viz_parser.add_argument(
        "--collapse-loops",
        action="store_true",
        default=True,
        help="Collapse loop patterns in visualization (default: True)",
    )
    viz_parser.add_argument(
        "--no-collapse-loops",
        dest="collapse_loops",
        action="store_false",
        help="Don't collapse loop patterns",
    )
    viz_parser.add_argument(
        "--model-name",
        help="Variable name of the model to visualize (default: auto-detect)",
    )

    # Trace command
    trace_parser = subparsers.add_parser(
        "trace", help="Trace a Python script and save the execution graph"
    )
    trace_parser.add_argument("script", help="Path to Python script to trace")
    trace_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum call depth to trace (default: unlimited)",
    )
    trace_parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude file patterns (can be repeated)",
    )
    trace_parser.add_argument(
        "--value-policy",
        choices=["none", "stats_only", "preview", "full"],
        default="stats_only",
        help="Tensor value capture policy (default: stats_only)",
    )
    trace_parser.add_argument(
        "--output",
        "-o",
        default="trace.json",
        help="Output file path (default: trace.json)",
    )
    trace_parser.add_argument(
        "--no-torch-hooks",
        action="store_true",
        help="Disable torch.nn.Module hooks",
    )

    # View command
    view_parser = subparsers.add_parser("view", help="View a saved trace file")
    view_parser.add_argument("trace_file", help="Path to trace JSON file")
    view_parser.add_argument(
        "--web",
        action="store_true",
        help="Start web viewer (default: text output)",
    )
    view_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    view_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    view_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    return parser


def trace_command(args: argparse.Namespace) -> int:
    """Execute the trace command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    script_path = Path(args.script)

    # Check if script exists
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}", file=sys.stderr)
        return 1

    # Create trace configuration
    exclude_patterns = [
        r".*site-packages.*",
        r".*dist-packages.*",
        r".*<frozen.*",
    ]
    exclude_patterns.extend(args.exclude)

    config = TraceConfig(
        max_depth=args.max_depth,
        exclude_patterns=exclude_patterns,
        value_policy=args.value_policy,
        torch_module_hooks=not args.no_torch_hooks,
    )

    # Create trace runtime
    tracer = TraceRuntime(config)

    # Prepare script execution environment
    script_path_abs = script_path.resolve()
    original_argv = sys.argv.copy()
    original_path = sys.path.copy()

    graph = None
    try:
        # Set up sys.argv for the target script
        sys.argv = [str(script_path_abs)]

        # Add script directory to path
        script_dir = str(script_path_abs.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # Start tracing
        tracer.start()

        # Execute the script
        try:
            runpy.run_path(str(script_path_abs), run_name="__main__")
        except SystemExit:
            # Allow the script to exit normally
            pass
        except Exception as e:
            print(f"Error executing script: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    finally:
        # Always stop tracing and build graph
        try:
            graph = tracer.stop()
        except Exception as e:
            print(f"Error stopping tracer: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

        # Restore original sys.argv and sys.path
        sys.argv = original_argv
        sys.path = original_path

    # Check if graph was built successfully
    if graph is None:
        print("Error: Failed to build trace graph", file=sys.stderr)
        return 1

    # Serialize to output file
    try:
        serializer = GraphSerializer()
        serializer.serialize(graph, args.output)
        print(f"Trace saved to: {args.output}")
    except Exception as e:
        print(f"Error saving trace: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    # Print summary
    num_calls = len(graph.function_calls)
    num_edges = len(graph.edges)
    dataflow_edges = len([e for e in graph.edges if e.kind == "dataflow"])

    print(f"Summary:")
    print(f"  Function calls: {num_calls}")
    print(f"  Total edges: {num_edges}")
    print(f"  Dataflow edges: {dataflow_edges}")

    return 0


def view_command(args: argparse.Namespace) -> int:
    """Execute the view command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    trace_file = Path(args.trace_file)

    # Check if file exists
    if not trace_file.exists():
        print(f"Error: Trace file not found: {trace_file}", file=sys.stderr)
        return 1

    # Check if web viewer is requested
    if hasattr(args, "web") and args.web:
        # Start web viewer
        try:
            from vode.view.server import ViewServer

            host = getattr(args, "host", "127.0.0.1")
            port = getattr(args, "port", 8000)
            no_browser = getattr(args, "no_browser", False)

            server = ViewServer(trace_file, host, port)
            server.run(open_browser=not no_browser)
            return 0
        except Exception as e:
            print(f"Error starting web viewer: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    # Default: text rendering
    try:
        serializer = GraphSerializer()
        graph = serializer.deserialize(str(trace_file))
    except Exception as e:
        print(f"Error loading trace: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    # Render the graph
    try:
        renderer = TextRenderer()
        output = renderer.render(graph)
        print(output)
    except Exception as e:
        print(f"Error rendering trace: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    return 0


def visualize_command(args: argparse.Namespace) -> int:
    """Execute the visualize command (direct script visualization).

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    script_path = Path(args.script)

    # Check if script exists
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate from script name
        script_stem = script_path.stem
        output_path = f"{script_stem}_{args.mode}.{args.format}"

    # Prepare script execution environment
    script_path_abs = script_path.resolve()
    original_argv = sys.argv.copy()
    original_path = sys.path.copy()

    try:
        # Set up sys.argv for the target script
        sys.argv = [str(script_path_abs)] + args.script_args

        # Add script directory to path
        script_dir = str(script_path_abs.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # Track model creation
        _track_model_creation()

        # Execute the script
        try:
            script_globals = runpy.run_path(str(script_path_abs), run_name="__main__")
        except SystemExit:
            # Allow the script to exit normally
            pass
        except Exception as e:
            print(f"Error executing script: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

        # Detect model
        model = None
        if args.model_name:
            # Use specified model name
            if args.model_name in script_globals:
                model = script_globals[args.model_name]
            else:
                print(
                    f"Error: Model '{args.model_name}' not found in script",
                    file=sys.stderr,
                )
                return 1
        else:
            # Auto-detect: use the last created model
            if _model_registry:
                model = _model_registry[-1]
                print(f"Auto-detected model: {model.__class__.__name__}")
            else:
                print("Error: No PyTorch models detected in script", file=sys.stderr)
                print(
                    "Hint: Use --model-name to specify the model variable name",
                    file=sys.stderr,
                )
                return 1

        # Verify it's a PyTorch model
        try:
            import torch.nn as nn

            if not isinstance(model, nn.Module):
                print(
                    f"Error: Detected object is not a PyTorch model: {type(model)}",
                    file=sys.stderr,
                )
                return 1
        except ImportError:
            print("Error: PyTorch is not installed", file=sys.stderr)
            return 1

        # Visualize the model
        try:
            # Import the vode wrapper
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from vode.visualize import vode

            # Handle dynamic mode
            if args.mode == "dynamic":
                print(
                    "Warning: Dynamic mode requires sample input. Using default input shapes.",
                    file=sys.stderr,
                )
                print(
                    "Hint: Modify your script to call vode() directly for custom inputs.",
                    file=sys.stderr,
                )
                # Try to infer input shape from model
                # For now, skip dynamic mode without inputs
                print(
                    "Error: Dynamic mode not supported in CLI without explicit inputs",
                    file=sys.stderr,
                )
                print(
                    "Use static mode or call vode() directly in your script",
                    file=sys.stderr,
                )
                return 1

            # Static mode
            result_path = vode(
                model,
                mode=args.mode,
                output=output_path,
                max_depth=args.depth,
                format=args.format,
                collapse_loops=args.collapse_loops,
            )

            print(f"Visualization saved to: {result_path}")
            return 0

        except Exception as e:
            print(f"Error visualizing model: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    finally:
        # Restore original sys.argv and sys.path
        sys.argv = original_argv
        sys.path = original_path
        _restore_model_tracking()


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Pre-parse to detect if first arg is a subcommand or a script file
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        # Check if it's a subcommand
        if first_arg in ["trace", "view", "viz"]:
            # Let argparse handle it normally
            parser = create_parser()
            args = parser.parse_args()
        elif first_arg.startswith("-"):
            # It's a flag, show help
            parser = create_parser()
            parser.print_help()
            return 1
        else:
            # Assume it's a script file - inject "viz" subcommand
            sys.argv.insert(1, "viz")
            parser = create_parser()
            args = parser.parse_args()
    else:
        # No arguments, show help
        parser = create_parser()
        parser.print_help()
        return 1

    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return 1

    # Execute the appropriate command
    if args.command == "viz":
        return visualize_command(args)
    elif args.command == "trace":
        return trace_command(args)
    elif args.command == "view":
        return view_command(args)
    else:
        print(f"Error: Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
