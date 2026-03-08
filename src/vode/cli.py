"""Command-line interface for Vode tracing system.

This module provides the main CLI entry point for tracing Python scripts
and viewing saved traces.
"""

import argparse
import sys
import runpy
from pathlib import Path

from vode.trace.tracer import TraceRuntime, TraceConfig
from vode.trace.serializer import GraphSerializer
from vode.trace.renderer import TextRenderer
from vode.trace.dataflow_resolver import DataflowResolver


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="vode",
        description="Vode: Function-level execution tracer for Python/PyTorch",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

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
    if hasattr(args, 'web') and args.web:
        # Start web viewer
        try:
            from vode.view.server import ViewServer
            
            host = getattr(args, 'host', '127.0.0.1')
            port = getattr(args, 'port', 8000)
            no_browser = getattr(args, 'no_browser', False)
            
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


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return 1

    # Execute the appropriate command
    if args.command == "trace":
        return trace_command(args)
    elif args.command == "view":
        return view_command(args)
    else:
        print(f"Error: Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
