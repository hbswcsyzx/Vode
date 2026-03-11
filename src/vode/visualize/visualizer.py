"""Main visualization API for VODE.

Provides high-level functions to visualize computation graphs.
"""

import os
from pathlib import Path
from typing import Literal

from vode.core.graph import ComputationGraph
from .graphviz_renderer import GraphvizRenderer


def visualize_static(
    graph: ComputationGraph,
    output_path: str,
    max_depth: int | None = None,
    format: Literal["svg", "png", "pdf", "gv"] = "svg",
    rankdir: Literal["LR", "TB"] = "LR",
    collapse_loops: bool = True,
) -> str:
    """Visualize a static computation graph.

    Args:
        graph: ComputationGraph to visualize
        output_path: Output file path (extension will be added if needed)
        max_depth: Maximum depth to render (None for full tree)
        format: Output format ('svg', 'png', 'pdf', 'gv')
        rankdir: Graph direction ('LR' for left-right, 'TB' for top-bottom)
        collapse_loops: Whether to collapse loop nodes

    Returns:
        Path to the generated file

    Raises:
        ImportError: If graphviz is not installed
        ValueError: If format is invalid
    """
    return _visualize(
        graph=graph,
        output_path=output_path,
        max_depth=max_depth,
        format=format,
        rankdir=rankdir,
        collapse_loops=collapse_loops,
    )


def visualize_dynamic(
    graph: ComputationGraph,
    output_path: str,
    max_depth: int | None = None,
    format: Literal["svg", "png", "pdf", "gv"] = "svg",
    rankdir: Literal["LR", "TB"] = "LR",
    collapse_loops: bool = True,
) -> str:
    """Visualize a dynamic computation graph.

    Args:
        graph: ComputationGraph to visualize
        output_path: Output file path (extension will be added if needed)
        max_depth: Maximum depth to render (None for full tree)
        format: Output format ('svg', 'png', 'pdf', 'gv')
        rankdir: Graph direction ('LR' for left-right, 'TB' for top-bottom)
        collapse_loops: Whether to collapse loop nodes

    Returns:
        Path to the generated file

    Raises:
        ImportError: If graphviz is not installed
        ValueError: If format is invalid
    """
    return _visualize(
        graph=graph,
        output_path=output_path,
        max_depth=max_depth,
        format=format,
        rankdir=rankdir,
        collapse_loops=collapse_loops,
    )


def _visualize(
    graph: ComputationGraph,
    output_path: str,
    max_depth: int | None,
    format: str,
    rankdir: str,
    collapse_loops: bool,
) -> str:
    """Internal visualization function.

    Args:
        graph: ComputationGraph to visualize
        output_path: Output file path
        max_depth: Maximum depth to render
        format: Output format
        rankdir: Graph direction
        collapse_loops: Whether to collapse loops

    Returns:
        Path to the generated file

    Raises:
        ImportError: If graphviz is not installed
        ValueError: If format is invalid
    """
    # Validate format
    valid_formats = {"svg", "png", "pdf", "gv"}
    if format not in valid_formats:
        raise ValueError(f"Invalid format '{format}'. Must be one of {valid_formats}")

    # Render to DOT format
    renderer = GraphvizRenderer(graph)
    dot_source = renderer.render(
        max_depth=max_depth,
        collapse_loops=collapse_loops,
        rankdir=rankdir,
    )

    # Handle output path
    output_path = Path(output_path)

    # If format is 'gv', just save the DOT source
    if format == "gv":
        output_file = output_path.with_suffix(".gv")
        output_file.write_text(dot_source)
        return str(output_file)

    # Otherwise, use graphviz to render
    try:
        from graphviz import Source
    except ImportError as e:
        raise ImportError(
            "graphviz package is required for rendering. "
            "Install it with: pip install graphviz\n"
            "Note: You also need the graphviz system package installed."
        ) from e

    # Create graphviz Source object
    source = Source(dot_source)

    # Render to file
    output_file = output_path.with_suffix("")  # Remove extension
    rendered_path = source.render(
        filename=str(output_file),
        format=format,
        cleanup=True,  # Remove intermediate .gv file
    )

    return rendered_path


def visualize(
    graph: ComputationGraph,
    output_path: str = "model.svg",
    max_depth: int | None = None,
    format: Literal["svg", "png", "pdf", "gv"] = "svg",
    rankdir: Literal["LR", "TB"] = "LR",
    collapse_loops: bool = True,
) -> str:
    """Visualize a computation graph (auto-detects static/dynamic).

    Args:
        graph: ComputationGraph to visualize
        output_path: Output file path (default: 'model.svg')
        max_depth: Maximum depth to render (None for full tree)
        format: Output format ('svg', 'png', 'pdf', 'gv')
        rankdir: Graph direction ('LR' for left-right, 'TB' for top-bottom)
        collapse_loops: Whether to collapse loop nodes

    Returns:
        Path to the generated file

    Raises:
        ImportError: If graphviz is not installed
        ValueError: If format is invalid
    """
    return _visualize(
        graph=graph,
        output_path=output_path,
        max_depth=max_depth,
        format=format,
        rankdir=rankdir,
        collapse_loops=collapse_loops,
    )
