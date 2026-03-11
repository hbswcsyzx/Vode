"""Convenience wrapper for VODE - all-in-one capture and visualization.

Provides a simple API: vode(model, *args, mode='static', output='model.svg')
"""

from typing import Any, Literal

import torch.nn as nn

from vode.capture import capture_static, capture_dynamic
from vode.visualize import visualize


def vode(
    model: nn.Module,
    *args: Any,
    mode: Literal["static", "dynamic"] = "static",
    output: str = "model.svg",
    max_depth: int | None = None,
    format: Literal["svg", "png", "pdf", "gv"] = "svg",
    rankdir: Literal["LR", "TB"] = "LR",
    collapse_loops: bool = True,
    compute_stats: bool = False,
    **kwargs: Any,
) -> str:
    """All-in-one function: capture computation graph and visualize it.

    This is the main entry point for VODE. It captures the computation graph
    of a model and renders it to a visual format.

    Args:
        model: PyTorch model to visualize
        *args: Input arguments to pass to the model (required for dynamic mode)
        mode: Capture mode ('static' or 'dynamic')
        output: Output file path (default: 'model.svg')
        max_depth: Maximum depth to render (None for full tree)
        format: Output format ('svg', 'png', 'pdf', 'gv')
        rankdir: Graph direction ('LR' for left-right, 'TB' for top-bottom)
        collapse_loops: Whether to collapse loop nodes
        compute_stats: Whether to compute tensor statistics (dynamic mode only)
        **kwargs: Additional keyword arguments to pass to the model (dynamic mode)

    Returns:
        Path to the generated visualization file

    Raises:
        ValueError: If mode is invalid or if dynamic mode is used without inputs
        ImportError: If graphviz is not installed (for non-gv formats)

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        >>>
        >>> # Static capture (no input needed)
        >>> vode(model, mode='static', output='my_model.svg')
        'my_model.svg'
        >>>
        >>> # Dynamic capture (input required)
        >>> x = torch.randn(1, 10)
        >>> vode(model, x, mode='dynamic', max_depth=3, output='model_limited.png')
        'model_limited.png'
    """
    # Validate mode
    if mode not in ("static", "dynamic"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'static' or 'dynamic'")

    # Validate inputs for dynamic mode
    if mode == "dynamic" and not args:
        raise ValueError(
            "Dynamic mode requires input arguments. Pass model inputs as positional arguments."
        )

    # Infer format from output path if not explicitly set
    if format == "svg" and output != "model.svg":
        # Check if output has a different extension
        if output.endswith(".png"):
            format = "png"
        elif output.endswith(".pdf"):
            format = "pdf"
        elif output.endswith(".gv"):
            format = "gv"

    # Capture computation graph
    if mode == "static":
        graph = capture_static(model)
    else:
        graph = capture_dynamic(model, *args, compute_stats=compute_stats, **kwargs)

    # Visualize
    output_path = visualize(
        graph=graph,
        output_path=output,
        max_depth=max_depth,
        format=format,
        rankdir=rankdir,
        collapse_loops=collapse_loops,
    )

    return output_path
