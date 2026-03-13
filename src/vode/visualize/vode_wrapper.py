"""Convenience wrapper for VODE - all-in-one capture and visualization.

Provides a simple API: vode(model, *args, mode='static', output='model.svg')
"""

from typing import Any, Literal

import torch.nn as nn

from vode.capture import capture_static_execution_graph, capture_dynamic_execution_graph
from vode.visualize.graphviz_renderer import render_execution_graph


def vode(
    model: nn.Module,
    *args: Any,
    mode: Literal["static", "dynamic"] = "static",
    output: str = "model.svg",
    max_depth: int = 1,
    rankdir: Literal["LR", "TB"] = "LR",
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
        max_depth: Maximum depth to render (default: 1)
        rankdir: Graph direction ('LR' for left-right, 'TB' for top-bottom)
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

    # Infer format from output path
    format = "svg"
    if output.endswith(".png"):
        format = "png"
    elif output.endswith(".pdf"):
        format = "pdf"
    elif output.endswith(".gv"):
        format = "gv"

    # Capture execution graph
    if mode == "static":
        root = capture_static_execution_graph(model)
    else:
        # For dynamic mode, pass the first arg as input_data
        root = capture_dynamic_execution_graph(model, args[0] if args else None)

    # Render to graphviz
    dot = render_execution_graph(root, max_depth=max_depth, rankdir=rankdir)

    # Save output
    if format == "gv":
        # Save DOT source directly
        with open(output, "w") as f:
            f.write(dot.source)
        return output
    else:
        # Render to image format
        output_base = output.rsplit(".", 1)[0] if "." in output else output
        rendered_path = dot.render(output_base, format=format, cleanup=True)
        return rendered_path
