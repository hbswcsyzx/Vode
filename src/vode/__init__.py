"""Vode - Function-level execution tracer for Python/PyTorch.

This package provides tools for tracing and visualizing Python execution.
"""

__version__ = "0.1.0"

# Re-export main components for convenience
from vode.trace import (
    TraceRuntime,
    TraceConfig,
    TraceGraph,
    GraphSerializer,
    TextRenderer,
)

__all__ = [
    "TraceRuntime",
    "TraceConfig",
    "TraceGraph",
    "GraphSerializer",
    "TextRenderer",
]
