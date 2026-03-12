"""VODE Core - Core data structures for computation graph visualization.

This module provides the foundational data structures for VODE's recursive
descent modeling approach where everything is represented as:
    input -> operation -> output
"""

__version__ = "0.1.0"

from .nodes import (
    ExecutionNode,
    TensorInfo,
    OperationInfo,
)
from .utils import (
    generate_node_id,
    format_shape,
    format_dtype,
    format_device,
    is_tensor_like,
    get_tensor_info,
    compute_tensor_stats,
    sanitize_name,
    get_module_info,
    truncate_string,
)

__all__ = [
    # Version
    "__version__",
    # Node classes
    "ExecutionNode",
    "TensorInfo",
    "OperationInfo",
    # Utilities
    "generate_node_id",
    "format_shape",
    "format_dtype",
    "format_device",
    "is_tensor_like",
    "get_tensor_info",
    "compute_tensor_stats",
    "sanitize_name",
    "get_module_info",
    "truncate_string",
]
