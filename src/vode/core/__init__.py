"""VODE Core - Core data structures for computation graph visualization.

This module provides the foundational data structures for VODE's recursive
descent modeling approach where everything is represented as:
    input -> operation -> output
"""

__version__ = "0.1.0"

# Node classes
from .nodes import (
    Node,
    TensorNode,
    ModuleNode,
    FunctionNode,
    LoopNode,
    ExecutionNode,
    TensorInfo,
    OperationInfo,
)

# Graph container
from .graph import ComputationGraph

# Serialization
from .serializer import (
    serialize_graph,
    deserialize_graph,
    save_graph,
    load_graph,
    serialize_execution_node,
    deserialize_execution_node,
    save_execution_node,
    load_execution_node,
)

# Validation
from .validator import (
    ValidationError,
    validate_graph,
    validate_execution_node,
    validate_json_data,
)

# Type definitions
from .types import (
    NodeType,
    EdgeType,
    LoopType,
    CaptureMode,
    OutputFormat,
    GraphDirection,
    NodeTypeEnum,
    EdgeTypeEnum,
    LoopTypeEnum,
    CaptureModeEnum,
)

# Utilities
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
    "Node",
    "TensorNode",
    "ModuleNode",
    "FunctionNode",
    "LoopNode",
    "ExecutionNode",
    "TensorInfo",
    "OperationInfo",
    # Graph
    "ComputationGraph",
    # Serialization
    "serialize_graph",
    "deserialize_graph",
    "save_graph",
    "load_graph",
    "serialize_execution_node",
    "deserialize_execution_node",
    "save_execution_node",
    "load_execution_node",
    # Validation
    "ValidationError",
    "validate_graph",
    "validate_execution_node",
    "validate_json_data",
    # Types
    "NodeType",
    "EdgeType",
    "LoopType",
    "CaptureMode",
    "OutputFormat",
    "GraphDirection",
    "NodeTypeEnum",
    "EdgeTypeEnum",
    "LoopTypeEnum",
    "CaptureModeEnum",
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

