"""VODE - Visualization of Deep Execution.

A Python code execution visualization tool with specialized support for PyTorch models.
Uses a recursive descent modeling approach: input -> operation -> output.

Main Features:
- Computation flow capture for PyTorch models (static and dynamic)
- Graphviz-based static export (PNG, SVG, PDF)
- Interactive viewing (future)
- Visual programming interface (future)
"""

__version__ = "0.1.0"

# Core data structures
from .core import (
    # Nodes
    Node,
    TensorNode,
    ModuleNode,
    FunctionNode,
    LoopNode,
    ExecutionNode,
    TensorInfo,
    OperationInfo,
    # Serialization
    serialize_execution_node,
    deserialize_execution_node,
    save_execution_node,
    load_execution_node,
    # Validation
    ValidationError,
    validate_execution_node,
    # Types
    NodeType,
    EdgeType,
    LoopType,
    CaptureMode,
    OutputFormat,
    # Utilities
    generate_node_id,
    sanitize_name,
)

# Capture mechanisms
from .capture import (
    # Base
    BaseTracer,
    # Computation flow - static
    capture_static_execution_graph,
    # Computation flow - dynamic
    DynamicExecutionCapture,
    capture_dynamic_execution_graph,
)

# Visualization
from .visualize import (
    GraphvizRenderer,
    vode,
)

__all__ = [
    # Version
    "__version__",
    # Core - Nodes
    "Node",
    "TensorNode",
    "ModuleNode",
    "FunctionNode",
    "LoopNode",
    "ExecutionNode",
    "TensorInfo",
    "OperationInfo",
    # Core - Serialization
    "serialize_execution_node",
    "deserialize_execution_node",
    "save_execution_node",
    "load_execution_node",
    # Core - Validation
    "ValidationError",
    "validate_execution_node",
    # Core - Types
    "NodeType",
    "EdgeType",
    "LoopType",
    "CaptureMode",
    "OutputFormat",
    # Core - Utilities
    "generate_node_id",
    "sanitize_name",
    # Capture - Base
    "BaseTracer",
    # Capture - Computation flow (static)
    "capture_static_execution_graph",
    # Capture - Computation flow (dynamic)
    "DynamicExecutionCapture",
    "capture_dynamic_execution_graph",
    # Visualization
    "GraphvizRenderer",
    "vode",
]
