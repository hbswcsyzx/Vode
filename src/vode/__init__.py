"""VODE - Visualization of Deep Execution.

A Python code execution visualization tool with specialized support for PyTorch models.
Uses a recursive descent modeling approach: input -> operation -> output.

Main Features:
- Dual-mode capture: Function flow (sys.settrace) and Computation flow (PyTorch hooks)
- Static and dynamic capture modes
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
    # Graph
    ComputationGraph,
    # Serialization
    serialize_graph,
    deserialize_graph,
    save_graph,
    load_graph,
    # Validation
    ValidationError,
    validate_graph,
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
    # Function flow
    FunctionTracer,
    capture_function_flow,
    # Computation flow - static
    StaticCapture,
    capture_static,
    capture_static_execution_graph,
    # Computation flow - dynamic
    DynamicCapture,
    capture_dynamic,
    DynamicExecutionCapture,
    capture_dynamic_execution_graph,
)

# Visualization
from .visualize import (
    GraphvizRenderer,
    visualize,
    visualize_static,
    visualize_dynamic,
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
    # Core - Graph
    "ComputationGraph",
    # Core - Serialization
    "serialize_graph",
    "deserialize_graph",
    "save_graph",
    "load_graph",
    # Core - Validation
    "ValidationError",
    "validate_graph",
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
    # Capture - Function flow
    "FunctionTracer",
    "capture_function_flow",
    # Capture - Computation flow (static)
    "StaticCapture",
    "capture_static",
    "capture_static_execution_graph",
    # Capture - Computation flow (dynamic)
    "DynamicCapture",
    "capture_dynamic",
    "DynamicExecutionCapture",
    "capture_dynamic_execution_graph",
    # Visualization
    "GraphvizRenderer",
    "visualize",
    "visualize_static",
    "visualize_dynamic",
    "vode",
]

