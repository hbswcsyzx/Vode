"""Type definitions for VODE core.

Defines enums and type aliases used throughout the VODE core module.
"""

from enum import Enum
from typing import Literal


# Node type literals
NodeType = Literal["Node", "TensorNode", "ModuleNode", "FunctionNode", "LoopNode", "ExecutionNode"]

# Edge type literals
EdgeType = Literal["dataflow", "control", "hierarchy"]

# Loop type literals
LoopType = Literal["for", "while", "recursive", "sequential", "modulelist"]

# Capture mode literals
CaptureMode = Literal["static", "dynamic", "function", "computation"]

# Output format literals
OutputFormat = Literal["svg", "png", "pdf", "gv"]

# Graph direction literals
GraphDirection = Literal["LR", "TB", "BT", "RL"]


class NodeTypeEnum(Enum):
    """Enumeration of node types."""
    NODE = "Node"
    TENSOR = "TensorNode"
    MODULE = "ModuleNode"
    FUNCTION = "FunctionNode"
    LOOP = "LoopNode"
    EXECUTION = "ExecutionNode"


class EdgeTypeEnum(Enum):
    """Enumeration of edge types."""
    DATAFLOW = "dataflow"  # Data flowing between operations
    CONTROL = "control"    # Control flow (e.g., conditionals, loops)
    HIERARCHY = "hierarchy"  # Parent-child structural relationships


class LoopTypeEnum(Enum):
    """Enumeration of loop types."""
    FOR = "for"
    WHILE = "while"
    RECURSIVE = "recursive"
    SEQUENTIAL = "sequential"  # nn.Sequential
    MODULELIST = "modulelist"  # nn.ModuleList


class CaptureModeEnum(Enum):
    """Enumeration of capture modes."""
    STATIC = "static"  # Static structure capture (no forward pass)
    DYNAMIC = "dynamic"  # Dynamic runtime capture (with forward pass)
    FUNCTION = "function"  # Function flow capture (sys.settrace)
    COMPUTATION = "computation"  # Computation flow capture (PyTorch hooks)
