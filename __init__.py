"""VODE - Visualization of Deep Execution.

A Python code execution visualization tool with specialized support for PyTorch models.
Uses a recursive descent modeling approach: input -> operation -> output.
"""

__version__ = "0.1.0"

from .core import (
    Node,
    TensorNode,
    ModuleNode,
    FunctionNode,
    LoopNode,
    ComputationGraph,
)
from .capture import capture_static, capture_dynamic
from .visualize import (
    GraphvizRenderer,
    visualize,
    visualize_static,
    visualize_dynamic,
    vode,
)

__all__ = [
    "__version__",
    "Node",
    "TensorNode",
    "ModuleNode",
    "FunctionNode",
    "LoopNode",
    "ComputationGraph",
    "capture_static",
    "capture_dynamic",
    "GraphvizRenderer",
    "visualize",
    "visualize_static",
    "visualize_dynamic",
    "vode",
]
