"""Capture mechanisms for PyTorch models and Python functions.

This module provides dual-mode capture:
1. Function Flow: Captures Python function call flow using sys.settrace()
2. Computation Flow: Captures PyTorch module execution using hooks

Both static (structure-only) and dynamic (runtime) capture are supported.
"""

# Base tracer
from .base import BaseTracer

# Computation flow capture (PyTorch hooks) - ExecutionNode API only
from .computation_tracer import (
    capture_static_execution_graph,
    DynamicExecutionCapture,
    capture_dynamic_execution_graph,
)

# Utilities
from .hooks import register_forward_hooks, remove_hooks
from .recorder_tensor import RecorderTensor

__all__ = [
    # Base
    "BaseTracer",
    # Computation flow - static
    "capture_static_execution_graph",
    # Computation flow - dynamic
    "DynamicExecutionCapture",
    "capture_dynamic_execution_graph",
    # Utilities
    "register_forward_hooks",
    "remove_hooks",
    "RecorderTensor",
]
