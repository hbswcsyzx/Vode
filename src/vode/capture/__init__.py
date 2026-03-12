"""Capture mechanisms for PyTorch models and Python functions.

This module provides dual-mode capture:
1. Function Flow: Captures Python function call flow using sys.settrace()
2. Computation Flow: Captures PyTorch module execution using hooks

Both static (structure-only) and dynamic (runtime) capture are supported.
"""

# Base tracer
from .base import BaseTracer

# Function flow capture (sys.settrace)
from .function_tracer import FunctionTracer, capture_function_flow

# Computation flow capture (PyTorch hooks)
from .computation_tracer import (
    StaticCapture,
    capture_static,
    capture_static_execution_graph,
)

# Import dynamic capture from dynamic_capture.py (keeping backward compatibility)
from .dynamic_capture import (
    DynamicCapture,
    capture_dynamic,
    DynamicExecutionCapture,
    capture_dynamic_execution_graph,
)

# Utilities
from .hooks import register_forward_hooks, remove_hooks
from .recorder_tensor import RecorderTensor

__all__ = [
    # Base
    "BaseTracer",
    # Function flow
    "FunctionTracer",
    "capture_function_flow",
    # Computation flow - static
    "StaticCapture",
    "capture_static",
    "capture_static_execution_graph",
    # Computation flow - dynamic
    "DynamicCapture",
    "capture_dynamic",
    "DynamicExecutionCapture",
    "capture_dynamic_execution_graph",
    # Utilities
    "register_forward_hooks",
    "remove_hooks",
    "RecorderTensor",
]
