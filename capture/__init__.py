"""Capture mechanisms for PyTorch models.

This module provides static and dynamic capture mechanisms for tracing
model execution and building computation graphs.
"""

from .static_capture import capture_static, StaticCapture
from .dynamic_capture import capture_dynamic, DynamicCapture

__all__ = [
    "capture_static",
    "StaticCapture",
    "capture_dynamic",
    "DynamicCapture",
]
