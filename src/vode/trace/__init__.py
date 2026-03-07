"""Trace module for Vode - function-level execution tracing.

This module provides the core tracing functionality for capturing and analyzing
Python/PyTorch execution traces.
"""

from vode.trace.models import (
    EdgeKind,
    FunctionCallNode,
    GraphEdge,
    TensorMeta,
    TensorStats,
    TensorValuePolicy,
    TraceGraph,
    ValuePreview,
    VariableRecord,
)
from vode.trace.tracer import TraceConfig, TraceRuntime
from vode.trace.value_extractor import ValueExtractor
from vode.trace.dataflow_resolver import DataflowResolver
from vode.trace.serializer import GraphSerializer
from vode.trace.renderer import TextRenderer

__all__ = [
    "EdgeKind",
    "FunctionCallNode",
    "GraphEdge",
    "TensorMeta",
    "TensorStats",
    "TensorValuePolicy",
    "TraceGraph",
    "ValuePreview",
    "VariableRecord",
    "TraceConfig",
    "TraceRuntime",
    "ValueExtractor",
    "DataflowResolver",
    "GraphSerializer",
    "TextRenderer",
]
