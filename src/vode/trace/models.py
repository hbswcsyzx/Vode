"""Core data models for Vode's function-level tracing system.

This module defines the foundational data structures for capturing and representing
function call traces, data flow, and tensor metadata in Python/PyTorch programs.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

# Type aliases for node and edge kinds
NodeKind = Literal["function_call", "variable", "parameter", "buffer"]
EdgeKind = Literal["call_tree", "dataflow", "owns"]
TensorValuePolicy = Literal["full", "preview", "stats_only", "none"]


@dataclass
class TensorMeta:
    """Metadata for a tensor object.

    Attributes:
        shape: Tensor dimensions, e.g. [3, 224, 224]
        dtype: Data type string, e.g. 'torch.float32'
        device: Device location, e.g. 'cuda:0' or 'cpu'
        requires_grad: Whether gradient tracking is enabled
        numel: Total number of elements in the tensor
    """

    shape: list[int] | None
    dtype: str | None
    device: str | None
    requires_grad: bool | None
    numel: int | None


@dataclass
class TensorStats:
    """Statistical summary of tensor values.

    Attributes:
        min: Minimum value in the tensor
        max: Maximum value in the tensor
        mean: Mean value across all elements
        std: Standard deviation of values
    """

    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None


@dataclass
class ValuePreview:
    """Preview representation of a value.

    Attributes:
        text: Human-readable text representation
        data: Structured data preview (e.g., first N elements)
    """

    text: str | None = None
    data: Any | None = None


@dataclass
class VariableRecord:
    """Record of a variable or parameter at a function boundary.

    Attributes:
        id: Unique identifier for this variable record
        slot_path: Path describing the variable's position, e.g. 'arg.input' or 'return.0'
        display_name: Human-readable name for display purposes
        runtime_object_id: Python runtime object id (id(obj)) for this session
        type_name: Type name of the object, e.g. 'torch.Tensor'
        tensor_meta: Tensor metadata if this is a tensor
        tensor_stats: Statistical summary if this is a tensor
        preview: Value preview data
        producer_call_id: ID of the function call that produced this object
        consumer_call_ids: IDs of function calls that consume this object
    """

    id: str
    slot_path: str
    display_name: str
    runtime_object_id: int | None
    type_name: str
    tensor_meta: TensorMeta | None
    tensor_stats: TensorStats | None
    preview: ValuePreview | None
    producer_call_id: str | None
    consumer_call_ids: list[str] = field(default_factory=list)


@dataclass
class FunctionCallNode:
    """Node representing a single function call event.

    Attributes:
        id: Unique identifier for this call, e.g. 'call:42'
        parent_id: ID of the parent call in the call tree
        qualified_name: Fully qualified function name, e.g. 'module.Class.method'
        display_name: Shortened name for display purposes
        filename: Source file where the function is defined
        lineno: Line number where the call occurred
        depth: Call stack depth
        arg_variable_ids: IDs of input argument variables
        return_variable_ids: IDs of return value variables
        metadata: Additional metadata (e.g., module_path, module_class for torch.nn)
    """

    id: str
    parent_id: str | None
    qualified_name: str
    display_name: str
    filename: str
    lineno: int
    depth: int
    arg_variable_ids: list[str] = field(default_factory=list)
    return_variable_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Edge connecting nodes in the trace graph.

    Attributes:
        id: Unique identifier for this edge
        src_id: Source node ID
        dst_id: Destination node ID
        kind: Type of relationship this edge represents
    """

    id: str
    src_id: str
    dst_id: str
    kind: EdgeKind


@dataclass
class TraceGraph:
    """Complete trace graph structure.

    Attributes:
        root_call_ids: IDs of top-level function calls (entry points)
        function_calls: All function call nodes in the trace
        variables: All variable records captured at function boundaries
        edges: All edges representing call tree and dataflow relationships
    """

    root_call_ids: list[str]
    function_calls: list[FunctionCallNode]
    variables: list[VariableRecord]
    edges: list[GraphEdge]
