"""Core node classes for VODE computation graph.

All nodes follow the recursive descent pattern: input -> operation -> output
"""

from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================================
# New Data Structures for Input->Op->Output Modeling
# ============================================================================


@dataclass
class TensorInfo:
    """Represents tensor metadata in the execution graph.

    This is a lightweight representation of tensor information used in the
    recursive descent model. Unlike TensorNode, this focuses purely on
    tensor metadata without graph connectivity.

    Attributes:
        name: Tensor name or identifier
        shape: Tensor shape tuple (e.g., (1, 10, 20))
        dtype: Data type string (e.g., 'torch.float32')
        device: Device string (e.g., 'cpu', 'cuda:0')
    """

    name: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
        }


@dataclass
class OperationInfo:
    """Represents operation metadata in the execution graph.

    This captures the essential information about an operation (module or function)
    in the computation graph. Operations can be composite (expandable) or atomic.

    Attributes:
        op_type: Operation type (e.g., 'Linear', 'ReLU', 'Sequential')
        op_name: Human-readable operation name
        params_count: Number of learnable parameters (0 for parameterless ops)
        is_composite: Whether this operation can be expanded into sub-operations
        is_loop: Whether this operation represents a loop structure
        loop_type: Type of loop ('sequential', 'modulelist', 'reuse')
        iteration_count: Number of iterations in the loop (if known)
    """

    op_type: str
    op_name: str
    params_count: int = 0
    is_composite: bool = False
    is_loop: bool = False
    loop_type: str | None = None
    iteration_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "op_type": self.op_type,
            "op_name": self.op_name,
            "params_count": self.params_count,
            "is_composite": self.is_composite,
            "is_loop": self.is_loop,
            "loop_type": self.loop_type,
            "iteration_count": self.iteration_count,
        }


@dataclass
class ExecutionNode:
    """Core node representing the (inputs, operation, outputs) triple.

    This is the fundamental unit in the recursive descent model. Each node
    represents a single execution step with:
    - inputs: List of input tensors
    - operation: The operation being performed
    - outputs: List of output tensors

    Nodes can be recursively expanded: composite operations can be expanded
    into a sequence of child nodes, allowing hierarchical visualization at
    different depth levels.

    Attributes:
        node_id: Unique identifier for this node
        name: Display name for the node
        depth: Current depth in the hierarchy (0 = root)
        inputs: List of input tensor information
        operation: The operation being performed
        outputs: List of output tensor information
        children: Child nodes (for recursive expansion)
        is_expandable: Whether this node can be expanded
        is_expanded: Whether this node is currently expanded
        parent: Reference to parent node (None for root)
    """

    node_id: str
    name: str
    depth: int
    inputs: list[TensorInfo]
    operation: OperationInfo
    outputs: list[TensorInfo]
    children: list["ExecutionNode"] = field(default_factory=list)
    is_expandable: bool = False
    is_expanded: bool = False
    parent: Optional["ExecutionNode"] = None

    def can_expand(self) -> bool:
        """Check if this node can be expanded.

        Returns:
            True if the node is expandable and has children, False otherwise
        """
        return self.is_expandable and len(self.children) > 0

    def expand(self) -> None:
        """Mark this node as expanded.

        This indicates that the node's children should be shown in visualization
        instead of the node itself.
        """
        if self.can_expand():
            self.is_expanded = True

    def collapse(self) -> None:
        """Mark this node as collapsed.

        This indicates that the node should be shown as a single operation
        rather than its expanded children.
        """
        self.is_expanded = False

    def add_child(self, child: "ExecutionNode") -> None:
        """Add a child node to this node.

        Args:
            child: The child node to add
        """
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            # If we have children, we're expandable
            if not self.is_expandable:
                self.is_expandable = True

    def get_depth(self) -> int:
        """Get the current depth of this node in the hierarchy.

        Returns:
            The depth value (0 for root nodes)
        """
        return self.depth

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary containing all node information
        """
        return {
            "node_id": self.node_id,
            "name": self.name,
            "depth": self.depth,
            "inputs": [inp.to_dict() for inp in self.inputs],
            "operation": self.operation.to_dict(),
            "outputs": [out.to_dict() for out in self.outputs],
            "children": [child.to_dict() for child in self.children],
            "is_expandable": self.is_expandable,
            "is_expanded": self.is_expanded,
            "parent_id": self.parent.node_id if self.parent else None,
        }
