"""Core node classes for VODE computation graph.

All nodes follow the recursive descent pattern: input -> operation -> output
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class Node:
    """Base node class supporting recursive descent pattern.

    All nodes represent: input -> operation -> output

    Attributes:
        node_id: Unique identifier for this node
        name: Human-readable name
        depth: Nesting depth in the computation graph
        parents: List of parent node IDs (structural hierarchy)
        children: List of child node IDs (structural hierarchy)
        input_ids: List of input node IDs (dataflow)
        output_ids: List of output node IDs (dataflow)
        metadata: Additional metadata for this node
    """

    node_id: str
    name: str
    depth: int = 0

    # Graph structure (hierarchical)
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)

    # Recursive descent components (dataflow)
    input_ids: list[str] = field(default_factory=list)
    output_ids: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_root(self) -> bool:
        """Check if this node is a root node (no parents)."""
        return len(self.parents) == 0

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (no children)."""
        return len(self.children) == 0

    def add_child(self, child_id: str) -> None:
        """Add a child node ID to this node."""
        if child_id not in self.children:
            self.children.append(child_id)

    def add_parent(self, parent_id: str) -> None:
        """Add a parent node ID to this node."""
        if parent_id not in self.parents:
            self.parents.append(parent_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "depth": self.depth,
            "parents": self.parents,
            "children": self.children,
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
            "metadata": self.metadata,
            "type": self.__class__.__name__,
        }


@dataclass
class TensorNode(Node):
    """Represents a tensor in the computation graph.

    For recursive descent: This is data flowing between operations.

    Attributes:
        tensor_id: Unique identifier for the tensor object (id(tensor))
        shape: Tensor shape tuple
        dtype: Data type string (e.g., 'torch.float32')
        device: Device string (e.g., 'cpu', 'cuda:0')
        requires_grad: Whether tensor requires gradient
        stats: Optional statistics (min, max, mean, std)
        is_aux: Flag for auxiliary/intermediate tensors
        main_node_id: Reference to main tensor if this is auxiliary
    """

    tensor_id: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    device: str | None = None
    requires_grad: bool = False

    # Statistics (for dynamic capture)
    stats: dict[str, float] | None = None

    # Auxiliary flag (for intermediate tensors in modules)
    is_aux: bool = False
    main_node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert tensor node to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "tensor_id": self.tensor_id,
                "shape": self.shape,
                "dtype": self.dtype,
                "device": self.device,
                "requires_grad": self.requires_grad,
                "stats": self.stats,
                "is_aux": self.is_aux,
                "main_node_id": self.main_node_id,
            }
        )
        return base_dict


@dataclass
class ModuleNode(Node):
    """Represents an nn.Module execution.

    For recursive descent:
    - input_ids: Input tensor node IDs
    - operation: Module forward pass
    - output_ids: Output tensor node IDs

    Attributes:
        module_type: Type name of the module (e.g., 'Linear', 'Conv2d')
        module_instance_id: id(module) for tracking instance reuse
        input_shapes: List of input tensor shapes
        output_shapes: List of output tensor shapes
        is_container: Whether this is a leaf module (no submodules)
        is_parameterless: Whether module has no learnable parameters
        params: Optional parameter information
    """

    module_type: str | None = None
    module_instance_id: int | None = None

    # Shape information
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)

    # Module properties
    is_container: bool = False
    is_parameterless: bool = False

    # Parameters info (optional)
    params: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert module node to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "module_type": self.module_type,
                "module_instance_id": self.module_instance_id,
                "input_shapes": self.input_shapes,
                "output_shapes": self.output_shapes,
                "is_container": self.is_container,
                "is_parameterless": self.is_parameterless,
                "params": self.params,
            }
        )
        return base_dict


@dataclass
class FunctionNode(Node):
    """Represents a torch function or Python function call.

    For recursive descent:
    - input_ids: Input tensor/value node IDs
    - operation: Function execution
    - output_ids: Output tensor/value node IDs

    Attributes:
        func_name: Name of the function
        func_id: id(function) for tracking
        input_shapes: List of input tensor shapes
        output_shapes: List of output tensor shapes
        is_inplace: Whether this is an inplace operation
        qualified_name: Full module.function path
    """

    func_name: str | None = None
    func_id: int | None = None

    # Shape information
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)

    # Function metadata
    is_inplace: bool = False
    qualified_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert function node to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "func_name": self.func_name,
                "func_id": self.func_id,
                "input_shapes": self.input_shapes,
                "output_shapes": self.output_shapes,
                "is_inplace": self.is_inplace,
                "qualified_name": self.qualified_name,
            }
        )
        return base_dict


@dataclass
class LoopNode(Node):
    """Represents a detected loop structure.

    For recursive descent:
    - input_ids: Loop input variables
    - operation: Loop body (collapsed or expanded)
    - output_ids: Loop output variables

    Attributes:
        loop_type: Type of loop ('for', 'while', 'recursive')
        iteration_count: Number of iterations (if known)
        body_node_ids: Node IDs in the loop body
        is_collapsed: Whether to show collapsed or expanded view
        recursive_call_id: For recursive calls, the ID of the recursive call
    """

    loop_type: Literal["for", "while", "recursive"] = "for"
    iteration_count: int | None = None

    # Loop body representation
    body_node_ids: list[str] = field(default_factory=list)
    is_collapsed: bool = True

    # For recursive calls
    recursive_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert loop node to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "loop_type": self.loop_type,
                "iteration_count": self.iteration_count,
                "body_node_ids": self.body_node_ids,
                "is_collapsed": self.is_collapsed,
                "recursive_call_id": self.recursive_call_id,
            }
        )
        return base_dict


# ============================================================================
# Stage 4: New Data Structures for Input->Op->Output Modeling
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
    """

    op_type: str
    op_name: str
    params_count: int = 0
    is_composite: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "op_type": self.op_type,
            "op_name": self.op_name,
            "params_count": self.params_count,
            "is_composite": self.is_composite,
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
