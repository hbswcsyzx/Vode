"""Core node classes for VODE computation graph.

All nodes follow the recursive descent pattern: input -> operation -> output
"""

from dataclasses import dataclass, field
from typing import Any, Literal


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
            'node_id': self.node_id,
            'name': self.name,
            'depth': self.depth,
            'parents': self.parents,
            'children': self.children,
            'input_ids': self.input_ids,
            'output_ids': self.output_ids,
            'metadata': self.metadata,
            'type': self.__class__.__name__,
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
        base_dict.update({
            'tensor_id': self.tensor_id,
            'shape': self.shape,
            'dtype': self.dtype,
            'device': self.device,
            'requires_grad': self.requires_grad,
            'stats': self.stats,
            'is_aux': self.is_aux,
            'main_node_id': self.main_node_id,
        })
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
        base_dict.update({
            'module_type': self.module_type,
            'module_instance_id': self.module_instance_id,
            'input_shapes': self.input_shapes,
            'output_shapes': self.output_shapes,
            'is_container': self.is_container,
            'is_parameterless': self.is_parameterless,
            'params': self.params,
        })
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
        base_dict.update({
            'func_name': self.func_name,
            'func_id': self.func_id,
            'input_shapes': self.input_shapes,
            'output_shapes': self.output_shapes,
            'is_inplace': self.is_inplace,
            'qualified_name': self.qualified_name,
        })
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
        base_dict.update({
            'loop_type': self.loop_type,
            'iteration_count': self.iteration_count,
            'body_node_ids': self.body_node_ids,
            'is_collapsed': self.is_collapsed,
            'recursive_call_id': self.recursive_call_id,
        })
        return base_dict
