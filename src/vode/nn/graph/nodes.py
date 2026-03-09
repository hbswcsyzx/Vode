"""Node classes for VODE neural network visualization.

This module defines the node hierarchy for representing neural network components:
- Node: Base class with parent-child relationships
- TensorNode: Represents tensors (input/output/intermediate)
- ModuleNode: Represents nn.Module instances
- FunctionNode: Represents torch functions (relu, cat, etc.)
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """Base node class for graph representation.

    All nodes support parent-child relationships and depth tracking
    for hierarchical visualization.

    Attributes:
        node_id: Unique identifier for this node
        name: Human-readable name
        depth: Depth in the hierarchy (0 = root)
        parents: List of parent node IDs
        children: List of child node IDs
    """

    node_id: str
    name: str
    depth: int = 0
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)

    def add_parent(self, parent_id: str) -> None:
        """Add a parent node relationship.

        Args:
            parent_id: ID of the parent node
        """
        if parent_id not in self.parents:
            self.parents.append(parent_id)

    def add_child(self, child_id: str) -> None:
        """Add a child node relationship.

        Args:
            child_id: ID of the child node
        """
        if child_id not in self.children:
            self.children.append(child_id)

    def remove_parent(self, parent_id: str) -> None:
        """Remove a parent node relationship.

        Args:
            parent_id: ID of the parent node to remove
        """
        if parent_id in self.parents:
            self.parents.remove(parent_id)

    def remove_child(self, child_id: str) -> None:
        """Remove a child node relationship.

        Args:
            child_id: ID of the child node to remove
        """
        if child_id in self.children:
            self.children.remove(child_id)


@dataclass
class TensorNode(Node):
    """Node representing a tensor in the computation graph.

    Stores tensor metadata including shape, dtype, device, and unique tensor ID.
    Used for input tensors, output tensors, and intermediate tensors.

    Attributes:
        tensor_id: Unique identifier for the tensor object
        shape: Tensor shape as tuple (e.g., (1, 3, 224, 224))
        dtype: Data type (e.g., 'float32', 'int64')
        device: Device location (e.g., 'cpu', 'cuda:0')
        stats: Optional tensor statistics (min, max, mean, std)
    """

    tensor_id: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    device: str | None = None
    stats: dict[str, Any] | None = None

    def set_metadata(
        self,
        tensor_id: str,
        shape: tuple[int, ...],
        dtype: str,
        device: str,
    ) -> None:
        """Set tensor metadata.

        Args:
            tensor_id: Unique identifier for the tensor
            shape: Tensor shape
            dtype: Data type
            device: Device location
        """
        self.tensor_id = tensor_id
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def set_stats(
        self,
        min_val: float,
        max_val: float,
        mean_val: float,
        std_val: float,
    ) -> None:
        """Set tensor statistics.

        Args:
            min_val: Minimum value
            max_val: Maximum value
            mean_val: Mean value
            std_val: Standard deviation
        """
        self.stats = {
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "std": std_val,
        }


@dataclass
class ModuleNode(Node):
    """Node representing an nn.Module in the computation graph.

    Stores module information including type, input/output shapes,
    and parameters.

    Attributes:
        module_type: Type of the module (e.g., 'Conv2d', 'Linear')
        input_shapes: List of input tensor shapes
        output_shapes: List of output tensor shapes
        params: Optional parameter information
    """

    module_type: str | None = None
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)
    params: dict[str, Any] | None = None

    def set_module_info(
        self,
        module_type: str,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
    ) -> None:
        """Set module information.

        Args:
            module_type: Type of the module
            input_shapes: List of input tensor shapes
            output_shapes: List of output tensor shapes
        """
        self.module_type = module_type
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes

    def set_params(self, params: dict[str, Any]) -> None:
        """Set module parameters.

        Args:
            params: Dictionary of parameter information
        """
        self.params = params


@dataclass
class FunctionNode(Node):
    """Node representing a torch function in the computation graph.

    Stores function information including name, input/output shapes,
    and operation metadata.

    Attributes:
        func_name: Name of the function (e.g., 'relu', 'cat', 'add')
        input_shapes: List of input tensor shapes
        output_shapes: List of output tensor shapes
        metadata: Optional operation metadata
    """

    func_name: str | None = None
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)
    metadata: dict[str, Any] | None = None

    def set_function_info(
        self,
        func_name: str,
        input_shapes: list[tuple[int, ...]],
        output_shapes: list[tuple[int, ...]],
    ) -> None:
        """Set function information.

        Args:
            func_name: Name of the function
            input_shapes: List of input tensor shapes
            output_shapes: List of output tensor shapes
        """
        self.func_name = func_name
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Set operation metadata.

        Args:
            metadata: Dictionary of operation metadata
        """
        self.metadata = metadata
