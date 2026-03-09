"""RecorderTensor subclass for tracking tensor operations during forward pass.

This module provides a torch.Tensor subclass that intercepts operations to build
a dataflow graph during model execution.
"""

from __future__ import annotations

from typing import Any, Callable
from collections.abc import Iterable, Mapping

import torch

from vode.nn.graph.nodes import TensorNode, FunctionNode


class RecorderTensor(torch.Tensor):
    """Subclass of torch.Tensor that records operations for dataflow capture.

    This class intercepts torch operations via __torch_function__ to track
    tensor dataflow during forward propagation. Each RecorderTensor maintains
    references to its associated TensorNode objects in the dataflow graph.

    Attributes:
        tensor_nodes: List of TensorNode objects tracking this tensor's role
                     in the computation graph

    Example:
        >>> import torch
        >>> from vode.nn.capture.recorder_tensor import RecorderTensor
        >>> from vode.nn.graph.nodes import TensorNode
        >>>
        >>> # Create a tensor node
        >>> node = TensorNode(node_id="t1", name="input", depth=0)
        >>>
        >>> # Create recorder tensor
        >>> x = torch.randn(2, 3)
        >>> rec_x = x.as_subclass(RecorderTensor)
        >>> rec_x.tensor_nodes = [node]
    """

    @staticmethod
    def __new__(
        cls: type[RecorderTensor], x: Any, *args: Any, **kwargs: Any
    ) -> RecorderTensor:
        """Create a new RecorderTensor instance.

        Args:
            cls: The RecorderTensor class
            x: Input tensor data
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            New RecorderTensor instance
        """
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(
        self, x: Any, tensor_nodes: TensorNode | list[TensorNode] | None = None
    ) -> None:
        """Initialize RecorderTensor with associated TensorNode(s).

        Args:
            x: Input tensor data
            tensor_nodes: TensorNode or list of TensorNodes to associate
        """
        # Store tensor nodes
        if tensor_nodes is None:
            self.tensor_nodes: list[TensorNode] = []
        elif isinstance(tensor_nodes, TensorNode):
            self.tensor_nodes = [tensor_nodes]
        else:
            self.tensor_nodes = tensor_nodes

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Intercept torch operations to record dataflow.

        This method is called for all torch operations on RecorderTensor objects.
        It creates FunctionNode objects to track operations and wraps outputs
        as RecorderTensors with new TensorNode objects.

        Args:
            func: The torch function being called
            types: Tuple of types involved in the operation
            args: Positional arguments to the function
            kwargs: Keyword arguments to the function

        Returns:
            Result of the torch operation, wrapped as RecorderTensor if applicable
        """
        if kwargs is None:
            kwargs = {}

        # Collect all input TensorNodes from args and kwargs
        input_nodes: list[TensorNode] = _collect_tensor_nodes([args, kwargs])

        # Execute the original torch function
        out = super().__torch_function__(func, types, args, kwargs)

        # If no RecorderTensor inputs or no tensor outputs, return as-is
        if not input_nodes:
            return out
        if not _has_tensor_output(out):
            return out

        # Get context from first input node
        first_node = input_nodes[0]
        cur_depth = first_node.depth

        # Create FunctionNode for this operation
        func_name = getattr(func, "__name__", str(func))
        func_node_id = f"func_{id(func)}_{id(out)}"

        func_node = FunctionNode(
            node_id=func_node_id, name=func_name, depth=cur_depth, func_name=func_name
        )

        # Extract input/output shapes
        input_shapes = _collect_shapes([args, kwargs])
        output_shapes = _collect_shapes(out)

        func_node.set_function_info(
            func_name=func_name, input_shapes=input_shapes, output_shapes=output_shapes
        )

        # Connect input nodes to function node
        for input_node in input_nodes:
            input_node.add_child(func_node_id)
            func_node.add_parent(input_node.node_id)

        # Wrap output tensors as RecorderTensors with new TensorNodes
        _attach_output_nodes(out, func_node, cur_depth)

        return out


def _collect_tensor_nodes(data: Any) -> list[TensorNode]:
    """Recursively collect TensorNodes from RecorderTensors in nested data.

    Args:
        data: Data structure potentially containing RecorderTensors

    Returns:
        List of TensorNode objects found
    """
    nodes: list[TensorNode] = []

    if isinstance(data, RecorderTensor):
        # Check if tensor_nodes attribute exists to avoid infinite recursion
        if hasattr(data, "tensor_nodes"):
            nodes.extend(data.tensor_nodes)
    elif isinstance(data, Mapping):
        for value in data.values():
            nodes.extend(_collect_tensor_nodes(value))
    elif isinstance(data, Iterable) and not isinstance(data, (str, torch.Tensor)):
        for item in data:
            nodes.extend(_collect_tensor_nodes(item))

    return nodes


def _has_tensor_output(data: Any) -> bool:
    """Check if data contains any torch.Tensor objects.

    Args:
        data: Data structure to check

    Returns:
        True if any tensors found, False otherwise
    """
    if isinstance(data, torch.Tensor):
        return True
    elif isinstance(data, Mapping):
        return any(_has_tensor_output(v) for v in data.values())
    elif isinstance(data, Iterable) and not isinstance(data, (str, torch.Tensor)):
        return any(_has_tensor_output(item) for item in data)
    return False


def _collect_shapes(data: Any) -> list[tuple[int, ...]]:
    """Recursively collect tensor shapes from nested data.

    Args:
        data: Data structure potentially containing tensors

    Returns:
        List of tensor shapes as tuples
    """
    shapes: list[tuple[int, ...]] = []

    if isinstance(data, torch.Tensor):
        # Use super().__getattribute__ to avoid triggering __torch_function__
        shapes.append(tuple(torch.Tensor.size(data)))
    elif isinstance(data, Mapping):
        for value in data.values():
            shapes.extend(_collect_shapes(value))
    elif isinstance(data, Iterable) and not isinstance(data, (str, torch.Tensor)):
        for item in data:
            shapes.extend(_collect_shapes(item))

    return shapes


def _attach_output_nodes(data: Any, parent_func_node: FunctionNode, depth: int) -> None:
    """Recursively attach TensorNodes to output tensors.

    Wraps output tensors as RecorderTensors and creates TensorNode objects
    to track them in the dataflow graph.

    Args:
        data: Output data from torch operation
        parent_func_node: FunctionNode that produced this output
        depth: Current depth in the computation graph
    """
    if isinstance(data, torch.Tensor) and not isinstance(data, RecorderTensor):
        # Convert to RecorderTensor and attach node
        rec_tensor = data.as_subclass(RecorderTensor)

        tensor_node_id = f"tensor_{id(rec_tensor)}"
        tensor_node = TensorNode(
            node_id=tensor_node_id,
            name="hidden_tensor",
            depth=depth,
            tensor_id=str(id(rec_tensor)),
            shape=tuple(torch.Tensor.size(rec_tensor)),
            dtype=str(torch.Tensor.dtype.__get__(rec_tensor)),
            device=str(torch.Tensor.device.__get__(rec_tensor)),
        )

        # Connect to parent function node
        tensor_node.add_parent(parent_func_node.node_id)
        parent_func_node.add_child(tensor_node_id)

        rec_tensor.tensor_nodes = [tensor_node]

    elif isinstance(data, RecorderTensor):
        # Already a RecorderTensor, create new node
        tensor_node_id = f"tensor_{id(data)}"
        tensor_node = TensorNode(
            node_id=tensor_node_id,
            name="hidden_tensor",
            depth=depth,
            tensor_id=str(id(data)),
            shape=tuple(torch.Tensor.size(data)),
            dtype=str(torch.Tensor.dtype.__get__(data)),
            device=str(torch.Tensor.device.__get__(data)),
        )

        # Connect to parent function node
        tensor_node.add_parent(parent_func_node.node_id)
        parent_func_node.add_child(tensor_node_id)

        if hasattr(data, "tensor_nodes"):
            data.tensor_nodes.append(tensor_node)
        else:
            data.tensor_nodes = [tensor_node]

    elif isinstance(data, Mapping):
        for value in data.values():
            _attach_output_nodes(value, parent_func_node, depth)
    elif isinstance(data, (list, tuple)) and not isinstance(data, (str, torch.Tensor)):
        for item in data:
            _attach_output_nodes(item, parent_func_node, depth)
