"""DataflowCapture context manager for tracking tensor operations during forward pass.

This module provides a context manager that wraps nn.Module.__call__ to capture
the dataflow graph during model execution.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Iterable, Mapping

import torch
from torch import nn

from vode.nn.graph.builder import DataflowGraph
from vode.nn.graph.nodes import TensorNode, ModuleNode
from vode.nn.capture.recorder_tensor import RecorderTensor


# Store original nn.Module.__call__ for restoration
_ORIG_MODULE_CALL = nn.Module.__call__


class DataflowCapture:
    """Context manager for capturing dataflow during model forward pass.

    This class wraps nn.Module.__call__ to intercept module executions and
    build a dataflow graph showing how tensors flow through the network.

    Attributes:
        model: The nn.Module to capture dataflow from
        graph: DataflowGraph storing captured nodes and edges
        current_depth: Current depth in the module hierarchy

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from vode.nn.capture import DataflowCapture
        >>>
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        >>> input_data = torch.randn(1, 10)
        >>>
        >>> with DataflowCapture(model) as capture:
        ...     output = model(input_data)
        ...     graph = capture.get_graph()
        >>>
        >>> print(f"Captured {len(graph.get_nodes())} nodes")
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize DataflowCapture for a model.

        Args:
            model: The nn.Module to capture dataflow from
        """
        self.model = model
        self.graph = DataflowGraph()
        self.current_depth = 0
        self._orig_call = _ORIG_MODULE_CALL

    def __enter__(self) -> DataflowCapture:
        """Enter context: replace nn.Module.__call__ with wrapper.

        Returns:
            Self for context manager protocol
        """
        # Replace nn.Module.__call__ with our wrapper
        nn.Module.__call__ = self._create_module_wrapper()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        """Exit context: restore original nn.Module.__call__.

        Args:
            exc_type: Exception type if raised
            exc_value: Exception value if raised
            exc_traceback: Exception traceback if raised
        """
        # Restore original nn.Module.__call__
        nn.Module.__call__ = self._orig_call

    def get_graph(self) -> DataflowGraph:
        """Get the captured dataflow graph.

        Returns:
            DataflowGraph containing captured nodes and edges
        """
        return self.graph

    def _create_module_wrapper(self) -> Any:
        """Create wrapper function for nn.Module.__call__.

        Returns:
            Wrapper function that intercepts module calls
        """

        def module_call_wrapper(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
            """Wrapper for nn.Module.__call__ that captures dataflow.

            Args:
                module: The module being called
                *args: Positional arguments to forward
                **kwargs: Keyword arguments to forward

            Returns:
                Output from module forward pass
            """
            # Collect input RecorderTensors and their nodes
            input_nodes = _collect_tensor_nodes([args, kwargs])
            input_tensors = _collect_recorder_tensors([args, kwargs])

            # If no RecorderTensor inputs, just call original
            if not input_nodes:
                return self._orig_call(module, *args, **kwargs)

            # Determine depth from input nodes
            cur_depth = input_nodes[0].depth if input_nodes else self.current_depth

            # Create ModuleNode for this module
            module_type = type(module).__name__
            module_node_id = f"module_{module_type}_{id(module)}"

            module_node = ModuleNode(
                node_id=module_node_id,
                name=module_type,
                depth=cur_depth,
                module_type=module_type,
            )

            # Extract input shapes
            input_shapes = _collect_shapes([args, kwargs])

            # Add module node to graph
            try:
                self.graph.add_node(module_node)
            except ValueError:
                # Node already exists, skip
                pass

            # Connect input tensor nodes to module node
            for input_node in input_nodes:
                try:
                    self.graph.add_edge(
                        src_id=input_node.node_id, dst_id=module_node_id, label="input"
                    )
                except ValueError:
                    # Edge or node doesn't exist, skip
                    pass

            # Update depth for nested calls
            prev_depth = self.current_depth
            self.current_depth = cur_depth + 1

            # Call original forward
            output = self._orig_call(module, *args, **kwargs)

            # Restore depth
            self.current_depth = prev_depth

            # Extract output shapes
            output_shapes = _collect_shapes(output)

            # Set module info
            module_node.set_module_info(
                module_type=module_type,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
            )

            # Wrap output tensors as RecorderTensors
            output = _wrap_as_recorder_tensors(
                output, module_node, cur_depth + 1, self.graph
            )

            return output

        return module_call_wrapper


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


def _collect_recorder_tensors(data: Any) -> list[RecorderTensor]:
    """Recursively collect RecorderTensors from nested data.

    Args:
        data: Data structure potentially containing RecorderTensors

    Returns:
        List of RecorderTensor objects found
    """
    tensors: list[RecorderTensor] = []

    if isinstance(data, RecorderTensor):
        tensors.append(data)
    elif isinstance(data, Mapping):
        for value in data.values():
            tensors.extend(_collect_recorder_tensors(value))
    elif isinstance(data, Iterable) and not isinstance(data, (str, torch.Tensor)):
        for item in data:
            tensors.extend(_collect_recorder_tensors(item))

    return tensors


def _collect_shapes(data: Any) -> list[tuple[int, ...]]:
    """Recursively collect tensor shapes from nested data.

    Args:
        data: Data structure potentially containing tensors

    Returns:
        List of tensor shapes as tuples
    """
    shapes: list[tuple[int, ...]] = []

    if isinstance(data, torch.Tensor):
        # Use torch.Tensor.size to avoid triggering __torch_function__
        shapes.append(tuple(torch.Tensor.size(data)))
    elif isinstance(data, Mapping):
        for value in data.values():
            shapes.extend(_collect_shapes(value))
    elif isinstance(data, Iterable) and not isinstance(data, (str, torch.Tensor)):
        for item in data:
            shapes.extend(_collect_shapes(item))

    return shapes


def _wrap_as_recorder_tensors(
    data: Any, parent_module_node: ModuleNode, depth: int, graph: DataflowGraph
) -> Any:
    """Recursively wrap output tensors as RecorderTensors.

    Creates TensorNode objects for output tensors and adds them to the graph,
    connecting them to the parent module node.

    Args:
        data: Output data from module forward
        parent_module_node: ModuleNode that produced this output
        depth: Current depth in the computation graph
        graph: DataflowGraph to add nodes to

    Returns:
        Data with tensors wrapped as RecorderTensors
    """
    if isinstance(data, torch.Tensor) and not isinstance(data, RecorderTensor):
        # Convert to RecorderTensor
        rec_tensor = data.as_subclass(RecorderTensor)

        # Create TensorNode
        tensor_node_id = f"tensor_{id(rec_tensor)}"
        tensor_node = TensorNode(
            node_id=tensor_node_id,
            name="output_tensor",
            depth=depth,
            tensor_id=str(id(rec_tensor)),
            shape=tuple(torch.Tensor.size(rec_tensor)),
            dtype=str(torch.Tensor.dtype.__get__(rec_tensor)).replace("torch.", ""),
            device=str(torch.Tensor.device.__get__(rec_tensor)),
        )

        # Add to graph
        try:
            graph.add_node(tensor_node)
            graph.add_edge(
                src_id=parent_module_node.node_id, dst_id=tensor_node_id, label="output"
            )
        except ValueError:
            # Node or edge already exists
            pass

        # Attach tensor node
        rec_tensor.tensor_nodes = [tensor_node]
        return rec_tensor

    elif isinstance(data, RecorderTensor):
        # Already a RecorderTensor, create new node
        tensor_node_id = f"tensor_{id(data)}_out"
        tensor_node = TensorNode(
            node_id=tensor_node_id,
            name="output_tensor",
            depth=depth,
            tensor_id=str(id(data)),
            shape=tuple(torch.Tensor.size(data)),
            dtype=str(torch.Tensor.dtype.__get__(data)).replace("torch.", ""),
            device=str(torch.Tensor.device.__get__(data)),
        )

        # Add to graph
        try:
            graph.add_node(tensor_node)
            graph.add_edge(
                src_id=parent_module_node.node_id, dst_id=tensor_node_id, label="output"
            )
        except ValueError:
            # Node or edge already exists
            pass

        # Append tensor node
        if hasattr(data, "tensor_nodes"):
            data.tensor_nodes.append(tensor_node)
        else:
            data.tensor_nodes = [tensor_node]
        return data

    elif isinstance(data, Mapping):
        return {
            key: _wrap_as_recorder_tensors(value, parent_module_node, depth, graph)
            for key, value in data.items()
        }
    elif isinstance(data, tuple):
        return tuple(
            _wrap_as_recorder_tensors(item, parent_module_node, depth, graph)
            for item in data
        )
    elif isinstance(data, list):
        return [
            _wrap_as_recorder_tensors(item, parent_module_node, depth, graph)
            for item in data
        ]
    else:
        return data
