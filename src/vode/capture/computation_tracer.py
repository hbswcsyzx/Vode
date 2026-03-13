"""Computation flow tracer for PyTorch models.

Captures PyTorch model computation flow using hooks and module inspection.
Supports both static (structure-only) and dynamic (runtime) capture modes.

According to plans/01_FEATURES.md:
- Computation Flow captures PyTorch module execution
- Static mode: Inspects structure without running forward pass
- Dynamic mode: Runs forward pass to capture actual tensor shapes and data flow
"""

from typing import Any
import torch
import torch.nn as nn

from vode.core import (
    ExecutionNode,
    TensorInfo,
    OperationInfo,
    sanitize_name,
)


# Leaf module types (no submodules to traverse)
LEAF_MODULES = {
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.ReLU,
    nn.LeakyReLU,
    nn.GELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
    nn.LogSoftmax,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.Embedding,
    nn.EmbeddingBag,
    nn.LSTM,
    nn.GRU,
    nn.RNN,
    nn.LSTMCell,
    nn.GRUCell,
    nn.RNNCell,
}


def _count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a module.

    Args:
        module: PyTorch module

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _module_to_operation_info(module: nn.Module, name: str = "") -> OperationInfo:
    """Convert PyTorch module to OperationInfo.

    Args:
        module: PyTorch module
        name: Module name (optional)

    Returns:
        OperationInfo with module metadata
    """
    module_type = module.__class__.__name__
    op_name = name if name else module_type
    params_count = _count_parameters(module)

    # Check if module is composite (has children)
    children = list(module.children())
    is_composite = len(children) > 0 and not type(module) in LEAF_MODULES

    # Detect loop structures
    is_loop = False
    loop_type = None
    iteration_count = None

    if isinstance(module, nn.Sequential):
        is_loop = True
        loop_type = "sequential"
        iteration_count = len(children)
    elif isinstance(module, nn.ModuleList):
        is_loop = True
        loop_type = "modulelist"
        iteration_count = len(children)

    return OperationInfo(
        op_type=module_type,
        op_name=op_name,
        params_count=params_count,
        is_composite=is_composite,
        is_loop=is_loop,
        loop_type=loop_type,
        iteration_count=iteration_count,
    )


def _build_execution_node_recursive(
    module: nn.Module, name: str, depth: int, node_id_prefix: str
) -> ExecutionNode:
    """Recursively build ExecutionNode hierarchy from PyTorch module.

    Args:
        module: PyTorch module to convert
        name: Module name
        depth: Current depth in hierarchy
        node_id_prefix: Prefix for generating node IDs

    Returns:
        ExecutionNode with children populated
    """
    # Generate node ID
    node_id = f"{node_id_prefix}_{sanitize_name(name)}" if name else node_id_prefix

    # Create operation info
    operation = _module_to_operation_info(module, name)

    # Create ExecutionNode (static capture has no runtime tensor info)
    node = ExecutionNode(
        node_id=node_id,
        name=name if name else module.__class__.__name__,
        depth=depth,
        inputs=[],  # Static capture doesn't have runtime data
        operation=operation,
        outputs=[],  # Static capture doesn't have runtime data
        children=[],
        is_expandable=operation.is_composite,
        is_expanded=False,
    )

    # Recursively process children if composite
    if operation.is_composite:
        for child_name, child_module in module.named_children():
            child_node = _build_execution_node_recursive(
                child_module, child_name, depth + 1, node_id
            )
            node.add_child(child_node)

    return node


def capture_static_execution_graph(model: nn.Module) -> ExecutionNode:
    """Capture PyTorch model structure as ExecutionNode hierarchy.

    compatible with the new renderer. Unlike the old capture_static(),
    this returns a single root ExecutionNode with recursive children.

    Args:
        model: PyTorch model to capture

    Returns:
        Root ExecutionNode with complete hierarchy

    Raises:
        TypeError: If model is not an nn.Module

    Example:
        >>> import torch.nn as nn
        >>> from vode.capture import capture_static_execution_graph
        >>>
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 10)
        ... )
        >>> root = capture_static_execution_graph(model)
        >>> print(f"Root has {len(root.children)} children")
        >>> print(f"Is expandable: {root.is_expandable}")
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model).__name__}")

    # Build ExecutionNode hierarchy
    root = _build_execution_node_recursive(
        module=model, name="", depth=0, node_id_prefix="root"
    )

    return root


def _flatten_tensors(
    data: Any, result: list[torch.Tensor] | None = None
) -> list[torch.Tensor]:
    """Recursively flatten nested structures to extract all tensors.

    Args:
        data: Input data (tensor, tuple, list, dict, or nested)
        result: Accumulator list for tensors

    Returns:
        List of all tensors found in the structure
    """
    if result is None:
        result = []

    if isinstance(data, torch.Tensor):
        result.append(data)
    elif isinstance(data, dict):
        for value in data.values():
            _flatten_tensors(value, result)
    elif isinstance(data, (tuple, list)):
        for item in data:
            _flatten_tensors(item, result)

    return result


def _tensor_to_tensor_info(tensor: torch.Tensor, name: str) -> TensorInfo:
    """Convert PyTorch tensor to TensorInfo.

    Args:
        tensor: PyTorch tensor
        name: Name for the tensor

    Returns:
        TensorInfo with tensor metadata
    """
    return TensorInfo(
        name=name,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
    )


def _count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a module.

    Args:
        module: PyTorch module

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _module_to_operation_info(module: nn.Module, name: str = "") -> OperationInfo:
    """Convert PyTorch module to OperationInfo.

    Args:
        module: PyTorch module
        name: Module name (optional)

    Returns:
        OperationInfo with module metadata
    """
    module_type = module.__class__.__name__
    op_name = name if name else module_type
    params_count = _count_parameters(module)

    # Check if module is composite (has children)
    children = list(module.children())
    is_composite = len(children) > 0

    # Detect loop structures
    is_loop = False
    loop_type = None
    iteration_count = None

    if isinstance(module, nn.Sequential):
        is_loop = True
        loop_type = "sequential"
        iteration_count = len(children)
    elif isinstance(module, nn.ModuleList):
        is_loop = True
        loop_type = "modulelist"
        iteration_count = len(children)

    return OperationInfo(
        op_type=module_type,
        op_name=op_name,
        params_count=params_count,
        is_composite=is_composite,
        is_loop=is_loop,
        loop_type=loop_type,
        iteration_count=iteration_count,
    )


class DynamicExecutionCapture:
    """Captures PyTorch model execution as ExecutionNode hierarchy.

    Uses forward hooks to capture runtime tensor information while building
    the ExecutionNode hierarchy that matches the module structure.

    Attributes:
        model: PyTorch model to capture
        root_node: Root ExecutionNode
    """

    def __init__(self, model: nn.Module):
        """Initialize dynamic execution capture.

        Args:
            model: PyTorch model to capture

        Raises:
            TypeError: If model is not an nn.Module
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(model).__name__}")

        self.model = model
        self.root_node: ExecutionNode | None = None

        # Tracking state
        self._hooks: list[Any] = []
        self._module_to_node: dict[int, ExecutionNode] = {}  # id(module) -> node
        self._module_to_name: dict[int, str] = {}  # id(module) -> name
        self._module_call_count: dict[int, int] = {}  # id(module) -> call count
        self._node_counter = 0

    def _get_module_name(self, module: nn.Module) -> str:
        """Get the name of a module from the model.

        Args:
            module: Module instance

        Returns:
            Module name or class name if not found
        """
        module_id = id(module)
        if module_id in self._module_to_name:
            return self._module_to_name[module_id]

        for name, mod in self.model.named_modules():
            self._module_to_name[id(mod)] = name if name else mod.__class__.__name__

        return self._module_to_name.get(module_id, module.__class__.__name__)

    def _build_module_hierarchy(self) -> None:
        """Build ExecutionNode hierarchy matching module structure."""
        # Create nodes for all modules
        for name, module in self.model.named_modules():
            module_id = id(module)
            depth = name.count(".") if name else 0

            # Create operation info
            operation = _module_to_operation_info(module, name)

            # Create ExecutionNode (tensors will be populated by hooks)
            node = ExecutionNode(
                node_id=f"node_{self._node_counter}",
                name=name if name else module.__class__.__name__,
                depth=depth,
                inputs=[],
                operation=operation,
                outputs=[],
                children=[],
                is_expandable=operation.is_composite,
                is_expanded=False,
            )

            self._module_to_node[module_id] = node
            self._node_counter += 1

            # Set root node
            if not name:  # Root module
                self.root_node = node

        # Build parent-child relationships
        for name, module in self.model.named_modules():
            if not name:  # Skip root
                continue

            module_id = id(module)
            node = self._module_to_node[module_id]

            # Find parent
            parts = name.split(".")
            if len(parts) == 1:
                # Direct child of root
                parent_module = self.model
            else:
                # Find parent module
                parent_name = ".".join(parts[:-1])
                parent_module = dict(self.model.named_modules())[parent_name]

            parent_id = id(parent_module)
            if parent_id in self._module_to_node:
                parent_node = self._module_to_node[parent_id]
                parent_node.add_child(node)

    def _pre_forward_hook(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        """Hook called before module forward pass.

        Args:
            module: Module being executed
            inputs: Input arguments to the module
        """
        module_id = id(module)
        if module_id not in self._module_to_node:
            return

        # Track module call count for reuse detection
        self._module_call_count[module_id] = (
            self._module_call_count.get(module_id, 0) + 1
        )

        node = self._module_to_node[module_id]

        # Flatten inputs to get all tensors
        input_tensors = _flatten_tensors(inputs)

        # Convert to TensorInfo
        node.inputs = [
            _tensor_to_tensor_info(tensor, f"input_{idx}")
            for idx, tensor in enumerate(input_tensors)
        ]

    def _post_forward_hook(
        self, module: nn.Module, inputs: tuple[Any, ...], outputs: Any
    ) -> None:
        """Hook called after module forward pass.

        Args:
            module: Module that was executed
            inputs: Input arguments to the module
            outputs: Output from the module
        """
        module_id = id(module)
        if module_id not in self._module_to_node:
            return

        node = self._module_to_node[module_id]

        # Flatten outputs to get all tensors
        output_tensors = _flatten_tensors(outputs)

        # Convert to TensorInfo
        node.outputs = [
            _tensor_to_tensor_info(tensor, f"output_{idx}")
            for idx, tensor in enumerate(output_tensors)
        ]

    def _register_hooks(self) -> None:
        """Register forward hooks on all modules."""
        for module in self.model.modules():
            # Register pre-hook (before forward)
            pre_hook = module.register_forward_pre_hook(self._pre_forward_hook)
            self._hooks.append(pre_hook)

            # Register post-hook (after forward)
            post_hook = module.register_forward_hook(self._post_forward_hook)
            self._hooks.append(post_hook)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def capture(self, *args: Any, **kwargs: Any) -> ExecutionNode:
        """Capture model execution with sample inputs.

        Args:
            *args: Positional arguments for model forward pass
            **kwargs: Keyword arguments for model forward pass

        Returns:
            Root ExecutionNode with complete hierarchy and runtime data

        Raises:
            RuntimeError: If forward pass fails
        """
        # Build module hierarchy
        self._build_module_hierarchy()

        # Register hooks
        self._register_hooks()

        try:
            # Run forward pass with no_grad to avoid gradient overhead
            with torch.no_grad():
                _ = self.model(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Forward pass failed during capture: {e}") from e
        finally:
            # Always remove hooks
            self._remove_hooks()

        # Detect module reuse and mark as loops
        self._detect_module_reuse()

        if self.root_node is None:
            raise RuntimeError("Failed to capture root node")

        return self.root_node

    def _detect_module_reuse(self) -> None:
        """Detect modules that were called multiple times and mark as reuse loops."""
        for module_id, call_count in self._module_call_count.items():
            if call_count > 1:
                node = self._module_to_node.get(module_id)
                if node:
                    # Only mark as reuse if not already marked as sequential/modulelist
                    if not node.operation.is_loop:
                        node.operation.is_loop = True
                        node.operation.loop_type = "reuse"
                        node.operation.iteration_count = call_count


def capture_dynamic_execution_graph(model: nn.Module, input_data: Any) -> ExecutionNode:
    """Capture PyTorch model execution as ExecutionNode hierarchy.

    with runtime tensor information. Unlike the old capture_dynamic(),
    this returns a single root ExecutionNode with recursive children.

    Args:
        model: PyTorch model to capture
        input_data: Sample input for forward pass (tensor, tuple, dict, etc.)

    Returns:
        Root ExecutionNode with complete hierarchy and runtime data

    Raises:
        TypeError: If model is not an nn.Module
        RuntimeError: If forward pass fails

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from vode.capture import capture_dynamic_execution_graph
        >>>
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 10)
        ... )
        >>> x = torch.randn(5, 10)
        >>> root = capture_dynamic_execution_graph(model, x)
        >>> print(f"Root has {len(root.children)} children")
        >>> print(f"Input shape: {root.inputs[0].shape}")
        >>> print(f"Output shape: {root.outputs[0].shape}")
    """
    capturer = DynamicExecutionCapture(model)

    # Handle different input types
    if isinstance(input_data, torch.Tensor):
        return capturer.capture(input_data)
    elif isinstance(input_data, (tuple, list)):
        return capturer.capture(*input_data)
    elif isinstance(input_data, dict):
        return capturer.capture(**input_data)
    else:
        return capturer.capture(input_data)
