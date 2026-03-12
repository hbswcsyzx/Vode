"""Dynamic capture for PyTorch models.

Traces actual runtime execution to capture tensor shapes, dtypes, devices, and data flow.
"""

from typing import Any
from collections import defaultdict
import torch
import torch.nn as nn

from vode.core import (
    ComputationGraph,
    ModuleNode,
    TensorNode,
    LoopNode,
    ExecutionNode,
    TensorInfo,
    OperationInfo,
    generate_node_id,
    sanitize_name,
)


def _extract_tensor_metadata(
    tensor: torch.Tensor, compute_stats: bool = False
) -> dict[str, Any]:
    """Extract metadata from a tensor.

    Args:
        tensor: Tensor to extract metadata from
        compute_stats: Whether to compute statistics (min, max, mean, std)

    Returns:
        Dictionary with tensor metadata
    """
    metadata = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad,
    }

    if compute_stats and tensor.numel() > 0:
        try:
            with torch.no_grad():
                metadata["stats"] = {
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                    "mean": float(tensor.mean().item()),
                    "std": float(tensor.std().item()),
                }
        except (RuntimeError, ValueError):
            # Some dtypes don't support these operations
            metadata["stats"] = None
    else:
        metadata["stats"] = None

    return metadata


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


class DynamicCapture:
    """Captures PyTorch model execution dynamically.

    Uses forward hooks to trace actual runtime execution and capture
    tensor shapes, dtypes, devices, and data flow.

    Attributes:
        model: PyTorch model to capture
        compute_stats: Whether to compute tensor statistics
        graph: Resulting computation graph
    """

    def __init__(self, model: nn.Module, compute_stats: bool = False):
        """Initialize dynamic capture.

        Args:
            model: PyTorch model to capture
            compute_stats: Whether to compute tensor statistics (min, max, mean, std)

        Raises:
            TypeError: If model is not an nn.Module
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(model).__name__}")

        self.model = model
        self.compute_stats = compute_stats
        self.graph = ComputationGraph()

        # Tracking state
        self._hooks: list[Any] = []
        self._execution_order: list[tuple[str, nn.Module]] = []
        self._module_to_node_id: dict[int, str] = {}  # id(module) -> node_id
        self._module_call_count: dict[int, int] = defaultdict(int)  # Track reuse
        self._tensor_to_node_id: dict[int, str] = {}  # id(tensor) -> node_id
        self._node_counter = 0

    def _generate_node_id(self, prefix: str) -> str:
        """Generate unique node ID.

        Args:
            prefix: Prefix for the node ID

        Returns:
            Unique node ID
        """
        node_id = f"{prefix}_{self._node_counter}"
        self._node_counter += 1
        return node_id

    def _create_tensor_nodes(
        self, tensors: list[torch.Tensor], prefix: str
    ) -> list[str]:
        """Create TensorNode instances for a list of tensors.

        Args:
            tensors: List of tensors
            prefix: Prefix for node naming (e.g., 'input', 'output')

        Returns:
            List of created tensor node IDs
        """
        node_ids = []

        for idx, tensor in enumerate(tensors):
            # Check if we've already created a node for this tensor
            tensor_id = id(tensor)
            if tensor_id in self._tensor_to_node_id:
                node_ids.append(self._tensor_to_node_id[tensor_id])
                continue

            # Extract metadata
            metadata = _extract_tensor_metadata(tensor, self.compute_stats)

            # Create tensor node
            node_id = self._generate_node_id(f"{prefix}_tensor")
            node = TensorNode(
                node_id=node_id,
                name=f"{prefix}_{idx}",
                tensor_id=str(tensor_id),
                shape=metadata["shape"],
                dtype=metadata["dtype"],
                device=metadata["device"],
                requires_grad=metadata["requires_grad"],
                stats=metadata["stats"],
            )

            self.graph.add_node(node)
            self._tensor_to_node_id[tensor_id] = node_id
            node_ids.append(node_id)

        return node_ids

    def _pre_forward_hook(self, module: nn.Module, inputs: tuple[Any, ...]) -> None:
        """Hook called before module forward pass.

        Args:
            module: Module being executed
            inputs: Input arguments to the module
        """
        # Flatten inputs to get all tensors
        input_tensors = _flatten_tensors(inputs)

        if not input_tensors:
            return

        # Track execution order
        module_id = id(module)
        module_name = self._get_module_name(module)
        self._execution_order.append((module_name, module))

        # Create or get module node
        if module_id not in self._module_to_node_id:
            node_id = self._generate_node_id("module")
            self._module_to_node_id[module_id] = node_id

            # Count parameters
            param_count = sum(p.numel() for p in module.parameters())
            trainable_params = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )

            # Create module node
            node = ModuleNode(
                node_id=node_id,
                name=sanitize_name(module_name),
                module_type=module.__class__.__name__,
                module_instance_id=module_id,
                is_parameterless=(param_count == 0),
                params={
                    "total": param_count,
                    "trainable": trainable_params,
                },
                metadata={
                    "class_name": module.__class__.__name__,
                },
            )

            self.graph.add_node(node)

        # Track module reuse
        self._module_call_count[module_id] += 1

        # Create input tensor nodes
        input_node_ids = self._create_tensor_nodes(input_tensors, "input")

        # Store input shapes on module node
        module_node_id = self._module_to_node_id[module_id]
        module_node = self.graph.get_node(module_node_id)
        if module_node and isinstance(module_node, ModuleNode):
            module_node.input_shapes = [t.shape for t in input_tensors]
            module_node.input_ids = input_node_ids

            # Add edges from inputs to module
            for input_id in input_node_ids:
                self.graph.add_edge(input_id, module_node_id)

    def _post_forward_hook(
        self, module: nn.Module, inputs: tuple[Any, ...], outputs: Any
    ) -> None:
        """Hook called after module forward pass.

        Args:
            module: Module that was executed
            inputs: Input arguments to the module
            outputs: Output from the module
        """
        # Flatten outputs to get all tensors
        output_tensors = _flatten_tensors(outputs)

        if not output_tensors:
            return

        module_id = id(module)
        module_node_id = self._module_to_node_id.get(module_id)

        if not module_node_id:
            return

        # Create output tensor nodes
        output_node_ids = self._create_tensor_nodes(output_tensors, "output")

        # Store output shapes on module node
        module_node = self.graph.get_node(module_node_id)
        if module_node and isinstance(module_node, ModuleNode):
            module_node.output_shapes = [t.shape for t in output_tensors]
            module_node.output_ids = output_node_ids

            # Add edges from module to outputs
            for output_id in output_node_ids:
                self.graph.add_edge(module_node_id, output_id)

    def _get_module_name(self, module: nn.Module) -> str:
        """Get the name of a module from the model.

        Args:
            module: Module instance

        Returns:
            Module name or class name if not found
        """
        for name, mod in self.model.named_modules():
            if mod is module:
                return name if name else module.__class__.__name__
        return module.__class__.__name__

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

    def _detect_loops(self) -> None:
        """Detect module reuse patterns as loops."""
        # Find modules that were called multiple times
        for module_id, call_count in self._module_call_count.items():
            if call_count > 1:
                module_node_id = self._module_to_node_id.get(module_id)
                if not module_node_id:
                    continue

                module_node = self.graph.get_node(module_node_id)
                if not module_node:
                    continue

                # Create loop node
                loop_node = LoopNode(
                    node_id=f"{module_node_id}_loop",
                    name=f"{module_node.name}_loop",
                    depth=module_node.depth,
                    loop_type="recursive",
                    iteration_count=call_count,
                    body_node_ids=[module_node_id],
                    is_collapsed=True,
                    recursive_call_id=module_node_id,
                )

                self.graph.detected_loops.append(loop_node)

    def capture(self, *args: Any, **kwargs: Any) -> ComputationGraph:
        """Capture model execution with sample inputs.

        Args:
            *args: Positional arguments for model forward pass
            **kwargs: Keyword arguments for model forward pass

        Returns:
            ComputationGraph with runtime execution data

        Raises:
            RuntimeError: If forward pass fails
        """
        # Reset state
        self._execution_order.clear()
        self._module_to_node_id.clear()
        self._module_call_count.clear()
        self._tensor_to_node_id.clear()
        self._node_counter = 0
        self.graph = ComputationGraph()

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

        # Detect loops from module reuse
        self._detect_loops()

        return self.graph


def capture_dynamic(
    model: nn.Module, *args: Any, compute_stats: bool = False, **kwargs: Any
) -> ComputationGraph:
    """Capture PyTorch model execution dynamically.

    Main API function for dynamic capture. Traces actual runtime execution
    to capture tensor shapes, dtypes, devices, and data flow.

    Args:
        model: PyTorch model to capture
        *args: Sample inputs for forward pass (tensors, tuples, dicts, etc.)
        compute_stats: Whether to compute tensor statistics (min, max, mean, std)
        **kwargs: Additional keyword arguments for forward pass

    Returns:
        ComputationGraph with runtime execution data

    Raises:
        TypeError: If model is not an nn.Module
        RuntimeError: If forward pass fails

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from vode.capture import capture_dynamic
        >>>
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 10)
        ... )
        >>> x = torch.randn(5, 10)
        >>> graph = capture_dynamic(model, x, compute_stats=True)
        >>> print(f"Captured {len(graph.nodes)} nodes")
    """
    capturer = DynamicCapture(model, compute_stats=compute_stats)
    return capturer.capture(*args, **kwargs)


# ============================================================================
# New ExecutionNode-based capture (Stage 4)
# ============================================================================


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

    return OperationInfo(
        op_type=module_type,
        op_name=op_name,
        params_count=params_count,
        is_composite=is_composite,
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

        if self.root_node is None:
            raise RuntimeError("Failed to capture root node")

        return self.root_node


def capture_dynamic_execution_graph(model: nn.Module, input_data: Any) -> ExecutionNode:
    """Capture PyTorch model execution as ExecutionNode hierarchy.

    This is the new Stage 4 API that builds ExecutionNode hierarchies
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
