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
    ComputationGraph,
    ModuleNode,
    LoopNode,
    ExecutionNode,
    TensorInfo,
    OperationInfo,
    generate_node_id,
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


class StaticCapture:
    """Captures PyTorch model structure statically.

    Traverses module hierarchy without executing forward passes.

    Attributes:
        model: PyTorch model to capture
        graph: Resulting computation graph
    """

    def __init__(self, model: nn.Module):
        """Initialize static capture.

        Args:
            model: PyTorch model to capture

        Raises:
            TypeError: If model is not an nn.Module
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(model).__name__}")

        self.model = model
        self.graph = ComputationGraph()
        self._module_to_node_id: dict[str, str] = {}

    def capture(self) -> ComputationGraph:
        """Capture model structure and return computation graph.

        Returns:
            ComputationGraph with module hierarchy
        """
        # Traverse all modules
        for name, module in self.model.named_modules():
            self._process_module(name, module)

        # Build hierarchy relationships
        self._build_hierarchy()

        # Detect loop patterns
        self._detect_loops()

        return self.graph

    def _process_module(self, name: str, module: nn.Module) -> None:
        """Process a single module and create node.

        Args:
            name: Module name from named_modules (empty string for root)
            module: Module instance
        """
        # Generate node ID
        node_id = name if name else "root"
        self._module_to_node_id[name] = node_id

        # Calculate depth from name
        depth = name.count(".") if name else 0

        # Count parameters
        param_count = sum(p.numel() for p in module.parameters())
        trainable_params = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )

        # Check if leaf module
        is_leaf = self._is_leaf_module(module)

        # Create module node
        node = ModuleNode(
            node_id=node_id,
            name=sanitize_name(name if name else module.__class__.__name__),
            depth=depth,
            module_type=module.__class__.__name__,
            module_instance_id=id(module),
            is_container=not is_leaf,
            is_parameterless=(param_count == 0),
            params={
                "total": param_count,
                "trainable": trainable_params,
            },
            metadata={
                "module_path": name,
                "class_name": module.__class__.__name__,
            },
        )

        self.graph.add_node(node)

    def _is_leaf_module(self, module: nn.Module) -> bool:
        """Check if module is a leaf (no submodules to traverse).

        Args:
            module: Module to check

        Returns:
            True if module is a leaf
        """
        # Check if it's a known leaf type
        if type(module) in LEAF_MODULES:
            return True

        # Check if it has no children (excluding self)
        children = list(module.children())
        return len(children) == 0

    def _build_hierarchy(self) -> None:
        """Build parent-child relationships from module names."""
        for name, node_id in self._module_to_node_id.items():
            if not name:  # Root node
                continue

            # Find parent by removing last component
            parts = name.split(".")
            if len(parts) == 1:
                # Direct child of root
                parent_id = "root"
            else:
                # Find parent by joining all but last part
                parent_name = ".".join(parts[:-1])
                parent_id = self._module_to_node_id.get(parent_name, "root")

            # Update node relationships
            node = self.graph.get_node(node_id)
            parent = self.graph.get_node(parent_id)

            if node and parent:
                node.add_parent(parent_id)
                parent.add_child(node_id)

    def _detect_loops(self) -> None:
        """Detect Sequential and ModuleList patterns as loops."""
        for name, module in self.model.named_modules():
            node_id = self._module_to_node_id.get(name, name if name else "root")

            # Detect Sequential
            if isinstance(module, nn.Sequential):
                self._create_loop_node(node_id, module, "sequential")

            # Detect ModuleList
            elif isinstance(module, nn.ModuleList):
                self._create_loop_node(node_id, module, "modulelist")

    def _create_loop_node(
        self, parent_id: str, module: nn.Module, loop_type: str
    ) -> None:
        """Create a LoopNode for Sequential/ModuleList.

        Args:
            parent_id: Parent module node ID
            module: Sequential or ModuleList instance
            loop_type: 'sequential' or 'modulelist'
        """
        parent_node = self.graph.get_node(parent_id)
        if not parent_node:
            return

        # Get child node IDs
        child_ids = self.graph.get_children(parent_id)

        # Create loop node
        loop_node = LoopNode(
            node_id=f"{parent_id}_loop",
            name=f"{parent_node.name}_loop",
            depth=parent_node.depth + 1,
            loop_type=loop_type,  # type: ignore
            iteration_count=len(child_ids),
            body_node_ids=child_ids,
            is_collapsed=True,
        )

        # Add to detected loops
        self.graph.detected_loops.append(loop_node)


def capture_static(model: nn.Module) -> ComputationGraph:
    """Capture PyTorch model structure statically.

    Main API function for static capture. Inspects model hierarchy
    without running forward passes.

    Args:
        model: PyTorch model to capture

    Returns:
        ComputationGraph with module hierarchy

    Raises:
        TypeError: If model is not an nn.Module

    Example:
        >>> import torch.nn as nn
        >>> from vode.capture import capture_static
        >>>
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 10)
        ... )
        >>> graph = capture_static(model)
        >>> print(f"Captured {len(graph.nodes)} nodes")
    """
    capturer = StaticCapture(model)
    return capturer.capture()


# ============================================================================
# ============================================================================


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
