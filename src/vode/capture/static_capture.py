"""Static capture for PyTorch models.

Inspects model structure without running forward passes.
"""

from typing import Any
import torch
import torch.nn as nn

from vode.core import (
    ComputationGraph,
    ModuleNode,
    LoopNode,
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
