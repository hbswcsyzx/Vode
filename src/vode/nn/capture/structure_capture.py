"""Structure capture mechanism for VODE neural network visualization.

This module provides the StructureCapture class for building a graph of the model
architecture during/after initialization by introspecting the module hierarchy.
"""

from typing import Any

import torch.nn as nn

from vode.nn.graph.builder import StructureGraph
from vode.nn.graph.nodes import ModuleNode


class StructureCapture:
    """Captures the static structure of a PyTorch model.

    Uses introspection via `named_modules()` and `named_children()` to build
    a hierarchical graph of the model architecture without executing forward pass.

    Example:
        >>> import torch.nn as nn
        >>> from vode.nn.capture import StructureCapture
        >>>
        >>> model = nn.Sequential(
        ...     nn.Conv2d(3, 64, 3),
        ...     nn.ReLU(),
        ...     nn.Linear(64, 10)
        ... )
        >>>
        >>> capturer = StructureCapture()
        >>> graph = capturer.capture(model)
        >>> print(f"Captured {len(graph.get_nodes())} nodes")
    """

    def capture(self, model: nn.Module) -> StructureGraph:
        """Capture the structure of a PyTorch model.

        Traverses the module hierarchy and builds a graph representation with:
        - ModuleNode for each module (including root)
        - Parent-child edges based on module containment
        - Depth tracking for hierarchical visualization

        Args:
            model: PyTorch model to capture

        Returns:
            StructureGraph containing nodes and edges representing the model structure

        Raises:
            ValueError: If model is None

        Example:
            >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
            >>> capturer = StructureCapture()
            >>> graph = capturer.capture(model)
            >>> nodes = graph.get_nodes()
            >>> print(f"Root node: {nodes[0].name}")
        """
        if model is None:
            raise ValueError("Model cannot be None")

        graph = StructureGraph()

        # Phase 1: Create nodes for all modules
        for name, module in model.named_modules():
            node = self._create_module_node(name, module)
            graph.add_node(node)

        # Phase 2: Build parent-child edges
        self._build_edges(model, graph)

        return graph

    def _create_module_node(self, name: str, module: nn.Module) -> ModuleNode:
        """Create a ModuleNode from a module.

        Args:
            name: Full module name (e.g., "layer1.conv1" or "" for root)
            module: The module instance

        Returns:
            ModuleNode with name, type, depth, and optional parameters
        """
        # Calculate depth: empty string = 0, "conv1" = 1, "layer1.conv1" = 2
        depth = name.count(".") + 1 if name else 0

        # Get module type
        module_type = type(module).__name__

        # Create node with unique ID (use name, or "root" for empty string)
        node_id = name if name else "root"

        node = ModuleNode(
            node_id=node_id,
            name=name,
            depth=depth,
            module_type=module_type,
        )

        # Optionally extract parameters
        params = self._extract_params(module)
        if params:
            node.set_params(params)

        return node

    def _extract_params(self, module: nn.Module) -> dict[str, Any] | None:
        """Extract parameter information from a module.

        Args:
            module: The module to extract parameters from

        Returns:
            Dictionary of parameter information, or None if no parameters
        """
        params = {}

        # Count total parameters
        total_params = sum(p.numel() for p in module.parameters())
        if total_params > 0:
            params["total_params"] = total_params

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        if trainable_params > 0:
            params["trainable_params"] = trainable_params

        # Extract common module attributes
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            params["in_features"] = module.in_features
            params["out_features"] = module.out_features

        if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
            params["in_channels"] = module.in_channels
            params["out_channels"] = module.out_channels

        if hasattr(module, "kernel_size"):
            params["kernel_size"] = module.kernel_size

        if hasattr(module, "stride"):
            params["stride"] = module.stride

        if hasattr(module, "padding"):
            params["padding"] = module.padding

        return params if params else None

    def _build_edges(self, model: nn.Module, graph: StructureGraph) -> None:
        """Build parent-child edges in the graph.

        Iterates through all modules and their direct children to create edges
        representing the containment hierarchy.

        Args:
            model: The root model
            graph: The graph to add edges to
        """
        for parent_name, parent_module in model.named_modules():
            # Get parent node ID
            parent_id = parent_name if parent_name else "root"

            # Iterate through direct children
            for child_name, child_module in parent_module.named_children():
                # Construct full child name
                if parent_name:
                    full_child_name = f"{parent_name}.{child_name}"
                else:
                    full_child_name = child_name

                # Add edge from parent to child
                try:
                    graph.add_edge(
                        src_id=parent_id,
                        dst_id=full_child_name,
                        label="contains",
                    )
                except ValueError:
                    # Skip if child node doesn't exist (shouldn't happen)
                    pass
