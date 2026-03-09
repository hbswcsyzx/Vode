"""Graph builder classes for VODE neural network visualization.

This module provides graph builders for constructing structure and dataflow graphs:
- StructureGraph: Holds nodes and edges for model structure
- DataflowGraph: Holds nodes and edges for dataflow
"""

from dataclasses import dataclass, field
from typing import Any

from vode.nn.graph.nodes import Node


@dataclass
class Edge:
    """Represents a directed edge between two nodes.

    Attributes:
        src_id: Source node ID
        dst_id: Destination node ID
        label: Optional edge label
        metadata: Optional edge metadata
    """

    src_id: str
    dst_id: str
    label: str | None = None
    metadata: dict[str, Any] | None = None


class StructureGraph:
    """Graph builder for model structure visualization.

    Captures the hierarchical structure of nn.Module instances,
    showing parent-child relationships and module organization.

    Attributes:
        nodes: Dictionary mapping node IDs to Node objects
        edges: List of Edge objects
    """

    def __init__(self) -> None:
        """Initialize an empty structure graph."""
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: Node object to add

        Raises:
            ValueError: If node with same ID already exists
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID '{node.node_id}' already exists")
        self.nodes[node.node_id] = node

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            src_id: Source node ID
            dst_id: Destination node ID
            label: Optional edge label
            metadata: Optional edge metadata

        Raises:
            ValueError: If source or destination node doesn't exist
        """
        if src_id not in self.nodes:
            raise ValueError(f"Source node '{src_id}' not found")
        if dst_id not in self.nodes:
            raise ValueError(f"Destination node '{dst_id}' not found")

        edge = Edge(src_id=src_id, dst_id=dst_id, label=label, metadata=metadata)
        self.edges.append(edge)

        # Update parent-child relationships
        self.nodes[src_id].add_child(dst_id)
        self.nodes[dst_id].add_parent(src_id)

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            Node object or None if not found
        """
        return self.nodes.get(node_id)

    def get_nodes(self) -> list[Node]:
        """Get all nodes in the graph.

        Returns:
            List of all Node objects
        """
        return list(self.nodes.values())

    def get_edges(self) -> list[Edge]:
        """Get all edges in the graph.

        Returns:
            List of all Edge objects
        """
        return self.edges

    def get_nodes_by_depth(self, depth: int) -> list[Node]:
        """Get all nodes at a specific depth.

        Args:
            depth: Depth level to filter by

        Returns:
            List of nodes at the specified depth
        """
        return [node for node in self.nodes.values() if node.depth == depth]

    def get_children(self, node_id: str) -> list[Node]:
        """Get all child nodes of a given node.

        Args:
            node_id: Parent node ID

        Returns:
            List of child Node objects
        """
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [
            self.nodes[child_id] for child_id in node.children if child_id in self.nodes
        ]

    def get_parents(self, node_id: str) -> list[Node]:
        """Get all parent nodes of a given node.

        Args:
            node_id: Child node ID

        Returns:
            List of parent Node objects
        """
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [
            self.nodes[parent_id]
            for parent_id in node.parents
            if parent_id in self.nodes
        ]

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.nodes.clear()
        self.edges.clear()


class DataflowGraph:
    """Graph builder for dataflow visualization.

    Captures the flow of tensors through operations during forward pass,
    showing how data transforms through the network.

    Attributes:
        nodes: Dictionary mapping node IDs to Node objects
        edges: List of Edge objects
    """

    def __init__(self) -> None:
        """Initialize an empty dataflow graph."""
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: Node object to add

        Raises:
            ValueError: If node with same ID already exists
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID '{node.node_id}' already exists")
        self.nodes[node.node_id] = node

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            src_id: Source node ID
            dst_id: Destination node ID
            label: Optional edge label
            metadata: Optional edge metadata

        Raises:
            ValueError: If source or destination node doesn't exist
        """
        if src_id not in self.nodes:
            raise ValueError(f"Source node '{src_id}' not found")
        if dst_id not in self.nodes:
            raise ValueError(f"Destination node '{dst_id}' not found")

        edge = Edge(src_id=src_id, dst_id=dst_id, label=label, metadata=metadata)
        self.edges.append(edge)

        # Update parent-child relationships
        self.nodes[src_id].add_child(dst_id)
        self.nodes[dst_id].add_parent(src_id)

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            Node object or None if not found
        """
        return self.nodes.get(node_id)

    def get_nodes(self) -> list[Node]:
        """Get all nodes in the graph.

        Returns:
            List of all Node objects
        """
        return list(self.nodes.values())

    def get_edges(self) -> list[Edge]:
        """Get all edges in the graph.

        Returns:
            List of all Edge objects
        """
        return self.edges

    def get_nodes_by_depth(self, depth: int) -> list[Node]:
        """Get all nodes at a specific depth.

        Args:
            depth: Depth level to filter by

        Returns:
            List of nodes at the specified depth
        """
        return [node for node in self.nodes.values() if node.depth == depth]

    def get_children(self, node_id: str) -> list[Node]:
        """Get all child nodes of a given node.

        Args:
            node_id: Parent node ID

        Returns:
            List of child Node objects
        """
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [
            self.nodes[child_id] for child_id in node.children if child_id in self.nodes
        ]

    def get_parents(self, node_id: str) -> list[Node]:
        """Get all parent nodes of a given node.

        Args:
            node_id: Child node ID

        Returns:
            List of parent Node objects
        """
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [
            self.nodes[parent_id]
            for parent_id in node.parents
            if parent_id in self.nodes
        ]

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.nodes.clear()
        self.edges.clear()
