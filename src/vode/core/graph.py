"""Computation graph container for VODE.

Manages the complete computation graph with nodes and edges.
"""

from dataclasses import dataclass, field
from typing import Any, Iterator

from .nodes import Node, LoopNode


@dataclass
class ComputationGraph:
    """Complete computation graph with all nodes and edges.

    Attributes:
        nodes: Dictionary mapping node_id to Node objects
        edges: List of edges as (source_id, target_id) tuples
        root_node_ids: List of root node IDs (entry points)
        node_hierarchy: Dictionary mapping parent_id to list of child_ids
        detected_loops: List of detected LoopNode instances
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)

    # Root nodes (entry points)
    root_node_ids: list[str] = field(default_factory=list)

    # Hierarchy tracking
    node_hierarchy: dict[str, list[str]] = field(default_factory=dict)

    # Loop tracking
    detected_loops: list[LoopNode] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: Node instance to add
        """
        self.nodes[node.node_id] = node

        # Update root nodes
        if node.is_root() and node.node_id not in self.root_node_ids:
            self.root_node_ids.append(node.node_id)

        # Update hierarchy
        for parent_id in node.parents:
            if parent_id not in self.node_hierarchy:
                self.node_hierarchy[parent_id] = []
            if node.node_id not in self.node_hierarchy[parent_id]:
                self.node_hierarchy[parent_id].append(node.node_id)

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by its ID.

        Args:
            node_id: Node identifier

        Returns:
            Node instance or None if not found
        """
        return self.nodes.get(node_id)

    def add_edge(self, source_id: str, target_id: str) -> None:
        """Add an edge between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
        """
        edge = (source_id, target_id)
        if edge not in self.edges:
            self.edges.append(edge)

    def get_children(self, node_id: str) -> list[str]:
        """Get child node IDs for a given node.

        Args:
            node_id: Parent node ID

        Returns:
            List of child node IDs
        """
        return self.node_hierarchy.get(node_id, [])

    def get_descendants(self, node_id: str) -> list[str]:
        """Get all descendant node IDs recursively.

        Args:
            node_id: Root node ID

        Returns:
            List of all descendant node IDs
        """
        descendants = []
        children = self.get_children(node_id)

        for child_id in children:
            descendants.append(child_id)
            descendants.extend(self.get_descendants(child_id))

        return descendants

    def traverse(
        self, start_node_id: str | None = None, order: str = "depth_first"
    ) -> Iterator[Node]:
        """Traverse the graph starting from a node.

        Args:
            start_node_id: Starting node ID (None for all roots)
            order: Traversal order ('depth_first' or 'breadth_first')

        Yields:
            Node instances in traversal order
        """
        if start_node_id is None:
            # Traverse from all roots
            for root_id in self.root_node_ids:
                yield from self.traverse(root_id, order)
            return

        node = self.get_node(start_node_id)
        if node is None:
            return

        visited = set()

        if order == "depth_first":
            yield from self._dfs(start_node_id, visited)
        elif order == "breadth_first":
            yield from self._bfs(start_node_id, visited)
        else:
            raise ValueError(f"Unknown traversal order: {order}")

    def _dfs(self, node_id: str, visited: set[str]) -> Iterator[Node]:
        """Depth-first search traversal.

        Args:
            node_id: Current node ID
            visited: Set of visited node IDs

        Yields:
            Node instances in DFS order
        """
        if node_id in visited:
            return

        visited.add(node_id)
        node = self.get_node(node_id)

        if node is not None:
            yield node

            for child_id in self.get_children(node_id):
                yield from self._dfs(child_id, visited)

    def _bfs(self, node_id: str, visited: set[str]) -> Iterator[Node]:
        """Breadth-first search traversal.

        Args:
            node_id: Starting node ID
            visited: Set of visited node IDs

        Yields:
            Node instances in BFS order
        """
        queue = [node_id]

        while queue:
            current_id = queue.pop(0)

            if current_id in visited:
                continue

            visited.add(current_id)
            node = self.get_node(current_id)

            if node is not None:
                yield node
                queue.extend(self.get_children(current_id))

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary representation.

        Returns:
            Dictionary with graph data
        """
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": self.edges,
            "root_node_ids": self.root_node_ids,
            "node_hierarchy": self.node_hierarchy,
            "detected_loops": [loop.to_dict() for loop in self.detected_loops],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the graph.

        Returns:
            Dictionary with graph statistics
        """
        from collections import Counter

        node_types = Counter(node.__class__.__name__ for node in self.nodes.values())

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "root_nodes": len(self.root_node_ids),
            "node_types": dict(node_types),
            "detected_loops": len(self.detected_loops),
            "max_depth": max((node.depth for node in self.nodes.values()), default=0),
        }
