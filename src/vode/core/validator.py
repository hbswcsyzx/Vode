"""Data validation for VODE graphs.

Validates ComputationGraph and ExecutionNode structures for correctness.
"""

from typing import Any
from .graph import ComputationGraph
from .nodes import Node, ExecutionNode


class ValidationError(Exception):
    """Raised when graph validation fails."""

    pass


def validate_graph(graph: ComputationGraph) -> bool:
    """Validate a ComputationGraph for structural correctness.

    Args:
        graph: ComputationGraph to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If graph is invalid
    """
    # Check that all nodes have unique IDs
    node_ids = set()
    for node_id in graph.nodes.keys():
        if node_id in node_ids:
            raise ValidationError(f"Duplicate node ID: {node_id}")
        node_ids.add(node_id)

    # Check that all edges reference existing nodes
    for source_id, target_id in graph.edges:
        if source_id not in graph.nodes:
            raise ValidationError(
                f"Edge references non-existent source node: {source_id}"
            )
        if target_id not in graph.nodes:
            raise ValidationError(
                f"Edge references non-existent target node: {target_id}"
            )

    # Check that all root nodes exist
    for root_id in graph.root_node_ids:
        if root_id not in graph.nodes:
            raise ValidationError(f"Root node does not exist: {root_id}")

    # Check that hierarchy references valid nodes
    for parent_id, child_ids in graph.node_hierarchy.items():
        if parent_id not in graph.nodes:
            raise ValidationError(
                f"Hierarchy references non-existent parent: {parent_id}"
            )
        for child_id in child_ids:
            if child_id not in graph.nodes:
                raise ValidationError(
                    f"Hierarchy references non-existent child: {child_id}"
                )

    # Check that parent-child relationships are consistent
    for node_id, node in graph.nodes.items():
        # Check parents
        for parent_id in node.parents:
            if parent_id not in graph.nodes:
                raise ValidationError(
                    f"Node {node_id} references non-existent parent: {parent_id}"
                )
            parent = graph.nodes[parent_id]
            if node_id not in parent.children:
                raise ValidationError(
                    f"Parent-child relationship inconsistent: "
                    f"{parent_id} -> {node_id}"
                )

        # Check children
        for child_id in node.children:
            if child_id not in graph.nodes:
                raise ValidationError(
                    f"Node {node_id} references non-existent child: {child_id}"
                )
            child = graph.nodes[child_id]
            if node_id not in child.parents:
                raise ValidationError(
                    f"Parent-child relationship inconsistent: "
                    f"{node_id} -> {child_id}"
                )

    return True


def validate_execution_node(node: ExecutionNode) -> bool:
    """Validate an ExecutionNode for structural correctness.

    Args:
        node: ExecutionNode to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If node is invalid
    """
    # Check that node has required fields
    if not node.node_id:
        raise ValidationError("ExecutionNode missing node_id")

    if not node.name:
        raise ValidationError(f"ExecutionNode {node.node_id} missing name")

    if node.depth < 0:
        raise ValidationError(f"ExecutionNode {node.node_id} has negative depth")

    # Check that operation is valid
    if not node.operation:
        raise ValidationError(f"ExecutionNode {node.node_id} missing operation")

    if not node.operation.op_type:
        raise ValidationError(f"ExecutionNode {node.node_id} operation missing op_type")

    # Check that expandable nodes have children
    if node.is_expandable and len(node.children) == 0:
        raise ValidationError(
            f"ExecutionNode {node.node_id} is marked expandable but has no children"
        )

    # Recursively validate children
    for child in node.children:
        validate_execution_node(child)

        # Check that child's parent reference is correct
        if child.parent != node:
            raise ValidationError(
                f"Child {child.node_id} has incorrect parent reference"
            )

        # Check that child's depth is correct
        if child.depth != node.depth + 1:
            raise ValidationError(
                f"Child {child.node_id} has incorrect depth "
                f"(expected {node.depth + 1}, got {child.depth})"
            )

    return True


def validate_json_data(data: dict[str, Any]) -> bool:
    """Validate JSON data structure for graph deserialization.

    Args:
        data: Dictionary from JSON deserialization

    Returns:
        True if valid

    Raises:
        ValidationError: If data is invalid
    """
    # Check required top-level keys
    if "nodes" not in data:
        raise ValidationError("JSON data missing 'nodes' key")

    # Check that nodes is a dictionary
    if not isinstance(data["nodes"], dict):
        raise ValidationError("'nodes' must be a dictionary")

    # Check that each node has required fields
    for node_id, node_data in data["nodes"].items():
        if not isinstance(node_data, dict):
            raise ValidationError(f"Node {node_id} data must be a dictionary")

        required_fields = ["node_id", "name", "depth", "type"]
        for field in required_fields:
            if field not in node_data:
                raise ValidationError(f"Node {node_id} missing required field: {field}")

    return True
