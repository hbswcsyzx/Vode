"""Data validation for VODE graphs.

Validates ExecutionNode structures for correctness.
"""

from typing import Any
from .nodes import ExecutionNode


class ValidationError(Exception):
    """Raised when graph validation fails."""

    pass


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
