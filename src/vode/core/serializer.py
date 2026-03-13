"""JSON serialization and deserialization for VODE graphs.

Handles conversion between ExecutionNode and JSON format.
"""

import json
from typing import Any
from pathlib import Path

from .nodes import (
    ExecutionNode,
    TensorInfo,
    OperationInfo,
)


def serialize_execution_node(node: ExecutionNode) -> str:
    """Serialize ExecutionNode to JSON string.

    Args:
        node: ExecutionNode to serialize

    Returns:
        JSON string representation
    """
    return json.dumps(node.to_dict(), indent=2)


def deserialize_execution_node(json_str: str) -> ExecutionNode:
    """Deserialize JSON string to ExecutionNode.

    Args:
        json_str: JSON string to deserialize

    Returns:
        Reconstructed ExecutionNode

    Raises:
        ValueError: If JSON is invalid or missing required fields
    """
    data = json.loads(json_str)

    # Reconstruct TensorInfo objects
    inputs = [TensorInfo(**inp) for inp in data.get("inputs", [])]
    outputs = [TensorInfo(**out) for out in data.get("outputs", [])]

    # Reconstruct OperationInfo
    operation = OperationInfo(**data.get("operation", {}))

    # Create node
    node = ExecutionNode(
        node_id=data["node_id"],
        name=data["name"],
        depth=data["depth"],
        inputs=inputs,
        operation=operation,
        outputs=outputs,
        is_expandable=data.get("is_expandable", False),
        is_expanded=data.get("is_expanded", False),
    )

    # Recursively reconstruct children
    for child_data in data.get("children", []):
        child_json = json.dumps(child_data)
        child = deserialize_execution_node(child_json)
        node.add_child(child)

    return node


def save_execution_node(node: ExecutionNode, filepath: str | Path) -> None:
    """Save ExecutionNode to JSON file.

    Args:
        node: ExecutionNode to save
        filepath: Path to output file
    """
    filepath = Path(filepath)
    json_str = serialize_execution_node(node)
    filepath.write_text(json_str)


def load_execution_node(filepath: str | Path) -> ExecutionNode:
    """Load ExecutionNode from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded ExecutionNode
    """
    filepath = Path(filepath)
    json_str = filepath.read_text()
    return deserialize_execution_node(json_str)
