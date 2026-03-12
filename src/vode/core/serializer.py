"""JSON serialization and deserialization for VODE graphs.

Handles conversion between ComputationGraph/ExecutionNode and JSON format.
"""

import json
from typing import Any
from pathlib import Path

from .graph import ComputationGraph
from .nodes import (
    Node,
    TensorNode,
    ModuleNode,
    FunctionNode,
    LoopNode,
    ExecutionNode,
    TensorInfo,
    OperationInfo,
)


def serialize_graph(graph: ComputationGraph) -> str:
    """Serialize ComputationGraph to JSON string.

    Args:
        graph: ComputationGraph to serialize

    Returns:
        JSON string representation
    """
    return json.dumps(graph.to_dict(), indent=2)


def deserialize_graph(json_str: str) -> ComputationGraph:
    """Deserialize JSON string to ComputationGraph.

    Args:
        json_str: JSON string to deserialize

    Returns:
        Reconstructed ComputationGraph

    Raises:
        ValueError: If JSON is invalid or missing required fields
    """
    data = json.loads(json_str)

    # Create empty graph
    graph = ComputationGraph()

    # Reconstruct nodes
    nodes_data = data.get("nodes", {})
    for node_id, node_dict in nodes_data.items():
        node_type = node_dict.get("type")

        # Create appropriate node type
        if node_type == "TensorNode":
            node = TensorNode(**{k: v for k, v in node_dict.items() if k != "type"})
        elif node_type == "ModuleNode":
            node = ModuleNode(**{k: v for k, v in node_dict.items() if k != "type"})
        elif node_type == "FunctionNode":
            node = FunctionNode(**{k: v for k, v in node_dict.items() if k != "type"})
        elif node_type == "LoopNode":
            node = LoopNode(**{k: v for k, v in node_dict.items() if k != "type"})
        else:
            node = Node(**{k: v for k, v in node_dict.items() if k != "type"})

        graph.add_node(node)

    # Reconstruct edges
    graph.edges = data.get("edges", [])

    # Reconstruct root nodes
    graph.root_node_ids = data.get("root_node_ids", [])

    # Reconstruct hierarchy
    graph.node_hierarchy = data.get("node_hierarchy", {})

    # Reconstruct loops
    loops_data = data.get("detected_loops", [])
    for loop_dict in loops_data:
        loop = LoopNode(**{k: v for k, v in loop_dict.items() if k != "type"})
        graph.detected_loops.append(loop)

    return graph


def save_graph(graph: ComputationGraph, filepath: str | Path) -> None:
    """Save ComputationGraph to JSON file.

    Args:
        graph: ComputationGraph to save
        filepath: Path to output file
    """
    filepath = Path(filepath)
    json_str = serialize_graph(graph)
    filepath.write_text(json_str)


def load_graph(filepath: str | Path) -> ComputationGraph:
    """Load ComputationGraph from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded ComputationGraph
    """
    filepath = Path(filepath)
    json_str = filepath.read_text()
    return deserialize_graph(json_str)


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
