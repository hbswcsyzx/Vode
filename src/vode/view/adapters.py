"""Data adapters to convert Stage 1 JSON format to frontend format."""

from typing import Any, Dict, List


def adapt_graph_for_frontend(stage1_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Stage 1 trace JSON to frontend-expected format.

    Args:
        stage1_data: Stage 1 JSON with structure:
            {
                "version": "1.0",
                "timestamp": "...",
                "graph": {
                    "root_call_ids": ["call_0"],
                    "function_calls": [...],
                    "variables": [...],
                    "edges": [...]
                }
            }

    Returns:
        Frontend format:
            {
                "version": "1.0",
                "root_node": "call_0",
                "nodes": {"call_0": {...}, ...},
                "dataflow_edges": [...]
            }
    """
    graph = stage1_data.get("graph", {})
    function_calls = graph.get("function_calls", [])
    edges = graph.get("edges", [])
    root_call_ids = graph.get("root_call_ids", [])

    # Build parent-child relationships from parent_id field
    children_map: Dict[str, List[str]] = {}
    for call in function_calls:
        parent_id = call.get("parent_id")
        child_id = call.get("id")
        if parent_id is not None:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(child_id)

    # Convert function_calls array to nodes dict
    nodes = {}
    for call in function_calls:
        node_id = call.get("id", "")
        nodes[node_id] = {
            "id": node_id,
            "function_name": call.get(
                "display_name", call.get("qualified_name", "unknown")
            ),
            "module": (
                call.get("qualified_name", "").rsplit(".", 1)[0]
                if "." in call.get("qualified_name", "")
                else ""
            ),
            "file": call.get("filename", ""),
            "line": call.get("lineno", 0),
            "depth": call.get("depth", 0),
            "args": {},  # Could be populated from variables if needed
            "return_value": None,  # Could be populated from variables if needed
            "children": children_map.get(node_id, []),
        }

    # Extract dataflow edges
    dataflow_edges = [
        {
            "from_node": edge.get("source", ""),
            "to_node": edge.get("target", ""),
            "from_value": edge.get("source_slot", ""),
            "to_param": edge.get("target_slot", ""),
        }
        for edge in edges
        if edge.get("kind") == "dataflow"
    ]

    # Get root node (use first root_call_id)
    root_node = root_call_ids[0] if root_call_ids else ""

    return {
        "version": stage1_data.get("version", "1.0"),
        "root_node": root_node,
        "nodes": nodes,
        "dataflow_edges": dataflow_edges,
    }
