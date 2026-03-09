"""REST API endpoints for Vode viewer."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException

from vode.view.adapters import adapt_graph_for_frontend

router = APIRouter(prefix="/api")

# Global state to hold loaded graph data
_graph_data: Optional[Dict[str, Any]] = None
_frontend_data_cache: Optional[Dict[str, Any]] = None


def set_graph_data(data: Dict[str, Any]) -> None:
    """Set the graph data to be served by the API."""
    global _graph_data, _frontend_data_cache
    _graph_data = data
    # Pre-convert and cache frontend format
    _frontend_data_cache = adapt_graph_for_frontend(data)


@router.get("/graph")
async def get_graph() -> Dict[str, Any]:
    """Get complete trace graph data in frontend format."""
    if _frontend_data_cache is None:
        raise HTTPException(status_code=500, detail="Graph data not loaded")

    # Return cached frontend format
    return _frontend_data_cache


@router.get("/node/{node_id}")
async def get_node(node_id: str) -> Dict[str, Any]:
    """Get detailed information for a specific node."""
    if _graph_data is None:
        raise HTTPException(status_code=500, detail="Graph data not loaded")

    # Search in function_calls
    for call in _graph_data.get("graph", {}).get("function_calls", []):
        if call.get("id") == node_id:
            return call

    raise HTTPException(status_code=404, detail=f"Node {node_id} not found")


@router.get("/search")
async def search_nodes(q: str, limit: int = 50) -> Dict[str, List[Dict[str, Any]]]:
    """Search nodes by function name or other criteria."""
    if _graph_data is None:
        raise HTTPException(status_code=500, detail="Graph data not loaded")

    q_lower = q.lower()
    results = []

    for call in _graph_data.get("graph", {}).get("function_calls", []):
        # Search in qualified_name, display_name, and filename
        if (
            q_lower in call.get("qualified_name", "").lower()
            or q_lower in call.get("display_name", "").lower()
            or q_lower in call.get("filename", "").lower()
        ):
            results.append(call)
            if len(results) >= limit:
                break

    return {"results": results}


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get graph statistics."""
    if _graph_data is None:
        raise HTTPException(status_code=500, detail="Graph data not loaded")

    graph = _graph_data.get("graph", {})
    function_calls = graph.get("function_calls", [])
    variables = graph.get("variables", [])
    edges = graph.get("edges", [])

    # Calculate max depth
    max_depth = max((call.get("depth", 0) for call in function_calls), default=0)

    return {
        "function_count": len(function_calls),
        "variable_count": len(variables),
        "edge_count": len(edges),
        "max_depth": max_depth,
    }
