"""API routes for VODE interactive visualization.

Defines REST API endpoints for graph data access and manipulation.
This is a placeholder for future implementation (Stage 3).
"""

from pathlib import Path
from typing import Optional

# Note: FastAPI is an optional dependency
try:
    from fastapi import APIRouter, HTTPException, Request
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None
    HTTPException = None
    BaseModel = None


if FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/api", tags=["graph"])
    
    class NodeQuery(BaseModel):
        """Query parameters for node details."""
        node_id: str
    
    class GraphQuery(BaseModel):
        """Query parameters for graph data."""
        max_depth: Optional[int] = None
        filter_type: Optional[str] = None
    
    @router.get("/graph")
    async def get_graph(request: Request, max_depth: Optional[int] = None):
        """Get the complete graph data.
        
        Args:
            request: FastAPI request object
            max_depth: Optional maximum depth to return
            
        Returns:
            Graph data as JSON
        """
        graph_file = request.app.state.graph_file
        
        if not graph_file:
            raise HTTPException(status_code=404, detail="No graph file loaded")
        
        graph_path = Path(graph_file)
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail=f"Graph file not found: {graph_file}")
        
        # Load and return graph data
        try:
            from vode.core import load_graph
            graph = load_graph(graph_path)
            return graph.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading graph: {str(e)}")
    
    @router.get("/node/{node_id}")
    async def get_node(request: Request, node_id: str):
        """Get details for a specific node.
        
        Args:
            request: FastAPI request object
            node_id: Node identifier
            
        Returns:
            Node data as JSON
        """
        graph_file = request.app.state.graph_file
        
        if not graph_file:
            raise HTTPException(status_code=404, detail="No graph file loaded")
        
        try:
            from vode.core import load_graph
            graph = load_graph(graph_file)
            node = graph.get_node(node_id)
            
            if not node:
                raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
            
            return node.to_dict()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading node: {str(e)}")
    
    @router.get("/stats")
    async def get_stats(request: Request):
        """Get graph statistics.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Graph statistics as JSON
        """
        graph_file = request.app.state.graph_file
        
        if not graph_file:
            raise HTTPException(status_code=404, detail="No graph file loaded")
        
        try:
            from vode.core import load_graph
            graph = load_graph(graph_file)
            return graph.get_stats()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading stats: {str(e)}")

else:
    # Placeholder when FastAPI is not available
    router = None
