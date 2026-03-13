"""Visualization module for VODE.

Provides tools to visualize computation graphs:
- Static export to images (PNG, SVG, PDF) using Graphviz
- Interactive viewing (future) using FastAPI + React Flow
"""

# Graphviz renderer
from .graphviz_renderer import GraphvizRenderer

# Wrapper function
from .vode_wrapper import vode

# Server (optional, for interactive viewing)
from .server import create_app, start_server

# Routes (optional, for interactive viewing)
from .routes import router

__all__ = [
    # Renderer
    "GraphvizRenderer",
    # Wrapper
    "vode",
    # Server (optional)
    "create_app",
    "start_server",
    "router",
]

