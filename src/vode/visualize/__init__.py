"""Visualization module for VODE.

Provides tools to visualize computation graphs:
- Static export to images (PNG, SVG, PDF) using Graphviz
- Interactive viewing (future) using FastAPI + React Flow
"""

# Graphviz renderer
from .graphviz_renderer import GraphvizRenderer

# High-level visualization API
from .visualizer import visualize, visualize_static, visualize_dynamic

# Wrapper function
from .vode_wrapper import vode

# Server (optional, for interactive viewing)
from .server import create_app, start_server

# Routes (optional, for interactive viewing)
from .routes import router

__all__ = [
    # Renderer
    "GraphvizRenderer",
    # Visualization API
    "visualize",
    "visualize_static",
    "visualize_dynamic",
    # Wrapper
    "vode",
    # Server (optional)
    "create_app",
    "start_server",
    "router",
]

