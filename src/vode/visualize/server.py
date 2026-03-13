"""FastAPI server for interactive VODE visualization.

Provides REST API for loading and exploring computation graphs interactively.
This is a placeholder for future implementation.
"""

from pathlib import Path
from typing import Optional

# Note: FastAPI is an optional dependency for future interactive viewing
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None


def create_app(graph_file: Optional[str] = None) -> "FastAPI":
    """Create FastAPI application for interactive visualization.

    Args:
        graph_file: Optional path to graph JSON file to load

    Returns:
        FastAPI application instance

    Raises:
        ImportError: If FastAPI is not installed
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for interactive visualization. "
            "Install it with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="VODE Interactive Viewer",
        description="Interactive visualization for VODE computation graphs",
        version="0.1.0",
    )

    # Mount static files (frontend build artifacts)
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Import and include routes
    from .routes import router

    app.include_router(router)

    # Store graph file path in app state
    app.state.graph_file = graph_file

    @app.get("/")
    async def root():
        """Serve the main HTML page."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "VODE Interactive Viewer (frontend not built yet)"}

    return app


def start_server(
    graph_file: str, host: str = "127.0.0.1", port: int = 8000, auto_open: bool = True
) -> None:
    """Start the interactive visualization server.

    Args:
        graph_file: Path to graph JSON file
        host: Host to bind to
        port: Port to bind to
        auto_open: Whether to automatically open browser

    Raises:
        ImportError: If required dependencies are not installed
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI and uvicorn are required for interactive visualization. "
            "Install them with: pip install fastapi uvicorn"
        )

    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required for running the server. "
            "Install it with: pip install uvicorn"
        )

    # Create app
    app = create_app(graph_file)

    # Open browser if requested
    if auto_open:
        import webbrowser
        import threading

        def open_browser():
            import time

            time.sleep(1)  # Wait for server to start
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

    # Start server
    print(f"Starting VODE Interactive Viewer at http://{host}:{port}")
    print(f"Loading graph from: {graph_file}")
    print("Press Ctrl+C to stop")

    uvicorn.run(app, host=host, port=port, log_level="info")
