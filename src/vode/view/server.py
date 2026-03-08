"""FastAPI server for Vode viewer."""

import json
import logging
import webbrowser
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from vode.view.api import router, set_graph_data

logger = logging.getLogger(__name__)


class ViewServer:
    """Web server for viewing trace files."""

    def __init__(self, trace_file: Path, host: str = "127.0.0.1", port: int = 8000):
        """Initialize the view server.

        Args:
            trace_file: Path to trace JSON file
            host: Server host address
            port: Server port
        """
        self.trace_file = trace_file
        self.host = host
        self.port = port
        self.app = FastAPI(title="Vode Viewer")
        self.graph_data: Optional[dict] = None

        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self) -> None:
        """Configure CORS middleware for development."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        """Set up API routes and static file serving."""
        # Include API router
        self.app.include_router(router)

        # Serve frontend static files
        frontend_dist = Path(__file__).parent / "frontend" / "dist"
        if frontend_dist.exists():
            self.app.mount(
                "/assets",
                StaticFiles(directory=frontend_dist / "assets"),
                name="assets",
            )

            @self.app.get("/", response_class=HTMLResponse)
            async def root():
                index_file = frontend_dist / "index.html"
                if index_file.exists():
                    return index_file.read_text()
                return self._fallback_html()

        else:
            # Fallback if frontend not built
            @self.app.get("/", response_class=HTMLResponse)
            async def root():
                return self._fallback_html()

    def _fallback_html(self) -> str:
        """Return fallback HTML when frontend is not built."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vode Viewer</title>
        </head>
        <body>
            <h1>Vode Viewer</h1>
            <p>Frontend not built yet. Please run:</p>
            <pre>cd vode/src/vode/view/frontend && npm install && npm run build</pre>
            <p>API endpoints available:</p>
            <ul>
                <li><a href="/api/graph">/api/graph</a> - Get complete graph data</li>
                <li><a href="/api/stats">/api/stats</a> - Get graph statistics</li>
                <li>/api/node/{id} - Get node details</li>
                <li>/api/search?q=query - Search nodes</li>
            </ul>
        </body>
        </html>
        """

    def load_trace(self) -> None:
        """Load and parse trace JSON file."""
        logger.info(f"Loading trace file: {self.trace_file}")

        if not self.trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {self.trace_file}")

        with open(self.trace_file, "r") as f:
            self.graph_data = json.load(f)

        # Set graph data in API module
        set_graph_data(self.graph_data)

        logger.info("Trace file loaded successfully")

    def run(self, open_browser: bool = True) -> None:
        """Start the server and optionally open browser.

        Args:
            open_browser: Whether to automatically open browser
        """
        # Load trace file
        self.load_trace()

        url = f"http://{self.host}:{self.port}"

        # Print startup message
        print(f"\n{'='*60}")
        print(f"Vode Web Viewer")
        print(f"{'='*60}")
        print(f"Server starting at: {url}")
        print(f"Trace file: {self.trace_file}")
        print(f"\nPress Ctrl+C to stop the server")
        print(f"{'='*60}\n")

        # Open browser if requested
        if open_browser:
            logger.info(f"Opening browser at {url}")
            webbrowser.open(url)

        # Start server
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
