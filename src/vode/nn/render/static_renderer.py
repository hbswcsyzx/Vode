"""Static renderer for converting Graphviz .gv files to image formats.

This module provides functionality to render .gv files to static image formats
(SVG, PNG, PDF) using the Graphviz dot command.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal


class StaticRenderer:
    """Renderer for converting Graphviz .gv files to static image formats.

    This class provides methods to render .gv files to various static formats
    using the Graphviz dot command-line tool.

    Supported formats:
        - svg: Scalable Vector Graphics
        - png: Portable Network Graphics
        - pdf: Portable Document Format

    Example:
        >>> renderer = StaticRenderer()
        >>> renderer.render('model.gv', 'model.svg', format='svg')
        'model.svg'
    """

    SUPPORTED_FORMATS = {"svg", "png", "pdf"}

    def __init__(self):
        """Initialize the StaticRenderer.

        Raises:
            RuntimeError: If Graphviz is not installed or dot command is not found.
        """
        self._check_graphviz_installed()

    def _check_graphviz_installed(self) -> None:
        """Check if Graphviz dot command is available.

        Raises:
            RuntimeError: If dot command is not found in PATH.
        """
        if shutil.which("dot") is None:
            raise RuntimeError(
                "Graphviz is not installed or 'dot' command is not in PATH. "
                "Please install Graphviz:\n"
                "  - Ubuntu/Debian: sudo apt-get install graphviz\n"
                "  - macOS: brew install graphviz\n"
                "  - Windows: Download from https://graphviz.org/download/"
            )

    def render(
        self,
        gv_path: str,
        output_path: str,
        format: Literal["svg", "png", "pdf"] = "svg",
    ) -> str:
        """Render a .gv file to a static image format.

        Args:
            gv_path: Path to the input .gv file.
            output_path: Path where the output file should be saved.
            format: Output format ('svg', 'png', or 'pdf'). Defaults to 'svg'.

        Returns:
            Path to the generated output file.

        Raises:
            FileNotFoundError: If the input .gv file does not exist.
            ValueError: If the format is not supported.
            RuntimeError: If the rendering process fails.

        Example:
            >>> renderer = StaticRenderer()
            >>> renderer.render('model.gv', 'model.svg', format='svg')
            'model.svg'
            >>> renderer.render('model.gv', 'model.png', format='png')
            'model.png'
        """
        # Validate format
        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        # Check if input file exists
        gv_path_obj = Path(gv_path)
        if not gv_path_obj.exists():
            raise FileNotFoundError(f"Input file not found: {gv_path}")

        if not gv_path_obj.is_file():
            raise ValueError(f"Input path is not a file: {gv_path}")

        # Create output directory if it doesn't exist
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Build the dot command
        cmd = ["dot", f"-T{format}", str(gv_path_obj), "-o", str(output_path_obj)]

        try:
            # Execute the dot command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to render {gv_path} to {format} format.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {e.stderr}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while rendering {gv_path}: {e}"
            ) from e

        # Verify output file was created
        if not output_path_obj.exists():
            raise RuntimeError(
                f"Rendering completed but output file was not created: {output_path}"
            )

        return str(output_path_obj)

    @staticmethod
    def is_graphviz_available() -> bool:
        """Check if Graphviz is available without raising an exception.

        Returns:
            True if Graphviz dot command is available, False otherwise.

        Example:
            >>> if StaticRenderer.is_graphviz_available():
            ...     renderer = StaticRenderer()
            ...     renderer.render('model.gv', 'model.svg')
            ... else:
            ...     print("Please install Graphviz")
        """
        return shutil.which("dot") is not None
