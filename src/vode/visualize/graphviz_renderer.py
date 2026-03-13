"""Graphviz renderer for VODE computation graphs.

Renders ExecutionNode to graphviz DOT format with depth control.
"""

from typing import Any

from vode.core.nodes import (
    ExecutionNode,
    TensorInfo,
    OperationInfo,
)


class GraphvizRenderer:
    """Renders ExecutionNode to graphviz DOT format.

    Follows input->operation->output pattern with depth control.
    """

    # Color scheme for different node types
    COLORS = {
        "TensorNode": "lightyellow",
        "ModuleNode": "darkseagreen1",
        "FunctionNode": "aliceblue",
        "LoopNode": "lightblue",
        "collapsed": "lightgray",
    }

    def __init__(self):
        """Initialize renderer."""
        self.rendered_nodes: set[str] = set()
        self.node_counter = 0

    # ========================================================================
    # ExecutionNode Rendering
    # ========================================================================

    def render_execution_graph(
        self, root: ExecutionNode, max_depth: int = 1, rankdir: str = "LR"
    ) -> str:
        """Render ExecutionNode graph with three-column layout.

        for each node, with depth-based recursive expansion.

        Args:
            root: Root ExecutionNode to render
            max_depth: Maximum depth to expand (0 = show only root)
            rankdir: Graph direction ('LR' for left-right, 'TB' for top-bottom)

        Returns:
            DOT format string
        """
        # Step 1: Expand nodes to specified depth
        expanded_nodes = expand_to_depth(root, max_depth)

        # Step 2: Flatten to linear sequence
        operation_sequence = flatten_to_sequence(expanded_nodes)

        # Step 3: Render to Graphviz
        lines = [
            "digraph ExecutionGraph {",
            f"    graph [ordering=in rankdir={rankdir}]",
            '    node [align=left fontname="Linux libertine" fontsize=10 height=0.2 margin=0 ranksep=0.1 shape=plaintext style=filled]',
            "    edge [fontsize=10]",
            "",
        ]

        # Render each operation node
        for i, node in enumerate(operation_sequence):
            node_label = f"op{i}"
            html = self._render_execution_node_html(node)
            lines.append(f"    {node_label} [label=<{html}>]")

        # Add edges between consecutive operations
        lines.append("")
        for i in range(len(operation_sequence) - 1):
            lines.append(f"    op{i} -> op{i+1}")

        lines.append("}")
        return "\n".join(lines)

    def _render_execution_node_html(self, node: ExecutionNode) -> str:
        """Render ExecutionNode as HTML table with three columns.

        Format:
        ┌─────────────┬──────────────────┬─────────────┐
        │   INPUTS    │    OPERATION     │   OUTPUTS   │
        │  shape info │  op_type/name    │  shape info │
        └─────────────┴──────────────────┴─────────────┘

        Args:
            node: ExecutionNode to render

        Returns:
            HTML table string for Graphviz
        """
        # Format inputs
        inputs_html = self._format_tensors_for_column(node.inputs)

        # Format operation
        op_html = self._format_operation_for_column(node.operation)

        # Format outputs
        outputs_html = self._format_tensors_for_column(node.outputs)

        # Build three-column table (single line for DOT syntax compatibility)
        html = f'<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4"><TR><TD BGCOLOR="lightyellow">{inputs_html}</TD><TD BGCOLOR="darkseagreen1">{op_html}</TD><TD BGCOLOR="lightyellow">{outputs_html}</TD></TR></TABLE>'
        return html

    def _format_tensors_for_column(self, tensors: list[TensorInfo]) -> str:
        """Format list of TensorInfo for display in a column.

        Args:
            tensors: List of TensorInfo objects

        Returns:
            Formatted HTML string
        """
        if not tensors:
            return "-"

        parts = []
        for tensor in tensors:
            shape_str = self._format_shape(tensor.shape) if tensor.shape else "unknown"
            parts.append(f"{tensor.name}<BR/>{shape_str}")

        return "<BR/>".join(parts)

    def _format_operation_for_column(self, operation: OperationInfo) -> str:
        """Format OperationInfo for display in the operation column.

        Args:
            operation: OperationInfo object

        Returns:
            Formatted HTML string
        """
        parts = [f"<B>{operation.op_type}</B>"]

        if operation.op_name and operation.op_name != operation.op_type:
            parts.append(operation.op_name)

        if operation.params_count > 0:
            parts.append(f"{self._format_number(operation.params_count)} params")

        # Display loop information
        if operation.is_loop:
            loop_info = f"{operation.loop_type} loop"
            if operation.iteration_count:
                loop_info += f" ({operation.iteration_count}x)"
            parts.append(f"<I>{loop_info}</I>")

        if operation.is_composite:
            parts.append("(composite)")

        return "<BR/>".join(parts)

    def _format_shape(self, shape: tuple[int, ...] | None) -> str:
        """Format tensor shape for display.

        Args:
            shape: Tensor shape tuple

        Returns:
            Formatted shape string
        """
        if shape is None:
            return "unknown"
        return f'({", ".join(str(d) for d in shape)})'

    def _format_number(self, num: int) -> str:
        """Format large numbers with K/M/B suffixes.

        Args:
            num: Number to format

        Returns:
            Formatted number string
        """
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.1f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        else:
            return str(num)


# ============================================================================
# ============================================================================


def expand_to_depth(
    node: ExecutionNode, max_depth: int, current_depth: int = 0
) -> list[ExecutionNode]:
    """Recursively expand nodes to specified depth.

    At each depth level, we decide whether to show the node as-is (collapsed)
    or expand it into its children.

    Key behavior:
    - depth=0: Show only the top-level node (no expansion)
    - depth=1: Expand one level (show immediate children)
    - depth=N: Expand N levels deep

    Important: We never show both a parent and its children. If we expand a node,
    we only show its children, not the parent itself.

    Args:
        node: ExecutionNode to expand
        max_depth: Maximum depth to expand to
        current_depth: Current depth in the recursion (default 0)

    Returns:
        List of ExecutionNode objects at the target depth level
    """
    # Base case 1: At max depth, return the node as-is
    if current_depth >= max_depth:
        return [node]

    # Base case 2: Node is not expandable, return as-is
    if not node.can_expand():
        return [node]

    # Recursive case: Expand children
    result = []
    for child in node.children:
        result.extend(expand_to_depth(child, max_depth, current_depth + 1))

    return result


def flatten_to_sequence(nodes: list[ExecutionNode]) -> list[ExecutionNode]:
    """Flatten expanded nodes into a linear sequence.

    Converts a potentially nested tree structure into a linear sequence
    of operations for visualization. This handles the case where expansion
    might produce a tree, but we want to display it as a linear flow.

    For now, this is a simple pass-through since expand_to_depth already
    produces a flat list. In the future, this could handle more complex
    flattening logic (e.g., handling branches, merges, etc.).

    Args:
        nodes: List of ExecutionNode objects (potentially nested)

    Returns:
        Flattened list of ExecutionNode objects in execution order
    """
    # For now, just return the nodes as-is since expand_to_depth
    # already produces a flat list
    return nodes


# ============================================================================
# ============================================================================


def render_execution_graph(
    root: ExecutionNode, max_depth: int = 1, rankdir: str = "LR"
):
    """Render ExecutionNode graph to Graphviz format (standalone API).

    This is a convenience function that creates a GraphvizRenderer instance
    and calls its render_execution_graph method.

    Args:
        root: Root ExecutionNode to render
        max_depth: Maximum depth to expand (0 = show only root)
        rankdir: Graph direction ('LR' for left-right, 'TB' for top-bottom)

    Returns:
        Graphviz Digraph object

    Example:
        >>> from vode.capture import capture_static_execution_graph
        >>> from vode.visualize import render_execution_graph
        >>> import torch.nn as nn
        >>>
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        >>> root = capture_static_execution_graph(model)
        >>> dot = render_execution_graph(root, max_depth=1)
        >>> dot.render('output')  # Save to file
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz package is required for rendering. "
            "Install it with: pip install graphviz"
        )

    # Create renderer
    renderer = GraphvizRenderer()

    # Render to DOT string
    dot_string = renderer.render_execution_graph(root, max_depth, rankdir)

    # Convert to graphviz.Digraph object
    dot = graphviz.Source(dot_string)
    return dot
