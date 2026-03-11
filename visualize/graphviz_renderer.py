"""Graphviz renderer for VODE computation graphs.

Renders ComputationGraph to graphviz DOT format with depth control.
"""

from typing import Any

from vode.core.graph import ComputationGraph
from vode.core.nodes import Node, TensorNode, ModuleNode, FunctionNode, LoopNode


class GraphvizRenderer:
    """Renders ComputationGraph to graphviz DOT format.

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

    def __init__(self, graph: ComputationGraph):
        """Initialize renderer with a computation graph.

        Args:
            graph: ComputationGraph to render
        """
        self.graph = graph
        self.rendered_nodes: set[str] = set()
        self.node_counter = 0

    def render(
        self,
        max_depth: int | None = None,
        collapse_loops: bool = True,
        rankdir: str = "LR",
    ) -> str:
        """Render graph to graphviz DOT format.

        Args:
            max_depth: Maximum depth to render (None for full tree)
            collapse_loops: Whether to collapse loop nodes
            rankdir: Graph direction ('LR' for left-right, 'TB' for top-bottom)

        Returns:
            DOT format string
        """
        self.rendered_nodes.clear()
        self.node_counter = 0

        # Start DOT graph
        lines = [
            "digraph ComputationGraph {",
            f'    graph [ordering=in rankdir={rankdir} size="12.0,12.0"]',
            '    node [align=left fontname="Linux libertine" fontsize=10 height=0.2 margin=0 ranksep=0.1 shape=plaintext style=filled]',
            "    edge [fontsize=10]",
            "",
        ]

        # Render nodes
        for root_id in self.graph.root_node_ids:
            self._render_subtree(
                root_id, lines, max_depth, collapse_loops, current_depth=0
            )

        # Render edges
        lines.append("")

        # First, render explicit dataflow edges if they exist
        if self.graph.edges:
            for source_id, target_id in self.graph.edges:
                # Only render edges for nodes we've rendered
                if (
                    source_id in self.rendered_nodes
                    and target_id in self.rendered_nodes
                ):
                    source_label = self._get_node_label(source_id)
                    target_label = self._get_node_label(target_id)

                    # Get edge label (tensor shape if available)
                    edge_label = self._get_edge_label(source_id, target_id)
                    if edge_label:
                        lines.append(
                            f'    {source_label} -> {target_label} [label="{edge_label}"]'
                        )
                    else:
                        lines.append(f"    {source_label} -> {target_label}")
        else:
            # If no explicit edges, render hierarchical structure
            for node_id in self.rendered_nodes:
                node = self.graph.get_node(node_id)
                if node:
                    for child_id in node.children:
                        if child_id in self.rendered_nodes:
                            source_label = self._get_node_label(node_id)
                            target_label = self._get_node_label(child_id)
                            lines.append(f"    {source_label} -> {target_label}")

        lines.append("}")
        return "\n".join(lines)

    def _render_subtree(
        self,
        node_id: str,
        lines: list[str],
        max_depth: int | None,
        collapse_loops: bool,
        current_depth: int,
    ) -> None:
        """Recursively render a subtree.

        Args:
            node_id: Root node ID for this subtree
            lines: List to append DOT lines to
            max_depth: Maximum depth to render
            collapse_loops: Whether to collapse loops
            current_depth: Current depth in traversal
        """
        if node_id in self.rendered_nodes:
            return

        node = self.graph.get_node(node_id)
        if node is None:
            return

        # Check depth limit
        if max_depth is not None and current_depth >= max_depth:
            # Render collapsed node
            self._render_collapsed_node(node_id, lines)
            return

        # Render this node
        self._render_node(node, lines, collapse_loops)
        self.rendered_nodes.add(node_id)

        # Render children
        for child_id in self.graph.get_children(node_id):
            self._render_subtree(
                child_id, lines, max_depth, collapse_loops, current_depth + 1
            )

    def _render_node(self, node: Node, lines: list[str], collapse_loops: bool) -> None:
        """Render a single node to DOT format.

        Args:
            node: Node to render
            lines: List to append DOT lines to
            collapse_loops: Whether to collapse loops
        """
        node_label = self._get_node_label(node.node_id)

        if isinstance(node, TensorNode):
            html = self._render_tensor_node(node)
        elif isinstance(node, ModuleNode):
            html = self._render_module_node(node)
        elif isinstance(node, FunctionNode):
            html = self._render_function_node(node)
        elif isinstance(node, LoopNode):
            html = self._render_loop_node(node, collapse_loops)
        else:
            html = self._render_generic_node(node)

        lines.append(f"    {node_label} [label=<{html}>]")

    def _render_tensor_node(self, node: TensorNode) -> str:
        """Render TensorNode as HTML table.

        Args:
            node: TensorNode to render

        Returns:
            HTML table string
        """
        shape_str = self._format_shape(node.shape) if node.shape else "unknown"
        dtype_str = node.dtype.replace("torch.", "") if node.dtype else ""
        device_str = node.device if node.device else ""

        # Build info string
        info_parts = [shape_str]
        if dtype_str:
            info_parts.append(dtype_str)
        if device_str and device_str != "cpu":
            info_parts.append(device_str)

        info_str = ", ".join(info_parts)

        # Add stats if available
        stats_row = ""
        if node.stats:
            stats_str = ", ".join(f"{k}={v:.3f}" for k, v in node.stats.items())
            stats_row = f'<TR><TD COLSPAN="2">{stats_str}</TD></TR>'

        color = self.COLORS["TensorNode"]

        return f"""
                    <TABLE BORDER="0" CELLBORDER="1"
                    CELLSPACING="0" CELLPADDING="4">
                        <TR><TD>{node.name}<BR/>depth:{node.depth}</TD><TD>{info_str}</TD></TR>
                        {stats_row}
                    </TABLE>> fillcolor={color}"""

    def _render_module_node(self, node: ModuleNode) -> str:
        """Render ModuleNode as HTML table.

        Args:
            node: ModuleNode to render

        Returns:
            HTML table string
        """
        module_type = node.module_type or "Module"

        # Format input shapes
        input_str = (
            ", ".join(self._format_shape(s) for s in node.input_shapes)
            if node.input_shapes
            else "unknown"
        )

        # Format output shapes
        output_str = (
            ", ".join(self._format_shape(s) for s in node.output_shapes)
            if node.output_shapes
            else "unknown"
        )

        # Add parameter info if available
        param_info = ""
        if node.params:
            total_params = node.params.get("total_params", 0)
            if total_params > 0:
                param_info = f"<BR/>{self._format_number(total_params)} params"

        color = self.COLORS["ModuleNode"]

        return f"""
                    <TABLE BORDER="0" CELLBORDER="1"
                    CELLSPACING="0" CELLPADDING="4">
                    <TR>
                        <TD ROWSPAN="2">{module_type}<BR/>depth:{node.depth}{param_info}</TD>
                        <TD COLSPAN="2">input:</TD>
                        <TD COLSPAN="2">{input_str}</TD>
                    </TR>
                    <TR>
                        <TD COLSPAN="2">output:</TD>
                        <TD COLSPAN="2">{output_str}</TD>
                    </TR>
                    </TABLE>> fillcolor={color}"""

    def _render_function_node(self, node: FunctionNode) -> str:
        """Render FunctionNode as HTML table.

        Args:
            node: FunctionNode to render

        Returns:
            HTML table string
        """
        func_name = node.func_name or "function"

        # Format input shapes
        input_str = (
            ", ".join(self._format_shape(s) for s in node.input_shapes)
            if node.input_shapes
            else "unknown"
        )

        # Format output shapes
        output_str = (
            ", ".join(self._format_shape(s) for s in node.output_shapes)
            if node.output_shapes
            else "unknown"
        )

        # Add inplace indicator
        inplace_str = " (inplace)" if node.is_inplace else ""

        color = self.COLORS["FunctionNode"]

        return f"""
                    <TABLE BORDER="0" CELLBORDER="1"
                    CELLSPACING="0" CELLPADDING="4">
                    <TR>
                        <TD ROWSPAN="2">{func_name}{inplace_str}<BR/>depth:{node.depth}</TD>
                        <TD COLSPAN="2">input:</TD>
                        <TD COLSPAN="2">{input_str}</TD>
                    </TR>
                    <TR>
                        <TD COLSPAN="2">output:</TD>
                        <TD COLSPAN="2">{output_str}</TD>
                    </TR>
                    </TABLE>> fillcolor={color}"""

    def _render_loop_node(self, node: LoopNode, collapse: bool) -> str:
        """Render LoopNode as HTML table.

        Args:
            node: LoopNode to render
            collapse: Whether to show collapsed view

        Returns:
            HTML table string
        """
        loop_type = node.loop_type
        iter_count = node.iteration_count or "?"

        if collapse:
            body_info = f"{len(node.body_node_ids)} nodes"
        else:
            body_info = "expanded"

        color = self.COLORS["LoopNode"]

        return f"""
                    <TABLE BORDER="0" CELLBORDER="1"
                    CELLSPACING="0" CELLPADDING="4">
                        <TR><TD>{loop_type} loop<BR/>depth:{node.depth}</TD><TD>{iter_count} iterations<BR/>{body_info}</TD></TR>
                    </TABLE>> fillcolor={color}"""

    def _render_generic_node(self, node: Node) -> str:
        """Render generic Node as HTML table.

        Args:
            node: Node to render

        Returns:
            HTML table string
        """
        node_type = node.__class__.__name__

        return f"""
                    <TABLE BORDER="0" CELLBORDER="1"
                    CELLSPACING="0" CELLPADDING="4">
                        <TR><TD>{node_type}<BR/>{node.name}<BR/>depth:{node.depth}</TD></TR>
                    </TABLE>> fillcolor=white"""

    def _render_collapsed_node(self, node_id: str, lines: list[str]) -> None:
        """Render a collapsed node (beyond max_depth).

        Args:
            node_id: Node ID to render as collapsed
            lines: List to append DOT lines to
        """
        if node_id in self.rendered_nodes:
            return

        # Count how many nodes are collapsed
        descendants = self.graph.get_descendants(node_id)
        count = len(descendants) + 1

        node_label = self._get_node_label(node_id)
        color = self.COLORS["collapsed"]

        html = f"""
                    <TABLE BORDER="0" CELLBORDER="1"
                    CELLSPACING="0" CELLPADDING="4">
                        <TR><TD>... {count} more nodes</TD></TR>
                    </TABLE>> fillcolor={color}"""

        lines.append(f"    {node_label} [label=<{html}>]")
        self.rendered_nodes.add(node_id)

    def _get_node_label(self, node_id: str) -> str:
        """Get graphviz label for a node ID.

        Args:
            node_id: Node ID

        Returns:
            Graphviz node label (e.g., 'n0', 'n1')
        """
        # Use node_id directly but sanitize it
        return f'n{node_id.replace("-", "_").replace(":", "_")}'

    def _get_edge_label(self, source_id: str, target_id: str) -> str:
        """Get label for an edge (tensor shape if available).

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            Edge label string (empty if no label)
        """
        source_node = self.graph.get_node(source_id)

        # If source is a tensor, show its shape
        if isinstance(source_node, TensorNode) and source_node.shape:
            return self._format_shape(source_node.shape)

        return ""

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
