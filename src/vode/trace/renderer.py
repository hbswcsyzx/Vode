"""Text-based renderer for trace graphs.

This module provides a simple text-based tree visualization of function call traces.
"""

from vode.trace.models import TraceGraph, FunctionCallNode, VariableRecord, TensorMeta


class TextRenderer:
    """Renders a TraceGraph as a text-based tree structure."""

    def __init__(self):
        """Initialize the text renderer."""
        self.graph: TraceGraph | None = None
        self.call_map: dict[str, FunctionCallNode] = {}
        self.var_map: dict[str, VariableRecord] = {}
        self.children_map: dict[str, list[str]] = {}

    def render(self, graph: TraceGraph) -> str:
        """Render a TraceGraph as a text tree.

        Args:
            graph: TraceGraph to render

        Returns:
            String representation of the graph
        """
        self.graph = graph
        self._build_maps()

        lines = ["Function Call Tree:"]

        # Render each root call
        visited = set()
        for root_id in graph.root_call_ids:
            lines.extend(self._render_tree(root_id, visited, indent=""))

        # Add dataflow summary
        dataflow_edges = [e for e in graph.edges if e.kind == "dataflow"]
        lines.append("")
        lines.append(f"Dataflow Edges: {len(dataflow_edges)} edges")

        return "\n".join(lines)

    def _build_maps(self) -> None:
        """Build lookup maps for efficient access."""
        # Map call IDs to nodes
        self.call_map = {node.id: node for node in self.graph.function_calls}

        # Map variable IDs to variables
        self.var_map = {var.id: var for var in self.graph.variables}

        # Build parent-child relationships
        self.children_map = {}
        for node in self.graph.function_calls:
            if node.parent_id:
                if node.parent_id not in self.children_map:
                    self.children_map[node.parent_id] = []
                self.children_map[node.parent_id].append(node.id)

    def _render_tree(self, node_id: str, visited: set[str], indent: str) -> list[str]:
        """Render a node and its children recursively.

        Args:
            node_id: ID of the node to render
            visited: Set of already visited node IDs
            indent: Current indentation string

        Returns:
            List of lines representing the tree
        """
        if node_id in visited:
            return []

        visited.add(node_id)
        node = self.call_map.get(node_id)
        if not node:
            return []

        lines = []

        # Render the node itself
        lines.append(self._render_node(node, indent))

        # Get children
        children = self.children_map.get(node_id, [])

        if children:
            # Render parameters
            if node.arg_variable_ids:
                param_lines = self._format_variables(
                    node.arg_variable_ids, "Parameters"
                )
                for i, line in enumerate(param_lines):
                    if i == len(param_lines) - 1 and not node.return_variable_ids:
                        # Last parameter line and no returns
                        lines.append(f"{indent}│  └─ {line}")
                    else:
                        lines.append(f"{indent}│  ├─ {line}")

            # Render return values
            if node.return_variable_ids:
                return_lines = self._format_variables(
                    node.return_variable_ids, "Returns"
                )
                for i, line in enumerate(return_lines):
                    if i == len(return_lines) - 1:
                        # Last return line
                        lines.append(f"{indent}│  └─ {line}")
                    else:
                        lines.append(f"{indent}│  ├─ {line}")

            # Render children
            for i, child_id in enumerate(children):
                is_last = i == len(children) - 1
                if is_last:
                    child_lines = self._render_tree(child_id, visited, indent + "   ")
                else:
                    child_lines = self._render_tree(child_id, visited, indent + "│  ")
                lines.extend(child_lines)
        else:
            # No children, just render parameters and returns
            if node.arg_variable_ids:
                param_lines = self._format_variables(
                    node.arg_variable_ids, "Parameters"
                )
                for i, line in enumerate(param_lines):
                    if i == len(param_lines) - 1 and not node.return_variable_ids:
                        lines.append(f"{indent}   └─ {line}")
                    else:
                        lines.append(f"{indent}   ├─ {line}")

            if node.return_variable_ids:
                return_lines = self._format_variables(
                    node.return_variable_ids, "Returns"
                )
                for i, line in enumerate(return_lines):
                    if i == len(return_lines) - 1:
                        lines.append(f"{indent}   └─ {line}")
                    else:
                        lines.append(f"{indent}   ├─ {line}")

        return lines

    def _render_node(self, node: FunctionCallNode, indent: str) -> str:
        """Render a single function call node.

        Args:
            node: Node to render
            indent: Current indentation

        Returns:
            Formatted string for the node
        """
        # Extract filename (just the basename)
        import os

        filename = os.path.basename(node.filename)

        # Format: function_name() [file.py:line]
        return f"{indent}├─ {node.display_name}() [{filename}:{node.lineno}]"

    def _format_variables(self, variable_ids: list[str], label: str) -> list[str]:
        """Format a list of variables.

        Args:
            variable_ids: List of variable IDs
            label: Label for the variable group (e.g., "Parameters", "Returns")

        Returns:
            List of formatted lines
        """
        if not variable_ids:
            return []

        lines = []
        vars_to_show = [self.var_map.get(vid) for vid in variable_ids]
        vars_to_show = [v for v in vars_to_show if v is not None]

        if not vars_to_show:
            return []

        # Format each variable
        var_strs = []
        for var in vars_to_show:
            var_str = self._format_variable(var)
            var_strs.append(var_str)

        # Combine into a single line if possible
        if len(var_strs) == 1:
            lines.append(f"{label}: {var_strs[0]}")
        else:
            lines.append(f"{label}:")
            for var_str in var_strs:
                lines.append(f"  {var_str}")

        return lines

    def _format_variable(self, var: VariableRecord) -> str:
        """Format a single variable.

        Args:
            var: Variable to format

        Returns:
            Formatted string
        """
        # Start with name
        parts = [var.display_name]

        # Add type and tensor metadata
        if var.tensor_meta:
            tensor_str = self._format_tensor_meta(var.tensor_meta)
            parts.append(tensor_str)
        else:
            # Just show type name
            parts.append(f"({var.type_name})")

        return ": ".join(parts)

    def _format_tensor_meta(self, meta: TensorMeta) -> str:
        """Format tensor metadata.

        Args:
            meta: TensorMeta to format

        Returns:
            Formatted string like "Tensor[2, 3] (float32, cpu)"
        """
        parts = []

        # Shape
        if meta.shape is not None:
            shape_str = ", ".join(str(d) for d in meta.shape)
            parts.append(f"Tensor[{shape_str}]")
        else:
            parts.append("Tensor")

        # Dtype and device
        details = []
        if meta.dtype:
            # Simplify dtype (remove 'torch.' prefix)
            dtype = meta.dtype.replace("torch.", "")
            details.append(dtype)
        if meta.device:
            details.append(str(meta.device))

        if details:
            parts.append(f"({', '.join(details)})")

        return " ".join(parts)
