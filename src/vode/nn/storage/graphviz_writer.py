"""Graphviz writer for VODE neural network visualization.

This module converts StructureGraph and DataflowGraph objects into
Graphviz DOT format (.gv files) for visualization.
"""

import html
import json
from pathlib import Path

from vode.nn.graph.builder import DataflowGraph, StructureGraph
from vode.nn.graph.nodes import FunctionNode, ModuleNode, TensorNode


class GraphvizWriter:
    """Writer for converting graphs to Graphviz DOT format.

    Generates .gv files with HTML-like labels for rich node visualization
    and custom vode_* attributes for metadata preservation.
    """

    # Node colors by type
    NODE_COLORS = {
        "tensor": "lightyellow",
        "module": "darkseagreen1",
        "function": "aliceblue",
    }

    def __init__(self) -> None:
        """Initialize the Graphviz writer."""
        pass

    def _sanitize_node_id(self, node_id: str) -> str:
        """Sanitize node ID to only contain valid Graphviz characters.

        Args:
            node_id: Original node ID

        Returns:
            Sanitized node ID with only alphanumeric and underscores
        """
        # Replace invalid characters with underscores
        sanitized = node_id.replace(".", "_").replace("/", "_").replace("-", "_")
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
        return sanitized

    def write_structure_graph(self, graph: StructureGraph, output_path: str) -> None:
        """Write a structure graph to a Graphviz .gv file.

        Args:
            graph: StructureGraph object to write
            output_path: Path to output .gv file

        Example:
            >>> writer = GraphvizWriter()
            >>> writer.write_structure_graph(graph, "model_structure.gv")
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # Write graph header - let Graphviz auto-size
            f.write("strict digraph StructureGraph {\n")
            f.write("    graph [ordering=in rankdir=LR]\n")
            f.write("    node [style=filled shape=plaintext fontsize=10]\n")
            f.write("    edge [fontsize=10]\n\n")

            # Write nodes
            for node in graph.get_nodes():
                self._write_node(f, node)

            # Write edges
            f.write("\n")
            for edge in graph.get_edges():
                self._write_edge(f, edge)

            f.write("}\n")

    def write_dataflow_graph(self, graph: DataflowGraph, output_path: str) -> None:
        """Write a dataflow graph to a Graphviz .gv file.

        Args:
            graph: DataflowGraph object to write
            output_path: Path to output .gv file

        Example:
            >>> writer = GraphvizWriter()
            >>> writer.write_dataflow_graph(graph, "model_dataflow.gv")
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # Check for loop groups (Strategy B: Z-layout)
            loop_groups = getattr(graph, "loop_groups", [])
            loop_components = getattr(graph, "loop_components", {})

            # Build node-to-iteration mapping for Z-layout
            node_to_iteration = {}
            iteration_groups = {}
            if loop_groups:
                for group_indices, is_same_shape in loop_groups:
                    if not is_same_shape:  # Strategy B: different shapes
                        comp_list = list(loop_components.keys())
                        for iter_idx, comp_idx in enumerate(group_indices):
                            comp_root = comp_list[comp_idx]
                            comp_nodes = loop_components[comp_root]
                            if iter_idx not in iteration_groups:
                                iteration_groups[iter_idx] = []
                            for node_id in comp_nodes:
                                node_to_iteration[node_id] = iter_idx
                                iteration_groups[iter_idx].append(node_id)

            # Write graph header - always use LR, use ortho splines for cleaner Z-layout
            f.write("strict digraph DataflowGraph {\n")
            f.write("    graph [ordering=in rankdir=LR newrank=true splines=ortho]\n")
            f.write("    node [style=filled shape=plaintext fontsize=10]\n")
            f.write("    edge [fontsize=10]\n\n")

            # Write all nodes
            for node in graph.get_nodes():
                self._write_node(f, node)

            # Add group attributes to iteration nodes for horizontal alignment
            if iteration_groups and len(iteration_groups) > 1:
                f.write("\n    // Group attributes for horizontal alignment\n")
                for iter_idx in sorted(iteration_groups.keys()):
                    if iteration_groups[iter_idx]:
                        for node_id in iteration_groups[iter_idx]:
                            sanitized_id = self._sanitize_node_id(node_id)
                            f.write(f"    {sanitized_id} [group=g{iter_idx}]\n")

            # Add invisible anchor nodes for Z-layout vertical alignment
            if iteration_groups and len(iteration_groups) > 1:
                f.write("\n    // Invisible anchor nodes for Z-layout\n")
                sorted_iters = sorted(iteration_groups.keys())
                for iter_idx in sorted_iters:
                    f.write(
                        f"    anchor_{iter_idx} [style=invis shape=point width=0 height=0]\n"
                    )

                # Chain anchors vertically with consistent weight and exact length
                for i in range(len(sorted_iters) - 1):
                    f.write(
                        f"    anchor_{sorted_iters[i]} -> anchor_{sorted_iters[i+1]} [style=invis weight=100 len=1]\n"
                    )

                # Put each anchor with its first node in separate rank=same constraints
                f.write("\n")
                for iter_idx in sorted_iters:
                    if iteration_groups[iter_idx]:
                        first_node = self._sanitize_node_id(
                            iteration_groups[iter_idx][0]
                        )
                        f.write(f"    {{rank=same; anchor_{iter_idx}; {first_node}}}\n")

            # Write edges with constraint=false for cross-iteration edges in Z-layout
            f.write("\n")

            # Build set of cross-iteration edges
            cross_iteration_edges = set()
            if iteration_groups and len(iteration_groups) > 1:
                # Identify edges that connect different iterations
                for edge in graph.get_edges():
                    src_iter = node_to_iteration.get(edge.src_id)
                    dst_iter = node_to_iteration.get(edge.dst_id)
                    if (
                        src_iter is not None
                        and dst_iter is not None
                        and src_iter != dst_iter
                    ):
                        cross_iteration_edges.add((edge.src_id, edge.dst_id))

            for edge in graph.get_edges():
                is_cross_iter = (edge.src_id, edge.dst_id) in cross_iteration_edges
                self._write_edge(f, edge, constraint=not is_cross_iter)

            f.write("}\n")

    def _write_node(self, f, node, indent="    ") -> None:
        """Write a single node to the file.

        Args:
            f: File object to write to
            node: Node object (TensorNode, ModuleNode, or FunctionNode)
            indent: Indentation string for the node
        """
        if isinstance(node, TensorNode):
            self._write_tensor_node(f, node, indent)
        elif isinstance(node, ModuleNode):
            self._write_module_node(f, node, indent)
        elif isinstance(node, FunctionNode):
            self._write_function_node(f, node, indent)
        else:
            # Fallback for base Node
            self._write_generic_node(f, node, indent)

    def _write_tensor_node(self, f, node: TensorNode, indent="    ") -> None:
        """Write a TensorNode with simple table format.

        Args:
            f: File object to write to
            node: TensorNode object
            indent: Indentation string
        """
        # Sanitize node ID
        sanitized_id = self._sanitize_node_id(node.node_id)

        # Build label
        name = self._escape_html(node.name)
        shape_str = str(node.shape) if node.shape else "unknown"

        label = f"""<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR><TD>{name}<BR/>{shape_str}</TD></TR>
        </TABLE>
    >"""

        # Build attributes
        attrs = [f'fillcolor={self.NODE_COLORS["tensor"]}']
        attrs.append('vode_type="tensor"')
        attrs.append(f'vode_depth="{node.depth}"')

        if node.tensor_id:
            attrs.append(f'vode_tensor_id="{self._escape_attr(node.tensor_id)}"')
        if node.shape:
            attrs.append(f'vode_shape="{self._escape_attr(str(node.shape))}"')
        if node.dtype:
            attrs.append(f'vode_dtype="{self._escape_attr(node.dtype)}"')
        if node.device:
            attrs.append(f'vode_device="{self._escape_attr(node.device)}"')
        if node.stats:
            stats_json = json.dumps(node.stats)
            attrs.append(f'vode_stats="{self._escape_attr(stats_json)}"')

        # Write node with sanitized ID
        f.write(f"{indent}{sanitized_id} [label={label} {' '.join(attrs)}]\n")

    def _write_module_node(self, f, node: ModuleNode, indent="    ") -> None:
        """Write a ModuleNode with HTML table format (INPUT | OP | OUTPUT).

        Args:
            f: File object to write to
            node: ModuleNode object
            indent: Indentation string
        """
        # Sanitize node ID
        sanitized_id = self._sanitize_node_id(node.node_id)

        # Build operation info
        op_name = self._escape_html(node.module_type or node.name)
        op_info = f"{op_name}<BR/>depth: {node.depth}"

        # Build shape info
        input_shape_str = ""
        output_shape_str = ""

        if node.input_shapes:
            if len(node.input_shapes) == 1:
                input_shape_str = str(node.input_shapes[0])
            else:
                input_shape_str = f"{len(node.input_shapes)}x"

        if node.output_shapes:
            if len(node.output_shapes) == 1:
                output_shape_str = str(node.output_shapes[0])
            else:
                output_shape_str = f"{len(node.output_shapes)}x"

        # Build HTML label - simple 3-column layout per spec
        label = f"""<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR>
            <TD BGCOLOR="#F0F0F0">input<BR/>{input_shape_str}</TD>
            <TD>{op_info}</TD>
            <TD BGCOLOR="#F0F0F0">output<BR/>{output_shape_str}</TD>
        </TR>
        </TABLE>
    >"""

        # Build attributes
        attrs = [f'fillcolor={self.NODE_COLORS["module"]}']
        attrs.append('vode_type="module"')
        attrs.append(f'vode_depth="{node.depth}"')

        if node.module_type:
            attrs.append(f'vode_module_type="{self._escape_attr(node.module_type)}"')
        if node.input_shapes:
            attrs.append(
                f'vode_input_shapes="{self._escape_attr(str(node.input_shapes))}"'
            )
        if node.output_shapes:
            attrs.append(
                f'vode_output_shapes="{self._escape_attr(str(node.output_shapes))}"'
            )
        if node.params:
            params_json = json.dumps(node.params)
            attrs.append(f'vode_params="{self._escape_attr(params_json)}"')

        # Write node with sanitized ID
        f.write(f"{indent}{sanitized_id} [label={label} {' '.join(attrs)}]\n")

    def _write_function_node(self, f, node: FunctionNode, indent="    ") -> None:
        """Write a FunctionNode with HTML table format (INPUT | OP | OUTPUT).

        Args:
            f: File object to write to
            node: FunctionNode object
            indent: Indentation string
        """
        # Sanitize node ID
        sanitized_id = self._sanitize_node_id(node.node_id)

        # Build operation info
        op_name = self._escape_html(node.func_name or node.name)
        op_info = f"{op_name}<BR/>depth: {node.depth}"

        # Build shape info
        input_shape_str = ""
        output_shape_str = ""

        if node.input_shapes:
            if len(node.input_shapes) == 1:
                input_shape_str = str(node.input_shapes[0])
            else:
                input_shape_str = f"{len(node.input_shapes)}x"

        if node.output_shapes:
            if len(node.output_shapes) == 1:
                output_shape_str = str(node.output_shapes[0])
            else:
                output_shape_str = f"{len(node.output_shapes)}x"

        # Build HTML label - simple 3-column layout per spec
        label = f"""<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
        <TR>
            <TD BGCOLOR="#F0F0F0">input<BR/>{input_shape_str}</TD>
            <TD>{op_info}</TD>
            <TD BGCOLOR="#F0F0F0">output<BR/>{output_shape_str}</TD>
        </TR>
        </TABLE>
    >"""

        # Build attributes
        attrs = [f'fillcolor={self.NODE_COLORS["function"]}']
        attrs.append('vode_type="function"')
        attrs.append(f'vode_depth="{node.depth}"')

        if node.func_name:
            attrs.append(f'vode_func_name="{self._escape_attr(node.func_name)}"')
        if node.input_shapes:
            attrs.append(
                f'vode_input_shapes="{self._escape_attr(str(node.input_shapes))}"'
            )
        if node.output_shapes:
            attrs.append(
                f'vode_output_shapes="{self._escape_attr(str(node.output_shapes))}"'
            )
        if node.metadata:
            metadata_json = json.dumps(node.metadata)
            attrs.append(f'vode_metadata="{self._escape_attr(metadata_json)}"')

        # Write node with sanitized ID
        f.write(f"{indent}{sanitized_id} [label={label} {' '.join(attrs)}]\n")

    def _write_generic_node(self, f, node, indent="    ") -> None:
        """Write a generic Node (fallback).

        Args:
            f: File object to write to
            node: Node object
            indent: Indentation string
        """
        # Sanitize node ID
        sanitized_id = self._sanitize_node_id(node.node_id)

        name = self._escape_html(node.name)
        label = f'"{name}\\ndepth: {node.depth}"'

        attrs = ["fillcolor=lightgray"]
        attrs.append('vode_type="generic"')
        attrs.append(f'vode_depth="{node.depth}"')

        f.write(f"{indent}{sanitized_id} [label={label} {' '.join(attrs)}]\n")

    def _write_edge(self, f, edge, constraint=True) -> None:
        """Write a single edge to the file.

        Args:
            f: File object to write to
            edge: Edge object
            constraint: Whether this edge should constrain node positions (default True)
        """
        # Sanitize node IDs
        sanitized_src = self._sanitize_node_id(edge.src_id)
        sanitized_dst = self._sanitize_node_id(edge.dst_id)

        # Write edge with optional constraint attribute
        if not constraint:
            f.write(f"    {sanitized_src} -> {sanitized_dst} [constraint=false]\n")
        else:
            f.write(f"    {sanitized_src} -> {sanitized_dst}\n")

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters for use in HTML labels.

        Args:
            text: Text to escape

        Returns:
            Escaped text safe for HTML labels
        """
        return html.escape(text)

    def _escape_attr(self, text: str) -> str:
        """Escape special characters for use in Graphviz attributes.

        Args:
            text: Text to escape

        Returns:
            Escaped text safe for attribute values
        """
        # Escape quotes and backslashes
        return text.replace("\\", "\\\\").replace('"', '\\"')
