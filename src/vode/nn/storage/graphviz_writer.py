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
            # Write graph header
            f.write("strict digraph StructureGraph {\n")
            f.write('    graph [ordering=in rankdir=LR size="20,20"]\n')
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
            # Write graph header
            f.write("strict digraph DataflowGraph {\n")
            f.write('    graph [ordering=in rankdir=LR size="20,20"]\n')
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

    def _write_node(self, f, node) -> None:
        """Write a single node to the file.

        Args:
            f: File object to write to
            node: Node object (TensorNode, ModuleNode, or FunctionNode)
        """
        if isinstance(node, TensorNode):
            self._write_tensor_node(f, node)
        elif isinstance(node, ModuleNode):
            self._write_module_node(f, node)
        elif isinstance(node, FunctionNode):
            self._write_function_node(f, node)
        else:
            # Fallback for base Node
            self._write_generic_node(f, node)

    def _write_tensor_node(self, f, node: TensorNode) -> None:
        """Write a TensorNode with simple table format.

        Args:
            f: File object to write to
            node: TensorNode object
        """
        # Build label
        name = self._escape_html(node.name)
        shape_str = str(node.shape) if node.shape else "unknown"

        label = f"""<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
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

        # Write node
        f.write(f"    {node.node_id} [label={label} {' '.join(attrs)}]\n")

    def _write_module_node(self, f, node: ModuleNode) -> None:
        """Write a ModuleNode with HTML table format (INPUT | OP | OUTPUT).

        Args:
            f: File object to write to
            node: ModuleNode object
        """
        # Build input/output port symbols
        num_inputs = len(node.input_shapes) if node.input_shapes else 1
        num_outputs = len(node.output_shapes) if node.output_shapes else 1

        input_ports = "".join(["●<BR/>" for _ in range(num_inputs)])
        output_ports = "".join(["●<BR/>" for _ in range(num_outputs)])

        # Remove trailing <BR/>
        if input_ports:
            input_ports = input_ports[:-5]
        if output_ports:
            output_ports = output_ports[:-5]

        # Build operation info
        op_name = self._escape_html(node.module_type or node.name)
        op_info = f"{op_name}<BR/>depth: {node.depth}"

        # Build shape info
        input_shape_str = ""
        output_shape_str = ""

        if node.input_shapes:
            if len(node.input_shapes) == 1:
                input_shape_str = f"in: {node.input_shapes[0]}"
            else:
                input_shape_str = f"in: {len(node.input_shapes)} tensors"

        if node.output_shapes:
            if len(node.output_shapes) == 1:
                output_shape_str = f"out: {node.output_shapes[0]}"
            else:
                output_shape_str = f"out: {len(node.output_shapes)} tensors"

        # Build HTML label
        if input_ports or output_ports:
            # With port symbols
            label = f"""<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
        <TR>
            <TD ROWSPAN="2" BGCOLOR="#E8E8E8">{input_ports if input_ports else ""}</TD>
            <TD COLSPAN="2">{op_info}</TD>
            <TD ROWSPAN="2" BGCOLOR="#E8E8E8">{output_ports if output_ports else ""}</TD>
        </TR>
        <TR>
            <TD>{input_shape_str}</TD>
            <TD>{output_shape_str}</TD>
        </TR>
        </TABLE>
    >"""
        else:
            # Without port symbols (fallback)
            label = f"""<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
        <TR>
            <TD ROWSPAN="2" BGCOLOR="#E8E8E8">INPUT</TD>
            <TD COLSPAN="2">{op_info}</TD>
            <TD ROWSPAN="2" BGCOLOR="#E8E8E8">OUTPUT</TD>
        </TR>
        <TR>
            <TD>{input_shape_str}</TD>
            <TD>{output_shape_str}</TD>
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

        # Write node
        f.write(f"    {node.node_id} [label={label} {' '.join(attrs)}]\n")

    def _write_function_node(self, f, node: FunctionNode) -> None:
        """Write a FunctionNode with HTML table format (INPUT | OP | OUTPUT).

        Args:
            f: File object to write to
            node: FunctionNode object
        """
        # Build input/output port symbols
        num_inputs = len(node.input_shapes) if node.input_shapes else 1
        num_outputs = len(node.output_shapes) if node.output_shapes else 1

        input_ports = "".join(["●<BR/>" for _ in range(num_inputs)])
        output_ports = "".join(["●<BR/>" for _ in range(num_outputs)])

        # Remove trailing <BR/>
        if input_ports:
            input_ports = input_ports[:-5]
        if output_ports:
            output_ports = output_ports[:-5]

        # Build operation info
        op_name = self._escape_html(node.func_name or node.name)
        op_info = f"{op_name}<BR/>depth: {node.depth}"

        # Build shape info
        input_shape_str = ""
        output_shape_str = ""

        if node.input_shapes:
            if len(node.input_shapes) == 1:
                input_shape_str = f"in: {node.input_shapes[0]}"
            else:
                input_shape_str = f"in: {len(node.input_shapes)} tensors"

        if node.output_shapes:
            if len(node.output_shapes) == 1:
                output_shape_str = f"out: {node.output_shapes[0]}"
            else:
                output_shape_str = f"out: {len(node.output_shapes)} tensors"

        # Build HTML label
        if input_ports or output_ports:
            # With port symbols
            label = f"""<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
        <TR>
            <TD ROWSPAN="2" BGCOLOR="#E8E8E8">{input_ports if input_ports else ""}</TD>
            <TD COLSPAN="2">{op_info}</TD>
            <TD ROWSPAN="2" BGCOLOR="#E8E8E8">{output_ports if output_ports else ""}</TD>
        </TR>
        <TR>
            <TD>{input_shape_str}</TD>
            <TD>{output_shape_str}</TD>
        </TR>
        </TABLE>
    >"""
        else:
            # Without port symbols (fallback)
            label = f"""<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
        <TR>
            <TD ROWSPAN="2" BGCOLOR="#E8E8E8">INPUT</TD>
            <TD COLSPAN="2">{op_info}</TD>
            <TD ROWSPAN="2" BGCOLOR="#E8E8E8">OUTPUT</TD>
        </TR>
        <TR>
            <TD>{input_shape_str}</TD>
            <TD>{output_shape_str}</TD>
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

        # Write node
        f.write(f"    {node.node_id} [label={label} {' '.join(attrs)}]\n")

    def _write_generic_node(self, f, node) -> None:
        """Write a generic Node (fallback).

        Args:
            f: File object to write to
            node: Node object
        """
        name = self._escape_html(node.name)
        label = f'"{name}\\ndepth: {node.depth}"'

        attrs = ["fillcolor=lightgray"]
        attrs.append('vode_type="generic"')
        attrs.append(f'vode_depth="{node.depth}"')

        f.write(f"    {node.node_id} [label={label} {' '.join(attrs)}]\n")

    def _write_edge(self, f, edge) -> None:
        """Write a single edge to the file.

        Args:
            f: File object to write to
            edge: Edge object
        """
        if edge.label:
            label = self._escape_attr(edge.label)
            f.write(f'    {edge.src_id} -> {edge.dst_id} [label="{label}"]\n')
        else:
            f.write(f"    {edge.src_id} -> {edge.dst_id}\n")

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
