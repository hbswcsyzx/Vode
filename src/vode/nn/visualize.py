"""High-level API for VODE neural network visualization.

This module provides a simple user-facing interface for visualizing PyTorch models
by integrating capture, storage, and rendering components.
"""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from vode.nn.capture.structure_capture import StructureCapture
from vode.nn.capture.dataflow_capture import DataflowCapture
from vode.nn.capture.recorder_tensor import RecorderTensor
from vode.nn.storage.graphviz_writer import GraphvizWriter
from vode.nn.render.static_renderer import StaticRenderer


def visualize_model(
    model: nn.Module,
    input_data: torch.Tensor | tuple | dict,
    save_path: str = "model_viz",
    format: Literal["svg", "png", "pdf", "gv"] = "svg",
    graph_type: Literal["structure", "dataflow", "both"] = "both",
    depth_limit: int | None = None,
) -> dict[str, str]:
    """Visualize a PyTorch model's structure and/or dataflow.

    This function provides a simple interface to generate visualizations of PyTorch
    models. It can create structure graphs (showing module hierarchy) and/or dataflow
    graphs (showing tensor flow during forward pass).

    Args:
        model: PyTorch model to visualize
        input_data: Sample input for forward pass (required for dataflow graphs)
        save_path: Base path for output files (without extension)
        format: Output format - 'svg', 'png', 'pdf', or 'gv' (Graphviz source only)
        graph_type: Which graph(s) to generate - 'structure', 'dataflow', or 'both'
        depth_limit: Maximum depth to visualize (None = all levels). Not yet implemented.

    Returns:
        Dictionary mapping graph type to output file path.
        Example: {"structure": "model_viz_structure.svg", "dataflow": "model_viz_dataflow.svg"}

    Raises:
        ValueError: If model is None or graph_type is invalid
        RuntimeError: If Graphviz is not installed (when format != 'gv')

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from vode.nn import visualize_model
        >>>
        >>> # Create a simple model
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 5),
        ...     nn.ReLU(),
        ...     nn.Linear(5, 2)
        ... )
        >>>
        >>> # Generate both graphs as SVG
        >>> input_data = torch.randn(1, 10)
        >>> paths = visualize_model(
        ...     model,
        ...     input_data,
        ...     save_path="my_model",
        ...     format="svg",
        ...     graph_type="both"
        ... )
        >>> print(paths)
        {'structure': 'my_model_structure.svg', 'dataflow': 'my_model_dataflow.svg'}

        >>> # Generate only structure graph as PNG
        >>> paths = visualize_model(
        ...     model,
        ...     input_data,
        ...     save_path="my_model",
        ...     format="png",
        ...     graph_type="structure"
        ... )
        >>> print(paths)
        {'structure': 'my_model_structure.png'}

        >>> # Generate Graphviz source files only (no rendering)
        >>> paths = visualize_model(
        ...     model,
        ...     input_data,
        ...     save_path="my_model",
        ...     format="gv",
        ...     graph_type="both"
        ... )
        >>> print(paths)
        {'structure': 'my_model_structure.gv', 'dataflow': 'my_model_dataflow.gv'}
    """
    # Validate inputs
    if model is None:
        raise ValueError("Model cannot be None")

    if graph_type not in {"structure", "dataflow", "both"}:
        raise ValueError(
            f"Invalid graph_type '{graph_type}'. "
            "Must be 'structure', 'dataflow', or 'both'"
        )

    # Create output directory if needed
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Initialize components
    writer = GraphvizWriter()
    renderer = None
    if format != "gv":
        renderer = StaticRenderer()

    result_paths: dict[str, str] = {}

    # Generate structure graph
    if graph_type in {"structure", "both"}:
        structure_path = _generate_structure_graph(
            model=model,
            save_path=save_path,
            format=format,
            writer=writer,
            renderer=renderer,
            depth_limit=depth_limit,
        )
        result_paths["structure"] = structure_path

    # Generate dataflow graph
    if graph_type in {"dataflow", "both"}:
        dataflow_path = _generate_dataflow_graph(
            model=model,
            input_data=input_data,
            save_path=save_path,
            format=format,
            writer=writer,
            renderer=renderer,
            depth_limit=depth_limit,
        )
        result_paths["dataflow"] = dataflow_path

    return result_paths


def _generate_structure_graph(
    model: nn.Module,
    save_path: str,
    format: str,
    writer: GraphvizWriter,
    renderer: StaticRenderer | None,
    depth_limit: int | None,
) -> str:
    """Generate structure graph for a model.

    Args:
        model: PyTorch model to visualize
        save_path: Base path for output files
        format: Output format
        writer: GraphvizWriter instance
        renderer: StaticRenderer instance (or None if format='gv')
        depth_limit: Maximum depth to visualize (not yet implemented)

    Returns:
        Path to generated file
    """
    # Capture structure
    capturer = StructureCapture()
    graph = capturer.capture(model)

    # TODO: Apply depth_limit filtering if specified
    if depth_limit is not None:
        # Filter nodes by depth
        pass

    # Write to .gv file
    gv_path = f"{save_path}_structure.gv"
    writer.write_structure_graph(graph, gv_path)

    # Render if needed
    if format == "gv":
        return gv_path
    else:
        output_path = f"{save_path}_structure.{format}"
        renderer.render(gv_path, output_path, format=format)
        return output_path


def _generate_dataflow_graph(
    model: nn.Module,
    input_data: torch.Tensor | tuple | dict,
    save_path: str,
    format: str,
    writer: GraphvizWriter,
    renderer: StaticRenderer | None,
    depth_limit: int | None,
) -> str:
    """Generate dataflow graph for a model.

    Args:
        model: PyTorch model to visualize
        input_data: Sample input for forward pass
        save_path: Base path for output files
        format: Output format
        writer: GraphvizWriter instance
        renderer: StaticRenderer instance (or None if format='gv')
        depth_limit: Maximum depth to visualize (not yet implemented)

    Returns:
        Path to generated file
    """
    # Wrap input data as RecorderTensor
    wrapped_input = _wrap_input_data(input_data)

    # Capture dataflow
    with DataflowCapture(model) as capture:
        output = model(wrapped_input)
        graph = capture.get_graph()

        # Filter by depth - show only the deepest level by default
        if depth_limit is None:
            # Find the maximum depth
            all_nodes = graph.get_nodes()
            if all_nodes:
                max_depth = max(node.depth for node in all_nodes)
                # Keep only nodes at max depth, plus input/output tensors
                from vode.nn.graph.nodes import TensorNode, ModuleNode

                filtered_nodes = {}
                filtered_edges = []

                for node in all_nodes:
                    if isinstance(node, TensorNode) and node.name in [
                        "input",
                        "output",
                    ]:
                        # Keep input/output tensors
                        filtered_nodes[node.node_id] = node
                    elif isinstance(node, ModuleNode) and node.depth == max_depth:
                        # Keep modules at max depth
                        filtered_nodes[node.node_id] = node

                # Keep only edges between filtered nodes
                for edge in graph.get_edges():
                    if edge.src_id in filtered_nodes and edge.dst_id in filtered_nodes:
                        filtered_edges.append(edge)

                # Replace graph nodes and edges
                graph.nodes = filtered_nodes
                graph.edges = filtered_edges
        else:
            # Filter by specified depth
            from vode.nn.graph.nodes import TensorNode, ModuleNode

            filtered_nodes = {}
            filtered_edges = []

            for node in graph.get_nodes():
                if isinstance(node, TensorNode) and node.name in ["input", "output"]:
                    filtered_nodes[node.node_id] = node
                elif isinstance(node, ModuleNode) and node.depth == depth_limit:
                    filtered_nodes[node.node_id] = node

            for edge in graph.get_edges():
                if edge.src_id in filtered_nodes and edge.dst_id in filtered_nodes:
                    filtered_edges.append(edge)

            graph.nodes = filtered_nodes
            graph.edges = filtered_edges

        # Add final output tensor node
        from vode.nn.graph.nodes import TensorNode

        if isinstance(output, torch.Tensor):
            output_node_id = f"tensor_output_{id(output)}"
            output_node = TensorNode(
                node_id=output_node_id,
                name="output",
                depth=0,
                tensor_id=str(id(output)),
                shape=tuple(torch.Tensor.size(output)),
                dtype=str(torch.Tensor.dtype.__get__(output)).replace("torch.", ""),
                device=str(torch.Tensor.device.__get__(output)),
            )
            try:
                graph.add_node(output_node)
                # Connect last module to output
                if hasattr(output, "parent_module_node"):
                    parent_id = output.parent_module_node.node_id
                    if parent_id in graph.nodes:
                        graph.add_edge(src_id=parent_id, dst_id=output_node_id)
            except (ValueError, AttributeError):
                pass

    # Write to .gv file
    gv_path = f"{save_path}_dataflow.gv"
    writer.write_dataflow_graph(graph, gv_path)

    # Render if needed
    if format == "gv":
        return gv_path
    else:
        output_path = f"{save_path}_dataflow.{format}"
        renderer.render(gv_path, output_path, format=format)
        return output_path


def _wrap_input_data(
    input_data: torch.Tensor | tuple | dict,
) -> torch.Tensor | tuple | dict:
    """Wrap input data tensors as RecorderTensors.

    Args:
        input_data: Input data to wrap

    Returns:
        Input data with tensors wrapped as RecorderTensors
    """
    if isinstance(input_data, torch.Tensor):
        # Wrap single tensor
        rec_tensor = input_data.as_subclass(RecorderTensor)
        # Initialize tensor_nodes attribute
        rec_tensor.tensor_nodes = []
        return rec_tensor
    elif isinstance(input_data, tuple):
        # Wrap tuple of tensors
        return tuple(_wrap_input_data(item) for item in input_data)
    elif isinstance(input_data, dict):
        # Wrap dict of tensors
        return {key: _wrap_input_data(value) for key, value in input_data.items()}
    else:
        # Return as-is for non-tensor data
        return input_data
