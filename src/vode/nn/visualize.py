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
    debug: bool = False,
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
        depth_limit: Depth level to visualize (None = deepest level). When None, shows
            the maximum depth level across all branches. When set to N, shows level N.
        debug: Enable debug output showing edge creation process

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
            debug=debug,
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


def _compare_component_structure(comp1_nodes, comp2_nodes, graph, filtered_edges):
    """Compare if two components have the same structure.

    Returns:
        (is_same_structure, is_same_shapes) tuple
    """
    from vode.nn.graph.nodes import ModuleNode

    if len(comp1_nodes) != len(comp2_nodes):
        return False, False

    # Get node types in order for both components
    def get_ordered_types_and_shapes(nodes, edges):
        # Find start node (no incoming edges)
        start = None
        for n in nodes:
            if not any(e.dst_id == n for e in edges):
                start = n
                break
        if not start:
            return [], []

        # Follow the chain
        types = []
        shapes = []
        current = start
        visited = set()
        while current and current not in visited:
            visited.add(current)
            node = graph.get_node(current)
            if node and isinstance(node, ModuleNode):
                types.append(node.module_type)
                shapes.append((node.input_shapes, node.output_shapes))
            # Find next
            next_nodes = [
                e.dst_id for e in edges if e.src_id == current and e.dst_id in nodes
            ]
            current = next_nodes[0] if next_nodes else None
        return types, shapes

    types1, shapes1 = get_ordered_types_and_shapes(comp1_nodes, filtered_edges)
    types2, shapes2 = get_ordered_types_and_shapes(comp2_nodes, filtered_edges)

    is_same_structure = types1 == types2
    is_same_shapes = shapes1 == shapes2

    return is_same_structure, is_same_shapes


def _detect_loop_patterns(
    comp_list, components, comp_to_parent, graph, filtered_edges, debug
):
    """Detect consecutive repeated patterns at parent module level.

    Returns:
        List of (group_indices, is_same_shape) tuples
    """
    loop_groups = []
    i = 0

    while i < len(comp_list):
        group = [i]

        # Check consecutive components for same structure
        j = i + 1
        while j < len(comp_list):
            is_same_struct, is_same_shape = _compare_component_structure(
                components[comp_list[i]],
                components[comp_list[j]],
                graph,
                filtered_edges,
            )

            # Also check parent module type if available
            parent_i = comp_to_parent.get(comp_list[i])
            parent_j = comp_to_parent.get(comp_list[j])
            same_parent_type = False
            if parent_i and parent_j:
                # Get module type from graph nodes
                parent_node_i = graph.get_node(parent_i)
                parent_node_j = graph.get_node(parent_j)
                if parent_node_i and parent_node_j:
                    same_parent_type = (
                        parent_node_i.module_type == parent_node_j.module_type
                    )

            if debug:
                print(
                    f"[DEBUG] Comparing comp {i} vs {j}: struct={is_same_struct}, shape={is_same_shape}, parent_type={same_parent_type}"
                )

            if is_same_struct and same_parent_type:
                group.append(j)
                j += 1
            else:
                break

        if len(group) > 1:
            # Found a loop pattern
            _, is_same_shape = _compare_component_structure(
                components[comp_list[group[0]]],
                components[comp_list[group[1]]],
                graph,
                filtered_edges,
            )
            loop_groups.append((group, is_same_shape))
            if debug:
                print(
                    f"[DEBUG] Found loop: indices {group}, same_shape={is_same_shape}"
                )

        i = j if j > i + 1 else i + 1

    return loop_groups


def _apply_loop_visualization(
    loop_groups, components, comp_list, filtered_nodes, filtered_edges, graph, debug
):
    """Apply visualization strategies for detected loops.

    Strategy A (same shapes): Collapse to single component with loop edge
    Strategy B (different shapes): Keep all components in Z-layout
    """
    from vode.nn.graph.builder import Edge

    nodes_to_remove = set()
    edges_to_add = []

    for group_indices, is_same_shape in loop_groups:
        if is_same_shape:
            # Strategy A: Collapse identical loops
            if debug:
                print(
                    f"[DEBUG] Applying Strategy A (collapse) for group {group_indices}"
                )

            # Keep only first component
            first_comp_root = comp_list[group_indices[0]]
            first_comp_nodes = components[first_comp_root]

            # Find first and last nodes in first component
            first_node = None
            last_node = None
            for node_id in first_comp_nodes:
                if not any(
                    e.dst_id == node_id
                    for e in filtered_edges
                    if e.src_id in first_comp_nodes and e.dst_id in first_comp_nodes
                ):
                    first_node = node_id
                if not any(
                    e.src_id == node_id
                    for e in filtered_edges
                    if e.src_id in first_comp_nodes and e.dst_id in first_comp_nodes
                ):
                    last_node = node_id

            # Remove other components in the loop
            for idx in group_indices[1:]:
                comp_root = comp_list[idx]
                for node_id in components[comp_root]:
                    nodes_to_remove.add(node_id)

            # Add loop edge with "xN" label
            if first_node and last_node:
                loop_count = len(group_indices)
                edges_to_add.append(
                    Edge(src_id=last_node, dst_id=first_node, label=f"x{loop_count}")
                )
                if debug:
                    print(
                        f"[DEBUG] Added loop edge: {last_node} -> {first_node} (x{loop_count})"
                    )
        else:
            # Strategy B: Z-layout (keep all components, add rank attributes)
            if debug:
                print(
                    f"[DEBUG] Applying Strategy B (Z-layout) for group {group_indices}"
                )
            # For now, just keep all nodes - rank attributes would be added in GraphvizWriter

    # Remove nodes
    for node_id in nodes_to_remove:
        if node_id in filtered_nodes:
            del filtered_nodes[node_id]

    # Remove edges involving removed nodes
    filtered_edges = [
        e
        for e in filtered_edges
        if e.src_id not in nodes_to_remove and e.dst_id not in nodes_to_remove
    ]

    # Add new loop edges
    filtered_edges.extend(edges_to_add)

    return filtered_nodes, filtered_edges


def _generate_dataflow_graph(
    model: nn.Module,
    input_data: torch.Tensor | tuple | dict,
    save_path: str,
    format: str,
    writer: GraphvizWriter,
    renderer: StaticRenderer | None,
    depth_limit: int | None,
    debug: bool = False,
) -> str:
    """Generate dataflow graph for a model.

    Args:
        model: PyTorch model to visualize
        input_data: Sample input for forward pass
        save_path: Base path for output files
        format: Output format
        writer: GraphvizWriter instance
        renderer: StaticRenderer instance (or None if format='gv')
        depth_limit: Depth level to visualize (None = deepest level)

    Returns:
        Path to generated file
    """
    # Wrap input data as RecorderTensor
    wrapped_input = _wrap_input_data(input_data)

    # Capture dataflow
    with DataflowCapture(model) as capture:
        output = model(wrapped_input)
        graph = capture.get_graph()

        # Filter by depth - show only one hierarchy level at a time
        from vode.nn.graph.nodes import TensorNode, ModuleNode

        all_nodes = graph.get_nodes()
        if not all_nodes:
            pass  # Empty graph, nothing to filter
        else:
            if debug:
                print(f"[DEBUG] Initial edges count: {len(graph.edges)}")
                for edge in graph.edges:
                    print(f"[DEBUG]   {edge.src_id} -> {edge.dst_id}")

            # Determine target depth
            if depth_limit is None:
                # Find max depth for each branch and show deepest level
                module_nodes = [n for n in all_nodes if isinstance(n, ModuleNode)]
                if module_nodes:
                    # Get max depth across all modules
                    target_depth = max(node.depth for node in module_nodes)
                else:
                    target_depth = 0
            else:
                target_depth = depth_limit

            # Filter nodes: keep only modules at target depth
            filtered_nodes = {}
            node_to_parent_map = {}  # Store parent info before clearing

            for node in all_nodes:
                if isinstance(node, ModuleNode) and node.depth == target_depth:
                    # Store parent info before clearing
                    if hasattr(node, "parents") and node.parents:
                        for parent_id in node.parents:
                            parent_node = graph.get_node(parent_id)
                            if (
                                parent_node
                                and isinstance(parent_node, ModuleNode)
                                and parent_node.depth == target_depth - 1
                            ):
                                node_to_parent_map[node.node_id] = parent_id
                                break

                    # Clear parent-child relationships to avoid edge duplication
                    node.parents = []
                    node.children = []
                    filtered_nodes[node.node_id] = node

            # Filter edges: keep only edges between modules at target depth
            filtered_edges = []
            for edge in graph.get_edges():
                if edge.src_id in filtered_nodes and edge.dst_id in filtered_nodes:
                    filtered_edges.append(edge)

            # Rebuild cross-parent connections using Union-Find
            if target_depth > 0:
                from vode.nn.graph.builder import Edge
                from vode.nn.graph.nodes import ModuleNode

                # Union-Find to find connected components at target depth
                parent_uf = {}

                def find(x):
                    if x not in parent_uf:
                        parent_uf[x] = x
                    if parent_uf[x] != x:
                        parent_uf[x] = find(parent_uf[x])
                    return parent_uf[x]

                def union(x, y):
                    px, py = find(x), find(y)
                    if px != py:
                        parent_uf[px] = py

                # Build connected components from filtered_edges
                for edge in filtered_edges:
                    union(edge.src_id, edge.dst_id)

                # Group nodes by component
                components = {}
                for node_id in filtered_nodes.keys():
                    root = find(node_id)
                    if root not in components:
                        components[root] = []
                    components[root].append(node_id)

                if debug:
                    print(f"[DEBUG] Found {len(components)} connected components")

                # Map components to parent modules
                # Strategy: Find depth-1 parents in execution order and map to components
                comp_to_parent = {}
                parent_to_comp = {}

                # Get all depth-1 modules in execution order
                depth1_parents = []
                all_edges = graph.get_edges()
                visited = set()

                # Find depth-1 modules by traversing edges
                for edge in all_edges:
                    for node_id in [edge.src_id, edge.dst_id]:
                        node = graph.get_node(node_id)
                        if (
                            node
                            and isinstance(node, ModuleNode)
                            and node.depth == target_depth - 1
                        ):
                            if node_id not in visited:
                                depth1_parents.append(node_id)
                                visited.add(node_id)

                if debug:
                    print(f"[DEBUG] Found {len(depth1_parents)} depth-1 parents")

                # Map components to parents (1-to-1 in execution order)
                comp_list = list(components.keys())
                for i in range(min(len(comp_list), len(depth1_parents))):
                    comp_to_parent[comp_list[i]] = depth1_parents[i]
                    parent_to_comp[depth1_parents[i]] = comp_list[i]
                    if debug:
                        parent_node = graph.get_node(depth1_parents[i])
                        parent_type = (
                            parent_node.module_type if parent_node else "unknown"
                        )
                        print(
                            f"[DEBUG] Component {i} -> Parent {depth1_parents[i]} ({parent_type})"
                        )

                # Detect consecutive repeated patterns at parent level
                comp_list = list(components.keys())
                loop_groups = _detect_loop_patterns(
                    comp_list, components, comp_to_parent, graph, filtered_edges, debug
                )

                # Apply visualization strategies
                filtered_nodes, filtered_edges = _apply_loop_visualization(
                    loop_groups,
                    components,
                    comp_list,
                    filtered_nodes,
                    filtered_edges,
                    graph,
                    debug,
                )

                # Store loop group metadata in graph for rendering
                graph.loop_groups = loop_groups
                graph.loop_components = components

                # Connect components in sequence
                for i in range(len(comp_list) - 1):
                    src_nodes = components[comp_list[i]]
                    dst_nodes = components[comp_list[i + 1]]

                    last_node = None
                    for node_id in src_nodes:
                        has_out = any(e.src_id == node_id for e in filtered_edges)
                        if not has_out:
                            last_node = node_id
                            break

                    first_node = None
                    for node_id in dst_nodes:
                        has_in = any(e.dst_id == node_id for e in filtered_edges)
                        if not has_in:
                            first_node = node_id
                            break

                    if last_node and first_node:
                        exists = any(
                            e.src_id == last_node and e.dst_id == first_node
                            for e in filtered_edges
                        )
                        if not exists:
                            filtered_edges.append(
                                Edge(src_id=last_node, dst_id=first_node)
                            )
                            if debug:
                                print(
                                    f"[DEBUG] Connected comp {i}->{i+1}: {last_node} -> {first_node}"
                                )

            if debug:
                print(f"[DEBUG] Filtered edges count: {len(filtered_edges)}")
                for edge in filtered_edges:
                    print(f"[DEBUG]   {edge.src_id} -> {edge.dst_id}")

            # Replace graph nodes and edges
            graph.nodes = filtered_nodes
            graph.edges = filtered_edges

            # Add Input and Output nodes
            if filtered_nodes:
                # Get module IDs BEFORE adding Input/Output nodes
                module_ids = list(filtered_nodes.keys())

                # Get input shape from wrapped_input
                if isinstance(wrapped_input, torch.Tensor):
                    input_shape = tuple(torch.Tensor.size(wrapped_input))
                    input_dtype = str(
                        torch.Tensor.dtype.__get__(wrapped_input)
                    ).replace("torch.", "")
                    input_device = str(torch.Tensor.device.__get__(wrapped_input))
                else:
                    input_shape = None
                    input_dtype = "unknown"
                    input_device = "unknown"

                # Get output shape from output
                if isinstance(output, torch.Tensor):
                    output_shape = tuple(torch.Tensor.size(output))
                    output_dtype = str(torch.Tensor.dtype.__get__(output)).replace(
                        "torch.", ""
                    )
                    output_device = str(torch.Tensor.device.__get__(output))
                else:
                    output_shape = None
                    output_dtype = "unknown"
                    output_device = "unknown"

                # Create Input node
                input_node = TensorNode(
                    node_id="input_tensor",
                    name="Input",
                    depth=-1,
                    tensor_id="input",
                    shape=input_shape,
                    dtype=input_dtype,
                    device=input_device,
                )
                graph.nodes["input_tensor"] = input_node

                # Create Output node
                output_node = TensorNode(
                    node_id="output_tensor",
                    name="Output",
                    depth=-1,
                    tensor_id="output",
                    shape=output_shape,
                    dtype=output_dtype,
                    device=output_device,
                )
                graph.nodes["output_tensor"] = output_node

                # Find first and last module nodes based on edges
                # Use saved module_ids (before Input/Output were added)
                if module_ids:
                    # Find nodes with no incoming edges (first nodes)
                    nodes_with_incoming = {edge.dst_id for edge in filtered_edges}
                    first_nodes = [
                        nid for nid in module_ids if nid not in nodes_with_incoming
                    ]

                    # Find nodes with no outgoing edges (last nodes)
                    nodes_with_outgoing = {edge.src_id for edge in filtered_edges}
                    last_nodes = [
                        nid for nid in module_ids if nid not in nodes_with_outgoing
                    ]

                    if debug:
                        print(f"[DEBUG] module_ids: {module_ids}")
                        print(f"[DEBUG] first_nodes: {first_nodes}")
                        print(f"[DEBUG] last_nodes: {last_nodes}")

                    # Create new edges list with Input/Output connections
                    from vode.nn.graph.builder import Edge

                    new_edges = list(filtered_edges)

                    # Connect Input to first nodes
                    for first_node_id in first_nodes:
                        new_edges.append(
                            Edge(src_id="input_tensor", dst_id=first_node_id)
                        )

                    # Connect last nodes to Output
                    for last_node_id in last_nodes:
                        new_edges.append(
                            Edge(src_id=last_node_id, dst_id="output_tensor")
                        )

                    if debug:
                        print(f"[DEBUG] Final edges count: {len(new_edges)}")
                        for edge in new_edges:
                            print(f"[DEBUG]   {edge.src_id} -> {edge.dst_id}")

                    graph.edges = new_edges

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
