"""Dataflow resolution system for building edges between function calls.

This module implements the dataflow resolver that tracks object identity through
the call tree to build edges connecting function calls based on data flow.
"""

from collections import defaultdict
from typing import Any

from vode.trace.models import (
    EdgeKind,
    GraphEdge,
    TensorMeta,
    TraceGraph,
    VariableRecord,
)


class DataflowResolver:
    """Resolves dataflow edges by tracking object identity through function calls.

    This class builds edges between function calls by:
    1. Tracking which function produced each object (by runtime_object_id)
    2. Tracking which functions consume each object
    3. Creating DATAFLOW edges from producers to consumers
    4. Creating CALL_TREE edges for parent-child relationships
    """

    def __init__(self, trace_graph: TraceGraph):
        """Initialize the dataflow resolver.

        Args:
            trace_graph: The trace graph with function calls and variables populated
        """
        self.trace_graph = trace_graph

        # Registry mapping runtime_object_id -> list of (node_id, slot_path, is_return)
        # is_return=True means this is a return value, False means parameter
        self.object_registry: dict[int, list[tuple[str, str, bool]]] = defaultdict(list)

        # Registry for tensor metadata matching (shape, dtype, device)
        # Maps (shape_tuple, dtype, device) -> list of (node_id, slot_path, is_return)
        self.tensor_registry: dict[tuple, list[tuple[str, str, bool]]] = defaultdict(
            list
        )

        # Track edges we've created to avoid duplicates
        self.created_edges: set[tuple[str, str, str]] = set()  # (src_id, dst_id, kind)

        # Edge ID counter
        self.edge_id_counter = 0

        # Build variable lookup by call_id for quick access
        self.variables_by_call: dict[str, list[VariableRecord]] = defaultdict(list)
        self._build_variable_lookup()

    def _build_variable_lookup(self) -> None:
        """Build a lookup table of variables by their associated call."""
        for var in self.trace_graph.variables:
            # Extract call_id from variable id (format: var:{runtime_id}:{slot_path})
            # We need to find which call this variable belongs to
            # Variables are associated with calls through arg_variable_ids and return_variable_ids
            pass

        # Build lookup from function calls
        for call in self.trace_graph.function_calls:
            for var_id in call.arg_variable_ids:
                var = self._get_variable_by_id(var_id)
                if var:
                    self.variables_by_call[call.id].append(var)

            for var_id in call.return_variable_ids:
                var = self._get_variable_by_id(var_id)
                if var:
                    self.variables_by_call[call.id].append(var)

    def _get_variable_by_id(self, var_id: str) -> VariableRecord | None:
        """Get a variable by its ID.

        Args:
            var_id: Variable ID to look up

        Returns:
            VariableRecord if found, None otherwise
        """
        for var in self.trace_graph.variables:
            if var.id == var_id:
                return var
        return None

    def resolve(self) -> list[GraphEdge]:
        """Main entry point for dataflow resolution.

        Returns:
            List of GraphEdge objects representing dataflow and call tree relationships
        """
        edges: list[GraphEdge] = []

        # Step 1: Build object registry
        self._build_object_registry()

        # Step 2: Create CALL_TREE edges for parent-child relationships
        call_tree_edges = self._create_call_tree_edges()
        edges.extend(call_tree_edges)

        # Step 3: Resolve parameter edges (dataflow from producers to consumers)
        dataflow_edges = self._resolve_parameter_edges()
        edges.extend(dataflow_edges)

        return edges

    def _build_object_registry(self) -> None:
        """Build registry mapping object IDs to their producing nodes.

        This creates two registries:
        1. object_registry: Maps runtime_object_id to (node_id, slot_path, is_return)
        2. tensor_registry: Maps (shape, dtype, device) to (node_id, slot_path, is_return)
        """
        # Process all function calls
        for call in self.trace_graph.function_calls:
            # Register return values as producers
            for var_id in call.return_variable_ids:
                var = self._get_variable_by_id(var_id)
                if var and var.runtime_object_id is not None:
                    # Register by object ID
                    self.object_registry[var.runtime_object_id].append(
                        (call.id, var.slot_path, True)
                    )

                    # Also register by tensor metadata if applicable
                    if var.tensor_meta is not None:
                        tensor_key = self._make_tensor_key(var.tensor_meta)
                        if tensor_key is not None:
                            self.tensor_registry[tensor_key].append(
                                (call.id, var.slot_path, True)
                            )

            # Register parameters as consumers (for tracking)
            for var_id in call.arg_variable_ids:
                var = self._get_variable_by_id(var_id)
                if var and var.runtime_object_id is not None:
                    # Register by object ID
                    self.object_registry[var.runtime_object_id].append(
                        (call.id, var.slot_path, False)
                    )

                    # Also register by tensor metadata if applicable
                    if var.tensor_meta is not None:
                        tensor_key = self._make_tensor_key(var.tensor_meta)
                        if tensor_key is not None:
                            self.tensor_registry[tensor_key].append(
                                (call.id, var.slot_path, False)
                            )

    def _make_tensor_key(self, tensor_meta: TensorMeta) -> tuple | None:
        """Create a hashable key from tensor metadata.

        Args:
            tensor_meta: Tensor metadata to convert to key

        Returns:
            Tuple of (shape, dtype, device) or None if incomplete
        """
        if (
            tensor_meta.shape is None
            or tensor_meta.dtype is None
            or tensor_meta.device is None
        ):
            return None

        # Convert shape list to tuple for hashing
        shape_tuple = tuple(tensor_meta.shape)
        return (shape_tuple, tensor_meta.dtype, tensor_meta.device)

    def _create_call_tree_edges(self) -> list[GraphEdge]:
        """Create CALL_TREE edges for parent-child relationships.

        Returns:
            List of GraphEdge objects representing call tree structure
        """
        edges: list[GraphEdge] = []

        for call in self.trace_graph.function_calls:
            if call.parent_id is not None:
                edge = self._create_edge(
                    call.parent_id,
                    None,  # No specific slot for call tree edges
                    call.id,
                    None,
                    "call_tree",
                )
                if edge:
                    edges.append(edge)

        return edges

    def _resolve_parameter_edges(self) -> list[GraphEdge]:
        """Resolve dataflow edges by connecting parameters to their sources.

        For each function call parameter, find which function produced that object
        and create a DATAFLOW edge from producer to consumer.

        Returns:
            List of GraphEdge objects representing dataflow relationships
        """
        edges: list[GraphEdge] = []

        # Build a map of variables by ID for quick lookup
        variables_by_id = {var.id: var for var in self.trace_graph.variables}

        for call in self.trace_graph.function_calls:
            # Process each parameter
            for var_id in call.arg_variable_ids:
                var = variables_by_id.get(var_id)
                if not var or var.runtime_object_id is None:
                    continue

                # Find producers of this object
                producers = self._match_by_identity(var.runtime_object_id)

                # If no exact match, try tensor metadata matching
                if not producers and var.tensor_meta is not None:
                    producers = self._match_by_tensor_identity(var.tensor_meta)

                # Create edges from each producer to this consumer
                for producer_call_id, producer_slot_path in producers:
                    # Don't create self-edges
                    if producer_call_id == call.id:
                        continue

                    # Update variable record with producer information
                    var.producer_call_id = producer_call_id

                    # Find the producer variable and update its consumer list
                    producer_call = next(
                        (
                            c
                            for c in self.trace_graph.function_calls
                            if c.id == producer_call_id
                        ),
                        None,
                    )
                    if producer_call:
                        for producer_var_id in producer_call.return_variable_ids:
                            producer_var = variables_by_id.get(producer_var_id)
                            if (
                                producer_var
                                and producer_var.runtime_object_id
                                == var.runtime_object_id
                            ):
                                if call.id not in producer_var.consumer_call_ids:
                                    producer_var.consumer_call_ids.append(call.id)

                    # Create dataflow edge
                    edge = self._create_edge(
                        producer_call_id,
                        producer_slot_path,
                        call.id,
                        var.slot_path,
                        "dataflow",
                    )
                    if edge:
                        edges.append(edge)

        return edges

    def _match_by_identity(self, obj_id: int) -> list[tuple[str, str]]:
        """Find nodes that produced an object with the given runtime ID.

        Args:
            obj_id: Runtime object ID to search for

        Returns:
            List of (node_id, slot_path) tuples for producers
        """
        results: list[tuple[str, str]] = []

        if obj_id in self.object_registry:
            for node_id, slot_path, is_return in self.object_registry[obj_id]:
                # Only return producers (is_return=True)
                if is_return:
                    results.append((node_id, slot_path))

        return results

    def _match_by_tensor_identity(
        self, tensor_meta: TensorMeta
    ) -> list[tuple[str, str]]:
        """Find nodes that produced a tensor with matching metadata.

        This handles cases where tensor identity changes but data is the same.

        Args:
            tensor_meta: Tensor metadata to match

        Returns:
            List of (node_id, slot_path) tuples for producers
        """
        results: list[tuple[str, str]] = []

        tensor_key = self._make_tensor_key(tensor_meta)
        if tensor_key is None:
            return results

        if tensor_key in self.tensor_registry:
            for node_id, slot_path, is_return in self.tensor_registry[tensor_key]:
                # Only return producers (is_return=True)
                if is_return:
                    results.append((node_id, slot_path))

        return results

    def _create_edge(
        self,
        from_node: str,
        from_slot: str | None,
        to_node: str,
        to_slot: str | None,
        kind: EdgeKind,
    ) -> GraphEdge | None:
        """Create a graph edge with deduplication.

        Args:
            from_node: Source node ID
            from_slot: Source slot path (optional)
            to_node: Destination node ID
            to_slot: Destination slot path (optional)
            kind: Edge kind (call_tree, dataflow, owns)

        Returns:
            GraphEdge if created, None if duplicate
        """
        # Check for duplicates
        edge_key = (from_node, to_node, kind)
        if edge_key in self.created_edges:
            return None

        # Mark as created
        self.created_edges.add(edge_key)

        # Generate edge ID
        edge_id = f"edge:{self.edge_id_counter}"
        self.edge_id_counter += 1

        # For now, edges connect nodes, not specific slots
        # Slot information is preserved in the variable records
        return GraphEdge(
            id=edge_id,
            src_id=from_node,
            dst_id=to_node,
            kind=kind,
        )
