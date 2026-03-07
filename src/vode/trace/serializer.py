"""Graph serialization module for saving and loading trace graphs.

This module provides JSON-based serialization for TraceGraph objects, handling
dataclasses, enums, and special types like TensorMeta.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vode.trace.models import (
    EdgeKind,
    FunctionCallNode,
    GraphEdge,
    TensorMeta,
    TensorStats,
    TraceGraph,
    ValuePreview,
    VariableRecord,
)


class GraphSerializer:
    """Serializes and deserializes TraceGraph objects to/from JSON."""

    VERSION = "1.0"

    def serialize(self, graph: TraceGraph, output_path: str) -> None:
        """Save a TraceGraph to a JSON file.

        Args:
            graph: TraceGraph to serialize
            output_path: Path to output JSON file
        """
        # Convert graph to dictionary
        graph_dict = self._to_dict(graph)

        # Add metadata
        serialized = {
            "version": self.VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "graph": graph_dict,
        }

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, default=self._json_default)

    def deserialize(self, input_path: str) -> TraceGraph:
        """Load a TraceGraph from a JSON file.

        Args:
            input_path: Path to input JSON file

        Returns:
            Deserialized TraceGraph object
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check version compatibility
        version = data.get("version", "unknown")
        if version != self.VERSION:
            print(
                f"Warning: File version {version} may not be compatible with {self.VERSION}"
            )

        # Extract graph data
        graph_data = data.get("graph", {})

        # Reconstruct graph
        return self._from_dict(graph_data)

    def _to_dict(self, graph: TraceGraph) -> dict[str, Any]:
        """Convert TraceGraph to dictionary.

        Args:
            graph: TraceGraph to convert

        Returns:
            Dictionary representation of the graph
        """
        return {
            "root_call_ids": graph.root_call_ids,
            "function_calls": [
                self._serialize_node(node) for node in graph.function_calls
            ],
            "variables": [self._serialize_variable(var) for var in graph.variables],
            "edges": [self._serialize_edge(edge) for edge in graph.edges],
        }

    def _from_dict(self, data: dict[str, Any]) -> TraceGraph:
        """Convert dictionary to TraceGraph.

        Args:
            data: Dictionary representation of graph

        Returns:
            Reconstructed TraceGraph object
        """
        # Deserialize function calls
        function_calls = [
            self._deserialize_node(node_data)
            for node_data in data.get("function_calls", [])
        ]

        # Deserialize variables
        variables = [
            self._deserialize_variable(var_data)
            for var_data in data.get("variables", [])
        ]

        # Deserialize edges
        edges = [
            self._deserialize_edge(edge_data) for edge_data in data.get("edges", [])
        ]

        return TraceGraph(
            root_call_ids=data.get("root_call_ids", []),
            function_calls=function_calls,
            variables=variables,
            edges=edges,
        )

    def _serialize_node(self, node: FunctionCallNode) -> dict[str, Any]:
        """Serialize a FunctionCallNode to dictionary.

        Args:
            node: Node to serialize

        Returns:
            Dictionary representation
        """
        return {
            "id": node.id,
            "parent_id": node.parent_id,
            "qualified_name": node.qualified_name,
            "display_name": node.display_name,
            "filename": node.filename,
            "lineno": node.lineno,
            "depth": node.depth,
            "arg_variable_ids": node.arg_variable_ids,
            "return_variable_ids": node.return_variable_ids,
            "metadata": node.metadata,
        }

    def _deserialize_node(self, data: dict[str, Any]) -> FunctionCallNode:
        """Deserialize a FunctionCallNode from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed FunctionCallNode
        """
        return FunctionCallNode(
            id=data["id"],
            parent_id=data.get("parent_id"),
            qualified_name=data["qualified_name"],
            display_name=data["display_name"],
            filename=data["filename"],
            lineno=data["lineno"],
            depth=data["depth"],
            arg_variable_ids=data.get("arg_variable_ids", []),
            return_variable_ids=data.get("return_variable_ids", []),
            metadata=data.get("metadata", {}),
        )

    def _serialize_variable(self, var: VariableRecord) -> dict[str, Any]:
        """Serialize a VariableRecord to dictionary.

        Args:
            var: Variable to serialize

        Returns:
            Dictionary representation
        """
        return {
            "id": var.id,
            "slot_path": var.slot_path,
            "display_name": var.display_name,
            "runtime_object_id": var.runtime_object_id,
            "type_name": var.type_name,
            "tensor_meta": (
                self._serialize_tensor_meta(var.tensor_meta)
                if var.tensor_meta
                else None
            ),
            "tensor_stats": (
                self._serialize_tensor_stats(var.tensor_stats)
                if var.tensor_stats
                else None
            ),
            "preview": self._serialize_preview(var.preview) if var.preview else None,
            "producer_call_id": var.producer_call_id,
            "consumer_call_ids": var.consumer_call_ids,
        }

    def _deserialize_variable(self, data: dict[str, Any]) -> VariableRecord:
        """Deserialize a VariableRecord from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed VariableRecord
        """
        return VariableRecord(
            id=data["id"],
            slot_path=data["slot_path"],
            display_name=data["display_name"],
            runtime_object_id=data.get("runtime_object_id"),
            type_name=data["type_name"],
            tensor_meta=(
                self._deserialize_tensor_meta(data.get("tensor_meta"))
                if data.get("tensor_meta")
                else None
            ),
            tensor_stats=(
                self._deserialize_tensor_stats(data.get("tensor_stats"))
                if data.get("tensor_stats")
                else None
            ),
            preview=(
                self._deserialize_preview(data.get("preview"))
                if data.get("preview")
                else None
            ),
            producer_call_id=data.get("producer_call_id"),
            consumer_call_ids=data.get("consumer_call_ids", []),
        )

    def _serialize_edge(self, edge: GraphEdge) -> dict[str, Any]:
        """Serialize a GraphEdge to dictionary.

        Args:
            edge: Edge to serialize

        Returns:
            Dictionary representation
        """
        return {
            "id": edge.id,
            "src_id": edge.src_id,
            "dst_id": edge.dst_id,
            "kind": edge.kind,  # EdgeKind is already a string literal
        }

    def _deserialize_edge(self, data: dict[str, Any]) -> GraphEdge:
        """Deserialize a GraphEdge from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed GraphEdge
        """
        return GraphEdge(
            id=data["id"],
            src_id=data["src_id"],
            dst_id=data["dst_id"],
            kind=data["kind"],
        )

    def _serialize_tensor_meta(self, meta: TensorMeta) -> dict[str, Any]:
        """Serialize TensorMeta to dictionary.

        Args:
            meta: TensorMeta to serialize

        Returns:
            Dictionary representation
        """
        return {
            "shape": meta.shape,
            "dtype": meta.dtype,
            "device": meta.device,
            "requires_grad": meta.requires_grad,
            "numel": meta.numel,
        }

    def _deserialize_tensor_meta(self, data: dict[str, Any]) -> TensorMeta:
        """Deserialize TensorMeta from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed TensorMeta
        """
        return TensorMeta(
            shape=data.get("shape"),
            dtype=data.get("dtype"),
            device=data.get("device"),
            requires_grad=data.get("requires_grad"),
            numel=data.get("numel"),
        )

    def _serialize_tensor_stats(self, stats: TensorStats) -> dict[str, Any]:
        """Serialize TensorStats to dictionary.

        Args:
            stats: TensorStats to serialize

        Returns:
            Dictionary representation
        """
        return {
            "min": stats.min,
            "max": stats.max,
            "mean": stats.mean,
            "std": stats.std,
        }

    def _deserialize_tensor_stats(self, data: dict[str, Any]) -> TensorStats:
        """Deserialize TensorStats from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed TensorStats
        """
        return TensorStats(
            min=data.get("min"),
            max=data.get("max"),
            mean=data.get("mean"),
            std=data.get("std"),
        )

    def _serialize_preview(self, preview: ValuePreview) -> dict[str, Any]:
        """Serialize ValuePreview to dictionary.

        Args:
            preview: ValuePreview to serialize

        Returns:
            Dictionary representation
        """
        return {
            "text": preview.text,
            "data": preview.data,
        }

    def _deserialize_preview(self, data: dict[str, Any]) -> ValuePreview:
        """Deserialize ValuePreview from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed ValuePreview
        """
        return ValuePreview(
            text=data.get("text"),
            data=data.get("data"),
        )

    def _json_default(self, obj: Any) -> Any:
        """Custom JSON encoder for special types.

        Args:
            obj: Object to encode

        Returns:
            JSON-serializable representation

        Raises:
            TypeError: If object cannot be serialized
        """
        # Handle dataclasses
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)

        # Handle other types as needed
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
