"""Integration tests for Stage 2 Web Viewer.

Tests the complete Stage 2 pipeline including:
- Trace file loading
- API endpoints
- Server configuration
"""

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vode.trace.tracer import TraceRuntime, TraceConfig
from vode.trace.serializer import GraphSerializer
from vode.view.server import ViewServer
from vode.view.api import router, set_graph_data


def test_trace_to_json_pipeline():
    """Test generating a trace and saving to JSON for viewer consumption."""

    def simple_func(x):
        return x * 2

    def caller(n):
        return simple_func(n) + 1

    # Generate trace
    config = TraceConfig(max_depth=10)
    runtime = TraceRuntime(config)

    runtime.start()
    result = caller(5)
    graph = runtime.stop()

    assert result == 11
    assert len(graph.function_calls) >= 2

    # Serialize to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        serializer = GraphSerializer()
        serializer.serialize(graph, temp_path)

        # Verify JSON structure
        with open(temp_path, "r") as f:
            data = json.load(f)

        assert "version" in data
        assert "graph" in data
        assert "function_calls" in data["graph"]
        assert "variables" in data["graph"]
        assert "edges" in data["graph"]

    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_api_graph_endpoint():
    """Test /api/graph endpoint returns complete graph data."""

    # Create minimal test data
    test_data = {
        "version": "1.0",
        "timestamp": "2026-03-07T15:00:00.000000+00:00",
        "graph": {
            "root_call_ids": ["call_0"],
            "function_calls": [
                {
                    "id": "call_0",
                    "qualified_name": "test_func",
                    "display_name": "test_func",
                    "filename": "test.py",
                    "lineno": 1,
                    "depth": 0,
                    "parent_id": None,
                    "arg_variable_ids": [],
                    "return_variable_ids": [],
                }
            ],
            "variables": [],
            "edges": [],
        },
    }

    # Set graph data
    set_graph_data(test_data)

    # Test API
    client = TestClient(router)
    response = client.get("/graph")

    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0"
    assert "graph" in data
    assert len(data["graph"]["function_calls"]) == 1


def test_api_node_endpoint():
    """Test /api/node/{id} endpoint returns node details."""

    test_data = {
        "version": "1.0",
        "graph": {
            "function_calls": [
                {
                    "id": "call_123",
                    "qualified_name": "module.func",
                    "display_name": "func",
                    "filename": "test.py",
                    "lineno": 10,
                }
            ]
        },
    }

    set_graph_data(test_data)

    client = TestClient(router)
    response = client.get("/node/call_123")

    assert response.status_code == 200
    node = response.json()
    assert node["id"] == "call_123"
    assert node["display_name"] == "func"


def test_api_node_not_found():
    """Test /api/node/{id} returns 404 for missing node."""

    test_data = {"version": "1.0", "graph": {"function_calls": []}}

    set_graph_data(test_data)

    client = TestClient(router)
    response = client.get("/node/nonexistent")

    assert response.status_code == 404


def test_api_search_endpoint():
    """Test /api/search endpoint finds matching nodes."""

    test_data = {
        "version": "1.0",
        "graph": {
            "function_calls": [
                {
                    "id": "call_1",
                    "qualified_name": "module.compute",
                    "display_name": "compute",
                    "filename": "main.py",
                },
                {
                    "id": "call_2",
                    "qualified_name": "module.process",
                    "display_name": "process",
                    "filename": "utils.py",
                },
            ]
        },
    }

    set_graph_data(test_data)

    client = TestClient(router)
    response = client.get("/search?q=compute")

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["display_name"] == "compute"


def test_api_stats_endpoint():
    """Test /api/stats endpoint returns graph statistics."""

    test_data = {
        "version": "1.0",
        "graph": {
            "function_calls": [
                {"id": "c1", "depth": 0},
                {"id": "c2", "depth": 1},
                {"id": "c3", "depth": 2},
            ],
            "variables": [{"id": "v1"}, {"id": "v2"}],
            "edges": [{"source": "c1", "target": "c2", "kind": "call_tree"}],
        },
    }

    set_graph_data(test_data)

    client = TestClient(router)
    response = client.get("/stats")

    assert response.status_code == 200
    stats = response.json()
    assert stats["function_count"] == 3
    assert stats["variable_count"] == 2
    assert stats["edge_count"] == 1
    assert stats["max_depth"] == 2


def test_server_initialization():
    """Test ViewServer can be initialized with trace file."""

    # Create temp trace file
    test_data = {
        "version": "1.0",
        "graph": {
            "root_call_ids": [],
            "function_calls": [],
            "variables": [],
            "edges": [],
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        server = ViewServer(temp_path, host="127.0.0.1", port=8000)
        assert server.trace_file == temp_path
        assert server.host == "127.0.0.1"
        assert server.port == 8000
        assert server.app is not None

    finally:
        temp_path.unlink(missing_ok=True)


def test_server_load_trace():
    """Test ViewServer can load and parse trace file."""

    test_data = {
        "version": "1.0",
        "graph": {
            "root_call_ids": ["call_0"],
            "function_calls": [{"id": "call_0", "display_name": "main"}],
            "variables": [],
            "edges": [],
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        server = ViewServer(temp_path)
        server.load_trace()

        assert server.graph_data is not None
        assert server.graph_data["version"] == "1.0"
        assert len(server.graph_data["graph"]["function_calls"]) == 1

    finally:
        temp_path.unlink(missing_ok=True)


def test_server_missing_trace_file():
    """Test ViewServer raises error for missing trace file."""

    nonexistent_path = Path("/tmp/nonexistent_trace_file.json")
    server = ViewServer(nonexistent_path)

    with pytest.raises(FileNotFoundError):
        server.load_trace()
