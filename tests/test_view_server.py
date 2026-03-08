"""Tests for the view server module."""

import json
import tempfile
from pathlib import Path

import pytest

from vode.view.server import ViewServer
from vode.view.api import set_graph_data, get_graph, get_node, search_nodes, get_stats


def test_view_server_initialization():
    """Test ViewServer initialization."""
    trace_file = Path("test_trace.json")
    server = ViewServer(trace_file, host="localhost", port=9000)

    assert server.trace_file == trace_file
    assert server.host == "localhost"
    assert server.port == 9000
    assert server.app is not None


def test_load_trace():
    """Test loading a trace file."""
    # Create a temporary trace file
    trace_data = {
        "version": "1.0",
        "timestamp": "2026-03-07T14:00:00+00:00",
        "graph": {
            "root_call_ids": ["call_0"],
            "function_calls": [
                {
                    "id": "call_0",
                    "qualified_name": "test.func",
                    "display_name": "func",
                    "filename": "test.py",
                    "lineno": 10,
                    "depth": 0,
                }
            ],
            "variables": [],
            "edges": [],
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(trace_data, f)
        temp_path = Path(f.name)

    try:
        server = ViewServer(temp_path)
        server.load_trace()

        assert server.graph_data is not None
        assert server.graph_data["version"] == "1.0"
        assert len(server.graph_data["graph"]["function_calls"]) == 1
    finally:
        temp_path.unlink()


def test_load_trace_file_not_found():
    """Test loading a non-existent trace file."""
    server = ViewServer(Path("nonexistent.json"))

    with pytest.raises(FileNotFoundError):
        server.load_trace()


@pytest.mark.asyncio
async def test_api_get_graph():
    """Test the /api/graph endpoint."""
    test_data = {
        "version": "1.0",
        "graph": {"function_calls": [], "variables": [], "edges": []},
    }

    set_graph_data(test_data)
    result = await get_graph()

    assert result == test_data


@pytest.mark.asyncio
async def test_api_get_node():
    """Test the /api/node/{id} endpoint."""
    test_data = {
        "graph": {
            "function_calls": [
                {"id": "call_0", "qualified_name": "test.func"},
                {"id": "call_1", "qualified_name": "test.func2"},
            ],
        },
    }

    set_graph_data(test_data)
    result = await get_node("call_0")

    assert result["id"] == "call_0"
    assert result["qualified_name"] == "test.func"


@pytest.mark.asyncio
async def test_api_search_nodes():
    """Test the /api/search endpoint."""
    test_data = {
        "graph": {
            "function_calls": [
                {"id": "call_0", "qualified_name": "test.func", "display_name": "func"},
                {
                    "id": "call_1",
                    "qualified_name": "other.method",
                    "display_name": "method",
                },
            ],
        },
    }

    set_graph_data(test_data)
    result = await search_nodes("func")

    assert len(result["results"]) == 1
    assert result["results"][0]["id"] == "call_0"


@pytest.mark.asyncio
async def test_api_get_stats():
    """Test the /api/stats endpoint."""
    test_data = {
        "graph": {
            "function_calls": [
                {"id": "call_0", "depth": 0},
                {"id": "call_1", "depth": 1},
                {"id": "call_2", "depth": 2},
            ],
            "variables": [{"id": "var_0"}, {"id": "var_1"}],
            "edges": [{"source": "call_0", "target": "call_1"}],
        },
    }

    set_graph_data(test_data)
    result = await get_stats()

    assert result["function_count"] == 3
    assert result["variable_count"] == 2
    assert result["edge_count"] == 1
    assert result["max_depth"] == 2
