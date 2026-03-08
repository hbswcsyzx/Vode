"""Test script to verify API adapter works correctly."""

import json
from pathlib import Path
from vode.view.adapters import adapt_graph_for_frontend

# Load a sample trace file
trace_file = Path("trace.json")
if not trace_file.exists():
    print(f"Error: {trace_file} not found")
    exit(1)

with open(trace_file) as f:
    stage1_data = json.load(f)

print("Stage 1 data structure:")
print(f"  version: {stage1_data.get('version')}")
print(f"  root_call_ids: {stage1_data.get('graph', {}).get('root_call_ids')}")
print(
    f"  function_calls count: {len(stage1_data.get('graph', {}).get('function_calls', []))}"
)
print(f"  edges count: {len(stage1_data.get('graph', {}).get('edges', []))}")

# Convert to frontend format
frontend_data = adapt_graph_for_frontend(stage1_data)

print("\nFrontend data structure:")
print(f"  version: {frontend_data.get('version')}")
print(f"  root_node: {frontend_data.get('root_node')}")
print(f"  nodes count: {len(frontend_data.get('nodes', {}))}")
print(f"  dataflow_edges count: {len(frontend_data.get('dataflow_edges', []))}")

# Show first node
if frontend_data.get("nodes"):
    first_node_id = list(frontend_data["nodes"].keys())[0]
    first_node = frontend_data["nodes"][first_node_id]
    print(f"\nFirst node example:")
    print(f"  id: {first_node.get('id')}")
    print(f"  function_name: {first_node.get('function_name')}")
    print(f"  file: {first_node.get('file')}")
    print(f"  children: {first_node.get('children')}")

print("\nAdapter test completed successfully!")
