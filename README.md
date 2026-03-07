# Vode

A lightweight function-level execution tracing and visualization tool for Python.

## Overview

Vode captures function call trees, parameters, return values, and dataflow relationships during Python program execution. It's designed for debugging, understanding code flow, and visualizing execution patterns.

**Current Status**: Stage 1 (Trace functionality) - ✅ Complete

## Features

- ✅ **Zero-code modification tracing** using `sys.settrace()`
- ✅ **Function call tree capture** with parent-child relationships
- ✅ **Parameter and return value recording** with configurable policies
- ✅ **Dataflow edge resolution** by object ID matching
- ✅ **PyTorch tensor support** with shape/dtype/device/stats extraction
- ✅ **JSON serialization** for persistence
- ✅ **Text rendering** for quick inspection
- ⏳ **Web visualization** (Stage 2 - planned)

## Installation

**Requirements**: Python 3.10+

```bash
cd vode
pip install -e .
```

**Optional**: Install PyTorch for tensor tracing support.

## Quick Start

### CLI Usage

```bash
# Trace a Python script
vode trace script.py

# Save trace to file
vode trace script.py --output trace.json

# Configure tracing
vode trace script.py --max-depth 5 --exclude "torch.*"
```

### Programmatic Usage

```python
from vode.trace import TraceRuntime, TraceConfig

# Configure tracing
config = TraceConfig(max_depth=10, value_policy="preview")
runtime = TraceRuntime(config)

# Trace execution
runtime.start()
result = my_function()
graph = runtime.stop()

# Render as text
from vode.trace.renderer import TextRenderer
print(TextRenderer().render(graph))

# Serialize to JSON
from vode.trace.serializer import GraphSerializer
json_data = GraphSerializer().serialize(graph)
```

## Example

```python
# example.py
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def compute(x, y):
    sum_val = add(x, y)
    prod_val = multiply(x, y)
    return sum_val, prod_val

if __name__ == '__main__':
    result = compute(3, 4)
    print(f"Result: {result}")
```

Run with tracing:

```bash
vode trace example.py
```

Output:

```
Result: (7, 12)
Trace saved to: trace.json
Summary:
  Function calls: 4
  Total edges: 3
  Dataflow edges: 0
```

## What Vode Can Do

- Trace function calls at any depth
- Capture parameters and return values
- Track dataflow between functions
- Extract PyTorch tensor metadata (shape, dtype, device, stats)
- Filter by depth and module patterns
- Serialize traces to JSON
- Render call trees as text

See [`docs/stage1/capabilities.md`](docs/stage1/capabilities.md) for details.

## What Vode Cannot Do

- Cannot trace intra-function statement-level execution
- Cannot trace tensor operations (use [torchview](https://github.com/mert-kurttutan/torchview) for that)
- Cannot trace C/C++ extension internals
- Cannot provide web visualization yet (Stage 2)

## Architecture

```
vode/
├── src/vode/
│   ├── cli.py              # CLI interface
│   └── trace/              # Trace functionality
│       ├── models.py       # Data models
│       ├── tracer.py       # Tracing engine (sys.settrace)
│       ├── value_extractor.py    # Value extraction
│       ├── dataflow_resolver.py  # Dataflow analysis
│       ├── serializer.py   # JSON serialization
│       └── renderer.py     # Text rendering
├── tests/                  # Test suite
└── docs/stage1/           # Documentation
```

## Development

```bash
# Run tests
pytest tests/

# Run integration tests
pytest tests/test_integration.py -v
```

**Test Results**: All 10 integration tests pass (100% success rate).

## Documentation

- [`docs/stage1/capabilities.md`](docs/stage1/capabilities.md) - Feature scope and examples
- [`docs/stage1/todo.md`](docs/stage1/todo.md) - Technical implementation plan
- [`docs/stage1/report.md`](docs/stage1/report.md) - Implementation report (Chinese)

## Roadmap

- [x] Stage 1: Trace functionality
- [ ] Stage 2: Web visualization
- [ ] Stage 3: Performance profiling
- [ ] Stage 4: Multi-framework support

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
