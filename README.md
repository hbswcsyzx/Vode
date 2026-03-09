# VODE

A comprehensive visualization and debugging tool for Python execution and PyTorch neural networks.

## Overview

VODE provides two powerful capabilities:

- **Execution Tracing**: Capture and visualize function call trees for any Python code
- **Neural Network Visualization**: Dual-graph visualization of PyTorch model architecture and dataflow

## Features

### Execution Tracing

- Zero-code modification tracing using `sys.settrace()`
- Function call tree capture with parent-child relationships
- Parameter and return value recording
- Dataflow edge resolution by object ID matching
- PyTorch tensor metadata extraction (shape, dtype, device, stats)
- Interactive web visualization

### Neural Network Visualization

- **Dual-graph architecture**: Separate structure and dataflow graphs
- **Structure graph**: Captures model architecture during initialization
- **Dataflow graph**: Captures tensor operations during forward pass
- **Horizontal layout**: INPUT-OP-OUTPUT node design for clear dataflow
- **Interactive inspection**: Click nodes to view detailed tensor statistics
- **Multiple formats**: Export to SVG, PNG, PDF, or view in browser
- **Graphviz storage**: Extensible .gv format for custom processing

## Installation

**Requirements**: Python 3.10+

```bash
cd vode
pip install -e .
```

**Optional**: Install PyTorch for neural network visualization.

## Quick Start

### Function Tracing

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
```

### Neural Network Visualization

```python
from vode.nn import visualize_model
import torch
import torch.nn as nn

# Define your model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Visualize
input_data = torch.randn(1, 3, 224, 224)
visualize_model(
    model, 
    input_data,
    save_path='model_viz',
    format='svg'  # or 'png', 'pdf', 'web'
)
```

## Architecture

```
vode/
├── src/vode/
│   ├── cli.py              # CLI interface
│   ├── trace/              # Execution tracing
│   │   ├── models.py       # Data models
│   │   ├── tracer.py       # Tracing engine
│   │   ├── serializer.py   # JSON serialization
│   │   └── renderer.py     # Text rendering
│   ├── nn/                 # Neural network visualization
│   │   ├── capture/        # Capture mechanisms
│   │   ├── graph/          # Graph building
│   │   ├── storage/        # Graphviz storage
│   │   └── render/         # Rendering layer
│   └── view/               # Web viewer
│       ├── frontend/       # React frontend
│       └── server.py       # API server
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Documentation

**Execution Tracing**:

- [Capabilities](docs/stage1/capabilities.md) - Feature scope and examples
- [Quick Start](docs/stage2/quickstart.md) - Getting started guide

**Neural Network Visualization**:

- [Overview](docs/stage3/overview.md) - High-level design
- [Architecture](docs/stage3/architecture.md) - System architecture
- [Node Design](docs/stage3/node_design.md) - Node structure and layout
- [Capture Mechanism](docs/stage3/capture_mechanism.md) - How data is captured
- [Rendering](docs/stage3/rendering.md) - Web vs static rendering

## Development

```bash
# Run tests
pytest tests/

# Run integration tests
pytest tests/test_integration.py -v
```

## Roadmap

- [x] Function-level execution tracing
- [x] Web-based visualization
- [ ] Neural network dual-graph visualization
- [ ] Performance profiling integration
- [ ] Multi-framework support (TensorFlow, JAX)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
