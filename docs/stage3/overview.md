# VODE Stage 3: Dual-Graph Neural Network Visualization

## Executive Summary

Stage 3 transforms VODE into a specialized neural network visualization tool inspired by torchview, with two complementary graph types:

1. **Model Structure Graph**: Captures the static architecture during `__init__`
2. **Dataflow Graph**: Captures dynamic tensor flow during `forward()`

## Key Design Decisions

### 1. Data Format: Graphviz (.gv)

**Rationale**:

- Text-based, human-readable, version-controllable
- Highly extensible with custom attributes
- Native support for multiple output formats (SVG, PNG, PDF)
- Industry-standard with robust tooling ecosystem

### 2. Capture Mechanism: RecorderTensor Pattern

**Inspired by torchview**:

- Subclass `torch.Tensor` to track data flow
- Intercept `nn.Module.__call__` during forward pass
- Override `__torch_function__` to capture operations
- Use context manager for temporary instrumentation

### 3. Dual-Graph Architecture

**Model Structure Graph** (Static):

- Captures module hierarchy during initialization
- Shows layer composition and connections
- Useful for understanding architecture

**Dataflow Graph** (Dynamic):

- Captures actual tensor operations during execution
- Shows data transformations and shapes
- Useful for debugging and optimization

### 4. Node Design: INPUT-OP-OUTPUT Structure

All nodes follow a horizontal layout:

```text
┌─────┬──────────┬─────┐
│ IN  │    OP    │ OUT │
│ ●   │  Conv2d  │  ●  │
│ ●   │ depth: 3 │  ●  │
└─────┴──────────┴─────┘
```

**Low-level nodes** (torch operations):

- Display tensor shapes in input/output sections
- Example: `(1, 3, 224, 224)` shown directly

**High-level nodes** (user functions):

- Display only "input" and "output" labels
- Shapes accessible via interactive inspection

### 5. Rendering Modes

**Web Mode** (Interactive):

- Click nodes to inspect details
- Zoom, pan, filter operations
- Real-time shape/stats display
- Hover tooltips

**Static Mode** (PDF/SVG/PNG):

- Compact layout like torchview
- Essential information only
- Print-friendly formatting

## Technical Stack

- **Backend**: Python 3.10+, PyTorch
- **Graph Format**: Graphviz DOT language
- **Frontend**: React + TypeScript (web mode)
- **Rendering**: Graphviz (static), D3.js/Cytoscape.js (web)

## Stage 3 Goals

1. ✅ Implement RecorderTensor-based capture
2. ✅ Generate dual graphs (structure + dataflow)
3. ✅ Support INPUT-OP-OUTPUT node layout
4. ✅ Enable interactive web inspection
5. ✅ Export to multiple static formats
6. ✅ Maintain extensibility for future enhancements

## Comparison with Torchview

| Feature       | Torchview         | VODE Stage 3                 |
|---------------|-------------------|------------------------------|
| Graph Type    | Single (dataflow) | Dual (structure + dataflow)  |
| Node Layout   | Vertical          | Horizontal (INPUT-OP-OUTPUT) |
| Interactivity | Static only       | Web + Static                 |
| Data Format   | Graphviz          | Graphviz                     |
| Inspection    | None              | Click-to-inspect details     |
| Scope         | PyTorch only      | PyTorch + extensible         |

## Next Steps

See detailed documentation:

- [`architecture.md`](architecture.md) - System architecture
- [`data_format.md`](data_format.md) - Graphviz format spec
- [`node_design.md`](node_design.md) - Node structure details
- [`capture_mechanism.md`](capture_mechanism.md) - Capture strategy
- [`rendering.md`](rendering.md) - Rendering implementation
