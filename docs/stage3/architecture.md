# VODE Stage 3: System Architecture

## Overview

VODE Stage 3 implements a dual-graph neural network visualization system with two capture modes and multiple rendering backends.

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Code"
        Model[PyTorch Model]
        Init[__init__ execution]
        Forward[forward execution]
    end
    
    subgraph "Capture Layer"
        StructCapture[Structure Capturer]
        DataCapture[Dataflow Capturer]
        RecTensor[RecorderTensor]
    end
    
    subgraph "Graph Builder"
        StructGraph[Structure Graph Builder]
        DataGraph[Dataflow Graph Builder]
        NodeFactory[Node Factory]
    end
    
    subgraph "Storage Layer"
        GVWriter[Graphviz Writer]
        GVFile[.gv Files]
    end
    
    subgraph "Rendering Layer"
        StaticRender[Static Renderer]
        WebRender[Web Renderer]
        SVG[SVG/PNG/PDF]
        WebUI[Interactive Web UI]
    end
    
    Init --> StructCapture
    Forward --> DataCapture
    DataCapture --> RecTensor
    
    StructCapture --> StructGraph
    DataCapture --> DataGraph
    RecTensor --> DataGraph
    
    StructGraph --> NodeFactory
    DataGraph --> NodeFactory
    
    NodeFactory --> GVWriter
    GVWriter --> GVFile
    
    GVFile --> StaticRender
    GVFile --> WebRender
    
    StaticRender --> SVG
    WebRender --> WebUI
    
    style Model fill:#90ee90
    style GVFile fill:#ffd700
    style WebUI fill:#add8e6
```

## Component Architecture

### 1. Capture Layer

```mermaid
graph LR
    subgraph "Structure Capture"
        SC1[Module Inspector]
        SC2[Hierarchy Tracker]
        SC3[Connection Analyzer]
    end
    
    subgraph "Dataflow Capture"
        DC1[RecorderTensor]
        DC2[Module Wrapper]
        DC3[Function Interceptor]
        DC4[Context Manager]
    end
    
    SC1 --> SC2
    SC2 --> SC3
    
    DC4 --> DC2
    DC4 --> DC3
    DC2 --> DC1
    DC3 --> DC1
```

**Structure Capture** (during `__init__`):

- Inspects module hierarchy via `named_modules()`
- Tracks parent-child relationships
- Identifies layer types and parameters

**Dataflow Capture** (during `forward`):

- Uses RecorderTensor subclass
- Wraps `nn.Module.__call__`
- Intercepts `__torch_function__`
- Operates within context manager scope

### 2. Graph Builder Layer

```mermaid
graph TB
    subgraph "Node Types"
        TN[TensorNode]
        MN[ModuleNode]
        FN[FunctionNode]
    end
    
    subgraph "Graph Structure"
        Root[Root Container]
        Hierarchy[Node Hierarchy Dict]
        EdgeList[Edge List]
    end
    
    subgraph "Node Factory"
        Create[Create Node]
        Connect[Connect Edges]
        SetAttrs[Set Attributes]
    end
    
    TN --> Create
    MN --> Create
    FN --> Create
    
    Create --> Hierarchy
    Connect --> EdgeList
    SetAttrs --> Hierarchy
    
    Root --> Hierarchy
```

**Node Types**:

- **TensorNode**: Represents tensors (input/output/intermediate)
- **ModuleNode**: Represents nn.Module instances
- **FunctionNode**: Represents torch functions (relu, cat, etc.)

**Graph Structure**:

- Root container holds input nodes
- Hierarchy dict maintains parent-child relationships
- Edge list stores all connections

### 3. Storage Layer (Graphviz Format)

```mermaid
graph LR
    subgraph "Graph Object"
        Nodes[Node Collection]
        Edges[Edge Collection]
        Attrs[Graph Attributes]
    end
    
    subgraph "Graphviz Writer"
        NodeWriter[Node Writer]
        EdgeWriter[Edge Writer]
        AttrWriter[Attribute Writer]
    end
    
    subgraph "Output"
        DOT[.gv File]
        Meta[Metadata JSON]
    end
    
    Nodes --> NodeWriter
    Edges --> EdgeWriter
    Attrs --> AttrWriter
    
    NodeWriter --> DOT
    EdgeWriter --> DOT
    AttrWriter --> DOT
    
    Nodes --> Meta
    Edges --> Meta
```

**Graphviz Format Benefits**:

- Text-based, diff-friendly
- Extensible with custom attributes
- Multiple output formats (SVG, PNG, PDF)
- Industry standard

### 4. Rendering Layer

```mermaid
graph TB
    subgraph "Input"
        GV[.gv File]
        JSON[Metadata JSON]
    end
    
    subgraph "Static Renderer"
        GVEngine[Graphviz Engine]
        LayoutEngine[Layout Algorithm]
    end
    
    subgraph "Web Renderer"
        Parser[DOT Parser]
        D3[D3.js / Cytoscape]
        React[React Components]
    end
    
    subgraph "Output"
        SVG[SVG]
        PNG[PNG]
        PDF[PDF]
        WebUI[Interactive UI]
    end
    
    GV --> GVEngine
    GV --> Parser
    JSON --> React
    
    GVEngine --> LayoutEngine
    LayoutEngine --> SVG
    LayoutEngine --> PNG
    LayoutEngine --> PDF
    
    Parser --> D3
    D3 --> React
    React --> WebUI
```

## Source Code Structure

```
vode/src/vode/
├── nn/                          # NEW: Neural network visualization
│   ├── __init__.py
│   ├── capture/                 # Capture mechanisms
│   │   ├── __init__.py
│   │   ├── recorder_tensor.py   # RecorderTensor subclass
│   │   ├── structure_capture.py # Structure graph capture
│   │   ├── dataflow_capture.py  # Dataflow graph capture
│   │   └── context.py           # Context manager
│   ├── graph/                   # Graph building
│   │   ├── __init__.py
│   │   ├── nodes.py             # Node classes
│   │   ├── builder.py           # Graph builder
│   │   └── hierarchy.py         # Hierarchy management
│   ├── storage/                 # Storage layer
│   │   ├── __init__.py
│   │   ├── graphviz_writer.py   # .gv file writer
│   │   └── metadata.py          # Metadata extraction
│   └── render/                  # Rendering layer
│       ├── __init__.py
│       ├── static.py            # Static rendering (SVG/PNG/PDF)
│       └── web.py               # Web rendering utilities
├── view/                        # Existing web viewer
│   ├── frontend/                # React frontend
│   │   └── src/
│   │       ├── components/
│   │       │   ├── StructureView.tsx    # NEW
│   │       │   ├── DataflowView.tsx     # Enhanced
│   │       │   └── NodeInspector.tsx    # NEW
│   │       └── utils/
│   │           └── graphviz_parser.ts   # NEW
│   └── server.py                # Enhanced API
└── cli.py                       # Enhanced CLI
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant StructCapture
    participant DataCapture
    participant GraphBuilder
    participant GVWriter
    participant Renderer
    
    User->>CLI: vode visualize model.py
    CLI->>StructCapture: Capture structure
    StructCapture->>GraphBuilder: Build structure graph
    CLI->>DataCapture: Run forward pass
    DataCapture->>GraphBuilder: Build dataflow graph
    GraphBuilder->>GVWriter: Write .gv files
    GVWriter->>Renderer: Render graphs
    Renderer->>User: Display/Export
```

## Key Design Patterns

### 1. Context Manager Pattern

```python
with DataflowCapture(model) as capture:
    output = model(input_data)
    graph = capture.get_graph()
```

### 2. Visitor Pattern

```python
class NodeVisitor:
    def visit_tensor_node(self, node): ...
    def visit_module_node(self, node): ...
    def visit_function_node(self, node): ...
```

### 3. Builder Pattern

```python
builder = GraphBuilder()
builder.add_node(node)
builder.add_edge(src, dst)
graph = builder.build()
```

## Extensibility Points

1. **Custom Node Types**: Add new node classes inheriting from `Node`
2. **Custom Attributes**: Extend metadata extraction
3. **Custom Renderers**: Implement new rendering backends
4. **Custom Layouts**: Add layout algorithms for different use cases

## Performance Considerations

- **Lazy Evaluation**: Build graphs on-demand
- **Streaming**: Write .gv files incrementally
- **Caching**: Cache parsed graphs for web rendering
- **Filtering**: Support depth/pattern filtering to reduce graph size
