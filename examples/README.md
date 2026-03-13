# VODE Examples

This directory contains example scripts demonstrating VODE's capabilities for visualizing PyTorch model architectures.

## Directory Structure

```
examples/
├── simple_example.py      # Basic Sequential model (~100 lines)
├── advanced_example.py    # ResNet-style architecture (~300 lines)
├── complex_example.py     # Transformer architecture (~600 lines)
├── output/                # Generated visualization files
│   ├── simple/           # Simple example outputs
│   ├── advanced/         # Advanced example outputs
│   └── complex/          # Complex example outputs
└── README.md             # This file
```

## Examples Overview

### 1. Simple Example (`simple_example.py`)

**Complexity:** Basic  
**Lines:** ~100  
**Architecture:** Sequential feedforward network  
**Features:**

- 3 Linear layers with ReLU activations
- Basic parameter counting
- Demonstrates fundamental VODE usage

### 2. Advanced Example (`advanced_example.py`)

**Complexity:** Intermediate  
**Lines:** ~300  
**Architecture:** ResNet-style with residual blocks  
**Features:**

- Custom residual blocks with skip connections
- Nested Sequential modules
- ModuleList for repeated blocks
- Multi-stage architecture
- Depth-based expansion control

### 3. Complex Example (`complex_example.py`)

**Complexity:** Advanced  
**Lines:** ~600  
**Architecture:** Transformer encoder-decoder  
**Features:**

- Multi-head self-attention mechanisms
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Encoder-decoder structure
- Deep nested hierarchies (6+ levels)
- 37M+ parameters

## Usage

Each example can be run in two ways:

### Method 1: Direct Python Execution

Run the script directly to execute the model and generate visualizations:

```bash
# Simple example
python simple_example.py

# Advanced example
python advanced_example.py

# Complex example
python complex_example.py
```

**Output:** Generates `.gv` (Graphviz DOT) files in `output/<example_name>/` directory.

### Method 2: VODE CLI Analysis

Use the VODE command-line tool to analyze the script:

```bash
# Simple example with depth 1
vode --depth 1 simple_example.py

# Advanced example with depth 2
vode --depth 2 advanced_example.py

# Complex example with depth 3
vode --depth 3 complex_example.py
```

**Output:**

- Script executes and generates its own visualizations
- VODE also captures the model and generates an additional visualization

## Visualizing Output Files

The examples generate `.gv` (Graphviz DOT) files. To convert them to images:

```bash
# Navigate to output directory
cd output/simple

# Convert to PNG
dot -Tpng simple_depth1.gv -o simple_depth1.png

# Convert to SVG (recommended for scalability)
dot -Tsvg simple_dynamic.gv -o simple_dynamic.svg

# Convert to PDF
dot -Tpdf simple_depth0.gv -o simple_depth0.pdf
```

## Understanding Depth Levels

VODE uses depth-based expansion to control visualization detail:

- **Depth 0:** Show only the root module (high-level overview)
- **Depth 1:** Show immediate children (main components)
- **Depth 2:** Show sub-components (detailed structure)
- **Depth 3+:** Show deep internals (attention mechanisms, etc.)

### Example: Simple Model

```bash
python simple_example.py
```

Generates:

- `simple_depth0.gv` - Root only (SimpleModel)
- `simple_depth1.gv` - 5 layers (fc1, relu1, fc2, relu2, fc3)
- `simple_dynamic.gv` - With runtime tensor shapes

### Example: Advanced Model

```bash
python advanced_example.py
```

Generates:

- `advanced_depth0.gv` - Root only (AdvancedModel)
- `advanced_depth1.gv` - 8 main stages
- `advanced_depth2.gv` - Stage internals (14 modules)
- `advanced_dynamic.gv` - With runtime shapes

### Example: Complex Model

```bash
python complex_example.py
```

Generates:

- `complex_depth0.gv` - Root only (ComplexTransformer)
- `complex_depth1.gv` - 7 top-level modules
- `complex_depth2.gv` - 9 modules (encoder/decoder layers)
- `complex_depth3.gv` - 13 modules (attention details)
- `complex_dynamic.gv` - With runtime shapes

## Static vs Dynamic Capture

### Static Capture

- Analyzes model structure without running forward pass
- No runtime data required
- Shows module hierarchy and parameter counts
- Fast and lightweight

### Dynamic Capture

- Runs forward pass with sample input
- Captures actual tensor shapes, dtypes, devices
- Shows runtime data flow
- Requires sample input data

## Visualization Features

VODE visualizations use a three-column layout:

```
┌─────────────┬──────────────────┬─────────────┐
│   INPUTS    │    OPERATION     │   OUTPUTS   │
│  shape info │  op_type/name    │  shape info │
│             │  param count     │             │
└─────────────┴──────────────────┴─────────────┘
```

**Color Coding:**

- Light yellow: Input/Output tensors
- Dark sea green: Operations (modules)

## Tips

1. **Start Simple:** Begin with `simple_example.py` to understand basic concepts
2. **Explore Depths:** Try different depth levels to find the right detail level
3. **Use SVG:** SVG format is recommended for large models (scalable, searchable)
4. **Compare Modes:** Run both static and dynamic capture to see the difference
5. **CLI Integration:** Use VODE CLI for quick analysis without modifying code

## Requirements

- Python 3.8+
- PyTorch
- VODE package
- Graphviz (for rendering .gv files to images)

Install Graphviz:

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Or use conda
conda install graphviz
```

## Troubleshooting

**Issue:** `.gv` files generated but can't convert to images  
**Solution:** Install Graphviz system package (not just Python package)

**Issue:** "No module named 'vode'"  
**Solution:** Install VODE package or run from project root

**Issue:** Out of memory with complex models  
**Solution:** Use lower depth levels or static capture mode

## Next Steps

After exploring these examples:

1. Try visualizing your own models
2. Experiment with different depth levels
3. Compare static vs dynamic capture
4. Integrate VODE into your development workflow

For more information, see the main VODE documentation.
