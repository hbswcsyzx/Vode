# Vode Test Suite

This directory contains test fixtures organized by difficulty level to validate the function-level tracing capabilities.

## Test Levels

### Level 1: Basic Python Functions

- Simple function calls
- Basic data flow
- No PyTorch dependencies
- Tests core trace runtime

### Level 2: Simple Neural Networks

- Basic `nn.Module` usage
- Single layer models
- Simple forward passes
- Tests `torch.nn` integration

### Level 3: Medium Complexity Networks

- Multi-layer models
- Nested modules
- Multiple operations per forward
- Tests module hierarchy tracking

### Level 4: Complex Neural Networks

- Advanced architectures
- Multiple inputs/outputs
- Dict/tuple returns
- Custom modules
- Tests complex dataflow scenarios

### Level 5: Real-World Models

- Production-scale models
- Reference to `unifolm-world-model-action`
- Tests scalability and filtering

## Running Tests

Each level can be traced independently:

```bash
# Level 1
vode python vode/tests/level_1_basic/simple_functions.py

# Level 2
vode python vode/tests/level_2_simple_nn/linear_model.py

# And so on...
```
