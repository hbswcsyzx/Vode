"""
Level 2: Simple Neural Networks
Tests basic nn.Module tracing with single-layer models.
"""

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    """Simple single-layer linear model."""
    
    def __init__(self, input_size=10, output_size=5):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)


class TwoLayerNet(nn.Module):
    """Two-layer network with activation."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        h = self.fc1(x)
        h = torch.relu(h)
        y = self.fc2(h)
        return y


def main():
    """Main entry point for testing."""
    print("Level 2: Simple Neural Networks")
    
    # Test single layer
    print("\n1. Testing LinearModel...")
    model1 = LinearModel()
    input1 = torch.randn(3, 10)
    output1 = model1(input1)
    print(f"Input shape: {input1.shape}")
    print(f"Output shape: {output1.shape}")
    
    # Test two layers with activation
    print("\n2. Testing TwoLayerNet...")
    model2 = TwoLayerNet()
    input2 = torch.randn(3, 10)
    output2 = model2(input2)
    print(f"Input shape: {input2.shape}")
    print(f"Output shape: {output2.shape}")
    
    return output2


if __name__ == "__main__":
    main()
