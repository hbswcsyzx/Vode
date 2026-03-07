"""
Level 3: Medium Complexity Networks
Tests multi-layer models with nested modules and various operations.
"""

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """A reusable MLP block."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class NestedNetwork(nn.Module):
    """Network with nested module structure."""
    
    def __init__(self):
        super().__init__()
        self.encoder = MLPBlock(10, 20, 15)
        self.decoder = MLPBlock(15, 20, 5)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class MultiPathNetwork(nn.Module):
    """Network with multiple execution paths."""
    
    def __init__(self):
        super().__init__()
        self.path1 = nn.Linear(10, 5)
        self.path2 = nn.Linear(10, 5)
        self.combine = nn.Linear(10, 5)
    
    def forward(self, x):
        out1 = torch.relu(self.path1(x))
        out2 = torch.sigmoid(self.path2(x))
        combined = torch.cat([out1, out2], dim=-1)
        result = self.combine(combined)
        return result


class ResidualBlock(nn.Module):
    """Simple residual block."""
    
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        out = out + residual  # Residual connection
        return torch.relu(out)


def main():
    """Main entry point for testing."""
    print("Level 3: Medium Complexity Networks")
    
    # Test nested modules
    print("\n1. Testing NestedNetwork...")
    model1 = NestedNetwork()
    input1 = torch.randn(4, 10)
    output1 = model1(input1)
    print(f"Input shape: {input1.shape}")
    print(f"Output shape: {output1.shape}")
    
    # Test multi-path network
    print("\n2. Testing MultiPathNetwork...")
    model2 = MultiPathNetwork()
    input2 = torch.randn(4, 10)
    output2 = model2(input2)
    print(f"Input shape: {input2.shape}")
    print(f"Output shape: {output2.shape}")
    
    # Test residual connections
    print("\n3. Testing ResidualBlock...")
    model3 = ResidualBlock(10)
    input3 = torch.randn(4, 10)
    output3 = model3(input3)
    print(f"Input shape: {input3.shape}")
    print(f"Output shape: {output3.shape}")
    
    return output3


if __name__ == "__main__":
    main()
