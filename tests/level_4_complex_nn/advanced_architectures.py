"""
Level 4: Complex Neural Networks
Tests advanced architectures with multiple inputs/outputs, dict/tuple returns, and custom modules.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class AttentionModule(nn.Module):
    """Simple attention mechanism."""
    
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Simplified attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights


class MultiOutputModel(nn.Module):
    """Model with multiple outputs returned as dict."""
    
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 20)
        self.classifier = nn.Linear(20, 5)
        self.regressor = nn.Linear(20, 1)
    
    def forward(self, x) -> Dict[str, torch.Tensor]:
        features = torch.relu(self.encoder(x))
        logits = self.classifier(features)
        value = self.regressor(features)
        
        return {
            'features': features,
            'logits': logits,
            'value': value
        }


class MultiInputModel(nn.Module):
    """Model with multiple inputs."""
    
    def __init__(self):
        super().__init__()
        self.input1_proj = nn.Linear(10, 15)
        self.input2_proj = nn.Linear(5, 15)
        self.combine = nn.Linear(30, 10)
    
    def forward(self, input1, input2):
        proj1 = self.input1_proj(input1)
        proj2 = self.input2_proj(input2)
        combined = torch.cat([proj1, proj2], dim=-1)
        output = self.combine(combined)
        return output


class TransformerBlock(nn.Module):
    """Simplified transformer block."""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights


class ConditionalModel(nn.Module):
    """Model with conditional execution paths."""
    
    def __init__(self):
        super().__init__()
        self.path_a = nn.Linear(10, 5)
        self.path_b = nn.Linear(10, 5)
        self.gate = nn.Linear(10, 1)
    
    def forward(self, x, use_path_a=True):
        gate_value = torch.sigmoid(self.gate(x))
        
        if use_path_a:
            output = self.path_a(x)
        else:
            output = self.path_b(x)
        
        # Weight by gate
        output = output * gate_value
        return output


def main():
    """Main entry point for testing."""
    print("Level 4: Complex Neural Networks")
    
    # Test attention module
    print("\n1. Testing AttentionModule...")
    model1 = AttentionModule(16)
    input1 = torch.randn(2, 8, 16)  # batch, seq_len, dim
    output1, weights1 = model1(input1)
    print(f"Input shape: {input1.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Attention weights shape: {weights1.shape}")
    
    # Test multi-output model
    print("\n2. Testing MultiOutputModel...")
    model2 = MultiOutputModel()
    input2 = torch.randn(4, 10)
    outputs2 = model2(input2)
    print(f"Input shape: {input2.shape}")
    for key, value in outputs2.items():
        print(f"  {key}: {value.shape}")
    
    # Test multi-input model
    print("\n3. Testing MultiInputModel...")
    model3 = MultiInputModel()
    input3a = torch.randn(4, 10)
    input3b = torch.randn(4, 5)
    output3 = model3(input3a, input3b)
    print(f"Input1 shape: {input3a.shape}")
    print(f"Input2 shape: {input3b.shape}")
    print(f"Output shape: {output3.shape}")
    
    # Test transformer block
    print("\n4. Testing TransformerBlock...")
    model4 = TransformerBlock(32)
    input4 = torch.randn(2, 10, 32)  # batch, seq_len, dim
    output4, attn4 = model4(input4)
    print(f"Input shape: {input4.shape}")
    print(f"Output shape: {output4.shape}")
    
    # Test conditional model
    print("\n5. Testing ConditionalModel...")
    model5 = ConditionalModel()
    input5 = torch.randn(4, 10)
    output5a = model5(input5, use_path_a=True)
    output5b = model5(input5, use_path_a=False)
    print(f"Input shape: {input5.shape}")
    print(f"Output (path A) shape: {output5a.shape}")
    print(f"Output (path B) shape: {output5b.shape}")
    
    return output4


if __name__ == "__main__":
    main()
